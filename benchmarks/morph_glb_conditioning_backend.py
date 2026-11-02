"""Run the frozen MorphFlow pipeline with dataset or GLB-encoded conditioning.

This backend runs in its own process so its imports and checkpoint loading use
the historical project and TRELLIS checkout. It writes SS, SLat and raw mesh /
Gaussian outputs; exporting textures and measuring renders belongs to the
parent runner. --validate-only runs the dataset control at alpha=0.5 without
requiring any encoded assets.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid


VERSION = 1


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp.' + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_stat(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def single_component(value, label):
    if not isinstance(value, str) or not value or value in ('.', '..') or Path(value).name != value or '/' in value:
        raise ValueError(f'{label} must be one path component')
    return value


def verified_checkpoints(root, manifest, manifest_digest):
    """Hash immutable checkpoint files once, shared across the pair workers."""
    directory = root / 'runtime'
    directory.mkdir(parents=True, exist_ok=True)
    cache_path = directory / 'conditioning_backend_checkpoint_validation.json'
    identity = {'manifest_sha256': manifest_digest, 'files': {}}
    for target in ('ss', 'slat'):
        spec = manifest['checkpoints'][target]
        path = Path(spec['path']).expanduser().resolve()
        identity['files'][target] = {'stat': file_stat(path), 'expected_sha256': spec.get('sha256')}
        if spec.get('bytes') is not None and path.stat().st_size != int(spec['bytes']):
            raise ValueError(f'Checkpoint size differs from manifest: {path}')
    with (directory / 'conditioning_backend_checkpoint_validation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text())
            if cached.get('identity') == identity and cached.get('status') == 'complete':
                return cached['verified']
        verified = {}
        for target, value in identity['files'].items():
            path = Path(value['stat']['path'])
            actual = sha256(path)
            if value['expected_sha256'] and actual != value['expected_sha256']:
                raise ValueError(f'Checkpoint SHA256 differs from manifest: {path}')
            if file_stat(path) != value['stat']:
                raise RuntimeError(f'Checkpoint changed during hashing: {path}')
            verified[target] = {**value['stat'], 'sha256': actual}
        write_json(cache_path, {'identity': identity, 'status': 'complete', 'verified': verified})
        return verified


def frozen_imports(manifest):
    project = Path(manifest['frozen_project']).expanduser().resolve()
    trellis_root = Path(manifest['frozen_trellis']).expanduser().resolve()
    required = [project / 'eval_validation_latents.py', project / 'benchmarks/threeway_textured_benchmark.py',
                trellis_root / 'trellis/__init__.py']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    for prefix in ('models', 'modules', 'trellis', 'eval_validation_latents', 'threeway_textured_benchmark'):
        if prefix in sys.modules:
            raise RuntimeError(f'{prefix} was imported before selecting the frozen source')
    sys.path[:0] = [str(project), str(trellis_root), str(project / 'benchmarks')]
    os.environ['TRELLIS_REPO'] = str(trellis_root)
    os.environ.setdefault('ATTN_BACKEND', 'xformers')
    os.environ.setdefault('SPCONV_ALGO', 'native')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    import torch
    evaluation = importlib.import_module('eval_validation_latents')
    benchmark = importlib.import_module('threeway_textured_benchmark')
    import trellis
    for module, expected in ((evaluation, project), (benchmark, project), (trellis, trellis_root)):
        if not Path(module.__file__).resolve().is_relative_to(expected):
            raise RuntimeError(f'Unexpected imported source: {module.__file__}')
    for name in ('models.morph_flow', 'models.morph_slat_flow', 'models.lora'):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(project):
            raise RuntimeError(f'Wrong frozen model import: {module.__file__}')
    source_identity = {}
    for path in [*required, project / 'models/morph_flow.py', project / 'models/morph_slat_flow.py',
                 project / 'models/lora.py']:
        source_identity[str(path)] = sha256(path)
    return torch, evaluation, benchmark, source_identity


def load_pack(torch, paths, branch):
    before = {str(path): file_stat(path) for path in paths}
    if branch == 'dataset_control':
        feats = torch.load(paths[0], map_location='cpu', weights_only=True)
        coords = torch.load(paths[1], map_location='cpu', weights_only=True)
    else:
        pack = torch.load(paths[0], map_location='cpu', weights_only=True)
        feats, coords = pack['feats'], pack['coords']
    feats = torch.as_tensor(feats).detach().float().cpu().contiguous()
    coords = torch.as_tensor(coords).detach().cpu()
    if feats.ndim != 2 or feats.shape[1] != 8 or not len(feats) or not torch.isfinite(feats).all():
        raise ValueError(f'Invalid raw SLat feature pack: {paths}')
    if coords.ndim != 2 or coords.shape[0] != len(feats) or coords.shape[1] not in (3, 4):
        raise ValueError(f'Invalid SLat coordinate shape: {paths}')
    if torch.is_floating_point(coords) and not torch.equal(coords, coords.round()):
        raise ValueError(f'Noninteger SLat coordinates: {paths}')
    coords = coords.int().contiguous()
    if coords.shape[1] == 3:
        coords = torch.cat([torch.zeros((len(coords), 1), dtype=torch.int32), coords], dim=1)
    if (coords[:, 0] != 0).any() or (coords[:, 1:] < 0).any() or (coords[:, 1:] >= 64).any():
        raise ValueError(f'Source coordinates must use one batch and spatial grid 0..63: {paths}')
    if len(torch.unique(coords, dim=0)) != len(coords):
        raise ValueError(f'Duplicate source coordinates: {paths}')
    identity = []
    for path in paths:
        if file_stat(path) != before[str(path)]:
            raise RuntimeError(f'Source SLat changed during loading: {path}')
        identity.append({**before[str(path)], 'sha256': sha256(path)})
    return {'feats': feats, 'coords': coords}, {'kind': branch, 'files': identity,
                                             'voxels': len(coords), 'latent_scale': 'raw_decoder'}


def tensor_comparison(torch, actual, expected):
    out = {'actual_shape': list(actual.shape), 'reference_shape': list(expected.shape),
           'actual_dtype': str(actual.dtype), 'reference_dtype': str(expected.dtype)}
    if actual.shape != expected.shape:
        return {**out, 'equal': False, 'max_abs': None, 'allclose_atol_1e_6_rtol_1e_5': False}
    a, b = actual.detach().cpu(), expected.detach().cpu()
    equal = bool(torch.equal(a, b)) and a.dtype == b.dtype
    difference = (a.double() - b.double()).abs()
    max_abs = float(difference.max()) if difference.numel() else 0.0
    if not math.isfinite(max_abs):
        max_abs = None
    return {**out, 'equal': equal, 'max_abs': max_abs,
            'allclose_atol_1e_6_rtol_1e_5': bool(torch.allclose(a.double(), b.double(), atol=1e-6, rtol=1e-5))}


def compare_raw(torch, fresh, reference, reference_path):
    fields = {}

    def compare(actual, expected, key):
        if torch.is_tensor(actual) and torch.is_tensor(expected):
            fields[key] = tensor_comparison(torch, actual, expected)
        elif isinstance(actual, dict) and isinstance(expected, dict):
            if set(actual) != set(expected):
                fields[key + '.keys'] = {'equal': False, 'actual': sorted(actual), 'reference': sorted(expected)}
            for name in sorted(set(actual) | set(expected)):
                if name in actual and name in expected:
                    compare(actual[name], expected[name], key + '.' + name)
                else:
                    fields[key + '.' + name] = {'equal': False, 'missing_in': 'actual' if name not in actual else 'reference'}
        elif isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
            if len(actual) != len(expected):
                fields[key] = {'equal': False, 'actual_length': len(actual), 'reference_length': len(expected)}
            else:
                for index, (a, b) in enumerate(zip(actual, expected)):
                    compare(a, b, key + f'[{index}]')
        else:
            equal = bool(actual == expected) if actual is not None and expected is not None else actual is expected
            fields[key] = {'equal': equal}

    for name in ('vertices', 'faces', 'vertex_attrs', 'res', 'gaussian'):
        if name not in fresh or name not in reference:
            fields[name] = {'equal': False, 'missing': True}
        else:
            compare(fresh[name], reference[name], name)
    errors = [value['max_abs'] for value in fields.values() if value.get('max_abs') is not None]
    return {'reference_raw': str(reference_path), 'reference_sha256': sha256(reference_path),
            'exact_numeric_reproduction': all(value['equal'] for value in fields.values()),
            'topology_equal': fields.get('faces', {}).get('equal', False),
            'max_abs_over_tensor_fields': max(errors, default=None), 'fields': fields,
            'comparison_excludes': 'Serialization bytes and metadata; mesh geometry/attributes and full Gaussian parameters are compared.'}


def save_tensor(torch, path, value):
    temporary = path.with_name('.' + path.name + '.tmp.' + uuid.uuid4().hex)
    torch.save(value, temporary)
    temporary.replace(path)


def ready_state(path, identity):
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if data.get('identity') != identity:
        raise ValueError(f'State belongs to another manifest/source/configuration: {path}')
    if data.get('status') != 'complete':
        return None
    for item in data.get('files', []):
        file = path.parent / item['name']
        if not file.is_file() or file.stat().st_size != item['bytes'] or sha256(file) != item['sha256']:
            print(f'[backend] Invalid cached artifact; regenerating: {file}', flush=True)
            return None
    if not data.get('files'):
        return None
    return data


def run(args):
    root = args.run_root.expanduser().resolve()
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_digest = sha256(manifest_path)
    if not 0 <= args.index < len(manifest['pairs']):
        raise ValueError('--index is outside the manifest pair list')
    pair = manifest['pairs'][args.index]
    pair_dir = single_component(pair['pair_dir'], 'pair_dir')
    for name in ('src1', 'src2'):
        single_component(pair[name], name)
    sampling = manifest['sampling']
    for name in ('ss_steps', 'slat_steps'):
        if int(sampling[name]) < 1:
            raise ValueError(f'{name} must be positive')
    for name in ('cfg_ss', 'cfg_slat'):
        if not math.isfinite(float(sampling[name])) or float(sampling[name]) < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if sampling['mixed_precision'] not in ('no', 'fp16', 'bf16'):
        raise ValueError('Unsupported mixed precision')
    for name in ('tf32_matmul', 'tf32_cudnn'):
        if not isinstance(sampling[name], bool):
            raise ValueError(f'{name} must be a JSON boolean')
    alphas = [float(value) for value in manifest['alphas']]
    if not alphas or any(not math.isfinite(value) or not 0 <= value <= 1 for value in alphas):
        raise ValueError('Manifest alphas must be finite and in [0,1]')
    if args.validate_only:
        if not any(abs(value - 0.5) < 1e-10 for value in alphas):
            raise ValueError('--validate-only requires alpha=0.5 in the manifest')
        alphas = [next(value for value in alphas if abs(value - 0.5) < 1e-10)]
    branches = ['dataset_control'] if args.validate_only else ['dataset_control', 'glb_encoded']
    source_paths = {}
    for branch in branches:
        source_paths[branch] = []
        for field in ('src1', 'src2'):
            source = pair[field]
            directory = Path(manifest['data_root']).expanduser().resolve() / 'assets' / source
            paths = [directory / 'slat_feats.pt', directory / 'slat_coords.pt'] if branch == 'dataset_control' else [root / 'assets' / source / 'encoding/latent.pt']
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
            source_paths[branch].append(paths)
    pair_out = root / 'pairs' / pair_dir
    pair_out.mkdir(parents=True, exist_ok=True)
    owner_path = pair_out / 'conditioning_backend_identity.json'
    owner = {'version': VERSION, 'manifest_sha256': manifest_digest, 'pair_id': pair['pair_id'],
             'pair_dir': pair_dir, 'src1': pair['src1'], 'src2': pair['src2']}
    with (pair_out / '.conditioning_backend.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Pair backend already running: {pair_out}') from exc
        if owner_path.is_file() and json.loads(owner_path.read_text()) != owner:
            raise ValueError(f'Pair output belongs to another frozen manifest: {pair_out}')
        write_json(owner_path, owner)
        try:
            return run_locked(root, manifest, manifest_digest, pair, pair_out, sampling, alphas, branches, source_paths)
        except Exception as exc:
            write_json(pair_out / 'conditioning_backend_error.json', {
                **owner, 'error': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
            raise


def run_locked(root, manifest, manifest_digest, pair, pair_out, sampling, alphas, branches, source_paths):
    checkpoints = verified_checkpoints(root, manifest, manifest_digest)
    torch, evaluation, benchmark, sources = frozen_imports(manifest)
    if not torch.cuda.is_available():
        raise RuntimeError('Run this backend inside an allocated CUDA GPU job')
    if sampling['mixed_precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Frozen bf16 sampling requires a supported GPU')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = sampling['tf32_matmul']
    torch.backends.cudnn.allow_tf32 = sampling['tf32_cudnn']
    device = torch.device('cuda')
    packs, pack_identity = {}, {}
    for branch in branches:
        loaded = [load_pack(torch, paths, branch) for paths in source_paths[branch]]
        packs[branch] = [item[0] for item in loaded]
        pack_identity[branch] = [item[1] for item in loaded]
    alpha_dirs = [benchmark.alpha_dir(alpha) for alpha in alphas]
    if len(set(alpha_dirs)) != len(alpha_dirs):
        raise ValueError('Alpha values collide in historical six-decimal output directory names')
    models = {}
    checkpoint_details = {}
    for target in ('ss', 'slat'):
        checkpoint = torch.load(checkpoints[target]['path'], map_location='cpu', weights_only=False, mmap=True)
        if evaluation.detect_flow_target(checkpoint) != target:
            raise ValueError(f'Wrong checkpoint flow target: {target}')
        model_type = evaluation.detect_model_type(checkpoint, sampling.get('trellis_model', 'auto'))
        models[target] = evaluation.build_model(checkpoint, model_type, target).to(device).eval().requires_grad_(False)
        checkpoint_details[target] = {'epoch': checkpoint.get('epoch'), 'step': checkpoint.get('step'),
                                      'model_type': model_type, 'val_loss': checkpoint.get('val_loss')}
        del checkpoint
    ss_decoder, mesh_decoder, sparse_cls = evaluation.load_decoders('ss', device)
    from trellis.models import from_pretrained
    gaussian_decoder = from_pretrained('microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16').to(device).eval().requires_grad_(False)
    flow = models['ss'].sparse_structure_flow
    template = torch.zeros((1, int(flow.in_channels), int(flow.resolution), int(flow.resolution), int(flow.resolution)),
                           device=device, dtype=torch.float32)
    started = time.perf_counter()
    completed, generated, skipped = [], 0, 0
    for branch in branches:
        src1, src2 = packs[branch]
        for alpha, directory_name in zip(alphas, alpha_dirs):
            folder = pair_out / branch / directory_name
            folder.mkdir(parents=True, exist_ok=True)
            state_path = folder / 'state.json'
            identity = {'version': VERSION, 'manifest_sha256': manifest_digest,
                        'pair_id': pair['pair_id'], 'branch': branch, 'alpha': alpha,
                        'ss_seed': int(pair['ss_seed']), 'slat_seed': int(pair['slat_seed']),
                        'sampling': sampling, 'source_packs': pack_identity[branch],
                        'checkpoints': checkpoints, 'frozen_sources': sources,
                        'mesh_decoder_precision': 'float32', 'ss_occupancy_threshold': 0.0}
            ready = ready_state(state_path, identity)
            if ready is not None:
                skipped += 1
                completed.append(ready)
                print(f'[backend] Reuse {branch} {directory_name}', flush=True)
                continue
            write_json(state_path, {'identity': identity, 'status': 'running'})
            step_started = time.perf_counter()
            batch = {'src1_feats': src1['feats'], 'src1_coords': src1['coords'],
                     'src2_feats': src2['feats'], 'src2_coords': src2['coords'],
                     'alpha': torch.tensor([alpha], dtype=torch.float32)}
            benchmark.seed_everything(int(pair['ss_seed']))
            ss = evaluation.sample_ss(models['ss'], batch, template, int(sampling['ss_steps']), device,
                                      float(sampling['cfg_ss']), sampling['mixed_precision'])
            if not torch.isfinite(ss).all():
                raise FloatingPointError(f'Nonfinite SS output: {branch} {alpha}')
            coords = evaluation.ss_coords_from_latent(ss_decoder, ss, device, sampling['mixed_precision'])
            if coords.numel() == 0:
                raise RuntimeError(f'Empty predicted SS structure: {branch} {alpha}')
            benchmark.seed_everything(int(pair['slat_seed']))
            slat = evaluation.sample_slat_on_coords(models['slat'], batch, coords, int(sampling['slat_steps']),
                                                     device, float(sampling['cfg_slat']), sampling['mixed_precision'])
            if slat is None or not torch.isfinite(slat.feats).all():
                raise FloatingPointError(f'Invalid SLat output: {branch} {alpha}')
            mesh = benchmark.decode_slat_mesh(mesh_decoder, sparse_cls, slat.feats, slat.coords, device)
            gaussian = benchmark.decode_slat_gaussian(gaussian_decoder, sparse_cls, slat.feats, slat.coords, device)
            if not mesh.success or not torch.isfinite(mesh.vertices).all():
                raise RuntimeError(f'Empty or nonfinite decoded mesh: {branch} {alpha}')
            metadata = {'method': 'morphflow', 'pair_id': int(pair['pair_id']), 'alpha': alpha,
                        'conditioning_branch': branch, 'ss_seed': int(pair['ss_seed']),
                        'slat_seed': int(pair['slat_seed']), 'manifest_sha256': manifest_digest,
                        'source_packs': pack_identity[branch]}
            save_tensor(torch, folder / 'ss.pt', {'latent': ss.detach().float().cpu(), 'coords': coords.detach().int().cpu()})
            save_tensor(torch, folder / 'slat.pt', {'feats': slat.feats.detach().float().cpu(), 'coords': slat.coords.detach().int().cpu()})
            raw_path = folder / 'mesh_raw.pt'
            benchmark.save_raw_mesh(mesh, raw_path, metadata, gaussian=gaussian)
            torch.cuda.synchronize()
            reproduction = None
            if branch == 'dataset_control':
                reference_path = Path(pair['baseline_raw'][directory_name]).expanduser().resolve()
                reference = torch.load(reference_path, map_location='cpu', weights_only=False)
                original_metadata = reference.get('metadata', {})
                for key, expected in (('pair_id', int(pair['pair_id'])), ('ss_seed', int(pair['ss_seed'])),
                                      ('slat_seed', int(pair['slat_seed']))):
                    if key in original_metadata and int(original_metadata[key]) != expected:
                        raise ValueError(f'Historical baseline has another {key}: {reference_path}')
                if 'alpha' in original_metadata and abs(float(original_metadata['alpha']) - alpha) > 1e-10:
                    raise ValueError(f'Historical baseline has another alpha: {reference_path}')
                fresh = torch.load(raw_path, map_location='cpu', weights_only=False)
                reproduction = compare_raw(torch, fresh, reference, reference_path)
                reproduction['raw_storage'] = 'fresh_regeneration'
                if reproduction['exact_numeric_reproduction']:
                    temporary = raw_path.with_name('.mesh_raw.historical_link.' + uuid.uuid4().hex)
                    try:
                        os.link(reference_path, temporary)
                        temporary.replace(raw_path)
                        reproduction['raw_storage'] = 'hardlink_to_exact_historical_numeric_reference'
                    except OSError:
                        temporary.unlink(missing_ok=True)
                write_json(folder / 'control_reproduction.json', reproduction)
                print(f"[backend] Control reproduction {directory_name}: exact={reproduction['exact_numeric_reproduction']} max_abs={reproduction['max_abs_over_tensor_fields']}", flush=True)
                del fresh, reference
            filenames = ['ss.pt', 'slat.pt', 'mesh_raw.pt', 'mesh_raw.appearance.json']
            if reproduction is not None:
                filenames.append('control_reproduction.json')
            files = [{'name': name, 'bytes': (folder / name).stat().st_size, 'sha256': sha256(folder / name)} for name in filenames]
            for spec in checkpoints.values():
                current = file_stat(spec['path'])
                if any(current[key] != spec[key] for key in ('path', 'bytes', 'mtime_ns')):
                    raise RuntimeError(f"Checkpoint changed during inference: {spec['path']}")
            state = {'identity': identity, 'status': 'complete', 'files': files,
                     'seconds': time.perf_counter() - step_started,
                     'predicted_voxels': int(len(coords)), 'vertices': int(len(mesh.vertices)), 'faces': int(len(mesh.faces)),
                     'raw_path': str(raw_path), 'checkpoint_details': checkpoint_details,
                     'control_reproduction': reproduction}
            write_json(state_path, state)
            completed.append(state)
            generated += 1
            print(f"[backend] COMPLETE pair={pair['pair_id']} branch={branch} alpha={alpha:.10f} voxels={len(coords)} seconds={state['seconds']:.1f}", flush=True)
            del batch, ss, coords, slat, mesh, gaussian
            torch.cuda.empty_cache()
    result = {'status': 'complete', 'manifest_sha256': manifest_digest, 'pair_id': pair['pair_id'],
              'branches': branches, 'alphas': alphas, 'states_completed': len(completed),
              'states_generated': generated, 'states_reused': skipped,
              'exact_dataset_controls': sum(bool(item.get('control_reproduction', {}).get('exact_numeric_reproduction'))
                                            for item in completed if item.get('control_reproduction')),
              'seconds': time.perf_counter() - started, 'checkpoint_details': checkpoint_details}
    write_json(pair_out / 'conditioning_backend_summary.json', result)
    (pair_out / 'conditioning_backend_error.json').unlink(missing_ok=True)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True, help='Zero-based index into manifest pairs')
    parser.add_argument('--validate-only', action='store_true', help='Only regenerate the dataset control at alpha=0.5')
    args = parser.parse_args(argv)
    run(args)


if __name__ == '__main__':
    main()
