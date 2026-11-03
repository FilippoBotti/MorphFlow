"""Matched full-model morphing: GLB-reencoded source SLats versus dataset SLats.

Historical predictions remain untouched. Source GLBs are materialized from the
historical decoded source meshes/Gaussians; only the new GLB encoder outputs
condition the glb_encoded branch. Sampling runs in an isolated frozen runtime.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
DEFAULT_BASELINE = WORKSPACE / 'outputs/threeway_benchmark/full_ss17_slat12_n50_seed42_ss3_4_4.5_5_slat1.5_20261026'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def alpha_dir(alpha):
    return ('alpha_%.6f' % alpha).replace('.', 'p')


def prepare(args):
    baseline = args.baseline.resolve()
    original = json.loads((baseline / 'run_manifest.json').read_text())
    oldplan = json.loads((baseline / 'shared/plan.json').read_text())
    checkpoint_meta = json.loads((baseline / 'checkpoint_metadata.json').read_text())
    data = Path(oldplan['data_root']).resolve()
    test_assets = set(json.loads((data / 'split_assets.json').read_text())['test'])
    selected = oldplan['pairs'][:args.count]
    if len(selected) != args.count:
        raise ValueError('Not enough historical pairs')
    cell = baseline / 'cfg_ss_5_slat_1.5'
    pairs, assets = [], {}
    for index, old in enumerate(selected):
        pair = {key: old[key] for key in ('pair_id', 'pair_dir', 'src1', 'src2', 'metadata_index', 'pair_seed')}
        pair.update(index=index, ss_seed=int(old['pair_seed']), slat_seed=int(old['pair_seed']) + 10000)
        pair['source_raw'] = {}
        for label in ('src1', 'src2'):
            name = pair[label]
            if name not in test_assets:
                raise ValueError('Source outside the v4 test split: ' + name)
            raw = baseline / 'shared/references_raw' / pair['pair_dir'] / (label + '_raw.pt')
            if not raw.is_file():
                raise FileNotFoundError(raw)
            pair['source_raw'][label] = str(raw)
            directory = data / 'assets' / name
            for filename in ('slat_feats.pt', 'slat_coords.pt'):
                if not (directory / filename).is_file():
                    raise FileNotFoundError(directory / filename)
            assets.setdefault(name, {'asset': name, 'data_dir': str(directory),
                                      'reference_raw': str(raw), 'raw_identity': identity(raw)})
        pair['baseline_raw'] = {}
        for alpha in oldplan['alphas']:
            raw = cell / 'morphflow' / pair['pair_dir'] / alpha_dir(alpha) / 'mesh_raw.pt'
            if not raw.is_file():
                raise FileNotFoundError(raw)
            pair['baseline_raw'][alpha_dir(alpha)] = str(raw)
        pairs.append(pair)
    checkpoints = {}
    for flow in ('ss', 'slat'):
        saved = next(row for row in checkpoint_meta if row['flow_target'] == flow)
        path = baseline / 'checkpoints' / flow / 'checkpoint.pt'
        checkpoints[flow] = {**saved, **identity(path), 'sha256': original['checkpoints'][flow]['sha256']}
    if (checkpoints['ss']['epoch'], checkpoints['slat']['epoch']) != (17, 12):
        raise ValueError('Unexpected selected full checkpoints')
    frozen_project = (baseline / 'source_snapshot/MorphFlow').resolve()
    frozen_trellis = (baseline / 'source_snapshot/TRELLIS').resolve()
    for path in (frozen_project / 'eval_validation_latents.py', frozen_trellis / 'trellis/__init__.py'):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = {
        'version': 1, 'baseline_run': str(baseline), 'baseline_cell': str(cell),
        'historical_plan': str(baseline / 'shared/plan.json'),
        'selection': 'first ten already-evaluated pairs in the frozen 50-pair plan; no outcome-based selection',
        'count': len(pairs), 'data_root': str(data), 'split': 'test',
        'frozen_project': str(frozen_project), 'frozen_trellis': str(frozen_trellis),
        'encoder_trellis': str(WORKSPACE / 'src/TRELLIS'), 'checkpoints': checkpoints,
        'alphas': oldplan['alphas'], 'pairs': pairs, 'assets': list(assets.values()),
        'sampling': {'ss_steps': 25, 'slat_steps': 25, 'cfg_ss': 5.0, 'cfg_slat': 1.5,
                     'mixed_precision': 'bf16', 'model_type': 'image_large',
                     'tf32_matmul': False, 'tf32_cudnn': True,
                     'alpha_convention': '1=src1, 0=src2',
                     'seed_policy': 'SS=42+100000*pair_id; SLat=SS+10000; reset at every alpha and in both branches',
                     'slat_noise_caveat': 'Same seed; when predicted coordinate sets differ, row-wise random noise is assigned to different spatial positions.'},
        'encoding': {'num_views': 150, 'render_resolution': 512, 'dino_resolution': 518,
                     'lighting': 'uniform_texture',
                     'lighting_description': 'Base-color texture emission=1; no extra lights or shading; Standard sRGB, exposure=0, gamma=1. Dataset v4 is generated from FLUX images, not a fixed Blender light rig.',
                     'voxel_resolution': 64, 'feature_model': 'dinov2_vitl14_reg',
                     'sample_posterior': False, 'latent_scale': 'raw_decoder',
                     'input_glb': 'full historical FP32 source mesh with official Gaussian-baked texture',
                     'texture_size': 2048, 'simplify': 0, 'fill_holes': False},
        'export': {'textured_midpoints': not args.skip_textured_midpoints, 'texture_size': 2048,
                   'all_states': 'Y-up vertex-colored GLBs plus original Gaussian appearance',
                   'mesh_decoder': 'historical FP32 decoder in BOTH morphing branches',
                   'external_source_appearance_transfer': False},
        'dataflow': {'dataset_control': 'dataset source SLat -> full SS -> full SLat -> TRELLIS decoders',
                     'glb_encoded': 'source GLB -> official multiview SLat encoder -> full SS -> full SLat -> TRELLIS decoders'},
    }
    target = args.run_root / 'manifest.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise FileExistsError('Run manifest differs; choose another run root')
    write_json(target, manifest)
    print(json.dumps({'run_root': str(args.run_root), 'pairs': len(pairs), 'unique_sources': len(assets),
                      'states_per_pair': len(oldplan['alphas']), 'checkpoints': checkpoints}, indent=2))


def export_raw_glb(raw_path, out_path, codec, textured=False):
    """Export the already-decoded historical/full-model mesh without redecoding."""
    import numpy as np
    import torch
    import trimesh
    from threeway_textured_benchmark import load_raw_as_trellis_mesh, load_raw_gaussian, glb_validation_error
    codec._init()
    raw_path, out_path = Path(raw_path), Path(out_path)
    provenance = out_path.with_suffix('.json')
    config = {'raw': identity(raw_path), 'textured': textured, 'texture_size': 2048,
              'simplify': 0, 'fill_holes': False, 'axes': 'official Z-up to GLB Y-up',
              'appearance': 'Gaussian decoded from the same latent' if textured else 'decoded vertex RGB'}
    if provenance.is_file() and out_path.is_file():
        previous = json.loads(provenance.read_text())
        if previous['identity'] != config:
            raise ValueError('Incompatible export cache: ' + str(out_path))
        if previous.get('status') == 'complete' and glb_validation_error(out_path) is None:
            return previous
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(provenance, {'identity': config, 'status': 'running'})
    started = time.perf_counter()
    if textured:
        from trellis.utils.postprocessing_utils import to_glb
        mesh = load_raw_as_trellis_mesh(raw_path)
        gaussian = load_raw_gaussian(raw_path)
        print('TEXTURE_EXPORT_START ' + str(out_path), flush=True)
        with torch.inference_mode(False), torch.enable_grad():
            output = to_glb(gaussian, mesh, simplify=0.0, fill_holes=False, texture_size=2048, verbose=False)
    else:
        pack = torch.load(raw_path, map_location='cpu', weights_only=False)
        vertices = pack['vertices'].float().numpy()
        attrs = pack.get('vertex_attrs')
        rgb = np.full((len(vertices), 3), 180, dtype=np.uint8) if attrs is None else np.clip(attrs[:, :3].float().numpy() * 255, 0, 255).astype(np.uint8)
        rgba = np.c_[rgb, np.full(len(rgb), 255, dtype=np.uint8)]
        rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        output = trimesh.Trimesh(vertices @ rotation, pack['faces'].numpy(), vertex_colors=rgba, process=False)
    temporary = out_path.with_name(out_path.stem + '.tmp.glb')
    output.export(temporary)
    temporary.replace(out_path)
    error = glb_validation_error(out_path)
    if error:
        raise RuntimeError(error)
    result = {'identity': config, 'status': 'complete', 'glb': str(out_path),
              'vertices': len(output.vertices), 'faces': len(output.faces),
              'seconds': time.perf_counter() - started}
    write_json(provenance, result)
    print('GLB_EXPORTED ' + str(out_path), flush=True)
    del output
    gc.collect()
    torch.cuda.empty_cache()
    return result


def backend(args, validate_only=False):
    manifest = json.loads((args.run_root / 'manifest.json').read_text())
    pair = manifest['pairs'][args.index]
    folder = args.run_root / 'pairs' / pair['pair_dir']
    folder.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONPATH=manifest['frozen_project'] + ':' + manifest['frozen_trellis'],
               TRELLIS_REPO=manifest['frozen_trellis'])
    command = [sys.executable, str(PROJECT / 'benchmarks/morph_glb_conditioning_backend.py'),
               '--run-root', str(args.run_root), '--index', str(args.index)]
    if validate_only:
        command.append('--validate-only')
    prefix = 'canary' if validate_only else 'generation'
    print('BACKEND_START ' + prefix, flush=True)
    with (folder / (prefix + '.out')).open('a') as stdout, (folder / (prefix + '.err')).open('a') as stderr:
        result = subprocess.run(command, env=env, cwd=manifest['frozen_project'], stdout=stdout, stderr=stderr)
    if result.returncode:
        print((folder / (prefix + '.err')).read_text()[-6000:], file=sys.stderr, flush=True)
        raise RuntimeError('Frozen generation failed; see ' + str(folder / (prefix + '.err')))
    print('BACKEND_COMPLETE ' + prefix, flush=True)


def worker(args):
    import torch
    from trellis_asset_codec import OfficialAssetCodec
    from benchmark_trellis_codec import latent_comparison
    manifest = json.loads((args.run_root / 'manifest.json').read_text())
    pair = manifest['pairs'][args.index]
    output = args.run_root / 'pairs' / pair['pair_dir']
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'result.json').is_file():
            print('Already complete: ' + pair['pair_dir'], flush=True)
            return
        started = time.perf_counter()
        codec = OfficialAssetCodec(trellis_root=manifest['encoder_trellis'], blender_bin=args.blender,
                                    texture_size=2048, dino_batch_size=4, seed=42,
                                    lighting=manifest['encoding'].get('lighting', 'official'))
        try:
            for label in ('src1', 'src2'):
                name = pair[label]
                asset = next(a for a in manifest['assets'] if a['asset'] == name)
                directory = args.run_root / 'assets' / name
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / '.lock').open('a') as asset_lock:
                    fcntl.flock(asset_lock, fcntl.LOCK_EX)
                    write_json(output / 'status.json', {'stage': 'source_glb', 'asset': name})
                    export_raw_glb(asset['reference_raw'], directory / 'source.glb', codec, textured=True)
                    export_raw_glb(asset['reference_raw'], directory / 'source_vertex.glb', codec, textured=False)
                    write_json(output / 'status.json', {'stage': 'encoding_150', 'asset': name})
                    encoded = codec.encode(directory / 'source.glb', directory / 'encoding', num_views=150)
                    original = {'feats': torch.load(Path(asset['data_dir']) / 'slat_feats.pt', map_location='cpu', weights_only=True).float(),
                                'coords': torch.load(Path(asset['data_dir']) / 'slat_coords.pt', map_location='cpu', weights_only=True).int()}
                    write_json(directory / 'encoding_diagnostics.json', latent_comparison(original, encoded))
                    from morph_glb_conditioning_report import report
                    report(args.run_root)
            codec.release()
            write_json(output / 'status.json', {'stage': 'full_model_morphing'})
            backend(args)
            write_json(output / 'status.json', {'stage': 'export_morph_glbs'})
            for branch in ('dataset_control', 'glb_encoded'):
                for alpha in manifest['alphas']:
                    directory = output / branch / alpha_dir(alpha)
                    export_raw_glb(directory / 'mesh_raw.pt', directory / 'prediction.glb', codec)
                    if manifest['export']['textured_midpoints'] and abs(alpha - 0.5) < 1e-10:
                        export_raw_glb(directory / 'mesh_raw.pt', directory / 'prediction_textured.glb', codec, textured=True)
            codec.release()
            write_json(output / 'status.json', {'stage': 'comparison_metrics'})
            from morph_glb_conditioning_metrics import evaluate_pair
            result = evaluate_pair(args.run_root, args.index)
            result['seconds_total'] = time.perf_counter() - started
            write_json(output / 'result.json', result)
            write_json(output / 'status.json', {'stage': 'complete', 'seconds': result['seconds_total']})
            (output / 'error.json').unlink(missing_ok=True)
            print('PAIR_COMPLETE ' + pair['pair_dir'], flush=True)
            from morph_glb_conditioning_report import report
            report(args.run_root)
        except Exception as exc:
            write_json(output / 'error.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
            raise


def status(args):
    manifest = json.loads((args.run_root / 'manifest.json').read_text())
    for pair in manifest['pairs']:
        folder = args.run_root / 'pairs' / pair['pair_dir']
        current = json.loads((folder / 'status.json').read_text()) if (folder / 'status.json').exists() else {'stage': 'pending'}
        print(json.dumps({'pair_id': pair['pair_id'], **current}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'worker', 'canary', 'report', 'status'])
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--skip-textured-midpoints', action='store_true')
    parser.add_argument('--blender', type=Path, default=WORKSPACE / 'tools/blender-3.0.1-linux-x64/blender')
    args = parser.parse_args()
    args.run_root = args.run_root.expanduser().resolve()
    if args.mode == 'canary':
        backend(args, validate_only=True)
    elif args.mode == 'report':
        from morph_glb_conditioning_report import report
        report(args.run_root)
    else:
        {'prepare': prepare, 'worker': worker, 'status': status}[args.mode](args)


if __name__ == '__main__':
    main()
