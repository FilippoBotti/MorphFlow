"""Resumable GLB/SLat -> MorphFlow SS -> SLat -> officially textured GLBs.

alpha=1 selects src1; alpha=0 selects src2. Endpoints are sampled by the
model, never replaced with the input. Models are loaded in separate stages
to limit GPU memory. Input SLat features must be in raw TRELLIS decoder scale.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import time
import traceback
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
VERSION = 1
MODEL_REPO = 'microsoft/TRELLIS-image-large/ckpts'
SS_DECODER = 'ss_dec_conv3d_16l8_fp16'
GS_DECODER = 'slat_dec_gs_swin8_B_64l8gs32_fp16'
GS_FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.tmp.' + uuid.uuid4().hex)
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path, old=None):
    path = Path(path).resolve()
    stat = path.stat()
    info = {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if old and all(old.get(k) == v for k, v in info.items()):
        info['sha256'] = old['sha256']
    else:
        info['sha256'] = sha256(path)
    return info


def alpha_tag(alpha):
    short = format(float(alpha), '.8g')
    value = short if float(short) == alpha else repr(float(alpha))
    return 'alpha_' + value.replace('.', 'p').replace('-', 'm')


def parse_alphas(values):
    result = [float(v) for value in values for v in value.replace(',', ' ').split()]
    if not result or any(not math.isfinite(v) or not 0 <= v <= 1 for v in result):
        raise ValueError('Alphas must be finite numbers in [0,1]')
    if len(set(result)) != len(result):
        raise ValueError('Duplicate alpha values')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('src1', type=Path, help='GLB, SLat .pt/.npz, or asset directory')
    parser.add_argument('src2', type=Path, help='Same input type as src1')
    parser.add_argument('--alphas', nargs='+', required=True, help='0 0.25 0.5 0.75 1, or a quoted comma-separated list')
    parser.add_argument('--ss-checkpoint', type=Path, required=True)
    parser.add_argument('--slat-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--blender', type=Path, help='Blender executable; overrides BLENDER_BIN')
    parser.add_argument('--trellis-root', type=Path, default=os.environ.get('TRELLIS_REPO'))
    parser.add_argument('--steps-ss', '--steps', type=int, default=25)
    parser.add_argument('--steps-slat', '--slat-steps', type=int, default=25)
    parser.add_argument('--cfg-ss', type=float, default=5.0)
    parser.add_argument('--cfg-slat', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--slat-seed', type=int, help='Default: seed+10000')
    parser.add_argument('--mixed-precision', choices=('auto', 'bf16', 'fp16', 'no'), default='auto')
    parser.add_argument('--encoding-views', type=int, default=150)
    parser.add_argument('--dino-batch-size', type=int, default=4)
    parser.add_argument('--lighting', choices=('uniform_texture', 'official'), default='uniform_texture')
    parser.add_argument('--glb-up-axis', choices=('y', 'z'), default='y',
                        help='Actual source up axis: y for standard glTF, z for GLBs storing Z-up coordinates')
    parser.add_argument('--src1-up-axis', choices=('y', 'z'), help='Override --glb-up-axis for src1')
    parser.add_argument('--src2-up-axis', choices=('y', 'z'), help='Override --glb-up-axis for src2')
    parser.add_argument('--texture-size', type=int, default=2048)
    parser.add_argument('--simplify', type=float, default=0.0, help='Fraction of triangles removed; default preserves all geometry')
    parser.add_argument('--fill-holes', action='store_true')
    parser.add_argument('--allow-tf32', action='store_true')
    parser.add_argument('--cudnn-tf32', type=int, choices=(0, 1),
                        help='Override convolution TF32 separately; default follows --allow-tf32')
    parser.add_argument('--mode', choices=('run', 'check'), default='run')
    args = parser.parse_args(argv)
    try:
        args.alphas = parse_alphas(args.alphas)
    except ValueError as exc:
        parser.error(str(exc))
    if min(args.steps_ss, args.steps_slat, args.encoding_views, args.dino_batch_size, args.texture_size) < 1:
        parser.error('Steps, views, batch size and texture size must be positive')
    if any(not math.isfinite(v) or v < 0 for v in (args.cfg_ss, args.cfg_slat)):
        parser.error('CFG values must be finite and nonnegative')
    if not math.isfinite(args.simplify) or not 0 <= args.simplify < 1:
        parser.error('--simplify must be in [0,1)')
    if args.seed < 0 or (args.slat_seed is not None and args.slat_seed < 0):
        parser.error('Seeds must be nonnegative')
    args.slat_seed = args.seed + 10000 if args.slat_seed is None else args.slat_seed
    args.cudnn_tf32 = args.allow_tf32 if args.cudnn_tf32 is None else bool(args.cudnn_tf32)
    args.src1_up_axis = args.src1_up_axis or args.glb_up_axis
    args.src2_up_axis = args.src2_up_axis or args.glb_up_axis
    for name in ('src1', 'src2', 'ss_checkpoint', 'slat_checkpoint'):
        value = getattr(args, name).expanduser().resolve()
        if not value.exists() or (name.endswith('checkpoint') and not value.is_file()):
            parser.error('Missing ' + name + ': ' + str(value))
        setattr(args, name, value)
    args.input_type = 'glb' if args.src1.suffix.lower() == '.glb' else 'slat'
    if (args.src2.suffix.lower() == '.glb') != (args.input_type == 'glb'):
        parser.error('Provide two GLBs or two SLat inputs')
    args.output_dir = args.output_dir.expanduser().resolve()
    args.trellis_root = Path(args.trellis_root or PROJECT.parent / 'TRELLIS').expanduser().resolve()
    if not (args.trellis_root / 'trellis/models/__init__.py').is_file():
        parser.error('Set --trellis-root or TRELLIS_REPO to a TRELLIS checkout')
    if args.input_type == 'glb':
        args.blender = resolve_blender(args.blender)
        for path in (args.src1, args.src2):
            with path.open('rb') as stream:
                if stream.read(4) != b'glTF':
                    parser.error('Invalid GLB: ' + str(path))
        if not (args.trellis_root / 'dataset_toolkits/blender_script/render.py').is_file():
            parser.error('GLB encoding needs the official TRELLIS dataset_toolkits')
    else:
        if args.src1_up_axis != 'y' or args.src2_up_axis != 'y':
            parser.error('Up-axis correction applies to GLBs; encoded SLat coordinates cannot be rotated independently of their features')
        args.blender = None
        for path in (args.src1, args.src2):
            input_files(path, 'slat')
    return args


def resolve_blender(override=None):
    explicit = override or os.environ.get('BLENDER_BIN')
    candidates = [explicit] if explicit else [shutil.which('blender'),
        '/home/filippo/projects/morph_old/.cache/blender/blender-3.0.1-linux-x64/blender',
        str(PROJECT.parents[1] / 'tools/blender-3.0.1-linux-x64/blender')]
    for candidate in candidates:
        if candidate:
            executable = Path(candidate).expanduser().resolve()
            if executable.is_file() and os.access(executable, os.X_OK):
                return executable
    raise FileNotFoundError('Blender is not executable: ' + str(explicit or 'set --blender or BLENDER_BIN'))


def input_files(path, kind):
    if kind == 'glb':
        return [path]
    if path.is_dir():
        files = [path / 'slat_feats.pt', path / 'slat_coords.pt']
        if all(file.is_file() for file in files):
            return files
        for name in ('latent.pt', 'slat.pt'):
            if (path / name).is_file():
                return [path / name]
        raise ValueError('SLat directory needs slat_feats.pt + slat_coords.pt, latent.pt, or slat.pt: ' + str(path))
    if path.suffix.lower() not in ('.pt', '.npz'):
        raise ValueError('SLat input must be a .pt/.npz pack or an asset directory: ' + str(path))
    return [path]


def load_tensor(path, checkpoint=False):
    import torch
    kwargs = {'map_location': 'cpu', 'weights_only': not checkpoint}
    if checkpoint:
        try:
            return torch.load(path, mmap=True, **kwargs)
        except RuntimeError as exc:
            if 'mmap' not in str(exc):
                raise
    return torch.load(path, **kwargs)


def validate_pack(pack):
    import torch
    if not isinstance(pack, dict) or not all(k in pack for k in ('feats', 'coords')):
        raise ValueError('Expected a SLat dict with feats [N,8] and coords [N,3 or 4]')
    scale = pack.get('latent_scale', pack.get('metadata', {}).get('latent_scale', 'raw_decoder'))
    if scale != 'raw_decoder':
        raise ValueError('Expected raw_decoder SLat scale; received ' + str(scale))
    feats = torch.as_tensor(pack['feats']).detach().float().cpu().contiguous()
    coords = torch.as_tensor(pack['coords']).detach().cpu()
    if feats.ndim != 2 or feats.shape[1] != 8 or not len(feats) or not torch.isfinite(feats).all():
        raise ValueError('SLat feats must be finite and have shape [N,8]')
    if coords.ndim != 2 or coords.shape[0] != len(feats) or coords.shape[1] not in (3, 4):
        raise ValueError('SLat coords must have shape [N,3 or 4]')
    if not torch.isfinite(coords).all() or (torch.is_floating_point(coords) and not torch.equal(coords, coords.round())):
        raise ValueError('SLat coordinates must be finite integers')
    if (coords[:, -3:] < 0).any() or (coords[:, -3:] >= 64).any():
        raise ValueError('SLat spatial coordinates must lie in [0,63]')
    if coords.shape[1] == 4 and (coords[:, 0] != 0).any():
        raise ValueError('Provide one asset per input (batch index zero)')
    coords = coords.int().contiguous()
    if coords.shape[1] == 3:
        coords = torch.cat([torch.zeros((len(coords), 1), dtype=torch.int32), coords], dim=1)
    if len(torch.unique(coords, dim=0)) != len(coords):
        raise ValueError('Duplicate SLat coordinates')
    return {'feats': feats, 'coords': coords}


def read_slat(path):
    files = input_files(path, 'slat')
    if len(files) == 2:
        return validate_pack({'feats': load_tensor(files[0]), 'coords': load_tensor(files[1])})
    if files[0].suffix.lower() == '.npz':
        import numpy as np
        with np.load(files[0], allow_pickle=False) as pack:
            return validate_pack({k: pack[k].copy() for k in ('feats', 'coords')})
    return validate_pack(load_tensor(files[0]))


def save_tensor(path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.tmp.' + uuid.uuid4().hex)
    torch.save(value, temp)
    temp.replace(path)


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def release():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def ready(folder, stage):
    marker = folder / (stage + '.json')
    if not marker.is_file():
        return False
    try:
        saved = json.loads(marker.read_text())
        return saved.get('status') == 'complete' and bool(saved.get('files')) and all(
            (folder / f['name']).is_file() and (folder / f['name']).stat().st_size == f['bytes']
            and sha256(folder / f['name']) == f['sha256'] for f in saved['files'])
    except (OSError, ValueError, KeyError):
        return False


def invalidate(folder, stages):
    for stage in stages:
        (folder / (stage + '.json')).unlink(missing_ok=True)


def glb_error(path, textured=False):
    try:
        with path.open('rb') as stream:
            magic, version, length = struct.unpack('<4sII', stream.read(12))
            if magic != b'glTF' or version != 2 or length != path.stat().st_size:
                return 'Invalid GLB header or truncated file'
            size, kind = struct.unpack('<II', stream.read(8))
            if kind != 0x4e4f534a or size > length - 20:
                return 'Invalid GLB JSON chunk'
            document = json.loads(stream.read(size))
            if not document.get('meshes'):
                return 'GLB contains no mesh'
            if textured and not (document.get('images') and document.get('textures')):
                return 'GLB is missing its embedded texture'
    except (OSError, ValueError, struct.error) as exc:
        return str(exc)
    return None


def commit(folder, stage, names, **metadata):
    write_json(folder / (stage + '.json'), {'status': 'complete', **metadata,
        'files': [{'name': name, 'bytes': (folder / name).stat().st_size,
                   'sha256': sha256(folder / name)} for name in names]})


def runtime(args):
    os.environ['TRELLIS_REPO'] = str(args.trellis_root)
    sys.path.insert(0, str(args.trellis_root))
    os.environ.setdefault('ATTN_BACKEND', 'xformers')
    os.environ.setdefault('SPCONV_ALGO', 'native')
    import torch
    import trellis
    if not Path(trellis.__file__).resolve().is_relative_to(args.trellis_root):
        raise RuntimeError('A different TRELLIS checkout was imported: ' + str(trellis.__file__))
    if args.mode == 'run' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; on Leonardo submit with sbatch')
    precision = args.mixed_precision
    if precision == 'auto':
        precision = 'bf16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'fp16'
    if precision == 'bf16' and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support bf16; use --mixed-precision fp16')
    args.precision = precision
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.cudnn_tf32
    return torch


def preflight(args):
    import importlib
    from eval_validation_latents import checkpoint_args, detect_flow_target, detect_model_type
    for name in ('nvdiffrast.torch', 'diff_gaussian_rasterization', 'xatlas', 'trellis.utils.postprocessing_utils'):
        importlib.import_module(name)
    metadata = {}
    for target in ('ss', 'slat'):
        path = getattr(args, target + '_checkpoint')
        checkpoint = load_tensor(path, checkpoint=True)
        config = checkpoint_args(checkpoint)
        if detect_flow_target(checkpoint) != target:
            raise ValueError('Wrong checkpoint flow target for ' + target + ': ' + str(path))
        if config.get('ss_flow_arch', 'standard') != 'standard' or config.get('slat_condition_source', 'slat') != 'slat':
            raise ValueError('Only standard, SLat-conditioned checkpoints are supported')
        metadata[target] = {'path': str(path), 'epoch': checkpoint.get('epoch'), 'step': checkpoint.get('step'),
                            'model_type': detect_model_type(checkpoint, 'auto')}
        del checkpoint
    if args.blender:
        version = subprocess.check_output([str(args.blender), '--version'], text=True, timeout=30).splitlines()[0]
        if not version.startswith('Blender 3.'):
            raise RuntimeError('GLB rendering requires Blender 3.x; found ' + version)
        print('BLENDER ' + str(args.blender) + ' | ' + version, flush=True)
        metadata['blender'] = {'path': str(args.blender), 'version': version}
    if args.input_type == 'slat':
        for path in (args.src1, args.src2):
            read_slat(path)
    return metadata


def configuration(args, old=None):
    old = old or {}
    previous_files = {f['path']: f for f in old.get('inputs', []) + old.get('checkpoints', [])}
    inputs = [fingerprint(path, previous_files.get(str(path))) for source in (args.src1, args.src2)
              for path in input_files(source, args.input_type)]
    checkpoints = [fingerprint(path, previous_files.get(str(path)))
                   for path in (args.ss_checkpoint, args.slat_checkpoint)]
    core_paths = [PROJECT / 'eval_validation_latents.py', PROJECT / 'models/morph_flow.py',
                  PROJECT / 'models/morph_slat_flow.py', PROJECT / 'models/lora.py',
                  args.trellis_root / 'trellis/utils/postprocessing_utils.py']
    if args.input_type == 'glb':
        core_paths += [PROJECT / 'trellis_asset_codec.py',
                       args.trellis_root / 'dataset_toolkits/blender_script/render.py']
    return {'version': VERSION, 'inputs': inputs, 'checkpoints': checkpoints,
            'source_paths': [str(args.src1), str(args.src2)], 'input_type': args.input_type,
            'trellis_root': str(args.trellis_root), 'source_hashes': {str(p): sha256(p) for p in core_paths},
            'blender': str(args.blender) if args.blender else None,
            'sampling': {'steps_ss': args.steps_ss, 'steps_slat': args.steps_slat,
                         'cfg_ss': args.cfg_ss, 'cfg_slat': args.cfg_slat, 'ss_seed': args.seed,
                         'slat_seed': args.slat_seed, 'mixed_precision': args.precision, 'allow_tf32': args.allow_tf32,
                         **({'cudnn_tf32': args.cudnn_tf32} if args.cudnn_tf32 != args.allow_tf32 else {})},
            'encoding': {'views': args.encoding_views, 'dino_batch_size': args.dino_batch_size, 'lighting': args.lighting,
                         **({'src1_up_axis': args.src1_up_axis, 'src2_up_axis': args.src2_up_axis}
                            if args.input_type == 'glb' else {})},
            'texture': {'size': args.texture_size, 'simplify': args.simplify, 'fill_holes': args.fill_holes}}


def load_model(path, target, device):
    from eval_validation_latents import build_model, detect_model_type
    checkpoint = load_tensor(path, checkpoint=True)
    model = build_model(checkpoint, detect_model_type(checkpoint, 'auto'), target).eval().requires_grad_(False)
    del checkpoint
    return model.to(device)


def prepare_inputs(args):
    from trellis_asset_codec import OfficialAssetCodec
    import inspect
    if args.input_type == 'glb' and 'input_up_axis' not in inspect.signature(OfficialAssetCodec).parameters:
        raise RuntimeError('Update trellis_asset_codec.py together with monster-exp: the installed encoder lacks explicit input-axis support')
    codec = OfficialAssetCodec(trellis_root=args.trellis_root, blender_bin=args.blender,
                               dino_batch_size=args.dino_batch_size, seed=args.seed, lighting=args.lighting,
                               **({'input_up_axis': args.src1_up_axis} if args.input_type == 'glb' else {}))
    packs = []
    for label, source in (('src1', args.src1), ('src2', args.src2)):
        codec.input_up_axis = getattr(args, label + '_up_axis')
        folder = args.output_dir / 'inputs' / label
        folder.mkdir(parents=True, exist_ok=True)
        if ready(folder, 'input'):
            pack = validate_pack(load_tensor(folder / 'slat.pt'))
        else:
            print('INPUT ' + label + ' ' + str(source) +
                  (' (up=' + codec.input_up_axis + ')' if args.input_type == 'glb' else ''), flush=True)
            for previous in (args.output_dir / 'morphs').glob('alpha_*'):
                invalidate(previous, ('ss', 'slat', 'mesh', 'gaussian', 'export'))
            pack = codec.encode(source, folder / 'encoding', num_views=args.encoding_views) if args.input_type == 'glb' else read_slat(source)
            save_tensor(folder / 'slat.pt', {**pack, 'metadata': {'latent_scale': 'raw_decoder', 'source': str(source)}})
            commit(folder, 'input', ['slat.pt'])
        packs.append(pack)
    codec.release()
    del codec
    release()
    return packs


def sample_sources(args, packs, torch, alphas=None):
    from eval_validation_latents import sample_ss, ss_coords_from_latent, sample_slat_on_coords
    from trellis.models import from_pretrained
    device = torch.device('cuda')
    batch = {'src1_feats': packs[0]['feats'], 'src1_coords': packs[0]['coords'],
             'src2_feats': packs[1]['feats'], 'src2_coords': packs[1]['coords']}
    folders = [(alpha, args.output_dir / 'morphs' / alpha_tag(alpha)) for alpha in (alphas or args.alphas)]
    for _, folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    pending = [(a, f) for a, f in folders if not ready(f, 'ss')]
    if pending:
        print('STAGE SS', flush=True)
        model = load_model(args.ss_checkpoint, 'ss', device)
        decoder = from_pretrained(MODEL_REPO + '/' + SS_DECODER).to(device).eval().requires_grad_(False)
        flow = model.sparse_structure_flow
        template = torch.zeros((1, flow.in_channels, flow.resolution, flow.resolution, flow.resolution), device=device)
        for alpha, folder in pending:
            invalidate(folder, ('slat', 'mesh', 'gaussian', 'export'))
            seed_everything(args.seed)
            batch['alpha'] = torch.tensor([alpha], dtype=torch.float32)
            with torch.no_grad():
                latent = sample_ss(model, batch, template, args.steps_ss, device, args.cfg_ss, args.precision)
                coords = ss_coords_from_latent(decoder, latent, device, args.precision)
            if not torch.isfinite(latent).all() or not len(coords):
                raise RuntimeError('Nonfinite SS or empty occupancy at alpha=' + str(alpha))
            save_tensor(folder / 'ss.pt', {'latent': latent.float().cpu(), 'coords': coords.int().cpu()})
            commit(folder, 'ss', ['ss.pt'], alpha=alpha, voxels=len(coords))
            print('SS_COMPLETE alpha=' + str(alpha), flush=True)
            del latent, coords
        del model, decoder, flow, template
        release()
    pending = [(a, f) for a, f in folders if not ready(f, 'slat')]
    if pending:
        print('STAGE SLAT', flush=True)
        model = load_model(args.slat_checkpoint, 'slat', device)
        for alpha, folder in pending:
            invalidate(folder, ('mesh', 'gaussian', 'export'))
            seed_everything(args.slat_seed)
            coords = load_tensor(folder / 'ss.pt')['coords'].to(device)
            batch['alpha'] = torch.tensor([alpha], dtype=torch.float32)
            with torch.no_grad():
                latent = sample_slat_on_coords(model, batch, coords, args.steps_slat, device, args.cfg_slat, args.precision)
            if latent is None or not torch.isfinite(latent.feats).all():
                raise RuntimeError('Invalid SLat at alpha=' + str(alpha))
            save_tensor(folder / 'slat.pt', {'feats': latent.feats.float().cpu(), 'coords': latent.coords.int().cpu(),
                                            'metadata': {'latent_scale': 'raw_decoder', 'alpha': alpha}})
            commit(folder, 'slat', ['slat.pt'], alpha=alpha)
            print('SLAT_COMPLETE alpha=' + str(alpha), flush=True)
            del latent, coords
        del model
        release()
    return folders


def decode_assets(folders, torch):
    from eval_validation_latents import load_decoders
    from trellis.models import from_pretrained
    from trellis.modules.sparse import SparseTensor
    pending = [(a, f) for a, f in folders if not ready(f, 'mesh')]
    if pending:
        print('STAGE MESH_DECODER_FP32', flush=True)
        _, decoder, _ = load_decoders('slat', torch.device('cuda'))
        decoder.requires_grad_(False)
        for alpha, folder in pending:
            invalidate(folder, ('export',))
            pack = load_tensor(folder / 'slat.pt')
            sparse = SparseTensor(feats=pack['feats'].cuda(), coords=pack['coords'].cuda())
            with torch.no_grad():
                mesh = decoder(sparse)[0]
            if not mesh.success or not torch.isfinite(mesh.vertices).all():
                raise RuntimeError('Mesh decoding failed at alpha=' + str(alpha))
            save_tensor(folder / 'mesh.pt', {'vertices': mesh.vertices.float().cpu(), 'faces': mesh.faces.long().cpu(),
                        'vertex_attrs': mesh.vertex_attrs.float().cpu() if mesh.vertex_attrs is not None else None,
                        'res': int(mesh.res)})
            commit(folder, 'mesh', ['mesh.pt'], alpha=alpha)
            del pack, sparse, mesh
        del decoder
        release()
    pending = [(a, f) for a, f in folders if not ready(f, 'gaussian')]
    if pending:
        print('STAGE GAUSSIAN_DECODER', flush=True)
        decoder = from_pretrained(MODEL_REPO + '/' + GS_DECODER).cuda().eval().requires_grad_(False)
        for alpha, folder in pending:
            invalidate(folder, ('export',))
            pack = load_tensor(folder / 'slat.pt')
            sparse = SparseTensor(feats=pack['feats'].cuda(), coords=pack['coords'].cuda())
            with torch.no_grad():
                gaussian = decoder(sparse)[0]
            state = {'init_params': gaussian.init_params, 'active_sh_degree': int(gaussian.active_sh_degree),
                     'tensors': {name: getattr(gaussian, name).float().cpu() if getattr(gaussian, name) is not None else None
                                 for name in GS_FIELDS}}
            if any(not torch.isfinite(v).all() for v in state['tensors'].values() if v is not None):
                raise RuntimeError('Nonfinite Gaussian appearance at alpha=' + str(alpha))
            save_tensor(folder / 'gaussian.pt', state)
            commit(folder, 'gaussian', ['gaussian.pt'], alpha=alpha)
            del pack, sparse, gaussian, state
        del decoder
        release()


def export_assets(args, folders, torch):
    import numpy as np
    import trimesh
    from trellis.representations import Gaussian, MeshExtractResult
    from trellis.utils.postprocessing_utils import to_glb
    for alpha, folder in folders:
        if ready(folder, 'export'):
            print('REUSE textured alpha=' + str(alpha), flush=True)
            continue
        seed_everything(args.seed)
        print('TEXTURE_START alpha=' + str(alpha) + ' (full UV atlas, 100 views, optimized texture)', flush=True)
        data, state = load_tensor(folder / 'mesh.pt'), load_tensor(folder / 'gaussian.pt')
        vertices, faces = data['vertices'].numpy(), data['faces'].numpy()
        attrs = data['vertex_attrs']
        colors = np.clip(attrs[:, :3].numpy() * 255, 0, 255).astype(np.uint8) if attrs is not None else None
        rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        vertex_mesh = trimesh.Trimesh(vertices @ rotation, faces, vertex_colors=colors, process=False)
        temp = folder / 'prediction.tmp.glb'
        vertex_mesh.export(temp)
        temp.replace(folder / 'prediction.glb')
        mesh = MeshExtractResult(vertices=data['vertices'].cuda(), faces=data['faces'].cuda(),
                                 vertex_attrs=attrs.cuda() if attrs is not None else None, res=data['res'])
        gaussian = Gaussian(**state['init_params'], device='cuda')
        for name, value in state['tensors'].items():
            setattr(gaussian, name, value.cuda() if value is not None else None)
        gaussian.active_sh_degree = state['active_sh_degree']
        # Texture baking optimizes parameters: do not wrap it in no_grad/inference_mode.
        with torch.inference_mode(False), torch.enable_grad():
            output = to_glb(gaussian, mesh, simplify=args.simplify, fill_holes=args.fill_holes,
                            texture_size=args.texture_size, verbose=False)
        temp = folder / 'prediction_textured.tmp.glb'
        output.export(temp)
        temp.replace(folder / 'prediction_textured.glb')
        for name in ('prediction.glb', 'prediction_textured.glb'):
            error = glb_error(folder / name, textured=name == 'prediction_textured.glb')
            if error:
                raise RuntimeError(name + ': ' + error)
        commit(folder, 'export', ['prediction.glb', 'prediction_textured.glb'], alpha=alpha,
               texture_size=args.texture_size, appearance='Gaussian decoder of generated SLat; no source texture transfer')
        print('TEXTURE_COMPLETE ' + str(folder / 'prediction_textured.glb'), flush=True)
        del data, state, mesh, gaussian, output, vertex_mesh, attrs
        release()


def gallery(root, alphas, status):
    results, rows = [], []
    for alpha in alphas:
        folder = root / 'morphs' / alpha_tag(alpha)
        complete = ready(folder, 'export')
        item = {'alpha': alpha, 'status': 'complete' if complete else 'pending',
                'textured_glb': str(folder / 'prediction_textured.glb'), 'slat': str(folder / 'slat.pt')}
        results.append(item)
        links = []
        for name, label in (('prediction_textured.glb', 'Textured GLB'), ('prediction.glb', 'Vertex-color GLB'), ('slat.pt', 'SLat')):
            if (folder / name).is_file():
                links.append('<a href="' + escape((folder / name).relative_to(root).as_posix(), quote=True) + '">' + label + '</a>')
        rows.append('<tr><td>' + str(alpha) + '</td><td>' + item['status'] + '</td><td>' + ' | '.join(links) + '</td></tr>')
    summary = {'status': status, 'alpha_convention': '1=src1, 0=src2', 'results': results,
               'texturing': 'Official TRELLIS Gaussian-based texture baking; 100 views; no source-color transfer',
               'axes': 'GLBs Y-up; SLat/mesh.pt internal Z-up'}
    write_json(root / 'summary.json', summary)
    page = ('<!doctype html><meta charset="utf-8"><title>MorphFlow assets</title>'
            '<style>body{font:16px system-ui;margin:32px}td,th{padding:12px;border:1px solid #ccc}table{border-collapse:collapse}</style>'
            '<h1>MorphFlow: ' + status + '</h1><p>alpha=1: source 1; alpha=0: source 2. '
            'Every requested alpha is sampled by the model. Textures come from the generated SLat.</p>'
            '<table><tr><th>alpha</th><th>Status</th><th>Outputs</th></tr>' + ''.join(rows) + '</table>')
    (root / 'index.html').write_text(page)
    return summary


def run(args, metadata, torch):
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process is already using ' + str(root)) from exc
        manifest_path = root / 'run.json'
        old = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        if old is None and any(p.name != '.run.lock' for p in root.iterdir()):
            raise FileExistsError('Choose an empty output directory or resume a directory containing run.json')
        config = configuration(args, old.get('config') if old else None)
        if old and old['config'] != config:
            raise ValueError('Inputs/checkpoints/settings changed. Choose a different --output-dir. Only extending --alphas is allowed.')
        all_alphas = list(dict.fromkeys((old.get('all_alphas', []) if old else []) + args.alphas))
        manifest = {'status': 'running', 'config': config, 'checkpoints': metadata,
                    'requested_alphas': args.alphas, 'all_alphas': all_alphas,
                    'runner_sha256': sha256(Path(__file__)),
                    'runtime': {'python': sys.executable, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                                'gpu': torch.cuda.get_device_name(0), 'trellis': str(args.trellis_root)},
                    'alpha_convention': '1=src1, 0=src2',
                    'sampling': 'Common SS/SLat seed across alphas; sparse support changes spatial noise assignment.'}
        write_json(manifest_path, manifest)
        started = time.monotonic()
        try:
            packs = prepare_inputs(args)
            # Codec intentionally disables TF32; restore the requested flow setting.
            torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
            torch.backends.cudnn.allow_tf32 = args.cudnn_tf32
            folders = sample_sources(args, packs, torch, all_alphas)
            del packs
            decode_assets(folders, torch)
            export_assets(args, folders, torch)
            manifest.update(status='complete', seconds_last_invocation=time.monotonic() - started,
                            peak_gpu_allocated_gib=max((old or {}).get('peak_gpu_allocated_gib', 0), torch.cuda.max_memory_allocated() / 2**30),
                            peak_gpu_reserved_gib=max((old or {}).get('peak_gpu_reserved_gib', 0), torch.cuda.max_memory_reserved() / 2**30))
            write_json(manifest_path, manifest)
            summary = gallery(root, all_alphas, 'complete')
            (root / 'last_error.json').unlink(missing_ok=True)
            print(json.dumps(summary, indent=2), flush=True)
        except Exception as exc:
            manifest['status'] = 'interrupted'
            write_json(manifest_path, manifest)
            write_json(root / 'last_error.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
            gallery(root, all_alphas, 'incomplete')
            raise


def main(argv=None):
    args = parse_args(argv)
    torch = runtime(args)
    metadata = preflight(args)
    if args.mode == 'check':
        print(json.dumps({'status': 'check_passed', 'input_type': args.input_type,
                          'checkpoints': metadata, 'trellis_root': str(args.trellis_root),
                          'precision': args.precision, 'alphas': args.alphas}, indent=2), flush=True)
        return
    run(args, metadata, torch)


if __name__ == '__main__':
    main()
