"""Print a shareable compatibility report; --gpu exercises the CUDA renderers."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trellis-root', type=Path, default=os.environ.get('TRELLIS_REPO'))
    parser.add_argument('--blender', type=Path, default=os.environ.get('BLENDER_BIN'))
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--input-type', choices=('glb', 'slat'), default='glb')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    trellis = Path(args.trellis_root or PROJECT.parent / 'TRELLIS').expanduser().resolve()
    sys.path.insert(0, str(trellis))
    os.environ['TRELLIS_REPO'] = str(trellis)
    os.environ.setdefault('ATTN_BACKEND', 'xformers')
    os.environ.setdefault('SPCONV_ALGO', 'native')
    report = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
              'trellis_root': str(trellis), 'cuda_home': os.environ.get('CUDA_HOME'),
              'nvcc': shutil.which('nvcc'), 'modules': {}, 'errors': [], 'warnings': []}
    modules = ['torch', 'trellis', 'xformers', 'spconv.pytorch', 'nvdiffrast.torch',
               'diff_gaussian_rasterization', 'utils3d', 'xatlas', 'trimesh',
               'trellis.utils.postprocessing_utils', 'eval_validation_latents']
    if args.input_type == 'glb':
        modules.append('open3d')
    for name in modules:
        try:
            module = importlib.import_module(name)
            report['modules'][name] = {'path': str(getattr(module, '__file__', '')), 'version': str(getattr(module, '__version__', ''))}
            if name == 'trellis' and not Path(module.__file__).resolve().is_relative_to(trellis):
                raise RuntimeError('TRELLIS is imported from a different checkout')
        except Exception as exc:
            report['errors'].append(name + ': ' + type(exc).__name__ + ': ' + str(exc))
    if 'torch' in report['modules']:
        import torch
        report['torch_cuda'] = torch.version.cuda
        report['cuda_available'] = torch.cuda.is_available()
        report['gpus'] = [{'name': torch.cuda.get_device_properties(i).name,
                           'vram_gib': torch.cuda.get_device_properties(i).total_memory / 2**30}
                          for i in range(torch.cuda.device_count())]
        if args.gpu:
            try:
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA unavailable; activate the environment or request a GPU')
                import nvdiffrast.torch as dr
                context = dr.RasterizeCudaContext()
                positions = torch.tensor([[[-.5, -.5, 0, 1], [.5, -.5, 0, 1], [0, .5, 0, 1]]], device='cuda')
                triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda')
                raster, _ = dr.rasterize(context, positions, triangles, resolution=(32, 32))
                if not (raster[..., 3] > 0).any():
                    raise RuntimeError('CUDA rasterizer returned an empty image')
                import utils3d
                if args.input_type == 'glb':
                    # Encoding requires this legacy utils3d projection API.
                    utils3d.torch.intrinsics_from_fov_xy(torch.tensor(1.0), torch.tensor(1.0))
                report['gpu_rasterizer'] = 'passed'
                report['bf16'] = torch.cuda.is_bf16_supported()
            except Exception as exc:
                report['errors'].append('GPU test: ' + type(exc).__name__ + ': ' + str(exc))
    if args.input_type == 'glb':
        try:
            from morph_assets import resolve_blender
            blender = resolve_blender(args.blender)
            version = subprocess.check_output([str(blender), '--version'], text=True, timeout=30).splitlines()[0]
            if not version.startswith('Blender 3.'):
                raise RuntimeError('Use Blender 3.x for the multiview renderer')
            report['blender'] = {'path': str(blender), 'version': version}
            if not (trellis / 'dataset_toolkits/blender_script/render.py').is_file():
                raise FileNotFoundError('Missing official dataset_toolkits/blender_script/render.py')
        except Exception as exc:
            report['errors'].append('Blender/encoding: ' + str(exc))
    if not report['nvcc']:
        report['warnings'].append('nvcc not in PATH; CUDA extensions may need CUDA_HOME on their first build.')
    report['status'] = 'compatible' if not report['errors'] else 'needs_attention'
    if args.report:
        target = args.report.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if not report['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
