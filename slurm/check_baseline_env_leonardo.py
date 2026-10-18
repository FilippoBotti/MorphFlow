"""Verify fork isolation and compiled runtime extensions, optionally on CUDA."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()
import torch
torch.set_num_threads(4)
modules = ['torchvision', 'xformers.ops', 'spconv.pytorch', 'kaolin', 'utils3d',
           'lpips', 'pytorch_fid.inception', 'scipy', 'pytorch3d.loss', 'nvdiffrast.torch',
           'diffoctreerast', 'diff_gaussian_rasterization', 'trellis.pipelines']
paths = {name: importlib.import_module(name).__file__ for name in modules}
expected = 'MorphAny3D' if 'morphany3d' in sys.prefix else 'Interp3D'
assert f'/src/{expected}/' in paths['trellis.pipelines'], paths['trellis.pipelines']
assert sys.prefix == str(Path(sys.executable).parent.parent), (sys.prefix, sys.executable)
assert 'morphflow/lib' not in ':'.join(sys.path), sys.path
if args.gpu:
    assert torch.cuda.is_available()
    from pytorch3d.loss import chamfer_distance
    points = torch.randn(1, 32, 3, device='cuda')
    loss, _ = chamfer_distance(points, points)
    assert loss.item() < 1e-6
    import nvdiffrast.torch as dr
    ctx = dr.RasterizeCudaContext()
    verts = torch.tensor([[[-.5, -.5, 0., 1.], [.5, -.5, 0., 1.], [0., .5, 0., 1.]]], device='cuda')
    tris = torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda')
    raster, _ = dr.rasterize(ctx, verts, tris, resolution=[32, 32])
    assert raster[..., -1].max().item() == 1
    from trellis.pipelines import TrellisImageTo3DPipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large')
    pipeline.cuda()
    assert len(pipeline.models) == 7, list(pipeline.models)
    from PIL import Image
    image = Image.new('RGBA', (64, 64), (255, 0, 0, 255))
    # The alpha mask avoids calling rembg during this GPU runtime check.
    image.putpixel((0, 0), (0, 0, 0, 0))
    cond = pipeline.get_cond([pipeline.preprocess_image(image)])
    assert torch.isfinite(cond['cond']).all()
    print('GPU_PIPELINE_READY', expected, list(pipeline.models), flush=True)
print(json.dumps({'python': sys.executable, 'prefix': sys.prefix,
                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'gpu': torch.cuda.get_device_name(0) if args.gpu else None,
                  'modules': paths}, indent=2), flush=True)
