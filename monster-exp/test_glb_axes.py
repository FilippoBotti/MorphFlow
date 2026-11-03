"""GPU/Blender regression: equivalent Y-up and Z-up GLBs must encode the same scene.

Only rendering and the normalized mesh are compared; no trained model is loaded.
The test verifies geometry/camera alignment, not the semantic up axis of an asset.
"""
import argparse
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--trellis-root', type=Path, required=True)
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import trimesh
    from PIL import Image
    from scipy.spatial import cKDTree
    from trellis_asset_codec import OfficialAssetCodec
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError('Choose an empty test output directory')
    scene = trimesh.load(args.source, force='scene', process=False)
    scene.export(root / 'fixture_y.glb')
    rotated = scene.copy()
    rotated.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    rotated.export(root / 'fixture_z.glb')
    codec = OfficialAssetCodec(trellis_root=args.trellis_root, blender_bin=args.blender,
                               lighting='uniform_texture', input_up_axis='y', seed=42)
    codec._init()
    codec._render(root / 'fixture_y.glb', root / 'canonical_y', 4)
    codec.input_up_axis = 'z'
    codec._render(root / 'fixture_z.glb', root / 'corrected_z', 4)
    meshes = [trimesh.load(root / name / 'mesh.ply', process=False) for name in ['canonical_y', 'corrected_z']]
    va, vb = [mesh.vertices for mesh in meshes]
    errors = [cKDTree(va).query(vb)[0], cKDTree(vb).query(va)[0]]
    distance = float(max(error.max() for error in errors))
    psnr = []
    for index in range(4):
        images = [np.asarray(Image.open(root / name / ('%03d.png' % index)).convert('RGBA'), dtype=np.float64) / 255
                  for name in ['canonical_y', 'corrected_z']]
        mse = float(np.mean((images[0] - images[1])**2))
        psnr.append(min(120.0, -10 * math.log10(max(mse, 1e-12))))
    report = {'status': 'passed' if distance < 1e-4 and min(psnr) > 35 else 'failed',
              'normalized_mesh_max_distance': distance, 'render_psnr_rgba': psnr,
              'description': 'Equivalent source scenes after correction before normalization/render/mesh export.'}
    (root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    assert report['status'] == 'passed', report


if __name__ == '__main__':
    main()
