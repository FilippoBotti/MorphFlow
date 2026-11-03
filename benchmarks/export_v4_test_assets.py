"""Export v4 test SLat sources with the exact direct codec benchmark pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))

from benchmark_trellis_codec import prepare, write_json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_source(item):
    import torch
    folder = Path(item['directory'])
    feats = torch.load(folder / 'slat_feats.pt', map_location='cpu', weights_only=True).float().contiguous()
    coords = torch.load(folder / 'slat_coords.pt', map_location='cpu', weights_only=True).int().contiguous()
    if coords.shape[1] == 3:
        coords = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32), coords], 1)
    return {'feats': feats, 'coords': coords}


def save_asset(args, item, source, decode_metadata):
    destination = args.assets / (item['asset'] + '.glb')
    metadata = destination.with_suffix('.json')
    sha = digest(source)
    if destination.exists():
        if digest(destination) != sha:
            raise FileExistsError('A different asset already exists: ' + str(destination))
    else:
        temporary = destination.with_suffix('.tmp.glb')
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    write_json(metadata, {'status': 'complete', 'asset': item['asset'], 'split': 'test',
                         'source_slat': item['directory'], 'sha256': sha,
                         'method': 'OfficialAssetCodec.decode(source SLat, textured=True)',
                         'reference_overview': str(args.reference / 'overview.png'),
                         'decode': decode_metadata, 'glb': str(destination)})


def prepare_exports(args):
    from trellis_asset_codec import MESH_DECODER, GAUSSIAN_DECODER
    prepare(args)
    args.assets.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.output / 'manifest.json').read_text())
    write_json(args.output / 'export_config.json', {
        'assets': str(args.assets), 'reference': str(args.reference),
        'pipeline': 'dataset source SLat -> official mesh and Gaussian decoders -> official to_glb',
        'texture_size': args.texture_size, 'simplify': 0, 'fill_holes': False,
        'source_appearance': 'Gaussian decoded directly from original dataset source SLat',
        'flow_sampling': False, 'glb_reencoding': False,
        'code_sha256': {str(p): digest(p) for p in (Path(__file__), PROJECT / 'trellis_asset_codec.py',
                         BASE / 'src/TRELLIS/trellis/utils/postprocessing_utils.py')}})
    reused = []
    for item in manifest['assets']:
        folder = args.reference / ("%02d_%s" % (item['index'], item['asset']))
        source, provenance = folder / 'direct/textured.glb', folder / 'direct/decode.json'
        if not source.is_file() or not provenance.is_file():
            continue
        old = json.loads(provenance.read_text())
        pack = load_source(item)
        latent_sha = hashlib.sha256(pack['feats'].numpy().tobytes() + pack['coords'].numpy().tobytes()).hexdigest()
        expected = {'latent_sha256': latent_sha, 'latent_scale': 'raw_decoder',
                    'mesh_decoder': MESH_DECODER, 'gaussian_decoder': GAUSSIAN_DECODER,
                    'textured': True, 'seed': 42, 'texture_size': args.texture_size,
                    'simplify': 0.0, 'fill_holes': False, 'trellis_root': str(BASE / 'src/TRELLIS')}
        if old.get('status') != 'complete' or any(old['identity'].get(k) != v for k, v in expected.items()):
            raise ValueError('Reference decoding differs from requested pipeline: ' + str(provenance))
        save_asset(args, item, source, old)
        preview = folder / 'official_150/comparison/source_view_07.png'
        if preview.is_file():
            (args.output / 'previews').mkdir(exist_ok=True)
            shutil.copy2(preview, args.output / 'previews' / (item['asset'] + '.png'))
        reused.append(item['index'])
    pending = [item['index'] for item in manifest['assets'] if item['index'] not in reused]
    write_json(args.output / 'preparation.json', {'reused_reference_indices': reused, 'pending_indices': pending})
    print(json.dumps({'reused': reused, 'pending': pending}, indent=2))


def render_preview(args, item):
    from trellis_codec_metrics import render_glb, heldout_cameras
    import numpy as np
    from PIL import Image
    preview = args.output / 'previews' / (item['asset'] + '.png')
    view = render_glb(args.assets / (item['asset'] + '.glb'), cameras=heldout_cameras()[-1:], resolution=512)[0]
    preview.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.dstack((view['rgb'], view['alpha']))
    temporary = preview.with_suffix('.tmp.png')
    Image.fromarray(np.clip(rgba * 255, 0, 255).astype(np.uint8), 'RGBA').save(temporary)
    temporary.replace(preview)


def refresh_reference_previews(args):
    manifest = json.loads((args.output / 'manifest.json').read_text())
    reused = json.loads((args.output / 'preparation.json').read_text())['reused_reference_indices']
    for index in reused:
        item = manifest['assets'][index]
        render_preview(args, item)
        print('PREVIEW_COMPLETE ' + item['asset'], flush=True)


def worker(args):
    from trellis_asset_codec import OfficialAssetCodec
    manifest = json.loads((args.output / 'manifest.json').read_text())
    item = manifest['assets'][args.index]
    status = args.output / 'status' / (item['asset'] + '.json')
    try:
        write_json(status, {'stage': 'decoding', 'index': args.index, 'asset': item['asset']})
        codec = OfficialAssetCodec(trellis_root=BASE / 'src/TRELLIS',
                                   texture_size=manifest['export']['texture_size'], seed=manifest['seed'])
        decoded = args.output / 'decoded' / item['asset']
        result = codec.decode(load_source(item), decoded, textured=True)
        save_asset(args, item, decoded / 'textured.glb', result)
        codec.release()
        preview = args.output / 'previews' / (item['asset'] + '.png')
        if not preview.is_file():
            render_preview(args, item)
        write_json(status, {'stage': 'complete', 'index': args.index, 'asset': item['asset']})
        print('ASSET_COMPLETE ' + item['asset'], flush=True)
    except Exception as exc:
        write_json(status, {'stage': 'failed', 'index': args.index, 'asset': item['asset'],
                            'error': str(exc), 'traceback': traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'worker', 'previews'))
    parser.add_argument('--dataset', type=Path, default=Path('/leonardo_scratch/fast/IscrC_MORPHFL/mbarezzi/datasets/morphing_dataset_v4'))
    parser.add_argument('--output', type=Path, default=BASE / 'outputs/test_v4_50_gaussian2048_20261103')
    parser.add_argument('--assets', type=Path, default=BASE / 'assets/test_v4_50_gaussian2048')
    parser.add_argument('--reference', type=Path, default=BASE / 'outputs/trellis_codec_v4_test10_20261030')
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--texture-size', type=int, default=2048)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    for key in ('dataset', 'output', 'assets', 'reference'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    {'prepare': prepare_exports, 'worker': worker, 'previews': refresh_reference_previews}[args.mode](args)


if __name__ == '__main__':
    main()
