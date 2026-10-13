#!/usr/bin/env python3
"""Audit MorphFlow Interp3D v4 (Python >=3.6, no torch required).

Covers generation state, paired alpha schedules, every target/asset tensor file,
endpoint aliases, split leakage and source images required by TRELLIS prior.
Optional --repair-image-manifests writes only missing image manifests and does
not edit tensors, alpha schedules or metadata.
"""
from __future__ import print_function
import argparse
import collections
import json
import os
from pathlib import Path
import sys
import time
import zipfile

EXPECTED = {'train': 50000, 'val': 2000, 'test': 2000}
PAIRS = {'train': 12500, 'val': 500, 'test': 500}
CORE = ('ss_latent.pt', 'slat_feats.pt', 'slat_coords.pt', 'structured_latent.pt', 'occupancy.pt')
SRC_FILE_KEYS = ('src1_ss_latent', 'src1_slat_feats', 'src1_slat_coords', 'src1_occupancy',
                 'src2_ss_latent', 'src2_slat_feats', 'src2_slat_coords', 'src2_occupancy',
                 'target_ss_latent', 'target_slat_feats', 'target_slat_coords', 'target_occupancy')
EXTS = ('.png', '.jpg', '.jpeg', '.webp')


def load(path):
    with open(str(path), 'r') as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--repair-image-manifests', action='store_true')
    args = p.parse_args()
    root = Path(args.out).resolve()
    bad = []
    warn = []
    stats = collections.Counter()
    t0 = time.time()

    def error(msg):
        if len(bad) < 40:
            bad.append(str(msg))
        stats['errors'] += 1

    def warning(msg):
        if len(warn) < 20:
            warn.append(str(msg))
        stats['warnings'] += 1

    def require_file(path):
        try:
            st = path.stat()
            if not path.is_file() or st.st_size <= 0:
                raise OSError('missing, non-regular or empty')
            stats['file_checks'] += 1
            return True
        except (OSError, ValueError) as exc:
            error('{}: {}'.format(path, exc))
            return False

    print('== MORPHFLOW V4 DATASET AUDIT ==', flush=True)
    print('ROOT:', root, flush=True)
    if not root.is_dir():
        raise SystemExit('FAIL: dataset directory does not exist')
    needed = ['run_state.json', 'plan.json', 'split_assets.json', 'splits.json',
              'metadata.json', 'metadata_train.json', 'metadata_val.json',
              'metadata_test.json', 'missing_targets.json']
    for name in needed:
        if not require_file(root / name):
            print('FAIL: required JSON missing: ' + name)
            return 1
    try:
        state = load(root / 'run_state.json')
        plan = load(root / 'plan.json')
        split_assets = load(root / 'split_assets.json')
        splits = load(root / 'splits.json')
        missing = load(root / 'missing_targets.json')
        grouped = {s: load(root / ('metadata_' + s + '.json')) for s in EXPECTED}
        combined = load(root / 'metadata.json')
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit('FAIL reading JSON: {}'.format(exc))

    print('Generator job: {} | phase={} | status={}'.format(state.get('job_id'),
            state.get('phase'), state.get('status')), flush=True)
    if state.get('phase') != 'complete' or state.get('status') != 'complete':
        error('Generation not complete (state/phase).')
    if missing:
        error('missing_targets.json has {} entries'.format(len(missing)))
    if plan.get('k') != 4 or plan.get('num_pairs') != 13500 or plan.get('num_targets') != 54000:
        error('Unexpected plan size: k={} pairs={} targets={}'.format(
            plan.get('k'), plan.get('num_pairs'), plan.get('num_targets')))
    if plan.get('train_virtual_after_swap') != 100000:
        error('Expected 100000 virtual training samples')
    if plan.get('pair_quotas') != PAIRS:
        error('Wrong pair quotas: {}'.format(plan.get('pair_quotas')))
    if not isinstance(plan.get('pairs'), list) or len(plan['pairs']) != 13500:
        error('Incorrect plan.pairs length')
    if not isinstance(plan.get('assets'), list):
        error('Missing plan.assets list')
    for split, n in EXPECTED.items():
        if len(grouped[split]) != n:
            error('metadata_{} length={} expected {}'.format(split, len(grouped[split]), n))
    if len(combined) != 54000:
        error('metadata.json length={} expected 54000'.format(len(combined)))
    if isinstance(splits, dict) and splits.get('missing_planned_targets', 0) != 0:
        error('splits.json reports incomplete targets')

    # Distinct asset split membership; assert metadata sources respect it.
    asset_membership = {}
    for split in EXPECTED:
        for asset in split_assets.get(split, []):
            if asset in asset_membership:
                error('Asset leakage {}: {} and {}'.format(asset, asset_membership[asset], split))
            asset_membership[asset] = split
    print('Assets in split assignments:',
          {s: len(split_assets.get(s, [])) for s in EXPECTED}, flush=True)

    image_root = Path(plan.get('images_root', '/hpc/scratch/marco.barezzi/3d_dataset/flux_outputs'))
    source_misses = []
    repairs = 0
    asset_names = set()
    for asset in plan.get('assets', []):
        name = asset['name']
        split = asset['split']
        asset_names.add(name)
        if asset_membership.get(name) != split:
            error('Asset {} has inconsistent split {}'.format(name, split))
        d = root / 'assets' / name
        for fn in CORE + ('ready.json',):
            require_file(d / fn)
        im = Path(asset['image'])
        if not im.is_file():
            error('Original FLUX image is missing: ' + str(im))
        # Mimic DataLoader lookup for a flat FLUX image folder.
        named_candidates = [image_root / (name + e) for e in EXTS]
        named_candidates.extend(image_root / (name + e) for e in (
            '_image.png', '_input.png', '_source.png', '_generated.png'))
        named_candidates.extend(image_root / name / e for e in ('image.png', 'input.png', 'source.png', 'output.png'))
        manifest_path = d / 'manifest.json'
        manifest_ok = False
        if manifest_path.is_file():
            try:
                raw = load(manifest_path).get('image')
                if raw:
                    candidate = Path(raw)
                    if not candidate.is_absolute():
                        candidate = image_root / candidate
                    manifest_ok = candidate.is_file()
            except (ValueError, OSError, AttributeError):
                pass
        if not manifest_ok and not any(x.is_file() for x in named_candidates):
            if args.repair_image_manifests and im.is_file() and im.parent.resolve() == image_root.resolve():
                # Do not overwrite existing manifest; only create if missing.
                if manifest_path.exists():
                    error('Existing invalid manifest needs manual review: ' + str(manifest_path))
                else:
                    tmp = d / ('.manifest.json.{}.tmp'.format(os.getpid()))
                    with open(str(tmp), 'w') as f:
                        json.dump({'image': im.name, 'note': 'v4 source image lookup'}, f)
                    os.rename(str(tmp), str(manifest_path))
                    repairs += 1
            else:
                source_misses.append(name)
    if source_misses:
        error('TRELLIS prior source images not discoverable for {} assets; e.g. {}'.format(
            len(source_misses), source_misses[:8]))
    if repairs:
        print('Created {} image manifests for TRELLIS prior lookup'.format(repairs), flush=True)
    print('Source assets validated:', len(asset_names), flush=True)

    expected_per_pair = {}
    for item in plan.get('pairs', []):
        pair_key = (item['split'], item['src_1'], item['src_2'])
        if pair_key in expected_per_pair:
            error('Plan duplicates pair {}'.format(pair_key))
        expected_per_pair[pair_key] = tuple(item['alphas'])

    counts = collections.Counter()
    uniques = set()
    by_pair = collections.defaultdict(list)
    endpoints = collections.Counter()
    done_target_paths = set()
    for split, entries in grouped.items():
        for e in entries:
            a, b = e.get('src_1'), e.get('src_2')
            alpha = e.get('alpha')
            pair = (split, a, b)
            counts[split] += 1
            if asset_membership.get(a) != split or asset_membership.get(b) != split or a == b:
                error('Source-split mismatch: {}'.format(pair))
            if not isinstance(alpha, (int, float)) or not (0.0 <= alpha <= 1.0):
                error('Invalid alpha for {}'.format(pair))
                continue
            if alpha in (0.0, 1.0):
                endpoints[(split, int(alpha))] += 1
            key = (split, a, b, alpha)
            if key in uniques:
                error('Duplicate metadata target: {}'.format(key))
            uniques.add(key)
            by_pair[pair].append(alpha)
            td = root / e['target_dir']
            tkey = str(td)
            if tkey not in done_target_paths:
                done_target_paths.add(tkey)
                for fn in CORE + ('ready.json',):
                    require_file(td / fn)
                if alpha in (0.0, 1.0):
                    ref = root / 'assets' / (a if alpha == 1.0 else b)
                    if not td.is_symlink() or td.resolve() != ref.resolve():
                        error('Wrong endpoint link: {} -> expected {}'.format(td, ref))
            # Validate literal paths the DataLoader uses.
            for keyname in SRC_FILE_KEYS:
                pathvalue = e.get(keyname)
                if not pathvalue or not str(pathvalue).endswith('.pt'):
                    error('Bad metadata path {} for {}'.format(keyname, key))
                # All target / asset files already verified above.
    if len(by_pair) != 13500:
        error('Distinct pairs={} expected 13500'.format(len(by_pair)))
    for pair, vals in by_pair.items():
        expected_alphas = expected_per_pair.get(pair)
        if expected_alphas is None or tuple(sorted(vals)) != tuple(sorted(expected_alphas)):
            error('Metadata vs plan alpha mismatch for {}'.format(pair))
        if len(vals) != 4 or len(set(vals)) != 4:
            error('Pair {} has invalid alpha set: {}'.format(pair, vals))
    for split in EXPECTED:
        if sum(1 for p in by_pair if p[0] == split) != PAIRS[split]:
            error('Wrong number of distinct {} pairs'.format(split))
        if endpoints[(split, 0)] != (500 if split == 'train' else 20):
            error('Incorrect alpha=0 count for {}: {}'.format(split, endpoints[(split, 0)]))
        if endpoints[(split, 1)] != (500 if split == 'train' else 20):
            error('Incorrect alpha=1 count for {}: {}'.format(split, endpoints[(split, 1)]))
    print('Metadata target counts:', dict(counts), flush=True)
    print('Endpoint counts:', dict(endpoints), flush=True)
    print('Distinct pairs:', len(by_pair), '| Checked target directories:', len(done_target_paths), flush=True)
    print('Checked tensor/ready files:', stats['file_checks'], flush=True)

    # Verify some ZIP CRCs, which catches truncated / partly written torch.save.
    # Torch saves files in ZIP format by default. This cannot detect numerical NaNs.
    targets_for_crc = sorted(done_target_paths)
    for i in range(32):
        if not targets_for_crc:
            break
        sample_dir = Path(targets_for_crc[i * (len(targets_for_crc) - 1) // 31])
        fn = CORE[i % len(CORE)]
        src = sample_dir / fn
        try:
            with zipfile.ZipFile(str(src), 'r') as z:
                first_bad = z.testzip()
                if first_bad:
                    raise ValueError('Bad CRC in {}'.format(first_bad))
            stats['crc_checks'] += 1
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            error('Corrupted PyTorch archive {}: {}'.format(src, exc))
    print('PyTorch tensor ZIP/CRC samples:', stats['crc_checks'], flush=True)
    if warn:
        print('WARNINGS:')
        for x in warn[:15]:
            print(' -', x)
    if bad:
        print('ERRORS (first {} of {}):'.format(len(bad), stats['errors']))
        for x in bad:
            print(' -', x)
        print('RESULT: FAIL | elapsed {:.1f}s'.format(time.time()-t0), flush=True)
        return 1
    print('RESULT: PASS | elapsed {:.1f}s'.format(time.time()-t0), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
