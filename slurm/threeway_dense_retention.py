#!/usr/bin/env python3
"""Release dense raw assets only after all expected LPIPS pair metrics are merged.

Never releases the seven-state protocol. Keeps all PNGs, source references,
pair metrics, timings and summaries. A durable completion marker supports
resuming after a timeout during cleanup without regenerating retired assets.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


METRICS = ('weighted_endpoint_lpips_mean', 'weighted_endpoint_lpips_auc',
           'weighted_endpoint_lpips_full_mean', 'weighted_endpoint_lpips_full_auc')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root, method, pair=None):
    root = Path(root).resolve()
    plan_path = root/'plan.json'
    plan = json.loads(plan_path.read_text())
    if plan.get('metric_mode') != 'weighted-endpoints-only' or plan.get('k_intermediate') != 48:
        raise ValueError('Raw retention cleanup is restricted to the 48-intermediate dense protocol')
    expected = {p['pair_dir'] for p in plan['pairs']}
    if len(expected) != plan['num_pairs'] or any(Path(p).name != p or not p.startswith('pair_') for p in expected):
        raise ValueError('Invalid dense pair identities')
    output = root/'metrics'/method
    batch_path = output/'metrics_batch.json'
    if pair is None:
        batch = json.loads(batch_path.read_text())
        if (batch.get('metric_mode') != 'weighted-endpoints-only' or batch.get('num_pairs') != len(expected)
                or batch.get('num_generated_frames') != 48 or batch.get('num_sequence_frames') != 50
                or batch.get('missing_pairs') or {p['sequence'] for p in batch['pairs']} != expected):
            raise ValueError('Dense metrics are not complete for every planned pair')
    else:
        if pair not in expected:
            raise ValueError('Pair is not in the frozen dense plan')
        expected = {pair}
        batch = {'num_views_per_pair':64}
    hashes = {}
    for pair in sorted(expected):
        path = output/pair/'metrics.json'
        result = json.loads(path.read_text())
        if (result.get('metric_mode') != 'weighted-endpoints-only' or result.get('num_generated_frames') != 48
                or result.get('num_sequence_frames') != 50 or len(result.get('weighted_endpoint_curve', [])) != 50
                or not result.get('weighted_endpoint_includes_reference_endpoints')):
            raise ValueError('Invalid dense pair metrics: '+pair)
        if result.get('num_views') != batch.get('num_views_per_pair'):
            raise ValueError('Different view counts: '+pair)
        if not all(math.isfinite(float(result['aggregate_across_views'][m]['mean'])) for m in METRICS):
            raise ValueError('Non-finite dense pair metrics: '+pair)
        hashes[pair] = sha256(path)
    return dict(plan_sha256=sha256(plan_path), batch_sha256=sha256(batch_path) if pair is None else None,
                pair_metrics_sha256=hashes, num_pairs=len(expected), method=method, pair=pair)


def release(root, method, pair=None):
    root = Path(root).resolve()
    manifest = validate(root, method, pair)
    marker = (root/method/pair if pair else root/method)/'raw_retired.json'
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    temporary.replace(marker)
    count, size = 0, 0
    for pair in manifest['pair_metrics_sha256']:
        directory = root/method/pair
        if directory.is_symlink():
            raise ValueError('Unexpected symlink at generated pair root')
        for path in directory.glob('alpha_*/mesh_raw.*'):
            if path.name not in ('mesh_raw.pt', 'mesh_raw.appearance.json'):
                continue
            if path.parent.is_symlink():
                raise ValueError('Unexpected symlink at alpha directory')
            size += path.lstat().st_size
            path.unlink()
            count += 1
    print('DENSE RAW RETIRED: method=%s pairs=%d files=%d bytes=%d; renders and metrics retained'
          % (method, manifest['num_pairs'], count, size), flush=True)
    return manifest


def retired(root, method, pair=None):
    root = Path(root).resolve()
    marker = (root/method/pair if pair else root/method)/'raw_retired.json'
    if not marker.is_file():
        return False
    recorded = json.loads(marker.read_text())
    return recorded == validate(root, method, pair)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--method', choices=['morphflow','morphany3d','interp3d'], required=True)
    parser.add_argument('--pair', help='Stream and release a single planned pair before whole-method merge')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--release', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.release:
        release(args.run_root, args.method, args.pair)
    elif not retired(args.run_root, args.method, args.pair):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
