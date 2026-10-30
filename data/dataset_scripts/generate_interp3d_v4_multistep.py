#!/usr/bin/env python3
"""Plan and finalize jointly generated Interp3D datasets with 8 and 2 bins.

Both views have the same sample budget. The 8-bin pairs are a prefix of the
2-bin pairs; their 2-bin targets are selected from the generated 8-bin targets.
The remaining pairs need only two teacher targets. GPU generation uses the
existing generate_interp3d_v4.py assets/pairs commands on the master root.

All tensors live in the master root. Each view has independent metadata and
relative assets/targets links, so moving the complete root preserves both.
This module requires only the Python standard library.
"""
import argparse
import math
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

try:
    from . import generate_interp3d_v4 as base
except ImportError:
    # Also support direct CLI execution and importlib-based CPU tests.
    _script_dir = str(Path(__file__).resolve().parent)
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    import generate_interp3d_v4 as base


VIEW_NAMES = {'steps8': 'morphing_dataset_v4_steps8',
              'steps2': 'morphing_dataset_v4_steps2'}
FORMAT = 'morphflow_interp3d_v4_multistep'
SAMPLING_VERSION = 1


def endpoint_count(num_targets, fraction):
    """Return the legacy even endpoint count, split equally between 0 and 1."""
    return int(round(num_targets * fraction / 2.0)) * 2


def _alpha_schedules(num_pairs, k, seed, split, alpha_min, alpha_max,
                     endpoints_per_side):
    if 2 * endpoints_per_side > num_pairs:
        raise ValueError('Too many endpoint targets for one endpoint per pair')
    rng = random.Random(base.stable_seed('interp3d-v4-multistep-lhs', seed, split, k))
    schedules = []
    for _ in range(num_pairs):
        values = [round(alpha_min + (alpha_max - alpha_min) *
                        (j + rng.random()) / float(k), 6) for j in range(k)]
        if len(set(values)) != k or any(not 0 < v < 1 for v in values):
            raise ValueError('Alpha range is too narrow for distinct six-decimal samples')
        schedules.append(sorted(values, reverse=True))
    order = list(range(num_pairs))
    rng.shuffle(order)
    for i in order[:endpoints_per_side]:
        schedules[i][0] = 1.0
    for i in order[endpoints_per_side:2 * endpoints_per_side]:
        schedules[i][-1] = 0.0
    return schedules


def build_alpha_schedules(num_pairs8, num_pairs2, seed, split,
                          endpoint_fraction, alpha_min, alpha_max):
    """Return actual teacher schedules and selected 2-bin schedules.

    Shared pairs occupy the first num_pairs8 positions. One selected target
    comes from each group of four fine bins. Each view keeps its own balanced
    endpoint quota; unselected endpoints are excluded from random selection.
    Extra 2-bin pairs absorb the rest of the 2-bin view's endpoint quota.
    """
    if num_pairs8 < 0 or num_pairs2 < num_pairs8:
        raise ValueError('Require 0 <= num_pairs8 <= num_pairs2')
    if not math.isfinite(endpoint_fraction) or not 0 <= endpoint_fraction < 1:
        raise ValueError('Endpoint fraction must be in [0,1)')
    if not 0 < alpha_min < alpha_max < 1:
        raise ValueError('Require 0 < alpha_min < alpha_max < 1')
    per_side8 = endpoint_count(num_pairs8 * 8, endpoint_fraction) // 2
    per_side2 = endpoint_count(num_pairs2 * 2, endpoint_fraction) // 2
    shared = _alpha_schedules(num_pairs8, 8, seed, split, alpha_min,
                              alpha_max, per_side8)
    # A coarse bin merges four fine bins. Preserve the corresponding fraction
    # of fine-view endpoints, subject to the complete coarse-view quota.
    shared_per_side2 = min(int(round(per_side8 / 4.0)), per_side2)
    rng = random.Random(base.stable_seed('interp3d-v4-multistep-subset', seed, split))
    preserve = {}
    for endpoint in (1.0, 0.0):
        candidates = [i for i, alphas in enumerate(shared) if endpoint in alphas]
        rng.shuffle(candidates)
        preserve[endpoint] = set(candidates[:shared_per_side2])
    selected = []
    for i, alphas in enumerate(shared):
        chosen = []
        for half, endpoint in ((alphas[:4], 1.0), (alphas[4:], 0.0)):
            if i in preserve[endpoint]:
                chosen.append(endpoint)
            else:
                chosen.append(rng.choice([a for a in half if 0 < a < 1]))
        selected.append(chosen)
    extra = _alpha_schedules(num_pairs2 - num_pairs8, 2, seed,
                             split + '-extra', alpha_min, alpha_max,
                             per_side2 - shared_per_side2)
    return shared + extra, selected + [list(a) for a in extra]


def _target_counts(pairs):
    counts = {split: {'interior': 0, 'alpha_0': 0, 'alpha_1': 0}
              for split in base.SPLITS}
    for pair in pairs:
        for alpha in pair['alphas']:
            key = 'alpha_0' if alpha == 0 else 'alpha_1' if alpha == 1 else 'interior'
            counts[pair['split']][key] += 1
    return counts


def _view_plan(master, key):
    spec = master['views'][key]
    pairs = []
    active = {split: set() for split in base.SPLITS}
    for source in master['pairs']:
        if key == 'steps8' and not source['shared_with_steps8']:
            continue
        pair = {name: source[name] for name in
                ('pair_id', 'split', 'src_1', 'src_2', 'seed')}
        pair['alphas'] = list(source['alphas'] if key == 'steps8'
                              else source['selected_alphas_steps2'])
        pair['teacher_alphas'] = list(source['alphas'])
        active[pair['split']].update((pair['src_1'], pair['src_2']))
        pairs.append(pair)
    names = set().union(*active.values())
    plan = {name: master[name] for name in
            ('version', 'seed', 'endpoint_fraction', 'alpha_min', 'alpha_max',
             'v3_root', 'images_root', 'teacher', 'alpha_definition', 'symmetry',
             'endpoint_target_policy', 'sampling_version')}
    plan.update({'format': 'morphflow_interp3d_v4_nested_view',
                 'view': key, 'master_root': '..', 'k': spec['k'],
                 'pair_quotas': spec['pair_quotas'],
                 'num_samples_per_pair': spec['k'],
                 'sampling': '%d equal alpha bins per pair; targets reused from master' % spec['k'],
                 'num_pairs': len(pairs), 'num_targets': spec['num_targets'],
                 'train_virtual_after_swap': master['sample_quotas']['train'] * 2,
                 'target_counts': _target_counts(pairs), 'pairs': pairs,
                 'assets': [a for a in master['assets'] if a['name'] in names],
                 'split_assets': {s: sorted(active[s]) for s in base.SPLITS}})
    return plan


def _ensure_link(path, target):
    if path.is_symlink():
        if os.readlink(str(path)) != target:
            raise RuntimeError('Existing link points elsewhere: ' + str(path))
    elif os.path.lexists(str(path)):
        raise RuntimeError('Refusing to replace existing path: ' + str(path))
    else:
        os.symlink(target, str(path))


def _prepare_views(out, master):
    """Validate all existing view paths before creating missing ones."""
    pending = []
    for key in VIEW_NAMES:
        root = out / VIEW_NAMES[key]
        plan = _view_plan(master, key)
        if os.path.lexists(str(root)) and (root.is_symlink() or not root.is_dir()):
            raise RuntimeError('View root must be a directory: ' + str(root))
        path = root / 'plan.json'
        if path.exists() and base.read_json(path) != plan:
            raise RuntimeError('Incompatible view plan: ' + str(path))
        if root.is_dir():
            for name in ('assets', 'targets'):
                link = root / name
                if os.path.lexists(str(link)) and (
                        not link.is_symlink() or os.readlink(str(link)) != '../' + name):
                    raise RuntimeError('Refusing to replace view storage: ' + str(link))
        pending.append((root, path, plan))
    for root, path, plan in pending:
        root.mkdir(parents=True, exist_ok=True)
        _ensure_link(root / 'assets', '../assets')
        _ensure_link(root / 'targets', '../targets')
        if not path.exists():
            base.save_json(path, plan)


def command_plan(args):
    out = Path(args.out).resolve()
    v3 = Path(args.v3_root).resolve()
    images = Path(args.images_root).resolve()
    samples = {s: getattr(args, s + '_samples') for s in base.SPLITS}
    if any(n < 0 or n % 8 for n in samples.values()) or sum(samples.values()) == 0:
        raise ValueError('Sample quotas must be nonnegative multiples of 8, with at least one pair')
    if sum(samples.values()) > base.MAX_DATASET_SAMPLES:
        raise ValueError('Each dataset exceeds the stored-target cap')
    if samples['train'] * 2 > base.MAX_VIRTUAL_TRAIN_SAMPLES:
        raise ValueError('Source-swapped training cap exceeded')
    if not 0 < args.alpha_min < args.alpha_max < 1:
        raise ValueError('Require 0 < alpha_min < alpha_max < 1')
    if not math.isfinite(args.endpoint_fraction) or not 0 <= args.endpoint_fraction < 1:
        raise ValueError('Endpoint fraction must be in [0,1)')
    config = {'format': FORMAT, 'version': 4, 'sampling_version': SAMPLING_VERSION,
              'sample_quotas': samples, 'seed': args.seed,
              'endpoint_fraction': args.endpoint_fraction,
              'alpha_min': args.alpha_min, 'alpha_max': args.alpha_max,
              'v3_root': str(v3), 'images_root': str(images)}
    plan_path = out / 'plan.json'
    if plan_path.exists():
        master = base.read_json(plan_path)
        if any(master.get(k) != v for k, v in config.items()):
            raise RuntimeError('Incompatible plan in OUT_ROOT: use a new output root')
        _prepare_views(out, master)
        print('Reusing joint plan:', plan_path, flush=True)
        return
    # A plan is committed before any storage/view directories. If no plan is
    # present, existing dataset paths cannot belong to a resumable joint run.
    for name in ('assets', 'targets') + tuple(VIEW_NAMES.values()):
        if os.path.lexists(str(out / name)):
            raise RuntimeError('Dataset paths exist without a joint plan: ' + str(out / name))
    inventory = base.image_inventory(images)
    assignments = base.read_json(v3 / 'split_assets.json')
    split_images = {}
    seen = set()
    for split in base.SPLITS:
        available = sorted(set(assignments[split]).intersection(inventory))
        if seen.intersection(available):
            raise RuntimeError('Asset leakage between splits')
        missing = set(assignments[split]).difference(inventory)
        if missing:
            print('WARNING: split %s: %d missing images excluded' %
                  (split, len(missing)), flush=True)
        seen.update(available)
        split_images[split] = available
    views = {key: {'name': name, 'k': k,
                   'pair_quotas': {s: samples[s] // k for s in base.SPLITS},
                   'num_targets': sum(samples.values()),
                   'train_virtual_after_swap': samples['train'] * 2}
             for key, name, k in (('steps8', VIEW_NAMES['steps8'], 8),
                                  ('steps2', VIEW_NAMES['steps2'], 2))}
    pairs = []
    active = {s: set() for s in base.SPLITS}
    for split in base.SPLITS:
        n8, n2 = samples[split] // 8, samples[split] // 2
        selected_pairs = base.balanced_pairs(split_images[split], n2, args.seed, split)
        teacher, selected2 = build_alpha_schedules(
            n8, n2, args.seed, split, args.endpoint_fraction,
            args.alpha_min, args.alpha_max)
        for i, ((a, b), alphas, subset) in enumerate(zip(selected_pairs, teacher, selected2)):
            pairs.append({'pair_id': len(pairs), 'split': split,
                          'src_1': a, 'src_2': b,
                          'seed': base.stable_seed(args.seed, 'pair', a, b),
                          'alphas': alphas, 'selected_alphas_steps2': subset,
                          'shared_with_steps8': i < n8})
            active[split].update((a, b))
    master = dict(config)
    master.update({'teacher': 'Interp3D', 'k': None,
                   'variable_alphas_per_pair': True, 'num_samples_per_pair': None,
                   'alpha_definition': 'alpha=1 -> src_1; alpha=0 -> src_2',
                   'sampling': '8 equal bins on shared pairs; 2 equal bins on extra pairs',
                   'symmetry': 'same canonical target; source swap with complementary alpha',
                   'endpoint_target_policy': 'relative symlink to cached source latent',
                   'pair_quotas': views['steps2']['pair_quotas'],
                   'train_virtual_after_swap': None,
                   'num_pairs': len(pairs),
                   'num_targets': sum(len(p['alphas']) for p in pairs),
                   'target_counts': _target_counts(pairs), 'pairs': pairs,
                   'assets': [{'name': n, 'split': s, 'image': inventory[n]}
                              for s in base.SPLITS for n in sorted(active[s])],
                   'split_assets': {s: sorted(active[s]) for s in base.SPLITS},
                   'views': views})
    out.mkdir(parents=True, exist_ok=True)
    base.save_json(plan_path, master)
    (out / 'assets').mkdir(exist_ok=True)
    (out / 'targets').mkdir(exist_ok=True)
    _prepare_views(out, master)
    print('JOINT PLAN: %d pairs; %d canonical targets' %
          (master['num_pairs'], master['num_targets']), flush=True)
    for key, spec in views.items():
        print('%s: pairs=%s targets=%d train virtual=%d' %
              (spec['name'], spec['pair_quotas'], spec['num_targets'],
               samples['train'] * 2), flush=True)


def command_finalize(args):
    out = Path(args.out).resolve()
    master = base.read_json(out / 'plan.json')
    if master.get('format') != FORMAT:
        raise RuntimeError('Expected a joint multistep master plan')
    _prepare_views(out, master)
    missing = []
    for pair in master['pairs']:
        a, b = pair['src_1'], pair['src_2']
        sources_ready = (base.check_ready(out / 'assets' / a) and
                         base.check_ready(out / 'assets' / b))
        for alpha in pair['alphas']:
            if sources_ready and alpha in (0.0, 1.0):
                base.ensure_endpoint_alias(out, a, b, alpha)
            if not sources_ready or not base.check_ready(base.target_path(out, a, b, alpha)):
                missing.append({'split': pair['split'], 'src_1': a,
                                'src_2': b, 'alpha': alpha})
    base.save_json(out / 'missing_targets.json', missing)
    if missing:
        raise RuntimeError('Incomplete master dataset (%d targets missing); resume the same job' %
                           len(missing))
    summary = {'teacher': 'Interp3D', 'canonical_targets': master['num_targets'],
               'canonical_pairs': master['num_pairs'], 'views': {}}
    for key, spec in master['views'].items():
        root = out / spec['name']
        base.command_finalize(SimpleNamespace(out=str(root)))
        summary['views'][key] = {'root': spec['name'], 'k': spec['k'],
                                 'num_targets': spec['num_targets'],
                                 'pair_quotas': spec['pair_quotas']}
    base.save_json(out / 'generation_summary.json', summary)
    print('COMPLETE: both dataset views share %d canonical targets' %
          master['num_targets'], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='mode')
    plan = sub.add_parser('plan')
    plan.add_argument('--v3-root', required=True)
    plan.add_argument('--images-root', required=True)
    plan.add_argument('--out', required=True)
    plan.add_argument('--train-samples', type=int, default=50000)
    plan.add_argument('--val-samples', type=int, default=2000)
    plan.add_argument('--test-samples', type=int, default=2000)
    plan.add_argument('--endpoint-fraction', type=float, default=0.02)
    plan.add_argument('--seed', type=int, default=42)
    plan.add_argument('--alpha-min', type=float, default=0.001)
    plan.add_argument('--alpha-max', type=float, default=0.999)
    finalize = sub.add_parser('finalize')
    finalize.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.mode == 'plan':
        command_plan(args)
    elif args.mode == 'finalize':
        command_finalize(args)
    else:
        ap.error('Choose plan/finalize')


if __name__ == '__main__':
    main()
