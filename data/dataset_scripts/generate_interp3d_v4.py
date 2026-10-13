#!/usr/bin/env python3
"""Interp3D teacher -> MorphFlow v3-compatible latent dataset v4.

Modes:
  plan      CPU only; reuse disjoint v3 ASSET splits, build new balanced pairs.
  assets    GPU; compute each source latent once (deterministic per asset).
  pairs     GPU; run native Interp3D interpolation and capture SS/SLat latents.
  finalize  CPU only; build metadata with rare exact endpoint source aliases.

Requires the Interp3D TRELLIS fork for GPU modes; does not modify that fork.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import uuid
import threading
import time
from functools import lru_cache

SPLITS = ('train', 'val', 'test')
MAX_DATASET_SAMPLES = 100000
ALPHAS_PER_PAIR = 4
MAX_VIRTUAL_TRAIN_SAMPLES = 100000
ACTIVE_REPORTER = None
REQUIRED = ('ss_latent.pt', 'slat_feats.pt', 'slat_coords.pt',
            'structured_latent.pt', 'occupancy.pt')
IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.webp')


def stable_seed(*parts):
    digest = hashlib.sha256('|'.join(map(str, parts)).encode('utf8')).digest()
    return int.from_bytes(digest[:8], 'little') % (2 ** 31 - 1)


def stable_shard(name, num_shards):
    return stable_seed('shard', name) % num_shards


def asset_name(path):
    return re.sub(r'[^A-Za-z0-9_.=-]+', '_', path.stem).strip('_') or 'asset'


def alpha_slug(alpha):
    return ('%.6f' % alpha).rstrip('0').rstrip('.').replace('-', 'm').replace('.', 'p')


def read_json(path):
    with open(str(path), 'r') as f:
        return json.load(f)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with open(str(temp), 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(str(temp), str(path))


def check_ready(root):
    return all((root / name).is_file() for name in REQUIRED) and (root / 'ready.json').is_file()


def image_inventory(images_dir):
    d = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXT:
            continue
        name = asset_name(path)
        if name in d:
            raise RuntimeError('Image filename collision: ' + name)
        d[name] = str(path.absolute())
    return d


def target_path(out, a, b, alpha):
    return out / 'targets' / (a + '+' + b) / ('alpha_' + alpha_slug(alpha))


def balanced_pairs(names, quota, seed, split):
    """Randomized round-robin: unique edges, near-equal asset degree, O(pairs).

    Unlike rejection of a whole random perfect matching, this construction
    remains efficient at 50,000 training pairs and cannot repeat an edge.
    """
    names = sorted(names)
    n = len(names)
    if quota < 0 or quota > n * (n - 1) // 2:
        raise ValueError('%s: %d unique pairs impossible for %d images' % (split, quota, n))
    if quota == 0:
        return []
    rng = random.Random(stable_seed('round-robin-v4', seed, split))
    rng.shuffle(names)
    slots = list(names)
    if n % 2:
        slots.append(None)
    width = len(slots)
    pairs = []
    for _ in range(width - 1):
        for j in range(width // 2):
            a, b = slots[j], slots[width - j - 1]
            if a is None or b is None:
                continue
            pairs.append(tuple(sorted((a, b))))
            if len(pairs) == quota:
                return pairs
        slots = [slots[0], slots[-1]] + slots[1:-1]
    raise RuntimeError('Internal error: insufficient round-robin pairs')


def assign_pair_alpha_sets(num_pairs, k, seed, split, endpoint_fraction, alpha_min, alpha_max):
    """Deterministic per-pair Latin-hypercube samples of alpha.

    Every pair has k physically supervised targets. Exactly 2% of the
    resulting physical targets are endpoint aliases, equally alpha=0 and
    alpha=1. Endpoint targets replace (never add to) one interior alpha
    in DIFFERENT pairs. Alpha values are sorted descending (A -> B).
    """
    if not 0.0 <= endpoint_fraction < 1.0:
        raise ValueError('endpoint fraction must be in [0,1)')
    count_endpoint = int(round(num_pairs * k * endpoint_fraction / 2.0)) * 2
    if count_endpoint > num_pairs:
        raise ValueError('Too many endpoint targets for one-per-pair policy')
    rng = random.Random(stable_seed('interp3d-v4-lhs-k4', seed, split))
    schedules = []
    for i in range(num_pairs):
        alphas = [round(alpha_min + (alpha_max-alpha_min) * (j+rng.random()) / float(k), 6)
                  for j in range(k)]
        schedules.append(sorted(alphas, reverse=True))
    chosen = list(range(num_pairs))
    rng.shuffle(chosen)
    half = count_endpoint // 2
    for idx in chosen[:half]:
        schedules[idx][0] = 1.0
    for idx in chosen[half:count_endpoint]:
        schedules[idx][-1] = 0.0
    flat = [a for alphas in schedules for a in alphas]
    assert len(flat) == num_pairs*k
    assert flat.count(0.0) == flat.count(1.0) == half
    assert len({tuple(x) for x in schedules}) == num_pairs or num_pairs <= 1
    assert all(len(set(xs)) == k and xs == sorted(xs, reverse=True) for xs in schedules)
    return schedules


def split_pair_counts(args):
    quotas = {'train': args.train_pairs, 'val': args.val_pairs, 'test': args.test_pairs}
    if any(x < 0 for x in quotas.values()):
        raise ValueError('Pair quotas must be nonnegative')
    return quotas


def command_plan(args):
    v3 = Path(args.v3_root).resolve()
    images = Path(args.images_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / 'plan.json'
    quotas = split_pair_counts(args)
    n_pairs = sum(quotas.values())
    n_samples = n_pairs * ALPHAS_PER_PAIR
    if n_samples < 1:
        raise ValueError('At least one pair required')
    if n_samples > MAX_DATASET_SAMPLES:
        raise ValueError('Stored targets exceed cap: %d > %d' % (n_samples, MAX_DATASET_SAMPLES))
    n_virtual_train = 2 * quotas['train'] * ALPHAS_PER_PAIR
    if n_virtual_train > MAX_VIRTUAL_TRAIN_SAMPLES:
        raise ValueError('Source-swapped training cap exceeded: %d > %d' %
                         (n_virtual_train, MAX_VIRTUAL_TRAIN_SAMPLES))
    if not 0.0 < args.alpha_min < args.alpha_max < 1.0:
        raise ValueError('Require 0 < alpha_min < alpha_max < 1')
    if not 0.0 <= args.endpoint_fraction < 1.0:
        raise ValueError('Require 0 <= endpoint_fraction < 1')
    expected = {'format': 'morphflow_interp3d_v4_four_alpha_per_pair', 'version': 4,
                'pair_quotas': quotas, 'seed': args.seed, 'k': ALPHAS_PER_PAIR,
                'endpoint_fraction': args.endpoint_fraction,
                'alpha_min': args.alpha_min, 'alpha_max': args.alpha_max,
                'v3_root': str(v3), 'images_root': str(images)}
    if plan_path.exists():
        old = read_json(plan_path)
        if any(old.get(key) != value for key, value in expected.items()):
            raise RuntimeError('Incompatible plan in OUT_ROOT: use a new OUT_NAME')
        print('Reusing plan:', plan_path, flush=True)
        return
    inventory = image_inventory(images)
    split_assets_v3 = read_json(v3 / 'split_assets.json')
    split_images = {}
    seen = set()
    for split in SPLITS:
        available = sorted(set(split_assets_v3[split]).intersection(inventory))
        missing = set(split_assets_v3[split]).difference(inventory)
        if missing:
            print(
                'WARNING: split %s: %d images missing, excluded: %s' %
                (split, len(missing), sorted(missing)[:10]),
                flush=True
            )
        if seen.intersection(available):
            raise RuntimeError('Asset leakage between splits')
        seen.update(available)
        split_images[split] = available
        print('SPLIT %s: images=%d' % (split, len(available)), flush=True)
    if len(inventory) != len(seen):
        print('WARNING: %d images not in v3 split assignment; not used' %
              (len(inventory) - len(seen)), flush=True)
    pairs = []
    active = {split: set() for split in SPLITS}
    counts = {}
    for split in SPLITS:
        selected = balanced_pairs(split_images[split], quotas[split], args.seed, split)
        alpha_sets = assign_pair_alpha_sets(len(selected), ALPHAS_PER_PAIR, args.seed,
                                           split, args.endpoint_fraction,
                                           args.alpha_min, args.alpha_max)
        flat = [v for values in alpha_sets for v in values]
        counts[split] = {'interior': sum(0 < a < 1 for a in flat),
                         'alpha_0': flat.count(0.0), 'alpha_1': flat.count(1.0)}
        for (a, b), alphas in zip(selected, alpha_sets):
            active[split].update((a,b))
            pairs.append({'pair_id': len(pairs), 'split': split,
                          'src_1': a, 'src_2': b,
                          'seed': stable_seed(args.seed, 'pair', a, b),
                          'alphas': alphas})
    assets = [{'name': name, 'split': split, 'image': inventory[name]}
              for split in SPLITS for name in sorted(active[split])]
    plan = dict(expected)
    plan.update({'teacher': 'Interp3D',
                 'alpha_definition': 'alpha=1 -> src_1; alpha=0 -> src_2',
                 'sampling': '4 randomized stratified alpha positions per pair, with 2% endpoints',
                 'symmetry': 'same canonical target; source swap with complementary alpha',
                 'endpoint_target_policy': 'relative symlink to cached source latent',
                 'num_samples_per_pair': ALPHAS_PER_PAIR,
                 'max_dataset_samples': MAX_DATASET_SAMPLES,
                 'train_virtual_after_swap': n_virtual_train,
                 'num_pairs': len(pairs), 'num_targets': n_samples,
                 'target_counts': counts, 'pairs': pairs, 'assets': assets,
                 'split_assets': {split: sorted(active[split]) for split in SPLITS}})
    save_json(plan_path, plan)
    print('PLAN:', plan_path, 'pairs:', len(pairs), 'physical targets:', n_samples,
          'assets:', len(assets), 'virtual train:', n_virtual_train, flush=True)
    for split in SPLITS:
        print('%s: pairs=%d targets=%d alpha0=%d alpha1=%d interior=%d assets=%d' %
              (split, quotas[split], quotas[split]*ALPHAS_PER_PAIR,
               counts[split]['alpha_0'], counts[split]['alpha_1'],
               counts[split]['interior'], len(active[split])), flush=True)
    print('Example alpha sequences:', [p['alphas'] for p in pairs[:5]], flush=True)


class Reporter(object):
    """Atomic, low-IO worker heartbeat and progress, resilient to job restarts."""
    def __init__(self, out, phase, shard, total, targets_per_pair=1):
        self.out = Path(out)
        self.path = self.out/'.progress'/('%s_gpu_%d.json' % (phase, shard))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.start = time.time()
        self.targets_per_pair = targets_per_pair
        self.state = {'run_id': os.environ.get('MFV4_RUN_ID', 'manual'),
                      'job_id': os.environ.get('SLURM_JOB_ID', 'manual'),
                      'phase': phase, 'gpu_shard': shard, 'total': total,
                      'completed': 0, 'generated': 0, 'reused': 0,
                      'target_total': total*targets_per_pair,
                      'target_completed': 0, 'target_generated': 0,
                      'target_reused': 0,
                      'interior': 0, 'endpoint': 0,
                      'per_split': {s: 0 for s in SPLITS},
                      'per_split_targets': {s: 0 for s in SPLITS},
                      'status': 'running', 'current_item': '', 'last_error': '',
                      'compute_seconds': 0.0}
        self.write()
        self.thread = threading.Thread(target=self.heartbeat)
        self.thread.daemon = True
        self.thread.start()

    def write(self):
        with self.lock:
            payload = dict(self.state)
            payload['per_split'] = dict(self.state['per_split'])
            payload['per_split_targets'] = dict(self.state['per_split_targets'])
            payload['heartbeat_utc_epoch'] = time.time()
            payload['elapsed_seconds'] = payload['heartbeat_utc_epoch'] - self.start
        save_json(self.path, payload)

    def heartbeat(self):
        while not self.stop_event.wait(15):
            try:
                self.write()
            except Exception as exc:
                print('WARNING: progress heartbeat failed:', exc, flush=True)

    def current(self, name):
        with self.lock:
            self.state['current_item'] = str(name)

    def complete(self, split, result, kind, duration=0.0, targets_generated=0):
        with self.lock:
            self.state['completed'] += 1
            self.state['per_split'][split] += 1
            self.state['target_completed'] += self.targets_per_pair
            self.state['per_split_targets'][split] += self.targets_per_pair
            self.state['target_generated'] += targets_generated
            self.state['target_reused'] += self.targets_per_pair - targets_generated
            self.state['compute_seconds'] += float(duration)
            self.state['generated' if result == 'generated' else 'reused'] += 1
            if kind in ('endpoint', 'interior'):
                self.state[kind] += 1
        count = self.state['completed']
        if count == 1 or count % 25 == 0 or count == self.state['total']:
            self.write()
            print('[progress] %s gpu=%d %d/%d generated=%d reused=%d current=%s' %
                  (self.state['phase'], self.state['gpu_shard'], count,
                   self.state['total'], self.state['generated'],
                   self.state['reused'], self.state['current_item']), flush=True)

    def finish(self):
        with self.lock:
            self.state['status'] = 'complete'
            self.state['current_item'] = ''
        self.write()

    def fail(self, exc):
        with self.lock:
            self.state['status'] = 'failed'
            self.state['last_error'] = str(exc)
        self.write()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.write()


def torch_load(path):
    import torch
    try:
        return torch.load(str(path), map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location='cpu')


def save_tensor(path, tensor):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    torch.save(tensor, str(tmp))
    os.replace(str(tmp), str(path))


def occupancy_from_ss(pipeline, ss):
    import torch
    with torch.no_grad():
        output = pipeline.models['sparse_structure_decoder'](ss)
    if isinstance(output, (tuple, list)):
        output = output[0]
    output = (output > 0).detach().cpu().contiguous()
    if output.ndim == 5 and output.shape[:2] == (1, 1):
        return output[0, 0].contiguous()
    if output.ndim == 5 and output.shape[0] == 1:
        return output[0].contiguous()
    return output


def save_latents(pipeline, out_dir, ss, feats, coords, info):
    # Atomic file writes; ready.json committed last for reliable resume.
    import torch
    ss = ss.detach().cpu()
    feats = feats.detach().cpu()
    coords = coords.detach().cpu().to(torch.int32)
    save_tensor(out_dir / 'ss_latent.pt', ss)
    save_tensor(out_dir / 'slat_feats.pt', feats)
    save_tensor(out_dir / 'slat_coords.pt', coords)
    save_tensor(out_dir / 'structured_latent.pt', {'feats': feats, 'coords': coords})
    occupancy = occupancy_from_ss(pipeline, ss.to(pipeline.device))
    save_tensor(out_dir / 'occupancy.pt', occupancy)
    save_json(out_dir / 'ready.json', info)


def prepare_pipeline(model_id):
    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly ONE visible GPU per worker')
    print('GPU:', torch.cuda.get_device_name(0), 'Torch:', torch.__version__, flush=True)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_id)
    pipeline.cuda()
    return pipeline


def visible_image_path(path):
    src = os.environ.get('MORPHFLOW_INPUTS_HOST_PREFIX', '').rstrip('/')
    dst = os.environ.get('MORPHFLOW_INPUTS_CONTAINER_PREFIX', '').rstrip('/')
    if src and dst and (path == src or path.startswith(src + '/')):
        return dst + path[len(src):]
    return path


def command_assets(args):
    import torch
    from PIL import Image
    out = Path(args.out).resolve()
    plan = read_json(out / 'plan.json')
    assets = [a for a in plan['assets']
              if stable_shard(a['name'], args.num_shards) == args.shard_index]
    global ACTIVE_REPORTER
    ACTIVE_REPORTER = Reporter(out, 'assets', args.shard_index, len(assets))
    print('ASSET SHARD %d/%d: %d assets' %
          (args.shard_index, args.num_shards, len(assets)), flush=True)
    if not assets:
        ACTIVE_REPORTER.finish()
        return
    pipeline = prepare_pipeline(args.model_id)
    for i, a in enumerate(assets):
        name = a['name']
        ACTIVE_REPORTER.current(name)
        save_dir = out / 'assets' / name
        if check_ready(save_dir):
            ACTIVE_REPORTER.complete(a['split'], 'reused', 'asset')
            continue
        item_start = time.time()
        with Image.open(visible_image_path(a['image'])) as im:
            img = pipeline.preprocess_image(im.copy())
        cond = pipeline.get_cond([img])
        seed = stable_seed(plan['seed'], 'endpoint', name)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            coords, ss = pipeline.sample_sparse_structure(cond, None, 1, {})
            slat = pipeline.sample_slat(cond, coords, {})
            save_latents(pipeline, save_dir, ss, slat.feats, slat.coords,
                         {'kind': 'asset', 'name': name, 'split': a['split'],
                          'teacher': 'Interp3D', 'seed': seed})
        ACTIVE_REPORTER.complete(a['split'], 'generated', 'asset', time.time()-item_start)
        del slat, ss, coords, cond
        torch.cuda.empty_cache()
    ACTIVE_REPORTER.finish()


def reset_interp3d_attention_state(pipeline):
    # Same necessary reset used in MorphFlow's tested three-way benchmark.
    for name in ('sparse_structure_flow_model', 'slat_flow_model'):
        model = pipeline.models.get(name)
        if model is None:
            continue
        for attr, value in (('mode', 'normal'), ('alpha', 0.5),
                            ('step_id0', 0), ('step_id1', 0), ('step_id2', 0)):
            if hasattr(model, attr):
                setattr(model, attr, value)
        for cache_name in ('kv_dict_img0', 'kv_dict_img1'):
            cache = getattr(model, cache_name, None)
            if cache is not None and hasattr(cache, 'clear'):
                cache.clear()
        for block in getattr(model, 'blocks', []):
            attn = getattr(block, 'self_attn', None)
            if attn is not None:
                if hasattr(attn, 'attn_phase'):
                    attn.attn_phase = 'normal'
                if hasattr(attn, 'alpha'):
                    attn.alpha = 0.5
                if hasattr(attn, 'step_id'):
                    attn.step_id = 0


def command_pairs(args):
    import torch
    from PIL import Image
    from trellis.modules import sparse as sp
    import trellis.pipelines.trellis_image_to_3d_morphing as interp_module
    out = Path(args.out).resolve()
    plan = read_json(out / 'plan.json')
    image_by_name = {a['name']: a['image'] for a in plan['assets']}
    pairs = [p for p in plan['pairs']
             if p['pair_id'] % args.num_shards == args.shard_index]
    global ACTIVE_REPORTER
    ACTIVE_REPORTER = Reporter(out, 'pairs', args.shard_index, len(pairs), plan['k'])
    print('PAIR SHARD %d/%d: %d pairs, %d alphas each' %
          (args.shard_index, args.num_shards, len(pairs), plan['k']), flush=True)
    if not pairs:
        ACTIVE_REPORTER.finish()
        return
    pipeline = prepare_pipeline(args.model_id)

    # Interp3D defaults to a Beta(6,6) alpha list; the v4 teacher must use
    # each pair's deterministic stratified schedule, INCLUDING the same alpha
    # positions at both SS and SLat generation (native Interp3D trajectory).
    current_schedule = {'alphas': None, 'called': False}
    def pair_specific_alpha_list(coords0, coords1, steps=5):
        a = current_schedule['alphas']
        if a is None or len(a) + 2 != int(steps):
            raise RuntimeError('Interp3D invoked with an unexpected alpha schedule size')
        current_schedule['called'] = True
        # Source image one corresponds to internal t=0 / MorphFlow alpha=1.
        if any(x <= 0.0 or x >= 1.0 for x in a) or a != sorted(a, reverse=True):
            raise RuntimeError('Internal alpha sequence must be strictly interior and descending')
        return torch.tensor([0.0] + [1.0-x for x in a] + [1.0],
                            dtype=torch.float32, device=coords0.device)
    interp_module.adaptive_alpha_list_by_chamfer = pair_specific_alpha_list

    @lru_cache(maxsize=32)
    def cached_asset(name):
        root = out / 'assets' / name
        if not check_ready(root):
            raise RuntimeError('Source asset not ready: ' + name)
        return (torch_load(root / 'ss_latent.pt'),
                torch_load(root / 'slat_feats.pt'),
                torch_load(root / 'slat_coords.pt'))

    original_generate = pipeline.generate_3d_single
    original_ss_sampler = pipeline.sparse_structure_sampler.sample
    original_slat_morphing = pipeline.sample_slat_morphing
    capture = {'ss': [], 'slat': []}
    current = {'src_1': None, 'src_2': None}

    def patched_generate(*pargs, **kwargs):
        alpha = float(pargs[0])
        if abs(alpha) < 1.e-7 or abs(alpha - 1.0) < 1.e-7:
            name = current['src_1'] if alpha < 0.5 else current['src_2']
            ss_cpu, feats_cpu, coords_cpu = cached_asset(name)
            ss = ss_cpu.to(pipeline.device)
            feats = feats_cpu.to(pipeline.device)
            coords = coords_cpu.to(pipeline.device)
            slat = sp.SparseTensor(feats=feats, coords=coords)
            return {}, slat, coords, ss
        return original_generate(*pargs, **kwargs)

    def capture_ss_sampler(*pargs, **kwargs):
        samples = original_ss_sampler(*pargs, **kwargs)
        if isinstance(samples, list):
            capture['ss'] = [s.samples.detach().cpu() for s in samples]
        return samples

    def capture_slat_morphing(*pargs, **kwargs):
        slats = original_slat_morphing(*pargs, **kwargs)
        capture['slat'] = [(s.feats.detach().cpu(), s.coords.detach().cpu())
                           for s in slats]
        return slats

    pipeline.generate_3d_single = patched_generate
    pipeline.sparse_structure_sampler.sample = capture_ss_sampler
    pipeline.sample_slat_morphing = capture_slat_morphing

    for idx, pair in enumerate(pairs):
        a, b = pair['src_1'], pair['src_2']
        alphas = pair['alphas']  # MorphFlow order: descending from A to B
        interiors = [alpha for alpha in alphas if 0.0 < alpha < 1.0]
        ACTIVE_REPORTER.current('pair=%d %s+%s alphas=%s' %
                                (pair['pair_id'], a, b, ','.join(map(str, alphas))))
        item_start = time.time()
        # Endpoint links are exact source-latent targets, not interpolations.
        for alpha in alphas:
            if alpha in (0.0, 1.0):
                ensure_endpoint_alias(out, a, b, alpha)
        target_dirs = [target_path(out, a, b, alpha) for alpha in alphas]
        ready_before = {alpha: check_ready(target_path(out, a, b, alpha))
                        for alpha in alphas}
        if all(ready_before.values()):
            ACTIVE_REPORTER.complete(pair['split'], 'reused', 'interior',
                                     targets_generated=0)
            continue
        current_schedule['alphas'] = interiors
        current_schedule['called'] = False
        current['src_1'], current['src_2'] = a, b
        capture['ss'], capture['slat'] = [], []
        reset_interp3d_attention_state(pipeline)
        with Image.open(visible_image_path(image_by_name[a])) as im:
            image1 = im.copy()
        with Image.open(visible_image_path(image_by_name[b])) as im:
            image2 = im.copy()
        torch.manual_seed(pair['seed'])
        torch.cuda.manual_seed_all(pair['seed'])
        with torch.no_grad():
            outputs, coords = pipeline.run_interpolation_morphing(
                image1, image2, seed=pair['seed'],
                num_search_iterations=len(interiors) + 2, formats=[])
            if not current_schedule['called']:
                raise RuntimeError('Interp3D did not use requested alpha sequence')
            if len(outputs) != len(interiors) + 2:
                raise RuntimeError('Interp3D output count mismatch')
            if len(capture['ss']) != len(interiors) or len(capture['slat']) != len(interiors):
                raise RuntimeError('Interp3D capture mismatch: SS=%d SLat=%d expected=%d' %
                                   (len(capture['ss']), len(capture['slat']), len(interiors)))
            generated_targets = 0
            for j, alpha in enumerate(interiors):
                target_dir = target_path(out, a, b, alpha)
                if check_ready(target_dir):
                    continue
                ss = capture['ss'][j]
                feats, coords_j = capture['slat'][j]
                save_latents(pipeline, target_dir, ss, feats, coords_j,
                             {'kind': 'target', 'teacher': 'Interp3D',
                              'src_1': a, 'src_2': b, 'split': pair['split'],
                              'alpha': alpha, 'interp3d_t': 1-alpha,
                              'pair_seed': pair['seed'], 'sequence_index': alphas.index(alpha),
                              'sequence_alphas': alphas})
                generated_targets += 1
        ACTIVE_REPORTER.complete(pair['split'], 'generated', 'interior',
                                 time.time()-item_start,
                                 targets_generated=generated_targets)
        del outputs, coords, image1, image2
        capture['ss'], capture['slat'] = [], []
        if idx % 20 == 0:
            torch.cuda.empty_cache()
    ACTIVE_REPORTER.finish()


def ensure_endpoint_alias(out, a, b, alpha):
    """Physical target directory alias for alpha=0/1, with zero tensor duplication.

    Both forward and reverse orientations use these same canonical target bytes.
    Relative symlinks remain valid when moving the dataset directory as a unit.
    """
    if alpha not in (0.0, 1.0):
        raise ValueError('Only 0/1 are endpoint aliases')
    name = a if alpha == 1.0 else b
    source = out / 'assets' / name
    if not check_ready(source):
        raise RuntimeError('Endpoint source not complete: ' + str(source))
    dest = target_path(out, a, b, alpha)
    dest.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(str(source), str(dest.parent))
    if dest.is_symlink():
        if os.readlink(str(dest)) != relative:
            raise RuntimeError('Endpoint alias points at wrong source: ' + str(dest))
    elif os.path.lexists(str(dest)):
        raise RuntimeError('Cannot safely replace existing endpoint target path: ' + str(dest))
    else:
        os.symlink(relative, str(dest))
    if not check_ready(dest):
        raise RuntimeError('Created incomplete endpoint alias: ' + str(dest))
    return dest


def make_metadata(pair, alpha, k):
    a, b = pair['src_1'], pair['src_2']
    target_rel = 'targets/%s+%s/alpha_%s' % (a, b, alpha_slug(alpha))
    kind = 'endpoint' if alpha in (0.0, 1.0) else 'intermediate'
    position = (pair['alphas'].index(alpha) if 0.0 < alpha < 1.0 else -1)
    ans = {'split': pair['split'], 'src_1': a, 'src_2': b,
           'target': '%s+%s/alpha_%s' % (a, b, alpha_slug(alpha)),
           'alpha': alpha, 'alpha_requested': alpha,
           'alpha_definition': 'alpha is fraction of src_1; alpha*src_1+(1-alpha)*src_2',
           'alpha_sequence_index': position,
           'sequence_intermediates': len([a for a in pair['alphas'] if 0 < a < 1]),
           'sequence_alphas': list(pair['alphas']),
           'src1_dir': 'assets/'+a, 'src2_dir': 'assets/'+b,
           'target_dir': target_rel,
           'teacher_model': 'Interp3D', 'interp3d_t': float(1-alpha),
           'target_kind': kind,
           'symmetry_policy': 'reuse_same_target_after_source_swap'}
    for key, name in (('src1', a), ('src2', b), ('target', None)):
        prefix = ('assets/' + name) if name else target_rel
        ans[key+'_slat_feats'] = prefix+'/slat_feats.pt'
        ans[key+'_slat_coords'] = prefix+'/slat_coords.pt'
        ans[key+'_ss_latent'] = prefix+'/ss_latent.pt'
        ans[key+'_structured_latent'] = prefix+'/structured_latent.pt'
        ans[key+'_occupancy'] = prefix+'/occupancy.pt'
    return ans


def command_finalize(args):
    out = Path(args.out).resolve()
    plan = read_json(out / 'plan.json')
    entries = []
    counts = {split: 0 for split in SPLITS}
    incomplete = []
    for pair in plan['pairs']:
        a, b = pair['src_1'], pair['src_2']
        source_ready = check_ready(out/'assets'/a) and check_ready(out/'assets'/b)
        if source_ready:
            for alpha in pair['alphas']:
                if alpha in (0.0, 1.0):
                    ensure_endpoint_alias(out, a, b, alpha)
        for alpha in pair['alphas']:
            if source_ready and check_ready(target_path(out, a, b, alpha)):
                entries.append(make_metadata(pair, alpha, plan['k']))
                counts[pair['split']] += 1
            else:
                incomplete.append({'src_1': a, 'src_2': b, 'alpha': alpha,
                                   'split': pair['split']})
    entries.sort(key=lambda x: (x['split'], x['src_1'], x['src_2'], x['alpha']))
    save_json(out/'metadata.json', entries)
    for split in SPLITS:
        save_json(out/('metadata_'+split+'.json'),
                  [e for e in entries if e['split'] == split])
    save_json(out/'split_assets.json', plan['split_assets'])
    save_json(out/'splits.json', {'assets': plan['split_assets'],
                                  'counts': {s: len(plan['split_assets'][s]) for s in SPLITS},
                                  'metadata_counts': counts,
                                  'missing_planned_targets': len(incomplete),
                                  'config': {'teacher': 'Interp3D',
                                             'alpha_samples_per_pair': plan['k'],
                                             'endpoint_fraction': plan['endpoint_fraction'],
                                             'symmetry': 'source swap with alpha complement',
                                             'num_pairs': plan['num_pairs'],
                                             'train_virtual_after_swap': plan['train_virtual_after_swap']}})
    save_json(out/'missing_targets.json', incomplete)
    print('FINAL: train=%d val=%d test=%d total=%d/%d missing=%d' %
          (counts['train'], counts['val'], counts['test'], len(entries),
           plan['num_targets'], len(incomplete)), flush=True)
    if incomplete:
        raise RuntimeError('Incomplete dataset: rerun Slurm job to resume')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='mode')
    p = sub.add_parser('plan')
    p.add_argument('--v3-root', required=True)
    p.add_argument('--images-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--train-pairs', type=int, default=12500)
    p.add_argument('--val-pairs', type=int, default=500)
    p.add_argument('--test-pairs', type=int, default=500)
    p.add_argument('--endpoint-fraction', type=float, default=0.02)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--alpha-min', type=float, default=0.001)
    p.add_argument('--alpha-max', type=float, default=0.999)
    for name in ('assets', 'pairs'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True)
        p.add_argument('--num-shards', type=int, default=8)
        p.add_argument('--shard-index', type=int, required=True)
        p.add_argument('--model-id', default='microsoft/TRELLIS-image-large')
    p = sub.add_parser('finalize')
    p.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.mode == 'plan':
        command_plan(args)
    else:
        try:
            if args.mode == 'assets':
                command_assets(args)
            elif args.mode == 'pairs':
                command_pairs(args)
            elif args.mode == 'finalize':
                command_finalize(args)
            else:
                ap.error('Choose plan/assets/pairs/finalize')
        except BaseException as exc:
            if ACTIVE_REPORTER is not None:
                ACTIVE_REPORTER.fail(exc)
            raise
        finally:
            if ACTIVE_REPORTER is not None:
                ACTIVE_REPORTER.close()


if __name__ == '__main__':
    main()
