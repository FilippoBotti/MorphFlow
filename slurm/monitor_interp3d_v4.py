#!/usr/bin/env python3
"""Live, read-only Interp3D v4 status viewer + Slurm state helper.

Python 3.6+ standard library; no Torch, no container required.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import time
import uuid


def read(path):
    try:
        with open(str(path)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with open(str(temp), 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(str(temp), str(path))


def set_phase(args):
    path = Path(args.out) / 'run_state.json'
    old = read(path) or {}
    if old.get('run_id') != args.run_id:
        old = {'started_at_epoch': time.time(), 'job_id': args.job_id,
               'run_id': args.run_id}
    old.update({'phase': args.phase, 'status': args.status,
                'updated_at_epoch': time.time(), 'job_id': args.job_id})
    if args.message:
        old['message'] = args.message
    atomic_json(path, old)


def hms(seconds):
    if seconds is None or seconds < 0:
        return '--'
    seconds = int(seconds)
    if seconds >= 86400:
        return '%dd %02dh' % (seconds // 86400, (seconds % 86400) // 3600)
    return '%02d:%02d:%02d' % (seconds // 3600, (seconds // 60) % 60, seconds % 60)


def status_data(out):
    root = Path(out)
    state = read(root / 'run_state.json') or {}
    plan = read(root / 'plan.json') or {}
    phase = state.get('phase', 'uninitialized')
    workers = []
    for phase_name in ('assets', 'pairs'):
        for i in range(8):
            st = read(root / '.progress' / ('%s_gpu_%d.json' % (phase_name, i)))
            if st and st.get('run_id') == state.get('run_id'):
                workers.append(st)
    relevant = [w for w in workers if w['phase'] == phase]
    completed = sum(w.get('completed', 0) for w in relevant)
    total = sum(w.get('total', 0) for w in relevant)
    generated = sum(w.get('generated', 0) for w in relevant)
    reused = sum(w.get('reused', 0) for w in relevant)
    target_completed = sum(w.get('target_completed', w.get('completed', 0)) for w in relevant)
    target_total = sum(w.get('target_total', w.get('total', 0)) for w in relevant)
    target_generated = sum(w.get('target_generated', w.get('generated', 0)) for w in relevant)
    target_reused = sum(w.get('target_reused', w.get('reused', 0)) for w in relevant)
    compute_seconds = sum(w.get('compute_seconds', 0) for w in relevant)
    # ETA based on newly generated items only: resuming skips many quickly.
    eta = None
    if generated and total >= completed:
        rate_seconds = compute_seconds / generated
        eta = (total - completed) * rate_seconds / max(1, len(relevant))
    by_split = {x: sum(w.get('per_split', {}).get(x, 0) for w in relevant)
                for x in ('train', 'val', 'test')}
    by_split_targets = {x: sum(w.get('per_split_targets', {}).get(x, 0) for w in relevant)
                        for x in ('train', 'val', 'test')}
    return {'state': state, 'plan': {'num_pairs': plan.get('num_pairs'),
                                   'num_targets': plan.get('num_targets'),
                                   'k': plan.get('k'),
                                   'pair_quotas': plan.get('pair_quotas'),
                                   'target_counts': plan.get('target_counts'),
                                   'num_assets': len(plan.get('assets', [])),
                                   'train_virtual_after_swap': plan.get('train_virtual_after_swap')},
            'phase': phase, 'workers': sorted(relevant, key=lambda x: x['gpu_shard']),
            'total': total, 'completed': completed, 'generated': generated,
            'reused': reused, 'by_split': by_split,
            'target_completed': target_completed, 'target_total': target_total,
            'target_generated': target_generated, 'target_reused': target_reused,
            'by_split_targets': by_split_targets,
            'eta_seconds': eta, 'timestamp': time.time()}


def show(data):
    s, p = data['state'], data['plan']
    print('\n' + '=' * 86)
    print('Interp3D v4 | job=%s | phase=%s | state=%s' %
          (s.get('job_id', '?'), data['phase'], s.get('status', '?')))
    if s.get('message'):
        print('Message:', s['message'])
    if p.get('num_pairs') is not None:
        print('PLAN: %d pairs x %s alphas = %s physical targets | assets=%d | virtual train=%s' %
              (p['num_pairs'], p.get('k'), p.get('num_targets'), p['num_assets'],
               p.get('train_virtual_after_swap')))
        print('Splits:', p['pair_quotas'], '| alpha types:', p['target_counts'])
    done, total = data['completed'], data['total']
    pct = 100.0 * done / total if total else 0.0
    print('PAIRS/ASSETS: %d/%d (%.2f%%), newly processed=%d, reused=%d, ETA~%s' %
          (done, total, pct, data['generated'], data['reused'], hms(data['eta_seconds'])))
    if data['phase'] == 'pairs':
        print('TARGETS: %d/%d (%.2f%%), new=%d, reused/endpoints=%d' %
              (data['target_completed'], data['target_total'],
               100.0*data['target_completed']/max(1, data['target_total']),
               data['target_generated'], data['target_reused']))
    quotas = p.get('pair_quotas') or {}
    print('Completed by split:', ', '.join('%s=%s/%s pairs (%s targets)' %
          (key, data['by_split'].get(key, 0), quotas.get(key, '?'),
           data['by_split_targets'].get(key, 0))
          for key in ('train', 'val', 'test')))
    print('-' * 86)
    print('GPU | done / total | new | reused | age(s) | elapsed | status   | current')
    print('-' * 86)
    for w in data['workers']:
        age = data['timestamp'] - w.get('heartbeat_utc_epoch', 0)
        status = str(w.get('status', '?'))
        if age > 120 and status == 'running':
            status = 'STALE'
        print('%3s | %5d / %-5d |%5d| %6d | %6.0f | %7s | %-8s | %.44s' %
              (w['gpu_shard'], w.get('completed', 0), w.get('total', 0),
               w.get('generated', 0), w.get('reused', 0), age,
               hms(w.get('elapsed_seconds')), status, w.get('current_item', '')))
        if w.get('last_error'):
            print('    ERROR:', w['last_error'][:180])
    if not data['workers']:
        print('Waiting for worker progress files (the job may be planning or preflighting).')
    print('=' * 86)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode')
    st = sub.add_parser('status')
    st.add_argument('--out', required=True)
    st.add_argument('--watch', type=float, default=0, metavar='SECONDS')
    st.add_argument('--json', action='store_true')
    ss = sub.add_parser('set-phase')
    ss.add_argument('--out', required=True)
    ss.add_argument('--job-id', required=True)
    ss.add_argument('--run-id', required=True)
    ss.add_argument('--phase', required=True)
    ss.add_argument('--status', default='running')
    ss.add_argument('--message', default='')
    args = p.parse_args()
    if args.mode == 'set-phase':
        set_phase(args)
    elif args.mode == 'status':
        while True:
            data = status_data(args.out)
            if args.json:
                print(json.dumps(data), flush=True)
            else:
                show(data)
            if args.watch <= 0:
                break
            time.sleep(max(1, args.watch))
    else:
        p.error('Select status or set-phase')


if __name__ == '__main__':
    main()
