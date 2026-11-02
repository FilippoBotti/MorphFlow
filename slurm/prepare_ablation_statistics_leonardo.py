#!/usr/bin/env python3
"""Freeze a four-model, full-test paired evaluation before observing new scores."""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shlex
import shutil
import subprocess

BASE = Path('/leonardo_work/IscrC_MORPHFL/mbarezzi')
PROJECT = BASE / 'src/MorphFlow'
OLD = BASE / 'outputs/threeway_benchmark/ablation_suite_20261026'
PILOT = BASE / 'outputs/threeway_benchmark/baseline_ss10_slat34_n50_seed42_cfg11_cfg515_20261023'
FULL = BASE / 'outputs/threeway_benchmark/full_ss17_slat12_n50_seed42_ss3_4_4.5_5_slat1.5_20261026'
REPAIRS = BASE / 'outputs/threeway_benchmark/interp3ddata_ss17_slat12_cfg9_seed42_7states_20261021/runtime'
VARIANTS = ('baseline', 'gate_only', 'prior_only', 'full')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def build_pairs(template, entries):
    unique = {}
    for index, entry in enumerate(entries):
        assert entry.get('split') == 'test'
        unique.setdefault((entry['src_1'], entry['src_2']), index)
    old = {(p['src1'], p['src2']) for p in template['pairs']}
    assert len(old) == len(template['pairs']) == 50 and old <= unique.keys()
    remaining = [pair for pair in unique if pair not in old]
    random.Random(42).shuffle(remaining)
    # This selection uses source identities only, never model outputs.
    used = {source for pair in old for source in pair}
    confirmation = set()
    for a, b in remaining:
        if a not in used and b not in used:
            confirmation.add((a, b))
            used.update((a, b))
    helper = load_module('threeway_layout_stats', PROJECT / 'slurm/threeway_ateneo.py')
    pairs = [dict(p) for p in template['pairs']]
    for pair_id, (a, b) in enumerate(remaining, start=50):
        def slug(text):
            import re
            return re.sub(r'[^a-zA-Z0-9._-]+', '_', text).strip('_')[:36]
        pairs.append(dict(pair_id=pair_id, pair_dir='pair_{:04d}_{}_to_{}'.format(pair_id, slug(a), slug(b)),
                          src1=a, src2=b, metadata_index=unique[a, b], pair_seed=42 + pair_id * 100000,
                          src1_image=str(Path(template['source_images_root']) / (a + '.png')),
                          src2_image=str(Path(template['source_images_root']) / (b + '.png'))))
    parent = {}
    def find(a):
        parent.setdefault(a, a)
        if parent[a] != a:
            parent[a] = find(parent[a])
        return parent[a]
    for a, b in unique:
        parent[find(a)] = find(b)
    component_names = {a: i for i, a in enumerate(sorted({find(a) for a, b in unique}))}
    for pair in pairs:
        identity = pair['src1'], pair['src2']
        pair['pilot'] = identity in old
        pair['confirmation'] = identity in confirmation
        pair['source_component'] = component_names[find(pair['src1'])]
        assert entries[pair['metadata_index']]['src_1'] == pair['src1']
    # Process the fresh object-disjoint cohort first, without changing pair IDs/seeds.
    pairs.sort(key=lambda p: (not p['confirmation'], p['pair_id']))
    sources = [s for p in pairs if p['confirmation'] for s in (p['src1'], p['src2'])]
    assert len(sources) == len(set(sources)) and not (set(sources) & {s for p in old for s in p})
    return pairs


def prepare(args):
    root = Path(args.run_root).resolve()
    assert not root.exists(), 'Use a new run root; existing frozen evaluations are never replaced.'
    reader = load_module('ablation_metadata_stats', PROJECT / 'slurm/prepare_ablation_suite_leonardo.py')
    layout = load_module('ablation_layout_stats', PROJECT / 'slurm/threeway_ateneo.py')
    sources = {
        'baseline': {'ss': OLD/'checkpoint_bank/ss_base/checkpoint.pt', 'slat': OLD/'checkpoint_bank/slat_base/checkpoint.pt'},
        'gate_only': {'ss': OLD/'checkpoint_bank/ss_gate/checkpoint.pt', 'slat': OLD/'checkpoint_bank/slat_gate/checkpoint.pt'},
        'prior_only': {'ss': OLD/'checkpoint_bank/ss_prior/checkpoint.pt', 'slat': OLD/'checkpoint_bank/slat_prior/checkpoint.pt'},
        'full': {flow: FULL/('checkpoints/'+flow+'/checkpoint.pt') for flow in ('ss','slat')},
    }
    metadata = {name: {flow: reader.metadata(path) for flow, path in rows.items()} for name, rows in sources.items()}
    candidates = [BASE/'checkpoints/ateneo_v4_ss17/ss_best_e17.pt',
                  BASE/'checkpoints/v4_interp3d_ss_dual_tgate_p25g4_lr1e-4_bs_1_leonardo/morphflow_best.pt',
                  BASE/'checkpoints/v4_interp3d_slat_dual_tgate_p25g4_lr5e-5_bs1_leonardo/morphflow_best.pt']
    available = [reader.metadata(path) for path in candidates]
    for flow in ('ss','slat'):
        best = min((m for m in available if m['flow_target'] == flow), key=lambda m:m['val_loss'])
        selected = metadata['full'][flow]
        assert (best['epoch'],best['step'],best['val_loss']) == (selected['epoch'],selected['step'],selected['val_loss'])
    for name, flows in metadata.items():
        for flow, m in flows.items():
            assert m['flow_target'] == flow
            assert m['expectations'] == dict(separate_cond_gate='token' if name in ('gate_only','full') else 'alpha',
                                               augment_source_swap=int(name=='full'),
                                               trellis_prior_weight=0.1 if name in ('prior_only','full') else 0.)
    template = json.loads((PILOT/'shared/plan.json').read_text())
    entries = json.loads(Path(template['metadata']).read_text())
    pairs = build_pairs(template, entries)
    assert len(pairs) == 500
    data_root = Path(template['data_root'])
    assets = {s for p in pairs for s in (p['src1'],p['src2'])}
    missing = [str(data_root/'assets'/s/f) for s in sorted(assets)
               for f in ('ss_latent.pt','slat_feats.pt','slat_coords.pt') if not (data_root/'assets'/s/f).is_file()]
    assert not missing, 'Missing source inputs: '+str(missing[:20])
    root.mkdir(parents=True)
    write(root/'checkpoint_metadata.json', metadata)
    write(root/'full_checkpoint_selection.json', dict(rule='Minimum saved validation loss per flow, fixed before new test scores.', candidates=available, selected=metadata['full']))
    input_file = root/'input_snapshot/metadata_test.json'
    input_file.parent.mkdir()
    shutil.copyfile(template['metadata'], input_file)
    plan = dict(template, pairs=pairs, metadata=str(input_file), num_pairs=len(pairs), requested_num_pairs=len(pairs),
                available_test_pairs=len(pairs), metadata_index_basis='unfiltered')
    write(root/'plan.json', plan)
    components = collections.Counter(p['source_component'] for p in pairs)
    protocol = dict(
        variants=list(VARIANTS), cfg_ss=5., cfg_slat=1.5, seed=42, states=7, intermediate_states=5,
        alphas=plan['alphas'], alpha_schedule='Beta(5,5) deterministic interior quantiles',
        views=64, fid_views_per_state=1000, fid_aggregation='concatenated',
        primary_metrics=['FID','PDV'], secondary_metrics=['PPL','EPRE'],
        bootstrap_samples=10000, permutation_samples=99999, statistics_seed=42028,
        primary_cohort='confirmation_object_disjoint', confirmation_count=sum(p['confirmation'] for p in pairs),
        confirmation_selection='Greedy source-disjoint selection in Random(42)-shuffled new pairs, excluding every source appearing in the original 50 pairs. Fixed from identities before outputs.',
        all_pairs_count=len(pairs), original_pilot_count=50, unique_assets=len(assets),
        source_components=len(components), source_component_sizes=sorted(components.values(),reverse=True),
        inference='Conditional on four fixed trained checkpoints; no claim of across-training-seed variability or matched training budgets.',
        tests='Six two-sided paired sign-flip contrasts for each primary metric; Holm adjustment over all 12 primary tests. Secondary tests form a separate exploratory family.',
        uncertainty='Paired bootstrap on the object-disjoint confirmation cohort. Descriptive full-test intervals resample entire connected source components, with equal pair weighting within each resample.',
        stopping='All 500 pairs per variant are scheduled. No early stopping or checkpoint selection based on new results. Final hypothesis tests only after complete evaluation.',
        retention='Keep per-pair metrics, per-view CSVs and 025.png for seven states; remove only newly generated work batches after verified durable commit.',
        references={'paired_permutation':'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html',
                    'holm':'https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html'})
    write(root/'statistical_protocol.json',protocol)
    source_manifest = {}
    for name, path in layout.source_files(PROJECT, {'TRELLIS':BASE/'src/TRELLIS','MorphAny3D':BASE/'src/MorphAny3D'}).items():
        dst=root/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
        source_manifest[name]=layout.digest(dst)
    for name in ('resume_threeway_mesh.py','mesh_channel_blocks.py','resume_threeway_render_cache.py'):
        dst=root/'runtime'/name;dst.parent.mkdir(exist_ok=True);shutil.copyfile(REPAIRS/name,dst)
    write(root/'runtime/render_repair.json',dict(old_root=str(root),new_root=str(root),
          frozen_evaluator=str(root/'source_snapshot/MorphFlow/benchmarks/eval_textured_benchmark.py')))
    pins={}
    for name, flows in sources.items():
        pins[name]={}
        cell=root/'variants'/name/'cfg_ss_5_slat_1.5'
        cell.mkdir(parents=True)
        (cell/'plan.json').symlink_to(root/'plan.json')
        for flow,path in flows.items():
            dst=root/'variants'/name/'checkpoints'/flow/'checkpoint.pt';dst.parent.mkdir(parents=True,exist_ok=True)
            immutable=path.resolve();dst.symlink_to(immutable);st=immutable.stat()
            pins[name][flow]=dict(path=str(dst),frozen_existing_source=str(immutable),bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    write(root/'run_manifest.json',dict(methods=['morphflow'],variants=list(VARIANTS),checkpoints=pins,
          metadata=metadata,metadata_sha256=layout.digest(input_file),sources=source_manifest,
          num_pairs=len(pairs),cfg_ss=5.,cfg_slat=1.5,batch_size=args.batch_size,planned_continuations=args.jobs))
    env=dict(RUN_ROOT=str(root),DATA_ROOT_HOST=str(data_root),SOURCE_IMAGES_ROOT_HOST=template['source_images_root'],
             ORIGINAL_ENV=str(BASE/'envs/morphflow'),METRIC_ENV=str(BASE/'envs/morphany3d'))
    (root/'launch.env').write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items()))
    launcher=root/'source_snapshot/MorphFlow/slurm/benchmark_ablation_statistics_leonardo.slurm'
    subprocess.run(['bash','-n',str(launcher)],check=True)
    submissions=[]
    if args.submit:
        previous=None
        for index in range(args.jobs):
            cmd=['sbatch','--parsable','--job-name=ablation_stats_'+('start' if index==0 else 'resume'+str(index))]
            if previous:cmd.append('--dependency=afterany:'+previous)
            previous=subprocess.check_output(cmd+[str(launcher),str(root)],text=True).strip().split(';')[0]
            submissions.append(dict(job_id=previous,continuation=index,run_root=str(root)))
            write(root/'submissions.json',submissions)
    print(json.dumps(dict(run_root=str(root),pairs=len(pairs),confirmation_pairs=protocol['confirmation_count'],
                         variants=list(VARIANTS),full_checkpoints=metadata['full'],submissions=submissions),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',required=True)
    p.add_argument('--batch-size',type=int,default=5)
    p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--submit',action='store_true')
    args=p.parse_args()
    if not 1<=args.batch_size<=5 or not 1<=args.jobs<=4:p.error('batch-size 1..5; jobs 1..4')
    prepare(args)
