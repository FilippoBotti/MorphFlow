#!/usr/bin/env python3
"""Freeze the requested component ablations and a 50-pair Interp3DData baseline."""
import argparse
import builtins
import collections
import csv
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import shlex
import shutil
import subprocess
import zipfile

BASE = Path('/leonardo_work/IscrC_MORPHFL/mbarezzi')
PROJECT = BASE/'src/MorphFlow'
SUITE = BASE/'outputs/threeway_benchmark/ablation_suite_20261026'
PREVIOUS = BASE/'outputs/threeway_benchmark/baseline_ss10_slat34_n50_seed42_cfg11_cfg515_20261023'
EXTERNAL = BASE/'outputs/threeway_benchmark/interp3ddata_ss17_slat12_cfg9_seed42_7states_20261021'
BANK = BASE/'checkpoints/ateneo_baselines'
SS = BANK/'ablations_ss'
PATHS = {
    'ss_base': SS/'v4_ss_ablation_baseline_steps250000_e10_s42_g2_bs1/checkpoints/morphflow_best.pt',
    'ss_gate': SS/'v4_ss_ablation_gate_only_steps250000_e10_s42_g2_bs1/checkpoints/morphflow_best.pt',
    'ss_prior': SS/'v4_ss_ablation_prior_only_steps62500_e10_s42_g8_bs1/checkpoints/morphflow_best.pt',
    'ss_swap': SS/'v4_ss_ablation_swap_only_steps125000_e10_s42_g8_bs1/checkpoints/morphflow_best.pt',
    'slat_base': BANK/'v4_slat_ablation_baseline_steps250000_e40_s42_g8_bs1/checkpoints/morphflow_best.pt',
    'slat_gate': BANK/'v4_slat_ablation_gate_only_steps250000_e20_s42_g2_bs2/checkpoints/morphflow_best.pt',
    'slat_prior': BANK/'v4_slat_ablation_prior_only_steps125000_e20_s42_g8_bs1/checkpoints/morphflow_best.pt',
}
JOBS = [
    ('gate_only_both_n50_seed42', 'gate_only_both', 'v4_test', 'ss_gate', 'slat_gate', [('1','1'),('5','1.5')]),
    ('prior_only_both_n50_seed42', 'prior_only_both', 'v4_test', 'ss_prior', 'slat_prior', [('1','1'),('5','1.5')]),
    ('swap_ss_only_n50_seed42', 'swap_ss_only', 'v4_test', 'ss_swap', 'slat_base', [('1','1'),('5','1.5')]),
    ('baseline_interp3ddata_n50_seed42_cfg515', 'baseline', 'interp3ddata', 'ss_base', 'slat_base', [('5','1.5')]),
]

class MetadataReader(pickle.Unpickler):
    """Read trusted training metadata without importing Torch or materializing tensors."""
    def persistent_load(self, pid):
        return None
    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        if module == 'argparse' and name == 'Namespace':
            return argparse.Namespace
        if module in ('builtins', '__builtin__'):
            return getattr(builtins, name)
        if module.startswith('torch'):
            return lambda *args, **kwargs: None
        raise ValueError((module, name))

def metadata(path):
    with zipfile.ZipFile(str(path)) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        with archive.open(name) as stream:
            checkpoint = MetadataReader(stream).load()
    args = checkpoint['args']
    if not isinstance(args, dict):
        args = vars(args)
    return dict(original_path=str(path), flow_target=args['flow_target'], epoch=checkpoint['epoch'],
                step=checkpoint.get('step'), val_loss=checkpoint['val_loss'],
                planned_train_steps=args.get('expected_train_steps'), train_bs=args['train_bs'],
                train_epochs=args['train_epochs'],
                expectations={k: args[k] for k in ('separate_cond_gate', 'augment_source_swap', 'trellis_prior_weight')})

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')

def prepare(submit=False):
    assert not SUITE.exists(), 'Use a fresh suite root to avoid replacing frozen runs'
    meta = {key: metadata(path) for key, path in PATHS.items()}
    for key, row in meta.items():
        assert row['flow_target'] == key.split('_')[0]
        e = row['expectations']
        assert e['separate_cond_gate'] == ('token' if key.endswith('gate') else 'alpha')
        assert e['augment_source_swap'] == int(key.endswith('swap'))
        assert e['trellis_prior_weight'] == (0.1 if key.endswith('prior') else 0)
    assert meta['slat_prior']['train_bs'] == meta['slat_base']['train_bs'] == 1
    SUITE.mkdir(parents=True)
    write(SUITE/'checkpoint_inventory.json', meta)
    helper_spec = importlib.util.spec_from_file_location('threeway_layout', str(PROJECT/'slurm/threeway_ateneo.py'))
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    print('Freeze common source snapshot', flush=True)
    sources = helper.source_files(PROJECT, {'TRELLIS': BASE/'src/TRELLIS', 'MorphAny3D': BASE/'src/MorphAny3D'})
    source_hashes = {}
    for name, path in sources.items():
        target = SUITE/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(path), str(target))
        source_hashes[name] = helper.digest(target)
    pinned = {}
    old_manifest = json.loads((PREVIOUS/'run_manifest.json').read_text())
    for key, path in PATHS.items():
        target = SUITE/'checkpoint_bank'/key/'checkpoint.pt'
        if key.endswith('base'):
            flow = key.split('_')[0]
            existing = PREVIOUS/('checkpoints/'+flow+'/checkpoint.pt')
            old = old_manifest['checkpoints'][flow]
            assert existing.stat().st_size == path.stat().st_size == old['bytes']
            previous_meta = next(m for m in json.loads((PREVIOUS/'checkpoint_metadata.json').read_text()) if m['flow_target'] == flow)
            assert previous_meta['epoch'] == meta[key]['epoch'] and previous_meta['val_loss'] == meta[key]['val_loss']
            target.parent.mkdir(parents=True)
            target.symlink_to(existing)
            pinned[key] = dict(old, snapshot=str(target), reused_existing_snapshot=str(existing))
        else:
            print('Pin', key, 'best epoch', meta[key]['epoch'], 'step', meta[key]['step'], flush=True)
            pinned[key] = helper.copy_checkpoint(path, target)
    write(SUITE/'checkpoint_bank_manifest.json', pinned)
    selection = dict(chosen='slat_prior', reason='Same 8-GPU batch-1/global-8 training setup as SLat baseline; lower validation loss than global-4 H100 run. Chosen before benchmark metrics.',
                     selected=meta['slat_prior'], excluded=metadata(BANK/'v4_slat_ablation_prior_only_steps125000_e10_s42_g2_bs2/checkpoints/morphflow_best.pt'))
    write(SUITE/'slat_prior_selection.json', selection)
    prepared = []
    for name, ablation, dataset, ss_key, slat_key, grid in JOBS:
        run = SUITE/name
        run.mkdir()
        (run/'source_snapshot').symlink_to('../source_snapshot')
        reference = PREVIOUS if dataset == 'v4_test' else EXTERNAL
        plan = json.loads((reference/'shared/plan.json').read_text())
        assert plan['seed'] == 42 and plan['k_intermediate'] == 5 and plan['beta_shape'] == 5
        if dataset == 'interp3ddata':
            original_count = len(plan['pairs'])
            plan['pairs'] = sorted(random.Random(42).sample(plan['pairs'], 50), key=lambda p: p['pair_id'])
            plan['num_pairs'] = plan['requested_num_pairs'] = 50
            plan['subset_selection'] = dict(population_size=original_count, num_pairs=50, seed=42,
                    rule='random.Random(42).sample(existing full-plan pairs,50), preserve original pair IDs/noise seeds')
        assert len(plan['pairs']) == plan['num_pairs'] == 50
        data = Path(plan['data_root'])
        images = Path(plan['source_images_root'])
        original_metadata = Path(plan['metadata'])
        if not original_metadata.is_file():
            original_metadata = reference/'input_snapshot/metadata_test.json'
        metadata_path = run/'input_snapshot/metadata_test.json'
        metadata_path.parent.mkdir()
        shutil.copyfile(str(original_metadata), str(metadata_path))
        entries = json.loads(metadata_path.read_text())
        plan['metadata'] = str(metadata_path)
        for pair in plan['pairs']:
            entry = entries[pair['metadata_index']]
            assert (entry['src_1'], entry['src_2']) == (pair['src1'], pair['src2'])
            for field in ('src1_image', 'src2_image'):
                assert Path(pair[field]).is_file(), pair[field]
            for asset in (pair['src1'], pair['src2']):
                for file in ('ss_latent.pt','slat_feats.pt','slat_coords.pt'):
                    assert (data/'assets'/asset/file).is_file()
            for endpoint in ('src1_raw.pt','src2_raw.pt'):
                assert (reference/'shared/references_raw'/pair['pair_dir']/endpoint).is_file()
        write(run/'shared/plan.json', plan)
        # Complete, immutable canonical references/cache from the previous run;
        # every new run has its own generations, metrics and phase logs.
        (run/'shared/references_raw').symlink_to(reference/'shared/references_raw')
        (run/'shared/reference_metrics').symlink_to(reference/'shared/reference_metrics')
        for target, key in [('ss', ss_key), ('slat', slat_key)]:
            file = run/('checkpoints/'+target+'/checkpoint.pt')
            file.parent.mkdir(parents=True)
            file.symlink_to(SUITE/'checkpoint_bank'/key/'checkpoint.pt')
        (run/'cfg_grid.txt').write_text(''.join(cfg+' '+slat+'\n' for cfg,slat in grid))
        for cfg,slat in grid:
            cell=run/('cfg_ss_'+cfg+'_slat_'+slat)
            cell.mkdir()
            (cell/'plan.json').symlink_to('../shared/plan.json')
            (cell/'references_raw').symlink_to('../shared/references_raw')
        for file in ('resume_threeway_mesh.py','mesh_channel_blocks.py','resume_threeway_render_cache.py'):
            dest=run/'runtime'/file
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(str(EXTERNAL/'runtime'/file),str(dest))
        write(run/'runtime/render_repair.json',dict(old_root=str(run),new_root=str(run),frozen_evaluator=str(run/'source_snapshot/MorphFlow/benchmarks/eval_textured_benchmark.py')))
        config=dict(RUN_ROOT=str(run), ABLATION_NAME=ablation, DATASET_NAME=dataset,
                    DATA_ROOT_HOST=str(data), SOURCE_IMAGES_ROOT_HOST=str(images),
                    ORIGINAL_ENV=str(BASE/'envs/morphflow'), METRIC_ENV=str(BASE/'envs/morphany3d'),
                    CHECKPOINT_PATH=str(run/'checkpoints/ss/checkpoint.pt'), SLAT_CHECKPOINT_PATH=str(run/'checkpoints/slat/checkpoint.pt'))
        (run/'launch.env').write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in config.items()))
        write(run/'checkpoint_metadata.json',[meta[ss_key],meta[slat_key]])
        write(run/'run_manifest.json',dict(methods=['morphflow'],configuration=config,cfg_pairs=grid,num_pairs=50,
             k_intermediate=5,seed=42,alpha_schedule='beta',beta_shape=5,num_views=64,fid_num_views=1000,
             checkpoint_keys=dict(ss=ss_key,slat=slat_key),checkpoints={k:pinned[k] for k in (ss_key,slat_key)},
             checkpoint_expectations=dict(ss=meta[ss_key]['expectations'],slat=meta[slat_key]['expectations']),
             reference_plan=str(reference/'shared/plan.json'),reference_pair_ids=[p['pair_id'] for p in plan['pairs']],
             sources=source_hashes,metadata_sha256=helper.digest(metadata_path),weighted_dense=False,export_glb=False))
        launcher=run/'source_snapshot/MorphFlow/slurm/benchmark_ablation_morphflow_leonardo.slurm'
        subprocess.run(['bash','-n',str(launcher)],check=True)
        row=dict(run=name,ablation=ablation,dataset=dataset,ss_checkpoint=ss_key,slat_checkpoint=slat_key,cfg_pairs=grid,run_root=str(run))
        if submit:
            jobname=dict(gate_only_both='ablat_gate_both',prior_only_both='ablat_prior_both',swap_ss_only='ablat_swap_ss',baseline='base_interp3d50')[ablation]
            output=subprocess.check_output(['sbatch','--parsable','--job-name='+jobname,str(launcher),str(run)]).decode().strip()
            row['job_id']=output.split(';')[0]
        prepared.append(row)
        write(SUITE/'jobs.json',prepared)
        print('PREPARED',json.dumps(row),flush=True)
    with (SUITE/'jobs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(prepared[0]));writer.writeheader();writer.writerows(prepared)
    print('DONE',SUITE,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit',action='store_true')
    prepare(parser.parse_args().submit)
