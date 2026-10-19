"""Exercise the native one-GPU launcher's real layout with fake GPU workers.

The three Python executables below stand in for the existing, separate model
environments. Inference/rendering are simulated; the actual shell launcher and
stdlib snapshot/layout/summarizer run unchanged. No model package is imported.
"""

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SLURM_DIR = Path(__file__).resolve().parent

FAKE_PYTHON = r'''import csv
import json
import os
from pathlib import Path
import sys

command = sys.argv[1:]
environment = str(Path(sys.argv[0]).parent.parent)
with open(os.environ['FAKE_NATIVE_JOURNAL'], 'a') as stream:
    stream.write(json.dumps(dict(command=command, environment=environment,
        pythonpath=os.environ.get('PYTHONPATH'),
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))) + '\n')

if Path(command[0]).name == 'threeway_ateneo.py' or (
        command[0] == '-c' and 'requested_configuration' in command[1]):
    os.execv(sys.executable,[sys.executable]+command)

def option(flag):
    return command[command.index(flag)+1]

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))

def plan(root):
    return json.loads((Path(root)/'plan.json').read_text())

def selected_methods():
    if '--methods' not in command:
        return ['morphflow', 'morphany3d', 'interp3d']
    start = command.index('--methods')+1
    stop = next((i for i in range(start, len(command)) if command[i].startswith('--')), len(command))
    return command[start:stop]

weighted = {key: dict(mean=.1, std=.01, median=.1) for key in
    ['weighted_endpoint_lpips_mean', 'weighted_endpoint_lpips_auc',
     'weighted_endpoint_lpips_full_mean', 'weighted_endpoint_lpips_full_auc']}
standard = dict(weighted, **{key: dict(mean=1., std=.1, median=1.) for key in
    ['fid_endpoint_manifold', 'fid_endpoint_manifold_mean_per_state',
     'fid_endpoint_manifold_pooled', 'ppl_full_normalized_reference_endpoints', 'pdv_full']})

if command[0] == '-c':
    # Import/device and checkpoint metadata preflights are GPU work stand-ins.
    if 'Path(sys.argv[1]).write_text' in command[1]:
        write_json(Path(os.environ['RUN_ROOT'])/'checkpoint_metadata.json',
                   [dict(flow_target='ss'), dict(flow_target='slat')])
elif Path(command[0]).name == 'prepare_interp3ddata.py':
    root = Path(option('--output-root'))
    assert '--prepare' in command or '--index-only' in command
    root.mkdir(parents=True, exist_ok=True)
    (root/'images').mkdir(exist_ok=True)
    pairs = [dict(src_1='easy__a', src_2='easy__b', alpha=.5),
             dict(src_1='hard__c', src_2='hard__d', alpha=.5)]
    write_json(root/'metadata_test.json', pairs)
    write_json(root/'split_assets.json', dict(test=['easy__a','easy__b','hard__c','hard__d']))
    write_json(root/'dataset_provenance.json', dict(endpoint_seed=42, evaluated_pairs=2))
    if '--prepare' in command:
        for name in ('easy__a','easy__b','hard__c','hard__d'):
            asset = root/'assets'/name
            asset.mkdir(parents=True, exist_ok=True)
            for filename in ('ss_latent.pt','slat_feats.pt','slat_coords.pt'):
                (asset/filename).write_bytes(b'fake canonical endpoint')
            write_json(asset/'endpoint_ready.json', dict(seed=42))
elif Path(command[0]).name == 'weighted_endpoint_report.py':
    root = Path(option('--run-root'))
    for directory in [root]+[p.parent for p in root.glob('cfg_ss_*/final_metrics.json')]:
        report = directory/'weighted_endpoint_lpips'
        report.mkdir(exist_ok=True)
        (report/'evolution.png').write_bytes(b'CPU plot fixture')
        (report/'evolution.pdf').write_bytes(b'CPU plot fixture')
        (report/'curves.csv').write_text('alpha,weighted_lpips_mean\n0.5,0.1\n')
        (report/'summary.csv').write_text('method,mean\nmorphflow,0.1\n')
        write_json(report/'summary.json', dict(metric='weighted_endpoint_lpips'))
else:
    operation = command[1]
    if os.environ.get('FAKE_FAIL_OPERATION') == operation:
        print('simulated native worker failure: '+operation,file=sys.stderr)
        raise SystemExit(17)
    if operation == 'check-fid-weights':
        print('fake cached FID weights')
    elif operation == 'plan':
        root = Path(option('--run-root'))
        dense = '--protocol' in command and option('--protocol') == 'weighted-endpoints-50'
        if dense:
            assert option('--k') == '48' and option('--alpha-schedule') == 'uniform'
            assert '--beta-shape' not in command
            alphas = [i/49 for i in range(48,0,-1)]
        else:
            assert option('--k') == '5' and option('--alpha-schedule') == 'beta'
            assert option('--beta-shape') == '5'
            alphas = [.7,.6,.5,.4,.3]
        manifest = json.loads((Path(os.environ['RUN_ROOT'])/'run_manifest.json').read_text())
        assert option('--num-pairs') == '0'
        assert Path(option('--metadata')).read_bytes() == (Path(os.environ['RUN_ROOT'])/'input_snapshot/metadata_test.json').read_bytes()
        pairs = [dict(pair_id=i,pair_dir='pair_%04d_a_to_b'%i) for i in range(manifest['evaluated_pairs'])]
        write_json(root/'plan.json', dict(pairs=pairs,num_pairs=len(pairs),alphas=alphas,
            sequence_alphas=[1.]+alphas+[0.],k_intermediate=len(alphas),num_sequence_frames=len(alphas)+2,
            metric_mode='weighted-endpoints-only' if dense else 'all'))
    elif operation in {'morphflow','morphany3d','interp3d'}:
        assert option('--steps') == '25' and option('--slat-steps') == '25'
        root = Path(option('--run-root'))
        assert len(plan(root)['pairs']) == 2
        if operation == 'morphflow':
            assert Path(option('--checkpoint-path')).read_bytes() == b'SS chosen epoch17'
            assert Path(option('--slat-checkpoint-path')).read_bytes() == b'SLat chosen epoch9'
            assert '/checkpoints/ss/checkpoint.pt' in option('--checkpoint-path')
        for pair in plan(root)['pairs']:
            for i in range(len(plan(root)['alphas'])):
                raw = root/operation/pair['pair_dir']/('alpha_'+str(i))/'mesh_raw.pt'
                raw.parent.mkdir(parents=True,exist_ok=True)
                raw.write_bytes(b'appearance-aware mesh/Gaussian fixture')
                write_json(raw.with_suffix('.appearance.json'),dict(format_version=2))
        timing = root/'timings'/(operation+'_shard_0.jsonl')
        timing.parent.mkdir(parents=True,exist_ok=True)
        timing.write_text('{}\n')
    elif operation == 'shard':
        assert option('--num-views') == '64' and option('--lpips-net') == 'vgg'
        assert option('--render-backend') == 'gaussian'
        if '--metric-mode' in command:
            assert option('--metric-mode') == 'weighted-endpoints-only'
            assert not any(argument.startswith('--fid-') for argument in command)
        else:
            assert option('--fid-num-views') == '1000'
            assert option('--fid-aggregation') == 'both'
        write_json(Path(option('--output-dir'))/'fixture_shard.json',[0,1])
    elif operation == 'merge':
        assert option('--expected-num-pairs') == '2'
        dense = '--metric-mode' in command
        if not dense:
            assert option('--fid-aggregation') == 'both'
        write_json(Path(option('--output-dir'))/'metrics_batch.json',
                   dict(num_pairs=2,missing_pairs=[],aggregate_across_pairs=weighted if dense else standard))
    elif operation == 'aggregate':
        root = Path(option('--run-root'))
        data = dict(plan(root),methods={})
        for method in selected_methods():
            data['methods'][method] = json.loads((root/'metrics'/method/'metrics_batch.json').read_text())
        write_json(root/'final_metrics.json',data)
        with (root/'final_metrics.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=['method','metric','mean','std','median'])
            writer.writeheader()
            writer.writerows(dict(method=method,metric=metric,**value)
                for method,result in data['methods'].items() for metric,value in result['aggregate_across_pairs'].items())
    else:
        raise AssertionError('Unexpected native command: '+repr(command))
'''


class NativeInterp3DDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='interp3ddata_native_')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root/'project'
        (self.project/'slurm').mkdir(parents=True)
        for filename in ('benchmark_interp3ddata_leonardo.slurm','threeway_ateneo.py'):
            shutil.copyfile(SLURM_DIR/filename,self.project/'slurm'/filename)
        (self.project/'benchmarks').mkdir()
        for filename in ('prepare_interp3ddata.py','threeway_textured_benchmark.py',
                         'eval_textured_benchmark.py','weighted_endpoint_report.py'):
            (self.project/'benchmarks'/filename).write_text('# native GPU fixture\n')
        self.journal = self.root/'native_calls.jsonl'
        self.run_root = self.root/'run'
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith('BASH_FUNC_') or key in {'BASH_ENV','ENV','CONFIG_FILE','SIF','PROTOCOL_PYTHON'}:
                self.env.pop(key)
        self.env.update(dict(BASE_DIR=str(self.root),PROJECT_DIR=str(self.project),HOST_PYTHON=sys.executable,
            RUN_ROOT=str(self.run_root),RUN_NAME='native_fixture',FINAL_ARCHIVE=str(self.root/'reports'),
            DATASET_INPUT_ROOT=str(self.root/'raw'),DATA_ROOT_HOST=str(self.root/'prepared'),
            SOURCE_IMAGES_ROOT_HOST=str(self.root/'prepared/images'),
            CFG_VALUES='2 5',SLAT_CFG_VALUES='1.5',N='0',SEED='42',WEIGHTED_DENSE='1',EXPORT_GLB='0',
            SLURM_JOB_ID='90301',SLURM_JOB_NUM_NODES='1',SLURM_CPUS_PER_TASK='4',
            CUDA_VISIBLE_DEVICES='GPU-native-test',FAKE_NATIVE_JOURNAL=str(self.journal)))
        (self.root/'raw').mkdir()
        for key in ('ORIGINAL_ENV','MORPHANY_ENV','INTERP_ENV'):
            environment = self.root/key
            (environment/'bin').mkdir(parents=True)
            executable = environment/'bin/python'
            executable.write_text('#!'+sys.executable+'\n'+FAKE_PYTHON)
            executable.chmod(0o755)
            self.env[key] = str(environment)
        for key in ('TRELLIS_REPO_HOST','MORPHANY_REPO','INTERP_REPO'):
            repository = self.root/key
            (repository/'trellis').mkdir(parents=True)
            (repository/'trellis/__init__.py').write_text('FORK='+repr(key)+'\n')
            self.env[key] = str(repository)
        for key,data in (('CHECKPOINT_PATH',b'SS chosen epoch17'),('SLAT_CHECKPOINT_PATH',b'SLat chosen epoch9')):
            checkpoint = self.root/(key+'.pt')
            checkpoint.write_bytes(data)
            self.env[key] = str(checkpoint)
        self.environment_file = self.root/'env.sh'
        self.environment_file.write_text('export MORPHFLOW_ENV="'+self.env['ORIGINAL_ENV']+'"\n'
            'export MORPHFLOW_REPO="'+str(self.project)+'"\n'
            'export TRELLIS_REPO="'+self.env['TRELLIS_REPO_HOST']+'"\n')
        self.env['MF_ENV_FILE'] = str(self.environment_file)

    def run_launcher(self, mode='--run'):
        result = subprocess.run(['bash',str(self.project/'slurm/benchmark_interp3ddata_leonardo.slurm'),mode],
            cwd=self.project,env=self.env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
        return result

    def calls(self):
        return [json.loads(line) for line in self.journal.read_text().splitlines()]

    def test_two_protocols_share_baselines_across_cfg_and_keep_native_envs_isolated(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        calls = self.calls()
        operations = {}
        for call in calls:
            command = call['command']
            if len(command)>1 and command[0]!='-c':
                operations.setdefault(command[1],[]).append(call)
        self.assertEqual(len(operations['plan']),2)
        self.assertEqual(len(operations['morphflow']),4)
        for method in ('morphany3d','interp3d'):
            self.assertEqual(len(operations[method]),2,method+' must run once per protocol')
        self.assertNotIn('texture',operations)
        self.assertNotIn('texture-status',operations)
        mapping = {'morphflow':('ORIGINAL_ENV','TRELLIS'), 'morphany3d':('MORPHANY_ENV','MorphAny3D'),
                   'interp3d':('INTERP_ENV','Interp3D')}
        for method,(key,fork) in mapping.items():
            for call in operations[method]:
                self.assertEqual(call['environment'],self.env[key])
                self.assertEqual(call['gpu'],'GPU-native-test')
                paths = call['pythonpath'].split(os.pathsep)
                self.assertEqual(paths[:2],[str(self.run_root/'source_snapshot'/fork),
                                           str(self.run_root/'source_snapshot/MorphFlow')])
        for operation in ('shard','merge','aggregate'):
            self.assertTrue(operations[operation])
            for call in operations[operation]:
                self.assertEqual(call['environment'],self.env['MORPHANY_ENV'])
        manifest = json.loads((self.run_root/'run_manifest.json').read_text())
        self.assertEqual(manifest['evaluated_pairs'],2)
        self.assertEqual(manifest['configuration']['EXECUTION_BACKEND'],'native')
        self.assertNotIn('SIF',manifest['configuration'])
        self.assertEqual((self.run_root/'checkpoints/ss/checkpoint.pt').read_bytes(),b'SS chosen epoch17')
        for workload in (self.run_root,self.run_root/'weighted_50'):
            grid = json.loads((workload/'cfg_grid_metrics.json').read_text())
            self.assertEqual(grid['completed_configurations'],2)
            for cfg in ('2','5'):
                cell = workload/('cfg_ss_'+cfg+'_slat_1.5')
                self.assertTrue((cell/'morphany3d').is_symlink())
                self.assertTrue((cell/'interp3d').is_symlink())
                self.assertTrue((cell/'final_metrics.csv').is_file())
        self.assertFalse(list(self.run_root.rglob('*.glb')))
        self.assertTrue((self.root/'reports/cfg_grid_metrics.csv').is_file())
        self.assertTrue((self.root/'reports/weighted_50/weighted_endpoint_lpips/evolution.png').is_file())

    def test_check_indexes_images_without_preparing_gpu_endpoints_or_run_snapshot(self):
        result = self.run_launcher('--check')
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        preparation = [call for call in self.calls()
                       if Path(call['command'][0]).name == 'prepare_interp3ddata.py']
        self.assertEqual(len(preparation),1)
        self.assertIn('--index-only',preparation[0]['command'])
        self.assertEqual(preparation[0]['environment'],self.env['INTERP_ENV'])
        self.assertFalse(self.run_root.exists())
        self.assertFalse((self.root/'prepared/assets').exists())
        for call in self.calls():
            if call['command'][0] == '-c':
                self.assertNotIn('torch',call['command'][1])
        self.assertIn('CHECK PASSED',result.stdout)

    def test_failed_native_phase_stops_job_and_reports_its_phase_log(self):
        self.env['FAKE_FAIL_OPERATION'] = 'morphany3d'
        result = self.run_launcher()
        self.assertEqual(result.returncode,17,result.stdout+'\n'+result.stderr)
        self.assertIn('morphany3d_generation failed',result.stderr)
        self.assertIn('simulated native worker failure: morphany3d',result.stderr)
        generated = [call['command'][1] for call in self.calls()
                     if len(call['command'])>1 and call['command'][0]!='-c']
        self.assertNotIn('interp3d',generated)
        self.assertNotIn('shard',generated)
        self.assertFalse((self.run_root/'cfg_grid_metrics.json').exists())


if __name__ == '__main__':
    unittest.main()
