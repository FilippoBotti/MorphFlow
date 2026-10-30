"""CPU checks for the eight-stage training commands and scheduler dependencies."""
import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / 'slurm'
QUEUE = SLURM / 'queue_train_v4_steps8.sh'
RUNTIME = SLURM / 'train_morphflow_v4_steps8_ateneo.sh'
WRAPPERS = {kind: SLURM / ('train_morphflow_v4_steps8_%s.slurm' % kind)
            for kind in ('ss', 'slat')}


def training_parser():
    # Use the real parser and its argument helpers without loading GPU models.
    namespace = {'argparse': argparse, 'os': os}
    for source, name in (('modules/trellis_prior_training.py', 'add_trellis_prior_args'),
                         ('modules/prior_gradient_balance.py', 'add_balance_args'),
                         ('train.py', 'build_parser')):
        path = ROOT / source
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == name)
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(module, str(path), 'exec'), namespace)
    return namespace['build_parser']()


def clean_environment():
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(('BASH_FUNC_', 'SINGULARITYENV_', 'SLURM_')) or name in (
                'BASH_ENV', 'PROJECT_DIR', 'DATA_PARENT_HOST', 'DATA_ROOT_HOST',
                'DATA_PARENT_IN', 'DATA_ROOT_IN', 'RUN_NAME', 'RESUME_FROM',
                'AUTO_RESUME', 'CONDITION_GATE', 'SOURCE_SWAP', 'NGPUS', 'SEED',
                'TRAIN_BS', 'TRAIN_EPOCHS', 'TRAIN_LR', 'EXPECTED_TRAIN_STEPS',
                'MAX_TRAIN_STEPS', 'TARGET_TRAIN_STEPS', 'TRELLIS_PRIOR_WEIGHT',
                'LORA_ATTENTION_SCOPE', 'LORA_TARGET_MODULES', 'CUDA_VISIBLE_DEVICES'):
            env.pop(name, None)
    return env


class TrainingCommandTests(unittest.TestCase):
    def dry_run(self, kind):
        env = clean_environment()
        env['PROJECT_DIR'] = str(ROOT)
        result = subprocess.run(['bash', str(WRAPPERS[kind]), '--dry-run'],
                                env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        line = next((line for line in result.stdout.splitlines()
                     if line.startswith('TRAIN_COMMAND=')), None)
        self.assertIsNotNone(line, result.stdout)
        return result.stdout, shlex.split(line.split('=', 1)[1])

    def test_syntax_and_requested_node_gpu_allocations(self):
        for script in [QUEUE, RUNTIME] + list(WRAPPERS.values()):
            result = subprocess.run(['bash', '-n', str(script)],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    universal_newlines=True)
            self.assertEqual(result.returncode, 0, result.stdout)
        for kind, node, gres in (('ss', 'wn49', 'gpu:l40s_vbd:8'),
                                 ('slat', 'wn51', 'gpu:h100_vbd:2')):
            source = WRAPPERS[kind].read_text()
            self.assertIn('#SBATCH --nodelist=' + node, source)
            self.assertIn('#SBATCH --gres=' + gres, source)
            self.assertIn('#SBATCH --account=g_vbd', source)
            self.assertIn('#SBATCH --partition=gpu_vbd', source)

    def test_actual_commands_use_all_lora_projections_and_requested_training_budgets(self):
        for kind, gpus, epochs, rate in (('ss', '8', '40', '1e-4'),
                                        ('slat', '2', '20', '5e-5')):
            output, args = self.dry_run(kind)
            for flag, value in (('--num_processes', gpus), ('--flow_target', kind),
                                ('--train_bs', '1'), ('--train_epochs', epochs),
                                ('--lr', rate), ('--cond_lr', rate),
                                ('--flow_lr', rate), ('--lora_lr', rate),
                                ('--lora_attention_scope', 'all'), ('--use_lora', '1'),
                                ('--lora_target_modules', 'to_qkv,to_q,to_kv'),
                                ('--augment_source_swap', '1'),
                                ('--expected_train_samples', '100000'),
                                ('--separate_cond_gate', 'token')):
                self.assertIn(flag, args, (kind, output))
                self.assertEqual(args[args.index(flag) + 1], value, (kind, flag, output))
            self.assertNotIn('--max_train_steps', args)
            self.assertNotIn('--expected_train_steps', args)
            parsed = training_parser().parse_args(args[args.index('main.py') + 1:])
            self.assertEqual(parsed.flow_target, kind)
            self.assertEqual(parsed.train_epochs, int(epochs))
            self.assertEqual(parsed.lr, float(rate))
            self.assertIsNone(parsed.max_train_steps)
            dataset = args[args.index('--root_dir') + 1]
            self.assertTrue(dataset.endswith('/morphing_dataset_v4_steps8'), dataset)
            self.assertIn('steps8', args[args.index('--run_name') + 1])
            self.assertIn('GLOBAL_BATCH=' + gpus, output)


class RuntimeBoundaryTests(unittest.TestCase):
    """Exercise the real host runtime without executing its GPU container."""
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='steps8_runtime_')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.store = self.root / 'dataset'
        self.view = self.store / 'morphing_dataset_v4_steps8'
        self.view.mkdir(parents=True)
        for name in ('assets', 'targets'):
            (self.store / name).mkdir()
            (self.view / name).symlink_to('../' + name)
        (self.view / 'plan.json').write_text(json.dumps({'k': 8}))
        (self.view / 'run_state.json').write_text(json.dumps({'phase': 'complete', 'status': 'complete'}))
        for split, count in (('train', 50000), ('val', 2000), ('test', 2000)):
            (self.view / ('metadata_' + split + '.json')).write_text(json.dumps([{}] * count))
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.capture = self.root / 'container.json'
        module = self.bin / 'module'
        module.write_text('#!/bin/bash\nexit 0\n')
        module.chmod(0o755)
        singularity = self.bin / 'singularity'
        singularity.write_text('#!' + sys.executable + '''
import json, os, sys
with open(os.environ['CAPTURE_CONTAINER'], 'w') as handle:
    json.dump({'args': sys.argv[1:], 'stdin': sys.stdin.read(),
               'visible': os.environ['SINGULARITYENV_CUDA_VISIBLE_DEVICES'],
               'target': os.environ['SINGULARITYENV_FLOW_TARGET']}, handle)
''')
        singularity.chmod(0o755)
        self.env = clean_environment()
        self.env.update(PROJECT_DIR=str(ROOT), DATA_PARENT_HOST=str(self.store),
                        DATA_ROOT_HOST=str(self.view), ARCHIVE_HOST=str(self.root / 'archive'),
                        TRELLIS_BASE=str(self.root / 'trellis'),
                        FLUX_ROOT_HOST=str(self.root / 'images'),
                        BENCH_HF_HOME=str(self.root / 'hf-home'),
                        BENCH_TORCH_HOME=str(self.root / 'torch-home'),
                        SIF=str(self.root / 'container.sif'), CAPTURE_CONTAINER=str(self.capture),
                        PATH=str(self.bin) + os.pathsep + self.env['PATH'],
                        SLURM_JOB_ID='90001', SLURM_JOB_NUM_NODES='1')
        for directory in ('TRELLIS_BASE', 'FLUX_ROOT_HOST', 'BENCH_HF_HOME', 'BENCH_TORCH_HOME'):
            Path(self.env[directory]).mkdir()
        (Path(self.env['TRELLIS_BASE']) / 'env').mkdir()
        dino = Path(self.env['BENCH_TORCH_HOME']) / 'hub/facebookresearch_dinov2_main'
        dino.mkdir(parents=True)
        (dino / 'hubconf.py').write_text('# fixture\n')
        Path(self.env['SIF']).touch()

    def invoke(self, kind):
        return subprocess.run(['bash', str(WRAPPERS[kind])], env=self.env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True)

    def test_parent_dataset_bind_and_allocated_gpu_mask_cross_container_boundary(self):
        for kind, count, mask in (('ss', 8, '0,1,2,3,4,5,6,7'), ('slat', 2, '3,5')):
            self.env.update(CUDA_VISIBLE_DEVICES=mask, SLURM_GPUS_ON_NODE=str(count))
            result = self.invoke(kind)
            self.assertEqual(result.returncode, 0, result.stdout)
            captured = json.loads(self.capture.read_text())
            self.assertEqual(captured['visible'], mask)
            self.assertEqual(captured['target'], kind)
            self.assertIn(str(self.store) + ':/data/morphing_dataset_v4_steps8_steps2:ro', captured['args'])
            self.assertNotIn(str(self.view) + ':/data/morphing_dataset_v4_steps8:ro', captured['args'])
            args = captured['args']
            self.assertEqual(args[args.index('--flow_target') + 1], kind)
            self.assertEqual(args[args.index('--lora_target_modules') + 1], 'to_qkv,to_q,to_kv')
            self.assertIn('torch.cuda.device_count() != expected', captured['stdin'])

    def test_wrong_gpu_allocation_fails_before_container_is_started(self):
        self.env.update(CUDA_VISIBLE_DEVICES='3', SLURM_GPUS_ON_NODE='2')
        result = self.invoke('slat')
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('NGPUS differs from CUDA_VISIBLE_DEVICES', result.stdout)
        self.assertFalse(self.capture.exists())

    def test_existing_checkpoint_cannot_be_overwritten_with_resume_disabled(self):
        self.env.update(CUDA_VISIBLE_DEVICES='3,5', SLURM_GPUS_ON_NODE='2')
        result = self.invoke('slat')
        self.assertEqual(result.returncode, 0, result.stdout)
        run = Path(self.env['ARCHIVE_HOST']) / 'morphflow_runs/v4_steps8_slat_lora_all_bs1_lr5e5_e20'
        (run / 'checkpoints/morphflow_last.pt').write_bytes(b'checkpoint fixture')
        self.capture.unlink()
        self.env['AUTO_RESUME'] = '0'
        result = self.invoke('slat')
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('Existing checkpoints', result.stdout)
        self.assertFalse(self.capture.exists())


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='steps8_queue_')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.store = self.root / 'dataset'
        self.view = self.store / 'morphing_dataset_v4_steps8'
        self.view.mkdir(parents=True)
        for name in ('assets', 'targets'):
            (self.store / name).mkdir()
            (self.view / name).symlink_to('../' + name)
        self.plan = {'format': 'morphflow_interp3d_v4_nested_view', 'version': 4,
                     'k': 8, 'num_targets': 54000, 'num_pairs': 6750,
                     'pair_quotas': {'train': 6250, 'val': 250, 'test': 250},
                     'train_virtual_after_swap': 100000,
                     'endpoint_fraction': 0.02, 'master_root': '..'}
        self.write_json(self.view / 'plan.json', self.plan)
        self.write_json(self.store / 'plan.json', {
            'format': 'morphflow_interp3d_v4_multistep', 'version': 4,
            'k': None, 'views': {'steps8': {'name': self.view.name, 'k': 8}}})
        for split, count in (('train', 50000), ('val', 2000), ('test', 2000)):
            self.write_json(self.view / ('metadata_' + split + '.json'), [{}] * count)
        self.state('complete', 'complete')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.project = self.root / 'project'
        (self.project / 'slurm').mkdir(parents=True)
        # Queue checks only that these entry points exist. Real wrapper
        # commands and resource requests are exercised separately above.
        for script in [RUNTIME] + list(WRAPPERS.values()):
            (self.project / 'slurm' / script.name).write_text('#!/bin/bash\nexit 97\n')
        self.calls = self.root / 'submissions.jsonl'
        self.env = clean_environment()
        self.env.update(PROJECT_DIR=str(self.project), DATA_PARENT_HOST=str(self.store),
                        DATA_ROOT_HOST=str(self.view), CAPTURE_SBATCH=str(self.calls),
                        PATH=str(self.bin) + os.pathsep + self.env['PATH'],
                        FAKE_JOB_STATE='RUNNING',
                        FAKE_WN49_GRES='gpu:l40s_vbd:8(S:0-1)',
                        FAKE_WN51_GRES='gpu:h100_vbd:2(S:0-1)')
        self.tool('sinfo', '''
import os, sys
args = sys.argv[1:]
node = args[args.index('-n') + 1]
print(node + ' ' + os.environ['FAKE_' + node.upper() + '_GRES'])
''')
        self.tool('squeue', '''
import os
print(os.environ['FAKE_JOB_STATE'])
''')
        self.tool('sbatch', '''
import json, os, sys
path = os.environ['CAPTURE_SBATCH']
try:
    with open(path) as handle:
        count = len(handle.readlines())
except IOError:
    count = 0
with open(path, 'a') as handle:
    handle.write(json.dumps(sys.argv[1:]) + '\\n')
print(700001 + count)
''')

    @staticmethod
    def write_json(path, payload):
        path.write_text(json.dumps(payload))

    def state(self, phase, status, job='2594233'):
        payload = {'phase': phase, 'status': status, 'job_id': job, 'run_id': 'fixture'}
        self.write_json(self.store / 'run_state.json', payload)
        self.write_json(self.view / 'run_state.json', payload)

    def tool(self, name, source):
        path = self.bin / name
        path.write_text('#!' + sys.executable + '\n' + source.lstrip())
        path.chmod(0o755)

    def run_queue(self, *args):
        return subprocess.run(['bash', str(QUEUE)] + list(args), env=self.env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True)

    def submissions(self):
        if not self.calls.exists():
            return []
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def assert_two_jobs(self, dependency=None):
        submissions = self.submissions()
        self.assertEqual(len(submissions), 2, submissions)
        for args, kind in zip(submissions, ('ss', 'slat')):
            self.assertEqual(Path(args[-1]).name, WRAPPERS[kind].name)
            deps = [arg for arg in args if arg.startswith('--dependency=')]
            self.assertEqual(deps, [] if dependency is None else [dependency])
        return submissions

    def test_complete_dataset_submits_both_jobs_without_dependency(self):
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assert_two_jobs()

    def test_running_generation_is_dependency_for_both_independent_jobs(self):
        self.state('assets', 'running')
        # Metadata is published by finalize, after the generation job finishes.
        for path in self.view.glob('metadata_*.json'):
            path.unlink()
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stdout)
        submissions = self.assert_two_jobs('--dependency=afterok:2594233')
        self.assertNotIn('--dependency=afterok:700001', submissions[1])

    def test_explicit_generation_job_overrides_recorded_job(self):
        self.state('assets', 'running')
        result = self.run_queue('--generation-job', '2594999')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assert_two_jobs('--dependency=afterok:2594999')

    def test_failed_or_inactive_generation_submits_no_training(self):
        self.state('assets', 'failed')
        self.env['FAKE_JOB_STATE'] = 'FAILED'
        result = self.run_queue()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.submissions(), [])

    def test_gpu_type_or_capacity_mismatch_submits_neither_job(self):
        for bad_gres in ('gpu:h100_vbd:1(S:0)', 'gpu:a100_vbd:8(S:0-1)'):
            self.env['FAKE_WN51_GRES'] = bad_gres
            result = self.run_queue()
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertEqual(self.submissions(), [])

    def test_check_mode_never_submits_ready_or_waiting_training(self):
        for phase, status in (('complete', 'complete'), ('pairs', 'running')):
            self.state(phase, status)
            result = self.run_queue('--check')
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(self.submissions(), [])

    def test_wrong_dataset_stage_count_is_rejected_before_submission(self):
        self.plan['k'] = 2
        self.write_json(self.view / 'plan.json', self.plan)
        result = self.run_queue()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.submissions(), [])


if __name__ == '__main__':
    unittest.main()
