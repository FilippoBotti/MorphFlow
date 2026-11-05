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
STEP_QUEUE = SLURM / 'queue_train_v4_steps.sh'
RUNTIME = SLURM / 'train_morphflow_v4_steps8_ateneo.sh'
WRAPPERS = {kind: SLURM / ('train_morphflow_v4_steps8_%s.slurm' % kind)
            for kind in ('ss', 'slat')}
STEP_WRAPPERS = {kind: SLURM / ('train_morphflow_v4_steps_%s.slurm' % kind)
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
                'LORA_ATTENTION_SCOPE', 'LORA_TARGET_MODULES', 'CUDA_VISIBLE_DEVICES',
                'DATA_STAGES', 'SS_RUN_NAME', 'SLAT_RUN_NAME'):
            env.pop(name, None)
    return env


class TrainingCommandTests(unittest.TestCase):
    def dry_run(self, kind, overrides=None, wrapper=None):
        env = clean_environment()
        env['PROJECT_DIR'] = str(ROOT)
        env.update(overrides or {})
        result = subprocess.run(['bash', str(wrapper or WRAPPERS[kind]), '--dry-run'],
                                env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        line = next((line for line in result.stdout.splitlines()
                     if line.startswith('TRAIN_COMMAND=')), None)
        self.assertIsNotNone(line, result.stdout)
        return result.stdout, shlex.split(line.split('=', 1)[1])

    def test_syntax_and_requested_node_gpu_allocations(self):
        for script in [QUEUE, STEP_QUEUE, RUNTIME] + list(WRAPPERS.values()) + list(STEP_WRAPPERS.values()):
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

    def test_step_presets_use_exact_limits_and_cross_lora_on_both_datasets(self):
        for stages in (8, 2):
            for kind, gpus, epochs, rate, limit in (('ss', 8, 24, 1e-4, 300000),
                                                    ('slat', 2, 6, 5e-5, 300000)):
                output, command = self.dry_run(kind, {'DATA_STAGES': str(stages)}, STEP_WRAPPERS[kind])
                args = training_parser().parse_args(command[command.index('main.py') + 1:])
                self.assertEqual(args.max_train_steps, limit)
                self.assertEqual(args.expected_train_steps, limit)
                self.assertEqual(args.expected_train_samples, 100000)
                self.assertEqual(args.train_epochs, epochs)
                self.assertEqual(args.lr, rate)
                self.assertEqual(args.lora_lr, rate)
                self.assertEqual(args.lora_attention_scope, 'cross')
                self.assertEqual(args.lora_target_modules, 'to_q,to_kv')
                self.assertEqual(args.train_bs, 1)
                self.assertEqual(args.augment_source_swap, 1)
                self.assertTrue(args.root_dir.endswith('/morphing_dataset_v4_steps%d' % stages))
                self.assertIn('_s%d' % limit, args.run_name)
                self.assertIn('steps%d' % stages, args.run_name)
                self.assertEqual(command[command.index('--num_processes') + 1], str(gpus))
                source = STEP_WRAPPERS[kind].read_text()
                self.assertIn('#SBATCH --nodelist=' + ('wn49' if kind == 'ss' else 'wn51'), source)

    def test_step_limit_override_rounds_epoch_ceiling_without_rounding_step_budget(self):
        output, command = self.dry_run('ss', {'MAX_TRAIN_STEPS': '12501'}, STEP_WRAPPERS['ss'])
        args = training_parser().parse_args(command[command.index('main.py') + 1:])
        self.assertEqual(args.train_epochs, 2)
        self.assertEqual(args.max_train_steps, 12501)
        self.assertEqual(args.expected_train_steps, 12501)
        self.assertIn('OPTIMIZER_STEPS=12501', output)


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

    def invoke(self, kind, wrapper=None):
        return subprocess.run(['bash', str(wrapper or WRAPPERS[kind])], env=self.env,
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

    def test_step_configuration_is_recorded_and_resume_keeps_global_step_limit(self):
        self.env.update(CUDA_VISIBLE_DEVICES='3,5', SLURM_GPUS_ON_NODE='2')
        result = self.invoke('slat', STEP_WRAPPERS['slat'])
        self.assertEqual(result.returncode, 0, result.stdout)
        run = Path(self.env['ARCHIVE_HOST']) / 'morphflow_runs/v4_steps8_slat_lora_cross_bs1_lr5e-5_s300000'
        config = json.loads((run / 'run_config.json').read_text())
        self.assertEqual(config['max_train_steps'], 300000)
        self.assertEqual(config['lora_attention_scope'], 'cross')
        self.assertEqual(config['lora_target_modules'], 'to_q,to_kv')
        checkpoint = run / 'checkpoints/morphflow_last.pt'
        checkpoint.write_bytes(b'checkpoint fixture')
        result = self.invoke('slat', STEP_WRAPPERS['slat'])
        self.assertEqual(result.returncode, 0, result.stdout)
        args = json.loads(self.capture.read_text())['args']
        self.assertEqual(args[args.index('--max_train_steps') + 1], '300000')
        self.assertEqual(args[args.index('--resume_from') + 1],
                         '/archive/morphflow_runs/' + run.name + '/checkpoints/morphflow_last.pt')
        self.env['MAX_TRAIN_STEPS'] = '300001'
        self.env['RUN_NAME'] = run.name
        result = self.invoke('slat', STEP_WRAPPERS['slat'])
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('Run configuration changed', result.stdout)

    def test_step_runtime_accepts_the_two_stage_view_and_preserves_parent_bind(self):
        two = self.store / 'morphing_dataset_v4_steps2'
        two.mkdir()
        for source in self.view.glob('*.json'):
            (two / source.name).write_bytes(source.read_bytes())
        (two / 'plan.json').write_text(json.dumps({'k': 2}))
        for name in ('assets', 'targets'):
            (two / name).symlink_to('../' + name)
        self.env.update(CUDA_VISIBLE_DEVICES='3,5', SLURM_GPUS_ON_NODE='2',
                        DATA_STAGES='2', DATA_ROOT_HOST=str(two))
        result = self.invoke('slat', STEP_WRAPPERS['slat'])
        self.assertEqual(result.returncode, 0, result.stdout)
        args = json.loads(self.capture.read_text())['args']
        self.assertEqual(args[args.index('--root_dir') + 1],
                         '/data/morphing_dataset_v4_steps8_steps2/morphing_dataset_v4_steps2')
        self.assertEqual(args[args.index('--max_train_steps') + 1], '300000')


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
        for script in [RUNTIME] + list(WRAPPERS.values()) + list(STEP_WRAPPERS.values()):
            (self.project / 'slurm' / script.name).write_text('#!/bin/bash\nexit 97\n')
        self.calls = self.root / 'submissions.jsonl'
        self.environments = self.root / 'environments.jsonl'
        self.env = clean_environment()
        self.env.update(PROJECT_DIR=str(self.project), DATA_PARENT_HOST=str(self.store),
                        DATA_ROOT_HOST=str(self.view), CAPTURE_SBATCH=str(self.calls),
                        CAPTURE_SBATCH_ENV=str(self.environments),
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
with open(os.environ['CAPTURE_SBATCH_ENV'], 'a') as handle:
    fields = ('DATA_ROOT_HOST', 'DATA_STAGES', 'TRAIN_EPOCHS', 'MAX_TRAIN_STEPS',
              'NGPUS', 'TRAIN_BS', 'TRAIN_LR', 'RUN_NAME', 'LORA_ATTENTION_SCOPE')
    handle.write(json.dumps({key: os.environ[key] for key in fields}) + '\\n')
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

    def test_step_queue_submits_requested_budgets_and_does_not_inherit_epoch_run_settings(self):
        self.env.update(MAX_TRAIN_STEPS='99', LORA_ATTENTION_SCOPE='all', TRAIN_EPOCHS='999')
        result = subprocess.run(['bash', str(STEP_QUEUE)], env=self.env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        for command, kind in zip(self.submissions(), ('ss', 'slat')):
            self.assertEqual(Path(command[-1]).name, STEP_WRAPPERS[kind].name)
            self.assertFalse(any(value.startswith('--dependency=') for value in command))
        environments = [json.loads(line) for line in self.environments.read_text().splitlines()]
        for env, limit, epochs, rate, gpus in zip(environments, (300000, 300000), (24, 6), ('1e-4', '5e-5'), (8, 2)):
            self.assertEqual(env['MAX_TRAIN_STEPS'], str(limit))
            self.assertEqual(env['TRAIN_EPOCHS'], str(epochs))
            self.assertEqual(env['LORA_ATTENTION_SCOPE'], 'cross')
            self.assertEqual(env['TRAIN_LR'], rate)
            self.assertEqual(env['NGPUS'], str(gpus))
            self.assertEqual(env['TRAIN_BS'], '1')

    def test_step_queue_can_select_two_stage_dataset_and_custom_limits(self):
        two = self.store / 'morphing_dataset_v4_steps2'
        two.mkdir()
        plan = dict(self.plan, k=2, pair_quotas={'train': 25000, 'val': 1000, 'test': 1000})
        self.write_json(two / 'plan.json', plan)
        self.write_json(two / 'run_state.json', {'phase': 'complete', 'status': 'complete'})
        for source in self.view.glob('metadata_*.json'):
            (two / source.name).write_bytes(source.read_bytes())
        self.env.pop('DATA_ROOT_HOST')
        result = self.run_queue('--steps', '--dataset-stages', '2', '--ss-steps', '12501', '--slat-steps', '50001')
        self.assertEqual(result.returncode, 0, result.stdout)
        environments = [json.loads(line) for line in self.environments.read_text().splitlines()]
        for env, limit in zip(environments, (12501, 50001)):
            self.assertEqual(env['DATA_STAGES'], '2')
            self.assertEqual(env['DATA_ROOT_HOST'], str(two))
            self.assertEqual(env['MAX_TRAIN_STEPS'], str(limit))
            self.assertEqual(env['TRAIN_EPOCHS'], '2')

    def test_step_check_and_invalid_limits_submit_no_jobs(self):
        result = self.run_queue('--steps', '--check')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.submissions(), [])
        for arguments in (('--steps', '--ss-steps', '0'), ('--steps', '--slat-steps', '-1')):
            result = self.run_queue(*arguments)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertEqual(self.submissions(), [])


if __name__ == '__main__':
    unittest.main()
