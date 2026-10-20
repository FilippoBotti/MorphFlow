"""Exercise actual launcher arguments using a fake scheduler/container, without GPUs."""
import ast
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT/'slurm/train_ss_ablation_ateneo.sh'
EXPERIMENTS = {'baseline':('alpha','0','0.0','0',20), 'gate_only':('token','0','0.0','1',20),
               'swap_only':('alpha','1','0.0','0',10), 'prior_only':('alpha','0','0.1','0',20)}


def train_parser():
    # Load the real CLI definition without loading multi-billion-parameter models.
    from modules.trellis_prior_training import add_trellis_prior_args
    # This argument factory only calls argparse: load its real definition
    # directly, avoiding the unrelated module-level Torch/CUDA imports.
    balance_path=ROOT/'modules/prior_gradient_balance.py'
    balance_tree=ast.parse(balance_path.read_text())
    balance_function=next(node for node in balance_tree.body if isinstance(node,ast.FunctionDef) and node.name=='add_balance_args')
    balance_namespace={}
    exec(compile(ast.Module(body=[balance_function],type_ignores=[]),str(balance_path),'exec'),balance_namespace)
    add_balance_args=balance_namespace['add_balance_args']
    tree = ast.parse((ROOT/'train.py').read_text())
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='build_parser')
    namespace = dict(argparse=argparse,os=os,add_trellis_prior_args=add_trellis_prior_args,add_balance_args=add_balance_args)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(ROOT/'train.py'),'exec'),namespace)
    return namespace['build_parser']()


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.env = os.environ.copy()
        # The real HPC module() is an exported shell function and would shadow
        # our fake executable even with the temporary bin directory first.
        for key in list(self.env):
            if key.startswith('BASH_FUNC_'):
                self.env.pop(key)
        self.env.update(PROJECT_DIR=str(ROOT),TRAIN_BS='1',SEED='42',NGPUS='2')
        for name in ('RUN_NAME','RESUME_FROM','AUTO_RESUME','BASH_ENV'):
            self.env.pop(name,None)
        self.root = root

    def test_four_wrappers_are_syntactically_valid_and_request_two_a100s(self):
        for name in EXPERIMENTS:
            wrapper = ROOT/('slurm/train_ss_ablation_'+name+'_ateneo.slurm')
            subprocess.run(['bash','-n',str(wrapper)],check=True)
            text = wrapper.read_text()
            for expected in ('--nodelist=wn46','--gres=gpu:a100_vbd:2','--cpus-per-task=28','--mem=400G','--mail-type=BEGIN,END,FAIL'):
                self.assertIn(expected,text)
            result = subprocess.run(['bash',str(wrapper),'--dry-run'],env=self.env,check=True,text=True,stdout=subprocess.PIPE)
            gate,swap,prior,_,epochs = EXPERIMENTS[name]
            self.assertIn('EPOCHS='+str(epochs),result.stdout)
            self.assertIn('TRAIN_SAMPLES='+str(50000*(1+int(swap))),result.stdout)
            self.assertIn('GLOBAL_BATCH=2',result.stdout)
            self.assertIn('GATE='+gate,result.stdout)
            self.assertIn('SOURCE_SWAP='+swap,result.stdout)
            self.assertIn('PRIOR_WEIGHT='+prior,result.stdout)
        subprocess.run(['bash','-n',str(COMMON)],check=True)

    def test_dry_run_batch_scales_with_gpu_allocation_and_per_gpu_batch(self):
        self.env.update(NGPUS='8',TRAIN_BS='2')
        result = subprocess.run(['bash',str(COMMON),'baseline','--dry-run'],env=self.env,check=True,text=True,stdout=subprocess.PIPE)
        self.assertIn('GLOBAL_BATCH=16',result.stdout)
        self.assertIn('EPOCHS=20',result.stdout)

    def prepare_fake_runtime(self):
        root = self.root
        for name in ('dataset','archive','hf','torch','trellis/env/bin','bin','flux'):
            (root/name).mkdir(parents=True,exist_ok=True)
        (root/'trellis/env/bin/python').symlink_to(sys.executable)
        (root/'dataset/run_state.json').write_text(json.dumps({'phase':'complete','status':'complete'}))
        for split,count in (('train',50000),('val',2000),('test',2000)):
            (root/('dataset/metadata_'+split+'.json')).write_text(json.dumps([{}]*count))
        (root/'torch/hub/facebookresearch_dinov2_main').mkdir(parents=True)
        (root/'torch/hub/facebookresearch_dinov2_main/hubconf.py').write_text('# cached')
        (root/'container.sif').touch()
        (root/'bin/module').write_text('#!/bin/sh\nexit 0\n')
        capture = root/'capture.py'
        capture.write_text("import json,os,sys\nopen(os.environ['CAPTURE_PATH'],'w').write(json.dumps(sys.argv[1:]))\n")
        singularity = root/'bin/singularity'
        singularity.write_text('''#!SHEBANG
import os,subprocess,sys
text=sys.stdin.read()
segment=text[text.index('resume_args=()'):text.index('echo "RUN_DIR=')]
env=os.environ.copy()
env.update({k[len('SINGULARITYENV_'):]:v for k,v in os.environ.items() if k.startswith('SINGULARITYENV_')})
# Execute the original training command/arrays, replacing only accelerate.
prefix='accelerate() { "'+sys.executable+'" "'+os.environ['CAPTURE_SCRIPT']+'" "$@"; }\\n'
subprocess.run(['bash','-c',prefix+segment],env=env,check=True)
'''.replace('SHEBANG',sys.executable))
        for script in ('module','singularity'):
            (root/'bin'/script).chmod(0o755)
        self.env.update(PATH=str(root/'bin')+os.pathsep+self.env['PATH'],
                        SLURM_JOB_ID='123456',SLURM_JOB_NUM_NODES='1',SLURM_GPUS_ON_NODE='2',
                        CUDA_VISIBLE_DEVICES='0,2',ARCHIVE_HOST=str(root/'archive'),DATA_ROOT_HOST=str(root/'dataset'),
                        BENCH_HF_HOME=str(root/'hf'),BENCH_TORCH_HOME=str(root/'torch'),
                        TRELLIS_BASE=str(root/'trellis'),SIF=str(root/'container.sif'),FLUX_ROOT_HOST=str(root/'flux'),
                        CAPTURE_PATH=str(root/'captured.json'),CAPTURE_SCRIPT=str(capture))

    def run_fake(self,variant,check=True):
        result = subprocess.run(['bash',str(COMMON),variant],env=self.env,check=False,text=True,
                                stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if check:
            self.assertEqual(result.returncode,0,result.stdout)
        return result

    def test_real_four_training_commands_parse_and_differ_only_in_requested_components(self):
        self.prepare_fake_runtime()
        parser = train_parser()
        parsed = {}
        for name,(gate,swap,prior,train_gate,epochs) in EXPERIMENTS.items():
            self.run_fake(name)
            command = json.loads((self.root/'captured.json').read_text())
            self.assertEqual(command[0],'launch')
            self.assertIn('--multi_gpu',command)
            self.assertEqual(command[command.index('--num_processes')+1],'2')
            args = parser.parse_args(command[command.index('main.py')+1:])
            self.assertEqual(args.train_epochs,epochs)
            self.assertEqual(args.train_epochs*50000*(1+args.augment_source_swap),1000000)
            self.assertEqual(args.train_bs,1)
            self.assertEqual(args.flow_target,'ss')
            self.assertEqual(args.separate_cond,1)
            self.assertEqual(args.separate_cond_gate,gate)
            self.assertEqual(args.augment_source_swap,int(swap))
            self.assertEqual(args.trellis_prior_weight,float(prior))
            self.assertEqual(args.train_flow_alpha_gate,int(train_gate))
            self.assertNotIn('--source_swap_mode',command)
            self.assertEqual(args.seed,42)
            self.assertEqual(args.use_lora,1)
            self.assertEqual(args.use_semantic_token_matching,0)
            self.assertIsNone(args.init_from)
            self.assertIsNone(args.resume_from)
            parsed[name] = vars(args)
        baseline = parsed['baseline']
        for name in ('gate_only','swap_only','prior_only'):
            differences = {key for key in baseline if baseline[key]!=parsed[name][key]}
            expected = {'run_name'} | {'gate_only':{'separate_cond_gate','train_flow_alpha_gate'},
                                      'swap_only':{'augment_source_swap','train_epochs'},
                                      'prior_only':{'trellis_prior_weight'}}[name]
            self.assertEqual(differences,expected)

    def test_resume_restores_only_own_checkpoint_and_rejects_changed_batch(self):
        self.prepare_fake_runtime()
        self.env['RUN_NAME']='resume_check'
        self.run_fake('baseline')
        run=self.root/'archive/morphflow_runs/ablations_ss/resume_check'
        (run/'checkpoints/morphflow_last.pt').touch()
        self.run_fake('baseline')
        args=json.loads((self.root/'captured.json').read_text())
        self.assertEqual(args[args.index('--resume_from')+1],'/archive/morphflow_runs/ablations_ss/resume_check/checkpoints/morphflow_last.pt')
        self.assertEqual(args[args.index('--resume_optimizer')+1],'1')
        self.env['TRAIN_BS']='2'
        failed=self.run_fake('baseline',check=False)
        self.assertNotEqual(failed.returncode,0)
        self.assertIn('Configuration changed',failed.stdout)


if __name__=='__main__':
    unittest.main()
