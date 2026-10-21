"""Execute the actual SLat argument block without a container or model imports."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name('train_morphflow_v4_slat.slurm')


class SlatAblationTests(unittest.TestCase):
    def command(self, overrides, prior_guard=False):
        text = SCRIPT.read_text()
        settings = text[text.index('RUN_NAME="'):text.index('RUN_DIR_HOST=')]
        start = 'if python - "$TRELLIS_PRIOR_WEIGHT"' if prior_guard else 'budget_args=()'
        command = text[text.index(start):text.rindex('\nIN_CONTAINER')]
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory)/'capture.py'
            capture.write_text('import json,sys\nprint("CAPTURE="+json.dumps(sys.argv[1:]))\n')
            env = os.environ.copy()
            for key in ('CONDITION_GATE','SOURCE_SWAP','TRAIN_BS','TRAIN_EPOCHS',
                        'EXPECTED_TRAIN_STEPS','TRELLIS_PRIOR_WEIGHT','SEED','RUN_NAME'):
                env.pop(key,None)
            env.update(overrides)
            prefix = ('set -euo pipefail\n'
                      'DATA_ROOT_IN=/data/v4\nOUT_ROOT_IN=/archive/runs\n'
                      'FLUX_ROOT_IN=/data/flux\nRESUME_ARG=""\n'
                      'python() { command python3 "$@"; }\n'
                      'accelerate() { command python3 '+shlex.quote(str(capture))+' "$@"; }\n')
            result = subprocess.run(['bash','-c',prefix+settings+command],env=env,
                                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                    universal_newlines=True)
        if result.returncode:
            return result,None
        row=next(line for line in result.stdout.splitlines() if line.startswith('CAPTURE='))
        return result,json.loads(row[len('CAPTURE='):])

    def test_baseline_8_l40s_250k_updates_and_no_prior_import(self):
        result,args=self.command(dict(CONDITION_GATE='alpha',SOURCE_SWAP='0',
                                      TRELLIS_PRIOR_WEIGHT='0.0',TRAIN_EPOCHS='40',
                                      EXPECTED_TRAIN_STEPS='250000'),prior_guard=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for flag,value in [('--num_processes','8'),('--flow_target','slat'),
                           ('--separate_cond','1'),('--separate_cond_gate','alpha'),
                           ('--train_flow_alpha_gate','0'),('--augment_source_swap','0'),
                           ('--trellis_prior_weight','0.0'),('--train_epochs','40'),
                           ('--expected_train_samples','50000'),('--expected_train_steps','250000'),
                           ('--train_bs','1'),('--lr','5e-5'),('--slat_condition_source','slat')]:
            self.assertEqual(args[args.index(flag)+1],value)
        self.assertNotIn('--resume_from',args)

    def test_full_slat_defaults_preserved(self):
        result,args=self.command({})
        self.assertEqual(result.returncode,0,result.stderr)
        for flag,value in [('--separate_cond_gate','token'),('--train_flow_alpha_gate','1'),
                           ('--augment_source_swap','1'),('--trellis_prior_weight','0.1'),
                           ('--train_epochs','20'),('--expected_train_samples','100000')]:
            self.assertEqual(args[args.index(flag)+1],value)
        self.assertNotIn('--expected_train_steps',args)

    def test_wrong_epoch_budget_rejected(self):
        result,args=self.command(dict(SOURCE_SWAP='0',TRAIN_EPOCHS='20',EXPECTED_TRAIN_STEPS='250000'))
        self.assertNotEqual(result.returncode,0)
        self.assertIsNone(args)
        self.assertIn('Epochs/batch do not match',result.stderr)


if __name__=='__main__':unittest.main()
