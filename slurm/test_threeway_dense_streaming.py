import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name('threeway_dense_streaming.sh')


class DenseStreamingTests(unittest.TestCase):
    def run_stream(self, directory, retired_pair=False):
        root=Path(directory)
        shared=root/'weighted_50/shared'
        shared.mkdir(parents=True)
        plan=dict(pairs=[dict(pair_id=i,pair_dir='pair_%04d_a_to_b'%i) for i in range(2)])
        (shared/'plan.json').write_text(json.dumps(plan))
        cell=root/'weighted_50/cfg_ss_3_slat_1'
        cell.mkdir()
        (cell/'plan.json').symlink_to(shared/'plan.json')
        if retired_pair:
            marker=cell/'morphflow/pair_0000_a_to_b/raw_retired.json'
            marker.parent.mkdir(parents=True)
            marker.write_text('{}')
        (root/'grid.txt').write_text('3 1\n')
        shell='''
set -euo pipefail
RUN_ROOT="$TEST_ROOT"
SNAPSHOT="$RUN_ROOT/snapshot"
ORIGINAL_ENV=/env/morphflow
MORPHANY_ENV=/env/morphany
INTERP_ENV=/env/interp
SCRIPT=/repair/generation.py
EVALUATOR=/repair/evaluator.py
MORPHFLOW_DENSE_RETENTION_SCRIPT=/repair/retention.py
HELPER=/helper.py
ACTIVE_PROJECT=/frozen/MorphFlow
DATA_ROOT_HOST=/data
SOURCE_IMAGES_ROOT_HOST=/images
SOURCE_IMAGE_FILENAME=""
STEPS=25 SLAT_STEPS=25 MIXED_PRECISION=bf16
MORPHANY_TFSA_RAM_GB=12 MORPHANY_TFSA_MIN_FREE_GB=20
TASK_TMP="$RUN_ROOT/tmp"
ACTUAL_PAIRS=2 NUM_VIEWS=64 LPIPS_BATCH_SIZE=4
GRID_FILE="$RUN_ROOT/grid.txt"
run_python() {
 shift 2
 if [[ "$1" == /helper.py ]]; then
  if [[ "$2" == cell ]]; then echo "$RUN_ROOT/weighted_50/cfg_ss_3_slat_1"; fi
 else
  python3 "$@"
 fi
}
run_phase() { printf '%s\\n' "$*" >> "$RUN_ROOT/trace.txt"; }
die() { echo "$*" >&2; return 1; }
source "$STREAM_SCRIPT"
run_dense_streaming_grid
'''
        env=dict(os.environ,TEST_ROOT=str(root),STREAM_SCRIPT=str(SCRIPT))
        result=subprocess.run(['bash','-c',shell],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        self.assertEqual(result.returncode,0,result.stderr)
        return (root/'trace.txt').read_text().splitlines()

    def test_each_pair_is_scored_and_released_before_generating_the_next(self):
        with tempfile.TemporaryDirectory() as directory:
            trace=self.run_stream(directory)
            phases=[r.split()[0] for r in trace]
            for prefix in ['weighted50_morphany3d','weighted50_interp3d','weighted50_cfg_ss_3_slat_1']:
                for i in range(2):
                    g=phases.index(prefix+'_pair_%d_generation'%i)
                    self.assertEqual(phases[g+1],prefix+'_pair_%d_metrics'%i)
                    self.assertEqual(phases[g+2],prefix+'_pair_%d_retire_raw'%i)
                    self.assertIn('--shard-index %d --num-shards 2'%i,trace[g])
                    self.assertIn('--num-views 64 --resolution 512',trace[g+1])
                    self.assertIn('--metric-mode weighted-endpoints-only',trace[g+1])
                self.assertLess(phases.index(prefix+'_pair_1_retire_raw'),phases.index(prefix+'_merge'))
            self.assertIn('/env/morphflow',next(r for r in trace if r.startswith('weighted50_cfg_ss_3_slat_1_pair_0_generation ')))
            self.assertIn('/env/morphany',next(r for r in trace if r.startswith('weighted50_morphany3d_pair_0_generation ')))
            self.assertIn('/env/interp',next(r for r in trace if r.startswith('weighted50_interp3d_pair_0_generation ')))

    def test_retired_pair_is_verified_and_skipped_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            trace=self.run_stream(directory,retired_pair=True)
            phases=[r.split()[0] for r in trace]
            prefix='weighted50_cfg_ss_3_slat_1_pair_0'
            self.assertIn(prefix+'_verify_retained',phases)
            self.assertNotIn(prefix+'_generation',phases)
            self.assertNotIn(prefix+'_metrics',phases)


if __name__ == '__main__':
    unittest.main()
