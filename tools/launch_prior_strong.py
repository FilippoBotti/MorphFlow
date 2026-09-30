#!/usr/bin/env python3
"""Submit the new phase from the HOST login shell (no torch needed).

python3 tools/launch_prior_strong.py --dry-run
python3 tools/launch_prior_strong.py --submit
"""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess


DEFAULT_RUN = 'v3_ss_projectionStrong_ep14_r05_c015_g4'
ARCHIVE = '/hpc/archive/G_VBD/marco.barezzi/morphflow_runs'
DEFAULT_CKPT = (ARCHIVE + '/v3_ss_trellisProjection_t005_020_test2/checkpoints/'
                'morphflow_epoch_0014_step_0070000.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=DEFAULT_CKPT)
    parser.add_argument('--run-name', default=DEFAULT_RUN)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit('Checkpoint missing: ' + str(checkpoint))
    if '/' in args.run_name or args.run_name in ('', '.', '..'):
        raise SystemExit('run-name must be a directory name, not a path')
    output = Path(ARCHIVE) / args.run_name
    if (output / 'checkpoints').exists() and any((output / 'checkpoints').glob('*.pt')):
        raise SystemExit('Output contains checkpoints. Use a new run-name; do not overwrite the experiment.')
    launcher = root / 'slurm/train_morphflow_v3_prior_strong.slurm'
    if not launcher.is_file():
        raise SystemExit('Apply the patch before submitting')
    settings = dict(
        RUN_NAME=args.run_name, AUTO_RESUME='0', RESUME_FROM=str(checkpoint),
        TRELLIS_PRIOR_WEIGHT='1.0', TRELLIS_PRIOR_EVERY='2',
        TRELLIS_PRIOR_ROLLOUT_STEPS='8', TRELLIS_PRIOR_GRAD_STEPS='4',
        TRELLIS_PRIOR_MAX_ITEMS='1', TRELLIS_PRIOR_CHECKPOINT='1',
        TRELLIS_PRIOR_T_MIN='0.05', TRELLIS_PRIOR_T_MAX='0.20',
        TRELLIS_PRIOR_PROJECTION_CLIP_RATIO='0.15',
        TRELLIS_PRIOR_RMS_GUARD_WEIGHT='1.0',
        TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO='0.25',
        TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO='2.0',
        TRELLIS_PRIOR_GRAD_BALANCE='1', TRELLIS_PRIOR_RATIO_TARGET='0.5',
        TRELLIS_PRIOR_RATIO_MAX='1.0', TRELLIS_PRIOR_BALANCE_EMA='0.95',
        TRELLIS_PRIOR_LAMBDA_MIN='0.01', TRELLIS_PRIOR_LAMBDA_MAX='100.0',
        TRELLIS_PRIOR_PHASE_START_WEIGHT='0.1',
        TRELLIS_PRIOR_PHASE_WARMUP_STEPS='500', TRELLIS_PRIOR_GUARD_SCALE='0.1',
        TRELLIS_PRIOR_RESET_PHASE='1', TRELLIS_PRIOR_LOG_EVERY='20',
        TRELLIS_PRIOR_WEAK_PATIENCE='25',
        TRELLIS_PRIOR_PHASE_EVAL_STEPS='500,1500,3000',
    )
    env = os.environ.copy()
    env.update(settings)
    command = ['sbatch', '--parsable', '--export=ALL', '--nodelist=wn49',
               '--gres=gpu:l40s_vbd:8', '--job-name=mf_ss_prior_strong', str(launcher)]
    print('Checkpoint:', checkpoint)
    print('Output:', output)
    print('Resources: wn49, 8 GPUs. Epochs: 15 through 25.')
    print(' '.join(shlex.quote(s) for s in command))
    for key in sorted(settings):
        print('%s=%s' % (key, settings[key]))
    if args.submit:
        if shutil.which('sbatch') is None:
            raise SystemExit('Run this launcher on the HOST, not inside Singularity.')
        result = subprocess.run(command, env=env, cwd=str(root), check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True)
        print('SUBMITTED JOB:', result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip())


if __name__ == '__main__':
    main()
