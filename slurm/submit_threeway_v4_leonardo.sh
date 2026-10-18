#!/bin/bash
# Submit the verified v4 checkpoint comparison, using one Leonardo GPU.
set -euo pipefail
MODE="${1:---submit}"
case "$MODE" in --check|--submit) ;; *) echo "Usage: $0 [--check|--submit]" >&2; exit 1 ;; esac
export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
BASE_DIR="${BASE_DIR:-$WORK/$USER}"
PROJECT_DIR="${PROJECT_DIR:-$BASE_DIR/src/MorphFlow}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$BASE_DIR/checkpoints/threeway_v4_20261017/ss_best_epoch_13.pt}"
export SLAT_CHECKPOINT_PATH="${SLAT_CHECKPOINT_PATH:-$BASE_DIR/checkpoints/threeway_v4_20261017/slat_best_epoch_6.pt}"
export CFG_VALUES="${CFG_VALUES:-2 3 5}" SLAT_CFG_VALUES="${SLAT_CFG_VALUES:-1.5 2 2.5}"
export N="${N:-30}" K="${K:-5}" SEED="${SEED:-42}"
export RUN_ROOT="${RUN_ROOT:-$BASE_DIR/outputs/threeway_benchmark/threeway_v4_ss13_slat6_n${N}_beta5_cfg235_slat15225_20261017}"
mkdir -p "$BASE_DIR/logs"
bash "$PROJECT_DIR/slurm/benchmark_threeway_textured_leonardo.slurm" --check
[[ "$MODE" == --submit ]] || exit 0
JOB_ID="$(sbatch --parsable --export=ALL "$PROJECT_DIR/slurm/benchmark_threeway_textured_leonardo.slurm")"
export THREEWAY_SUBMITTED_JOB_ID="$JOB_ID"
source "$BASE_DIR/env.sh"
python - <<'PY'
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

run_root = Path(os.environ['RUN_ROOT'])
run_root.mkdir(parents=True, exist_ok=True)
payload = {
    'job_id': os.environ['THREEWAY_SUBMITTED_JOB_ID'],
    'submitted_at': datetime.now(ZoneInfo('Europe/Rome')).isoformat(),
    'settings': {name: os.environ[name] for name in (
        'CHECKPOINT_PATH', 'SLAT_CHECKPOINT_PATH', 'CFG_VALUES', 'SLAT_CFG_VALUES', 'N', 'K', 'SEED', 'RUN_ROOT')},
    'nodes': 1, 'gpus': 1, 'mail_user': 'marco.barezzi@unipr.it', 'mail_type': ['BEGIN', 'END', 'FAIL'],
}
(run_root / f"submission_{payload['job_id']}.json").write_text(json.dumps(payload, indent=2) + '\n')
print(f"Submitted {payload['job_id']}: {run_root}")
PY
