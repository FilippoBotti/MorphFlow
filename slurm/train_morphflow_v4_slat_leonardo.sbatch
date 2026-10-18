#!/bin/bash
#SBATCH --job-name=v4_slat_pr25g4_lr5e-5_bs1
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.err

# Port of train_morphflow_v4_ss.slurm: SLat, all learning rates 5e-5,
# two Leonardo Booster nodes x four GPUs, one sample per GPU (global batch 8).
# The shared launcher supplies the native environment, offline cache checks,
# output locks and one Accelerate launcher per node.
set -euo pipefail

export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
export MF_PROJECT_DIR="${MF_PROJECT_DIR:-$WORK/$USER/src/MorphFlow}"
export MF_DATA_ROOT="${MF_DATA_ROOT:-$FAST/$USER/datasets/morphing_dataset_v4}"
export FLOW_TARGET=slat
export RUN_NAME="${RUN_NAME:-v4_interp3d_slat_dual_tgate_p25g4_lr5e-5_bs1_leonardo}"
export TRAIN_BS="${TRAIN_BS:-1}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export WARMUP_STEPS="${WARMUP_STEPS:-10000}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-5e-5}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2}"
export CHECKPOINT_HALF_EPOCH="${CHECKPOINT_HALF_EPOCH:-0}"

# Match the v4 ateneo prior, including its warmup and RMS guard bounds.
export TRELLIS_PRIOR_WEIGHT="${TRELLIS_PRIOR_WEIGHT:-0.1}"
export TRELLIS_PRIOR_EVERY="${TRELLIS_PRIOR_EVERY:-4}"
export TRELLIS_PRIOR_WARMUP_STEPS="${TRELLIS_PRIOR_WARMUP_STEPS:-10000}"
export TRELLIS_PRIOR_ROLLOUT_STEPS="${TRELLIS_PRIOR_ROLLOUT_STEPS:-25}"
export TRELLIS_PRIOR_GRAD_STEPS="${TRELLIS_PRIOR_GRAD_STEPS:-4}"
export TRELLIS_PRIOR_MAX_ITEMS="${TRELLIS_PRIOR_MAX_ITEMS:-1}"
export TRELLIS_PRIOR_CHECKPOINT="${TRELLIS_PRIOR_CHECKPOINT:-1}"
export TRELLIS_PRIOR_T_MIN="${TRELLIS_PRIOR_T_MIN:-0.05}"
export TRELLIS_PRIOR_T_MAX="${TRELLIS_PRIOR_T_MAX:-0.20}"
export TRELLIS_PRIOR_PROJECTION_CLIP_RATIO="${TRELLIS_PRIOR_PROJECTION_CLIP_RATIO:-0.10}"
export TRELLIS_PRIOR_RMS_GUARD_WEIGHT="${TRELLIS_PRIOR_RMS_GUARD_WEIGHT:-1.0}"
export TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO="${TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO:-0.25}"
export TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO="${TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO:-2.0}"
export TRELLIS_PRIOR_GRAD_BALANCE="${TRELLIS_PRIOR_GRAD_BALANCE:-0}"

# Use original TRELLIS weights for a new run. Resume is explicit, as in the
# other Leonardo launchers: AUTO_RESUME=1 or RESUME_FROM=/absolute/path.pt.
export AUTO_RESUME="${AUTO_RESUME:-0}"

case "${1:---train}" in
    --train|--check|--prepare)
        python3 - "$MF_DATA_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
state = json.loads((root / "run_state.json").read_text())
if state.get("phase") != "complete" or state.get("status") != "complete":
    raise SystemExit(f"Dataset v4 non completo: {state}")
for split, count in (("train", 50000), ("val", 2000), ("test", 2000)):
    entries = json.loads((root / f"metadata_{split}.json").read_text())
    if len(entries) != count:
        raise SystemExit(f"Dataset v4 {split}: {len(entries)} != {count}")
print("Dataset v4 completo: train=50000 val=2000 test=2000", flush=True)
PY
        ;;
esac

exec bash "$MF_PROJECT_DIR/slurm/train_morphflow_v3_leonardo.sbatch" "$@"
