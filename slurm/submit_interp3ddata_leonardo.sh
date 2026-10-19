#!/bin/bash
# Explicitly execute this helper to check or submit the Interp3DData benchmark.
set -euo pipefail
MODE="${1:---submit}"
case "$MODE" in --check|--submit) ;; *) echo "Usage: $0 [--check|--submit]" >&2; exit 1 ;; esac
[[ $# -le 1 ]] || { echo "Configure using exported environment variables." >&2; exit 1; }
export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
BASE_DIR="${BASE_DIR:-$WORK/$USER}"
PROJECT_DIR="${PROJECT_DIR:-$BASE_DIR/src/MorphFlow}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$BASE_DIR/checkpoints/ateneo_v4_ss17/ss_best_e17.pt}"
export SLAT_CHECKPOINT_PATH="${SLAT_CHECKPOINT_PATH:-$BASE_DIR/checkpoints/v4_interp3d_slat_dual_tgate_p25g4_lr5e-5_bs1_leonardo/morphflow_best.pt}"
export N="${N:-0}" SEED="${SEED:-42}" CFG_VALUES="${CFG_VALUES:-1 2 3 5 7.5}" SLAT_CFG_VALUES="${SLAT_CFG_VALUES:-1 1.5 2}"
export WEIGHTED_DENSE="${WEIGHTED_DENSE:-1}" EXPORT_GLB="${EXPORT_GLB:-0}"
export RUN_NAME="${RUN_NAME:-interp3ddata_ss17_slatbest_cfg15_n${N}_seed${SEED}_raw}"
export RUN_ROOT="${RUN_ROOT:-$FAST/$USER/outputs/threeway_benchmark/$RUN_NAME}"
export FINAL_ARCHIVE="${FINAL_ARCHIVE:-$BASE_DIR/outputs/threeway_benchmark/$RUN_NAME}"
mkdir -p "$BASE_DIR/logs"
bash "$PROJECT_DIR/slurm/benchmark_interp3ddata_leonardo.slurm" --check
[[ "$MODE" == --submit ]] || exit 0
JOB_ID="$(sbatch --parsable --export=ALL "$PROJECT_DIR/slurm/benchmark_interp3ddata_leonardo.slurm")"
echo "Submitted $JOB_ID; run: $RUN_ROOT"
echo "Mail BEGIN/END/FAIL: marco.barezzi@unipr.it"
echo "tail -n 80 -F $BASE_DIR/logs/interp3ddata_threeway_${JOB_ID%%;*}.{out,err}"
