#!/bin/bash
#SBATCH --job-name=morph_glb_conditions
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%A_%a.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%A_%a.err
set -euo pipefail
export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
source "$WORK/$USER/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
export TORCH_EXTENSIONS_DIR="$WORK/$USER/cache/torch_extensions"
export MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.0
RUN_DIR="${1:?Pass prepared run root}"
TASK_MODE="${2:-worker}"
PAIR_INDEX="${SLURM_ARRAY_TASK_ID:-${3:-0}}"
exec srun "$WORK/$USER/envs/interp3d/bin/python" "$MORPHFLOW_REPO/benchmarks/benchmark_morph_glb_conditioning.py" \
    "$TASK_MODE" --run-root "$RUN_DIR" --index "$PAIR_INDEX"
