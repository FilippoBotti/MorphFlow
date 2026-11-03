#!/usr/bin/env bash
#SBATCH --job-name=glb_test50
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%A_%a.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%A_%a.err

set -euo pipefail
MORPH_BASE=/leonardo_work/IscrC_MORPHFL/mbarezzi
export WORK=/leonardo_work/IscrC_MORPHFL
export FAST=/leonardo_scratch/fast/IscrC_MORPHFL
source "$MORPH_BASE/env.sh"
export PYTHONPATH="$MORPH_BASE/src/MorphFlow:$MORPH_BASE/src/TRELLIS:$MORPH_BASE/src/MorphFlow/benchmarks"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.0
export TORCH_EXTENSIONS_DIR="$MORPH_BASE/cache/torch_extensions"
exec srun "$MORPH_BASE/envs/interp3d/bin/python" \
  "$MORPH_BASE/src/MorphFlow/benchmarks/test_portable_glb_test50.py" \
  "${1:?Use materialize or worker}" --index "${SLURM_ARRAY_TASK_ID:-0}" "${@:2}"
