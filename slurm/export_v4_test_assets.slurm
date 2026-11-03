#!/usr/bin/env bash
#SBATCH --job-name=v4_assets2048
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%A_%a.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%A_%a.err
set -euo pipefail
MORPH_BASE=/leonardo_work/IscrC_MORPHFL/mbarezzi
export WORK=/leonardo_work/IscrC_MORPHFL
export FAST=/leonardo_scratch/fast/IscrC_MORPHFL
source "$MORPH_BASE/env.sh"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.0
export TORCH_EXTENSIONS_DIR="$MORPH_BASE/cache/torch_extensions"
TASK_MODE="${EXPORT_MODE:-worker}"
if [[ "$TASK_MODE" == worker ]]; then
    ASSET_INDEX="${SLURM_ARRAY_TASK_ID:?Requires a Slurm array index}"
else
    ASSET_INDEX=0
fi
exec srun "$MORPH_BASE/envs/interp3d/bin/python" \
    "$MORPH_BASE/src/MorphFlow/benchmarks/export_v4_test_assets.py" "$TASK_MODE" \
    --index "$ASSET_INDEX" "$@"
