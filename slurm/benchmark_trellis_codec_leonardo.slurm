#!/bin/bash
#SBATCH --job-name=trellis_codec_v4
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
export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
source "$WORK/$USER/env.sh"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export TORCH_EXTENSIONS_DIR="$WORK/$USER/cache/torch_extensions"
export BLENDER_BIN="$WORK/$USER/tools/blender-3.0.1-linux-x64/blender"
OUTPUT_DIR="${1:?Pass the prepared benchmark output directory}"
ASSET_INDEX="${SLURM_ARRAY_TASK_ID:-${2:-0}}"
TASK_MODE="${3:-worker}"
# The baseline environment includes the official Gaussian and mesh rasterizers.
# trellis_asset_codec itself pins Python imports to the upstream TRELLIS checkout.
exec srun "$WORK/$USER/envs/interp3d/bin/python" "$MORPHFLOW_REPO/benchmarks/benchmark_trellis_codec.py" "$TASK_MODE" \
    --output "$OUTPUT_DIR" --index "$ASSET_INDEX" --blender "$BLENDER_BIN"
