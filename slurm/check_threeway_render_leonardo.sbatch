#!/bin/bash
#SBATCH --job-name=threeway_render_check
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=BEGIN,END,FAIL

# Verify the complete 64-view/1000-view metric path on an existing raw pair.
set -euo pipefail
export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
BASE="$WORK/$USER"
source "$BASE/env.sh"
export PYTHONPATH="$BASE/src/MorphAny3D:$MORPHFLOW_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON="$BASE/envs/morphany3d/bin/python"
RAW_RUN="${RAW_RUN:-$BASE/outputs/threeway_benchmark/threeway_smoke_20261017/cfg_ss_2_slat_1.5}"
CHECK_ROOT="${CHECK_ROOT:-$BASE/outputs/threeway_benchmark/render_check_20261017}"
mkdir -p "$CHECK_ROOT/runtime-tmp"
export TMPDIR="$CHECK_ROOT/runtime-tmp"
cd "$MORPHFLOW_REPO"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'
"$PYTHON" benchmarks/eval_textured_benchmark.py shard \
    --run-dir "$RAW_RUN/morphflow" --output-dir "$CHECK_ROOT/metrics/morphflow" \
    --pair-glob 'pair_0000_*' --stage all --num-views 64 --fid-num-views 1000 \
    --resolution 512 --camera-distribution hammersley --render-backend gaussian \
    --appearance materials --lpips-net vgg --lpips-batch-size 8 --fid-batch-size 16 \
    --fid-selection-seed 42 --device cuda --reference-cache-root "$CHECK_ROOT/reference_metrics"
"$PYTHON" benchmarks/eval_textured_benchmark.py merge \
    --output-dir "$CHECK_ROOT/metrics/morphflow" --expected-num-pairs 1 --fid-min-samples 512
echo "RENDER CHECK PASSED: $CHECK_ROOT/metrics/morphflow/metrics_batch.json"
