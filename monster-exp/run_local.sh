#!/usr/bin/env bash
# Run inside the installed environment: conda activate trellis
# bash monster-exp/run_local.sh src1.glb src2.glb --alphas 0 .5 1 ...
set -euo pipefail
MONSTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$MONSTER_DIR")"
MORPH_PYTHON="${MORPH_PYTHON:-python}"

# Respect an explicitly configured environment. These defaults match monster.
if [[ -z "${TRELLIS_REPO:-}" && -f /home/filippo/projects/TRELLIS/trellis/__init__.py ]]; then
    export TRELLIS_REPO=/home/filippo/projects/TRELLIS
fi
if [[ -z "${BLENDER_BIN:-}" && -x /home/filippo/projects/morph_old/.cache/blender/blender-3.0.1-linux-x64/blender ]]; then
    export BLENDER_BIN=/home/filippo/projects/morph_old/.cache/blender/blender-3.0.1-linux-x64/blender
fi
if [[ -z "${CUDA_HOME:-}" && -x /usr/local/cuda-12.4/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda-12.4
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
fi
export PYTHONPATH="$PROJECT_DIR${TRELLIS_REPO:+:$TRELLIS_REPO}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export ATTN_BACKEND="${ATTN_BACKEND:-xformers}"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-2}"
# Each run uses one visible GPU. Launch a second run with CUDA_VISIBLE_DEVICES=1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "${1:-}" == --doctor ]]; then
    shift
    exec "$MORPH_PYTHON" "$MONSTER_DIR/check_environment.py" "$@"
fi
exec "$MORPH_PYTHON" "$MONSTER_DIR/morph_assets.py" "$@"
