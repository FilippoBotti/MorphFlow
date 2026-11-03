#!/usr/bin/env bash
#SBATCH --job-name=morph_assets
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.err

# sbatch this-file src1.glb src2.glb --ss-checkpoint ss.pt --slat-checkpoint slat.pt \
#   --alphas 0 .25 .5 .75 1 --output-dir /path/run [--blender /path/blender]
# bash this-file --doctor [--gpu]   # --gpu only inside a GPU allocation
# bash this-file src1 ... --mode check
set -euo pipefail
MORPH_BASE="${MORPH_BASE:-/leonardo_work/IscrC_MORPHFL/mbarezzi}"
export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
source "${MF_ENV_FILE:-$MORPH_BASE/env.sh}"
MORPH_PROJECT="${MF_PROJECT_DIR:-$MORPH_BASE/src/MorphFlow}"
MORPH_PYTHON="${MORPH_PYTHON:-$MORPH_BASE/envs/interp3d/bin/python}"
export TRELLIS_REPO="${MF_TRELLIS_REPO:-$MORPH_BASE/src/TRELLIS}"
export BLENDER_BIN="${BLENDER_BIN:-$MORPH_BASE/tools/blender-3.0.1-linux-x64/blender}"
export PYTHONPATH="$MORPH_PROJECT:$TRELLIS_REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ATTN_BACKEND="${ATTN_BACKEND:-xformers}" SPCONV_ALGO="${SPCONV_ALGO:-native}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$MORPH_BASE/cache/torch_extensions}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" MAX_JOBS=1
mkdir -p "$TORCH_EXTENSIONS_DIR"

MORPH_DOCTOR=0
if [[ "${1:-}" == --doctor ]]; then
    MORPH_DOCTOR=1
    shift
    command=("$MORPH_PYTHON" "$MORPH_PROJECT/monster-exp/check_environment.py" "$@")
else
    command=("$MORPH_PYTHON" "$MORPH_PROJECT/monster-exp/morph_assets.py" "$@")
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    exec srun --ntasks=1 "${command[@]}"
fi
if [[ "$MORPH_DOCTOR" == 1 ]]; then
    exec "${command[@]}"
fi
case " $* " in
    *" --mode check "*|*" --doctor "*|*" --help "*) exec "${command[@]}" ;;
    *) echo "Submit generation with sbatch; use --mode check or --doctor on the login node." >&2; exit 1 ;;
esac
