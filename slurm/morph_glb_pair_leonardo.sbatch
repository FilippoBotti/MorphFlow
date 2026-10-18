#!/bin/bash
#SBATCH --job-name=morph_glb_pair
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
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=BEGIN,END,FAIL

# sbatch this-file src1.glb src2.glb --ss-checkpoint ss.pt --slat-checkpoint slat.pt \
#     --alpha 0.5 --output-dir /path/to/output [other morph_glb_pair.py options]
# bash this-file --check|--prepare src1.glb src2.glb [same options]
set -euo pipefail
MODE=run
case "${1:-}" in
    --check|--prepare) MODE="${1#--}"; shift ;;
    --help|-h) MODE=check ;;
esac
if [[ "$MODE" == run && -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Usare sbatch per la generazione, oppure --check/--prepare dal login node." >&2
    exit 1
fi
export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
source "${MF_ENV_FILE:-$WORK/$USER/env.sh}"
PROJECT_DIR="${PROJECT_DIR:-${MF_PROJECT_DIR:-$MORPHFLOW_REPO}}"
BLENDER_BIN="${BLENDER_BIN:-$WORK/$USER/tools/blender-3.0.1-linux-x64/blender}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$WORK/$USER/cache/torch_extensions}"
mkdir -p "$WORK/$USER/logs" "$TORCH_EXTENSIONS_DIR"

# Keep input/output paths relative to the submission directory, when relative.
command=(python "$PROJECT_DIR/morph_glb_pair.py" --blender "$BLENDER_BIN" "$@" --mode "$MODE")
printf 'Command: '; printf '%q ' "${command[@]}"; printf '\n'
if [[ "$MODE" == run ]]; then
    exec srun --ntasks=1 "${command[@]}"
else
    exec "${command[@]}"
fi
