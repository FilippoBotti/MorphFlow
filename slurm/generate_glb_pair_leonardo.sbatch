#!/bin/bash
#SBATCH --job-name=morphflow_glb_pair_prior_cfggrid
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

# sbatch this-file [first.glb second.glb [Python flags...]]
# bash this-file --prepare|--check [first.glb second.glb [Python flags...]]
set -euo pipefail
MODE=run
case "${1:-}" in
    --prepare|--check) MODE="${1#--}"; shift ;;
esac
if [[ "$MODE" == run && -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Use sbatch for generation, or --check/--prepare on the login node." >&2
    exit 1
fi
export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
source "${MF_ENV_FILE:-$WORK/$USER/env.sh}"
PROJECT_DIR="${MF_PROJECT_DIR:-$MORPHFLOW_REPO}"
SRC1="${SRC1:-$WORK/$USER/assets/Werewolf_Warrior.glb}"
SRC2="${SRC2:-$WORK/$USER/assets/pool3913.glb}"
if [[ $# -gt 0 ]]; then
    [[ $# -ge 2 ]] || { echo "Supply both GLB paths." >&2; exit 1; }
    SRC1="$1"; SRC2="$2"; shift 2
fi
# Pin the latest available prior-trained checkpoints at submission time.
# The newly queued dual-prior training has not produced checkpoints yet.
SS_CHECKPOINT="${SS_CHECKPOINT:-$WORK/$USER/checkpoints/ss_tangentScaleAnchor_r030_s020_g025_ep14_last.pt}"
SLAT_CHECKPOINT="${SLAT_CHECKPOINT:-$WORK/$USER/outputs/v3_slat_trellisPrior_r015_stat015_bs1_fromBest_leonardo/checkpoints/morphflow_epoch_0005_step_0050000.pt}"
BLENDER_BIN="${BLENDER_BIN:-$WORK/$USER/tools/blender-3.0.1-linux-x64/blender}"
RUN_NAME="${RUN_NAME:-werewolf_pool_prior_cfggrid}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK/$USER/outputs/glb_pair/$RUN_NAME/job_${SLURM_JOB_ID:-check}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ "$MODE" == prepare ]]; then
    export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
fi
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$WORK/$USER/cache/torch_extensions}"
mkdir -p "$WORK/$USER/logs" "$TORCH_EXTENSIONS_DIR"
command=(python "$PROJECT_DIR/generate_glb_pair.py" "$SRC1" "$SRC2"
    --ss-checkpoint "$SS_CHECKPOINT" --slat-checkpoint "$SLAT_CHECKPOINT"
    --blender "$BLENDER_BIN" --output-dir "$OUTPUT_DIR" --mode "$MODE"
    --alphas "${ALPHAS:-0,0.5,1}"
    --cfg-ss "${CFG_SS:-2,2.5,3}" --cfg-slat "${CFG_SLAT:-1.5,2,2.5}"
    --steps "${STEPS:-50}" --slat-steps "${SLAT_STEPS:-50}"
    --encoding-views "${ENCODING_VIEWS:-50}" --dino-batch-size "${DINO_BATCH_SIZE:-4}"
    --seed "${SEED:-42}" --mixed-precision "${MIXED_PRECISION:-bf16}" "$@")
printf 'Command: '; printf '%q ' "${command[@]}"; printf '\n'
if [[ "$MODE" == run ]]; then
    mkdir -p "$OUTPUT_DIR"
    exec 9>"$OUTPUT_DIR/.run.lock"
    flock -n 9 || { echo "Output directory already in use: $OUTPUT_DIR" >&2; exit 1; }
    exec srun --ntasks=1 "${command[@]}"
else
    exec "${command[@]}"
fi
