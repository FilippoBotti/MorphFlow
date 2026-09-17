#!/bin/bash
#SBATCH --job-name=morph_comparison
#SBATCH --account=g_vbd
#SBATCH --partition=gpu_vbd
#SBATCH --qos=gpu_vbd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s_vbd:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --open-mode=append

# Ateneo: same Singularity image/environment as generate_morphany3d_v3.slurm.
# Prefer submit_teacher_student_comparison.sh: it configures Slurm's own logs
# inside OUTPUT_DIR before submission, including scheduler/startup errors.
# Direct sbatch also redirects the running script's logs into OUTPUT_DIR.
# Set CHECKPOINT_PATH and SLAT_CHECKPOINT_PATH before submission. Other settings
# below can be overridden via exported variables. Extra CLI arguments are passed
# verbatim to Python (e.g. --dry-run, --stage teacher, --resume).
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/trellis-singularity/work/MorphFlow}"
OUTPUT_DIR="${OUTPUT_DIR:-/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/comparison/job_${SLURM_JOB_ID:-local}}"
# Honor the Python override before opening logs, including paths with spaces.
CLI_ARGS=("$@")
for ((arg_index=0; arg_index<${#CLI_ARGS[@]}; arg_index++)); do
    case "${CLI_ARGS[$arg_index]}" in
        --output-dir)
            arg_index=$((arg_index + 1))
            OUTPUT_DIR="${CLI_ARGS[$arg_index]:?--output-dir needs a value}"
            ;;
        --output-dir=*) OUTPUT_DIR="${CLI_ARGS[$arg_index]#--output-dir=}" ;;
    esac
done
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
exec >>"$OUTPUT_DIR/slurm-comparison-${SLURM_JOB_ID:-local}.out" \
     2>>"$OUTPUT_DIR/slurm-comparison-${SLURM_JOB_ID:-local}.err"
printf 'Job %s on %s; output=%s\n' "${SLURM_JOB_ID:-local}" "${SLURMD_NODENAME:-unknown}" "$OUTPUT_DIR"

module purge
module load "${SINGULARITY_MODULE:-singularity/3.8.7}"

MORPHANY3D_DIR="${MORPHANY3D_DIR:-$HOME/trellis-singularity/work/MorphAny3D}"
TRELLIS_BASE="${TRELLIS_BASE:-$HOME/trellis-singularity}"
SIF="${SIF:-$HOME/containers/trellis-py240-cu118.sif}"
ASSETS_DIR="${ASSETS_DIR:-/hpc/scratch/marco.barezzi/3d_dataset/flux_outputs}"
DATASET_DIR="${DATASET_DIR:-/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v3}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the trained SS checkpoint}"
SLAT_CHECKPOINT_PATH="${SLAT_CHECKPOINT_PATH:?Set SLAT_CHECKPOINT_PATH to the trained SLat checkpoint}"
NUM_PAIRS="${NUM_PAIRS:-10}"
NUM_INTERMEDIATES="${NUM_INTERMEDIATES:-5}"
SEED="${SEED:-42}"
RESUME="${RESUME:-0}"
STAGE="${STAGE:-all}"
MODEL_ID="${MODEL_ID:-microsoft/TRELLIS-image-large}"
STEPS="${STEPS:-50}"
SLAT_STEPS="${SLAT_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-3.0}"
SLAT_CFG_SCALE="${SLAT_CFG_SCALE:-3.0}"
TEACHER_SS_STEPS="${TEACHER_SS_STEPS:-25}"
TEACHER_SLAT_STEPS="${TEACHER_SLAT_STEPS:-25}"
TEACHER_SS_CFG="${TEACHER_SS_CFG:-7.5}"
TEACHER_SLAT_CFG="${TEACHER_SLAT_CFG:-3.0}"
TFSA_ALPHA="${TFSA_ALPHA:-0.8}"
TFSA_CACHE_MODE="${TFSA_CACHE_MODE:-memory}"
MAX_WORK_CACHE_GB="${MAX_WORK_CACHE_GB:-60}"
MIXED_PRECISION="${MIXED_PRECISION:-auto}"
# Set WORK_CACHE_ROOT=/nvme/$USER on nodes that offer local NVMe.
WORK_CACHE_ROOT="${WORK_CACHE_ROOT:-${SLURM_TMPDIR:-/tmp}}"

for directory in "$PROJECT_DIR" "$MORPHANY3D_DIR/trellis" "$TRELLIS_BASE/env" "$ASSETS_DIR" "$DATASET_DIR"; do
    if [ ! -d "$directory" ]; then
        echo "ERROR: directory not found: $directory" >&2
        exit 1
    fi
done
for file in "$SIF" "$CHECKPOINT_PATH" "$SLAT_CHECKPOINT_PATH" "$PROJECT_DIR/generate_teacher_student_comparison.py"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: file not found: $file" >&2
        exit 1
    fi
done
if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
    echo "ERROR: RESUME must be 0 or 1" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$WORK_CACHE_ROOT" "$TRELLIS_BASE/hf-home"
PROJECT_DIR="$(realpath "$PROJECT_DIR")"
MORPHANY3D_DIR="$(realpath "$MORPHANY3D_DIR")"
TRELLIS_BASE="$(realpath "$TRELLIS_BASE")"
SIF="$(realpath "$SIF")"
ASSETS_DIR="$(realpath "$ASSETS_DIR")"
DATASET_DIR="$(realpath "$DATASET_DIR")"
CHECKPOINT_PATH="$(realpath "$CHECKPOINT_PATH")"
SLAT_CHECKPOINT_PATH="$(realpath "$SLAT_CHECKPOINT_PATH")"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
CACHE_DIR="$(mktemp -d "$WORK_CACHE_ROOT/morph_comparison_${SLURM_JOB_ID:-local}_XXXXXX")"
CACHE_DIR="$(realpath "$CACHE_DIR")"
cleanup() { rm -rf -- "$CACHE_DIR"; }
trap cleanup EXIT

# Keep host paths inside the container: custom checkpoint/output paths work
# without archive-specific prefix substitution. Array arguments preserve spaces.
BIND_ARGS=(--bind "$HOME:$HOME" --bind "$TRELLIS_BASE:/opt/trellis")
declare -A SEEN_BINDS=()
add_bind() {
    local path="$1" mode="$2" key="$1:$2"
    if [[ -z "${SEEN_BINDS[$key]:-}" ]]; then
        BIND_ARGS+=(--bind "$path:$path:$mode")
        SEEN_BINDS[$key]=1
    fi
}
add_bind "$PROJECT_DIR" ro
add_bind "$MORPHANY3D_DIR" ro
add_bind "$ASSETS_DIR" ro
add_bind "$DATASET_DIR" ro
add_bind "$(dirname "$CHECKPOINT_PATH")" ro
add_bind "$(dirname "$SLAT_CHECKPOINT_PATH")" ro
if [ -d "$MODEL_ID" ]; then
    MODEL_ID="$(realpath "$MODEL_ID")"
    add_bind "$MODEL_ID" ro
fi
add_bind "$OUTPUT_DIR" rw
add_bind "$CACHE_DIR" rw

export SINGULARITYENV_PROJECT_DIR="$PROJECT_DIR"
export SINGULARITYENV_MORPHANY3D_DIR="$MORPHANY3D_DIR"
export SINGULARITYENV_OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
# Respect Slurm's assigned GPU visibility; never select a physical GPU manually.
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

ARGS=(
    --assets-dir "$ASSETS_DIR" --dataset-dir "$DATASET_DIR" --output-dir "$OUTPUT_DIR"
    --checkpoint-path "$CHECKPOINT_PATH" --slat-checkpoint-path "$SLAT_CHECKPOINT_PATH"
    --num-pairs "$NUM_PAIRS" --num-intermediates "$NUM_INTERMEDIATES" --seed "$SEED"
    --model-id "$MODEL_ID" --steps "$STEPS" --slat-steps "$SLAT_STEPS"
    --cfg-scale "$CFG_SCALE" --slat-cfg-scale "$SLAT_CFG_SCALE"
    --teacher-ss-steps "$TEACHER_SS_STEPS" --teacher-slat-steps "$TEACHER_SLAT_STEPS"
    --teacher-ss-cfg "$TEACHER_SS_CFG" --teacher-slat-cfg "$TEACHER_SLAT_CFG"
    --tfsa-alpha "$TFSA_ALPHA" --tfsa-cache-mode "$TFSA_CACHE_MODE"
    --max-work-cache-gb "$MAX_WORK_CACHE_GB" --mixed-precision "$MIXED_PRECISION"
    --work-cache-dir "$CACHE_DIR" --stage "$STAGE"
)
if [ "$RESUME" = 1 ]; then ARGS+=(--resume); fi
ARGS+=("$@")
printf 'Comparison: %s pairs, %s intermediates; output=%s\n' "$NUM_PAIRS" "$NUM_INTERMEDIATES" "$OUTPUT_DIR"

singularity exec --cleanenv --nv "${BIND_ARGS[@]}" "$SIF" bash -s -- "${ARGS[@]}" <<'IN_CONTAINER'
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/trellis/env
export HF_HOME=/opt/trellis/hf-home
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export PYTHONUNBUFFERED=1
export TRELLIS_REPO="$MORPHANY3D_DIR"
export PYTHONPATH="$MORPHANY3D_DIR:$PROJECT_DIR:${PYTHONPATH:-}"
cd "$PROJECT_DIR"
exec python generate_teacher_student_comparison.py "$@"
IN_CONTAINER
