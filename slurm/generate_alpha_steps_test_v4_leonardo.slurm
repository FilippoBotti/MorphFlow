#!/bin/bash
#SBATCH --job-name=alpha_steps_test_v4
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=BEGIN,END,FAIL

# Stessi parametri di generate_alpha_steps_test.slurm, con dataset v4 e env nativo.
# CHECKPOINT_PATH=/path/ss.pt SLAT_CHECKPOINT_PATH=/path/slat.pt \
# NUM_PAIRS=10 NUM_INTERMEDIATE_STEPS=5 sbatch slurm/generate_alpha_steps_test_v4_leonardo.slurm
# Dal login: bash questo-script --check (offline) oppure --prepare (cache decoder).
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }

MODE="${1:---generate}"
case "$MODE" in
    --generate|--check|--prepare) ;;
    *) die "Uso: $0 [--generate|--check|--prepare]" ;;
esac
[[ $# -le 1 ]] || die "Usare le variabili ambiente per modificare i parametri."
if [[ "$MODE" == --generate ]]; then
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "Lanciare con sbatch; --check e --prepare funzionano dal login node."
fi

export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
MF_ENV_FILE="${MF_ENV_FILE:-$WORK/$USER/env.sh}"
[[ -r "$MF_ENV_FILE" ]] || die "Ambiente non trovato: $MF_ENV_FILE"
source "$MF_ENV_FILE"

PROJECT_DIR="${PROJECT_DIR:-${MF_PROJECT_DIR:-$MORPHFLOW_REPO}}"
ROOT_DIR="${ROOT_DIR:-${MF_DATA_ROOT:-$FAST/$USER/datasets/morphing_dataset_v4}}"
SOURCE_IMAGES_ROOT="${SOURCE_IMAGES_ROOT:-${MF_SOURCE_IMAGES_ROOT:-$FAST/$USER/datasets/flux_outputs}}"
SOURCE_IMAGE_FILENAME="${SOURCE_IMAGE_FILENAME:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Impostare CHECKPOINT_PATH al checkpoint SS}"
SLAT_CHECKPOINT_PATH="${SLAT_CHECKPOINT_PATH:?Impostare SLAT_CHECKPOINT_PATH al checkpoint SLat}"
METADATA="${METADATA:-metadata_test.json}"

NUM_PAIRS="${NUM_PAIRS:-10}"
SELECTION="${SELECTION:-first}" # first | random | evenly_spaced
START_INDEX="${START_INDEX:-0}"
# Nel v4 ogni coppia compare con 4 alpha: selezionarla una sola volta.
DEDUPLICATE_PAIRS="${DEDUPLICATE_PAIRS:-1}"
NUM_INTERMEDIATE_STEPS="${NUM_INTERMEDIATE_STEPS:-5}"
INCLUDE_ENDPOINTS="${INCLUDE_ENDPOINTS:-0}"
DIRECTION="${DIRECTION:-src1_to_src2}"
ALPHAS="${ALPHAS:-}" # Opzionale: lista esplicita, sostituisce la griglia automatica.

SEED="${SEED:-42}"
STEPS="${STEPS:-50}"
SLAT_STEPS="${SLAT_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-3.0}"
SLAT_CFG_SCALE="${SLAT_CFG_SCALE:-3.0}"
TRELLIS_MODEL="${TRELLIS_MODEL:-image_large}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SAVE_LATENTS="${SAVE_LATENTS:-0}"
SAVE_VOXELS="${SAVE_VOXELS:-1}"
SAVE_SOURCES="${SAVE_SOURCES:-1}"
RUN_NAME="${RUN_NAME:-alpha_steps_test_v4}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK/$USER/outputs/alpha_steps_test_v4/$RUN_NAME}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
if [[ "$MODE" == --prepare ]]; then
    export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
else
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
mkdir -p "$WORK/$USER/logs"
cd "$PROJECT_DIR"
[[ -r generate_alpha_steps_validation.py ]] || die "Generatore non trovato in $PROJECT_DIR"
[[ -d "$ROOT_DIR/assets" && -d "$ROOT_DIR/targets" ]] || die "Dataset non trovato: $ROOT_DIR"
METADATA_PATH="$METADATA"
[[ "$METADATA_PATH" == /* ]] || METADATA_PATH="$ROOT_DIR/$METADATA_PATH"
[[ -r "$METADATA_PATH" ]] || die "Metadata non trovato: $METADATA_PATH"
[[ -r "$CHECKPOINT_PATH" ]] || die "Checkpoint SS non trovato: $CHECKPOINT_PATH"
[[ -r "$SLAT_CHECKPOINT_PATH" ]] || die "Checkpoint SLat non trovato: $SLAT_CHECKPOINT_PATH"
CHECKPOINT_PATH="$(realpath "$CHECKPOINT_PATH")"
SLAT_CHECKPOINT_PATH="$(realpath "$SLAT_CHECKPOINT_PATH")"

generate_args=(
    generate_alpha_steps_validation.py
    --root_dir "$ROOT_DIR" --metadata "$METADATA" --split test
    --checkpoint_path "$CHECKPOINT_PATH" --slat_checkpoint_path "$SLAT_CHECKPOINT_PATH"
    --num_pairs "$NUM_PAIRS" --selection "$SELECTION" --start_index "$START_INDEX"
    --deduplicate_pairs "$DEDUPLICATE_PAIRS"
    --num_intermediate_steps "$NUM_INTERMEDIATE_STEPS"
    --include_endpoints "$INCLUDE_ENDPOINTS" --direction "$DIRECTION"
    --seed "$SEED" --steps "$STEPS" --slat_steps "$SLAT_STEPS"
    --cfg_scale "$CFG_SCALE" --slat_cfg_scale "$SLAT_CFG_SCALE"
    --trellis_model "$TRELLIS_MODEL" --mixed_precision "$MIXED_PRECISION"
    --save_latents "$SAVE_LATENTS" --save_voxels "$SAVE_VOXELS" --save_sources "$SAVE_SOURCES"
)
[[ -z "$ALPHAS" ]] || generate_args+=(--alphas "$ALPHAS")
if [[ -d "$SOURCE_IMAGES_ROOT" ]]; then
    generate_args+=(--source_images_root "$SOURCE_IMAGES_ROOT")
    [[ -z "$SOURCE_IMAGE_FILENAME" ]] || generate_args+=(--source_image_filename "$SOURCE_IMAGE_FILENAME")
fi

python slurm/check_latent_inference_leonardo.py "$MODE" "${generate_args[@]}" --output_dir "$OUTPUT_DIR"
if [[ "$MODE" != --generate ]]; then
    echo "CHECK PASSED: ambiente, dataset, selezione, checkpoint e cache decoder disponibili."
    echo "NUM_PAIRS=$NUM_PAIRS NUM_INTERMEDIATE_STEPS=$NUM_INTERMEDIATE_STEPS INCLUDE_ENDPOINTS=$INCLUDE_ENDPOINTS"
    echo "STEPS=$STEPS SLAT_STEPS=$SLAT_STEPS CFG_SCALE=$CFG_SCALE SLAT_CFG_SCALE=$SLAT_CFG_SCALE"
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    exit 0
fi

srun --ntasks=1 python -c 'import torch; assert torch.cuda.is_available(), "GPU CUDA non disponibile"; print(torch.cuda.get_device_name(0))'
echo "JOB=$SLURM_JOB_ID NODES=${SLURM_JOB_NODELIST:-} ROOT_DIR=$ROOT_DIR METADATA=$METADATA"
job_output="$OUTPUT_DIR/job_${SLURM_JOB_ID}"
mkdir -p "$job_output"
command=(python "${generate_args[@]}" --output_dir "$job_output")
printf 'Generate command: '; printf '%q ' "${command[@]}"; printf '\n'
exec srun --ntasks=1 "${command[@]}"
