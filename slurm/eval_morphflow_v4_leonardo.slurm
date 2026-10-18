#!/bin/bash
#SBATCH --job-name=eval_v4_ss_slat_cfggrid
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=BEGIN,END,FAIL

# Eval del dataset v4: stessa pipeline SS/SLat e stessi parametri del launcher v3.
# Dal login: bash slurm/eval_morphflow_v4_leonardo.slurm --prepare
# Verifica offline: bash slurm/eval_morphflow_v4_leonardo.slurm --check
# Pipeline: CHECKPOINT_PATH=/percorso/ss.pt SLAT_CHECKPOINT_PATH=/percorso/slat.pt \
# sbatch slurm/eval_morphflow_v4_leonardo.slurm
# CFG_VALUES x SLAT_CFG_VALUES: griglia sulle stesse coppie, seed fisso.
# Richiede CHECKPOINT_PATH=SS e SLAT_CHECKPOINT_PATH=SLat.
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }

MODE="${1:---eval}"
case "$MODE" in
    --eval|--check|--prepare) ;;
    *) die "Uso: $0 [--eval|--check|--prepare]" ;;
esac
[[ $# -le 1 ]] || die "Usare le variabili ambiente per modificare i parametri."
if [[ "$MODE" == --eval ]]; then
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "Lanciare l'eval con sbatch; --check e --prepare funzionano dal login node."
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
EVAL_TYPE="${EVAL_TYPE:-eval_test_latents_v4}"
RUN_NAME="${RUN_NAME:-eval_v4_ss_slat_cfggrid}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK/$USER/outputs/$EVAL_TYPE/$RUN_NAME}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
SEED="${SEED:-42}"
STEPS="${STEPS:-50}"
SLAT_STEPS="${SLAT_STEPS:-50}"
CFG_VALUES="${CFG_VALUES:-1 1.5 2 2.5 3}"
# Anche SLAT_CFG_SCALE del launcher di ateneo e' accettato come CFG singolo.
SLAT_CFG_VALUES="${SLAT_CFG_VALUES:-${SLAT_CFG_SCALE:-1 1.5 2 2.5 3}}"
TRELLIS_MODEL="${TRELLIS_MODEL:-auto}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SAVE_LATENTS="${SAVE_LATENTS:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
# Download solo nella preparazione; i job usano la cache condivisa.
if [[ "$MODE" == --prepare ]]; then
    export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
else
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
mkdir -p "$WORK/$USER/logs"
cd "$PROJECT_DIR"
[[ -r eval_validation_latents.py ]] || die "Evaluator non trovato in $PROJECT_DIR"
[[ -d "$ROOT_DIR/assets" && -d "$ROOT_DIR/targets" ]] || die "Dataset non trovato: $ROOT_DIR"
METADATA_PATH="$METADATA"
[[ "$METADATA_PATH" == /* ]] || METADATA_PATH="$ROOT_DIR/$METADATA_PATH"
[[ -r "$METADATA_PATH" ]] || die "Metadata non trovato: $METADATA_PATH"
[[ -r "$CHECKPOINT_PATH" ]] || die "Checkpoint non trovato: $CHECKPOINT_PATH"
CHECKPOINT_PATH="$(realpath "$CHECKPOINT_PATH")"
if [[ -n "$SLAT_CHECKPOINT_PATH" ]]; then
    [[ -r "$SLAT_CHECKPOINT_PATH" ]] || die "Checkpoint SLat non trovato: $SLAT_CHECKPOINT_PATH"
    SLAT_CHECKPOINT_PATH="$(realpath "$SLAT_CHECKPOINT_PATH")"
fi
read -r -a cfg_values <<< "$CFG_VALUES"
read -r -a slat_cfg_values <<< "$SLAT_CFG_VALUES"
[[ ${#slat_cfg_values[@]} -gt 0 ]] || die "SLAT_CFG_VALUES vuoto."
[[ ${#cfg_values[@]} -gt 0 ]] || die "CFG_VALUES deve contenere almeno un valore."

eval_args=(
    eval_validation_latents.py
    --root_dir "$ROOT_DIR" --metadata "$METADATA"
    --checkpoint_path "$CHECKPOINT_PATH"
    --num_samples "$NUM_SAMPLES" --seed "$SEED" --steps "$STEPS"
    --trellis_model "$TRELLIS_MODEL" --mixed_precision "$MIXED_PRECISION"
    --save_latents "$SAVE_LATENTS"
)
if [[ -n "$SLAT_CHECKPOINT_PATH" ]]; then
    eval_args+=(--slat_checkpoint_path "$SLAT_CHECKPOINT_PATH"
                --slat_steps "$SLAT_STEPS")
fi
if [[ -d "$SOURCE_IMAGES_ROOT" ]]; then
    eval_args+=(--source_images_root "$SOURCE_IMAGES_ROOT")
    [[ -z "$SOURCE_IMAGE_FILENAME" ]] || eval_args+=(--source_image_filename "$SOURCE_IMAGE_FILENAME")
fi

# Valida gli stessi argomenti che verranno passati all'evaluator.
# Controlla dataset, selezione, checkpoint e decoder senza costruire i modelli.
python - "${cfg_values[@]}" "${slat_cfg_values[@]}" <<'PY_CFG'
import math
import sys
for raw in sys.argv[1:]:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise SystemExit(f"CFG non valido: {raw}")
PY_CFG
python slurm/check_latent_inference_leonardo.py "$MODE" "${eval_args[@]}" \
    --cfg_scale "${cfg_values[0]}" --slat_cfg_scale "${slat_cfg_values[0]}" \
    --output_dir "$OUTPUT_DIR"

if [[ "$MODE" != --eval ]]; then
    echo "CHECK PASSED: ambiente, checkpoint, metadata e cache decoder disponibili."
    echo "OUTPUT_DIR=$OUTPUT_DIR CFG_SS=$CFG_VALUES CFG_SLAT=$SLAT_CFG_VALUES"
    echo "NUM_SAMPLES=$NUM_SAMPLES STEPS=$STEPS SLAT_STEPS=$SLAT_STEPS SEED=$SEED"
    exit 0
fi

srun --ntasks=1 python -c 'import torch; assert torch.cuda.is_available(), "GPU CUDA non disponibile"; print(torch.cuda.get_device_name(0))'
echo "JOB=$SLURM_JOB_ID NODES=${SLURM_JOB_NODELIST:-} CHECKPOINT=$CHECKPOINT_PATH"
echo "METADATA=$METADATA NUM_SAMPLES=$NUM_SAMPLES SEED=$SEED STEPS=$STEPS"
echo "CFG_SS=$CFG_VALUES CFG_SLAT=$SLAT_CFG_VALUES SLAT_STEPS=$SLAT_STEPS"
for cfg in "${cfg_values[@]}"; do
    for slat_cfg in "${slat_cfg_values[@]}"; do
        cfg_dir="$OUTPUT_DIR/job_${SLURM_JOB_ID}/cfg_ss_${cfg}_slat_${slat_cfg}"
        mkdir -p "$cfg_dir"
        command=(python "${eval_args[@]}"
                 --cfg_scale "$cfg" --slat_cfg_scale "$slat_cfg"
                 --output_dir "$cfg_dir")
        printf 'Eval command: '; printf '%q ' "${command[@]}"; printf '\n'
        srun --ntasks=1 "${command[@]}"
    done
done
