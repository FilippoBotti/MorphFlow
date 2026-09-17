#!/bin/bash
#SBATCH --job-name=alpha_steps_test
#SBATCH --account=g_vbd
#SBATCH --partition=gpu_vbd
#SBATCH --qos=gpu_vbd
#SBATCH --nodes=1
#SBATCH --nodelist=wn46
#SBATCH --gres=gpu:rtx6000_vbd:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/final_dataset/alpha-%x-%j.out
#SBATCH --error=/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/final_dataset/alpha-%x-%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module purge
module load singularity/3.8.7

PROJECT_DIR="${PROJECT_DIR:-$HOME/trellis-singularity/work/MorphFlow}"
SIF="${SIF:-$HOME/containers/trellis-py240-cu118.sif}"
TRELLIS_BASE="${TRELLIS_BASE:-$HOME/trellis-singularity}"

DATA_ROOT_HOST="${ROOT_DIR:-/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v3}"
DATA_ROOT_IN="/data/morphing_dataset_v2"

SOURCE_IMAGES_ROOT_HOST="${SOURCE_IMAGES_ROOT:-/hpc/scratch/marco.barezzi/3d_dataset/flux_outputs}"
SOURCE_IMAGES_ROOT_IN="/data/flux_outputs"
SOURCE_IMAGE_FILENAME="${SOURCE_IMAGE_FILENAME:-}"

ARCHIVE_HOST="${ARCHIVE_HOST:-/hpc/archive/G_VBD/marco.barezzi}"
ARCHIVE_IN="/archive"

CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH=/path/to/ss_checkpoint.pt}"
SLAT_CHECKPOINT_PATH="${SLAT_CHECKPOINT_PATH:?Set SLAT_CHECKPOINT_PATH=/path/to/slat_checkpoint.pt}"

METADATA="${METADATA:-metadata_test.json}"

# Numero di coppie src_1/src_2 del test set
NUM_PAIRS="${NUM_PAIRS:-10}"

# first | random | evenly_spaced
SELECTION="${SELECTION:-first}"
START_INDEX="${START_INDEX:-0}"

# 0 = usa le righe del test metadata come coppie.
# Non ci interessa il target.
DEDUPLICATE_PAIRS="${DEDUPLICATE_PAIRS:-1}"

# Numero di alpha intermedi fra src1 e src2.
# Con 9:
# 0.9, 0.8, ..., 0.1
NUM_INTERMEDIATE_STEPS="${NUM_INTERMEDIATE_STEPS:-5}"

# Gli endpoint reali src1/src2 vengono gia' salvati.
INCLUDE_ENDPOINTS="${INCLUDE_ENDPOINTS:-0}"
DIRECTION="${DIRECTION:-src1_to_src2}"

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

RUN_NAME="${RUN_NAME:-$(basename "$(dirname "$(dirname "$CHECKPOINT_PATH")")")}"
OUTPUT_DIR_HOST="${OUTPUT_DIR:-$PROJECT_DIR/outputs/alpha_steps_test/$RUN_NAME}"

mkdir -p \
  "$OUTPUT_DIR_HOST" \
  "$ARCHIVE_HOST/morphflow_runs/final_dataset"

to_container_path() {
  local x="$1"

  if [[ "$x" == "$ARCHIVE_HOST"* ]]; then
    echo "$ARCHIVE_IN${x#$ARCHIVE_HOST}"
  elif [[ "$x" == "$DATA_ROOT_HOST"* ]]; then
    echo "$DATA_ROOT_IN${x#$DATA_ROOT_HOST}"
  elif [[ "$x" == "$SOURCE_IMAGES_ROOT_HOST"* ]]; then
    echo "$SOURCE_IMAGES_ROOT_IN${x#$SOURCE_IMAGES_ROOT_HOST}"
  else
    echo "$x"
  fi
}

CHECKPOINT_IN="$(to_container_path "$CHECKPOINT_PATH")"
SLAT_CHECKPOINT_IN="$(to_container_path "$SLAT_CHECKPOINT_PATH")"
OUTPUT_DIR_IN="$OUTPUT_DIR_HOST"

if [ ! -d "$DATA_ROOT_HOST" ]; then
  echo "ERROR: dataset not found: $DATA_ROOT_HOST"
  exit 1
fi

if [ ! -f "$DATA_ROOT_HOST/$METADATA" ]; then
  echo "ERROR: metadata not found: $DATA_ROOT_HOST/$METADATA"
  exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "ERROR: SS checkpoint not found: $CHECKPOINT_PATH"
  exit 1
fi

if [ ! -f "$SLAT_CHECKPOINT_PATH" ]; then
  echo "ERROR: SLat checkpoint not found: $SLAT_CHECKPOINT_PATH"
  exit 1
fi

SOURCE_BIND_ARG=()
SOURCE_IMAGES_ARG=""

if [ -d "$SOURCE_IMAGES_ROOT_HOST" ]; then
  SOURCE_BIND_ARG=(
    --bind "$SOURCE_IMAGES_ROOT_HOST:$SOURCE_IMAGES_ROOT_IN:ro"
  )

  SOURCE_IMAGES_ARG="--source_images_root $SOURCE_IMAGES_ROOT_IN"

  if [ -n "$SOURCE_IMAGE_FILENAME" ]; then
    SOURCE_IMAGES_ARG="$SOURCE_IMAGES_ARG --source_image_filename $SOURCE_IMAGE_FILENAME"
  fi
else
  echo "WARNING: source images root not found: $SOURCE_IMAGES_ROOT_HOST"
fi

echo "===== TEST PAIR ALPHA SWEEP ====="
echo "JOB_ID=$SLURM_JOB_ID"
echo "NODELIST=$SLURM_JOB_NODELIST"
echo "DATA_ROOT=$DATA_ROOT_HOST"
echo "METADATA=$METADATA"
echo "SS_CHECKPOINT=$CHECKPOINT_PATH"
echo "SLAT_CHECKPOINT=$SLAT_CHECKPOINT_PATH"
echo "NUM_PAIRS=$NUM_PAIRS"
echo "SELECTION=$SELECTION"
echo "START_INDEX=$START_INDEX"
echo "NUM_INTERMEDIATE_STEPS=$NUM_INTERMEDIATE_STEPS"
echo "DIRECTION=$DIRECTION"
echo "CFG_SCALE=$CFG_SCALE"
echo "SLAT_CFG_SCALE=$SLAT_CFG_SCALE"
echo "OUTPUT=$OUTPUT_DIR_HOST"
echo "================================="

singularity exec --cleanenv --nv \
  --bind "$TRELLIS_BASE:/opt/trellis" \
  --bind "$DATA_ROOT_HOST:$DATA_ROOT_IN:ro" \
  "${SOURCE_BIND_ARG[@]}" \
  --bind "$ARCHIVE_HOST:$ARCHIVE_IN" \
  --bind "$HOME:$HOME" \
  "$SIF" \
  bash -s <<IN_CONTAINER

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/trellis/env

export HF_HOME=/opt/trellis/hf-home
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export PYTHONUNBUFFERED=1

TRELLIS_REPO=""

for candidate in \
  /opt/trellis/work/TRELLIS/MorphAny3D \
  /opt/trellis/work/TRELLIS \
  "$PROJECT_DIR/../TRELLIS/MorphAny3D" \
  "$PROJECT_DIR/../TRELLIS" \
  "$HOME/trellis-singularity/work/TRELLIS/MorphAny3D" \
  "$HOME/trellis-singularity/work/TRELLIS"
do
  if [ -d "\$candidate/trellis" ]; then
    TRELLIS_REPO="\$candidate"
    break
  fi
done

if [ -z "\$TRELLIS_REPO" ]; then
  echo "ERROR: could not find TRELLIS repo containing trellis/"
  exit 1
fi

export TRELLIS_REPO
export PYTHONPATH="\$TRELLIS_REPO:\${PYTHONPATH:-}"

echo "TRELLIS_REPO=\$TRELLIS_REPO"

cd "$PROJECT_DIR"

python --version
python -c 'import torch; print("torch=", torch.__version__, "cuda=", torch.version.cuda, "avail=", torch.cuda.is_available(), "bf16=", torch.cuda.is_bf16_supported())'
nvidia-smi -L || true

python generate_alpha_steps_validation.py \
  --root_dir "$DATA_ROOT_IN" \
  --metadata "$METADATA" \
  --split test \
  --checkpoint_path "$CHECKPOINT_IN" \
  --slat_checkpoint_path "$SLAT_CHECKPOINT_IN" \
  --output_dir "$OUTPUT_DIR_IN" \
  --num_pairs "$NUM_PAIRS" \
  --start_index "$START_INDEX" \
  --selection "$SELECTION" \
  --deduplicate_pairs "$DEDUPLICATE_PAIRS" \
  --num_intermediate_steps "$NUM_INTERMEDIATE_STEPS" \
  --include_endpoints "$INCLUDE_ENDPOINTS" \
  --direction "$DIRECTION" \
  --steps "$STEPS" \
  --slat_steps "$SLAT_STEPS" \
  --cfg_scale "$CFG_SCALE" \
  --slat_cfg_scale "$SLAT_CFG_SCALE" \
  --seed "$SEED" \
  --trellis_model "$TRELLIS_MODEL" \
  --mixed_precision "$MIXED_PRECISION" \
  --save_latents "$SAVE_LATENTS" \
  --save_voxels "$SAVE_VOXELS" \
  --save_sources "$SAVE_SOURCES" \
  $SOURCE_IMAGES_ARG

IN_CONTAINER
