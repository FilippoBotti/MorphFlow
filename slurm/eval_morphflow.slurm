#!/bin/bash
#SBATCH --job-name=eval_morphflow
#SBATCH --account=g_vbd
#SBATCH --partition=gpu_vbd
#SBATCH --qos=gpu_vbd
#SBATCH --nodes=1
#SBATCH --nodelist=wn46
#SBATCH --gres=gpu:a100_vbd:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/final_dataset/eval-%x-%j.out
#SBATCH --error=/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/final_dataset/eval-%x-%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module purge
module load singularity/3.8.7

PROJECT_DIR="${PROJECT_DIR:-$HOME/trellis-singularity/work/MorphFlow}"
SIF="${SIF:-$HOME/containers/trellis-py240-cu118.sif}"
TRELLIS_BASE="${TRELLIS_BASE:-$HOME/trellis-singularity}"

DATA_ROOT_HOST="${ROOT_DIR:-/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v2}"
DATA_ROOT_IN="/data/morphing_dataset_v2"

SOURCE_IMAGES_ROOT_HOST="${SOURCE_IMAGES_ROOT:-/hpc/scratch/marco.barezzi/3d_dataset/flux_outputs}"
SOURCE_IMAGES_ROOT_IN="/data/flux_outputs"
SOURCE_IMAGE_FILENAME="${SOURCE_IMAGE_FILENAME:-}"

ARCHIVE_HOST="${ARCHIVE_HOST:-/hpc/archive/G_VBD/marco.barezzi}"
ARCHIVE_IN="/archive"

CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH=/path/to/checkpoint.pt}"
SLAT_CHECKPOINT_PATH="${SLAT_CHECKPOINT_PATH:-}"

METADATA="${METADATA:-metadata_test.json}"
EVAL_TYPE="${EVAL_TYPE:-eval_test_latents}"
RUN_NAME="${RUN_NAME:-$(basename "$(dirname "$(dirname "$CHECKPOINT_PATH")")")}"

NUM_SAMPLES="${NUM_SAMPLES:-10}"
SEED="${SEED:-42}"
STEPS="${STEPS:-25}"
SLAT_STEPS="${SLAT_STEPS:-25}"
CFG_VALUES="${CFG_VALUES:-1.0}"
SLAT_CFG_SCALE="${SLAT_CFG_SCALE:-1.0}"
TRELLIS_MODEL="${TRELLIS_MODEL:-image_large}"
MIXED_PRECISION="${MIXED_PRECISION:-auto}"
SAVE_LATENTS="${SAVE_LATENTS:-0}"

OUTPUT_DIR_HOST="${OUTPUT_DIR:-$PROJECT_DIR/outputs/$EVAL_TYPE/$RUN_NAME}"
mkdir -p "$OUTPUT_DIR_HOST" "$ARCHIVE_HOST/morphflow_runs/final_dataset"

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
OUTPUT_DIR_IN="$OUTPUT_DIR_HOST"

if [ ! -d "$DATA_ROOT_HOST" ]; then
  echo "ERROR: dataset not found: $DATA_ROOT_HOST"
  exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT_PATH"
  exit 1
fi

SLAT_ARG=""
SOURCE_IMAGES_ARG=""
SOURCE_BIND_ARG=()
if [ -d "$SOURCE_IMAGES_ROOT_HOST" ]; then
  SOURCE_IMAGES_ROOT_IN_PATH="$(to_container_path "$SOURCE_IMAGES_ROOT_HOST")"
  SOURCE_BIND_ARG=(--bind "$SOURCE_IMAGES_ROOT_HOST:$SOURCE_IMAGES_ROOT_IN:ro")
  SOURCE_IMAGES_ARG="--source_images_root $SOURCE_IMAGES_ROOT_IN_PATH"
  if [ -n "$SOURCE_IMAGE_FILENAME" ]; then
    SOURCE_IMAGES_ARG="$SOURCE_IMAGES_ARG --source_image_filename $SOURCE_IMAGE_FILENAME"
  fi
else
  echo "WARNING: source images root not found: $SOURCE_IMAGES_ROOT_HOST"
fi

if [ -n "$SLAT_CHECKPOINT_PATH" ]; then
  if [ ! -f "$SLAT_CHECKPOINT_PATH" ]; then
    echo "ERROR: SLat checkpoint not found: $SLAT_CHECKPOINT_PATH"
    exit 1
  fi
  SLAT_CHECKPOINT_IN="$(to_container_path "$SLAT_CHECKPOINT_PATH")"
  SLAT_ARG="--slat_checkpoint_path $SLAT_CHECKPOINT_IN --slat_steps $SLAT_STEPS --slat_cfg_scale $SLAT_CFG_SCALE"
fi

echo "===== EVAL SLURM ====="
echo "JOB_ID=$SLURM_JOB_ID"
echo "NODELIST=$SLURM_JOB_NODELIST"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATA_ROOT_HOST=$DATA_ROOT_HOST"
echo "CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "SLAT_CHECKPOINT_PATH=${SLAT_CHECKPOINT_PATH:-<none>}"
echo "OUTPUT_DIR_HOST=$OUTPUT_DIR_HOST"
echo "SOURCE_IMAGES_ROOT_HOST=${SOURCE_IMAGES_ROOT_HOST:-<none>}"
echo "EVAL_TYPE=$EVAL_TYPE"
echo "RUN_NAME=$RUN_NAME"
echo "CFG_VALUES=$CFG_VALUES"
echo "======================"

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

for CFG in $CFG_VALUES; do
  echo ">>> Eval CFG=\$CFG"
  python eval_validation_latents.py \
    --root_dir "$DATA_ROOT_IN" \
    --metadata "$METADATA" \
    --checkpoint_path "$CHECKPOINT_IN" \
    --output_dir "$OUTPUT_DIR_IN" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --steps "$STEPS" \
    --cfg_scale "\$CFG" \
    --trellis_model "$TRELLIS_MODEL" \
    --mixed_precision "$MIXED_PRECISION" \
    --save_latents "$SAVE_LATENTS" \
    $SLAT_ARG \
    $SOURCE_IMAGES_ARG
done
IN_CONTAINER
