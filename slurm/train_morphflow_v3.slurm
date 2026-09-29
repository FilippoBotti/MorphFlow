#!/bin/bash
#SBATCH --job-name=v3_ss_trellisPrior_teacherSoft0p6_semStrong_T0p05_align0p35_hub0p01
#SBATCH --account=g_vbd
#SBATCH --partition=gpu_vbd
#SBATCH --qos=gpu_vbd
#SBATCH --nodes=1
#SBATCH --nodelist=wn49
#SBATCH --gres=gpu:l40s_vbd:8
#SBATCH --cpus-per-task=16	
#SBATCH --mem=128G
#SBATCH --time=10-00:00:00
#SBATCH --output=/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/final_dataset/slurm-bootstrap-%x-%j.out
#SBATCH --error=/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/final_dataset/slurm-bootstrap-%x-%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=ALL

set -euo pipefail

module purge
module load singularity/3.8.7

PROJECT_DIR="$HOME/trellis-singularity/work/MorphFlow"
SIF="$HOME/containers/trellis-py240-cu118.sif"
TRELLIS_BASE="$HOME/trellis-singularity"

FLUX_ROOT_HOST="/hpc/scratch/marco.barezzi/3d_dataset/flux_outputs"
FLUX_ROOT_IN="/data/flux_outputs"

DATA_ROOT_HOST="/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v3"
DATA_ROOT_IN="/data/morphing_dataset_v3"

ARCHIVE_HOST="/hpc/archive/G_VBD/marco.barezzi"
OUT_ROOT_HOST="$ARCHIVE_HOST/morphflow_runs"
OUT_ROOT_IN="/archive/morphflow_runs"

RUN_NAME="${RUN_NAME:-$SLURM_JOB_NAME}"
# SS: LoRA cross-attn + semantic matching + hubness regularization.
# Cycle OFF; usage weight=0.01 cap=4; flow alpha embedder frozen/off.
SEMANTIC_CYCLE_LOSS_WEIGHT="${SEMANTIC_CYCLE_LOSS_WEIGHT:-0.0}"
SEMANTIC_CYCLE_LOSS_PROB="${SEMANTIC_CYCLE_LOSS_PROB:-0.0}"
SEMANTIC_USAGE_LOSS_WEIGHT="${SEMANTIC_USAGE_LOSS_WEIGHT:-0.01}"
SEMANTIC_USAGE_CAP="${SEMANTIC_USAGE_CAP:-4.0}"
# Frozen native TRELLIS SS prior on differentiable student rollouts.
# Weight ramps linearly over optimizer steps; cadence counts training microbatches.
# TRELLIS_PRIOR_WEIGHT=0 disables the extra model and rollout (baseline ablation).
TRELLIS_PRIOR_WEIGHT="${TRELLIS_PRIOR_WEIGHT:-0.1}"
TRELLIS_PRIOR_EVERY="${TRELLIS_PRIOR_EVERY:-4}"
TRELLIS_PRIOR_WARMUP_STEPS="${TRELLIS_PRIOR_WARMUP_STEPS:-10000}"
TRELLIS_PRIOR_ROLLOUT_STEPS="${TRELLIS_PRIOR_ROLLOUT_STEPS:-8}"
TRELLIS_PRIOR_GRAD_STEPS="${TRELLIS_PRIOR_GRAD_STEPS:-2}"
TRELLIS_PRIOR_MAX_ITEMS="${TRELLIS_PRIOR_MAX_ITEMS:-1}"
TRELLIS_PRIOR_CHECKPOINT="${TRELLIS_PRIOR_CHECKPOINT:-1}"
TRELLIS_PRIOR_T_MIN="${TRELLIS_PRIOR_T_MIN:-0.05}"
TRELLIS_PRIOR_T_MAX="${TRELLIS_PRIOR_T_MAX:-0.20}"
TRELLIS_PRIOR_PROJECTION_CLIP_RATIO="${TRELLIS_PRIOR_PROJECTION_CLIP_RATIO:-0.10}"
TRELLIS_PRIOR_RMS_GUARD_WEIGHT="${TRELLIS_PRIOR_RMS_GUARD_WEIGHT:-1.0}"
TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO="${TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO:-0.25}"
TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO="${TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO:-2.0}"
RUN_DIR_HOST="$OUT_ROOT_HOST/$RUN_NAME"
LOG_DIR_HOST="$RUN_DIR_HOST/logs"

mkdir -p "$OUT_ROOT_HOST/final_dataset" "$LOG_DIR_HOST" "$RUN_DIR_HOST/checkpoints" "$TRELLIS_BASE/hf-home" "$TRELLIS_BASE/tmp"

if [ ! -d "$DATA_ROOT_HOST" ]; then
  echo "ERROR: dataset not found: $DATA_ROOT_HOST"
  exit 1
fi

AUTO_RESUME="${AUTO_RESUME:-0}"
RESUME_FROM="${RESUME_FROM:-}"
RESUME_ARG=""

if [ "$AUTO_RESUME" = "1" ] && [ -z "$RESUME_FROM" ]; then
  # morphflow_last.pt viene sovrascritto alla fine di ogni epoca completata.
  if [ -f "$RUN_DIR_HOST/checkpoints/morphflow_last.pt" ]; then
    RESUME_FROM="$RUN_DIR_HOST/checkpoints/morphflow_last.pt"
  else
    # Compatibilità con run precedenti che non hanno morphflow_last.pt.
    latest_ckpt="$(
      find "$RUN_DIR_HOST/checkpoints"         -maxdepth 1         -type f         \(           -name 'morphflow_epoch_*.pt'           -o -name 'morphflow_best.pt'           -o -name 'morphflow_final_*.pt'         \)         -printf '%T@ %p\n' 2>/dev/null |
      sort -nr |
      head -n 1 |
      cut -d' ' -f2-
    )"

    if [ -n "$latest_ckpt" ]; then
      RESUME_FROM="$latest_ckpt"
    fi
  fi
fi

if [ -n "$RESUME_FROM" ] && [[ "$RESUME_FROM" == "$ARCHIVE_HOST"* ]]; then
  RESUME_FROM="/archive${RESUME_FROM#$ARCHIVE_HOST}"
fi

if [ -n "$RESUME_FROM" ]; then
  RESUME_ARG="--resume_from $RESUME_FROM --resume_strict 1 --resume_optimizer 1"
else
  RESUME_FROM="<none>"
fi

exec > >(tee -a "$LOG_DIR_HOST/train.log") 2>&1

echo "RUN_NAME=$RUN_NAME"
echo "RUN_DIR_HOST=$RUN_DIR_HOST"
echo "DATA_ROOT_HOST=$DATA_ROOT_HOST"
echo "AUTO_RESUME=$AUTO_RESUME"
echo "RESUME_FROM=$RESUME_FROM"
echo "SEMANTIC_CYCLE_LOSS_WEIGHT=$SEMANTIC_CYCLE_LOSS_WEIGHT"
echo "SEMANTIC_USAGE_LOSS_WEIGHT=$SEMANTIC_USAGE_LOSS_WEIGHT"
echo "SEMANTIC_USAGE_CAP=$SEMANTIC_USAGE_CAP"
echo "TRELLIS_PRIOR_WEIGHT=$TRELLIS_PRIOR_WEIGHT EVERY=$TRELLIS_PRIOR_EVERY WARMUP_STEPS=$TRELLIS_PRIOR_WARMUP_STEPS"
echo "TRELLIS_PRIOR_ROLLOUT_STEPS=$TRELLIS_PRIOR_ROLLOUT_STEPS GRAD_STEPS=$TRELLIS_PRIOR_GRAD_STEPS MAX_ITEMS=$TRELLIS_PRIOR_MAX_ITEMS CHECKPOINT=$TRELLIS_PRIOR_CHECKPOINT"
echo "TRELLIS_PRIOR_T_RANGE=$TRELLIS_PRIOR_T_MIN,$TRELLIS_PRIOR_T_MAX"
echo "TRELLIS_PRIOR_PROJECTION_CLIP_RATIO=$TRELLIS_PRIOR_PROJECTION_CLIP_RATIO"
echo "TRELLIS_PRIOR_RMS_GUARD weight=$TRELLIS_PRIOR_RMS_GUARD_WEIGHT low=$TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO high=$TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO"

export SINGULARITYENV_PROJECT_DIR="$PROJECT_DIR"
export SINGULARITYENV_DATA_ROOT_IN="$DATA_ROOT_IN"
export SINGULARITYENV_FLUX_ROOT_IN="$FLUX_ROOT_IN"
export SINGULARITYENV_OUT_ROOT_IN="$OUT_ROOT_IN"
export SINGULARITYENV_RUN_NAME="$RUN_NAME"
export SINGULARITYENV_RESUME_ARG="$RESUME_ARG"
export SINGULARITYENV_SEMANTIC_CYCLE_LOSS_WEIGHT="$SEMANTIC_CYCLE_LOSS_WEIGHT"
export SINGULARITYENV_SEMANTIC_CYCLE_LOSS_PROB="$SEMANTIC_CYCLE_LOSS_PROB"
export SINGULARITYENV_SEMANTIC_USAGE_LOSS_WEIGHT="$SEMANTIC_USAGE_LOSS_WEIGHT"
export SINGULARITYENV_SEMANTIC_USAGE_CAP="$SEMANTIC_USAGE_CAP"
export SINGULARITYENV_TRELLIS_PRIOR_WEIGHT="$TRELLIS_PRIOR_WEIGHT"
export SINGULARITYENV_TRELLIS_PRIOR_EVERY="$TRELLIS_PRIOR_EVERY"
export SINGULARITYENV_TRELLIS_PRIOR_WARMUP_STEPS="$TRELLIS_PRIOR_WARMUP_STEPS"
export SINGULARITYENV_TRELLIS_PRIOR_ROLLOUT_STEPS="$TRELLIS_PRIOR_ROLLOUT_STEPS"
export SINGULARITYENV_TRELLIS_PRIOR_GRAD_STEPS="$TRELLIS_PRIOR_GRAD_STEPS"
export SINGULARITYENV_TRELLIS_PRIOR_MAX_ITEMS="$TRELLIS_PRIOR_MAX_ITEMS"
export SINGULARITYENV_TRELLIS_PRIOR_CHECKPOINT="$TRELLIS_PRIOR_CHECKPOINT"
export SINGULARITYENV_TRELLIS_PRIOR_T_MIN="$TRELLIS_PRIOR_T_MIN"
export SINGULARITYENV_TRELLIS_PRIOR_T_MAX="$TRELLIS_PRIOR_T_MAX"
export SINGULARITYENV_TRELLIS_PRIOR_PROJECTION_CLIP_RATIO="$TRELLIS_PRIOR_PROJECTION_CLIP_RATIO"
export SINGULARITYENV_TRELLIS_PRIOR_RMS_GUARD_WEIGHT="$TRELLIS_PRIOR_RMS_GUARD_WEIGHT"
export SINGULARITYENV_TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO="$TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO"
export SINGULARITYENV_TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO="$TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO"
export SINGULARITYENV_HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

singularity exec --cleanenv --nv \
  --bind "$TRELLIS_BASE:/opt/trellis" \
  --bind "$DATA_ROOT_HOST:$DATA_ROOT_IN:ro" \
  --bind "$ARCHIVE_HOST:/archive" \
  --bind "$HOME:$HOME" \
  --bind "$FLUX_ROOT_HOST:$FLUX_ROOT_IN:ro" \
  "$SIF" \
  bash -s <<'IN_CONTAINER'
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/trellis/env

export HF_HOME=/opt/trellis/hf-home
export TMPDIR="/tmp/mf_${SLURM_JOB_ID:-$$}"
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=2

mkdir -p "$TMPDIR" "$OUT_ROOT_IN/$RUN_NAME"

cd "$PROJECT_DIR"

python --version
python -c 'import torch; print("torch=", torch.__version__, "cuda=", torch.version.cuda, "avail=", torch.cuda.is_available(), "bf16=", torch.cuda.is_bf16_supported())'
nvidia-smi -L || true

# Warm/check the shared cache once before launching the eight GPU processes.
# The native prior assets are required on resumed runs as well.
python - "$TRELLIS_PRIOR_WEIGHT" <<'PY'
import math
import os
import sys

weight = float(sys.argv[1])
if not math.isfinite(weight) or weight < 0:
    raise SystemExit("TRELLIS_PRIOR_WEIGHT must be finite and >= 0.")
if weight > 0:
    from huggingface_hub import hf_hub_download
    for suffix in ("json", "safetensors"):
        path = hf_hub_download(
            repo_id="microsoft/TRELLIS-image-large",
            filename=f"ckpts/ss_flow_img_dit_L_16l8_fp16.{suffix}",
            local_files_only=os.environ.get("HF_HUB_OFFLINE", "0").upper() in {"1", "ON", "YES", "TRUE"},
        )
        print(f"TRELLIS prior cache: {path}")
PY

accelerate launch \
  --multi-gpu \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  main.py \
  --root_dir "$DATA_ROOT_IN" \
  --metadata metadata_train.json \
  --val_metadata metadata_val.json \
  --out_dir "$OUT_ROOT_IN" \
  --run_name "$RUN_NAME" \
  --train_bs 2 \
  --val_bs 1 \
  --train_epochs 25 \
  --lr 1e-4 \
  --cond_lr 1e-4 \
  --flow_lr 5e-5 \
  --lora_lr 5e-5 \
  --weight_decay 1e-4 \
  --warmup_steps 10000 \
  --plateau_factor 0.5 \
  --plateau_patience 5 \
  --grad_clip 1.0 \
  --log_every 20 \
  --val_every 1 \
  --val_max_items 0 \
  --num_workers 8 \
  --lr_scheduler cosine \
  --cond_token_norm none \
  --cond_proj_norm layernorm \
  --cond_input_norm trellis \
  --cond_encoder_type sparse_conv3d \
  --cond_style_tokens 8 \
  --cond_use_occupancy 1 \
  --cond_hybrid_pool_stats 1 \
  --cond_residual_blocks_64 2 \
  --cond_residual_blocks_32 2 \
  --cond_residual_blocks_16 2 \
  --cfg_drop_prob 0.2 \
  --trellis_model image_large \
  --flow_target ss \
  --teacher_mid_weight 0.60 \
  --trellis_prior_weight "$TRELLIS_PRIOR_WEIGHT" \
  --trellis_prior_every "$TRELLIS_PRIOR_EVERY" \
  --trellis_prior_warmup_steps "$TRELLIS_PRIOR_WARMUP_STEPS" \
  --trellis_prior_rollout_steps "$TRELLIS_PRIOR_ROLLOUT_STEPS" \
  --trellis_prior_grad_steps "$TRELLIS_PRIOR_GRAD_STEPS" \
  --trellis_prior_max_items "$TRELLIS_PRIOR_MAX_ITEMS" \
  --trellis_prior_checkpoint "$TRELLIS_PRIOR_CHECKPOINT" \
  --trellis_prior_t_min "$TRELLIS_PRIOR_T_MIN" \
  --trellis_prior_t_max "$TRELLIS_PRIOR_T_MAX" \
  --trellis_prior_projection_clip_ratio "$TRELLIS_PRIOR_PROJECTION_CLIP_RATIO" \
  --trellis_prior_rms_guard_weight "$TRELLIS_PRIOR_RMS_GUARD_WEIGHT" \
  --trellis_prior_rms_guard_low_ratio "$TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO" \
  --trellis_prior_rms_guard_high_ratio "$TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO" \
  --slat_condition_source slat \
  --source_images_root "$FLUX_ROOT_IN" \
  --dino_model dinov2_vitl14_reg \
  --dino_dim 1024 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.1 \
  --lora_attention_scope cross \
  --lora_target_modules to_q,to_kv \
  --cond_resample_depth 1 \
  --cond_resample_heads 8 \
  --ss_flow_arch standard \
  --residual_interp_gate alpha \
  --residual_interp_gate_min 1e-3 \
  --residual_endpoint_prob 0 \
  --endpoint_loss_weight 0.0 \
  --symmetry_loss_weight 0.0 \
  --endpoint_loss_prob 0.0 \
  --symmetry_loss_prob 0.0 \
  --use_semantic_token_matching 1 \
  --semantic_match_max_align 0.35 \
  --semantic_match_temperature 0.05 \
  --semantic_match_dim 128 \
  --semantic_match_alpha_weight 1 \
  --semantic_match_detach_scores 0 \
  --semantic_match_exclude_style_tokens 1 \
  --semantic_cycle_loss_weight "$SEMANTIC_CYCLE_LOSS_WEIGHT" \
  --semantic_cycle_loss_prob "$SEMANTIC_CYCLE_LOSS_PROB" \
  --semantic_cycle_detach_targets 1 \
  --semantic_cycle_alpha_weight 1 \
  --semantic_usage_loss_weight "$SEMANTIC_USAGE_LOSS_WEIGHT" \
  --semantic_usage_cap "$SEMANTIC_USAGE_CAP" \
  --semantic_match_log_stats 1 \
  --separate_cond 1 \
  --separate_cond_gate token \
  --t_schedule logit_normal \
  --t_logit_mean 1.0 \
  --t_logit_std 1.0 \
  --trainable_scope cond_cross_attn \
  --freeze_modules none \
  --use_lora 1 \
  --train_flow_alpha_gate 1 \
  --train_flow_alpha_embedder 0 \
  --use_ema 0 \
  --mixed_precision bf16 \
  --allow_tf32 1 \
  --use_checkpoint 0 \
  --augment_source_swap 1 \
  --checkpoint_every 2 \
  $RESUME_ARG

echo "RUN_DIR=$OUT_ROOT_IN/$RUN_NAME"
IN_CONTAINER
