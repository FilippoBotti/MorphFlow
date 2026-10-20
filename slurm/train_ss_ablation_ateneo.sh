#!/bin/bash
# Shared implementation of the four SS ablations; submit the .slurm wrappers.
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }
ABLATION="${1:?Expected baseline, gate_only, swap_only or prior_only}"
shift
MODE="${1:---run}"
[[ $# -le 1 ]] || die "Use --dry-run or configure the launcher with environment variables."
[[ "$MODE" == --run || "$MODE" == --dry-run ]] || die "Unknown mode: $MODE"
CONDITION_GATE=alpha
SOURCE_SWAP=0
PRIOR_WEIGHT=0.0
case "$ABLATION" in
  baseline) ;;
  gate_only) CONDITION_GATE=token ;;
  swap_only) SOURCE_SWAP=1 ;;
  prior_only) PRIOR_WEIGHT=0.1 ;;
  *) die "Unknown SS ablation: $ABLATION" ;;
esac
TRAIN_ALPHA_GATE=0
[[ "$CONDITION_GATE" != token ]] || TRAIN_ALPHA_GATE=1
TRAIN_BS="${TRAIN_BS:-1}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-2}"
# Original v4 SS: 20 epochs * 100000 samples / (8 GPUs * batch 1).
TARGET_TRAIN_STEPS="${TARGET_TRAIN_STEPS:-250000}"
[[ "$TRAIN_BS" =~ ^[1-9][0-9]*$ ]] || die "TRAIN_BS must be positive."
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be a nonnegative integer."
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || die "NUM_WORKERS must be a nonnegative integer."
[[ "$TARGET_TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]] || die "TARGET_TRAIN_STEPS must be positive."
if [[ "$MODE" == --dry-run ]]; then
  NGPUS="${NGPUS:-${SLURM_GPUS_ON_NODE:-2}}"
else
  [[ -n "${SLURM_JOB_ID:-}" ]] || die "Submit a .slurm wrapper with sbatch."
  [[ "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || die "These launchers use one node."
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != NoDevFiles ]] || die "Slurm did not set CUDA_VISIBLE_DEVICES."
  IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
  NGPUS="${#visible_gpus[@]}"
  if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then
    [[ "$NGPUS" == "$SLURM_GPUS_ON_NODE" ]] || die "GPU visibility differs from the allocation."
  fi
fi
[[ "$NGPUS" =~ ^[1-9][0-9]*$ ]] || die "Invalid allocated GPU count: $NGPUS"
GLOBAL_BATCH=$((NGPUS * TRAIN_BS))
TRAIN_SAMPLES=$((50000 * (1 + SOURCE_SWAP)))
STEPS_PER_EPOCH=$(((TRAIN_SAMPLES + GLOBAL_BATCH - 1) / GLOBAL_BATCH))
(( TARGET_TRAIN_STEPS % STEPS_PER_EPOCH == 0 )) || die "Cannot reach exactly $TARGET_TRAIN_STEPS updates with whole epochs: global batch=$GLOBAL_BATCH, steps/epoch=$STEPS_PER_EPOCH. Choose a compatible batch or TARGET_TRAIN_STEPS."
TRAIN_EPOCHS=$((TARGET_TRAIN_STEPS / STEPS_PER_EPOCH))
RUN_NAME="${RUN_NAME:-v4_ss_ablation_${ABLATION}_steps${TARGET_TRAIN_STEPS}_e${TRAIN_EPOCHS}_s${SEED}_g${NGPUS}_bs${TRAIN_BS}}"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_.-]+$ && "$RUN_NAME" != . && "$RUN_NAME" != .. ]] || die "Invalid RUN_NAME."
echo "ABLATION=$ABLATION NODE=${SLURM_JOB_NODELIST:-wn46} GPUS=$NGPUS TRAIN_BS=$TRAIN_BS GLOBAL_BATCH=$GLOBAL_BATCH"
echo "EPOCHS=$TRAIN_EPOCHS ORIGINAL_TRAIN_SAMPLES=50000 TRAIN_SAMPLES=$TRAIN_SAMPLES SEED=$SEED GATE=$CONDITION_GATE SOURCE_SWAP=$SOURCE_SWAP PRIOR_WEIGHT=$PRIOR_WEIGHT"
echo "STEPS_PER_EPOCH=$STEPS_PER_EPOCH TARGET_TRAIN_STEPS=$TARGET_TRAIN_STEPS"
echo "RUN_NAME=$RUN_NAME"
if [[ "$MODE" == --dry-run ]]; then exit 0; fi

module purge
module load singularity/3.8.7

PROJECT_DIR="${PROJECT_DIR:-$HOME/trellis-singularity/work/MorphFlow}"
BENCH_HF_HOME="${BENCH_HF_HOME:-/hpc/scratch/marco.barezzi/morph-benchmark/shared-cache/hf-home}"
BENCH_TORCH_HOME="${BENCH_TORCH_HOME:-/hpc/scratch/marco.barezzi/morph-benchmark/shared-cache/torch-home}"
SIF="${SIF:-$HOME/containers/trellis-py240-cu118.sif}"
TRELLIS_BASE="${TRELLIS_BASE:-$HOME/trellis-singularity}"

FLUX_ROOT_HOST="${FLUX_ROOT_HOST:-/hpc/scratch/marco.barezzi/3d_dataset/flux_outputs}"
FLUX_ROOT_IN="/data/flux_outputs"

DATA_ROOT_HOST="${DATA_ROOT_HOST:-/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v4}"
DATA_ROOT_IN="/data/morphing_dataset_v4"

ARCHIVE_HOST="${ARCHIVE_HOST:-/hpc/archive/G_VBD/marco.barezzi}"
OUT_ROOT_HOST="$ARCHIVE_HOST/morphflow_runs/ablations_ss"
OUT_ROOT_IN="/archive/morphflow_runs/ablations_ss"

# Same v4 SS encoder/LoRA settings; matching and auxiliary losses stay off.
# The experiment switch changes only gate, swap or prior relative to baseline.
SEMANTIC_CYCLE_LOSS_WEIGHT="0.0"
SEMANTIC_CYCLE_LOSS_PROB="0.0"
SEMANTIC_USAGE_LOSS_WEIGHT="0.0"
SEMANTIC_USAGE_CAP="${SEMANTIC_USAGE_CAP:-4.0}"
# Frozen native TRELLIS SS prior on differentiable student rollouts.
# Weight ramps linearly over optimizer steps; cadence counts training microbatches.
# Only prior_only enables the extra frozen model and rollout.
TRELLIS_PRIOR_WEIGHT="$PRIOR_WEIGHT"
TRELLIS_PRIOR_EVERY="4"
TRELLIS_PRIOR_WARMUP_STEPS="10000"
TRELLIS_PRIOR_ROLLOUT_STEPS="25"
TRELLIS_PRIOR_GRAD_STEPS="4"
TRELLIS_PRIOR_MAX_ITEMS="1"
TRELLIS_PRIOR_CHECKPOINT="1"
TRELLIS_PRIOR_T_MIN="0.05"
TRELLIS_PRIOR_T_MAX="0.20"
TRELLIS_PRIOR_PROJECTION_CLIP_RATIO="0.10"
TRELLIS_PRIOR_RMS_GUARD_WEIGHT="1.0"
TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO="0.25"
TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO="2.0"
RUN_DIR_HOST="$OUT_ROOT_HOST/$RUN_NAME"
LOG_DIR_HOST="$RUN_DIR_HOST/logs"

mkdir -p "$LOG_DIR_HOST" "$RUN_DIR_HOST/checkpoints" "$TRELLIS_BASE/hf-home" "$TRELLIS_BASE/tmp"

if [ ! -d "$DATA_ROOT_HOST" ]; then
  echo "ERROR: dataset not found: $DATA_ROOT_HOST"
  exit 1
fi

python3 - "$DATA_ROOT_HOST" <<'PY_V4_DATA'
import json, os, sys
root = sys.argv[1]
with open(os.path.join(root, 'run_state.json')) as f:
    state = json.load(f)
if state.get('phase') != 'complete' or state.get('status') != 'complete':
    raise SystemExit('Dataset v4 non completo: %s' % state)
for split, count in [('train', 50000), ('val', 2000), ('test', 2000)]:
    with open(os.path.join(root, 'metadata_' + split + '.json')) as f:
        data = json.load(f)
    if len(data) != count:
        raise SystemExit('Dataset v4 %s: %d != %d' % (split, len(data), count))
print('Dataset v4 completo: train=50000 val=2000 test=2000')
PY_V4_DATA

# A run can resume only its own checkpoints and its recorded configuration.
exec 9>"$RUN_DIR_HOST/.training.lock"
flock -n 9 || die "Another job is already training $RUN_NAME"
AUTO_RESUME="${AUTO_RESUME:-1}"
[[ "$AUTO_RESUME" == 0 || "$AUTO_RESUME" == 1 ]] || die "AUTO_RESUME must be 0 or 1."
RESUME_FROM="${RESUME_FROM:-}"
if [[ -z "$RESUME_FROM" && "$AUTO_RESUME" == 1 && -f "$RUN_DIR_HOST/checkpoints/morphflow_last.pt" ]]; then
  RESUME_FROM="$RUN_DIR_HOST/checkpoints/morphflow_last.pt"
fi
if [[ -z "$RESUME_FROM" && -f "$RUN_DIR_HOST/checkpoints/morphflow_last.pt" ]]; then
  die "Existing checkpoint: use AUTO_RESUME=1 or choose a new RUN_NAME."
fi
RESUME_FROM_IN=""
if [[ -n "$RESUME_FROM" ]]; then
  RESUME_FROM="$(realpath -e "$RESUME_FROM")"
  resolved_run="$(realpath -e "$RUN_DIR_HOST")"
  [[ "$RESUME_FROM" == "$resolved_run/checkpoints/"* ]] || die "Resume only checkpoints belonging to this ablation; choose a fresh RUN_NAME to train from TRELLIS."
  RESUME_FROM_IN="$OUT_ROOT_IN/$RUN_NAME/${RESUME_FROM#"$resolved_run/"}"
fi
python3 - "$RUN_DIR_HOST/ablation_config.json" "$ABLATION" "$CONDITION_GATE" "$SOURCE_SWAP" "$PRIOR_WEIGHT" "$NGPUS" "$TRAIN_BS" "$SEED" "$DATA_ROOT_HOST" "$RESUME_FROM" "$TRAIN_EPOCHS" "$TARGET_TRAIN_STEPS" "$STEPS_PER_EPOCH" <<'PY_CONFIG'
import json
from pathlib import Path
import sys
path=Path(sys.argv[1])
keys=('experiment','condition_gate','source_swap','prior_weight','num_gpus','batch_per_gpu','seed','dataset_root')
configuration=dict(zip(keys,sys.argv[2:10]))
configuration.update(epochs=int(sys.argv[11]),original_train_samples=50000,
                     train_samples=50000*(1+int(sys.argv[4])),
                     target_train_steps=int(sys.argv[12]),steps_per_epoch=int(sys.argv[13]),
                     learning_rate=1e-4,warmup_steps=10000,global_batch=int(sys.argv[6])*int(sys.argv[7]))
if path.exists():
    previous=json.loads(path.read_text())
    if previous != configuration:
        raise SystemExit('Configuration changed for an existing ablation: choose a new RUN_NAME.\nPrevious: %s\nRequested: %s' % (previous,configuration))
else:
    if sys.argv[10]:
        raise SystemExit('Cannot resume a checkpoint without its ablation_config.json.')
    path.write_text(json.dumps(configuration,indent=2)+'\n')
print('Ablation configuration:',configuration)
PY_CONFIG

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
export SINGULARITYENV_RESUME_FROM_IN="$RESUME_FROM_IN"
export SINGULARITYENV_CONDITION_GATE="$CONDITION_GATE"
export SINGULARITYENV_SOURCE_SWAP="$SOURCE_SWAP"
export SINGULARITYENV_TRAIN_ALPHA_GATE="$TRAIN_ALPHA_GATE"
export SINGULARITYENV_TRAIN_BS="$TRAIN_BS"
export SINGULARITYENV_TRAIN_EPOCHS="$TRAIN_EPOCHS"
export SINGULARITYENV_TRAIN_SAMPLES="$TRAIN_SAMPLES"
export SINGULARITYENV_TARGET_TRAIN_STEPS="$TARGET_TRAIN_STEPS"
export SINGULARITYENV_SEED="$SEED"
export SINGULARITYENV_NGPUS="$NGPUS"
export SINGULARITYENV_NUM_WORKERS="$NUM_WORKERS"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export SINGULARITYENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export SINGULARITYENV_MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((15000 + SLURM_JOB_ID % 40000))}"
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
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1

[[ -r "$SIF" && -x "$TRELLIS_BASE/env/bin/python" ]] || die "Missing training container or /opt/trellis/env."
[[ -d "$BENCH_HF_HOME" && -d "$BENCH_TORCH_HOME" ]] || die "Missing shared benchmark caches."
flux_binds=()
if [[ "$ABLATION" == prior_only ]]; then
  [[ -d "$FLUX_ROOT_HOST" ]] || die "Prior requires source images: $FLUX_ROOT_HOST"
  test -s "$BENCH_TORCH_HOME/hub/facebookresearch_dinov2_main/hubconf.py" || die "DINOv2 cache incomplete."
  flux_binds=(--bind "$FLUX_ROOT_HOST:$FLUX_ROOT_IN:ro")
fi
singularity exec --cleanenv --nv \
  --bind "$TRELLIS_BASE:/opt/trellis" \
  --bind "$BENCH_HF_HOME:/opt/trellis/hf-home:rw" \
  --bind "$BENCH_TORCH_HOME:/opt/trellis/torch-home:rw" \
  --bind "$DATA_ROOT_HOST:$DATA_ROOT_IN:ro" \
  --bind "$ARCHIVE_HOST:/archive" \
  --bind "$HOME:$HOME" \
  ${flux_binds[@]+"${flux_binds[@]}"} \
  "$SIF" \
  bash -s <<'IN_CONTAINER'
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/trellis/env

export TORCH_HOME=/opt/trellis/torch-home
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
python - "$NGPUS" <<'PY_GPU'
import sys, torch
expected=int(sys.argv[1])
assert torch.cuda.device_count()==expected, (torch.cuda.device_count(),expected)
for index in range(expected):
    with torch.cuda.device(index):
        assert torch.cuda.is_bf16_supported(), 'BF16 is required; allocate homogeneous A100/L40S GPUs.'
    print('Allocated GPU',index,torch.cuda.get_device_name(index))
print('torch=',torch.__version__,'cuda=',torch.version.cuda)
PY_GPU
nvidia-smi -L || true

# Check the shared prior cache before launching the allocated training processes.
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

# DINO is used only by prior_only. The other three experiments never load it.
if [[ "$TRELLIS_PRIOR_WEIGHT" != 0.0 ]]; then
  python - <<'PY_DINO'
import torch
encoder=torch.hub.load('facebookresearch/dinov2:main','dinov2_vitl14_reg',pretrained=True,trust_repo=True)
del encoder
print('DINOv2 cache ready for the frozen prior.',flush=True)
PY_DINO
fi
resume_args=()
if [[ -n "$RESUME_FROM_IN" ]]; then
  resume_args=(--resume_from "$RESUME_FROM_IN" --resume_strict 1 --resume_optimizer 1)
fi
launch_args=(--num_processes "$NGPUS" --num_machines 1 --mixed_precision bf16 --dynamo_backend no --main_process_port "$MAIN_PROCESS_PORT")
if (( NGPUS > 1 )); then launch_args+=(--multi_gpu); fi

accelerate launch "${launch_args[@]}" \
  main.py \
  --root_dir "$DATA_ROOT_IN" \
  --metadata metadata_train.json \
  --val_metadata metadata_val.json \
  --out_dir "$OUT_ROOT_IN" \
  --run_name "$RUN_NAME" \
  --train_bs "$TRAIN_BS" \
  --seed "$SEED" \
  --val_bs 1 \
  --train_epochs "$TRAIN_EPOCHS" \
  --expected_train_samples "$TRAIN_SAMPLES" \
  --expected_train_steps "$TARGET_TRAIN_STEPS" \
  --lr 1e-4 \
  --cond_lr 1e-4 \
  --flow_lr 1e-4 \
  --lora_lr 1e-4 \
  --weight_decay 5e-5 \
  --warmup_steps 10000 \
  --plateau_factor 0.5 \
  --plateau_patience 5 \
  --grad_clip 1.0 \
  --log_every 20 \
  --val_every 1 \
  --val_max_items 0 \
  --num_workers "$NUM_WORKERS" \
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
  --cfg_drop_prob 0.0 \
  --trellis_model image_large \
  --flow_target ss \
  --teacher_mid_weight 1.0 \
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
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.1 \
  --lora_attention_scope cross \
  --lora_target_modules to_q,to_kv \
  --cond_resample_depth 1 \
  --cond_resample_heads 8 \
  --ss_flow_arch standard \
  --endpoint_loss_weight 0.0 \
  --symmetry_loss_weight 0.0 \
  --endpoint_loss_prob 0.0 \
  --symmetry_loss_prob 0.0 \
  --use_semantic_token_matching 0 \
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
  --semantic_match_log_stats 0 \
  --separate_cond 1 \
  --separate_cond_gate "$CONDITION_GATE" \
  --t_schedule logit_normal \
  --t_logit_mean 1.0 \
  --t_logit_std 1.0 \
  --trainable_scope full \
  --freeze_modules none \
  --use_lora 1 \
  --train_flow_alpha_gate "$TRAIN_ALPHA_GATE" \
  --train_flow_alpha_embedder 0 \
  --use_ema 0 \
  --mixed_precision bf16 \
  --allow_tf32 1 \
  --use_checkpoint 0 \
  --augment_source_swap "$SOURCE_SWAP" \
  --checkpoint_every 2 \
  ${resume_args[@]+"${resume_args[@]}"}

echo "RUN_DIR=$OUT_ROOT_IN/$RUN_NAME"
IN_CONTAINER
