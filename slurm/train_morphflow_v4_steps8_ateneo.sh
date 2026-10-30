#!/bin/bash
# Shared SS/SLat runtime for the v4 eight-stage dataset; submit the wrappers.
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }
FLOW_TARGET="${1:?Expected ss or slat}"
shift
MODE="${1:---run}"
[[ $# -le 1 ]] || die 'Use --dry-run or configure the launcher with environment variables.'
[[ "$MODE" == --run || "$MODE" == --dry-run ]] || die "Unknown mode: $MODE"
case "$FLOW_TARGET" in
  ss) DEFAULT_GPUS=8; DEFAULT_EPOCHS=40; DEFAULT_LR=1e-4; DEFAULT_RUN=v4_steps8_ss_lora_all_bs1_lr1e4_e40 ;;
  slat) DEFAULT_GPUS=2; DEFAULT_EPOCHS=20; DEFAULT_LR=5e-5; DEFAULT_RUN=v4_steps8_slat_lora_all_bs1_lr5e5_e20 ;;
  *) die "Unknown flow target: $FLOW_TARGET" ;;
esac
PROJECT_DIR="${PROJECT_DIR:-$HOME/trellis-singularity/work/MorphFlow}"
NGPUS="${NGPUS:-$DEFAULT_GPUS}"
TRAIN_BS="${TRAIN_BS:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-$DEFAULT_EPOCHS}"
TRAIN_LR="${TRAIN_LR:-$DEFAULT_LR}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_FROM="${RESUME_FROM:-}"
for value in "$NGPUS" "$TRAIN_BS" "$TRAIN_EPOCHS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die 'GPU count, batch and epochs must be positive integers.'
done
[[ "$SEED" =~ ^[0-9]+$ && "$NUM_WORKERS" =~ ^[0-9]+$ ]] || die 'Seed/workers must be nonnegative integers.'
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_.-]+$ && "$RUN_NAME" != . && "$RUN_NAME" != .. ]] || die 'Invalid RUN_NAME.'
[[ "$AUTO_RESUME" == 0 || "$AUTO_RESUME" == 1 ]] || die 'AUTO_RESUME must be 0 or 1.'
python3 - "$TRAIN_LR" <<'PY_LR'
import math, sys
try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit('TRAIN_LR must be a positive finite number')
if not math.isfinite(value) or value <= 0:
    raise SystemExit('TRAIN_LR must be a positive finite number')
PY_LR

DATA_PARENT_HOST="${DATA_PARENT_HOST:-/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v4_steps8_steps2}"
DATA_PARENT_HOST="${DATA_PARENT_HOST%/}"
DATA_ROOT_HOST="${DATA_ROOT_HOST:-$DATA_PARENT_HOST/morphing_dataset_v4_steps8}"
DATA_ROOT_HOST="${DATA_ROOT_HOST%/}"
[[ "$DATA_ROOT_HOST" == "$DATA_PARENT_HOST/morphing_dataset_v4_steps8" ]] || die 'DATA_ROOT_HOST must select morphing_dataset_v4_steps8 inside DATA_PARENT_HOST.'
DATA_PARENT_IN=/data/morphing_dataset_v4_steps8_steps2
DATA_ROOT_IN="$DATA_PARENT_IN/morphing_dataset_v4_steps8"
ARCHIVE_HOST="${ARCHIVE_HOST:-/hpc/archive/G_VBD/marco.barezzi}"
OUT_ROOT_HOST="$ARCHIVE_HOST/morphflow_runs"
OUT_ROOT_IN=/archive/morphflow_runs
RUN_DIR_HOST="$OUT_ROOT_HOST/$RUN_NAME"
FLUX_ROOT_HOST="${FLUX_ROOT_HOST:-/hpc/scratch/marco.barezzi/3d_dataset/flux_outputs}"
FLUX_ROOT_IN=/data/flux_outputs
BENCH_HF_HOME="${BENCH_HF_HOME:-/hpc/scratch/marco.barezzi/morph-benchmark/shared-cache/hf-home}"
BENCH_TORCH_HOME="${BENCH_TORCH_HOME:-/hpc/scratch/marco.barezzi/morph-benchmark/shared-cache/torch-home}"
SIF="${SIF:-$HOME/containers/trellis-py240-cu118.sif}"
TRELLIS_BASE="${TRELLIS_BASE:-$HOME/trellis-singularity}"
GLOBAL_BATCH=$((NGPUS * TRAIN_BS))
TRAIN_SAMPLES=100000
STEPS_PER_EPOCH=$(((TRAIN_SAMPLES + GLOBAL_BATCH - 1) / GLOBAL_BATCH))
OPTIMIZER_STEPS=$((TRAIN_EPOCHS * STEPS_PER_EPOCH))

echo "FLOW_TARGET=$FLOW_TARGET NGPUS=$NGPUS TRAIN_BS=$TRAIN_BS GLOBAL_BATCH=$GLOBAL_BATCH"
echo "TRAIN_EPOCHS=$TRAIN_EPOCHS TRAIN_LR=$TRAIN_LR TRAIN_SAMPLES=$TRAIN_SAMPLES SEED=$SEED"
echo "STEPS_PER_EPOCH=$STEPS_PER_EPOCH OPTIMIZER_STEPS=$OPTIMIZER_STEPS (informational; no step limit)"
echo "RUN_NAME=$RUN_NAME"
echo "DATA_PARENT_HOST=$DATA_PARENT_HOST"
echo "DATA_ROOT_HOST=$DATA_ROOT_HOST"
echo "DATA_ROOT_IN=$DATA_ROOT_IN"
echo 'LORA_ATTENTION_SCOPE=all LORA_TARGET_MODULES=to_qkv,to_q,to_kv GATE=token SOURCE_SWAP=1 PRIOR_WEIGHT=0.1'

# Build the model arguments once; dry-run prints the same arguments passed
# through the container boundary. All learning-rate groups share TRAIN_LR.
TRAIN_ARGS=(
  --root_dir "$DATA_ROOT_IN"
  --metadata metadata_train.json --val_metadata metadata_val.json
  --out_dir "$OUT_ROOT_IN" --run_name "$RUN_NAME"
  --train_bs "$TRAIN_BS" --val_bs 1 --train_epochs "$TRAIN_EPOCHS"
  --seed "$SEED" --expected_train_samples "$TRAIN_SAMPLES"
  --lr "$TRAIN_LR" --cond_lr "$TRAIN_LR" --flow_lr "$TRAIN_LR" --lora_lr "$TRAIN_LR"
  --weight_decay 5e-5 --warmup_steps 10000
  --plateau_factor 0.5 --plateau_patience 5 --grad_clip 1.0
  --log_every 20 --val_every 1 --val_max_items 0 --num_workers "$NUM_WORKERS"
  --lr_scheduler cosine
  --cond_token_norm none --cond_proj_norm layernorm --cond_input_norm trellis
  --cond_encoder_type sparse_conv3d --cond_style_tokens 8
  --cond_use_occupancy 1 --cond_hybrid_pool_stats 1
  --cond_residual_blocks_64 2 --cond_residual_blocks_32 2 --cond_residual_blocks_16 2
  --cfg_drop_prob 0.0 --trellis_model image_large --flow_target "$FLOW_TARGET"
  --teacher_mid_weight 1.0
  --trellis_prior_weight 0.1 --trellis_prior_every 4 --trellis_prior_warmup_steps 10000
  --trellis_prior_rollout_steps 25 --trellis_prior_grad_steps 4
  --trellis_prior_max_items 1 --trellis_prior_checkpoint 1
  --trellis_prior_t_min 0.05 --trellis_prior_t_max 0.20
  --trellis_prior_projection_clip_ratio 0.10
  --trellis_prior_rms_guard_weight 1.0 --trellis_prior_rms_guard_low_ratio 0.25
  --trellis_prior_rms_guard_high_ratio 2.0
  --slat_condition_source slat --source_images_root "$FLUX_ROOT_IN"
  --dino_model dinov2_vitl14_reg
  --lora_rank 32 --lora_alpha 64 --lora_dropout 0.1
  --lora_attention_scope all --lora_target_modules to_qkv,to_q,to_kv
  --cond_resample_depth 1 --cond_resample_heads 8 --ss_flow_arch standard
  --endpoint_loss_weight 0.0 --symmetry_loss_weight 0.0
  --endpoint_loss_prob 0.0 --symmetry_loss_prob 0.0
  --use_semantic_token_matching 0
  --semantic_match_max_align 0.35 --semantic_match_temperature 0.05 --semantic_match_dim 128
  --semantic_match_alpha_weight 1 --semantic_match_detach_scores 0
  --semantic_match_exclude_style_tokens 1
  --semantic_cycle_loss_weight 0.0 --semantic_cycle_loss_prob 0.0
  --semantic_cycle_detach_targets 1 --semantic_cycle_alpha_weight 1
  --semantic_usage_loss_weight 0.0 --semantic_usage_cap 4.0 --semantic_match_log_stats 0
  --separate_cond 1 --separate_cond_gate token
  --t_schedule logit_normal --t_logit_mean 1.0 --t_logit_std 1.0
  --trainable_scope full --freeze_modules none --use_lora 1
  --train_flow_alpha_gate 1 --train_flow_alpha_embedder 0
  --use_ema 0 --mixed_precision bf16 --allow_tf32 1 --use_checkpoint 0
  --augment_source_swap 1 --checkpoint_every 2
)
LAUNCH_ARGS=(--num_processes "$NGPUS" --num_machines 1 --mixed_precision bf16 --dynamo_backend no)
if (( NGPUS > 1 )); then LAUNCH_ARGS+=(--multi-gpu); fi
if [[ "$MODE" == --dry-run ]]; then
  printf 'TRAIN_COMMAND='
  printf '%q ' accelerate launch "${LAUNCH_ARGS[@]}" main.py "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || die 'Submit the SS or SLat .slurm wrapper with sbatch.'
[[ "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || die 'Use one node per training job.'
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != NoDevFiles ]] || die 'Slurm did not allocate visible CUDA devices.'
IFS=',' read -r -a allocated_gpus <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#allocated_gpus[@]}" == "$NGPUS" ]] || die 'NGPUS differs from CUDA_VISIBLE_DEVICES; match the wrapper and Slurm allocation.'
if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then
  [[ "$NGPUS" == "$SLURM_GPUS_ON_NODE" ]] || die 'NGPUS differs from SLURM_GPUS_ON_NODE.'
fi

module purge
module load singularity/3.8.7
[[ -f "$SIF" ]] || die "Container not found: $SIF"
for directory in "$PROJECT_DIR" "$TRELLIS_BASE/env" "$DATA_PARENT_HOST" "$DATA_ROOT_HOST" "$FLUX_ROOT_HOST" "$BENCH_HF_HOME" "$BENCH_TORCH_HOME"; do
  [[ -d "$directory" ]] || die "Missing directory: $directory"
done
[[ -s "$BENCH_TORCH_HOME/hub/facebookresearch_dinov2_main/hubconf.py" ]] || die 'DINOv2 source cache is incomplete.'

# Refuse an unfinished generation and catch accidental selection of v4 K=4/2.
python3 - "$DATA_ROOT_HOST" <<'PY_DATASET'
import json, os, sys
root = sys.argv[1]
def read(name):
    with open(os.path.join(root, name)) as handle:
        return json.load(handle)
state, plan = read('run_state.json'), read('plan.json')
if state.get('phase') != 'complete' or state.get('status') != 'complete':
    raise SystemExit('Eight-stage dataset generation is incomplete: %s' % state)
if plan.get('k') != 8:
    raise SystemExit('Expected eight alpha stages per pair, got %r' % plan.get('k'))
for split, count in (('train', 50000), ('val', 2000), ('test', 2000)):
    rows = read('metadata_' + split + '.json')
    if len(rows) != count:
        raise SystemExit('Unexpected %s sample count: %d != %d' % (split, len(rows), count))
print('Eight-stage dataset complete: train=50000 (100000 with swap), val=2000, test=2000')
PY_DATASET

mkdir -p "$RUN_DIR_HOST/logs" "$RUN_DIR_HOST/checkpoints"
exec 9>"$RUN_DIR_HOST/.training.lock"
flock -n 9 || die "Another job is training $RUN_NAME"
# An interrupted run can resume only the same training configuration.
export FLOW_TARGET NGPUS TRAIN_BS TRAIN_EPOCHS TRAIN_LR SEED RUN_NAME DATA_ROOT_HOST
python3 - "$RUN_DIR_HOST/run_config.json" <<'PY_CONFIG'
import json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
keys = ('FLOW_TARGET', 'NGPUS', 'TRAIN_BS', 'TRAIN_EPOCHS', 'TRAIN_LR', 'SEED', 'RUN_NAME', 'DATA_ROOT_HOST')
config = {key: os.environ[key] for key in keys}
config.update({'lora_attention_scope': 'all', 'lora_target_modules': 'to_qkv,to_q,to_kv',
               'lora_rank': 32, 'lora_alpha': 64, 'source_swap': 1, 'condition_gate': 'token',
               'prior_weight': 0.1, 'warmup_steps': 10000})
if path.exists():
    with path.open() as handle:
        if json.load(handle) != config:
            raise SystemExit('Run configuration changed: choose a new RUN_NAME')
else:
    if any((path.parent / 'checkpoints').glob('morphflow_*.pt')):
        raise SystemExit('Existing checkpoints have no run_config.json: choose a new RUN_NAME')
    temporary = path.with_name('.run_config.%d.tmp' % os.getpid())
    with temporary.open('w') as handle:
        json.dump(config, handle, indent=2)
    os.replace(str(temporary), str(path))
PY_CONFIG
if [[ "$AUTO_RESUME" == 1 && -z "$RESUME_FROM" ]]; then
  if [[ -f "$RUN_DIR_HOST/checkpoints/morphflow_last.pt" ]]; then
    RESUME_FROM="$RUN_DIR_HOST/checkpoints/morphflow_last.pt"
  else
    RESUME_FROM="$(python3 - "$RUN_DIR_HOST/checkpoints" <<'PY_RESUME'
from pathlib import Path
import sys
root = Path(sys.argv[1])
paths = [path for path in root.glob('morphflow_*.pt') if path.is_file()]
print(str(max(paths, key=lambda p: p.stat().st_mtime)) if paths else '')
PY_RESUME
)"
  fi
fi
if [[ -z "$RESUME_FROM" && "$AUTO_RESUME" == 0 ]]; then
  existing_checkpoint="$(find "$RUN_DIR_HOST/checkpoints" -maxdepth 1 -name 'morphflow_*.pt' -print -quit)"
  [[ -z "$existing_checkpoint" ]] || die 'Existing checkpoints: enable AUTO_RESUME or choose a new RUN_NAME.'
fi
if [[ -n "$RESUME_FROM" ]]; then
  [[ -f "$RESUME_FROM" ]] || die 'RESUME_FROM must be an existing checkpoint.'
  RESUME_FROM="$(realpath -e "$RESUME_FROM")"
  resolved_checkpoints="$(realpath -e "$RUN_DIR_HOST/checkpoints")"
  [[ "$RESUME_FROM" == "$resolved_checkpoints/"* ]] || die 'Resume only checkpoints from this run; choose a new RUN_NAME for a fresh training.'
  [[ "$RESUME_FROM" == "$ARCHIVE_HOST/"* ]] || die 'RESUME_FROM must be below ARCHIVE_HOST.'
  RESUME_IN="/archive${RESUME_FROM#"$ARCHIVE_HOST"}"
  TRAIN_ARGS+=(--resume_from "$RESUME_IN" --resume_strict 1 --resume_optimizer 1)
fi
exec > >(tee -a "$RUN_DIR_HOST/logs/train.log") 2>&1
echo "JOB=${SLURM_JOB_ID} NODE=${SLURM_JOB_NODELIST:-unknown} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "RUN_NAME=$RUN_NAME FLOW_TARGET=$FLOW_TARGET GPUS=$NGPUS GLOBAL_BATCH=$GLOBAL_BATCH LR=$TRAIN_LR EPOCHS=$TRAIN_EPOCHS"
echo "RESUME_FROM=${RESUME_FROM:-<none>}"

export SINGULARITYENV_PROJECT_DIR="$PROJECT_DIR"
export SINGULARITYENV_DATA_ROOT_IN="$DATA_ROOT_IN"
export SINGULARITYENV_FLUX_ROOT_IN="$FLUX_ROOT_IN"
export SINGULARITYENV_OUT_ROOT_IN="$OUT_ROOT_IN"
export SINGULARITYENV_RUN_NAME="$RUN_NAME"
export SINGULARITYENV_FLOW_TARGET="$FLOW_TARGET"
export SINGULARITYENV_NGPUS="$NGPUS"
export SINGULARITYENV_TRAIN_BS="$TRAIN_BS"
export SINGULARITYENV_TRAIN_EPOCHS="$TRAIN_EPOCHS"
export SINGULARITYENV_TRAIN_LR="$TRAIN_LR"
export SINGULARITYENV_SEED="$SEED"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export SINGULARITYENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1

singularity exec --cleanenv --nv \
  --bind "$TRELLIS_BASE:/opt/trellis" \
  --bind "$BENCH_HF_HOME:/opt/trellis/hf-home:rw" \
  --bind "$BENCH_TORCH_HOME:/opt/trellis/torch-home:rw" \
  --bind "$DATA_PARENT_HOST:$DATA_PARENT_IN:ro" \
  --bind "$ARCHIVE_HOST:/archive" \
  --bind "$HOME:$HOME" \
  --bind "$FLUX_ROOT_HOST:$FLUX_ROOT_IN:ro" \
  "$SIF" bash -s -- "${TRAIN_ARGS[@]}" <<'IN_CONTAINER'
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/trellis/env
export TORCH_HOME=/opt/trellis/torch-home
export HF_HOME=/opt/trellis/hf-home
export TMPDIR="/tmp/mf_steps8_${SLURM_JOB_ID}"
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=2
mkdir -p "$TMPDIR" "$OUT_ROOT_IN/$RUN_NAME"
cd "$PROJECT_DIR"
python - "$NGPUS" <<'PY_GPU'
import sys, torch
expected = int(sys.argv[1])
if not torch.cuda.is_available() or torch.cuda.device_count() != expected:
    raise SystemExit('Expected %d allocated GPUs, got %d' % (expected, torch.cuda.device_count()))
print('torch=', torch.__version__, 'cuda=', torch.version.cuda,
      'GPUs=', torch.cuda.device_count(), 'bf16=', torch.cuda.is_bf16_supported(), flush=True)
PY_GPU
nvidia-smi -L || true
# Two node jobs can warm their shared caches simultaneously: serialize this
# short check and use the existing local DINO source checkout.
(
  flock -x 8
  python - "$FLOW_TARGET" <<'PY_PRIOR'
from pathlib import Path
import sys
import torch
from huggingface_hub import hf_hub_download
kind = sys.argv[1]
prefix = ('ckpts/ss_flow_img_dit_L_16l8_fp16' if kind == 'ss'
          else 'ckpts/slat_flow_img_dit_L_64l8p2_fp16')
for extension in ('json', 'safetensors'):
    path = hf_hub_download(repo_id='microsoft/TRELLIS-image-large',
                           filename=prefix + '.' + extension, local_files_only=True)
    print('TRELLIS %s prior cache: %s' % (kind, path), flush=True)
dino_repo = Path(torch.hub.get_dir()) / 'facebookresearch_dinov2_main'
if not (dino_repo / 'hubconf.py').is_file():
    raise SystemExit('DINOv2 source cache is incomplete')
encoder = torch.hub.load(str(dino_repo), 'dinov2_vitl14_reg', source='local', pretrained=True)
del encoder
print('Prior cache verified for %s; starting training.' % kind, flush=True)
PY_PRIOR
) 8>"$TORCH_HOME/.train_v4_steps8_prior_cache.lock"
launch_args=(--num_processes "$NGPUS" --num_machines 1 --mixed_precision bf16 --dynamo_backend no)
if (( NGPUS > 1 )); then launch_args+=(--multi-gpu); fi
accelerate launch "${launch_args[@]}" main.py "$@"
echo "RUN_DIR=$OUT_ROOT_IN/$RUN_NAME"
IN_CONTAINER
