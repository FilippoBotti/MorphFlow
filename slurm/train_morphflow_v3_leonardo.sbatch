#!/bin/bash
#SBATCH --job-name=v3_slat_ablation_encDual_lora
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=ALL

# Un launcher Accelerate per nodo; ogni launcher crea 4 processi GPU.
# Uso e variabili modificabili: slurm/README_leonardo.md.
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }

MODE="${1:---train}"
case "$MODE" in
    --train|--check|--prepare|--comm-check) ;;
    *) die "Uso: $0 [--train|--check|--prepare|--comm-check]" ;;
esac
[[ $# -le 1 ]] || die "Usare le variabili ambiente per modificare i parametri."
if [[ "$MODE" == --train || "$MODE" == --comm-check ]]; then
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "Lanciare con sbatch; --check e --prepare funzionano dal login node."
fi

export MF_ENV_FILE="${MF_ENV_FILE:-/leonardo_work/IscrC_MORPHFL/mbarezzi/env.sh}"
# env.sh usa WORK e FAST; fissiamo i default del progetto anche nei job batch.
export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
[[ -r "$MF_ENV_FILE" ]] || die "Env non trovato: $MF_ENV_FILE"
source "$MF_ENV_FILE"

export MF_PROJECT_DIR="${MF_PROJECT_DIR:-$MORPHFLOW_REPO}"
export MF_DATA_ROOT="${MF_DATA_ROOT:-$FAST/$USER/datasets/morphing_dataset_v3}"
export MF_OUT_ROOT="${MF_OUT_ROOT:-$WORK/$USER/outputs}"
export MF_SOURCE_IMAGES_ROOT="${MF_SOURCE_IMAGES_ROOT:-$FAST/$USER/datasets/flux_outputs}"
FLOW_TARGET="${FLOW_TARGET:-slat}"
case "$FLOW_TARGET" in
    ss)
        DEFAULT_RUN_NAME=ss_ablation_encDual_lora_priorDual_s24_g6
        DEFAULT_INIT_FROM="$WORK/$USER/checkpoints/ss_ablation_encDual_lora/morphflow_best.pt"
        DEFAULT_TRAIN_EPOCHS=5
        DEFAULT_TRAIN_BS=1
        DEFAULT_WARMUP_STEPS=500
        DEFAULT_WEIGHT_DECAY=1e-4
        DEFAULT_GRAD_BALANCE=1
        DEFAULT_CHECKPOINT_EVERY=1
        DEFAULT_CHECKPOINT_HALF_EPOCH=1
        DEFAULT_TRELLIS_PRIOR_WEIGHT=1.01
        ;;
    slat)
        DEFAULT_RUN_NAME=slat_ablation_encDual_lora
        DEFAULT_INIT_FROM=""
        DEFAULT_TRAIN_EPOCHS=25
        DEFAULT_TRAIN_BS=2
        DEFAULT_WARMUP_STEPS=10000
        DEFAULT_WEIGHT_DECAY=5e-5
        DEFAULT_GRAD_BALANCE=0
        DEFAULT_CHECKPOINT_EVERY=2
        DEFAULT_CHECKPOINT_HALF_EPOCH=0
        DEFAULT_TRELLIS_PRIOR_WEIGHT=0.0
        ;;
    *) die "FLOW_TARGET deve essere ss o slat." ;;
esac
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
[[ "$RUN_NAME" != */* && "$RUN_NAME" != . && "$RUN_NAME" != .. ]] || die "RUN_NAME deve essere un nome, non un percorso."
RUN_DIR="$MF_OUT_ROOT/$RUN_NAME"
MF_CHECKPOINT_DIR="${MF_CHECKPOINT_DIR:-$WORK/$USER/checkpoints/$RUN_NAME}"
AUTO_RESUME="${AUTO_RESUME:-0}"
RESUME_FROM="${RESUME_FROM:-}"
# An explicit empty INIT_FROM disables the preset (also needed for resume).
DEFAULT_RESET_PHASE=1
if [[ -n "$RESUME_FROM" || "$AUTO_RESUME" == 1 ]]; then
    DEFAULT_INIT_FROM=""
    DEFAULT_RESET_PHASE=0
fi
INIT_FROM="${INIT_FROM-$DEFAULT_INIT_FROM}"
[[ -z "$RESUME_FROM" || -z "$INIT_FROM" ]] || die "Usare RESUME_FROM oppure INIT_FROM, non entrambi."
[[ "$AUTO_RESUME" == 0 || "$AUTO_RESUME" == 1 ]] || die "AUTO_RESUME deve essere 0 o 1."

# SLat mirrors train_morphflow_v3.slurm: baseline encDual, matching/prior off.
# SS starts a new five-epoch prior phase from the imported ablation best.
# Two nodes x four GPUs: global batch 16 for SLat, 8 for SS.
TRAIN_BS="${TRAIN_BS:-$DEFAULT_TRAIN_BS}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-$DEFAULT_TRAIN_EPOCHS}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# Prior TRELLIS locale disponibile per SS e SLat.
TRELLIS_PRIOR_WEIGHT="${TRELLIS_PRIOR_WEIGHT:-$DEFAULT_TRELLIS_PRIOR_WEIGHT}"
TRELLIS_PRIOR_EVERY="${TRELLIS_PRIOR_EVERY:-4}"
TRELLIS_PRIOR_WARMUP_STEPS="${TRELLIS_PRIOR_WARMUP_STEPS:-0}"
TRELLIS_PRIOR_ROLLOUT_STEPS="${TRELLIS_PRIOR_ROLLOUT_STEPS:-24}"
TRELLIS_PRIOR_GRAD_STEPS="${TRELLIS_PRIOR_GRAD_STEPS:-6}"
TRELLIS_PRIOR_MAX_ITEMS="${TRELLIS_PRIOR_MAX_ITEMS:-1}"
TRELLIS_PRIOR_CHECKPOINT="${TRELLIS_PRIOR_CHECKPOINT:-1}"
TRELLIS_PRIOR_T_MIN="${TRELLIS_PRIOR_T_MIN:-0.05}"
TRELLIS_PRIOR_T_MAX="${TRELLIS_PRIOR_T_MAX:-0.20}"
TRELLIS_PRIOR_PROJECTION_CLIP_RATIO="${TRELLIS_PRIOR_PROJECTION_CLIP_RATIO:-0.08}"
TRELLIS_PRIOR_RMS_GUARD_WEIGHT="${TRELLIS_PRIOR_RMS_GUARD_WEIGHT:-1.0}"
TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO="${TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO:-0.40}"
TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO="${TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO:-2.50}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Su A100 NVLink/InfiniBand restano abilitati, salvo override esplicito.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

cd "$MF_PROJECT_DIR"
[[ -f main.py ]] || die "main.py non trovato in $MF_PROJECT_DIR"
mkdir -p "$WORK/$USER/logs" "$MF_OUT_ROOT" "$HF_HOME"

if [[ "$MODE" != --comm-check ]]; then
    for item in metadata_train.json metadata_val.json; do
        [[ -r "$MF_DATA_ROOT/$item" ]] || die "Metadata non trovato: $MF_DATA_ROOT/$item"
    done
    for item in assets targets; do
        [[ -d "$MF_DATA_ROOT/$item" ]] || die "Directory non trovata: $MF_DATA_ROOT/$item"
    done

    if [[ "$AUTO_RESUME" == 1 && -z "$RESUME_FROM" && -z "$INIT_FROM" ]]; then
        RESUME_FROM="$(python - "$MF_CHECKPOINT_DIR" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
last = root / "morphflow_last.pt"
if last.is_file():
    print(last)
else:
    candidates = [p for pattern in ("morphflow_epoch_*.pt", "morphflow_best.pt", "morphflow_final_*.pt")
                  for p in root.glob(pattern) if p.is_file()]
    if candidates:
        print(max(candidates, key=lambda p: p.stat().st_mtime))
PY
        )"
    fi
    if [[ -n "$RESUME_FROM" ]]; then
        [[ -r "$RESUME_FROM" ]] || die "Checkpoint non trovato: $RESUME_FROM (usare un percorso assoluto Leonardo)."
        RESUME_FROM="$(realpath "$RESUME_FROM")"
    elif [[ -n "$INIT_FROM" ]]; then
        [[ -r "$INIT_FROM" ]] || die "Checkpoint INIT_FROM non trovato: $INIT_FROM"
        INIT_FROM="$(realpath "$INIT_FROM")"
    fi
    if [[ -z "$RESUME_FROM" ]] && compgen -G "$MF_CHECKPOINT_DIR/morphflow_*.pt" >/dev/null; then
        die "La destinazione contiene checkpoint: usare AUTO_RESUME=1, RESUME_FROM oppure un nuovo MF_CHECKPOINT_DIR."
    fi

    # --prepare popola la cache dal login; gli altri modi restano offline.
    # Il prior nativo va ricaricato anche quando lo student riparte da checkpoint.
    if [[ "$MODE" == --prepare ]]; then
        export HF_HUB_OFFLINE=0
    fi
    python - "$MODE" "$FLOW_TARGET" "$TRELLIS_PRIOR_WEIGHT" "$RESUME_FROM" <<'PY'
import math
import sys
from huggingface_hub import hf_hub_download
mode, flow_target, weight, resume = sys.argv[1:]
weight = float(weight)
if not math.isfinite(weight) or weight < 0:
    raise SystemExit("TRELLIS_PRIOR_WEIGHT deve essere finito e >= 0.")
filenames = set()
base = (
    "ckpts/ss_flow_img_dit_L_16l8_fp16"
    if flow_target == "ss"
    else "ckpts/slat_flow_img_dit_L_64l8p2_fp16"
)
if not resume:
    filenames.add(base + ".safetensors")
if weight > 0:
    filenames.add(base + ".safetensors")
    filenames.add(base + ".json")
for filename in sorted(filenames):
    try:
        path = hf_hub_download(
            repo_id="microsoft/TRELLIS-image-large",
            filename=filename,
            local_files_only=mode != "--prepare",
        )
    except Exception as exc:
        raise SystemExit(
            f"Pesi TRELLIS non disponibili: {exc}\n"
            f"Dal login node: FLOW_TARGET={flow_target} TRELLIS_PRIOR_WEIGHT={weight} "
            "bash slurm/train_morphflow_v3_leonardo.sbatch --prepare"
        ) from exc
    print(f"TRELLIS weights: {path}")

if weight > 0:
    from pathlib import Path
    from unittest.mock import patch
    import torch

    # Il prior usa DINO anche quando lo student e' condizionato sui latenti SLat.
    # Questi nomi corrispondono al --dino_model fissato nei train_args sotto.
    dino_model = "dinov2_vitl14_reg"
    hub = Path(torch.hub.get_dir())
    required = (
        hub / "facebookresearch_dinov2_main" / "hubconf.py",
        hub / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth",
    )
    prepare_hint = (
        f"FLOW_TARGET={flow_target} TRELLIS_PRIOR_WEIGHT={weight} "
        "bash slurm/train_morphflow_v3_leonardo.sbatch --prepare"
    )
    if mode != "--prepare":
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise SystemExit(
                "Cache DINOv2 incompleta: " + ", ".join(missing)
                + "\nDal login node: " + prepare_hint
            )
    if mode in ("--prepare", "--check"):
        torch.set_num_threads(2)
        def load_dino():
            return torch.hub.load(
                "facebookresearch/dinov2:main", dino_model,
                pretrained=True, trust_repo=True,
            )
        if mode == "--prepare":
            encoder = load_dino()
        else:
            # HF_HUB_OFFLINE non copre Torch Hub: vietare la rete anche qui.
            with patch("torch.hub.urlopen", side_effect=RuntimeError(
                "Download DINOv2 disabilitato in --check; eseguire " + prepare_hint
            )):
                encoder = load_dino()
        del encoder
    print(f"DINOv2 cache: {hub}")
PY
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
train_args=(
    main.py
    --root_dir "$MF_DATA_ROOT"
    --metadata metadata_train.json
    --val_metadata metadata_val.json
    --out_dir "$MF_OUT_ROOT"
    --run_name "$RUN_NAME"
    --checkpoint_dir "$MF_CHECKPOINT_DIR"
    --train_bs "$TRAIN_BS"
    --val_bs 1
    --train_epochs "$TRAIN_EPOCHS"
    --lr 5e-5
    --cond_lr 5e-5
    --flow_lr 5e-5
    --lora_lr 5e-5
    --weight_decay "${WEIGHT_DECAY:-$DEFAULT_WEIGHT_DECAY}"
    --warmup_steps "${WARMUP_STEPS:-$DEFAULT_WARMUP_STEPS}"
    --plateau_factor 0.5
    --plateau_patience 5
    --grad_clip 1.0
    --log_every 20
    --val_every 1
    --val_max_items 0
    --num_workers "$NUM_WORKERS"
    --lr_scheduler cosine
    --cond_token_norm none
    --cond_proj_norm layernorm
    --cond_input_norm trellis
    --cond_encoder_type sparse_conv3d
    --cond_style_tokens 8
    --cond_use_occupancy 1
    --cond_hybrid_pool_stats 1
    --cond_residual_blocks_64 2
    --cond_residual_blocks_32 2
    --cond_residual_blocks_16 2
    --cfg_drop_prob 0.0
    --trellis_model image_large
    --flow_target "$FLOW_TARGET"
    --teacher_mid_weight 1.0
    --trellis_prior_weight "$TRELLIS_PRIOR_WEIGHT"
    --trellis_prior_every "$TRELLIS_PRIOR_EVERY"
    --trellis_prior_warmup_steps "$TRELLIS_PRIOR_WARMUP_STEPS"
    --trellis_prior_rollout_steps "$TRELLIS_PRIOR_ROLLOUT_STEPS"
    --trellis_prior_grad_steps "$TRELLIS_PRIOR_GRAD_STEPS"
    --trellis_prior_max_items "$TRELLIS_PRIOR_MAX_ITEMS"
    --trellis_prior_checkpoint "$TRELLIS_PRIOR_CHECKPOINT"
    --trellis_prior_t_min "$TRELLIS_PRIOR_T_MIN"
    --trellis_prior_t_max "$TRELLIS_PRIOR_T_MAX"
    --trellis_prior_projection_clip_ratio "$TRELLIS_PRIOR_PROJECTION_CLIP_RATIO"
    --trellis_prior_tangent_projection "${TRELLIS_PRIOR_TANGENT_PROJECTION:-0}" \
    --trellis_prior_slat_stat_weight "${TRELLIS_PRIOR_SLAT_STAT_WEIGHT:-1.0}" \
    --trellis_prior_slat_std_low_ratio "${TRELLIS_PRIOR_SLAT_STD_LOW_RATIO:-0.65}" \
    --trellis_prior_slat_std_high_ratio "${TRELLIS_PRIOR_SLAT_STD_HIGH_RATIO:-1.50}" \
    --trellis_prior_slat_mean_tolerance "${TRELLIS_PRIOR_SLAT_MEAN_TOLERANCE:-0.50}" \
    --trellis_prior_rms_guard_weight "$TRELLIS_PRIOR_RMS_GUARD_WEIGHT"
    --trellis_prior_rms_guard_low_ratio "$TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO"
    --trellis_prior_rms_guard_high_ratio "$TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO"
    --trellis_prior_grad_balance "${TRELLIS_PRIOR_GRAD_BALANCE:-$DEFAULT_GRAD_BALANCE}" \
    --trellis_prior_ratio_target "${TRELLIS_PRIOR_RATIO_TARGET:-0.15}" \
    --trellis_prior_ratio_max "${TRELLIS_PRIOR_RATIO_MAX:-0.30}" \
    --trellis_prior_group_ratio_max "${TRELLIS_PRIOR_GROUP_RATIO_MAX:-0.40}" \
    --trellis_prior_scale_ratio_max "${TRELLIS_PRIOR_SCALE_RATIO_MAX:-0.15}" \
    --trellis_prior_guard_ratio_max "${TRELLIS_PRIOR_GUARD_RATIO_MAX:-0.20}" \
    --trellis_prior_balance_ema "${TRELLIS_PRIOR_BALANCE_EMA:-0.99}" \
    --trellis_prior_lambda_min "${TRELLIS_PRIOR_LAMBDA_MIN:-0.01}" \
    --trellis_prior_lambda_max "${TRELLIS_PRIOR_LAMBDA_MAX:-100.0}" \
    --trellis_prior_phase_start_weight "${TRELLIS_PRIOR_PHASE_START_WEIGHT:-0.05}" \
    --trellis_prior_phase_warmup_steps "${TRELLIS_PRIOR_PHASE_WARMUP_STEPS:-500}" \
    --trellis_prior_guard_scale "${TRELLIS_PRIOR_GUARD_SCALE:-0.10}" \
    --trellis_prior_reset_phase "${TRELLIS_PRIOR_RESET_PHASE:-$DEFAULT_RESET_PHASE}" \
    --trellis_prior_weak_patience "${TRELLIS_PRIOR_WEAK_PATIENCE:-25}" \
    --trellis_prior_log_every "${TRELLIS_PRIOR_LOG_EVERY:-20}" \
    --trellis_prior_phase_eval_steps "${TRELLIS_PRIOR_PHASE_EVAL_STEPS:-500,1500,3000}" \
    --slat_condition_source slat
    --source_images_root "$MF_SOURCE_IMAGES_ROOT"
    --dino_model dinov2_vitl14_reg
    --dino_dim 1024
    --lora_rank 32
    --lora_alpha 64
    --lora_dropout 0.1
    --lora_attention_scope cross
    --lora_target_modules to_q,to_kv
    --cond_resample_depth 1
    --cond_resample_heads 8
    --ss_flow_arch standard
    --residual_interp_gate alpha
    --residual_interp_gate_min 1e-3
    --residual_endpoint_prob 0
    --endpoint_loss_weight 0.0
    --symmetry_loss_weight 0.0
    --endpoint_loss_prob 0.0
    --symmetry_loss_prob 0.0
    --use_semantic_token_matching 0
    --semantic_match_max_align 0.35
    --semantic_match_temperature 0.05
    --semantic_match_dim 128
    --semantic_match_alpha_weight 1
    --semantic_match_detach_scores 0
    --semantic_match_exclude_style_tokens 1
    --semantic_cycle_loss_weight 0.0
    --semantic_cycle_loss_prob 0.0
    --semantic_cycle_detach_targets 1
    --semantic_cycle_alpha_weight 1
    --semantic_usage_loss_weight 0.0
    --semantic_usage_cap 4.0
    --semantic_match_log_stats 0
    --separate_cond 1
    --separate_cond_gate token
    --t_schedule logit_normal
    --t_logit_mean 1.0
    --t_logit_std 1.0
    --trainable_scope full
    --freeze_modules none
    --use_lora 1
    --train_flow_alpha_gate 1
    --train_flow_alpha_embedder 0
    --use_ema 0
    --mixed_precision bf16
    --allow_tf32 1
    --use_checkpoint 0
    --augment_source_swap 1
    --checkpoint_every "${CHECKPOINT_EVERY:-$DEFAULT_CHECKPOINT_EVERY}"
    --checkpoint_half_epoch "${CHECKPOINT_HALF_EPOCH:-$DEFAULT_CHECKPOINT_HALF_EPOCH}"
)
# The frozen prior needs source images even for a latent-conditioned student.
if [[ -n "$RESUME_FROM" ]]; then
    train_args+=(--resume_from "$RESUME_FROM" --resume_strict 1 --resume_optimizer 1)
elif [[ -n "$INIT_FROM" ]]; then
    train_args+=(--init_from "$INIT_FROM" --init_strict 1)
fi

if [[ "$MODE" == --prepare || "$MODE" == --check ]]; then
    # Import e parsing reali, senza costruire il modello o avviare il training.
    python - "${train_args[@]:1}" <<'PY'
import sys
import torch
from train import build_parser
from modules.trellis_prior_training import validate_trellis_prior_args
from modules.prior_gradient_balance import validate_balance_args
args = build_parser().parse_args(sys.argv[1:])
validate_trellis_prior_args(args)
validate_balance_args(args)
print(f"Python: {sys.executable}; torch: {torch.__version__}; CUDA build: {torch.version.cuda}")
print(f"Dataset: {args.root_dir}")
print(f"Output: {args.out_dir}/{args.run_name}")
print(f"Checkpoints: {args.checkpoint_dir}")
print(f"Flow: {args.flow_target}; epochs: {args.train_epochs}; batch/GPU: {args.train_bs}; init: {args.init_from}")
print(f"Prior: {args.trellis_prior_weight}; rollout/grad: {args.trellis_prior_rollout_steps}/{args.trellis_prior_grad_steps}")
print("Import, argomenti e cache: OK. GPU/NCCL da verificare con --comm-check via sbatch.")
PY
    exit 0
fi

export MF_NODE_COUNT="$SLURM_JOB_NUM_NODES"
export MF_GPUS_PER_NODE=4
export MF_NUM_PROCESSES=$((MF_NODE_COUNT * MF_GPUS_PER_NODE))
[[ "${SLURM_NTASKS:-0}" == "$MF_NODE_COUNT" ]] || die "Serve esattamente un task Slurm per nodo (--ntasks-per-node=1)."
node_list="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
export MASTER_ADDR="${MASTER_ADDR:-${node_list%%$'\n'*}}"
export MASTER_PORT="${MASTER_PORT:-$((15000 + SLURM_JOB_ID % 40000))}"
export SLURM_EXPORT_ENV=ALL

if [[ "$MODE" == --comm-check ]]; then
    train_args=("$MF_PROJECT_DIR/slurm/check_multinode.py")
    export NCCL_DEBUG=INFO
else
    mkdir -p "$RUN_DIR/logs" "$MF_CHECKPOINT_DIR"
    # Impedisce due training contemporanei sulla stessa directory di output.
    exec 9>"$RUN_DIR/.training.lock"
    flock -n 9 || die "Un altro training sta usando $RUN_DIR"
    exec 8>"$MF_CHECKPOINT_DIR/.training.lock"
    flock -n 8 || die "Un altro training sta usando $MF_CHECKPOINT_DIR"
    exec > >(tee -a "$RUN_DIR/logs/train_${SLURM_JOB_ID}.log") 2>&1
fi

echo "JOB=$SLURM_JOB_ID MODE=$MODE NODES=$SLURM_JOB_NODELIST"
echo "GPUs=$MF_NUM_PROCESSES (${MF_NODE_COUNT} x ${MF_GPUS_PER_NODE}) MASTER=$MASTER_ADDR:$MASTER_PORT"
echo "DATASET=$MF_DATA_ROOT RUN_DIR=$RUN_DIR CHECKPOINT_DIR=$MF_CHECKPOINT_DIR"
echo "AUTO_RESUME=$AUTO_RESUME RESUME_FROM=${RESUME_FROM:-<none>} INIT_FROM=${INIT_FROM:-<none>}"
echo "FLOW_TARGET=$FLOW_TARGET TRELLIS_PRIOR_WEIGHT=$TRELLIS_PRIOR_WEIGHT EVERY=$TRELLIS_PRIOR_EVERY WARMUP_STEPS=$TRELLIS_PRIOR_WARMUP_STEPS"
echo "TRAIN_BS/GPU=$TRAIN_BS GLOBAL_BATCH=$((TRAIN_BS * MF_NUM_PROCESSES))"
printf 'Training command:'; printf ' %q' "${train_args[@]}"; printf '\n'

# Le virgolette singole rinviano SLURM_PROCID al task sul nodo corretto.
# Nessun MPI launcher: srun avvia un Accelerate, che avvia quattro worker GPU.
srun --mpi=none --export=ALL --kill-on-bad-exit=1 \
    --nodes="$MF_NODE_COUNT" --ntasks="$MF_NODE_COUNT" --ntasks-per-node=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=none \
    bash -c '
        set -euo pipefail
        source "$MF_ENV_FILE"
        cd "$MF_PROJECT_DIR"
        export TMPDIR="/tmp/morphflow_${SLURM_JOB_ID}"
        mkdir -p "$TMPDIR"
        python -c "import torch; n=torch.cuda.device_count(); print(\"Visible GPUs:\", n, flush=True); assert n == 4, f\"Expected 4 GPUs, got {n}\"; assert torch.cuda.is_bf16_supported(), \"BF16 unavailable\""
        echo "HOST=$(hostname) MACHINE_RANK=$SLURM_PROCID PYTHON=$(command -v python)"
        exec python -m accelerate.commands.launch \
            --multi_gpu --gpu_ids all \
            --num_processes "$MF_NUM_PROCESSES" \
            --num_machines "$MF_NODE_COUNT" \
            --machine_rank "$SLURM_PROCID" \
            --main_process_ip "$MASTER_ADDR" \
            --main_process_port "$MASTER_PORT" \
            --same_network --rdzv_backend static \
            --max_restarts 0 \
            --num_cpu_threads_per_process "$OMP_NUM_THREADS" \
            --mixed_precision bf16 --dynamo_backend no \
            "$@"
    ' bash "${train_args[@]}"

echo "Completato: MODE=$MODE JOB=$SLURM_JOB_ID RUN_DIR=$RUN_DIR"
