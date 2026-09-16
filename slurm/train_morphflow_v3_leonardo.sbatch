#!/bin/bash
#SBATCH --job-name=v3_slat_lora_cross_r32_tokenGate_semMatch_loralr1e-4_leonardo
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=4
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
RUN_NAME="${RUN_NAME:-v3_slat_lora_cross_r32_tokenGate_semMatch_loralr1e-4_leonardo}"
[[ "$RUN_NAME" != */* && "$RUN_NAME" != . && "$RUN_NAME" != .. ]] || die "RUN_NAME deve essere un nome, non un percorso."
RUN_DIR="$MF_OUT_ROOT/$RUN_NAME"
AUTO_RESUME="${AUTO_RESUME:-0}"
RESUME_FROM="${RESUME_FROM:-}"
[[ "$AUTO_RESUME" == 0 || "$AUTO_RESUME" == 1 ]] || die "AUTO_RESUME deve essere 0 o 1."

# Parametri dell'allegato r32/semMatch, applicati al flow SLat.
# Con 4 nodi e train_bs=2: 16 GPU, batch globale 32.
TRAIN_BS="${TRAIN_BS:-2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
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

    if [[ "$AUTO_RESUME" == 1 && -z "$RESUME_FROM" ]]; then
        RESUME_FROM="$(python - "$RUN_DIR/checkpoints" <<'PY'
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
    elif compgen -G "$RUN_DIR/checkpoints/morphflow_*.pt" >/dev/null; then
        die "La run contiene checkpoint: usare AUTO_RESUME=1, RESUME_FROM oppure un nuovo RUN_NAME."
    fi

    # --prepare scarica SOLO i pesi SLat image_large necessari, una volta dal login.
    # Negli altri modi la cache viene controllata senza accessi a Internet.
    if [[ -z "$RESUME_FROM" ]]; then
        if [[ "$MODE" == --prepare ]]; then
            export HF_HUB_OFFLINE=0
        fi
        python - "$MODE" <<'PY'
import sys
from huggingface_hub import hf_hub_download
try:
    path = hf_hub_download(
        repo_id="microsoft/TRELLIS-image-large",
        filename="ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors",
        local_files_only=sys.argv[1] != "--prepare",
    )
except Exception as exc:
    raise SystemExit(
        f"Pesi TRELLIS non disponibili: {exc}\n"
        "Dal login node: bash slurm/train_morphflow_v3_leonardo.sbatch --prepare"
    ) from exc
print(f"TRELLIS weights: {path}")
PY
    fi
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
    --train_bs "$TRAIN_BS"
    --val_bs 1
    --train_epochs "$TRAIN_EPOCHS"
    --lr 1e-4
    --cond_lr 1e-4
    --flow_lr 1e-4
    --lora_lr 1e-4
    --weight_decay 1e-4
    --warmup_steps 20000
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
    --cfg_drop_prob 0.2
    --trellis_model image_large
    --flow_target slat
    --slat_condition_source slat
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
    --use_semantic_token_matching 1
    --semantic_match_max_align 0.25
    --semantic_match_temperature 0.1
    --semantic_match_dim 128
    --semantic_match_alpha_weight 1
    --semantic_match_detach_scores 0
    --semantic_match_exclude_style_tokens 1
    --semantic_cycle_loss_weight 0.01
    --semantic_cycle_loss_prob 0.25
    --semantic_cycle_detach_targets 1
    --semantic_cycle_alpha_weight 1
    --semantic_match_log_stats 1
    --separate_cond 1
    --separate_cond_gate token
    --t_schedule logit_normal
    --t_logit_mean 1.0
    --t_logit_std 1.0
    --trainable_scope cond_cross_attn
    --freeze_modules none
    --use_lora 1
    --train_flow_alpha_gate 1
    --train_flow_alpha_embedder 1
    --use_ema 0
    --mixed_precision bf16
    --allow_tf32 1
    --use_checkpoint 0
    --augment_source_swap 1
    --checkpoint_every 2
)
# Non serve --source_images_root: SLat con slat_condition_source=slat usa i latenti.
if [[ -n "$RESUME_FROM" ]]; then
    train_args+=(--resume_from "$RESUME_FROM" --resume_strict 1 --resume_optimizer 1)
fi

if [[ "$MODE" == --prepare || "$MODE" == --check ]]; then
    # Import e parsing reali, senza costruire il modello o avviare il training.
    python - "${train_args[@]:1}" <<'PY'
import sys
import torch
from train import build_parser
args = build_parser().parse_args(sys.argv[1:])
print(f"Python: {sys.executable}; torch: {torch.__version__}; CUDA build: {torch.version.cuda}")
print(f"Dataset: {args.root_dir}")
print(f"Output: {args.out_dir}/{args.run_name}")
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
    mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints"
    # Impedisce due training contemporanei sulla stessa directory di output.
    exec 9>"$RUN_DIR/.training.lock"
    flock -n 9 || die "Un altro training sta usando $RUN_DIR"
    exec > >(tee -a "$RUN_DIR/logs/train_${SLURM_JOB_ID}.log") 2>&1
fi

echo "JOB=$SLURM_JOB_ID MODE=$MODE NODES=$SLURM_JOB_NODELIST"
echo "GPUs=$MF_NUM_PROCESSES (${MF_NODE_COUNT} x ${MF_GPUS_PER_NODE}) MASTER=$MASTER_ADDR:$MASTER_PORT"
echo "DATASET=$MF_DATA_ROOT RUN_DIR=$RUN_DIR"
echo "AUTO_RESUME=$AUTO_RESUME RESUME_FROM=${RESUME_FROM:-<none>}"
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
