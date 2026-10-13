#!/bin/bash
#SBATCH --job-name=interp3d_v4
#SBATCH --account=g_vbd
#SBATCH --partition=gpu_vbd
#SBATCH --qos=gpu_vbd
#SBATCH --nodes=1
#SBATCH --nodelist=wn49
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s_vbd:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G
#SBATCH --time=10-00:00:00
#SBATCH --output=interp3d-v4-%j.out
#SBATCH --error=interp3d-v4-%j.err
#SBATCH --mail-type=END,FAIL

# Usage:
#   sbatch slurm/generate_interp3d_v4.slurm
# K=4 targets per pair; resume by repeating the SAME configuration and OUT_NAME.
# IMPORTANT: use a NEW OUT_NAME compared with the former fixed-alpha dataset v4.
set -euo pipefail
module purge
module load singularity/3.8.7

PROJECT_DIR="${PROJECT_DIR:-$HOME/trellis-singularity/work/MorphFlow}"
GENERATOR="$PROJECT_DIR/data/dataset_scripts/generate_interp3d_v4.py"
SIF="${SIF:-$HOME/containers/trellis-py240-cu118.sif}"
BENCH_ROOT="${BENCH_ROOT:-/hpc/scratch/marco.barezzi/morph-benchmark}"
DATA_PARENT="${DATA_PARENT:-/hpc/scratch/marco.barezzi/3d_dataset}"
V3_ROOT="${V3_ROOT:-$DATA_PARENT/morphing_dataset_v3}"
IMAGES_ROOT="${IMAGES_ROOT:-$DATA_PARENT/flux_outputs}"
OUT_NAME="${OUT_NAME:-morphing_dataset_v4}"
OUT_ROOT="${OUT_ROOT:-$DATA_PARENT/$OUT_NAME}"
INTERP_ENV="${INTERP_ENV:-$BENCH_ROOT/interp3d/env}"
INTERP_REPO="${INTERP_REPO:-$BENCH_ROOT/interp3d/work/Interp3D}"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-$BENCH_ROOT/morphany3d/toolchain/cuda118-gcc11}"
SHARED_CACHE="${SHARED_CACHE:-$BENCH_ROOT/shared-cache}"
TRAIN_PAIRS="${TRAIN_PAIRS:-12500}"
VAL_PAIRS="${VAL_PAIRS:-500}"
TEST_PAIRS="${TEST_PAIRS:-500}"
ENDPOINT_FRACTION="${ENDPOINT_FRACTION:-0.02}"
SEED="${SEED:-42}"
ALPHA_MIN="${ALPHA_MIN:-0.001}"
ALPHA_MAX="${ALPHA_MAX:-0.999}"
NGPUS=8
MODEL_ID="${MODEL_ID:-microsoft/TRELLIS-image-large}"

LOG_ROOT="${LOG_ROOT:-/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/dataset_v4_interp3d}"
LOG_DIR="$LOG_ROOT/job_${SLURM_JOB_ID}"
MONITOR="$PROJECT_DIR/slurm/monitor_interp3d_v4.py"
export MFV4_RUN_ID="${SLURM_JOB_ID:-manual}_$(date +%s)"
STATE_PHASE="starting"
mkdir -p "$OUT_ROOT" "$LOG_DIR" "$OUT_ROOT/.worker-home"
# One active generator per output dataset. Prevent concurrent corrupt writes.
exec 9>"$OUT_ROOT/.generation.lock"
if ! flock -n 9; then
    echo "ERROR: another job is already generating into $OUT_ROOT" >&2
    exit 1
fi

for file in "$GENERATOR" "$MONITOR" "$SIF"; do
    [[ -f "$file" ]] || { echo "ERROR missing file: $file" >&2; exit 1; }
done
for dir in "$PROJECT_DIR" "$V3_ROOT" "$IMAGES_ROOT" "$INTERP_ENV" "$INTERP_REPO" "$TOOLCHAIN_ROOT" \
  "$SHARED_CACHE/hf-home" "$SHARED_CACHE/torch-home" "$SHARED_CACHE/u2net"; do
    [[ -d "$dir" ]] || { echo "ERROR missing directory: $dir" >&2; exit 1; }
done

# If Slurm provides explicit GPU tokens, respect that allocation.
GPU_TOKENS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_TOKENS <<< "$CUDA_VISIBLE_DEVICES"
fi
if (( ${#GPU_TOKENS[@]} < NGPUS )); then
    GPU_TOKENS=(0 1 2 3 4 5 6 7)
fi

if [[ -n "${SLURM_TMPDIR:-}" && -d "$SLURM_TMPDIR" ]]; then
    LOCAL_ROOT="$SLURM_TMPDIR/interp3d_v4_${SLURM_JOB_ID}"
elif [[ -d /nvme && -w /nvme ]]; then
    LOCAL_ROOT="/nvme/$USER/interp3d_v4_${SLURM_JOB_ID}"
else
    LOCAL_ROOT="/tmp/$USER/interp3d_v4_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_ROOT"
set_phase() {
    python3 "$MONITOR" set-phase --out "$OUT_ROOT" \
        --job-id "${SLURM_JOB_ID:-manual}" --run-id "$MFV4_RUN_ID" \
        --phase "$1" --status "${2:-running}" --message "${3:-}"
}
on_exit() {
    rc=$?
    trap - EXIT
    if (( rc != 0 )); then
        set_phase "$STATE_PHASE" failed "exit_code=$rc; inspect $LOG_DIR" || true
    fi
    rm -rf "$LOCAL_ROOT" 2>/dev/null || true
    exit "$rc"
}
trap on_exit EXIT
trap 'echo "SIGTERM received" >&2; exit 143' TERM
trap 'echo "SIGINT received" >&2; exit 130' INT
set_phase starting running "preflight pending"

printf '\n=== Interp3D dataset v4 ===\n'
printf 'Node=%s | GPUs=%s | train_pairs=%s | val_pairs=%s | test_pairs=%s | endpoint_fraction=%s\n' \
    "${SLURMD_NODENAME:-unknown}" "$NGPUS" "$TRAIN_PAIRS" "$VAL_PAIRS" "$TEST_PAIRS" "$ENDPOINT_FRACTION"
printf 'v3=%s\nimages=%s\nout=%s\ninterp=%s\nlogs=%s\n' "$V3_ROOT" "$IMAGES_ROOT" "$OUT_ROOT" "$INTERP_REPO" "$LOG_DIR"
nvidia-smi -L || true

# Plan on HOST: no Torch or Singularity required. It reuses v3 train/val/test ASSET splits, NOT v3 pairs.
python3 "$GENERATOR" plan --v3-root "$V3_ROOT" --images-root "$IMAGES_ROOT" \
    --out "$OUT_ROOT" --train-pairs "$TRAIN_PAIRS" --val-pairs "$VAL_PAIRS" \
    --test-pairs "$TEST_PAIRS" --endpoint-fraction "$ENDPOINT_FRACTION" \
    --seed "$SEED" --alpha-min "$ALPHA_MIN" --alpha-max "$ALPHA_MAX"

run_container() {
    local shard="$1"; shift
    local gpu="${GPU_TOKENS[$shard]}"
    local runhome="$OUT_ROOT/.worker-home/shard_${shard}"
    local runtmp="$LOCAL_ROOT/shard_${shard}"
    mkdir -p "$runhome" "$runtmp"
    export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$gpu"
    export SINGULARITYENV_MFV4_RUN_ID="$MFV4_RUN_ID"
    export SINGULARITYENV_SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
    export SINGULARITYENV_OMP_NUM_THREADS=6
    export SINGULARITYENV_MKL_NUM_THREADS=6
    export SINGULARITYENV_OPENBLAS_NUM_THREADS=6
    export SINGULARITYENV_NUMEXPR_NUM_THREADS=6
    export SINGULARITYENV_MORPHFLOW_INPUTS_HOST_PREFIX="$DATA_PARENT"
    export SINGULARITYENV_MORPHFLOW_INPUTS_CONTAINER_PREFIX=/inputs
    singularity exec --cleanenv --containall --no-mount hostfs --nv \
        --bind "$INTERP_ENV:/opt/trellis/env:ro" \
        --bind "$TOOLCHAIN_ROOT:/opt/trellis/toolchain/cuda118-gcc11:ro" \
        --bind "$INTERP_REPO:/workspace/MethodRepo:ro" \
        --bind "$PROJECT_DIR:/workspace/MorphFlow:ro" \
        --bind "$DATA_PARENT:/inputs:ro" \
        --bind "$OUT_ROOT:/dataset:rw" \
        --bind "$SHARED_CACHE/hf-home:/opt/trellis/hf-home:rw" \
        --bind "$SHARED_CACHE/torch-home:/opt/trellis/torch-home:rw" \
        --bind "$SHARED_CACHE/u2net:/opt/trellis/u2net:rw" \
        --bind "$runhome:/runtime-home:rw" \
        --bind "$runtmp:/runtime-tmp:rw" \
        --pwd /workspace/MethodRepo \
        "$SIF" bash -s -- "$@" <<'INSIDE'
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/trellis/env
export HOME=/runtime-home
export TMPDIR=/runtime-tmp
export XDG_CACHE_HOME=/runtime-home/.cache
export HF_HOME=/opt/trellis/hf-home
export TORCH_HOME=/opt/trellis/torch-home
export U2NET_HOME=/opt/trellis/u2net
export CUDA_HOME=/opt/trellis/toolchain/cuda118-gcc11
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/opt/trellis/torch-home/extensions/nvdiffrast-035
export MAX_JOBS=4
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/workspace/MethodRepo:/workspace/MorphFlow
mkdir -p "$TORCH_EXTENSIONS_DIR"
exec "$@"
INSIDE
}

# One process verifies imports/CUDA before the eight workers start.
STATE_PHASE=preflight
set_phase preflight
printf 'Monitor from login node: python3 %s status --out %s --watch 30\n' "$MONITOR" "$OUT_ROOT"
echo '=== Serialized DINOv2 cache warmup ==='

TORCH_HUB_DIR="$SHARED_CACHE/torch-home/hub"
DINO_DIR="$TORCH_HUB_DIR/facebookresearch_dinov2_main"

mkdir -p "$TORCH_HUB_DIR"

(
    flock -x 200

    if [[ -d "$DINO_DIR" && ! -s "$DINO_DIR/hubconf.py" ]]; then
        BROKEN="${DINO_DIR}.broken-${SLURM_JOB_ID}-$(date +%s)"
        echo "Incomplete DINOv2 cache: moving to $BROKEN"
        mv "$DINO_DIR" "$BROKEN"
    fi

    echo "Initializing complete Interp3D pipeline on one GPU"

    run_container 0 python -c '
import sys
import torch
from trellis.pipelines import TrellisImageTo3DPipeline

print("CUDA available:", torch.cuda.is_available(), flush=True)
print("Visible GPUs:", torch.cuda.device_count(), flush=True)

assert torch.cuda.device_count() == 1

pipeline = TrellisImageTo3DPipeline.from_pretrained(
    sys.argv[1]
)

print("DINOv2 initialized successfully", flush=True)
print("Interp3D pipeline initialized successfully", flush=True)

del pipeline
' "$MODEL_ID"

    if [[ ! -s "$DINO_DIR/hubconf.py" ]]; then
        echo "ERROR: DINOv2 hubconf.py still missing" >&2
        exit 1
    fi

    echo "DINOv2 cache verified"

) 200>"$TORCH_HUB_DIR/.morphflow_v4_dinov2.lock"

run_shards() {
    local phase="$1"
    local -a pids=()
    local -a seen=()
    local shard rc pid marker failed=0 finished=0
    for (( shard=0; shard<NGPUS; shard++ )); do
        echo "Launching $phase shard=$shard GPU=${GPU_TOKENS[$shard]}"
        marker="$LOG_DIR/${phase}_gpu_${shard}.exit"
        rm -f "$marker"
        (
            set +e
            run_container "$shard" python /workspace/MorphFlow/data/dataset_scripts/generate_interp3d_v4.py \
                "$phase" --out /dataset --num-shards "$NGPUS" --shard-index "$shard" --model-id "$MODEL_ID"
            rc=$?
            printf '%s\n' "$rc" > "$marker"
            exit "$rc"
        ) > "$LOG_DIR/${phase}_gpu_${shard}.log" 2>&1 &
        pids+=("$!")
        seen+=(0)
    done
    # Poll all shards rather than waiting on shard 0 first: fail fast if any
    # worker crashes, and preserve its exit code in an inspectable marker.
    while (( finished < NGPUS )); do
        for (( shard=0; shard<NGPUS; shard++ )); do
            if (( seen[shard] == 1 )); then
                continue
            fi
            pid="${pids[$shard]}"
            marker="$LOG_DIR/${phase}_gpu_${shard}.exit"
            if [[ -f "$marker" ]]; then
                rc="$(cat "$marker")"
            elif ! kill -0 "$pid" 2>/dev/null; then
                rc=99  # killed before writing the exit marker
            else
                continue
            fi
            seen[$shard]=1
            finished=$((finished+1))
            wait "$pid" || true
            echo "$phase GPU=$shard exit_code=$rc ($finished/$NGPUS)"
            if [[ "$rc" != 0 ]]; then
                failed=1
                echo "ERROR: worker failed: $LOG_DIR/${phase}_gpu_${shard}.log" >&2
                break
            fi
        done
        if (( failed == 1 )); then
            for pid in "${pids[@]}"; do
                kill "$pid" 2>/dev/null || true
            done
            for pid in "${pids[@]}"; do
                wait "$pid" 2>/dev/null || true
            done
            return 1
        fi
        if (( finished < NGPUS )); then
            sleep 10
        fi
    done
}

# First source assets; only after all 8 shards finish begin target generation.
STATE_PHASE=assets
set_phase assets
run_shards assets
STATE_PHASE=pairs
set_phase pairs
run_shards pairs
STATE_PHASE=finalize
set_phase finalize
python3 "$GENERATOR" finalize --out "$OUT_ROOT"

# Verify that the unmodified MorphFlow v3 DataLoader accepts v4 metadata.
echo "=== MorphFlow DataLoader smoke test (train split) ==="
run_container 0 python /workspace/MorphFlow/data/morph_dataset.py --root /dataset --split train --batch-size 1 --augment-source-swap
STATE_PHASE=complete
set_phase complete complete "all targets finalized; loader smoke test passed"
echo "COMPLETE: $OUT_ROOT"
echo "Log files: $LOG_DIR"
