#!/bin/bash
#SBATCH --job-name=threeway_tex
#SBATCH --account=g_vbd
#SBATCH --partition=gpu_vbd
#SBATCH --qos=gpu_vbd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6000_vbd:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/hpc/archive/G_VBD/marco.barezzi/morph-benchmark/reports/threeway-%x-%j.out
#SBATCH --error=/hpc/archive/G_VBD/marco.barezzi/morph-benchmark/reports/threeway-%x-%j.err

# Multi-GPU, resumable benchmark:
#   MorphFlow vs MorphAny3D vs Interp3D
#   SAME uniform alpha schedule for every method
#   -> raw TRELLIS meshes
#   -> common texture baking to textured GLB
#   -> 64-view material renders
#   -> internal + FULL endpoint-inclusive LPIPS/PPL/PDV
#   -> pooled endpoint-manifold FID
#
# Parallelism: pair sharding across all GPUs on ONE node.
# Each heavy phase is split by pair_id % NGPUS.

set -euo pipefail

module purge
module load singularity/3.8.7

# -------------------------------
# Benchmark configuration
# -------------------------------
N="${N:-100}"
K="${K:-10}"
SEED="${SEED:-42}"

STEPS="${STEPS:-50}"
SLAT_STEPS="${SLAT_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-3.0}"
SLAT_CFG_SCALE="${SLAT_CFG_SCALE:-3.0}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"

TEXTURE_SIZE="${TEXTURE_SIZE:-1024}"
SIMPLIFY="${SIMPLIFY:-0.95}"
FILL_HOLES="${FILL_HOLES:-1}"

# Interp3D-style perceptual-view count, common to every method.
NUM_VIEWS="${NUM_VIEWS:-64}"
RENDER_RESOLUTION="${RENDER_RESOLUTION:-512}"
LPIPS_NET="${LPIPS_NET:-vgg}"
LPIPS_BATCH_SIZE="${LPIPS_BATCH_SIZE:-8}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-16}"
FID_MIN_SAMPLES="${FID_MIN_SAMPLES:-512}"

# Number of pair shards / GPUs used concurrently on this node.
# Must not exceed the GPU allocation requested from Slurm.
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    DETECTED_GPUS="$(printf '%s' "$SLURM_GPUS_ON_NODE" | grep -oE '[0-9]+$' || true)"
else
    DETECTED_GPUS=""
fi
NGPUS="${NGPUS:-${DETECTED_GPUS:-4}}"

# -------------------------------
# Host paths
# -------------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/trellis-singularity/work/MorphFlow}"
SIF="${SIF:-$HOME/containers/trellis-py240-cu118.sif}"
TRELLIS_BASE="${TRELLIS_BASE:-$HOME/trellis-singularity}"
ORIGINAL_ENV="${ORIGINAL_ENV:-$TRELLIS_BASE/env}"

BENCH_ROOT="${BENCH_ROOT:-/hpc/scratch/marco.barezzi/morph-benchmark}"
BENCH_ARCHIVE="${BENCH_ARCHIVE:-/hpc/archive/G_VBD/marco.barezzi/morph-benchmark}"
SHARED_CACHE="${SHARED_CACHE:-$BENCH_ROOT/shared-cache}"
TOOLCHAIN_ROOT_HOST="${TOOLCHAIN_ROOT_HOST:-$BENCH_ROOT/morphany3d/toolchain/cuda118-gcc11}"

MORPHANY_ENV="${MORPHANY_ENV:-$BENCH_ROOT/morphany3d/env}"
MORPHANY_REPO="${MORPHANY_REPO:-$BENCH_ROOT/morphany3d/work/MorphAny3D}"
INTERP_ENV="${INTERP_ENV:-$BENCH_ROOT/interp3d/env}"
INTERP_REPO="${INTERP_REPO:-$BENCH_ROOT/interp3d/work/Interp3D}"

DATA_PARENT="${DATA_PARENT:-/hpc/scratch/marco.barezzi/3d_dataset}"
DATASET_NAME="${DATASET_NAME:-morphing_dataset_v3}"
SOURCE_IMAGES_NAME="${SOURCE_IMAGES_NAME:-flux_outputs}"
METADATA="${METADATA:-metadata_test.json}"
SOURCE_IMAGE_FILENAME="${SOURCE_IMAGE_FILENAME:-}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH=/absolute/path/to/ss_checkpoint.pt}"
SLAT_CHECKPOINT_PATH="${SLAT_CHECKPOINT_PATH:?Set SLAT_CHECKPOINT_PATH=/absolute/path/to/slat_checkpoint.pt}"

RUN_NAME="${RUN_NAME:-threeway_uniform_n${N}_k${K}_seed${SEED}_v${NUM_VIEWS}}"
RUN_ROOT="$BENCH_ROOT/threeway-runs/$RUN_NAME"
FINAL_ARCHIVE="$BENCH_ARCHIVE/final-results/$RUN_NAME"
SCRIPT_IN="/workspace/MorphFlow/benchmarks/threeway_textured_benchmark.py"
EVAL_IN="/workspace/MorphFlow/benchmarks/eval_textured_benchmark.py"

# Portable Blender prepared during the smoke test.
BLENDER_ROOT_HOST="${BLENDER_ROOT_HOST:-$BENCH_ROOT/tools/blender-4.2.23-linux-x64}"
BLENDER_BIN_REL="${BLENDER_BIN_REL:-blender-headless-wrapper}"
BLENDER_BIN_IN="/opt/blender/$BLENDER_BIN_REL"

mkdir -p "$RUN_ROOT" "$RUN_ROOT/logs" "$RUN_ROOT/runtime-home" "$RUN_ROOT/tmp"
mkdir -p "$FINAL_ARCHIVE" "$BENCH_ARCHIVE/reports"

# -------------------------------
# Discover TRELLIS used by MorphFlow
# -------------------------------
TRELLIS_REPO_HOST="${TRELLIS_REPO_HOST:-}"
if [[ -z "$TRELLIS_REPO_HOST" ]]; then
    for candidate in \
        "$TRELLIS_BASE/work/TRELLIS/MorphAny3D" \
        "$TRELLIS_BASE/work/TRELLIS" \
        "$PROJECT_DIR/../TRELLIS/MorphAny3D" \
        "$PROJECT_DIR/../TRELLIS"
    do
        if [[ -d "$candidate/trellis" ]]; then
            TRELLIS_REPO_HOST="$candidate"
            break
        fi
    done
fi
[[ -n "$TRELLIS_REPO_HOST" && -d "$TRELLIS_REPO_HOST/trellis" ]] || {
    echo "ERROR: could not locate MorphFlow TRELLIS repo" >&2
    exit 1
}

# -------------------------------
# Preconditions
# -------------------------------
for d in \
    "$PROJECT_DIR" \
    "$ORIGINAL_ENV" \
    "$MORPHANY_ENV" \
    "$MORPHANY_REPO" \
    "$INTERP_ENV" \
    "$INTERP_REPO" \
    "$DATA_PARENT/$DATASET_NAME" \
    "$DATA_PARENT/$SOURCE_IMAGES_NAME" \
    "$SHARED_CACHE/hf-home" \
    "$SHARED_CACHE/torch-home" \
    "$SHARED_CACHE/u2net" \
    "$TOOLCHAIN_ROOT_HOST" \
    "$BLENDER_ROOT_HOST"
do
    [[ -d "$d" ]] || { echo "ERROR: missing directory: $d" >&2; exit 1; }
done

[[ -f "$PROJECT_DIR/benchmarks/threeway_textured_benchmark.py" ]] || { echo "ERROR: benchmark script missing" >&2; exit 1; }
[[ -f "$PROJECT_DIR/benchmarks/eval_textured_benchmark.py" ]] || { echo "ERROR: eval wrapper missing" >&2; exit 1; }
[[ -f "$CHECKPOINT_PATH" ]] || { echo "ERROR: SS checkpoint missing: $CHECKPOINT_PATH" >&2; exit 1; }
[[ -f "$SLAT_CHECKPOINT_PATH" ]] || { echo "ERROR: SLat checkpoint missing: $SLAT_CHECKPOINT_PATH" >&2; exit 1; }
[[ -x "$BLENDER_ROOT_HOST/$BLENDER_BIN_REL" ]] || { echo "ERROR: Blender wrapper missing/not executable" >&2; exit 1; }

SS_DIR="$(dirname "$CHECKPOINT_PATH")"
SS_FILE="$(basename "$CHECKPOINT_PATH")"
SLAT_DIR="$(dirname "$SLAT_CHECKPOINT_PATH")"
SLAT_FILE="$(basename "$SLAT_CHECKPOINT_PATH")"
SS_IN="/checkpoints/ss/$SS_FILE"
SLAT_IN="/checkpoints/slat/$SLAT_FILE"

# -------------------------------
# GPU mapping
# -------------------------------
GPU_TOKENS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_TOKENS <<< "$CUDA_VISIBLE_DEVICES"
fi
if (( ${#GPU_TOKENS[@]} < NGPUS )); then
    GPU_TOKENS=()
    for ((i=0; i<NGPUS; i++)); do GPU_TOKENS+=("$i"); done
fi

# -------------------------------
# Isolated container launcher
# -------------------------------
run_phase() {
    local phase="$1"
    local env_host="$2"
    local repo_host="$3"
    local gpu_token="$4"
    shift 4

    local phase_home="$RUN_ROOT/runtime-home/$phase"
    local phase_tmp="$RUN_ROOT/tmp/$phase"
    mkdir -p "$phase_home" "$phase_tmp"

    if [[ "$gpu_token" != "cpu" ]]; then
        export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$gpu_token"
    else
        unset SINGULARITYENV_CUDA_VISIBLE_DEVICES || true
    fi

    singularity exec --cleanenv --containall --no-mount hostfs --nv \
        --bind "$env_host:/opt/trellis/env:ro" \
        --bind "$TOOLCHAIN_ROOT_HOST:/opt/trellis/toolchain/cuda118-gcc11:ro" \
        --bind "$PROJECT_DIR:/workspace/MorphFlow:ro" \
        --bind "$repo_host:/workspace/MethodRepo:ro" \
        --bind "$DATA_PARENT:/inputs:ro" \
        --bind "$RUN_ROOT:/benchmark:rw" \
        --bind "$SHARED_CACHE/hf-home:/opt/trellis/hf-home:rw" \
        --bind "$SHARED_CACHE/torch-home:/opt/trellis/torch-home:rw" \
        --bind "$SHARED_CACHE/u2net:/opt/trellis/u2net:rw" \
        --bind "$SS_DIR:/checkpoints/ss:ro" \
        --bind "$SLAT_DIR:/checkpoints/slat:ro" \
        --bind "$phase_home:/runtime-home:rw" \
        --bind "$phase_tmp:/runtime-tmp:rw" \
        --bind "$BLENDER_ROOT_HOST:/opt/blender:ro" \
        --pwd /workspace/MorphFlow \
        "$SIF" \
        bash -s -- "$@" <<'IN_CONTAINER'
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
mkdir -p "$TORCH_EXTENSIONS_DIR"

export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/workspace/MethodRepo:/workspace/MorphFlow

exec "$@"
IN_CONTAINER
}

run_sharded() {
    local phase="$1"
    local env_host="$2"
    local repo_host="$3"
    shift 3
    local -a command=("$@")
    local -a pids=()
    local shard gpu log_out log_err

    echo "[$(date -Is)] START sharded phase=$phase shards=$NGPUS" | tee -a "$RUN_ROOT/progress.log"
    for ((shard=0; shard<NGPUS; shard++)); do
        gpu="${GPU_TOKENS[$shard]}"
        log_out="$RUN_ROOT/logs/${phase}_g${shard}.out"
        log_err="$RUN_ROOT/logs/${phase}_g${shard}.err"
        (
            run_phase "${phase}_g${shard}" "$env_host" "$repo_host" "$gpu" \
                "${command[@]}" --shard-index "$shard" --num-shards "$NGPUS"
        ) >"$log_out" 2>"$log_err" &
        local child_pid="$!"
        pids+=("$child_pid")
        echo "[$(date -Is)] $phase shard=$shard gpu=$gpu pid=$child_pid" | tee -a "$RUN_ROOT/progress.log"
    done

    local failed=0
    for ((shard=0; shard<NGPUS; shard++)); do
        if ! wait "${pids[$shard]}"; then
            failed=1
            echo "ERROR: $phase shard=$shard failed; tail:" >&2
            tail -n 80 "$RUN_ROOT/logs/${phase}_g${shard}.err" >&2 || true
        fi
    done
    (( failed == 0 )) || exit 1
    echo "[$(date -Is)] DONE sharded phase=$phase" | tee -a "$RUN_ROOT/progress.log"
}

# -------------------------------
# Provenance
# -------------------------------
{
    echo "===== THREE-WAY TEXTURED MULTI-GPU BENCHMARK ====="
    echo "JOB_ID=${SLURM_JOB_ID:-none}"
    echo "NODE=${SLURM_JOB_NODELIST:-none}"
    echo "RUN_NAME=$RUN_NAME"
    echo "RUN_ROOT=$RUN_ROOT"
    echo "N=$N"
    echo "K=$K"
    echo "SEED=$SEED"
    echo "NGPUS=$NGPUS"
    echo "GPU_TOKENS=${GPU_TOKENS[*]}"
    echo "SCHEDULE=uniform_shared_all_methods"
    echo "TEXTURE_SIZE=$TEXTURE_SIZE"
    echo "NUM_VIEWS=$NUM_VIEWS"
    echo "RENDER_RESOLUTION=$RENDER_RESOLUTION"
    echo "FID_PROTOCOL=pooled_endpoints_vs_two_center_states"
    echo "PROJECT_DIR=$PROJECT_DIR"
    echo "TRELLIS_REPO_HOST=$TRELLIS_REPO_HOST"
    echo "MORPHANY_REPO=$MORPHANY_REPO"
    echo "INTERP_REPO=$INTERP_REPO"
    echo "==============================================="
} | tee -a "$RUN_ROOT/progress.log"

# Save exact benchmark code used by the run.
cp -f "$PROJECT_DIR/benchmarks/threeway_textured_benchmark.py" "$RUN_ROOT/"
cp -f "$PROJECT_DIR/benchmarks/eval_textured_benchmark.py" "$RUN_ROOT/"
cp -f "$PROJECT_DIR/slurm/benchmark_threeway_textured.slurm" "$RUN_ROOT/" 2>/dev/null || true

# -------------------------------
# Preflight on one GPU (also prevents simultaneous nvdiffrast JIT builds)
# -------------------------------
run_phase preflight "$MORPHANY_ENV" "$MORPHANY_REPO" "${GPU_TOKENS[0]}" \
    python -c 'import torch, lpips, scipy, torchvision; import nvdiffrast.torch as dr; dr.RasterizeCudaContext(); print("preflight CUDA OK", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

run_phase fid_weights "$MORPHANY_ENV" "$MORPHANY_REPO" "cpu" \
    python "$EVAL_IN" check-fid-weights

run_phase blender_check "$MORPHANY_ENV" "$MORPHANY_REPO" "${GPU_TOKENS[0]}" \
    "$BLENDER_BIN_IN" --version

# -------------------------------
# 1) Fixed shared plan: same pairs + same UNIFORM alphas for all methods
# -------------------------------
run_phase plan "$ORIGINAL_ENV" "$TRELLIS_REPO_HOST" "cpu" \
    python "$SCRIPT_IN" plan \
        --run-root /benchmark \
        --data-root "/inputs/$DATASET_NAME" \
        --metadata "$METADATA" \
        --source-images-root "/inputs/$SOURCE_IMAGES_NAME" \
        --source-image-filename "$SOURCE_IMAGE_FILENAME" \
        --num-pairs "$N" \
        --k "$K" \
        --seed "$SEED"

# -------------------------------
# 2) Generation: pair-sharded over GPUs
# -------------------------------
run_sharded morphflow "$ORIGINAL_ENV" "$TRELLIS_REPO_HOST" \
    python "$SCRIPT_IN" morphflow \
        --run-root /benchmark \
        --data-root "/inputs/$DATASET_NAME" \
        --metadata "$METADATA" \
        --source-images-root "/inputs/$SOURCE_IMAGES_NAME" \
        --source-image-filename "$SOURCE_IMAGE_FILENAME" \
        --checkpoint-path "$SS_IN" \
        --slat-checkpoint-path "$SLAT_IN" \
        --steps "$STEPS" \
        --slat-steps "$SLAT_STEPS" \
        --cfg-scale "$CFG_SCALE" \
        --slat-cfg-scale "$SLAT_CFG_SCALE" \
        --mixed-precision "$MIXED_PRECISION" \
        --trellis-model image_large

run_sharded morphany3d "$MORPHANY_ENV" "$MORPHANY_REPO" \
    python "$SCRIPT_IN" morphany3d --run-root /benchmark

run_sharded interp3d "$INTERP_ENV" "$INTERP_REPO" \
    python "$SCRIPT_IN" interp3d --run-root /benchmark

# -------------------------------
# 3) Common texture baking: pair-sharded over GPUs
# -------------------------------
run_sharded texture "$MORPHANY_ENV" "$MORPHANY_REPO" \
    python "$SCRIPT_IN" texture \
        --run-root /benchmark \
        --texture-size "$TEXTURE_SIZE" \
        --simplify "$SIMPLIFY" \
        --fill-holes "$FILL_HOLES"

run_phase texture_status "$MORPHANY_ENV" "$MORPHANY_REPO" "cpu" \
    python "$SCRIPT_IN" texture-status --run-root /benchmark

# -------------------------------
# 4) Material-render perceptual evaluation, pair-sharded over GPUs
#    Includes original metrics + endpoint-inclusive FULL path metrics + FID features.
# -------------------------------
for METHOD in morphflow morphany3d interp3d; do
    run_sharded "metrics_${METHOD}" "$MORPHANY_ENV" "$MORPHANY_REPO" \
        python "$EVAL_IN" shard \
            --run-dir "/benchmark/$METHOD" \
            --output-dir "/benchmark/metrics/$METHOD" \
            --stage all \
            --skip-invalid-pairs \
            --num-views "$NUM_VIEWS" \
            --resolution "$RENDER_RESOLUTION" \
            --appearance materials \
            --lpips-net "$LPIPS_NET" \
            --lpips-batch-size "$LPIPS_BATCH_SIZE" \
            --fid-batch-size "$FID_BATCH_SIZE" \
            --device cuda \
            --strict-uniform-alphas \
            --blender-bin "$BLENDER_BIN_IN"

    # Deterministic merge + pooled batch-level FID.
    run_phase "metrics_merge_${METHOD}" "$MORPHANY_ENV" "$MORPHANY_REPO" "cpu" \
        python "$EVAL_IN" merge \
            --output-dir "/benchmark/metrics/$METHOD" \
            --fid-min-samples "$FID_MIN_SAMPLES"
done

# -------------------------------
# 5) Final three-method table
# -------------------------------
run_phase aggregate "$MORPHANY_ENV" "$MORPHANY_REPO" "cpu" \
    python "$SCRIPT_IN" aggregate --run-root /benchmark

# -------------------------------
# 6) Compact archive (large GLBs/renders remain in scratch)
# -------------------------------
cp -f "$RUN_ROOT/plan.json" "$FINAL_ARCHIVE/"
cp -f "$RUN_ROOT/progress.log" "$FINAL_ARCHIVE/"
cp -f "$RUN_ROOT/final_metrics.json" "$FINAL_ARCHIVE/"
cp -f "$RUN_ROOT/final_metrics.csv" "$FINAL_ARCHIVE/"
cp -f "$RUN_ROOT/texture_status.json" "$FINAL_ARCHIVE/"
cp -f "$RUN_ROOT/threeway_textured_benchmark.py" "$FINAL_ARCHIVE/"
cp -f "$RUN_ROOT/eval_textured_benchmark.py" "$FINAL_ARCHIVE/"
cp -f "$RUN_ROOT/benchmark_threeway_textured.slurm" "$FINAL_ARCHIVE/" 2>/dev/null || true

for METHOD in morphflow morphany3d interp3d; do
    mkdir -p "$FINAL_ARCHIVE/metrics/$METHOD"
    cp -f "$RUN_ROOT/metrics/$METHOD/metrics_batch.json" "$FINAL_ARCHIVE/metrics/$METHOD/"
    cp -f "$RUN_ROOT/metrics/$METHOD/per_pair.csv" "$FINAL_ARCHIVE/metrics/$METHOD/"
done

echo "DONE: $RUN_ROOT"
echo "FINAL METRICS: $FINAL_ARCHIVE/final_metrics.json"
