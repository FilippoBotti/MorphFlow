#!/bin/bash
#SBATCH --job-name=threeway_v4_interp3d
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
#SBATCH --output=threeway-l40s-%x-%j.out
#SBATCH --error=threeway-l40s-%x-%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=BEGIN,END,FAIL

# Parma/Ateneo: eight independent pair shards, one GPU per worker.
# Configure through CONFIG_FILE or environment variables; see the .env.example.
# --check validates snapshots, all selected test inputs and CPU imports.
# Default --run saves mesh/Gaussian raw assets and evaluates BOTH FIDs.
# EXPORT_GLB=1 additionally bakes every GLB; the default defers baking.
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }
MODE="${1:---run}"
case "$MODE" in --run|--check) ;; *) die "Usage: $0 [--run|--check]" ;; esac
[[ $# -le 1 ]] || die "Use CONFIG_FILE or environment variables for configuration."
if [[ -n "${CONFIG_FILE:-}" ]]; then
    [[ -f "$CONFIG_FILE" ]] || die "Missing CONFIG_FILE: $CONFIG_FILE"
    set -a
    source "$CONFIG_FILE"
    set +a
fi
if [[ "$MODE" == --run ]]; then
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "Submit this launcher with sbatch."
    [[ "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || die "This launcher requires one node."
fi

module purge
module load singularity/3.8.7
export PROJECT_DIR="${PROJECT_DIR:-$HOME/trellis-singularity/work/MorphFlow}"
export SIF="${SIF:-$HOME/containers/trellis-py240-cu118.sif}"
TRELLIS_BASE="${TRELLIS_BASE:-$HOME/trellis-singularity}"
export ORIGINAL_ENV="${ORIGINAL_ENV:-$TRELLIS_BASE/env}"
export BENCH_ROOT="${BENCH_ROOT:-/hpc/scratch/marco.barezzi/morph-benchmark}"
BENCH_ARCHIVE="${BENCH_ARCHIVE:-/hpc/archive/G_VBD/marco.barezzi/morph-benchmark}"
export MORPHANY_ENV="${MORPHANY_ENV:-$BENCH_ROOT/morphany3d/env}"
export MORPHANY_REPO="${MORPHANY_REPO:-$BENCH_ROOT/morphany3d/work/MorphAny3D}"
export INTERP_ENV="${INTERP_ENV:-$BENCH_ROOT/interp3d/env}"
export INTERP_REPO="${INTERP_REPO:-$BENCH_ROOT/interp3d/work/Interp3D}"
export PROTOCOL_PYTHON="${PROTOCOL_PYTHON:-$BENCH_ROOT/protocol-python}"
TOOLCHAIN_ROOT_HOST="${TOOLCHAIN_ROOT_HOST:-$BENCH_ROOT/morphany3d/toolchain/cuda118-gcc11}"
SHARED_CACHE="${SHARED_CACHE:-$BENCH_ROOT/shared-cache}"
DATA_PARENT="${DATA_PARENT:-/hpc/scratch/marco.barezzi/3d_dataset}"
DATASET_NAME="${DATASET_NAME:-morphing_dataset_v4}"
SOURCE_IMAGES_NAME="${SOURCE_IMAGES_NAME:-flux_outputs}"
[[ "$DATASET_NAME" =~ ^[a-zA-Z0-9_-]+$ && "$SOURCE_IMAGES_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || die "Dataset and image directory names must be simple basenames."
export DATA_ROOT_HOST="$DATA_PARENT/$DATASET_NAME"
export SOURCE_IMAGES_ROOT_HOST="$DATA_PARENT/$SOURCE_IMAGES_NAME"
export METADATA="${METADATA:-metadata_test.json}"
export SOURCE_IMAGE_FILENAME="${SOURCE_IMAGE_FILENAME:-}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to your SS checkpoint on Ateneo HPC.}"
export SLAT_CHECKPOINT_PATH="${SLAT_CHECKPOINT_PATH:?Set SLAT_CHECKPOINT_PATH to your SLat checkpoint on Ateneo HPC.}"
export N="${N:-0}" SEED="${SEED:-42}"
export CFG_VALUES="${CFG_VALUES:-2 3 5}" SLAT_CFG_VALUES="${SLAT_CFG_VALUES:-1.5 2 2.5}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
# Fixed paper counts. An incompatible inherited setting fails before generation.
export K="${K:-5}" STEPS="${STEPS:-25}" SLAT_STEPS="${SLAT_STEPS:-25}"
export BETA_SHAPE="${BETA_SHAPE:-5}" NUM_VIEWS="${NUM_VIEWS:-64}" FID_NUM_VIEWS="${FID_NUM_VIEWS:-1000}"
export LPIPS_NET="${LPIPS_NET:-vgg}" FID_AGGREGATION="${FID_AGGREGATION:-both}" RENDER_BACKEND="${RENDER_BACKEND:-gaussian}"
# Independent, additional 50-frame source-proximity evaluation (48 uniform intermediates).
export WEIGHTED_DENSE="${WEIGHTED_DENSE:-1}"
export EXPORT_GLB="${EXPORT_GLB:-0}"
export TEXTURE_SIZE="${TEXTURE_SIZE:-1024}" SIMPLIFY="${SIMPLIFY:-0.95}" FILL_HOLES="${FILL_HOLES:-1}"
LPIPS_BATCH_SIZE="${LPIPS_BATCH_SIZE:-4}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-8}"
MORPHANY_TFSA_RAM_GB_PER_WORKER="${MORPHANY_TFSA_RAM_GB_PER_WORKER:-12}"
MORPHANY_TFSA_MIN_NODE_FREE_GB="${MORPHANY_TFSA_MIN_NODE_FREE_GB:-32}"
NGPUS="${NGPUS:-8}"
[[ "$NGPUS" =~ ^[1-8]$ ]] || die "NGPUS must be between 1 and 8 allocated GPUs."
CPU_THREADS_PER_WORKER="${CPU_THREADS_PER_WORKER:-$(( ${SLURM_CPUS_PER_TASK:-64} / NGPUS ))}"
(( CPU_THREADS_PER_WORKER >= 1 )) || die "Not enough CPUs per GPU worker."
RUN_NAME="${RUN_NAME:-interp3d_fulltest_${SLURM_JOB_ID:-check}}"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_.-]+$ && "$RUN_NAME" != . && "$RUN_NAME" != .. ]] || die "Invalid RUN_NAME."
RUN_ROOT="${RUN_ROOT:-$BENCH_ROOT/threeway-runs/$RUN_NAME}"
FINAL_ARCHIVE="${FINAL_ARCHIVE:-$BENCH_ARCHIVE/final-results/$RUN_NAME}"
HOST_PYTHON="${HOST_PYTHON:-python3}"
HELPER_HOST="$PROJECT_DIR/slurm/threeway_ateneo.py"

export TRELLIS_REPO_HOST="${TRELLIS_REPO_HOST:-}"
if [[ -z "$TRELLIS_REPO_HOST" ]]; then
    for candidate in "$TRELLIS_BASE/work/TRELLIS/MorphAny3D" "$TRELLIS_BASE/work/TRELLIS" "$PROJECT_DIR/../TRELLIS/MorphAny3D" "$PROJECT_DIR/../TRELLIS"; do
        if [[ -d "$candidate/trellis" ]]; then TRELLIS_REPO_HOST="$candidate"; break; fi
    done
fi
for directory in "$PROJECT_DIR" "$ORIGINAL_ENV" "$MORPHANY_ENV" "$INTERP_ENV" \
    "$MORPHANY_REPO/trellis" "$INTERP_REPO/trellis" "$TRELLIS_REPO_HOST/trellis" \
    "$DATA_ROOT_HOST" "$SOURCE_IMAGES_ROOT_HOST" "$TOOLCHAIN_ROOT_HOST" \
    "$SHARED_CACHE/hf-home" "$SHARED_CACHE/torch-home" "$SHARED_CACHE/u2net"; do
    [[ -d "$directory" ]] || die "Missing directory: $directory"
done
for file in "$SIF" "$HELPER_HOST" "$CHECKPOINT_PATH" "$SLAT_CHECKPOINT_PATH"; do
    [[ -f "$file" ]] || die "Missing file: $file"
done
[[ -d "$PROTOCOL_PYTHON" ]] || die "Run first: bash $PROJECT_DIR/slurm/prepare_threeway_interp3d_l40s.sh --prepare"
echo "Environment map (read-only):"
echo "  MorphFlow generation: $ORIGINAL_ENV"
echo "  MorphAny3D generation: $MORPHANY_ENV"
echo "  Interp3D generation: $INTERP_ENV"
echo "  Common rendering/metrics: $MORPHANY_ENV; evaluation overlay: $PROTOCOL_PYTHON"
echo "  The evaluation overlay is excluded from MorphFlow and Interp3D environments."
if ! env PROJECT_DIR="$PROJECT_DIR" SIF="$SIF" BENCH_ROOT="$BENCH_ROOT" \
    MORPHANY_ENV="$MORPHANY_ENV" MORPHANY_REPO="$MORPHANY_REPO" \
    TOOLCHAIN_ROOT_HOST="$TOOLCHAIN_ROOT_HOST" SHARED_CACHE="$SHARED_CACHE" \
    PROTOCOL_PYTHON="$PROTOCOL_PYTHON" \
    bash "$PROJECT_DIR/slurm/prepare_threeway_interp3d_l40s.sh" --check; then
    die "Run first on the login node: bash $PROJECT_DIR/slurm/prepare_threeway_interp3d_l40s.sh --prepare"
fi
mkdir -p "$RUN_ROOT" "$FINAL_ARCHIVE"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd -P)"
exec 9>"$RUN_ROOT/.benchmark.lock"
flock -n 9 || die "Another job is already writing to $RUN_ROOT"
"$HOST_PYTHON" "$HELPER_HOST" prepare --run-root "$RUN_ROOT"
# Every phase executes the copied code and the copied method forks.
SNAPSHOT="$RUN_ROOT/source_snapshot"
HELPER_HOST="$SNAPSHOT/MorphFlow/slurm/threeway_ateneo.py"
ACTUAL_PAIRS="$("$HOST_PYTHON" "$HELPER_HOST" count --run-root "$RUN_ROOT")"
WORKERS="$NGPUS"
(( WORKERS <= ACTUAL_PAIRS )) || WORKERS="$ACTUAL_PAIRS"
GRID_FILE="$RUN_ROOT/cfg_grid.txt"
"$HOST_PYTHON" "$HELPER_HOST" grid --run-root "$RUN_ROOT" > "$GRID_FILE"
JOB_LOG_DIR="$RUN_ROOT/logs/job_${SLURM_JOB_ID:-check}_$(date +%s)"
mkdir -p "$JOB_LOG_DIR"

# Respect nonempty Slurm CUDA allocations; never invent GPU IDs outside them.
GPU_TOKENS=()
if [[ "$MODE" == --run && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_TOKENS <<< "$CUDA_VISIBLE_DEVICES"
    (( ${#GPU_TOKENS[@]} >= NGPUS )) || die "CUDA_VISIBLE_DEVICES exposes fewer than NGPUS GPUs."
elif [[ "$MODE" == --run ]]; then
    mapfile -t GPU_TOKENS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
    (( ${#GPU_TOKENS[@]} == NGPUS )) || die "No explicit GPU mapping and nvidia-smi count differs from NGPUS."
fi
declare -A SEEN_GPU_TOKENS=()
for gpu in "${GPU_TOKENS[@]}"; do
    [[ -n "$gpu" && -z "${SEEN_GPU_TOKENS[$gpu]:-}" ]] || die "Empty or duplicate allocated GPU token."
    SEEN_GPU_TOKENS[$gpu]=1
done

# TFSA overflow lives on wn49 NVMe, with an explicit alternative if necessary.
LOCAL_PARENT="${NODE_LOCAL_PARENT:-/nvme/$USER/morph-benchmark}"
if [[ "$MODE" == --check ]]; then LOCAL_PARENT="$RUN_ROOT/check-tmp"; fi
mkdir -p "$LOCAL_PARENT"
NODE_TMP="$(mktemp -d "$LOCAL_PARENT/threeway-${SLURM_JOB_ID:-check}.XXXXXX")"
ACTIVE_PIDS=()
cleanup() {
    local rc=$?
    trap - EXIT
    if (( ${#ACTIVE_PIDS[@]} )); then
        for child in "${ACTIVE_PIDS[@]}"; do kill "$child" 2>/dev/null || true; done
        for child in "${ACTIVE_PIDS[@]}"; do wait "$child" 2>/dev/null || true; done
    fi
    rm -rf -- "$NODE_TMP"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ "$MODE" == --run ]]; then
    NVME_MIN_GB="${NVME_MIN_GB:-$(( WORKERS * 64 + 20 ))}"
    available_kb="$(df -Pk "$NODE_TMP" | awk 'NR==2 {print $4}')"
    (( available_kb >= NVME_MIN_GB * 1024 * 1024 )) || die "Need at least $NVME_MIN_GB GiB free for private per-worker temporary storage."
fi

SCRIPT_IN=/workspace/MorphFlow/benchmarks/threeway_textured_benchmark.py
EVAL_IN=/workspace/MorphFlow/benchmarks/eval_textured_benchmark.py
SPLIT_BINDS=()
for split_name in split_assets.json splits.json; do
    if [[ -f "$RUN_ROOT/input_snapshot/$split_name" ]]; then
        SPLIT_BINDS+=(--bind "$RUN_ROOT/input_snapshot/$split_name:/inputs/$DATASET_NAME/$split_name:ro")
    fi
done
container_exec() {
    local phase="$1" environment="$2" repository="$3" gpu="$4"
    shift 4
    local phase_home="$RUN_ROOT/runtime-home/$phase" phase_tmp="$NODE_TMP/$phase"
    local protocol_binds=()
    # This native extension is compiled against MorphAny3D's Torch/CUDA ABI.
    # Other model environments must use only their own installed packages.
    export SINGULARITYENV_MORPHFLOW_METRIC_OVERLAY=0
    if [[ "$environment" == "$MORPHANY_ENV" ]]; then
        protocol_binds=(--bind "$PROTOCOL_PYTHON:/opt/protocol-python:ro")
        export SINGULARITYENV_MORPHFLOW_METRIC_OVERLAY=1
    fi
    mkdir -p "$phase_home" "$phase_tmp"
    if [[ "$gpu" == cpu ]]; then
        export SINGULARITYENV_CUDA_VISIBLE_DEVICES=""
    else
        export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$gpu"
    fi
    export SINGULARITYENV_OMP_NUM_THREADS="$CPU_THREADS_PER_WORKER"
    export SINGULARITYENV_MKL_NUM_THREADS="$CPU_THREADS_PER_WORKER"
    export SINGULARITYENV_OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_WORKER"
    export SINGULARITYENV_NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_WORKER"
    exec singularity exec --cleanenv --containall --no-mount hostfs --nv \
        --bind "$environment:/opt/trellis/env:ro" \
        --bind "$TOOLCHAIN_ROOT_HOST:/opt/trellis/toolchain/cuda118-gcc11:ro" \
        --bind "$SNAPSHOT/MorphFlow:/workspace/MorphFlow:ro" \
        --bind "$repository:/workspace/MethodRepo:ro" \
        --bind "$DATA_PARENT:/inputs:ro" \
        --bind "$DATA_PARENT:$DATA_PARENT:ro" \
        "${SPLIT_BINDS[@]}" \
        --bind "$RUN_ROOT:/benchmark:rw" \
        --bind "$RUN_ROOT/checkpoints/ss:/checkpoints/ss:ro" \
        --bind "$RUN_ROOT/checkpoints/slat:/checkpoints/slat:ro" \
        --bind "$SHARED_CACHE/hf-home:/opt/trellis/hf-home:rw" \
        --bind "$SHARED_CACHE/torch-home:/opt/trellis/torch-home:rw" \
        --bind "$SHARED_CACHE/u2net:/opt/trellis/u2net:rw" \
        "${protocol_binds[@]}" \
        --bind "$phase_home:/runtime-home:rw" \
        --bind "$phase_tmp:/runtime-tmp:rw" \
        --pwd /workspace/MorphFlow "$SIF" bash -s -- "$@" <<'IN_CONTAINER'
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/trellis/env
export HOME=/runtime-home TMPDIR=/runtime-tmp XDG_CACHE_HOME=/runtime-home/.cache
export HF_HOME=/opt/trellis/hf-home TORCH_HOME=/opt/trellis/torch-home U2NET_HOME=/opt/trellis/u2net
export CUDA_HOME=/opt/trellis/toolchain/cuda118-gcc11
export CUDACXX="$CUDA_HOME/bin/nvcc" CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc" CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX" PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/opt/trellis/torch-home/extensions/nvdiffrast-035 MAX_JOBS=2
export ATTN_BACKEND=xformers SPCONV_ALGO=native PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRELLIS_REPO=/workspace/MethodRepo PYTHONPATH=/workspace/MethodRepo:/workspace/MorphFlow
if [[ "${MORPHFLOW_METRIC_OVERLAY:-0}" == 1 ]]; then
    export PYTHONPATH="$PYTHONPATH:/opt/protocol-python/gaussian:/opt/protocol-python"
fi
mkdir -p "$TORCH_EXTENSIONS_DIR"
exec "$@"
IN_CONTAINER
}
# Serial calls get an isolated shell; asynchronous workers call container_exec
# directly so their tracked PID becomes Singularity (and receives termination).
run_container() ( container_exec "$@" )

run_sharded() {
    local phase="$1" environment="$2" repository="$3"
    shift 3
    local shard finished=0 remaining="$WORKERS" child
    ACTIVE_PIDS=()
    echo "[$(date -Is)] START $phase; $WORKERS pair shards" | tee -a "$RUN_ROOT/progress.log"
    for ((shard=0; shard<WORKERS; shard++)); do
        container_exec "${phase}_g${shard}" "$environment" "$repository" "${GPU_TOKENS[$shard]}" \
            "$@" --shard-index "$shard" --num-shards "$WORKERS" \
            >"$JOB_LOG_DIR/${phase}_g${shard}.out" 2>"$JOB_LOG_DIR/${phase}_g${shard}.err" &
        ACTIVE_PIDS+=("$!")
    done
    while (( remaining > 0 )); do
        for ((shard=0; shard<WORKERS; shard++)); do
            child="${ACTIVE_PIDS[$shard]}"
            [[ "$child" != done ]] || continue
            if ! kill -0 "$child" 2>/dev/null; then
                if ! wait "$child"; then
                    echo "ERROR: $phase shard $shard failed" >&2
                    tail -n 35 "$JOB_LOG_DIR/${phase}_g${shard}.err" >&2 || true
                    for child in "${ACTIVE_PIDS[@]}"; do [[ "$child" == done ]] || kill "$child" 2>/dev/null || true; done
                    for child in "${ACTIVE_PIDS[@]}"; do [[ "$child" == done ]] || wait "$child" 2>/dev/null || true; done
                    ACTIVE_PIDS=()
                    die "Phase $phase incomplete. Reuse RUN_NAME to resume after fixing the cause."
                fi
                ACTIVE_PIDS[$shard]=done
                remaining=$((remaining - 1))
            fi
        done
        (( remaining == 0 )) || sleep 1
    done
    ACTIVE_PIDS=()
    echo "[$(date -Is)] DONE $phase" | tee -a "$RUN_ROOT/progress.log"
}

preflight_gpu=cpu
[[ "$MODE" == --check ]] || preflight_gpu="${GPU_TOKENS[0]}"
for spec in "main|$ORIGINAL_ENV|$SNAPSHOT/TRELLIS" "morphany3d|$MORPHANY_ENV|$SNAPSHOT/MorphAny3D" "interp3d|$INTERP_ENV|$SNAPSHOT/Interp3D"; do
    IFS='|' read -r name environment repository <<< "$spec"
    run_container "preflight_$name" "$environment" "$repository" "$preflight_gpu" python -c \
        'import sys, pathlib, torch, trellis; from trellis.pipelines import TrellisImageTo3DPipeline; from trellis.renderers import GaussianRenderer; assert pathlib.Path(trellis.__file__).resolve().is_relative_to(pathlib.Path("/workspace/MethodRepo")); print(sys.prefix, trellis.__file__, torch.__version__); assert sys.prefix == "/opt/trellis/env"; gpu=len(sys.argv)>1 and sys.argv[1]=="gpu"; assert not gpu or torch.cuda.device_count()==1; print("GPU:", torch.cuda.get_device_name(0) if gpu else "CPU check")' \
        "$([[ "$MODE" == --run ]] && echo gpu || echo cpu)"
    if [[ "$MODE" == --run && "$name" != main ]]; then
        # Same serialized complete-pipeline/DINO warmup as generate_interp3d_v4.
        # Check all cached model weights before spending time on the full test.
        run_container "pipeline_warmup_$name" "$environment" "$repository" "$preflight_gpu" python -c \
            'import torch; from trellis.pipelines import TrellisImageTo3DPipeline; p=TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large"); p.cuda(); print("Complete baseline pipeline/DINO cache OK", torch.cuda.get_device_name(0)); del p; torch.cuda.empty_cache()'
    fi
done
if [[ "$EXPORT_GLB" == 1 ]]; then
    run_container preflight_export "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$preflight_gpu" python -c \
        'import sys, runpy; import lpips, pytorch_fid, scipy, xatlas, pyvista, pymeshfix, igraph, cv2, diff_gaussian_rasterization; from trellis.utils import postprocessing_utils; import nvdiffrast.torch as dr; dr.RasterizeCudaContext() if sys.argv[1]=="gpu" else None; runpy.run_path("/workspace/MorphFlow/slurm/prepare_gaussian_rasterizer.py")["smoke_gpu"]() if sys.argv[1]=="gpu" else None; print("Gaussian renderer, GLB baking and metric imports OK")' \
        "$([[ "$MODE" == --run ]] && echo gpu || echo cpu)"
else
    run_container preflight_metrics "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$preflight_gpu" python -c \
        'import sys, runpy; import lpips, pytorch_fid, scipy, diff_gaussian_rasterization; from trellis.utils import render_utils; runpy.run_path("/workspace/MorphFlow/slurm/prepare_gaussian_rasterizer.py")["smoke_gpu"]() if sys.argv[1]=="gpu" else None; print("Gaussian renderer and metric imports OK; GLB baking deferred")' \
        "$([[ "$MODE" == --run ]] && echo gpu || echo cpu)"
fi
run_container fid_weights "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" cpu python "$EVAL_IN" check-fid-weights
run_container checkpoint_check "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" cpu python -c \
    'import json, torch; from pathlib import Path; from eval_validation_latents import detect_flow_target; paths=["/checkpoints/ss/checkpoint.pt","/checkpoints/slat/checkpoint.pt"]; rows=[]; targets=[]
for p in paths:
 c=torch.load(p,map_location="cpu",mmap=True,weights_only=False); t=detect_flow_target(c); targets.append(t); rows.append(dict(path=p,flow_target=t,epoch=c.get("epoch"),val_loss=c.get("val_loss"))); del c
assert targets==["ss","slat"],targets
text=json.dumps(rows,indent=2); p=Path("/benchmark/.checkpoint_metadata.json.tmp"); p.write_text(text+"\n"); p.replace("/benchmark/checkpoint_metadata.json"); print(text)' | tee "$JOB_LOG_DIR/checkpoint_check.out"
run_container plan "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" cpu python "$SCRIPT_IN" plan \
    --run-root /benchmark/shared --data-root "/inputs/$DATASET_NAME" \
    --metadata /benchmark/input_snapshot/metadata_test.json --source-images-root "/inputs/$SOURCE_IMAGES_NAME" \
    --source-image-filename "$SOURCE_IMAGE_FILENAME" --num-pairs "$N" --k "$K" --seed "$SEED" --alpha-schedule beta --beta-shape "$BETA_SHAPE"
if [[ "$WEIGHTED_DENSE" == 1 ]]; then
    run_container plan_weighted50 "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" cpu python "$SCRIPT_IN" plan \
        --run-root /benchmark/weighted_50/shared --data-root "/inputs/$DATASET_NAME" \
        --metadata /benchmark/input_snapshot/metadata_test.json --source-images-root "/inputs/$SOURCE_IMAGES_NAME" \
        --source-image-filename "$SOURCE_IMAGE_FILENAME" --num-pairs "$N" --k 48 --seed "$SEED" \
        --protocol weighted-endpoints-50 --alpha-schedule uniform
fi
if [[ "$MODE" == --check ]]; then
    echo "CHECK PASSED: $ACTUAL_PAIRS pairs, checkpoint types, frozen code, three isolated forks and metric dependencies; EXPORT_GLB=$EXPORT_GLB."
    exit 0
fi

evaluate_method() {
    local cell_in="$1" method="$2" output_in="$3" phase="$4" metric_mode="$5"
    local metric_args=() merge_args=()
    if [[ "$metric_mode" == weighted-endpoints-only ]]; then
        metric_args=(--metric-mode weighted-endpoints-only)
        merge_args=(--metric-mode weighted-endpoints-only)
    else
        metric_args=(--fid-num-views "$FID_NUM_VIEWS" --fid-batch-size "$FID_BATCH_SIZE" --fid-selection-seed "$SEED" --fid-aggregation both)
        merge_args=(--fid-aggregation both --fid-min-samples 512)
    fi
    run_sharded "${phase}_shard" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" python "$EVAL_IN" shard \
        --run-dir "$cell_in/$method" --output-dir "$output_in" --stage all \
        --num-views "$NUM_VIEWS" --resolution 512 \
        --camera-distribution hammersley --render-backend gaussian --appearance materials --lpips-net vgg \
        --lpips-batch-size "$LPIPS_BATCH_SIZE" "${metric_args[@]}" \
        --reference-cache-root /benchmark/shared/reference_metrics --device cuda
    run_container "${phase}_merge" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" cpu python "$EVAL_IN" merge \
        --output-dir "$output_in" "${merge_args[@]}" --expected-num-pairs "$ACTUAL_PAIRS"
}

run_evaluation_grid() {
local evaluation="$1" base_in="$2" phase_prefix="$3" metric_mode="$4"
local BASELINES_DONE=0 cfg slat_cfg CELL_HOST CELL_NAME CELL_IN PHASE_NAME method
while read -r cfg slat_cfg; do
    CELL_HOST="$("$HOST_PYTHON" "$HELPER_HOST" cell --run-root "$RUN_ROOT" --evaluation "$evaluation" --cfg "$cfg" --slat-cfg "$slat_cfg")"
    CELL_NAME="$(basename "$CELL_HOST")"
    CELL_IN="$base_in/$CELL_NAME"
    PHASE_NAME="${phase_prefix}${CELL_NAME}"
    run_sharded "${PHASE_NAME}_generation" "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" python "$SCRIPT_IN" morphflow \
        --run-root "$CELL_IN" --data-root "/inputs/$DATASET_NAME" --metadata /benchmark/input_snapshot/metadata_test.json \
        --source-images-root "/inputs/$SOURCE_IMAGES_NAME" --source-image-filename "$SOURCE_IMAGE_FILENAME" \
        --checkpoint-path /checkpoints/ss/checkpoint.pt --slat-checkpoint-path /checkpoints/slat/checkpoint.pt \
        --steps "$STEPS" --slat-steps "$SLAT_STEPS" --cfg-scale "$cfg" --slat-cfg-scale "$slat_cfg" \
        --mixed-precision "$MIXED_PRECISION" --trellis-model image_large
    if (( BASELINES_DONE == 0 )); then
        run_sharded "${phase_prefix}morphany3d_generation" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" python "$SCRIPT_IN" morphany3d \
            --run-root "$base_in/shared" --steps "$STEPS" --slat-steps "$SLAT_STEPS" --tfsa-cache-mode hybrid \
            --tfsa-ram-gb "$MORPHANY_TFSA_RAM_GB_PER_WORKER" --tfsa-min-node-free-gb "$MORPHANY_TFSA_MIN_NODE_FREE_GB" \
            --local-cache-root /runtime-tmp/morphany3d --tfsa-spill-root /runtime-tmp/tfsa-spill
        run_sharded "${phase_prefix}interp3d_generation" "$INTERP_ENV" "$SNAPSHOT/Interp3D" python "$SCRIPT_IN" interp3d \
            --run-root "$base_in/shared" --steps "$STEPS" --slat-steps "$SLAT_STEPS"
        if [[ "$EXPORT_GLB" == 1 ]]; then
            run_sharded "${phase_prefix}shared_glbs" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" python "$SCRIPT_IN" texture \
                --run-root "$base_in/shared" --methods morphany3d interp3d --texture-size "$TEXTURE_SIZE" --simplify "$SIMPLIFY" --fill-holes "$FILL_HOLES"
            run_container "${phase_prefix}shared_glb_audit" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" cpu python "$SCRIPT_IN" texture-status \
                --run-root "$base_in/shared" --methods morphany3d interp3d
        fi
        for method in morphany3d interp3d; do
            evaluate_method "$base_in/shared" "$method" "$base_in/shared/metrics/$method" "${phase_prefix}${method}" "$metric_mode"
        done
        BASELINES_DONE=1
    fi
    if [[ "$EXPORT_GLB" == 1 ]]; then
        run_sharded "${PHASE_NAME}_glbs" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" python "$SCRIPT_IN" texture \
            --run-root "$CELL_IN" --methods morphflow --texture-size "$TEXTURE_SIZE" --simplify "$SIMPLIFY" --fill-holes "$FILL_HOLES"
        run_container "${PHASE_NAME}_glb_audit" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" cpu python "$SCRIPT_IN" texture-status --run-root "$CELL_IN"
    fi
    evaluate_method "$CELL_IN" morphflow "$CELL_IN/metrics/morphflow" "${PHASE_NAME}_metrics" "$metric_mode"
    # Baseline timing files exist after their first generation phase.
    "$HOST_PYTHON" "$HELPER_HOST" cell --run-root "$RUN_ROOT" --evaluation "$evaluation" --cfg "$cfg" --slat-cfg "$slat_cfg" >/dev/null
    run_container "${PHASE_NAME}_aggregate" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" cpu python "$SCRIPT_IN" aggregate --run-root "$CELL_IN"
    "$HOST_PYTHON" "$HELPER_HOST" summarize --run-root "$RUN_ROOT" --evaluation "$evaluation"
    run_container "${PHASE_NAME}_weighted_lpips" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" cpu \
        python /workspace/MorphFlow/benchmarks/weighted_endpoint_report.py --run-root "$base_in"
done < "$GRID_FILE"
"$HOST_PYTHON" "$HELPER_HOST" summarize --run-root "$RUN_ROOT" --evaluation "$evaluation" --require-complete
}

run_evaluation_grid interp3d /benchmark "" all
if [[ "$WEIGHTED_DENSE" == 1 ]]; then
    run_evaluation_grid weighted50 /benchmark/weighted_50 weighted50_ weighted-endpoints-only
fi

# Compact reports in archive; all raw assets, GLBs, images and features remain in RUN_ROOT.
cp "$RUN_ROOT/run_manifest.json" "$RUN_ROOT/checkpoint_metadata.json" "$FINAL_ARCHIVE/"
archive_evaluation() {
local source_root="$1" archive_root="$2" file directory destination
mkdir -p "$archive_root"
cp "$source_root/cfg_grid_metrics.csv" "$source_root/cfg_grid_metrics.json" "$archive_root/"
cp -R "$source_root/weighted_endpoint_lpips" "$archive_root/"
cp "$source_root/shared/plan.json" "$archive_root/plan.json"
for file in "$source_root"/cfg_ss_*/final_metrics.{json,csv} "$source_root"/cfg_ss_*/generation_timings.{json,csv} "$source_root"/cfg_ss_*/texture_status.json; do
    [[ -f "$file" ]] || continue
    destination="$archive_root/$(basename "$(dirname "$file")")"
    mkdir -p "$destination"
    cp "$file" "$destination/"
done
for directory in "$source_root"/cfg_ss_*/weighted_endpoint_lpips; do
    [[ -d "$directory" ]] || continue
    destination="$archive_root/$(basename "$(dirname "$directory")")"
    mkdir -p "$destination"
    cp -R "$directory" "$destination/"
done
}
archive_evaluation "$RUN_ROOT" "$FINAL_ARCHIVE"
if [[ "$WEIGHTED_DENSE" == 1 ]]; then
    archive_evaluation "$RUN_ROOT/weighted_50" "$FINAL_ARCHIVE/weighted_50"
    echo "Dense 50-frame reports: $RUN_ROOT/weighted_50/cfg_grid_metrics.csv"
    echo "Dense evolution: $RUN_ROOT/weighted_50/weighted_endpoint_lpips/evolution.png"
fi
echo "DONE: $RUN_ROOT/cfg_grid_metrics.csv"
if [[ "$EXPORT_GLB" == 1 ]]; then
    echo "GLBs: $RUN_ROOT/cfg_ss_*/morphflow/pair_*/alpha_*/pred_final.glb"
    echo "Baselines: $RUN_ROOT/shared/{morphany3d,interp3d}/pair_*/alpha_*/pred_final.glb"
else
    echo "GLB baking deferred for all models and protocols; mesh_raw.pt and Gaussian appearance retained in $RUN_ROOT."
fi
echo "Archive reports: $FINAL_ARCHIVE"
echo "Weighted LPIPS comparison: $RUN_ROOT/weighted_endpoint_lpips/evolution.png"
