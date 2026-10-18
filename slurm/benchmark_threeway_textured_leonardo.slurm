#!/bin/bash
#SBATCH --job-name=threeway_v4_cfggrid
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=BEGIN,END,FAIL

# Native Leonardo, one GPU. Paper Interp3D metrics on held-out v4 pairs.
# --check: offline CPU validation; --run: sbatch GPU benchmark.
# Resubmit the SAME RUN_ROOT/configuration to resume completed pairs/phases.
# Baselines and references are shared across the nine MorphFlow CFG cells.
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }
MODE="${1:---run}"
case "$MODE" in --run|--check) ;; *) die "Usage: $0 [--check|--run]" ;; esac
[[ $# -le 1 ]] || die "Configure the benchmark with environment variables."
if [[ "$MODE" == --run ]]; then
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "Submit --run with sbatch."
    [[ "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || die "This launcher requires one node."
fi

export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
BASE_DIR="${BASE_DIR:-$WORK/$USER}"
source "${MF_ENV_FILE:-$BASE_DIR/env.sh}"
export PROJECT_DIR="${PROJECT_DIR:-$MORPHFLOW_REPO}"
export MORPHANY_ENV="${MORPHANY_ENV:-$BASE_DIR/envs/morphany3d}"
export INTERP_ENV="${INTERP_ENV:-$BASE_DIR/envs/interp3d}"
export MORPHANY_REPO="${MORPHANY_REPO:-$BASE_DIR/src/MorphAny3D}"
export INTERP_REPO="${INTERP_REPO:-$BASE_DIR/src/Interp3D}"
export ROOT_DIR="${ROOT_DIR:-$FAST/$USER/datasets/morphing_dataset_v4}"
export SOURCE_IMAGES_ROOT="${SOURCE_IMAGES_ROOT:-$FAST/$USER/datasets/flux_outputs}"
export SOURCE_IMAGE_FILENAME="${SOURCE_IMAGE_FILENAME:-}"
export METADATA="${METADATA:-metadata_test.json}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a stable v4 SS checkpoint snapshot.}"
export SLAT_CHECKPOINT_PATH="${SLAT_CHECKPOINT_PATH:?Set SLAT_CHECKPOINT_PATH to a stable v4 SLat checkpoint snapshot.}"
export N="${N:-100}" K="${K:-5}" SEED="${SEED:-42}"
export STEPS="${STEPS:-25}" SLAT_STEPS="${SLAT_STEPS:-25}"
export CFG_VALUES="${CFG_VALUES:-2 3 5}" SLAT_CFG_VALUES="${SLAT_CFG_VALUES:-1.5 2 2.5}"
export ALPHA_SCHEDULE="${ALPHA_SCHEDULE:-beta}" BETA_SHAPE="${BETA_SHAPE:-5}"
export NUM_VIEWS="${NUM_VIEWS:-64}" FID_NUM_VIEWS="${FID_NUM_VIEWS:-1000}"
export RENDER_RESOLUTION="${RENDER_RESOLUTION:-512}" LPIPS_NET="${LPIPS_NET:-vgg}"
export FID_MIN_SAMPLES="${FID_MIN_SAMPLES:-512}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export RENDER_BACKEND="${RENDER_BACKEND:-gaussian}"
export BAKE_TEXTURES="${BAKE_TEXTURES:-0}"
export RUN_ROOT="${RUN_ROOT:-$BASE_DIR/outputs/threeway_benchmark/threeway_v4_beta5_n${N}_k${K}_seed${SEED}}"
export U2NET_HOME="${U2NET_HOME:-$BASE_DIR/cache/u2net}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$BASE_DIR/cache/torch_extensions/threeway}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MAX_JOBS="${MAX_JOBS:-4}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BLENDER_BIN="${BLENDER_BIN:-$BASE_DIR/tools/blender-3.0.1-linux-x64/blender}"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$BASE_DIR/logs"
cd "$PROJECT_DIR"
SCRIPT="$PROJECT_DIR/benchmarks/threeway_textured_benchmark.py"
EVALUATOR="$PROJECT_DIR/benchmarks/eval_textured_benchmark.py"
HELPER="$PROJECT_DIR/slurm/threeway_leonardo.py"
MAIN_ENV="$MORPHFLOW_ENV"
METADATA_PATH="$METADATA"
[[ "$METADATA_PATH" == /* ]] || METADATA_PATH="$ROOT_DIR/$METADATA_PATH"

# Each process sees one TRELLIS implementation. Main training env stays intact.
run_python() {
    local environment="$1" repository="$2"
    shift 2
    env PYTHONPATH="$repository:$PROJECT_DIR" PATH="$environment/bin:$PATH" \
        "$environment/bin/python" "$@"
}
for path in "$MAIN_ENV/bin/python" "$MORPHANY_ENV/bin/python" "$INTERP_ENV/bin/python" \
            "$CHECKPOINT_PATH" "$SLAT_CHECKPOINT_PATH" "$METADATA_PATH"; do
    [[ -r "$path" ]] || die "Missing input: $path"
done
for path in "$TRELLIS_REPO" "$MORPHANY_REPO" "$INTERP_REPO" "$SOURCE_IMAGES_ROOT"; do
    [[ -d "$path" ]] || die "Missing directory: $path"
done
run_python "$MAIN_ENV" "$TRELLIS_REPO" "$HELPER" check
for pair in "$MORPHANY_ENV:$MORPHANY_REPO" "$INTERP_ENV:$INTERP_REPO"; do
    run_python "${pair%%:*}" "${pair#*:}" -c \
        'import trellis, torch, lpips, scipy; from trellis.pipelines import TrellisImageTo3DPipeline; from trellis.renderers import GaussianRenderer; import diff_gaussian_rasterization; print("Baseline import OK:", trellis.__file__, torch.__version__)'
done
run_python "$MORPHANY_ENV" "$MORPHANY_REPO" "$EVALUATOR" check-fid-weights
if [[ "$BAKE_TEXTURES" == 1 || "$RENDER_BACKEND" == blender ]]; then
    [[ -x "$BLENDER_BIN" ]] || die "Blender not executable: $BLENDER_BIN"
    "$BLENDER_BIN" --version | head -1
fi
if [[ "$MODE" == --check ]]; then
    echo "CHECK PASSED: v4 checkpoints, dataset, isolated baseline imports and cached weights."
    exit 0
fi
run_python "$MORPHANY_ENV" "$MORPHANY_REPO" -c \
    'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1, "Expected one allocated GPU"; print(torch.cuda.get_device_name(0))'
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/.benchmark.lock"
flock -n 9 || die "A benchmark is already writing to $RUN_ROOT"
run_python "$MAIN_ENV" "$TRELLIS_REPO" "$HELPER" initialize

# Compute-node /tmp is a small tmpfs (10 GiB on the verified A100 node).
# TFSA can exhaust it during a single alpha; use the run's WORK allocation.
LOCAL_PARENT="${THREEWAY_TMP_ROOT:-$RUN_ROOT/runtime-tmp}"
mkdir -p "$LOCAL_PARENT"
TASK_TMP="$(mktemp -d "$LOCAL_PARENT/threeway-${SLURM_JOB_ID}.XXXXXX")"
cleanup() { rm -rf -- "$TASK_TMP"; }
trap cleanup EXIT
export TMPDIR="$TASK_TMP"
SHARED="$RUN_ROOT/shared"
mkdir -p "$SHARED"
run_python "$MAIN_ENV" "$TRELLIS_REPO" "$SCRIPT" plan \
    --run-root "$SHARED" --data-root "$ROOT_DIR" --metadata "$METADATA" \
    --source-images-root "$SOURCE_IMAGES_ROOT" --source-image-filename "$SOURCE_IMAGE_FILENAME" \
    --num-pairs "$N" --k "$K" --seed "$SEED" --alpha-schedule "$ALPHA_SCHEDULE" --beta-shape "$BETA_SHAPE"

evaluate_method() {
    local cell_root="$1" method="$2" output="$3"
    if [[ -f "$output/metrics_batch.json" ]]; then
        echo "Reuse completed metrics: $output"
        return
    fi
    run_python "$MORPHANY_ENV" "$MORPHANY_REPO" "$EVALUATOR" shard \
        --run-dir "$cell_root/$method" --output-dir "$output" --stage all \
        --num-views "$NUM_VIEWS" --fid-num-views "$FID_NUM_VIEWS" \
        --resolution "$RENDER_RESOLUTION" --camera-distribution hammersley \
        --render-backend "$RENDER_BACKEND" --appearance materials --lpips-net "$LPIPS_NET" \
        --lpips-batch-size 8 --fid-batch-size 16 --fid-selection-seed "$SEED" \
        --reference-cache-root "$SHARED/reference_metrics" \
        --device cuda --blender-bin "$BLENDER_BIN"
    run_python "$MORPHANY_ENV" "$MORPHANY_REPO" "$EVALUATOR" merge \
        --output-dir "$output" --fid-min-samples "$FID_MIN_SAMPLES" --expected-num-pairs "$N"
}

read -r -a cfg_values <<< "$CFG_VALUES"
read -r -a slat_values <<< "$SLAT_CFG_VALUES"
BASELINES_DONE=0
for cfg in "${cfg_values[@]}"; do
    for slat_cfg in "${slat_values[@]}"; do
        CELL="$(run_python "$MAIN_ENV" "$TRELLIS_REPO" "$HELPER" cell --cfg "$cfg" --slat-cfg "$slat_cfg")"
        echo "[$(date -Is)] CFG SS=$cfg SLat=$slat_cfg; $CELL"
        if [[ -f "$CELL/final_metrics.json" && -f "$CELL/final_metrics.csv" ]]; then
            echo "Reuse completed CFG cell"
            continue
        fi
        run_python "$MAIN_ENV" "$TRELLIS_REPO" "$SCRIPT" morphflow \
            --run-root "$CELL" --data-root "$ROOT_DIR" --metadata "$METADATA" \
            --source-images-root "$SOURCE_IMAGES_ROOT" --source-image-filename "$SOURCE_IMAGE_FILENAME" \
            --checkpoint-path "$CHECKPOINT_PATH" --slat-checkpoint-path "$SLAT_CHECKPOINT_PATH" \
            --steps "$STEPS" --slat-steps "$SLAT_STEPS" --cfg-scale "$cfg" --slat-cfg-scale "$slat_cfg" \
            --mixed-precision "$MIXED_PRECISION" --trellis-model image_large
        if [[ "$BASELINES_DONE" == 0 ]]; then
            run_python "$MORPHANY_ENV" "$MORPHANY_REPO" "$SCRIPT" morphany3d \
                --run-root "$SHARED" --steps "$STEPS" --slat-steps "$SLAT_STEPS" \
                --tfsa-cache-mode hybrid --tfsa-ram-gb 12 --tfsa-min-node-free-gb 20 \
                --local-cache-root "$TASK_TMP/morphany3d"
            run_python "$INTERP_ENV" "$INTERP_REPO" "$SCRIPT" interp3d \
                --run-root "$SHARED" --steps "$STEPS" --slat-steps "$SLAT_STEPS"
            BASELINES_DONE=1
        fi
        if [[ "$BAKE_TEXTURES" == 1 || "$RENDER_BACKEND" == blender ]]; then
            run_python "$MORPHANY_ENV" "$MORPHANY_REPO" "$SCRIPT" texture \
                --run-root "$CELL" --texture-size 1024 --simplify 0.95 --fill-holes 1
        fi
        for method in morphany3d interp3d morphflow; do
            evaluate_method "$CELL" "$method" "$CELL/metrics/$method"
        done
        # Baseline timing logs are generated once alongside the shared outputs.
        mkdir -p "$CELL/timings"
        for timing in "$SHARED"/timings/*.jsonl; do
            [[ ! -f "$timing" ]] || ln -sfn "$timing" "$CELL/timings/$(basename "$timing")"
        done
        run_python "$MORPHANY_ENV" "$MORPHANY_REPO" "$SCRIPT" aggregate --run-root "$CELL"
        run_python "$MAIN_ENV" "$TRELLIS_REPO" "$HELPER" summarize
    done
done
run_python "$MAIN_ENV" "$TRELLIS_REPO" "$HELPER" summarize
echo "DONE: $RUN_ROOT/cfg_grid_metrics.csv"
