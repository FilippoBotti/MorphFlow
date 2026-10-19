#!/bin/bash
#SBATCH --job-name=interp3ddata_threeway
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

# Native Leonardo environments: one A100, sequential phases, resumable outputs.
# --check indexes images and validates paths/configuration without model imports.
# --run prepares shared canonical endpoints on the GPU, then both protocols.
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }
MODE="${1:---run}"
case "$MODE" in --check|--run) ;; *) die "Usage: $0 [--check|--run]" ;; esac
[[ $# -le 1 ]] || die "Configure this launcher with environment variables."
if [[ "$MODE" == --run ]]; then
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "Submit --run with sbatch."
    [[ "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || die "This launcher requires one node."
fi

export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
BASE_DIR="${BASE_DIR:-$WORK/$USER}"
source "${MF_ENV_FILE:-$BASE_DIR/env.sh}"
export EXECUTION_BACKEND=native
export PROJECT_DIR="${PROJECT_DIR:-$MORPHFLOW_REPO}"
export ORIGINAL_ENV="${ORIGINAL_ENV:-$MORPHFLOW_ENV}"
export TRELLIS_REPO_HOST="${TRELLIS_REPO_HOST:-$TRELLIS_REPO}"
export MORPHANY_ENV="${MORPHANY_ENV:-$BASE_DIR/envs/morphany3d}"
export INTERP_ENV="${INTERP_ENV:-$BASE_DIR/envs/interp3d}"
export MORPHANY_REPO="${MORPHANY_REPO:-$BASE_DIR/src/MorphAny3D}"
export INTERP_REPO="${INTERP_REPO:-$BASE_DIR/src/Interp3D}"
export DATASET_INPUT_ROOT="${DATASET_INPUT_ROOT:-$BASE_DIR/datasets/Interp3DData}"
export DATA_ROOT_HOST="${DATA_ROOT_HOST:-$BASE_DIR/datasets/interp3ddata_prepared}"
export SOURCE_IMAGES_ROOT_HOST="${SOURCE_IMAGES_ROOT_HOST:-$DATA_ROOT_HOST/images}"
export METADATA="${METADATA:-metadata_test.json}"
export SOURCE_IMAGE_FILENAME="${SOURCE_IMAGE_FILENAME:-}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$BASE_DIR/checkpoints/ateneo_v4_ss17/ss_best_e17.pt}"
export SLAT_CHECKPOINT_PATH="${SLAT_CHECKPOINT_PATH:-$BASE_DIR/checkpoints/v4_interp3d_slat_dual_tgate_p25g4_lr5e-5_bs1_leonardo/morphflow_best.pt}"
export N="${N:-0}" SEED="${SEED:-42}"
export CFG_VALUES="${CFG_VALUES:-1 2 3 5 7.5}" SLAT_CFG_VALUES="${SLAT_CFG_VALUES:-1 1.5 2}"
export K="${K:-5}" STEPS="${STEPS:-25}" SLAT_STEPS="${SLAT_STEPS:-25}" BETA_SHAPE="${BETA_SHAPE:-5}"
export NUM_VIEWS="${NUM_VIEWS:-64}" FID_NUM_VIEWS="${FID_NUM_VIEWS:-1000}"
export LPIPS_NET="${LPIPS_NET:-vgg}" FID_AGGREGATION="${FID_AGGREGATION:-both}" RENDER_BACKEND="${RENDER_BACKEND:-gaussian}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}" WEIGHTED_DENSE="${WEIGHTED_DENSE:-1}" EXPORT_GLB="${EXPORT_GLB:-0}"
export TEXTURE_SIZE="${TEXTURE_SIZE:-1024}" SIMPLIFY="${SIMPLIFY:-0.95}" FILL_HOLES="${FILL_HOLES:-1}"
export U2NET_HOME="${U2NET_HOME:-$BASE_DIR/cache/u2net}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$BASE_DIR/cache/torch_extensions/threeway}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MAX_JOBS="${MAX_JOBS:-2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_NAME="${RUN_NAME:-interp3ddata_ss17_slatbest_cfg15_n${N}_seed${SEED}_raw}"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_.-]+$ && "$RUN_NAME" != . && "$RUN_NAME" != .. ]] || die "Invalid RUN_NAME."
export RUN_ROOT="${RUN_ROOT:-$FAST/$USER/outputs/threeway_benchmark/$RUN_NAME}"
FINAL_ARCHIVE="${FINAL_ARCHIVE:-$BASE_DIR/outputs/threeway_benchmark/$RUN_NAME}"
LPIPS_BATCH_SIZE="${LPIPS_BATCH_SIZE:-4}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-8}"
MORPHANY_TFSA_RAM_GB="${MORPHANY_TFSA_RAM_GB:-12}"
MORPHANY_TFSA_MIN_FREE_GB="${MORPHANY_TFSA_MIN_FREE_GB:-20}"
HELPER="$PROJECT_DIR/slurm/threeway_ateneo.py"
PREPARER="$PROJECT_DIR/benchmarks/prepare_interp3ddata.py"
mkdir -p "$BASE_DIR/logs" "$TORCH_EXTENSIONS_DIR"
cd "$PROJECT_DIR"

run_python() {
    local environment="$1" repository="$2"
    shift 2
    env TRELLIS_REPO="$repository" PYTHONPATH="$repository:$ACTIVE_PROJECT" \
        PATH="$environment/bin:$PATH" "$environment/bin/python" "$@"
}
ACTIVE_PROJECT="$PROJECT_DIR"
for path in "$ORIGINAL_ENV/bin/python" "$MORPHANY_ENV/bin/python" "$INTERP_ENV/bin/python" \
    "$CHECKPOINT_PATH" "$SLAT_CHECKPOINT_PATH" "$HELPER" "$PREPARER"; do
    [[ -f "$path" && -r "$path" ]] || die "Missing input: $path"
done
for directory in "$DATASET_INPUT_ROOT" "$TRELLIS_REPO_HOST/trellis" "$MORPHANY_REPO/trellis" "$INTERP_REPO/trellis"; do
    [[ -d "$directory" ]] || die "Missing directory: $directory"
done
echo "Native environment map:"
echo "  MorphFlow: $ORIGINAL_ENV; $TRELLIS_REPO_HOST"
echo "  MorphAny3D and common metrics: $MORPHANY_ENV; $MORPHANY_REPO"
echo "  Interp3D and canonical endpoint preparation: $INTERP_ENV; $INTERP_REPO"
echo "  Raw input: $DATASET_INPUT_ROOT; prepared dataset: $DATA_ROOT_HOST"
run_python "$INTERP_ENV" "$INTERP_REPO" "$PREPARER" \
    --input-root "$DATASET_INPUT_ROOT" --output-root "$DATA_ROOT_HOST" \
    --seed "$SEED" --steps "$STEPS" --slat-steps "$SLAT_STEPS" --index-only
run_python "$ORIGINAL_ENV" "$TRELLIS_REPO_HOST" -c \
    'import os,runpy; runpy.run_path(os.environ["PROJECT_DIR"]+"/slurm/threeway_ateneo.py")["requested_configuration"](os.environ); print("Native configuration and protocol counts OK")'
if [[ "$MODE" == --check ]]; then
    echo "CHECK PASSED: image index, env/checkpoint paths and protocol configuration. Model imports, checkpoint types and canonical endpoint sampling are checked in the GPU job."
    exit 0
fi
for spec in "$ORIGINAL_ENV|$TRELLIS_REPO_HOST" "$MORPHANY_ENV|$MORPHANY_REPO" "$INTERP_ENV|$INTERP_REPO"; do
    IFS='|' read -r environment repository <<< "$spec"
    run_python "$environment" "$repository" -c \
        'import pathlib,sys,torch,trellis; from trellis.pipelines import TrellisImageTo3DPipeline; from trellis.renderers import GaussianRenderer; assert pathlib.Path(trellis.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve()); assert pathlib.Path(sys.prefix).resolve()==pathlib.Path(sys.argv[2]).resolve(); print("Native import OK:",sys.prefix,trellis.__file__,torch.__version__)' \
        "$repository" "$environment"
done
run_python "$MORPHANY_ENV" "$MORPHANY_REPO" -c \
    'import torch,lpips,pytorch_fid,scipy,diff_gaussian_rasterization; fields=set(diff_gaussian_rasterization.GaussianRasterizationSettings._fields); assert {"kernel_size","subpixel_offset"}<=fields; print("Common Gaussian renderer and metric imports OK")'
run_python "$MORPHANY_ENV" "$MORPHANY_REPO" "$PROJECT_DIR/benchmarks/eval_textured_benchmark.py" check-fid-weights
run_python "$ORIGINAL_ENV" "$TRELLIS_REPO_HOST" -c \
    'import json,sys,torch; from eval_validation_latents import detect_flow_target; rows=[]
for filename,expected in zip(sys.argv[1:],["ss","slat"]):
 checkpoint=torch.load(filename,map_location="cpu",mmap=True,weights_only=False); actual=detect_flow_target(checkpoint); assert actual==expected,(filename,actual); rows.append(dict(path=filename,flow_target=actual,epoch=checkpoint.get("epoch"),val_loss=checkpoint.get("val_loss"))); del checkpoint
print(json.dumps(rows,indent=2))' "$CHECKPOINT_PATH" "$SLAT_CHECKPOINT_PATH"
run_python "$MORPHANY_ENV" "$MORPHANY_REPO" -c \
    'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1,"Expected one allocated GPU"; print("Allocated GPU:",torch.cuda.get_device_name(0))'
mkdir -p "$RUN_ROOT" "$FINAL_ARCHIVE"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd -P)"
FINAL_ARCHIVE="$(cd "$FINAL_ARCHIVE" && pwd -P)"
exec 9>"$RUN_ROOT/.benchmark.lock"
flock -n 9 || die "Another job is already writing to $RUN_ROOT"
run_python "$ORIGINAL_ENV" "$TRELLIS_REPO_HOST" "$HELPER" prepare --run-root "$RUN_ROOT"
SNAPSHOT="$RUN_ROOT/source_snapshot"
ACTIVE_PROJECT="$SNAPSHOT/MorphFlow"
HELPER="$ACTIVE_PROJECT/slurm/threeway_ateneo.py"
SCRIPT="$ACTIVE_PROJECT/benchmarks/threeway_textured_benchmark.py"
EVALUATOR="$ACTIVE_PROJECT/benchmarks/eval_textured_benchmark.py"
PREPARER="$ACTIVE_PROJECT/benchmarks/prepare_interp3ddata.py"
ACTUAL_PAIRS="$(run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$HELPER" count --run-root "$RUN_ROOT")"
GRID_FILE="$RUN_ROOT/cfg_grid.txt"
run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$HELPER" grid --run-root "$RUN_ROOT" > "$GRID_FILE"
JOB_LOG_DIR="$RUN_ROOT/logs/job_${SLURM_JOB_ID}_$(date +%s)"
mkdir -p "$JOB_LOG_DIR"
LOCAL_PARENT="${THREEWAY_TMP_ROOT:-$RUN_ROOT/runtime-tmp}"
mkdir -p "$LOCAL_PARENT"
TASK_TMP="$(mktemp -d "$LOCAL_PARENT/interp3ddata-${SLURM_JOB_ID}.XXXXXX")"
cleanup() { local status=$?; trap - EXIT; rm -rf -- "$TASK_TMP"; exit "$status"; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export TMPDIR="$TASK_TMP"
echo "Run: $RUN_ROOT"
echo "Phase logs: $JOB_LOG_DIR"
run_phase() {
    local phase="$1" environment="$2" repository="$3" status
    shift 3
    echo "[$(date -Is)] START $phase" | tee -a "$RUN_ROOT/progress.log"
    if run_python "$environment" "$repository" "$@" > "$JOB_LOG_DIR/$phase.out" 2> "$JOB_LOG_DIR/$phase.err"; then
        echo "[$(date -Is)] DONE $phase" | tee -a "$RUN_ROOT/progress.log"
    else
        status=$?
        echo "ERROR: $phase failed; see $JOB_LOG_DIR/$phase.{out,err}" >&2
        tail -n 40 "$JOB_LOG_DIR/$phase.err" >&2 || true
        return "$status"
    fi
}
run_phase gaussian_smoke "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" -c \
    'import runpy,sys; runpy.run_path(sys.argv[1])["smoke_gpu"]()' "$ACTIVE_PROJECT/slurm/prepare_gaussian_rasterizer.py"
run_phase canonical_endpoints "$INTERP_ENV" "$SNAPSHOT/Interp3D" "$PREPARER" \
    --input-root "$DATASET_INPUT_ROOT" --output-root "$DATA_ROOT_HOST" \
    --seed "$SEED" --steps "$STEPS" --slat-steps "$SLAT_STEPS" --prepare
run_phase checkpoint_metadata "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" -c \
    'import json,sys,torch; from pathlib import Path; from eval_validation_latents import detect_flow_target; rows=[]
for filename in sys.argv[2:]:
 checkpoint=torch.load(filename,map_location="cpu",mmap=True,weights_only=False); rows.append(dict(path=filename,flow_target=detect_flow_target(checkpoint),epoch=checkpoint.get("epoch"),val_loss=checkpoint.get("val_loss"))); del checkpoint
Path(sys.argv[1]).write_text(json.dumps(rows,indent=2)+"\n"); print(json.dumps(rows,indent=2))' \
    "$RUN_ROOT/checkpoint_metadata.json" "$RUN_ROOT/checkpoints/ss/checkpoint.pt" "$RUN_ROOT/checkpoints/slat/checkpoint.pt"
run_phase plan "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$SCRIPT" plan \
    --run-root "$RUN_ROOT/shared" --data-root "$DATA_ROOT_HOST" --metadata "$RUN_ROOT/input_snapshot/metadata_test.json" \
    --source-images-root "$SOURCE_IMAGES_ROOT_HOST" --source-image-filename "$SOURCE_IMAGE_FILENAME" \
    --num-pairs "$N" --k "$K" --seed "$SEED" --alpha-schedule beta --beta-shape "$BETA_SHAPE"
if [[ "$WEIGHTED_DENSE" == 1 ]]; then
    run_phase plan_weighted50 "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$SCRIPT" plan \
        --run-root "$RUN_ROOT/weighted_50/shared" --data-root "$DATA_ROOT_HOST" --metadata "$RUN_ROOT/input_snapshot/metadata_test.json" \
        --source-images-root "$SOURCE_IMAGES_ROOT_HOST" --source-image-filename "$SOURCE_IMAGE_FILENAME" \
        --num-pairs "$N" --k 48 --seed "$SEED" --protocol weighted-endpoints-50 --alpha-schedule uniform
fi

evaluate_method() {
    local cell="$1" method="$2" output="$3" phase="$4" metric_mode="$5"
    local metric_args=() merge_args=()
    if [[ "$metric_mode" == weighted-endpoints-only ]]; then
        metric_args=(--metric-mode weighted-endpoints-only)
        merge_args=(--metric-mode weighted-endpoints-only)
    else
        metric_args=(--fid-num-views "$FID_NUM_VIEWS" --fid-batch-size "$FID_BATCH_SIZE" --fid-selection-seed "$SEED" --fid-aggregation both)
        merge_args=(--fid-aggregation both --fid-min-samples 512)
    fi
    run_phase "${phase}_metrics" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$EVALUATOR" shard \
        --run-dir "$cell/$method" --output-dir "$output" --stage all --shard-index 0 --num-shards 1 \
        --num-views "$NUM_VIEWS" --resolution 512 --camera-distribution hammersley --render-backend gaussian \
        --appearance materials --lpips-net vgg --lpips-batch-size "$LPIPS_BATCH_SIZE" "${metric_args[@]}" \
        --reference-cache-root "$RUN_ROOT/shared/reference_metrics" --device cuda
    run_phase "${phase}_merge" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$EVALUATOR" merge \
        --output-dir "$output" "${merge_args[@]}" --expected-num-pairs "$ACTUAL_PAIRS"
}
run_evaluation_grid() {
    local evaluation="$1" base="$2" prefix="$3" metric_mode="$4"
    local baselines_done=0 cfg slat_cfg cell phase method
    while read -r cfg slat_cfg; do
        cell="$(run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$HELPER" cell --run-root "$RUN_ROOT" --evaluation "$evaluation" --cfg "$cfg" --slat-cfg "$slat_cfg")"
        phase="${prefix}$(basename "$cell")"
        run_phase "${phase}_generation" "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$SCRIPT" morphflow \
            --run-root "$cell" --data-root "$DATA_ROOT_HOST" --metadata "$RUN_ROOT/input_snapshot/metadata_test.json" \
            --source-images-root "$SOURCE_IMAGES_ROOT_HOST" --source-image-filename "$SOURCE_IMAGE_FILENAME" \
            --checkpoint-path "$RUN_ROOT/checkpoints/ss/checkpoint.pt" --slat-checkpoint-path "$RUN_ROOT/checkpoints/slat/checkpoint.pt" \
            --steps "$STEPS" --slat-steps "$SLAT_STEPS" --cfg-scale "$cfg" --slat-cfg-scale "$slat_cfg" \
            --mixed-precision "$MIXED_PRECISION" --trellis-model image_large
        if (( baselines_done == 0 )); then
            run_phase "${prefix}morphany3d_generation" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$SCRIPT" morphany3d \
                --run-root "$base/shared" --steps "$STEPS" --slat-steps "$SLAT_STEPS" --tfsa-cache-mode hybrid \
                --tfsa-ram-gb "$MORPHANY_TFSA_RAM_GB" --tfsa-min-node-free-gb "$MORPHANY_TFSA_MIN_FREE_GB" \
                --local-cache-root "$TASK_TMP/morphany3d" --tfsa-spill-root "$TASK_TMP/tfsa-spill"
            run_phase "${prefix}interp3d_generation" "$INTERP_ENV" "$SNAPSHOT/Interp3D" "$SCRIPT" interp3d \
                --run-root "$base/shared" --steps "$STEPS" --slat-steps "$SLAT_STEPS"
            if [[ "$EXPORT_GLB" == 1 ]]; then
                run_phase "${prefix}shared_glbs" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$SCRIPT" texture \
                    --run-root "$base/shared" --methods morphany3d interp3d --texture-size "$TEXTURE_SIZE" --simplify "$SIMPLIFY" --fill-holes "$FILL_HOLES"
                run_phase "${prefix}shared_glb_audit" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$SCRIPT" texture-status --run-root "$base/shared" --methods morphany3d interp3d
            fi
            for method in morphany3d interp3d; do
                evaluate_method "$base/shared" "$method" "$base/shared/metrics/$method" "${prefix}${method}" "$metric_mode"
            done
            baselines_done=1
        fi
        if [[ "$EXPORT_GLB" == 1 ]]; then
            run_phase "${phase}_glbs" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$SCRIPT" texture \
                --run-root "$cell" --methods morphflow --texture-size "$TEXTURE_SIZE" --simplify "$SIMPLIFY" --fill-holes "$FILL_HOLES"
            run_phase "${phase}_glb_audit" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$SCRIPT" texture-status --run-root "$cell"
        fi
        evaluate_method "$cell" morphflow "$cell/metrics/morphflow" "$phase" "$metric_mode"
        run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$HELPER" cell --run-root "$RUN_ROOT" --evaluation "$evaluation" --cfg "$cfg" --slat-cfg "$slat_cfg" >/dev/null
        run_phase "${phase}_aggregate" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$SCRIPT" aggregate --run-root "$cell"
        run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$HELPER" summarize --run-root "$RUN_ROOT" --evaluation "$evaluation"
        run_phase "${phase}_weighted_report" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$ACTIVE_PROJECT/benchmarks/weighted_endpoint_report.py" --run-root "$base"
    done < "$GRID_FILE"
    run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$HELPER" summarize --run-root "$RUN_ROOT" --evaluation "$evaluation" --require-complete
}
run_evaluation_grid interp3d "$RUN_ROOT" "" all
if [[ "$WEIGHTED_DENSE" == 1 ]]; then
    run_evaluation_grid weighted50 "$RUN_ROOT/weighted_50" weighted50_ weighted-endpoints-only
fi

archive_evaluation() {
    local source="$1" destination="$2" file directory target
    mkdir -p "$destination"
    cp "$source/cfg_grid_metrics.csv" "$source/cfg_grid_metrics.json" "$destination/"
    cp -R "$source/weighted_endpoint_lpips" "$destination/"
    cp "$source/shared/plan.json" "$destination/plan.json"
    for file in "$source"/cfg_ss_*/final_metrics.{json,csv} "$source"/cfg_ss_*/generation_timings.{json,csv} "$source"/cfg_ss_*/texture_status.json; do
        [[ -f "$file" ]] || continue
        target="$destination/$(basename "$(dirname "$file")")"
        mkdir -p "$target"
        cp "$file" "$target/"
    done
    for directory in "$source"/cfg_ss_*/weighted_endpoint_lpips; do
        [[ -d "$directory" ]] || continue
        target="$destination/$(basename "$(dirname "$directory")")"
        mkdir -p "$target"
        cp -R "$directory" "$target/"
    done
}
if [[ "$RUN_ROOT" != "$FINAL_ARCHIVE" ]]; then
    cp "$RUN_ROOT/run_manifest.json" "$RUN_ROOT/checkpoint_metadata.json" "$FINAL_ARCHIVE/"
    archive_evaluation "$RUN_ROOT" "$FINAL_ARCHIVE"
    if [[ "$WEIGHTED_DENSE" == 1 ]]; then archive_evaluation "$RUN_ROOT/weighted_50" "$FINAL_ARCHIVE/weighted_50"; fi
fi
echo "DONE: $RUN_ROOT/cfg_grid_metrics.csv"
if [[ "$WEIGHTED_DENSE" == 1 ]]; then echo "Dense 50-frame metrics: $RUN_ROOT/weighted_50/cfg_grid_metrics.csv"; fi
echo "Compact reports: $FINAL_ARCHIVE"
echo "EXPORT_GLB=$EXPORT_GLB; raw mesh/Gaussian assets remain available for later GLB export."
