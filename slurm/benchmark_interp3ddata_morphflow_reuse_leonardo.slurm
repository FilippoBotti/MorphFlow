#!/bin/bash
#SBATCH --job-name=interp3ddata_mf_cfg9
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

# Seven states only. Generates MorphFlow; both baseline results must already be
# present in RUN/shared. No baseline generation or evaluation is performed.
# Usage: sbatch THIS_FILE /absolute/run/root [--run|--check]
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }
[[ $# -ge 1 && $# -le 2 ]] || die "Usage: $0 RUN_ROOT [--run|--check]"
REQUESTED_ROOT="$(cd "$1" && pwd -P)"
MODE="${2:---run}"
case "$MODE" in --run|--check) ;; *) die "Unknown mode: $MODE" ;; esac
export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
BASE_DIR="$WORK/$USER"
source "$BASE_DIR/env.sh"
source "$REQUESTED_ROOT/launch.env"
[[ "$RUN_ROOT" == "$REQUESTED_ROOT" ]] || die "Run configuration points elsewhere"
[[ "$WEIGHTED_DENSE" == 0 && "$K" == 5 && "$EXPORT_GLB" == 0 ]] || die "Expected seven-state raw-only run"
if [[ "$MODE" == --run ]]; then
    [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || die "Submit on one GPU node"
fi
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MAX_JOBS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAPSHOT="$RUN_ROOT/source_snapshot"
ACTIVE_PROJECT="$SNAPSHOT/MorphFlow"
HELPER="$ACTIVE_PROJECT/slurm/threeway_ateneo.py"
SCRIPT="$MORPHFLOW_MESH_REPAIR_DIR/resume_threeway_mesh.py"
EVALUATOR="$MORPHFLOW_RENDER_REPAIR_WORKER"
run_python() {
    local environment="$1" repository="$2"
    shift 2
    env TRELLIS_REPO="$repository" PYTHONPATH="$repository:$ACTIVE_PROJECT" \
        PATH="$environment/bin:$PATH" "$environment/bin/python" "$@"
}
cd "$ACTIVE_PROJECT"
exec 9>"$RUN_ROOT/.benchmark.lock"
flock -n 9 || die "Another job is writing this run"
run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$SCRIPT" --verify-frozen --run-root "$RUN_ROOT"
run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" -c \
    'import hashlib,json,sys; from pathlib import Path
r=Path(sys.argv[1]); m=json.loads((r/"run_manifest.json").read_text()); p=json.loads((r/"shared/plan.json").read_text())
assert m["configuration"]["WEIGHTED_DENSE"]==0 and p["k_intermediate"]==5
for method in ("morphany3d","interp3d"):
 path=r/"shared/metrics"/method/"metrics_batch.json"; b=json.loads(path.read_text())
 assert hashlib.sha256(path.read_bytes()).hexdigest()==m["baseline_reuse"][method]["metrics_sha256"]
 assert b["num_pairs"]==m["evaluated_pairs"] and not b["missing_pairs"]
 assert {x["sequence"] for x in b["pairs"]}=={x["pair_dir"] for x in p["pairs"]}
print("Verified reused baseline metrics; dense evaluation disabled")' "$RUN_ROOT"
run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$HELPER" grid --run-root "$RUN_ROOT" > "$RUN_ROOT/cfg_grid.txt"
ACTUAL_PAIRS="$(run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$HELPER" count --run-root "$RUN_ROOT")"
if [[ "$MODE" == --check ]]; then
    echo "CHECK PASSED: frozen inputs, baseline metrics, seven-state protocol and CFG grid"
    exit 0
fi
JOB_LOG_DIR="$RUN_ROOT/logs/job_${SLURM_JOB_ID}_$(date +%s)"
mkdir -p "$JOB_LOG_DIR" "$RUN_ROOT/runtime-tmp"
TASK_TMP="$(mktemp -d "$RUN_ROOT/runtime-tmp/job_${SLURM_JOB_ID}.XXXXXX")"
cleanup() { local status=$?; trap - EXIT; rm -rf -- "$TASK_TMP"; exit "$status"; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export TMPDIR="$TASK_TMP"
run_phase() {
    local phase="$1" environment="$2" repository="$3" status
    shift 3
    echo "[$(date -Is)] START $phase" | tee -a "$RUN_ROOT/progress.log"
    if run_python "$environment" "$repository" "$@" > "$JOB_LOG_DIR/$phase.out" 2> "$JOB_LOG_DIR/$phase.err"; then
        echo "[$(date -Is)] DONE $phase" | tee -a "$RUN_ROOT/progress.log"
    else
        status=$?
        echo "ERROR: $phase failed; see $JOB_LOG_DIR/$phase.{out,err}" >&2
        tail -n 35 "$JOB_LOG_DIR/$phase.err" >&2 || true
        return "$status"
    fi
}
echo "Run: $RUN_ROOT"
echo "Phase logs: $JOB_LOG_DIR"
cat "$RUN_ROOT/checkpoint_metadata.json"
run_phase gpu_smoke "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" -c \
    'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(torch.cuda.get_device_name(0))'
run_phase mesh_channel_blocks_smoke "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$MORPHFLOW_MESH_REPAIR_DIR/mesh_channel_blocks.py"
run_phase gaussian_smoke "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" -c \
    'import runpy,sys; runpy.run_path(sys.argv[1])["smoke_gpu"]()' "$ACTIVE_PROJECT/slurm/prepare_gaussian_rasterizer.py"
while read -r cfg slat_cfg; do
    cell="$(run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$HELPER" cell --run-root "$RUN_ROOT" --cfg "$cfg" --slat-cfg "$slat_cfg")"
    phase="$(basename "$cell")"
    if [[ -s "$cell/final_metrics.json" && -s "$cell/final_metrics.csv" ]]; then
        echo "RESUME: $phase already complete"
        continue
    fi
    run_phase "${phase}_generation" "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$SCRIPT" morphflow \
        --run-root "$cell" --data-root "$DATA_ROOT_HOST" --metadata "$RUN_ROOT/input_snapshot/metadata_test.json" \
        --source-images-root "$SOURCE_IMAGES_ROOT_HOST" --source-image-filename "$SOURCE_IMAGE_FILENAME" \
        --checkpoint-path "$CHECKPOINT_PATH" --slat-checkpoint-path "$SLAT_CHECKPOINT_PATH" \
        --steps "$STEPS" --slat-steps "$SLAT_STEPS" --cfg-scale "$cfg" --slat-cfg-scale "$slat_cfg" \
        --mixed-precision "$MIXED_PRECISION" --trellis-model image_large
    run_phase "${phase}_metrics" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$EVALUATOR" shard \
        --run-dir "$cell/morphflow" --output-dir "$cell/metrics/morphflow" --stage all --shard-index 0 --num-shards 1 \
        --num-views "$NUM_VIEWS" --resolution 512 --camera-distribution hammersley --render-backend gaussian \
        --appearance materials --lpips-net vgg --lpips-batch-size 4 --fid-num-views "$FID_NUM_VIEWS" \
        --fid-batch-size 8 --fid-selection-seed "$SEED" --fid-aggregation both \
        --reference-cache-root "$RUN_ROOT/shared/reference_metrics" --device cuda
    run_phase "${phase}_merge" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$EVALUATOR" merge \
        --output-dir "$cell/metrics/morphflow" --fid-aggregation both --fid-min-samples 512 --expected-num-pairs "$ACTUAL_PAIRS"
    run_phase "${phase}_aggregate" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" "$SCRIPT" aggregate --run-root "$cell"
    run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$HELPER" summarize --run-root "$RUN_ROOT"
    run_phase "${phase}_weighted_report" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" \
        "$ACTIVE_PROJECT/benchmarks/weighted_endpoint_report.py" --run-root "$RUN_ROOT"
    run_phase "${phase}_profile_report" "$MORPHANY_ENV" "$SNAPSHOT/MorphAny3D" \
        "$ACTIVE_PROJECT/benchmarks/source_distance_profile_report.py" --run-root "$RUN_ROOT" --protocol interp3d --plots
done < "$RUN_ROOT/cfg_grid.txt"
run_python "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" "$HELPER" summarize --run-root "$RUN_ROOT" --require-complete
echo "DONE: $RUN_ROOT/cfg_grid_metrics.csv"
