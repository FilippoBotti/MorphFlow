#!/bin/bash
#SBATCH --job-name=baseline_mf_n50_cfg2
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

# Prepared immutable run; exactly two paired CFG settings, not a Cartesian grid.
set -euo pipefail
[[ $# == 1 ]] || { echo "Usage: sbatch $0 RUN_ROOT" >&2; exit 1; }
export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
BASE_DIR="$WORK/$USER"
source "$BASE_DIR/env.sh"
RUN_ROOT="$(cd "$1" && pwd -P)"
source "$RUN_ROOT/launch.env"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch' >&2; exit 1; }
[[ "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || exit 1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MAX_JOBS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ACTIVE_PROJECT="$RUN_ROOT/source_snapshot/MorphFlow"
TRELLIS_SNAPSHOT="$RUN_ROOT/source_snapshot/TRELLIS"
METRIC_SNAPSHOT="$RUN_ROOT/source_snapshot/MorphAny3D"
SCRIPT="$ACTIVE_PROJECT/benchmarks/threeway_textured_benchmark.py"
export MORPHFLOW_MESH_REPAIR_SCRIPT="$SCRIPT"
export MORPHFLOW_MESH_REPAIR_LOG_DIR="$RUN_ROOT/runtime"
export MORPHFLOW_RENDER_REPAIR_CONFIG="$RUN_ROOT/runtime/render_repair.json"
mkdir -p "$RUN_ROOT/runtime-tmp"
exec 9>"$RUN_ROOT/.benchmark.lock"
flock -n 9 || { echo 'Run already active' >&2; exit 1; }
TASK_TMP="$(mktemp -d "$RUN_ROOT/runtime-tmp/job_${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf -- "$TASK_TMP"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export TMPDIR="$TASK_TMP"
JOB_LOG_DIR="$RUN_ROOT/logs/job_${SLURM_JOB_ID}"
mkdir -p "$JOB_LOG_DIR"
cd "$ACTIVE_PROJECT"
run_python() {
    local environment="$1" repository="$2"
    shift 2
    env TRELLIS_REPO="$repository" PYTHONPATH="$repository:$ACTIVE_PROJECT" \
        PATH="$environment/bin:$PATH" "$environment/bin/python" "$@"
}
run_phase() {
    local phase="$1" environment="$2" repository="$3" status
    shift 3
    echo "[$(date -Is)] START $phase" | tee -a "$RUN_ROOT/progress.log"
    if run_python "$environment" "$repository" "$@" > "$JOB_LOG_DIR/$phase.out" 2> "$JOB_LOG_DIR/$phase.err"; then
        echo "[$(date -Is)] DONE $phase" | tee -a "$RUN_ROOT/progress.log"
    else
        status=$?
        echo "ERROR: $phase; $JOB_LOG_DIR/$phase.err" >&2
        tail -n 45 "$JOB_LOG_DIR/$phase.err" >&2 || true
        return "$status"
    fi
}
echo "Run: $RUN_ROOT"
echo "Phase logs: $JOB_LOG_DIR"
echo 'MorphFlow only: 50 test pairs, 5 Beta(5,5) intermediates + 2 canonical sources'
echo 'CFG pairs: SS=1/SLat=1 and SS=5/SLat=1.5; no dense evaluation or GLB export'
run_phase checkpoint_validation "$ORIGINAL_ENV" "$TRELLIS_SNAPSHOT" -c '
import json,sys,torch
from pathlib import Path
from eval_validation_latents import detect_flow_target,checkpoint_args
rows=[]
assert torch.cuda.is_available() and torch.cuda.device_count()==1
for path,target in zip(sys.argv[2:],["ss","slat"]):
 c=torch.load(path,map_location="cpu",mmap=True,weights_only=False);a=checkpoint_args(c)
 assert detect_flow_target(c)==target
 assert a["separate_cond_gate"]=="alpha" and not a.get("augment_source_swap",0) and a.get("trellis_prior_weight",0)==0
 rows.append(dict(path=path,flow_target=target,epoch=c.get("epoch"),val_loss=c.get("val_loss"),separate_cond_gate=a["separate_cond_gate"],augment_source_swap=a.get("augment_source_swap",0),trellis_prior_weight=a.get("trellis_prior_weight",0)));del c
Path(sys.argv[1]).write_text(json.dumps(rows,indent=2)+"\n");print(json.dumps(rows,indent=2));print(torch.cuda.get_device_name(0))
' "$RUN_ROOT/checkpoint_metadata.json" "$CHECKPOINT_PATH" "$SLAT_CHECKPOINT_PATH"
run_phase mesh_smoke "$ORIGINAL_ENV" "$TRELLIS_SNAPSHOT" "$RUN_ROOT/runtime/mesh_channel_blocks.py"
run_phase gaussian_smoke "$METRIC_ENV" "$METRIC_SNAPSHOT" -c \
    'import runpy,sys; runpy.run_path(sys.argv[1])["smoke_gpu"]()' "$ACTIVE_PROJECT/slurm/prepare_gaussian_rasterizer.py"
while read -r cfg slat_cfg; do
    cell="$RUN_ROOT/cfg_ss_${cfg}_slat_${slat_cfg}"
    phase="$(basename "$cell")"
    if [[ -s "$cell/final_metrics.json" ]]; then
        echo "RESUME: $phase complete"
        continue
    fi
    run_phase "${phase}_generation" "$ORIGINAL_ENV" "$TRELLIS_SNAPSHOT" "$RUN_ROOT/runtime/resume_threeway_mesh.py" morphflow \
        --run-root "$cell" --data-root "$DATA_ROOT_HOST" --metadata "$RUN_ROOT/input_snapshot/metadata_test.json" \
        --source-images-root "$SOURCE_IMAGES_ROOT_HOST" --source-image-filename '' \
        --checkpoint-path "$CHECKPOINT_PATH" --slat-checkpoint-path "$SLAT_CHECKPOINT_PATH" \
        --steps 25 --slat-steps 25 --cfg-scale "$cfg" --slat-cfg-scale "$slat_cfg" \
        --mixed-precision bf16 --trellis-model image_large
    run_phase "${phase}_metrics" "$METRIC_ENV" "$METRIC_SNAPSHOT" "$RUN_ROOT/runtime/resume_threeway_render_cache.py" shard \
        --run-dir "$cell/morphflow" --output-dir "$cell/metrics/morphflow" --stage all --shard-index 0 --num-shards 1 \
        --num-views 64 --resolution 512 --camera-distribution hammersley --render-backend gaussian \
        --appearance materials --lpips-net vgg --lpips-batch-size 4 --fid-num-views 1000 \
        --fid-batch-size 8 --fid-selection-seed 42 --fid-aggregation both \
        --reference-cache-root "$RUN_ROOT/shared/reference_metrics" --device cuda
    run_phase "${phase}_merge" "$METRIC_ENV" "$METRIC_SNAPSHOT" "$RUN_ROOT/runtime/resume_threeway_render_cache.py" merge \
        --output-dir "$cell/metrics/morphflow" --fid-aggregation both --fid-min-samples 512 --expected-num-pairs 50
    run_phase "${phase}_aggregate" "$METRIC_ENV" "$METRIC_SNAPSHOT" "$SCRIPT" aggregate --run-root "$cell" --methods morphflow
done < "$RUN_ROOT/cfg_grid.txt"
run_phase summary "$METRIC_ENV" "$METRIC_SNAPSHOT" -c '
import csv,json,sys
from pathlib import Path
r=Path(sys.argv[1]);rows=[];results={}
for name in ("cfg_ss_1_slat_1","cfg_ss_5_slat_1.5"):
 d=json.loads((r/name/"final_metrics.json").read_text());assert set(d["methods"])=={"morphflow"} and d["num_pairs"]==50
 results[name]=d
 with (r/name/"final_metrics.csv").open() as f:rows.extend(dict(x,configuration=name) for x in csv.DictReader(f))
with (r/"cfg_grid_metrics.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(r/"cfg_grid_metrics.json").write_text(json.dumps(results,indent=2)+"\n")
print("Completed 2/2 CFG cells:",r/"cfg_grid_metrics.csv")
' "$RUN_ROOT"
echo "DONE: $RUN_ROOT/cfg_grid_metrics.csv"
