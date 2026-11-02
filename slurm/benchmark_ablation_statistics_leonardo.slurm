#!/bin/bash
#SBATCH --job-name=ablation_stats
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
[[ $# == 1 || $# == 3 ]] || { echo "Usage: sbatch $0 RUN_ROOT [SHARD_INDEX SHARD_COUNT]" >&2; exit 1; }
SHARD_INDEX=${2:-0}
SHARD_COUNT=${3:-1}
[[ "$SHARD_INDEX" =~ ^[0-9]+$ && "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || exit 1
(( SHARD_INDEX < SHARD_COUNT )) || exit 1
printf -v SHARD_TAG 'shard_%03d_of_%03d' "$SHARD_INDEX" "$SHARD_COUNT"
BASE=/leonardo_work/IscrC_MORPHFL/mbarezzi
source "$BASE/env.sh"
source "$1/launch.env"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MAX_JOBS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PROJECT="$RUN_ROOT/source_snapshot/MorphFlow"
WORKER="$PROJECT/benchmarks/stream_ablation_statistics.py"
REPORT="$PROJECT/benchmarks/ablation_statistics_report.py"
allocation_lock="$RUN_ROOT/.allocation.lock"
if (( SHARD_COUNT > 1 )); then allocation_lock="$RUN_ROOT/.allocation.$SHARD_TAG.lock"; fi
exec 9>"$allocation_lock"
flock -n 9 || { echo 'This pair shard is already active'; exit 1; }
if [[ -f "$RUN_ROOT/reports/COMPLETE.json" ]]; then
    echo "Already complete: $RUN_ROOT/reports"
    exit 0
fi
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
[[ ${#visible_gpus[@]} == 4 ]] || { echo 'Exactly four allocated GPUs required' >&2; exit 1; }
variants=(baseline gate_only prior_only full)
workers=()
stop_workers() {
    trap '' TERM INT
    for pid in "${workers[@]+"${workers[@]}"}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${workers[@]+"${workers[@]}"}"; do wait "$pid" || true; done
    echo 'Allocation interrupted; durable commits will be resumed.'
    exit 75
}
trap stop_workers TERM INT
echo "SUITE: $RUN_ROOT"
echo "Pair batch shard: $SHARD_INDEX / $SHARD_COUNT"
echo '500 pairs; four fixed models; CFG SS=5 SLat=1.5; seven states; concatenated FID.'
echo "Progress: $RUN_ROOT/variants/MODEL/status.json"
for i in 0 1 2 3; do
    name="${variants[$i]}"
    log="$RUN_ROOT/logs/job_${SLURM_JOB_ID}/$name"
    mkdir -p "$log" "$RUN_ROOT/work/$name/tmp_$SHARD_TAG"
    env CUDA_VISIBLE_DEVICES="${visible_gpus[$i]}" TMPDIR="$RUN_ROOT/work/$name/tmp_$SHARD_TAG" \
        "$ORIGINAL_ENV/bin/python" "$WORKER" --run-root "$RUN_ROOT" --variant "$name" \
        --batch-shard-index "$SHARD_INDEX" --batch-shard-count "$SHARD_COUNT" \
        > "$log/worker.out" 2> "$log/worker.err" &
    workers+=("$!")
    echo "$name on GPU ${visible_gpus[$i]}: $log/worker.out"
done
status=0
for pid in "${workers[@]}"; do
    if wait "$pid"; then :; else code=$?; status="$code"; fi
done
if [[ "$status" != 0 ]]; then
    echo "Incomplete allocation (status=$status); see worker.err. Resume keeps verified pairs." >&2
    exit "$status"
fi
exec 8>"$RUN_ROOT/.report.lock"
flock 8
all_complete=1
for name in "${variants[@]}"; do
    [[ -f "$RUN_ROOT/variants/$name/COMPLETE.json" ]] || all_complete=0
done
if (( all_complete )); then
    if [[ ! -f "$RUN_ROOT/reports/COMPLETE.json" ]]; then
        "$ORIGINAL_ENV/bin/python" "$REPORT" --run-root "$RUN_ROOT" --require-complete
    fi
    echo "DONE: $RUN_ROOT/reports/index.html"
else
    echo "Pair shard complete; other pair shards are still pending."
fi
