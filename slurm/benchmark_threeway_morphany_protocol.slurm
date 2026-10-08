#!/bin/bash
#SBATCH --job-name=morphany_proto
#SBATCH --account=g_vbd
#SBATCH --partition=gpu_vbd
#SBATCH --qos=gpu_vbd
#SBATCH --nodes=1
#SBATCH --nodelist=wn49
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s_vbd:6
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/hpc/archive/G_VBD/marco.barezzi/morph-benchmark/reports/%x-%j.out
#SBATCH --error=/hpc/archive/G_VBD/marco.barezzi/morph-benchmark/reports/%x-%j.err

# MorphAny3D-style comparison protocol:
#   50 total morph frames = src + 48 intermediates + target
#   12 rendered views per state
#   50 source-target pairs
#   PPL/PDV = existing endpoint-inclusive ppl_full_sum / pdv_full
#   FID = all 50 sequence states vs original src1/src2 endpoints:
#         generated = 50 * 50 * 12 = 30,000 renders
#         reference = 50 *  2 * 12 =  1,200 renders

set -euo pipefail

# IMPORTANT: in MorphFlow benchmark K means INTERMEDIATE states only.
export N="${N:-50}"
export K="${K:-48}"
export SEED="${SEED:-42}"
export NUM_VIEWS="${NUM_VIEWS:-12}"
export RENDER_RESOLUTION="${RENDER_RESOLUTION:-512}"
export LPIPS_NET="${LPIPS_NET:-vgg}"
export RUN_NAME="${RUN_NAME:-threeway_morphanyproto_pairs50_frames50_seed42_v12}"

PROJECT_DIR="${PROJECT_DIR:-$HOME/trellis-singularity/work/MorphFlow}"
SIF="${SIF:-$HOME/containers/trellis-py240-cu118.sif}"
BENCH_ROOT="${BENCH_ROOT:-/hpc/scratch/marco.barezzi/morph-benchmark}"
BENCH_ARCHIVE="${BENCH_ARCHIVE:-/hpc/archive/G_VBD/marco.barezzi/morph-benchmark}"
SHARED_CACHE="${SHARED_CACHE:-$BENCH_ROOT/shared-cache}"
MORPHANY_ENV="${MORPHANY_ENV:-$BENCH_ROOT/morphany3d/env}"
RUN_ROOT="$BENCH_ROOT/threeway-runs/$RUN_NAME"
FINAL_ARCHIVE="$BENCH_ARCHIVE/final-results/$RUN_NAME"

cd "$PROJECT_DIR"

# Run the existing, resumable three-way pipeline with the protocol overrides.
# Its normal center-state FID is left intact as an auxiliary diagnostic.
bash slurm/benchmark_threeway_textured.slurm

# Post-pass: exact requested FID pool over ALL 50 sequence states.
module purge
module load singularity/3.8.7

POST_HOME="$RUN_ROOT/runtime-home/morphany-protocol-post"
POST_TMP="/dev/shm/${USER}/slurm-${SLURM_JOB_ID:-$$}/morphany-protocol-post"
mkdir -p "$POST_HOME" "$POST_TMP"
trap 'rm -rf "$POST_TMP" 2>/dev/null || true' EXIT

singularity exec --cleanenv --containall --no-mount hostfs --nv \
    --bind "$MORPHANY_ENV:/opt/trellis/env:ro" \
    --bind "$PROJECT_DIR:/workspace/MorphFlow:ro" \
    --bind "$RUN_ROOT:/benchmark:rw" \
    --bind "$SHARED_CACHE/torch-home:/opt/trellis/torch-home:rw" \
    --bind "$POST_HOME:/runtime-home:rw" \
    --bind "$POST_TMP:/runtime-tmp:rw" \
    --pwd /workspace/MorphFlow \
    "$SIF" \
    bash -lc '
        set -euo pipefail
        export HOME=/runtime-home
        export TMPDIR=/runtime-tmp
        export TORCH_HOME=/opt/trellis/torch-home
        export PYTHONUNBUFFERED=1
        export PYTHONDONTWRITEBYTECODE=1
        /opt/trellis/env/bin/python benchmarks/morphany_protocol_eval.py \
            --run-root /benchmark \
            --num-views 12 \
            --expected-pairs 50 \
            --expected-sequence-states 50 \
            --fid-batch-size 32 \
            --device cuda
    '

mkdir -p "$FINAL_ARCHIVE/metrics"
cp -f "$RUN_ROOT/morphany_protocol_metrics.json" "$FINAL_ARCHIVE/"
cp -f "$RUN_ROOT/morphany_protocol_metrics.csv" "$FINAL_ARCHIVE/"
for method in morphflow morphany3d interp3d; do
    mkdir -p "$FINAL_ARCHIVE/metrics/$method"
    cp -f "$RUN_ROOT/metrics/$method/metrics_batch.json" "$FINAL_ARCHIVE/metrics/$method/"
done

echo "MorphAny3D-style protocol complete."
echo "Results: $RUN_ROOT/morphany_protocol_metrics.csv"
