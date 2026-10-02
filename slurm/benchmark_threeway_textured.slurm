#!/bin/bash
#SBATCH --job-name=threeway_tex
#SBATCH --account=g_vbd
#SBATCH --partition=gpu_vbd
#SBATCH --qos=gpu_vbd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6000_vbd:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/hpc/archive/G_VBD/marco.barezzi/morph-benchmark/reports/threeway-%x-%j.out
#SBATCH --error=/hpc/archive/G_VBD/marco.barezzi/morph-benchmark/reports/threeway-%x-%j.err

# Resumable three-way benchmark:
#   MorphFlow (latents) vs MorphAny3D (images) vs Interp3D (images)
#   -> common TRELLIS texture baking
#   -> textured GLB perceptual sequence metrics.
#
# Defaults requested for the benchmark: N=100 test pairs, K=10 intermediates.
# Because common texture baking is expensive, this job is intentionally resumable:
# resubmitting the same RUN_NAME skips completed raw meshes and textured GLBs.

set -euo pipefail

module purge
module load singularity/3.8.7

# -------------------------------
# Fixed benchmark configuration
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

NUM_VIEWS="${NUM_VIEWS:-12}"
RENDER_RESOLUTION="${RENDER_RESOLUTION:-512}"
LPIPS_NET="${LPIPS_NET:-vgg}"

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

MORPHANY_ENV="${MORPHANY_ENV:-$BENCH_ROOT/morphany3d/env}"
MORPHANY_REPO="${MORPHANY_REPO:-$BENCH_ROOT/morphany3d/work/MorphAny3D}"
INTERP_ENV="${INTERP_ENV:-$BENCH_ROOT/interp3d/env}"
INTERP_REPO="${INTERP_REPO:-$BENCH_ROOT/interp3d/work/Interp3D}"

DATA_PARENT="${DATA_PARENT:-/hpc/scratch/marco.barezzi/3d_dataset}"
DATASET_NAME="${DATASET_NAME:-morphing_dataset_v3}"
SOURCE_IMAGES_NAME="${SOURCE_IMAGES_NAME:-flux_outputs}"
METADATA="${METADATA:-metadata_test.json}"
SOURCE_IMAGE_FILENAME="${SOURCE_IMAGE_FILENAME:-}"

# Required MorphFlow checkpoints.
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH=/absolute/path/to/ss_checkpoint.pt}"
SLAT_CHECKPOINT_PATH="${SLAT_CHECKPOINT_PATH:?Set SLAT_CHECKPOINT_PATH=/absolute/path/to/slat_checkpoint.pt}"

RUN_NAME="${RUN_NAME:-threeway_test_n${N}_k${K}_seed${SEED}}"
RUN_ROOT="$BENCH_ROOT/threeway-runs/$RUN_NAME"
FINAL_ARCHIVE="$BENCH_ARCHIVE/final-results/$RUN_NAME"
SCRIPT_IN="/workspace/MorphFlow/benchmarks/threeway_textured_benchmark.py"

# Optional Blender setup.
# If Blender already exists inside the SIF/PATH, leave BLENDER_ROOT_HOST empty.
# For a portable Blender folder on the host:
#   export BLENDER_ROOT_HOST=/path/to/blender-folder
#   export BLENDER_BIN_REL=blender
BLENDER_ROOT_HOST="${BLENDER_ROOT_HOST:-}"
BLENDER_BIN_REL="${BLENDER_BIN_REL:-blender}"
BLENDER_BIN_IN="${BLENDER_BIN_IN:-blender}"
if [[ -n "$BLENDER_ROOT_HOST" ]]; then
    BLENDER_BIN_IN="/opt/blender/$BLENDER_BIN_REL"
fi

mkdir -p "$RUN_ROOT" "$FINAL_ARCHIVE" "$BENCH_ARCHIVE/reports"
mkdir -p "$RUN_ROOT/runtime-home" "$RUN_ROOT/tmp"

# -------------------------------
# Discover the TRELLIS repo used by MorphFlow
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
if [[ -z "$TRELLIS_REPO_HOST" || ! -d "$TRELLIS_REPO_HOST/trellis" ]]; then
    echo "ERROR: could not locate the TRELLIS repo used by MorphFlow" >&2
    exit 1
fi

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
    "$SHARED_CACHE/u2net"
do
    [[ -d "$d" ]] || { echo "ERROR: missing directory: $d" >&2; exit 1; }
done

[[ -f "$PROJECT_DIR/benchmarks/threeway_textured_benchmark.py" ]] || {
    echo "ERROR: missing $PROJECT_DIR/benchmarks/threeway_textured_benchmark.py" >&2
    exit 1
}
[[ -f "$CHECKPOINT_PATH" ]] || { echo "ERROR: SS checkpoint not found: $CHECKPOINT_PATH" >&2; exit 1; }
[[ -f "$SLAT_CHECKPOINT_PATH" ]] || { echo "ERROR: SLat checkpoint not found: $SLAT_CHECKPOINT_PATH" >&2; exit 1; }

SS_DIR="$(dirname "$CHECKPOINT_PATH")"
SS_FILE="$(basename "$CHECKPOINT_PATH")"
SLAT_DIR="$(dirname "$SLAT_CHECKPOINT_PATH")"
SLAT_FILE="$(basename "$SLAT_CHECKPOINT_PATH")"
SS_IN="/checkpoints/ss/$SS_FILE"
SLAT_IN="/checkpoints/slat/$SLAT_FILE"

# -------------------------------
# Isolated container launcher
# -------------------------------
run_phase() {
    local phase="$1"
    local env_host="$2"
    local repo_host="$3"
    shift 3

    local phase_home="$RUN_ROOT/runtime-home/$phase"
    local phase_tmp="$RUN_ROOT/tmp/$phase"
    mkdir -p "$phase_home" "$phase_tmp"

    echo
    echo "============================================================"
    echo "PHASE: $phase"
    echo "ENV:   $env_host"
    echo "REPO:  $repo_host"
    echo "TIME:  $(date -Is)"
    echo "============================================================"

    if [[ -n "$BLENDER_ROOT_HOST" ]]; then
        singularity exec --cleanenv --containall --no-mount hostfs --nv \
            --bind "$env_host:/opt/trellis/env:ro" \
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
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/workspace/MethodRepo:/workspace/MorphFlow
exec "$@"
IN_CONTAINER
    else
        singularity exec --cleanenv --containall --no-mount hostfs --nv \
            --bind "$env_host:/opt/trellis/env:ro" \
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
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/workspace/MethodRepo:/workspace/MorphFlow
exec "$@"
IN_CONTAINER
    fi
}

# -------------------------------
# Header / provenance
# -------------------------------
{
    echo "===== THREE-WAY TEXTURED BENCHMARK ====="
    echo "JOB_ID=${SLURM_JOB_ID:-none}"
    echo "NODE=${SLURM_JOB_NODELIST:-none}"
    echo "RUN_NAME=$RUN_NAME"
    echo "RUN_ROOT=$RUN_ROOT"
    echo "N=$N"
    echo "K=$K"
    echo "SEED=$SEED"
    echo "TEXTURE_SIZE=$TEXTURE_SIZE"
    echo "NUM_VIEWS=$NUM_VIEWS"
    echo "PROJECT_DIR=$PROJECT_DIR"
    echo "TRELLIS_REPO_HOST=$TRELLIS_REPO_HOST"
    echo "MORPHANY_REPO=$MORPHANY_REPO"
    echo "INTERP_REPO=$INTERP_REPO"
    echo "========================================"
} | tee -a "$RUN_ROOT/progress.log"

# -------------------------------
# Early preflight: textured GLB + perceptual evaluation stack
# Only the MorphAny3D benchmark env needs these extra evaluation dependencies.
# -------------------------------
run_phase preflight "$MORPHANY_ENV" "$MORPHANY_REPO" \
    python -c 'import torch; import nvdiffrast.torch; import lpips; print("preflight python OK | torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu", torch.cuda.get_device_name(0))'

run_phase blender_check "$MORPHANY_ENV" "$MORPHANY_REPO" \
    "$BLENDER_BIN_IN" --version

# -------------------------------
# 1) Fixed test-set plan, shared by every method
# -------------------------------
run_phase plan "$ORIGINAL_ENV" "$TRELLIS_REPO_HOST" \
    python "$SCRIPT_IN" plan \
        --run-root /benchmark \
        --data-root "/inputs/$DATASET_NAME" \
        --metadata "$METADATA" \
        --source-images-root "/inputs/$SOURCE_IMAGES_NAME" \
        --num-pairs "$N" \
        --k "$K" \
        --seed "$SEED"

# -------------------------------
# 2) MorphFlow: source latents -> K raw meshes
# -------------------------------
run_phase morphflow "$ORIGINAL_ENV" "$TRELLIS_REPO_HOST" \
    python "$SCRIPT_IN" morphflow \
        --run-root /benchmark \
        --data-root "/inputs/$DATASET_NAME" \
        --metadata "$METADATA" \
        --source-images-root "/inputs/$SOURCE_IMAGES_NAME" \
        --checkpoint-path "$SS_IN" \
        --slat-checkpoint-path "$SLAT_IN" \
        --steps "$STEPS" \
        --slat-steps "$SLAT_STEPS" \
        --cfg-scale "$CFG_SCALE" \
        --slat-cfg-scale "$SLAT_CFG_SCALE" \
        --mixed-precision "$MIXED_PRECISION" \
        --trellis-model image_large

# -------------------------------
# 3) MorphAny3D: source images -> K raw meshes
# -------------------------------
run_phase morphany3d "$MORPHANY_ENV" "$MORPHANY_REPO" \
    python "$SCRIPT_IN" morphany3d --run-root /benchmark

# -------------------------------
# 4) Interp3D: source images -> K raw meshes
#    Same uniform alpha schedule used by the other methods.
# -------------------------------
run_phase interp3d "$INTERP_ENV" "$INTERP_REPO" \
    python "$SCRIPT_IN" interp3d --run-root /benchmark

# -------------------------------
# 5) Common texture baking for ALL methods and reference endpoints.
#    Uses MeshExtractResult.vertex_attrs and the same TRELLIS to_glb(mesh, mesh).
# -------------------------------
run_phase texture "$MORPHANY_ENV" "$MORPHANY_REPO" \
    python "$SCRIPT_IN" texture \
        --run-root /benchmark \
        --texture-size "$TEXTURE_SIZE" \
        --simplify "$SIMPLIFY" \
        --fill-holes "$FILL_HOLES"

# -------------------------------
# 6) Appearance-aware sequence metrics on the FINAL TEXTURED GLBs.
#    eval_perceptual_sequence.py renders GLB materials from identical cameras.
# -------------------------------
for METHOD in morphflow morphany3d interp3d; do
    run_phase "metrics_${METHOD}" "$MORPHANY_ENV" "$MORPHANY_REPO" \
        python /workspace/MorphFlow/eval_perceptual_sequence.py \
            --run_dir "/benchmark/$METHOD" \
            --output_dir "/benchmark/metrics/$METHOD" \
            --stage all \
            --skip_invalid_pairs \
            --num_views "$NUM_VIEWS" \
            --resolution "$RENDER_RESOLUTION" \
            --appearance materials \
            --lpips_net "$LPIPS_NET" \
            --device cuda \
            --strict_uniform_alphas \
            --blender_bin "$BLENDER_BIN_IN"
done

# -------------------------------
# 7) Compact final comparison JSON/CSV
# -------------------------------
run_phase aggregate "$MORPHANY_ENV" "$MORPHANY_REPO" \
    python "$SCRIPT_IN" aggregate --run-root /benchmark

# -------------------------------
# 8) Archive compact final results (not thousands of GLBs)
# -------------------------------
cp -f "$RUN_ROOT/plan.json" "$FINAL_ARCHIVE/"
cp -f "$RUN_ROOT/progress.log" "$FINAL_ARCHIVE/"
cp -f "$RUN_ROOT/final_metrics.json" "$FINAL_ARCHIVE/"
cp -f "$RUN_ROOT/final_metrics.csv" "$FINAL_ARCHIVE/"
cp -f "$RUN_ROOT/texture_status.json" "$FINAL_ARCHIVE/"

for METHOD in morphflow morphany3d interp3d; do
    mkdir -p "$FINAL_ARCHIVE/metrics/$METHOD"
    cp -f "$RUN_ROOT/metrics/$METHOD/metrics_batch.json" "$FINAL_ARCHIVE/metrics/$METHOD/"
    cp -f "$RUN_ROOT/metrics/$METHOD/per_pair.csv" "$FINAL_ARCHIVE/metrics/$METHOD/"
done

echo "DONE: $RUN_ROOT"
echo "FINAL METRICS: $FINAL_ARCHIVE/final_metrics.json"
