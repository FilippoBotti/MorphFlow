#!/bin/bash
#SBATCH --job-name=morphany3d_v3_20x10_50k
#SBATCH --account=g_vbd
#SBATCH --partition=gpu_vbd
#SBATCH --qos=gpu_vbd
#SBATCH --nodes=1
#SBATCH --nodelist=wn49
#SBATCH --gres=gpu:l40s_vbd:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G
#SBATCH --time=10-00:00:00
#SBATCH --output=/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/final_dataset/slurm-dataset-%x-%j.out
#SBATCH --error=/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/final_dataset/slurm-dataset-%x-%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=ALL

set -euo pipefail

module purge
module load singularity/3.8.7

PROJECT_DIR="$HOME/trellis-singularity/work/MorphFlow"
MORPHANY3D_DIR="$HOME/trellis-singularity/work/MorphAny3D"

SIF="$HOME/containers/trellis-py240-cu118.sif"
TRELLIS_BASE="$HOME/trellis-singularity"

FLUX_ROOT_HOST="/hpc/scratch/marco.barezzi/3d_dataset/flux_outputs"
FLUX_ROOT_IN="/data/flux_outputs"

DATA_ROOT_HOST="/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v3"
DATA_ROOT_IN="/data/morphing_dataset_v3"

ARCHIVE_HOST="/hpc/archive/G_VBD/marco.barezzi"
LOG_ROOT_HOST="$ARCHIVE_HOST/morphflow_runs/dataset_v3_generation"
LOG_DIR_HOST="$LOG_ROOT_HOST/job_${SLURM_JOB_ID}"

GENERATOR="$PROJECT_DIR/data/dataset_scripts/generate_morphany3d_split_dataset.py"

TARGET_TOTAL=50000
NUM_GPUS=8

# TFSA cache limit PER GPU/worker.
# Override at submit time if needed:
#   sbatch --export=ALL,MAX_WORK_CACHE_GB=100 ...
MAX_WORK_CACHE_GB="${MAX_WORK_CACHE_GB:-60}"
NVME_MARGIN_GB="${NVME_MARGIN_GB:-20}"

mkdir -p \
  "$DATA_ROOT_HOST" \
  "$LOG_DIR_HOST" \
  "$TRELLIS_BASE/hf-home"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: MorphFlow not found: $PROJECT_DIR"
    exit 1
fi

if [ ! -d "$MORPHANY3D_DIR" ]; then
    echo "ERROR: MorphAny3D not found: $MORPHANY3D_DIR"
    exit 1
fi

if [ ! -f "$GENERATOR" ]; then
    echo "ERROR: generator not found: $GENERATOR"
    exit 1
fi

if [ ! -d "$FLUX_ROOT_HOST" ]; then
    echo "ERROR: flux assets not found: $FLUX_ROOT_HOST"
    exit 1
fi

# Small control files only: worker.sh, etc.
CONTROL_TMP="/tmp/morphany3d_v3_${SLURM_JOB_ID}"

# Heavy MorphAny3D TFSA cache: wn49 local NVMe.
CACHE_ROOT_HOST="/nvme/${USER}/morphany3d_tfsa_cache/job_${SLURM_JOB_ID}"
CACHE_ROOT_IN="/morph_cache"

mkdir -p "$CONTROL_TMP"

if ! mkdir -p "$CACHE_ROOT_HOST"; then
    echo "ERROR: cannot create NVMe cache directory:"
    echo "  $CACHE_ROOT_HOST"
    exit 1
fi

if [ ! -w "$CACHE_ROOT_HOST" ]; then
    echo "ERROR: NVMe cache directory is not writable:"
    echo "  $CACHE_ROOT_HOST"
    exit 1
fi

echo
echo "=========================================="
echo " NVMe preflight"
echo "=========================================="
df -h /nvme
echo
echo "CACHE_ROOT_HOST=$CACHE_ROOT_HOST"
echo "CACHE_ROOT_IN=$CACHE_ROOT_IN"

# Dynamic NVMe safety check.
# Worst case = NUM_GPUS * MAX_WORK_CACHE_GB plus safety margin.
NVME_AVAIL_KB="$(df -Pk "$CACHE_ROOT_HOST" | awk 'NR==2 {print $4}')"
NVME_REQUIRED_GB=$((NUM_GPUS * MAX_WORK_CACHE_GB + NVME_MARGIN_GB))
NVME_MIN_KB=$((NVME_REQUIRED_GB * 1024 * 1024))

echo "MAX_WORK_CACHE_GB=$MAX_WORK_CACHE_GB GiB per worker"
echo "Required free NVMe >= $NVME_REQUIRED_GB GiB"

if [ "$NVME_AVAIL_KB" -lt "$NVME_MIN_KB" ]; then
    echo
    echo "ERROR: insufficient free space on /nvme."
    echo "Need at least $NVME_REQUIRED_GB GiB for:"
    echo "  $NUM_GPUS workers x $MAX_WORK_CACHE_GB GiB"
    echo "  + $NVME_MARGIN_GB GiB safety margin"
    df -h /nvme
    exit 1
fi

cleanup() {
    echo
    echo "Cleaning MorphAny3D temporary files..."
    rm -rf "$CONTROL_TMP"
    rm -rf "$CACHE_ROOT_HOST"
}
trap cleanup EXIT

echo "=========================================="
echo " MorphAny3D dataset v3"
echo "=========================================="
echo "PROJECT_DIR=$PROJECT_DIR"
echo "MORPHANY3D_DIR=$MORPHANY3D_DIR"
echo "FLUX_ROOT_HOST=$FLUX_ROOT_HOST"
echo "DATA_ROOT_HOST=$DATA_ROOT_HOST"
echo "TARGET_TOTAL=$TARGET_TOTAL"
echo "NUM_GPUS=$NUM_GPUS"
echo "CONTROL_TMP=$CONTROL_TMP"
echo "CACHE_ROOT_HOST=$CACHE_ROOT_HOST"
echo "CACHE_ROOT_IN=$CACHE_ROOT_IN"
echo

nvidia-smi -L || true


# ------------------------------------------------------------
# Worker eseguito dentro Singularity.
#
# Ogni worker:
#   - vede una sola GPU tramite CUDA_VISIBLE_DEVICES
#   - ricostruisce lo stesso piano globale
#   - genera soltanto il proprio batch
#
# Il generatore MorphFlow usa:
#   --num-batches 8
#   --batch-index 0..7
#
# per assegnare deterministicamente le coppie.
# ------------------------------------------------------------

WORKER_SCRIPT="$CONTROL_TMP/worker.sh"

cat > "$WORKER_SCRIPT" <<'WORKER'
#!/bin/bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/trellis/env

export HF_HOME=/opt/trellis/hf-home
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2

# È importante che venga usata la versione TRELLIS/MorphAny3D
# presente nel checkout MorphAny3D.
export PYTHONPATH="$MORPHANY3D_DIR:$PROJECT_DIR:${PYTHONPATH:-}"

mkdir -p "$WORK_CACHE_IN"

cd "$MORPHANY3D_DIR"

echo "=========================================="
echo "Worker batch $BATCH_INDEX / $NUM_BATCHES"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "WORK_CACHE_IN=$WORK_CACHE_IN"
echo "=========================================="

python3 --version

python3 -c '
import torch
print("torch =", torch.__version__)
print("cuda =", torch.version.cuda)
print("available =", torch.cuda.is_available())
print("device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU =", torch.cuda.get_device_name(0))
'

python3 -c '
import trellis
print("TRELLIS imported from:", trellis.__file__)
'

ARGS=(
  "$PROJECT_DIR/data/dataset_scripts/generate_morphany3d_split_dataset.py"

  --assets-dir "$FLUX_ROOT_IN"
  --output-dir "$DATA_ROOT_IN"

  --model-id microsoft/TRELLIS-image-large
  --seed 0

  --train-ratio 0.8
  --val-ratio 0.1
  --test-ratio 0.1
  --split-strategy random

  --target-total-samples 50000
  --pairs-per-asset 50

  --random-alphas-per-pair 10
  --alpha-min 0.01
  --alpha-max 0.99
  --alpha-jitter 0.011428571428571429

  --morphing-num 22

  --tfsa-alpha 0.8
  --disable-tfsa 0

  --tfsa-cache-mode memory
  --work-cache-dir "$WORK_CACHE_IN"
  --max-work-cache-gb "$MAX_WORK_CACHE_GB"

  --num-batches "$NUM_BATCHES"
  --batch-index "$BATCH_INDEX"

  --cuda-device 0
  --print-plan-details 0
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    ARGS+=(--dry-run)
fi

python3 "${ARGS[@]}"
WORKER

chmod +x "$WORKER_SCRIPT"


# ------------------------------------------------------------
# Variabili comuni esportate dentro Singularity
# ------------------------------------------------------------

export SINGULARITYENV_PROJECT_DIR="$PROJECT_DIR"
export SINGULARITYENV_MORPHANY3D_DIR="$MORPHANY3D_DIR"
export SINGULARITYENV_FLUX_ROOT_IN="$FLUX_ROOT_IN"
export SINGULARITYENV_DATA_ROOT_IN="$DATA_ROOT_IN"
export SINGULARITYENV_MAX_WORK_CACHE_GB="$MAX_WORK_CACHE_GB"

COMMON_BINDS=(
  --bind "$TRELLIS_BASE:/opt/trellis"
  --bind "$FLUX_ROOT_HOST:$FLUX_ROOT_IN:ro"
  --bind "$DATA_ROOT_HOST:$DATA_ROOT_IN"
  --bind "$CACHE_ROOT_HOST:$CACHE_ROOT_IN"
  --bind "$HOME:$HOME"
)


# ------------------------------------------------------------
# 1. Dry-run singolo.
#
# Qui NUM_BATCHES=1 apposta: vogliamo verificare il piano globale
# da 50.000 target prima di occupare le 8 GPU con la generazione.
# ------------------------------------------------------------

echo
echo "=========================================="
echo "Running global dry-run"
echo "=========================================="

SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 \
SINGULARITYENV_BATCH_INDEX=0 \
SINGULARITYENV_NUM_BATCHES=1 \
SINGULARITYENV_DRY_RUN=1 \
SINGULARITYENV_WORK_CACHE_IN="$CACHE_ROOT_IN/dry_run" \
singularity exec --cleanenv --nv \
  "${COMMON_BINDS[@]}" \
  "$SIF" \
  "$WORKER_SCRIPT"


# ------------------------------------------------------------
# Verifica forte del piano.
# ------------------------------------------------------------

python3 - "$DATA_ROOT_HOST/splits.dry_run.json" <<'CHECK'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])

if not path.is_file():
    raise SystemExit(f"Missing dry-run plan: {path}")

with path.open() as f:
    d = json.load(f)

cfg = d["config"]

print()
print("=== DRY RUN VALIDATION ===")
print("sequence_intermediates =", cfg["sequence_intermediates"])
print("targets_per_pair       =", cfg["targets_per_pair"])
print("alpha_jitter           =", cfg["alpha_jitter"])
print("target_sample_counts   =", cfg["target_sample_counts"])

assert cfg["sequence_intermediates"] == 20, cfg
assert cfg["targets_per_pair"] == 10, cfg

expected = {
    "train": 40000,
    "val": 5000,
    "test": 5000,
}

assert cfg["target_sample_counts"] == expected, (
    cfg["target_sample_counts"], expected
)

assert sum(expected.values()) == 50000

print("Dry-run OK: 50,000 total targets.")
print("==========================")
CHECK


# ------------------------------------------------------------
# 2. Generazione reale: 8 processi, uno per L40S.
# ------------------------------------------------------------

echo
echo "=========================================="
echo "Launching 8 MorphAny3D workers"
echo "=========================================="

declare -a PIDS=()
declare -a GPUS=()
declare -a LOGS=()

for gpu in $(seq 0 $((NUM_GPUS - 1))); do

    WORK_CACHE_HOST="$CACHE_ROOT_HOST/gpu_${gpu}"
    WORK_CACHE_IN="$CACHE_ROOT_IN/gpu_${gpu}"
    WORKER_LOG="$LOG_DIR_HOST/gpu_${gpu}.log"

    mkdir -p "$WORK_CACHE_HOST"

    echo "Launching batch=$gpu on GPU=$gpu"
    echo "  log=$WORKER_LOG"
    echo "  NVMe cache=$WORK_CACHE_HOST"

    (
        export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$gpu"
        export SINGULARITYENV_BATCH_INDEX="$gpu"
        export SINGULARITYENV_NUM_BATCHES="$NUM_GPUS"
        export SINGULARITYENV_DRY_RUN=0
        export SINGULARITYENV_WORK_CACHE_IN="$WORK_CACHE_IN"

        singularity exec --cleanenv --nv \
          "${COMMON_BINDS[@]}" \
          "$SIF" \
          "$WORKER_SCRIPT"
    ) >"$WORKER_LOG" 2>&1 &

    pid=$!

    PIDS+=("$pid")
    GPUS+=("$gpu")
    LOGS+=("$WORKER_LOG")
done


# ------------------------------------------------------------
# Attendi tutti i worker.
# ------------------------------------------------------------

FAIL=0

for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    gpu="${GPUS[$i]}"
    log="${LOGS[$i]}"

    if wait "$pid"; then
        rc=0
    else
        rc=$?
        FAIL=1
    fi

    echo "GPU=$gpu PID=$pid exit_code=$rc log=$log"
done

if [ "$FAIL" -ne 0 ]; then
    echo
    echo "ERROR: at least one MorphAny3D worker failed."
    echo "Worker logs:"
    printf '  %s\n' "${LOGS[@]}"
    exit 1
fi


# ------------------------------------------------------------
# 3. Verifica finale dataset.
# ------------------------------------------------------------

echo
echo "=========================================="
echo "Validating generated dataset"
echo "=========================================="

python3 - "$DATA_ROOT_HOST" <<'CHECK'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

expected = {
    "train": 40000,
    "val": 5000,
    "test": 5000,
}

counts = {}

for split, n_expected in expected.items():
    path = root / f"metadata_{split}.json"

    if not path.is_file():
        raise SystemExit(f"Missing metadata: {path}")

    with path.open() as f:
        data = json.load(f)

    counts[split] = len(data)

    print(
        f"{split:5s}: "
        f"{counts[split]:6d} / {n_expected:6d}"
    )

    if counts[split] != n_expected:
        print(
            f"WARNING: incomplete {split}: "
            f"{counts[split]} / {n_expected}"
        )


total = sum(counts.values())

print("total:", total)

if total != 50000:
    print(
        f"WARNING: dataset is still incomplete: "
        f"{total} / 50000 targets"
    )
    print(f"Missing targets: {50000 - total}")
else:
    print("Dataset target count complete: 50000 / 50000")


# Controlla anche la struttura 
selected_path = root / "pair_selected_indices.json"
sequence_path = root / "pair_sequence_alphas.json"

with selected_path.open() as f:
    selected = json.load(f)

with sequence_path.open() as f:
    sequences = json.load(f)

num_pairs = 0

for split in ("train", "val", "test"):

    for pair, indices in selected[split].items():
        if len(indices) != 10:
            raise SystemExit(
                f"{split}/{pair}: expected 10 selected indices, "
                f"got {len(indices)}"
            )

        seq = sequences[split][pair]

        if len(seq) != 20:
            raise SystemExit(
                f"{split}/{pair}: expected 20 sequence alphas, "
                f"got {len(seq)}"
            )

        num_pairs += 1


print("pairs:", num_pairs)

if num_pairs != 5000:
    raise SystemExit(
        f"Expected 5000 pairs, got {num_pairs}"
    )

print()
print("Dataset v3 validation PASSED")
print("  20 generated intermediate steps / pair")
print("  10 random saved targets / pair")
print("  5,000 pairs")
print("  50,000 targets total")
CHECK


echo
echo "=========================================="
echo "Dataset v3 generation pass completed"
echo "=========================================="
echo "DATA_ROOT_HOST=$DATA_ROOT_HOST"
echo "LOG_DIR_HOST=$LOG_DIR_HOST"
