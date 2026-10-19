#!/bin/bash
# Compose existing PNGs on CPU using the already installed MorphAny3D Python.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash slurm/compare_threeway_pngs_l40s.sh RUN_ROOT CFG_SS CFG_SLAT [OPTIONS]

Compare MorphFlow, MorphAny3D and Interp3D using only the existing 025.png files
from the original seven-state protocol: src1, five intermediates, src2.

Options are passed to benchmarks/compare_threeway_pngs.py:
  --pair ID_OR_DIRECTORY    Select one pair; repeat to select more than one.
  --tile-size PIXELS        Display tile size (default: 256).
  --overwrite              Replace existing comparison files.
  --skip-incomplete        Skip pairs whose PNGs are not all available yet.

Without --pair, process all pairs in the selected run. No inference, metrics,
GPU, installation or changes to the existing model environments are required.

Optional environment: COMPARISON_OUT, PROJECT_DIR, SIF, BENCH_ROOT, MORPHANY_ENV,
HOST_PYTHON (host Python used only to normalize the numerical CFG values).
HELP
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
if (( $# < 3 )); then
    usage >&2
    exit 2
fi
die() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "$1" ]] || die "Missing benchmark run: $1"
RUN_ROOT="$(cd "$1" && pwd -P)"
CFG_SS="$2"
CFG_SLAT="$3"
shift 3

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
SIF="${SIF:-$HOME/containers/trellis-py240-cu118.sif}"
BENCH_ROOT="${BENCH_ROOT:-/hpc/scratch/marco.barezzi/morph-benchmark}"
MORPHANY_ENV="${MORPHANY_ENV:-$BENCH_ROOT/morphany3d/env}"
HOST_PYTHON="${HOST_PYTHON:-python3}"

[[ -f "$SIF" ]] || die "Missing container: $SIF"
[[ -d "$MORPHANY_ENV" ]] || die "Missing existing MorphAny3D environment: $MORPHANY_ENV"
[[ -f "$PROJECT_DIR/benchmarks/compare_threeway_pngs.py" ]] || die "Missing comparison script: update the complete MorphFlow checkout."
CFG_CELL="$("$HOST_PYTHON" - "$CFG_SS" "$CFG_SLAT" <<'PY'
import math
import sys

try:
    values = [float(value) for value in sys.argv[1:]]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("CFG must be finite and nonnegative")
except ValueError as exc:
    raise SystemExit("ERROR: invalid CFG: " + str(exc))
print("cfg_ss_{}_slat_{}".format(*(format(value, ".12g") for value in values)))
PY
)"
COMPARISON_OUT="${COMPARISON_OUT:-$RUN_ROOT/comparisons/$CFG_CELL/view_025}"
mkdir -p "$COMPARISON_OUT"
COMPARISON_OUT="$(cd "$COMPARISON_OUT" && pwd -P)"

module purge
module load singularity/3.8.7
echo "Comparison PNGs: $COMPARISON_OUT"
singularity exec --cleanenv --containall --no-mount hostfs \
    --bind "$MORPHANY_ENV:/opt/trellis/env:ro" \
    --bind "$PROJECT_DIR:/workspace/MorphFlow:ro" \
    --bind "$RUN_ROOT:/benchmark:ro" \
    --bind "$COMPARISON_OUT:/comparisons:rw" \
    --pwd /workspace/MorphFlow "$SIF" bash -s -- "$CFG_SS" "$CFG_SLAT" "$@" <<'IN_CONTAINER'
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/trellis/env
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cfg_ss="$1"
cfg_slat="$2"
shift 2
exec python /workspace/MorphFlow/benchmarks/compare_threeway_pngs.py \
    --run-root /benchmark --cfg-ss "$cfg_ss" --cfg-slat "$cfg_slat" \
    --output-dir /comparisons "$@"
IN_CONTAINER
