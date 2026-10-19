#!/bin/bash
# Compare cached benchmark PNGs locally on Leonardo, using CPU only.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash slurm/compare_threeway_pngs_leonardo.sh RUN_ROOT CFG_SS CFG_SLAT [OPTIONS]

Create one comparison PNG per pair: MorphFlow, MorphAny3D and Interp3D in
three rows, using 025.png for src1, five intermediate alphas and src2.

Options passed to benchmarks/compare_threeway_pngs.py:
  --pair ID_OR_DIRECTORY    Select a pair; repeat to select several.
  --tile-size PIXELS        Tile size (default: 256).
  --output-dir DIRECTORY    Choose another output directory.
  --overwrite              Regenerate existing comparison PNGs.
  --skip-incomplete        Skip pairs with missing renders.

Default output: RUN_ROOT/comparisons/CFG_CELL/view_025/
Default Python: WORKSPACE/envs/morphflow/bin/python.
Optional environment: PROJECT_DIR, MORPHFLOW_WORKSPACE, COMPARISON_PYTHON,
COMPARISON_OUT. No container, GPU, model inference or installation is used.
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

RUN_ROOT="$1"
CFG_SS="$2"
CFG_SLAT="$3"
shift 3
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
MORPHFLOW_WORKSPACE="${MORPHFLOW_WORKSPACE:-$(cd "$PROJECT_DIR/../.." && pwd -P)}"
COMPARISON_PYTHON="${COMPARISON_PYTHON:-$MORPHFLOW_WORKSPACE/envs/morphflow/bin/python}"

[[ -d "$RUN_ROOT" ]] || die "Missing benchmark run: $RUN_ROOT"
[[ -x "$COMPARISON_PYTHON" ]] || die "Missing Python: $COMPARISON_PYTHON. Set COMPARISON_PYTHON to an existing Python with Pillow."
[[ -f "$PROJECT_DIR/benchmarks/compare_threeway_pngs.py" ]] || die "Missing benchmarks/compare_threeway_pngs.py."
if [[ -n "${COMPARISON_OUT:-}" ]]; then
    set -- --output-dir "$COMPARISON_OUT" "$@"
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
exec "$COMPARISON_PYTHON" "$PROJECT_DIR/benchmarks/compare_threeway_pngs.py" \
    --run-root "$RUN_ROOT" --cfg-ss "$CFG_SS" --cfg-slat "$CFG_SLAT" "$@"
