#!/bin/bash
# CPU-only composition of existing seven-state ablation render PNGs.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
MORPHFLOW_WORKSPACE="${MORPHFLOW_WORKSPACE:-$(cd "$PROJECT_DIR/../.." && pwd -P)}"
COMPARISON_PYTHON="${COMPARISON_PYTHON:-$MORPHFLOW_WORKSPACE/envs/morphflow/bin/python}"
[[ -x "$COMPARISON_PYTHON" ]] || { echo "Missing Pillow Python: $COMPARISON_PYTHON" >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
exec "$COMPARISON_PYTHON" "$PROJECT_DIR/benchmarks/compare_ablation_pngs.py" "$@"
