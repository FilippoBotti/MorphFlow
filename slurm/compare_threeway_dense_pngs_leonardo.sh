#!/bin/bash
# CPU-only comparison: 3 method rows x 50 states, using cached view 025.
set -euo pipefail
if [[ $# -lt 3 || "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "Usage: bash $0 RUN_ROOT CFG_SS CFG_SLAT [--pair ID] [--tile-size PIXELS] [--output-dir DIR] [--overwrite]"
    echo "Accepts the main run root or its weighted_50 directory. Default tile size: 256."
    [[ "${1:-}" == --help || "${1:-}" == -h ]] && exit 0
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
RUN_ROOT="$1"
CFG_SS="$2"
CFG_SLAT="$3"
shift 3
exec bash "$SCRIPT_DIR/compare_threeway_pngs_leonardo.sh" \
    "$RUN_ROOT" "$CFG_SS" "$CFG_SLAT" "$@" --protocol weighted50
