#!/bin/bash
# Default: 300000 optimizer updates per phase, cross-attention LoRA.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/queue_train_v4_steps8.sh" --steps "$@"
