#!/bin/bash
# Submit from the login node. Before '--': sbatch options; after '--': Python options.
# Example: bash slurm/submit_teacher_student_comparison.sh --time=2-00:00:00 -- --resume
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ARGS=()
PYTHON_ARGS=()
while [ "$#" -gt 0 ]; do
    if [ "$1" = -- ]; then
        shift
        PYTHON_ARGS=("$@")
        break
    fi
    SBATCH_ARGS+=("$1")
    shift
done

# The directory must exist before Slurm opens its logs, hence a submission ID
# (timestamp + PID) instead of the not-yet-known Slurm job ID in this default.
OUTPUT_DIR="${OUTPUT_DIR:-/hpc/archive/G_VBD/marco.barezzi/morphflow_runs/comparison/run_$(date +%Y%m%d_%H%M%S)_$$}"
for ((arg_index=0; arg_index<${#PYTHON_ARGS[@]}; arg_index++)); do
    case "${PYTHON_ARGS[$arg_index]}" in
        --output-dir)
            arg_index=$((arg_index + 1))
            OUTPUT_DIR="${PYTHON_ARGS[$arg_index]:?--output-dir needs a value}"
            ;;
        --output-dir=*) OUTPUT_DIR="${PYTHON_ARGS[$arg_index]#--output-dir=}" ;;
    esac
done
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
export OUTPUT_DIR

# Appending this absolute Python argument also keeps paths aligned when sbatch
# is submitted with a different working directory or environment export policy.
PYTHON_ARGS+=(--output-dir "$OUTPUT_DIR")
printf 'Output and logs: %s\n' "$OUTPUT_DIR"
exec sbatch "${SBATCH_ARGS[@]}" \
    --output="$OUTPUT_DIR/slurm-comparison-%j.out" \
    --error="$OUTPUT_DIR/slurm-comparison-%j.err" \
    --open-mode=append \
    "$SCRIPT_DIR/generate_teacher_student_comparison.slurm" "${PYTHON_ARGS[@]}"
