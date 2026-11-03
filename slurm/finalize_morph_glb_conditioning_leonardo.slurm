#!/bin/bash
#SBATCH --job-name=morph_glb_finalize
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.err
set -euo pipefail
export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
source "$WORK/$USER/env.sh"
exec "$WORK/$USER/envs/morphflow/bin/python" "$MORPHFLOW_REPO/benchmarks/morph_glb_conditioning_finalize.py" "${1:?run root}" "${2:-continue}"
