#!/bin/bash
#SBATCH --job-name=ss_ablation_gate_only
#SBATCH --account=g_vbd
#SBATCH --partition=gpu_vbd
#SBATCH --qos=gpu_vbd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=wn46
#SBATCH --gres=gpu:a100_vbd:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=400G
#SBATCH --time=10-00:00:00
#SBATCH --output=ss-ablation-%x-%j.out
#SBATCH --error=ss-ablation-%x-%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=BEGIN,END,FAIL

# Override resources with sbatch --nodelist=... --gres=gpu:TYPE:COUNT ...
# Batch = allocated GPUs * TRAIN_BS (default 1/GPU). 20 epochs; swap_only: 10 on doubled data.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/trellis-singularity/work/MorphFlow}"
exec bash "$PROJECT_DIR/slurm/train_ss_ablation_ateneo.sh" gate_only "$@"
