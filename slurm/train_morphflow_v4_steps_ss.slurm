#!/bin/bash
#SBATCH --job-name=v4_ss_cross_s300k
#SBATCH --account=g_vbd
#SBATCH --partition=gpu_vbd
#SBATCH --qos=gpu_vbd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=wn49
#SBATCH --gres=gpu:l40s_vbd:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10-00:00:00
#SBATCH --output=train-v4-steps-ss-%j.out
#SBATCH --error=train-v4-steps-ss-%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-$HOME/trellis-singularity/work/MorphFlow}"
export NGPUS="${NGPUS:-8}" TRAIN_BS="${TRAIN_BS:-1}" TRAIN_LR="${TRAIN_LR:-1e-4}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-300000}" DATA_STAGES="${DATA_STAGES:-8}"
[[ "$MAX_TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_TRAIN_STEPS must be positive.' >&2; exit 1; }
export LORA_ATTENTION_SCOPE=cross
exec bash "$PROJECT_DIR/slurm/train_morphflow_v4_steps8_ateneo.sh" ss "$@"
