#!/bin/bash
#SBATCH --job-name=v4_steps8_ss_lora_all
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
#SBATCH --output=train-v4-steps8-ss-%j.out
#SBATCH --error=train-v4-steps8-ss-%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=BEGIN,END,FAIL

# Batch is per GPU: default global batch=8; 40 full epochs with source swap.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/trellis-singularity/work/MorphFlow}"
export PROJECT_DIR
export NGPUS="${NGPUS:-8}" TRAIN_BS="${TRAIN_BS:-1}" TRAIN_EPOCHS="${TRAIN_EPOCHS:-40}" TRAIN_LR="${TRAIN_LR:-1e-4}"
export RUN_NAME="${RUN_NAME:-v4_steps8_ss_lora_all_bs1_lr1e4_e40}"
exec bash "$PROJECT_DIR/slurm/train_morphflow_v4_steps8_ateneo.sh" ss "$@"
