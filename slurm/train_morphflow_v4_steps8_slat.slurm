#!/bin/bash
#SBATCH --job-name=v4_steps8_slat_lora_all
#SBATCH --account=g_vbd
#SBATCH --partition=gpu_vbd
#SBATCH --qos=gpu_vbd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=wn51
#SBATCH --gres=gpu:h100_vbd:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10-00:00:00
#SBATCH --output=train-v4-steps8-slat-%j.out
#SBATCH --error=train-v4-steps8-slat-%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=BEGIN,END,FAIL

# Batch is per GPU: default global batch=2; 20 full epochs with source swap.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/trellis-singularity/work/MorphFlow}"
export PROJECT_DIR
export NGPUS="${NGPUS:-2}" TRAIN_BS="${TRAIN_BS:-1}" TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}" TRAIN_LR="${TRAIN_LR:-5e-5}"
export RUN_NAME="${RUN_NAME:-v4_steps8_slat_lora_all_bs1_lr5e5_e20}"
exec bash "$PROJECT_DIR/slurm/train_morphflow_v4_steps8_ateneo.sh" slat "$@"
