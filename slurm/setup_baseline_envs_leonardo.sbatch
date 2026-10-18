#!/bin/bash
#SBATCH --job-name=baseline_env_install
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
BASE="${BASE:-/leonardo_work/IscrC_MORPHFL/mbarezzi}"
bash "$BASE/src/MorphFlow/slurm/setup_baseline_envs_leonardo.sh" --gpu-install
