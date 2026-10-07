#!/bin/bash
#SBATCH --job-name=v3_ss_ablation_priorDual_s24_g6
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=ALL

# Five fresh epochs from ss_ablation_encDual_lora/morphflow_best.pt.
# The shared launcher owns all arguments, cache checks and multi-node setup.
set -euo pipefail
export FLOW_TARGET=ss
launcher="${MF_PROJECT_DIR:-/leonardo_work/IscrC_MORPHFL/mbarezzi/src/MorphFlow}/slurm/train_morphflow_v3_leonardo.sbatch"
exec bash "$launcher" "$@"
