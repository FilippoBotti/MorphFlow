#!/bin/bash
#SBATCH --job-name=morph_import
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.err

set -euo pipefail

source /leonardo_work/IscrC_MORPHFL/mbarezzi/env.sh

cd "$MORPHFLOW_REPO"

echo "===== ENV ====="
hostname
which python
python --version

echo
echo "===== CUDA ====="

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
PY

echo
echo "===== TRELLIS ====="

python - <<'PY'
import trellis
print("TRELLIS import: OK")
print("TRELLIS path:", trellis.__file__)
PY

echo
echo "===== MORPHFLOW ====="

python - <<'PY'
from models.morph_flow import MorphFlow
from models.morph_slat_flow import MorphSLatFlow
from models.morph_residual_flow import MorphResidualSSFlow

print("MorphFlow: OK")
print("MorphSLatFlow: OK")
print("MorphResidualSSFlow: OK")
PY

echo
echo "===== TRAIN.PY ====="

python train.py --help >/dev/null

echo "train.py import/parser: OK"
echo "MORPHFLOW ENVIRONMENT TEST: OK"
