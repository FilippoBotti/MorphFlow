#!/bin/bash
#SBATCH --job-name=morph_gpu_test
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.err
#SBATCH --mail-user=marco.barezzi@unipr.it
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

source /leonardo_work/IscrC_MORPHFL/mbarezzi/env.sh

echo "===== NODE ====="
hostname

echo
echo "===== NVIDIA ====="
nvidia-smi

echo
echo "===== PYTHON ====="
which python
python --version

echo
echo "===== TORCH GPU TEST ====="

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}")
    print(f"  memory: {p.total_memory / 1024**3:.2f} GiB")

x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
y = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
z = x @ y

print("matrix multiplication:", z.shape)
print("result device:", z.device)
print("GPU TEST OK")
PY

echo
echo "===== XFORMERS/SPCONV ====="

python - <<'PY'
import xformers
import spconv.pytorch as spconv

print("xformers:", xformers.__version__)
print("spconv: OK")
print("GPU ENVIRONMENT OK")
PY
