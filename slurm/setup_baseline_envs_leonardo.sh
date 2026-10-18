#!/bin/bash
# Independent environments for the two upstream TRELLIS forks. No packages are
# installed into envs/morphflow; copied files are never hard linked.
set -euo pipefail
BASE="${BASE:-/leonardo_work/IscrC_MORPHFL/mbarezzi}"
MODE="${1:---check}"
export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
source "$BASE/env.sh"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export U2NET_HOME="$BASE/cache/u2net"
export TORCH_EXTENSIONS_DIR="$BASE/cache/torch_extensions"
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS="${MAX_JOBS:-8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WHEELS="$BASE/cache/baseline_wheels"
mkdir -p "$WHEELS" "$BASE/logs" "$TORCH_EXTENSIONS_DIR"

case "$MODE" in
  --clone)
    # Run from a login node. This can take several minutes on the shared FS.
    if [[ ! -r "$WHEELS/pytorch3d-0.7.8-py310_cu121_pyt240.tar.bz2" ]]; then
      curl -L --fail -o "$WHEELS/pytorch3d-0.7.8-py310_cu121_pyt240.tar.bz2" \
        https://conda.anaconda.org/pytorch3d/linux-64/pytorch3d-0.7.8-py310_cu121_pyt240.tar.bz2
    fi
    for method in morphany3d interp3d; do
      target="$BASE/envs/$method"
      if [[ ! -e "$target/.baseline_env_isolated" ]]; then
        mkdir -p "$target"
        rsync -a --exclude='__pycache__/' --exclude='tests/' --exclude='test/' \
          "$BASE/envs/morphflow/" "$target/"
      fi
      python - "$BASE/envs/morphflow" "$target" <<'PY'
import pathlib, sys
old, new = sys.argv[1:]
for p in pathlib.Path(new, 'bin').iterdir():
    if p.is_symlink() or not p.is_file():
        continue
    data = p.read_bytes()
    if b'\0' not in data[:4096] and old.encode() in data:
        p.write_bytes(data.replace(old.encode(), new.encode()))
pathlib.Path(new, '.baseline_env_isolated').write_text('Independent copy of '+old+'\n')
PY
      "$target/bin/python" -m pip install 'iopath==0.1.10' 'pytorch-fid==0.3.0'
      # Official PyTorch3D build matching Python 3.10 / CUDA 12.1 / Torch 2.4.0.
      # It is distributed as a conda tarball, whose Python package and metadata
      # are also directly usable inside this native venv.
      tar -xjf "$WHEELS/pytorch3d-0.7.8-py310_cu121_pyt240.tar.bz2" -C "$target" \
        --wildcards 'lib/python3.10/site-packages/pytorch3d/*' \
        'lib/python3.10/site-packages/pytorch3d-*.egg-info/*'
      "$target/bin/python" -c 'import torch; from pytorch3d.loss import chamfer_distance; p=torch.randn(1,8,3); assert chamfer_distance(p,p)[0].item() < 1e-6'
      "$target/bin/python" -m pip freeze > "$target/requirements-inherited.txt"
    done
    python -m pip download --no-deps 'iopath==0.1.10' 'pytorch-fid==0.3.0' -d "$WHEELS"
    ;;
  --gpu-install)
    [[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch setup_baseline_envs_leonardo.sbatch'; exit 1; }
    py="$BASE/envs/morphany3d/bin/python"
    "$py" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
    for ext in nvdiffrast diffoctreerast mip-splatting/submodules/diff-gaussian-rasterization; do
      "$py" -m pip wheel --no-deps --no-build-isolation "$BASE/src/baseline_extensions/$ext" --wheel-dir "$WHEELS"
    done
    for method in morphany3d interp3d; do
      py="$BASE/envs/$method/bin/python"
      "$py" -m pip install --no-index --find-links "$WHEELS" nvdiffrast diffoctreerast diff-gaussian-rasterization pytorch-fid
      "$py" -m pip freeze > "$BASE/envs/$method/requirements-installed.txt"
    done
    "$0" --check
    for spec in 'morphany3d:MorphAny3D' 'interp3d:Interp3D'; do
      method="${spec%%:*}"
      repo="${spec#*:}"
      PYTHONPATH="$BASE/src/$repo" "$BASE/envs/$method/bin/python" "$BASE/src/MorphFlow/slurm/check_baseline_env_leonardo.py" --gpu
    done
    ;;
  --check)
    for spec in 'morphany3d:MorphAny3D' 'interp3d:Interp3D'; do
      method="${spec%%:*}"
      repo="${spec#*:}"
      PYTHONPATH="$BASE/src/$repo" "$BASE/envs/$method/bin/python" "$BASE/src/MorphFlow/slurm/check_baseline_env_leonardo.py"
    done
    ;;
  *) echo "Usage: $0 [--clone|--check|--gpu-install]" >&2; exit 2 ;;
esac
