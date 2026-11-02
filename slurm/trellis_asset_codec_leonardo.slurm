#!/bin/bash
#SBATCH --job-name=trellis_asset_codec
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs/%x_%j.err

# bash this-file --check
# sbatch this-file roundtrip asset.glb --output-dir /path/to/output [codec flags]
# sbatch this-file encode asset.glb --output-dir /path/to/encoding [codec flags]
# sbatch this-file decode latent.pt --output-dir /path/to/decoded [codec flags]
set -euo pipefail
MODE=run
case "${1:-}" in
    --check) MODE=check; shift ;;
    --help|-h) MODE=help ;;
esac
if [[ "$MODE" == run && -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit encoding/decoding with sbatch; use --check for offline login-node checks." >&2
    exit 1
fi
export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
BASE_DIR="${CODEC_BASE_DIR:-$WORK/$USER}"
source "${MF_ENV_FILE:-$BASE_DIR/env.sh}"
CODEC_PYTHON="${CODEC_PYTHON:-$BASE_DIR/envs/interp3d/bin/python}"
CODEC_SCRIPT="$MORPHFLOW_REPO/trellis_asset_codec.py"
# Use the environment with rasterizers, while retaining the official checkout.
export PYTHONPATH="$TRELLIS_REPO:$MORPHFLOW_REPO"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$BASE_DIR/cache/torch_extensions}"
export BLENDER_BIN="${BLENDER_BIN:-$BASE_DIR/tools/blender-3.0.1-linux-x64/blender}"
[[ -x "$CODEC_PYTHON" ]] || { echo "Missing codec Python: $CODEC_PYTHON" >&2; exit 1; }
[[ -r "$CODEC_SCRIPT" ]] || { echo "Missing codec script: $CODEC_SCRIPT" >&2; exit 1; }
[[ -r "$TRELLIS_REPO/trellis/__init__.py" ]] || { echo "Missing official TRELLIS checkout: $TRELLIS_REPO" >&2; exit 1; }
if [[ "$MODE" == check ]]; then
    [[ -x "$BLENDER_BIN" ]] || { echo "Missing executable Blender: $BLENDER_BIN" >&2; exit 1; }
    "$CODEC_PYTHON" - <<'PY'
import importlib
import os
from pathlib import Path
print('Checking offline codec dependencies; imports can be slow on the shared filesystem.', flush=True)
import torch
import trellis
from huggingface_hub import try_to_load_from_cache
from trellis_asset_codec import ENCODER, MESH_DECODER, GAUSSIAN_DECODER

official = Path(os.environ['TRELLIS_REPO']).resolve()
assert Path(trellis.__file__).resolve().is_relative_to(official), trellis.__file__
for module in ('open3d', 'utils3d', 'diff_gaussian_rasterization', 'nvdiffrast.torch',
               'xatlas', 'pyvista', 'pymeshfix', 'igraph', 'trellis.utils.postprocessing_utils'):
    print('Import:', module, flush=True)
    importlib.import_module(module)
hub = Path(torch.hub.get_dir())
for path in (hub / 'facebookresearch_dinov2_main/hubconf.py',
             hub / 'checkpoints/dinov2_vitl14_reg4_pretrain.pth'):
    assert path.is_file(), f'Missing offline DINO cache: {path}'
for model in (ENCODER, MESH_DECODER, GAUSSIAN_DECODER):
    for extension in ('json', 'safetensors'):
        filename = f'ckpts/{model}.{extension}'
        cached = try_to_load_from_cache('microsoft/TRELLIS-image-large', filename)
        assert isinstance(cached, str) and Path(cached).is_file(), f'Missing cached model: {filename}'
print('CHECK PASSED: official TRELLIS imports, rasterizers, Blender path, VAE and DINO caches')
print('TRELLIS:', trellis.__file__)
print('Python:', os.sys.executable)
PY
    exit 0
fi
if [[ "$MODE" == help ]]; then
    exec "$CODEC_PYTHON" "$CODEC_SCRIPT" "$@"
fi
exec srun --ntasks=1 "$CODEC_PYTHON" "$CODEC_SCRIPT" "$@"
