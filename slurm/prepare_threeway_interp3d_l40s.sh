#!/bin/bash
# Prepare evaluation-only assets from an Ateneo login node, without changing
# the working TRELLIS / MorphAny3D / Interp3D Conda environments.
set -euo pipefail

MODE="${1:---prepare}"
case "$MODE" in
    --prepare|--check) ;;
    --help|-h)
        cat <<'HELP'
Usage: bash slurm/prepare_threeway_interp3d_l40s.sh [--prepare|--check]

Run --prepare once on an Ateneo login node with internet access. It installs
pytorch-fid 0.3.0 and the TRELLIS mip-splatting Gaussian rasterizer in separate
evaluation overlays only when needed, then caches and verifies canonical weights.
The rasterizer is compiled for L40S with the existing CUDA 11.8/GCC11 toolchain;
no GPU is required for compilation when the target CUDA architecture is explicit.
The existing Conda environments are always mounted read-only.

--check is offline: it verifies imports and full SHA256 hashes of cached weights.
It does not download assets or install packages. No GPU is required.

Optional variables: BENCH_ROOT, SIF, MORPHANY_ENV, SHARED_CACHE,
PROTOCOL_PYTHON (default: $BENCH_ROOT/protocol-python), TOOLCHAIN_ROOT_HOST,
GAUSSIAN_CUDA_ARCH_LIST (default: 8.9), GAUSSIAN_BUILD_JOBS (default: 2).
HELP
        exit 0 ;;
    *) echo "ERROR: expected --prepare, --check, or --help" >&2; exit 2 ;;
esac
(( $# <= 1 )) || { echo "ERROR: unexpected arguments" >&2; exit 2; }

BENCH_ROOT="${BENCH_ROOT:-/hpc/scratch/marco.barezzi/morph-benchmark}"
SIF="${SIF:-$HOME/containers/trellis-py240-cu118.sif}"
MORPHANY_ENV="${MORPHANY_ENV:-$BENCH_ROOT/morphany3d/env}"
SHARED_CACHE="${SHARED_CACHE:-$BENCH_ROOT/shared-cache}"
PROTOCOL_PYTHON="${PROTOCOL_PYTHON:-$BENCH_ROOT/protocol-python}"
TOOLCHAIN_ROOT_HOST="${TOOLCHAIN_ROOT_HOST:-$BENCH_ROOT/morphany3d/toolchain/cuda118-gcc11}"
GAUSSIAN_CUDA_ARCH_LIST="${GAUSSIAN_CUDA_ARCH_LIST:-8.9}"
GAUSSIAN_BUILD_JOBS="${GAUSSIAN_BUILD_JOBS:-2}"
GAUSSIAN_PREPARER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/prepare_gaussian_rasterizer.py"

[[ "$PROTOCOL_PYTHON" = /* && "$PROTOCOL_PYTHON" != / ]] || {
    echo "ERROR: PROTOCOL_PYTHON must be an absolute directory path other than /" >&2
    exit 2
}
[[ -f "$SIF" ]] || { echo "ERROR: missing container: $SIF" >&2; exit 1; }
[[ -d "$MORPHANY_ENV" ]] || { echo "ERROR: missing existing environment: $MORPHANY_ENV" >&2; exit 1; }
[[ -d "$TOOLCHAIN_ROOT_HOST" ]] || { echo "ERROR: missing CUDA 11.8/GCC11 toolchain: $TOOLCHAIN_ROOT_HOST" >&2; exit 1; }
[[ -f "$GAUSSIAN_PREPARER" ]] || { echo "ERROR: missing helper: $GAUSSIAN_PREPARER; update the complete MorphFlow checkout" >&2; exit 1; }
if [[ "$MODE" == --check ]]; then
    for directory in "$PROTOCOL_PYTHON" "$SHARED_CACHE/torch-home"; do
        [[ -d "$directory" ]] || {
            echo "ERROR: missing $directory; run this helper with --prepare first" >&2
            exit 1
        }
    done
else
    mkdir -p "$(dirname "$PROTOCOL_PYTHON")" "$SHARED_CACHE/torch-home"
fi

# Coordinate preparation with other invocations on the same shared overlay.
OVERLAY_PARENT="$(cd "$(dirname "$PROTOCOL_PYTHON")" && pwd -P)"
OVERLAY_NAME="$(basename "$PROTOCOL_PYTHON")"
exec 9>"$OVERLAY_PARENT/.${OVERLAY_NAME}.prepare.lock"
flock -x 9

module purge
module load singularity/3.8.7

RUNTIME_ROOT="$(mktemp -d "$OVERLAY_PARENT/.threeway-protocol-prep.XXXXXXXX")"
cleanup() { rm -rf -- "$RUNTIME_ROOT"; }
trap cleanup EXIT
mkdir -p "$RUNTIME_ROOT/home" "$RUNTIME_ROOT/tmp"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1

singularity exec --cleanenv --containall --no-mount hostfs \
    --bind "$MORPHANY_ENV:/opt/trellis/env:ro" \
    --bind "$TOOLCHAIN_ROOT_HOST:/opt/trellis/toolchain/cuda118-gcc11:ro" \
    --bind "$GAUSSIAN_PREPARER:/prepare_gaussian_rasterizer.py:ro" \
    --bind "$OVERLAY_PARENT:/protocol-overlay-parent:rw" \
    --bind "$SHARED_CACHE/torch-home:/opt/trellis/torch-home:rw" \
    --bind "$RUNTIME_ROOT/home:/runtime-home:rw" \
    --bind "$RUNTIME_ROOT/tmp:/runtime-tmp:rw" \
    "$SIF" bash -s -- "$MODE" "$OVERLAY_NAME" "$GAUSSIAN_CUDA_ARCH_LIST" "$GAUSSIAN_BUILD_JOBS" <<'IN_CONTAINER'
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/trellis/env
export HOME=/runtime-home
export TMPDIR=/runtime-tmp
export TORCH_HOME=/opt/trellis/torch-home
export XDG_CACHE_HOME=/runtime-home/.cache
export PIP_CACHE_DIR=/runtime-tmp/pip-cache
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_HOME=/opt/trellis/toolchain/cuda118-gcc11
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc" CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX" PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="$3" MAX_JOBS="$4"
exec python - "$@" <<'PY'
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request

prepare = sys.argv[1] == '--prepare'
overlay = Path('/protocol-overlay-parent') / sys.argv[2]
expected_version = '0.3.0'
sys.path.insert(0, str(overlay))

def fid_version():
    try:
        # Looking up distribution metadata avoids caching a wrong imported
        # version before a newly prepared overlay is inserted.
        return importlib.metadata.version('pytorch-fid')
    except importlib.metadata.PackageNotFoundError:
        return None

if fid_version() != expected_version:
    if not prepare:
        raise SystemExit('Missing pytorch-fid 0.3.0. Run prepare_threeway_interp3d_l40s.sh --prepare.')
    if overlay.exists() and any(overlay.iterdir()):
        raise SystemExit('Protocol overlay is nonempty but does not provide pytorch-fid 0.3.0; choose a new PROTOCOL_PYTHON directory.')
    stage = Path(tempfile.mkdtemp(prefix='.' + overlay.name + '.stage-', dir=str(overlay.parent)))
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
            '--no-deps', '--no-compile', '--target', str(stage), 'pytorch-fid==0.3.0',
        ], check=True)
        check_env = dict(os.environ, PYTHONPATH=str(stage))
        subprocess.run([
            sys.executable, '-c',
            'import importlib.metadata; from pytorch_fid.inception import InceptionV3; '
            'assert importlib.metadata.version("pytorch-fid") == "0.3.0"',
        ], check=True, env=check_env)
        # Publish only a fully installed and importable package tree.
        if overlay.exists():
            overlay.rmdir()  # Only the empty directory allowed above.
        stage.rename(overlay)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    importlib.invalidate_caches()
elif prepare:
    overlay.mkdir(parents=True, exist_ok=True)

import torch
import torchvision
import lpips
from pytorch_fid.inception import FID_WEIGHTS_URL

# A CPU import of GaussianRenderer alone does not exercise this lazy native
# dependency. Install/probe the exact mip-splatting API before allocating GPUs.
subprocess.run([
    sys.executable, '/prepare_gaussian_rasterizer.py',
    '--prepare' if prepare else '--check',
    '--overlay', str(overlay / 'gaussian'),
    '--source-root', '/protocol-overlay-parent/gaussian-build',
], check=True)

if fid_version() != expected_version:
    raise SystemExit('Unexpected pytorch-fid version after preparation')

checkpoint_dir = Path(torch.hub.get_dir()) / 'checkpoints'
if prepare:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def require_weights(url, filename, prefix):
    target = checkpoint_dir / filename
    if target.is_file():
        digest = sha256(target)
        if not digest.startswith(prefix):
            raise SystemExit(f'Invalid cached weights {target}: SHA256={digest}; expected prefix {prefix}. Move the corrupt file aside before preparing again.')
        print(f'Verified {filename}: SHA256={digest}', flush=True)
        return target
    if not prepare:
        raise SystemExit(f'Missing cached weights {target}; run prepare_threeway_interp3d_l40s.sh --prepare.')
    fd, temporary_name = tempfile.mkstemp(prefix='.' + filename + '.', suffix='.download', dir=str(checkpoint_dir))
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        print(f'Downloading {filename} from {url}', flush=True)
        with urllib.request.urlopen(url) as source, temporary.open('wb') as destination:
            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        digest = sha256(temporary)
        if not digest.startswith(prefix):
            raise SystemExit(f'Download checksum mismatch for {filename}: SHA256={digest}; expected prefix {prefix}')
        temporary.replace(target)
        print(f'Cached {filename}: SHA256={digest}', flush=True)
    finally:
        temporary.unlink(missing_ok=True)
    return target

require_weights(FID_WEIGHTS_URL, 'pt_inception-2015-12-05-6726825d.pth', '6726825d')
vgg_weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
require_weights(vgg_weights.url, Path(vgg_weights.url).name, '397923af')

# These constructors use the caches just verified. They also catch incompatible
# Torch/torchvision/LPIPS builds on the login node, before allocating eight GPUs.
def reject_unexpected_download(*args, **kwargs):
    raise RuntimeError('Unexpected weight download after cache verification; check package versions and cache paths')

torch.hub.download_url_to_file = reject_unexpected_download
from pytorch_fid.inception import InceptionV3
fid = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True, normalize_input=True, use_fid_inception=True).eval()
del fid
perceptual = lpips.LPIPS(net='vgg', spatial=False).eval()
del perceptual
print(f'Protocol assets ready: torch={torch.__version__}, torchvision={torchvision.__version__}, pytorch-fid={fid_version()}, overlay={overlay}', flush=True)
PY
IN_CONTAINER

echo "READY: $PROTOCOL_PYTHON"
echo "Weights: $SHARED_CACHE/torch-home/hub/checkpoints"
