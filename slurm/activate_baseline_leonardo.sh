# Source this file with morphany3d or interp3d as the argument.
# Example: source slurm/activate_baseline_leonardo.sh interp3d
BASE="${BASE:-/leonardo_work/IscrC_MORPHFL/mbarezzi}"
BASELINE_METHOD="${1:?Specify morphany3d or interp3d}"
case "$BASELINE_METHOD" in
  morphany3d) BASELINE_REPO="$BASE/src/MorphAny3D" ;;
  interp3d) BASELINE_REPO="$BASE/src/Interp3D" ;;
  *) echo 'Specify morphany3d or interp3d' >&2; return 2 ;;
esac
export WORK="${WORK:-/leonardo_work/IscrC_MORPHFL}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrC_MORPHFL}"
source "$BASE/env.sh"
source "$BASE/envs/$BASELINE_METHOD/bin/activate"
export BASELINE_METHOD BASELINE_REPO
export TRELLIS_REPO="$BASELINE_REPO"
export PYTHONPATH="$BASELINE_REPO:$BASE/src/MorphFlow"
export PYTHONNOUSERSITE=1
export ATTN_BACKEND=xformers SPARSE_ATTN_BACKEND=xformers SPCONV_ALGO=native
export U2NET_HOME="$BASE/cache/u2net"
export TORCH_EXTENSIONS_DIR="$BASE/cache/torch_extensions"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
if [[ $- == *i* ]]; then
    PS1="($BASELINE_METHOD) [\\u@\\h \\W]\\$ "
fi
