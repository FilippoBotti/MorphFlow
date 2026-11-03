#!/bin/bash
#SBATCH --job-name=morph_glb_pair
#SBATCH --account=IscrC_MORPHFL
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs_gen/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_MORPHFL/mbarezzi/logs_gen/%x_%j.err

# Lancio: sbatch /leonardo_work/IscrC_MORPHFL/mbarezzi/src/MorphFlow/slurm/run_morph_glb_pair_leonardo.sbatch
# Modificare SRC1 e SRC2 qui sotto per usare altri GLB.
set -euo pipefail

BASE=/leonardo_work/IscrC_MORPHFL/mbarezzi
export WORK=/leonardo_work/IscrC_MORPHFL
export FAST=/leonardo_scratch/fast/IscrC_MORPHFL

SRC1="$BASE/assets/Werewolf_Warrior.glb"
SRC2="$BASE/assets/pool3913.glb"
SS_CHECKPOINT="$BASE/checkpoints/ateneo_v4_ss17/ss_best_e17.pt"
SLAT_CHECKPOINT="$BASE/checkpoints/v4_interp3d_slat_dual_tgate_p25g4_lr5e-5_bs1_leonardo/morphflow_best.pt"
BLENDER_BIN="$BASE/tools/blender-3.0.1-linux-x64/blender"
PROJECT_DIR="$BASE/src/MorphFlow"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Lanciare questo file con sbatch, senza argomenti." >&2
    exit 1
fi

source "$BASE/env.sh"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export TORCH_EXTENSIONS_DIR="$BASE/cache/torch_extensions"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

OUTPUT_DIR="$BASE/outputs/glb_pair/job_${SLURM_JOB_ID}"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$OUTPUT_DIR"
cd "$PROJECT_DIR"

for input_path in "$SRC1" "$SRC2" "$SS_CHECKPOINT" "$SLAT_CHECKPOINT" "$PROJECT_DIR/morph_glb_pair.py"; do
    if [[ ! -r "$input_path" ]]; then
        echo "File assente o non leggibile: $input_path" >&2
        exit 1
    fi
done
if [[ ! -x "$BLENDER_BIN" ]]; then
    echo "Blender assente o non eseguibile: $BLENDER_BIN" >&2
    exit 1
fi

command=(
    "$BASE/envs/morphflow/bin/python" "$PROJECT_DIR/morph_glb_pair.py"
    "$SRC1" "$SRC2"
    --ss-checkpoint "$SS_CHECKPOINT"
    --slat-checkpoint "$SLAT_CHECKPOINT"
    --blender "$BLENDER_BIN"
    --alpha 0.5
    --cfg-ss 5
    --cfg-slat 1.5
    --steps 25
    --slat-steps 25
    --encoding-views 50
    --dino-batch-size 4
    --seed 42
    --mixed-precision bf16
    --output-dir "$OUTPUT_DIR"
    --mode run
)
printf 'Output: %s\n' "$OUTPUT_DIR"
printf 'Command: '
printf '%q ' "${command[@]}"
printf '\n'
exec srun --ntasks=1 "${command[@]}"
