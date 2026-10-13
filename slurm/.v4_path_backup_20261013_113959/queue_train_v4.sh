#!/bin/bash
set -euo pipefail

cd "$HOME/trellis-singularity/work/MorphFlow"

python3 - <<'PY'
from pathlib import Path
import re

source = Path('slurm/train_morphflow_v3.slurm')
if not source.is_file():
    raise SystemExit('ERRORE: non trovo ' + str(source))
base = source.read_text()


def set_line(s, prefix, new_line):
    lines = s.splitlines(True)
    matches = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError('Attesa una sola riga %r, trovate %d' % (prefix, len(matches)))
    index = matches[0]
    lines[index] = new_line + '\n'
    return ''.join(lines)


def set_option(s, option, value):
    lines = s.splitlines(True)
    token = '--' + option + ' '
    matches = [i for i, line in enumerate(lines) if line.lstrip().startswith(token)]
    if len(matches) != 1:
        raise RuntimeError('Flag %s: atteso 1, trovati %d' % (token, len(matches)))
    i = matches[0]
    old = lines[i]
    indent = old[:len(old) - len(old.lstrip())]
    if not old.rstrip().endswith('\\'):
        raise RuntimeError('Flag %s non termina con continuazione bash' % token)
    lines[i] = indent + token + str(value) + ' \\\n'
    return ''.join(lines)


def inject_once(s, anchor, content):
    if s.count(anchor) != 1:
        raise RuntimeError('Anchor mancante o ambiguo: %r (count=%d)' % (anchor, s.count(anchor)))
    return s.replace(anchor, content + anchor, 1)

for target, lr, tag in (
    ('ss', '1e-4', 'v4_interp3d_ss_dual_tgate_p25g4'),
    ('slat', '5e-5', 'v4_interp3d_slat_dual_tgate_p25g4'),
):
    s = base
    s = set_line(s, '#SBATCH --job-name=', '#SBATCH --job-name=' + ('v4_ss_pr25g4' if target == 'ss' else 'v4_slat_pr25g4'))
    s = set_line(s, 'DATA_ROOT_HOST=', 'DATA_ROOT_HOST="/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v4_interp3d_12500x4_e02"')
    s = set_line(s, 'DATA_ROOT_IN=', 'DATA_ROOT_IN="/data/morphing_dataset_v4_interp3d_12500x4_e02"')
    s = set_line(s, 'RUN_NAME=', 'RUN_NAME="${RUN_NAME:-' + tag + '}"')
    s = set_line(s, 'AUTO_RESUME=', 'AUTO_RESUME="${AUTO_RESUME:-1}"')
    s = set_line(s, 'TRELLIS_PRIOR_WEIGHT=', 'TRELLIS_PRIOR_WEIGHT="${TRELLIS_PRIOR_WEIGHT:-0.1}"')
    s = set_line(s, 'TRELLIS_PRIOR_EVERY=', 'TRELLIS_PRIOR_EVERY="${TRELLIS_PRIOR_EVERY:-4}"')
    s = set_line(s, 'TRELLIS_PRIOR_ROLLOUT_STEPS=', 'TRELLIS_PRIOR_ROLLOUT_STEPS="${TRELLIS_PRIOR_ROLLOUT_STEPS:-25}"')
    s = set_line(s, 'TRELLIS_PRIOR_GRAD_STEPS=', 'TRELLIS_PRIOR_GRAD_STEPS="${TRELLIS_PRIOR_GRAD_STEPS:-4}"')
    s = set_line(s, 'TRELLIS_PRIOR_CHECKPOINT=', 'TRELLIS_PRIOR_CHECKPOINT="${TRELLIS_PRIOR_CHECKPOINT:-1}"')

    for option, value in (
        ('train_epochs', '20'),
        ('train_bs', '1'),
        ('num_workers', '2'),
        ('lr', lr),
        ('cond_lr', lr),
        ('flow_lr', lr),
        ('lora_lr', lr),
        ('flow_target', target),
    ):
        s = set_option(s, option, value)

    # Riutilizza la cache DINOv2 già stabilizzata durante la generazione.
    s = inject_once(s, 'SIF="$HOME/containers/trellis-py240-cu118.sif"\n',
        'BENCH_HF_HOME="/hpc/scratch/marco.barezzi/morph-benchmark/shared-cache/hf-home"\n'
        'BENCH_TORCH_HOME="/hpc/scratch/marco.barezzi/morph-benchmark/shared-cache/torch-home"\n'
    )
    s = inject_once(s, 'singularity exec --cleanenv --nv \\\n',
        'test -s "$BENCH_TORCH_HOME/hub/facebookresearch_dinov2_main/hubconf.py" || { echo "ERROR: DINOv2 cache incomplete" >&2; exit 1; }\n'
    )
    s = inject_once(s, '  --bind "$DATA_ROOT_HOST:$DATA_ROOT_IN:ro" \\\n',
        '  --bind "$BENCH_TORCH_HOME:/opt/trellis/torch-home:rw" \\\n'
    )
    s = inject_once(s, '  --bind "$BENCH_TORCH_HOME:/opt/trellis/torch-home:rw" \\\n',
        '  --bind "$BENCH_HF_HOME:/opt/trellis/hf-home:rw" \\\n'
    )
    s = inject_once(s, 'export HF_HOME=/opt/trellis/hf-home\n',
        'export TORCH_HOME=/opt/trellis/torch-home\n'
    )

    # Verifica che il dataset sia completamente generato, non solo pianificato.
    dataset_gate = '''python3 - "$DATA_ROOT_HOST" <<'PY_V4_DATA'
import json, os, sys
root = sys.argv[1]
with open(os.path.join(root, 'run_state.json')) as f:
    state = json.load(f)
if state.get('phase') != 'complete' or state.get('status') != 'complete':
    raise SystemExit('Dataset v4 non completo: %s' % state)
for split, count in [('train', 50000), ('val', 2000), ('test', 2000)]:
    with open(os.path.join(root, 'metadata_' + split + '.json')) as f:
        data = json.load(f)
    if len(data) != count:
        raise SystemExit('Dataset v4 %s: %d != %d' % (split, len(data), count))
print('Dataset v4 completo: train=50000 val=2000 test=2000')
PY_V4_DATA

'''
    s = inject_once(s, 'AUTO_RESUME="${AUTO_RESUME:-1}"\n', dataset_gate)

    # Il prior SLat richiede un checkpoint differente da quello SS.
    # Inizializzazione sequenziale di DINOv2 prima dei processi Accelerate.
    prior_gate = '''python - <<'PY_V4_PRIOR'
import torch
from huggingface_hub import hf_hub_download
kind = %r
prefix = ('ckpts/ss_flow_img_dit_L_16l8_fp16' if kind == 'ss'
          else 'ckpts/slat_flow_img_dit_L_64l8p2_fp16')
for ext in ('json', 'safetensors'):
    p = hf_hub_download(repo_id='microsoft/TRELLIS-image-large',
                        filename=prefix + '.' + ext)
    print('Prior TRELLIS %%s: %%s' %% (kind, p), flush=True)
print('TORCH_HOME:', torch.hub.get_dir(), flush=True)
encoder = torch.hub.load('facebookresearch/dinov2:main',
                         'dinov2_vitl14_reg', pretrained=True, trust_repo=True)
del encoder
print('Prior DINOv2 warmed up once; starting 8 training GPUs.', flush=True)
PY_V4_PRIOR

''' % target
    s = inject_once(s, 'accelerate launch \\\n', prior_gate)

    if '--separate_cond 1 \\' not in s or '--separate_cond_gate token \\' not in s:
        raise RuntimeError('Dual encoder/token gate missing')
    if '--augment_source_swap 1 \\' not in s or '--use_lora 1 \\' not in s:
        raise RuntimeError('Source swap or LoRA missing')
    out = Path('slurm/train_morphflow_v4_%s.slurm' % target)
    out.write_text(s)
    out.chmod(0o755)
    print('Creato:', out, '| lr:', lr, '| epochs: 20 | prior: 25/4 ogni 4 update')
PY

bash -n slurm/train_morphflow_v4_ss.slurm
bash -n slurm/train_morphflow_v4_slat.slurm

DATA_ROOT="/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v4_interp3d_12500x4_e02"
STATE="$DATA_ROOT/run_state.json"
if [[ ! -f "$STATE" ]]; then
    echo "ERRORE: nessuno stato della generazione v4 trovato in $STATE" >&2
    exit 1
fi

read -r GEN_JOB GEN_PHASE GEN_STATUS < <(python3 - "$STATE" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    s = json.load(f)
print('%s %s %s' % (s.get('job_id', ''), s.get('phase', ''), s.get('status', '')))
PY
)

DEP=()
if [[ "$GEN_PHASE" == 'complete' && "$GEN_STATUS" == 'complete' ]]; then
    echo "Dataset gia completo: avvio SS senza dipendenza."
else
    if [[ ! "$GEN_JOB" =~ ^[0-9]+$ ]]; then
        echo "ERRORE: job_id generazione non valido: $GEN_JOB" >&2
        exit 1
    fi
    SLURM_STATE="$(squeue -h -j "$GEN_JOB" -o '%T' | head -n 1 || true)"
    case "$SLURM_STATE" in
        RUNNING|PENDING|CONFIGURING|COMPLETING)
            DEP=("--dependency=afterok:$GEN_JOB")
            ;;
        *)
            echo "ERRORE: generazione $GEN_JOB non attiva (Slurm=$SLURM_STATE; phase=$GEN_PHASE status=$GEN_STATUS)." >&2
            exit 1
            ;;
    esac
fi

if [[ "$GEN_PHASE" == "complete" && "$GEN_STATUS" == "complete" ]]; then
    SS_JOB=$(sbatch --parsable --export=ALL slurm/train_morphflow_v4_ss.slurm)
else
    SS_JOB=$(sbatch --parsable --export=ALL "--dependency=afterok:$GEN_JOB" slurm/train_morphflow_v4_ss.slurm)
fi
SS_JOB="${SS_JOB%%;*}"
SLAT_JOB=$(sbatch --parsable --export=ALL "--dependency=afterok:$SS_JOB" slurm/train_morphflow_v4_slat.slurm)
SLAT_JOB="${SLAT_JOB%%;*}"

echo ''
echo '=============================================='
echo " GENERAZIONE V4: $GEN_JOB ($GEN_PHASE / $GEN_STATUS)"
echo " TRAIN SS 20ep LR=1e-4 prior25/4: $SS_JOB"
echo " TRAIN SLAT 20ep LR=5e-5 prior25/4: $SLAT_JOB"
echo ' Entrambi usano 8 L40S, in sequenza.'
echo '=============================================='
squeue -u "$USER" -o '%.18i %.26j %.10T %.10M %.28R'
