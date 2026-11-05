#!/bin/bash
# Run on the Ateneo login node after updating this checkout.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR%/slurm}}"
exec python3 - "$@" <<'PY'
"""Queue independent SS/SLat runs on a completed v4 dataset, by epochs or steps."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


def read_json(path):
    with path.open() as source:
        return json.load(source)


def run(command, environment=None):
    result = subprocess.run(command, env=environment, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode:
        raise RuntimeError('%s: %s' % (' '.join(command), result.stderr.strip()))
    return result.stdout.strip()


def require_resources(node, gpu_type, count):
    output = run(['sinfo', '-h', '-N', '-n', node, '-o', '%N %G'])
    for line in output.splitlines():
        fields = line.split(None, 1)
        if len(fields) != 2 or fields[0] != node:
            continue
        counts = re.findall(r'(?:^|,)gpu:' + re.escape(gpu_type) +
                            r':([0-9]+)(?=\(|,|$)', fields[1])
        if counts and sum(map(int, counts)) >= count:
            return
    raise RuntimeError('%s: richieste %d GPU %s; risorse disponibili: %s' %
                       (node, count, gpu_type, output or 'nodo non trovato'))


def generation_dependency(parent, dataset, explicit_job):
    state_path = parent / 'run_state.json'
    state = read_json(state_path) if state_path.is_file() else {}
    if not explicit_job and state.get('phase') == 'complete' and state.get('status') == 'complete':
        view_state = read_json(dataset / 'run_state.json')
        if view_state.get('phase') != 'complete' or view_state.get('status') != 'complete':
            raise RuntimeError('La vista del dataset non risulta completa: ' + str(dataset))
        for split, expected in (('train', 50000), ('val', 2000), ('test', 2000)):
            entries = read_json(dataset / ('metadata_' + split + '.json'))
            if not isinstance(entries, list) or len(entries) != expected:
                raise RuntimeError('Dataset %s: attesi %d campioni' % (split, expected))
        return None
    job = str(explicit_job or state.get('job_id', ''))
    if not re.fullmatch(r'[1-9][0-9]*', job):
        raise RuntimeError('Generazione non completa: manca un job valido in %s. '
                           'Avviare prima la generazione oppure usare --generation-job JOB.' % state_path)
    job_state = run(['squeue', '-h', '-j', job, '-o', '%T'])
    if job_state not in ('RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING'):
        raise RuntimeError('Generazione %s non attiva (Slurm=%s): riprenderla prima del training.' %
                           (job, job_state or 'non presente in coda'))
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='verifica piano e risorse senza inviare job')
    parser.add_argument('--generation-job', help='job di generazione da attendere con afterok')
    parser.add_argument('--dataset-stages', type=int, choices=(2, 8),
                        default=int(os.environ.get('DATA_STAGES', '8')))
    parser.add_argument('--steps', action='store_true', help='budget optimizer: 300000 per fase; LoRA cross')
    parser.add_argument('--ss-steps', type=int, help='numero totale di aggiornamenti SS')
    parser.add_argument('--slat-steps', type=int, help='numero totale di aggiornamenti SLat')
    args = parser.parse_args()
    steps_mode = args.steps or args.ss_steps is not None or args.slat_steps is not None
    project = Path(os.environ['PROJECT_DIR']).expanduser().resolve()
    parent = Path(os.environ.get('DATA_PARENT_HOST',
                  '/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v4_steps8_steps2')).expanduser().resolve()
    dataset = Path(os.environ.get('DATA_ROOT_HOST',
                   str(parent / ('morphing_dataset_v4_steps%d' % args.dataset_stages)))).expanduser().resolve()
    if dataset.parent != parent:
        raise RuntimeError('DATA_ROOT_HOST deve essere una vista nella directory DATA_PARENT_HOST.')
    for tool in ('sinfo', 'squeue', 'sbatch'):
        if shutil.which(tool) is None:
            raise RuntimeError('Comando %s assente: eseguire sul login HPC di Ateneo.' % tool)
    runtime = project / 'slurm/train_morphflow_v4_steps8_ateneo.sh'
    if not runtime.is_file():
        raise RuntimeError('Runtime mancante: ' + str(runtime))
    plan_path = dataset / 'plan.json'
    if plan_path.is_file():
        plan = read_json(plan_path)
        quotas = {s: n // args.dataset_stages for s, n in (('train', 50000), ('val', 2000), ('test', 2000))}
        if plan.get('k') != args.dataset_stages or plan.get('pair_quotas') != quotas:
            raise RuntimeError('Il piano deve essere quello a %d stadi con 50000/2000/2000 campioni.' % args.dataset_stages)
    dependency = generation_dependency(parent, dataset, args.generation_job)
    if not dependency and not plan_path.is_file():
        raise RuntimeError('Piano della vista richiesto mancante: ' + str(plan_path))
    jobs = []
    for target, node, gpu_type, gpu_count, epochs, lr, old_name, requested, default_steps in (
            ('ss', 'wn49', 'l40s_vbd', 8, 40, '1e-4',
             'v4_steps8_ss_lora_all_bs1_lr1e4_e40', args.ss_steps, 300000),
            ('slat', 'wn51', 'h100_vbd', 2, 20, '5e-5',
             'v4_steps8_slat_lora_all_bs1_lr5e5_e20', args.slat_steps, 300000)):
        limit = (default_steps if requested is None else requested) if steps_mode else 0
        if steps_mode and limit <= 0:
            raise RuntimeError('Il numero di step %s deve essere positivo.' % target)
        scope = 'cross' if steps_mode else 'all'
        if steps_mode:
            steps_per_epoch = (100000 + gpu_count - 1) // gpu_count
            epochs = (limit + steps_per_epoch - 1) // steps_per_epoch
            default_name = 'v4_steps%d_%s_lora_cross_bs1_lr%s_s%d' % (args.dataset_stages, target, lr, limit)
        elif args.dataset_stages != 8:
            default_name = old_name.replace('steps8', 'steps%d' % args.dataset_stages)
        else:
            default_name = old_name
        name = os.environ.get(target.upper() + '_RUN_NAME', default_name)
        jobs.append((target, node, gpu_type, gpu_count, epochs, lr, name, limit, scope))
    # Validate both jobs before submitting either one.
    commands = []
    for target, node, gpu_type, gpu_count, epochs, lr, name, limit, scope in jobs:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
            raise RuntimeError('RUN_NAME non valido: ' + name)
        preset = 'steps' if steps_mode else 'steps8'
        wrapper = project / ('slurm/train_morphflow_v4_%s_%s.slurm' % (preset, target))
        if not wrapper.is_file():
            raise RuntimeError('Launcher mancante: ' + str(wrapper))
        require_resources(node, gpu_type, gpu_count)
        command = ['sbatch', '--parsable', '--export=ALL']
        if steps_mode or args.dataset_stages != 8:
            suffix = 's%d' % limit if steps_mode else 'e%d' % epochs
            command.append('--job-name=v4_steps%d_%s_%s_%s' % (args.dataset_stages, target, scope, suffix))
        if dependency:
            command.append('--dependency=afterok:' + dependency)
        command.append(str(wrapper))
        environment = dict(os.environ, PROJECT_DIR=str(project), DATA_PARENT_HOST=str(parent),
                           DATA_ROOT_HOST=str(dataset), NGPUS=str(gpu_count), TRAIN_BS='1',
                           TRAIN_EPOCHS=str(epochs), TRAIN_LR=lr, RUN_NAME=name,
                           DATA_STAGES=str(args.dataset_stages), MAX_TRAIN_STEPS=str(limit),
                           LORA_ATTENTION_SCOPE=scope,
                           CONDITION_GATE='token', SOURCE_SWAP='1', TRELLIS_PRIOR_WEIGHT='0.1',
                           EXPECTED_TRAIN_STEPS='0')
        commands.append((target, command, environment))
        updates = limit or epochs * (100000 // gpu_count)
        print('%s: %s | %d %s | bs=1/GPU | lr=%s | epoche=%d | update=%d | LoRA=%s | %s' %
              (target.upper(), node, gpu_count, gpu_type, lr, epochs, updates, scope, name), flush=True)
    print('Dataset: ' + str(dataset), flush=True)
    print('Dipendenza: ' + ('afterok:' + dependency if dependency else 'dataset completo'), flush=True)
    if args.check:
        for unused_target, command, unused_environment in commands:
            print('CHECK: ' + ' '.join(shlex.quote(value) for value in command), flush=True)
        print('CHECK OK: nessun job inviato.', flush=True)
        return
    submitted = []
    try:
        for target, command, environment in commands:
            result = run(command, environment)
            job = result.split(';', 1)[0]
            if not re.fullmatch(r'[1-9][0-9]*', job):
                raise RuntimeError('Risposta sbatch inattesa: ' + result)
            submitted.append((target, job))
            print('%s_JOB=%s' % (target.upper(), job), flush=True)
    except RuntimeError:
        if submitted:
            print('Job gia inviati: ' + ', '.join('%s=%s' % item for item in submitted), file=sys.stderr)
        raise
    print('SS e SLat sono indipendenti e attendono soltanto la generazione.', flush=True)


try:
    main()
except (OSError, ValueError, KeyError, RuntimeError) as error:
    print('ERRORE: ' + str(error), file=sys.stderr)
    sys.exit(1)
PY
