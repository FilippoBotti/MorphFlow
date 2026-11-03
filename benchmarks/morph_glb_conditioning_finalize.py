"""Resume incomplete pairs once and finalize their gallery through Slurm dependencies."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def submit(arguments):
    result = subprocess.run(['sbatch', '--parsable', *map(str, arguments)],
                            check=True, text=True, capture_output=True)
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Unexpected sbatch response: ' + result.stdout)
    return int(job)


def finalize(root):
    from morph_glb_conditioning_report import report
    summary = report(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    expected = []
    for asset in manifest['assets']:
        expected += [root / 'assets' / asset['asset'] / name
                     for name in ('source.glb', 'source_vertex.glb')]
    for pair in manifest['pairs']:
        for branch in ('dataset_control', 'glb_encoded'):
            for alpha in manifest['alphas']:
                folder = root / 'pairs' / pair['pair_dir'] / branch / (('alpha_%.6f' % alpha).replace('.', 'p'))
                expected.append(folder / 'prediction.glb')
                if manifest['export']['textured_midpoints'] and abs(alpha - 0.5) < 1e-10:
                    expected.append(folder / 'prediction_textured.glb')
    missing = [str(p.relative_to(root)) for p in expected if not p.is_file() or not p.stat().st_size]
    verification = {'status': 'complete' if summary['status'] == 'complete' and not missing else 'incomplete',
                    'completed_pairs': summary['completed_pairs'],
                    'completed_alpha_comparisons': summary['completed_alpha_comparisons'],
                    'expected_glbs': len(expected), 'present_glbs': len(expected) - len(missing),
                    'missing': missing, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                    'validation': 'Exports are validated by each worker; finalization checks completion and expected artifact presence.'}
    write_json(root / 'verification.json', verification)
    report(root)  # Include the final verification in the portable preview.
    print(json.dumps(verification, indent=2), flush=True)
    return verification


def resume(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    execution = json.loads((root / 'execution.json').read_text())
    retries = {0: execution.get('retry_pair0_on_failure'), 1: execution.get('retry_pair1_on_failure')}
    job_ids = [str(job) for job in retries.values() if job]
    active = set()
    if job_ids:
        result = subprocess.run(['squeue', '-h', '-j', ','.join(job_ids), '-o', '%A'],
                                check=True, capture_output=True, text=True)
        active = {int(value.strip()) for value in result.stdout.splitlines() if value.strip().isdigit()}
    todo, dependencies = [], set()
    for index, pair in enumerate(manifest['pairs']):
        if (root / 'pairs' / pair['pair_dir'] / 'result.json').is_file():
            continue
        if retries.get(index) in active:
            dependencies.add(retries[index])
        else:
            todo.append(index)
    if todo:
        job = submit(['--job-name=morph_glb_resume', '--time=01:30:00',
                      '--array=' + ','.join(map(str, todo)) + '%10',
                      PROJECT / 'slurm/benchmark_morph_glb_conditioning_leonardo.slurm', root, 'worker'])
        execution.setdefault('automatic_resume_arrays', []).append({'job_id': job, 'pair_indices': todo})
        dependencies.add(job)
    if dependencies:
        job = submit(['--job-name=morph_glb_report',
                      '--dependency=afterany:' + ':'.join(map(str, sorted(dependencies))),
                      PROJECT / 'slurm/finalize_morph_glb_conditioning_leonardo.slurm', root, 'report'])
        execution['final_report_job'] = job
        write_json(root / 'execution.json', execution)
        print(json.dumps({'resume_indices': todo, 'waiting_on': sorted(dependencies), 'final_report_job': job}), flush=True)
    else:
        finalize(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('mode', choices=['continue', 'report'])
    args = parser.parse_args()
    root = args.run_root.resolve()
    try:
        (resume if args.mode == 'continue' else finalize)(root)
    except Exception as exc:
        write_json(root / 'finalization_error.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
