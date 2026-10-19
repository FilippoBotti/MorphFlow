"""Exercise the real eight-GPU shell orchestration using a fake container runtime.

The host layout/snapshot/summarizer and launcher are real. Only GPU/container
commands are replaced: their arguments and shard coverage are checked, and small
artifacts stand in for expensive inference, GLB baking and image metrics.
"""

import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


SLURM_DIR = Path(__file__).resolve().parent

FAKE_SINGULARITY = r'''#!/usr/bin/env python3
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

argv = sys.argv[1:]
binds = {}
bind_modes = {}
for position, value in enumerate(argv):
    if value == '--bind':
        host, guest, mode = argv[position + 1].rsplit(':', 2)
        binds[guest] = host
        bind_modes[guest] = mode
bash = argv.index('bash')
command = argv[argv.index('--', bash) + 1:]
# The shell is not executed by the fake runtime. Evaluate the overlay role used
# by its conditional package path, and record mounts to verify ABI isolation.
bootstrap = sys.stdin.read()
metric_overlay = os.environ.get('SINGULARITYENV_MORPHFLOW_METRIC_OVERLAY', '0')
gaussian_overlay_enabled = (metric_overlay == '1' and
                            '/opt/protocol-python' in binds and
                            '/opt/protocol-python/gaussian' in bootstrap)

def record(event, **extra):
    value = dict(event=event, command=command, bindings=binds,
                 gpu=os.environ.get('SINGULARITYENV_CUDA_VISIBLE_DEVICES'), pid=os.getpid(),
                 gaussian_overlay_enabled=gaussian_overlay_enabled,
                 metric_overlay=metric_overlay, binding_modes=bind_modes)
    value.update(extra)
    with open(os.environ['FAKE_CONTAINER_JOURNAL'], 'a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(value) + '\n')
        stream.flush()

def terminate(signum, frame):
    record('terminated', signal=signum)
    raise SystemExit(128 + signum)

signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)
record('start')

def translate(value):
    # Longest mount point wins, including checkpoint submounts and split files.
    for guest in sorted(binds, key=len, reverse=True):
        if value == guest or value.startswith(guest + '/'):
            return Path(binds[guest] + value[len(guest):])
    return Path(value)

def option(flag):
    return command[command.index(flag) + 1]

def methods(default=('morphflow', 'morphany3d', 'interp3d')):
    if '--methods' not in command:
        return list(default)
    start = command.index('--methods') + 1
    end = next((i for i in range(start, len(command)) if command[i].startswith('--')), len(command))
    return command[start:end]

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))

def plan_for(root):
    return json.loads((root / 'plan.json').read_text())

def selected(plan):
    shard, count = int(option('--shard-index')), int(option('--num-shards'))
    assert count >= 1 and 0 <= shard < count
    return [pair for pair in plan['pairs'] if pair['pair_id'] % count == shard]

def complete_glbs(root, method, pair):
    return all((root / method / pair['pair_dir'] / ('alpha_' + str(i)) / 'pred_final.glb').is_file()
               for i in range(len(plan_for(root)['alphas'])))

def complete_raws(root, method, pair):
    paths = [root / method / pair['pair_dir'] / ('alpha_' + str(i)) / 'mesh_raw.pt'
             for i in range(len(plan_for(root)['alphas']))]
    paths.extend(root / 'references_raw' / pair['pair_dir'] / (source + '_raw.pt')
                 for source in ('src1', 'src2'))
    return all(path.is_file() and path.with_suffix('.appearance.json').is_file()
               for path in paths)

def dense_metrics():
    return dict((key, value) for key, value in METRICS.items() if key.startswith('weighted_endpoint_'))

METRICS = {
    'weighted_endpoint_lpips_mean': {'mean': .15, 'std': .01, 'median': .15},
    'weighted_endpoint_lpips_auc': {'mean': .06, 'std': .004, 'median': .06},
    'weighted_endpoint_lpips_full_mean': {'mean': .144, 'std': .0096, 'median': .144},
    'weighted_endpoint_lpips_full_auc': {'mean': .062, 'std': .004, 'median': .062},
    'fid_endpoint_manifold': {'mean': 12., 'std': 1., 'median': 12.},
    'fid_endpoint_manifold_mean_per_state': {'mean': 15., 'std': 2., 'median': 15.},
    'fid_endpoint_manifold_pooled': {'mean': 7., 'std': 0., 'median': 7.},
    'ppl_full_normalized_reference_endpoints': {'mean': 3., 'std': .1, 'median': 3.},
    'pdv_full': {'mean': .001, 'std': .0001, 'median': .001},
}

if command[0] == '--check':
    # The separate protocol preparation helper is itself invoked by the launcher.
    assert '/protocol-overlay-parent' in binds and '/opt/trellis/env' in binds
    assert '/prepare_gaussian_rasterizer.py' in binds
    assert translate('/prepare_gaussian_rasterizer.py').is_file()
    assert '/opt/trellis/toolchain/cuda118-gcc11' in binds
    print('Fake offline protocol preparation verified')
elif command[:2] == ['python', '-c']:
    assert ('if [[ "${MORPHFLOW_METRIC_OVERLAY:-0}" == 1 ]]' in bootstrap and
            'PYTHONPATH=/workspace/MethodRepo:/workspace/MorphFlow\n' in bootstrap)
    if 'detect_flow_target' in command[2]:
        assert translate('/checkpoints/ss/checkpoint.pt').read_bytes().startswith(b'SS')
        assert translate('/checkpoints/slat/checkpoint.pt').read_bytes().startswith(b'SLat')
        rows = [{'flow_target': 'ss'}, {'flow_target': 'slat'}]
        write_json(translate('/benchmark/checkpoint_metadata.json'), rows)
        print(json.dumps(rows))
    else:
        assert '/workspace/MethodRepo' in binds
        if command[-1] in {'cpu', 'gpu'}:
            assert bool(os.environ.get('SINGULARITYENV_CUDA_VISIBLE_DEVICES')) == (command[-1] == 'gpu')
        else:
            # Serialized full baseline-pipeline warmup, before GPU shards start.
            assert 'from_pretrained' in command[2]
            assert os.environ.get('SINGULARITYENV_CUDA_VISIBLE_DEVICES')
        print('Fake import/device check verified')
elif len(command) > 1 and Path(command[1]).name == 'weighted_endpoint_report.py':
    assert not os.environ.get('SINGULARITYENV_CUDA_VISIBLE_DEVICES')
    root = translate(option('--run-root'))
    completed = list(root.glob('cfg_ss_*/final_metrics.json'))
    assert completed
    for directory in [root] + [p.parent for p in completed]:
        output = directory / 'weighted_endpoint_lpips'
        output.mkdir(exist_ok=True)
        (output / 'evolution.png').write_bytes(b'fake CPU weighted LPIPS report')
        (output / 'evolution.pdf').write_bytes(b'fake CPU weighted LPIPS report')
        (output / 'curves.csv').write_text('alpha,weighted_lpips_mean\n0.5,0.15\n')
        (output / 'summary.csv').write_text('method,weighted_endpoint_lpips_mean\nmorphflow,0.15\n')
        write_json(output / 'summary.json', dict(formula='alpha*d1+(1-alpha)*d2'))
elif command[2] == 'check-fid-weights':
    print('/opt/trellis/torch-home/hub/checkpoints/fake-fid.pth')
elif command[2] == 'plan':
    root = translate(option('--run-root'))
    run_root = translate('/benchmark')
    manifest = json.loads((run_root / 'run_manifest.json').read_text())
    assert int(option('--num-pairs')) == manifest['configuration']['N']
    dense = '--protocol' in command and option('--protocol') == 'weighted-endpoints-50'
    if dense:
        assert option('--k') == '48' and option('--alpha-schedule') == 'uniform'
        assert '--beta-shape' not in command
        alphas = [index/49 for index in range(48, 0, -1)]
    else:
        assert option('--k') == '5' and option('--alpha-schedule') == 'beta' and option('--beta-shape') == '5'
        alphas = [.7, .6, .5, .4, .3]
    assert translate(option('--metadata')).read_bytes() == (run_root / 'input_snapshot/metadata_test.json').read_bytes()
    count = manifest['evaluated_pairs']
    pairs = [dict(pair_id=i, pair_dir='pair_{:04d}_a_to_b'.format(i)) for i in range(count)]
    write_json(root / 'plan.json', dict(num_pairs=count, pairs=pairs, alphas=alphas,
        k_intermediate=len(alphas), num_sequence_frames=len(alphas)+2,
        sequence_alphas=[1.0]+alphas+[0.0],
        metric_mode='weighted-endpoints-only' if dense else 'all'))
elif command[2] in {'morphflow', 'morphany3d', 'interp3d'}:
    method = command[2]
    root = translate(option('--run-root'))
    plan = plan_for(root)
    assert option('--steps') == '25' and option('--slat-steps') == '25'
    assert os.environ.get('SINGULARITYENV_CUDA_VISIBLE_DEVICES')
    if os.environ.get('FAKE_FAIL_GENERATION') == method:
        if option('--shard-index') == '0':
            # Allow all siblings to register before provoking a worker failure.
            time.sleep(.3)
            record('failed')
            raise SystemExit(17)
        time.sleep(60)
        record('unexpected_survivor')
    if method == 'morphflow':
        assert translate(option('--checkpoint-path')).read_bytes().startswith(b'SS')
        assert translate(option('--slat-checkpoint-path')).read_bytes().startswith(b'SLat')
    if method == 'morphany3d':
        assert option('--local-cache-root').startswith('/runtime-tmp/')
        assert option('--tfsa-spill-root').startswith('/runtime-tmp/')
    for pair in selected(plan):
        output = root / method / pair['pair_dir']
        output.mkdir(parents=True, exist_ok=True)
        (output / 'raw.done').write_text(str(len(plan['alphas']))+' appearance-aware intermediates')
        paths = [output / ('alpha_' + str(i)) / 'mesh_raw.pt' for i in range(len(plan['alphas']))]
        if method == 'morphflow':
            paths.extend(root / 'references_raw' / pair['pair_dir'] / (source + '_raw.pt')
                         for source in ('src1', 'src2'))
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fake mesh and Gaussian payload')
            write_json(path.with_suffix('.appearance.json'), dict(format_version=2, appearance='trellis_gaussian'))
    timing = root / 'timings' / (method + '_shard_' + option('--shard-index') + '.jsonl')
    timing.parent.mkdir(parents=True, exist_ok=True)
    timing.write_text('{}\n')
elif command[2] == 'texture':
    root = translate(option('--run-root'))
    plan = plan_for(root)
    for pair in selected(plan):
        for method in methods():
            assert (root / method / pair['pair_dir'] / 'raw.done').is_file()
            for index in range(len(plan['alphas'])):
                output = root / method / pair['pair_dir'] / ('alpha_' + str(index)) / 'pred_final.glb'
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b'fake baked GLB')
elif command[2] == 'texture-status':
    root = translate(option('--run-root'))
    plan = plan_for(root)
    assert all(complete_glbs(root, method, pair) for method in methods() for pair in plan['pairs'])
    write_json(root / 'texture_status.json', dict(complete=True, methods=methods()))
elif command[2] == 'shard':
    dense = '--metric-mode' in command
    if dense:
        assert option('--metric-mode') == 'weighted-endpoints-only'
        assert not any(value.startswith('--fid-') for value in command)
    else:
        assert option('--fid-aggregation') == 'both' and option('--fid-num-views') == '1000'
    assert option('--num-views') == '64'
    assert option('--camera-distribution') == 'hammersley' and option('--render-backend') == 'gaussian'
    assert option('--lpips-net') == 'vgg'
    assert option('--reference-cache-root') == '/benchmark/shared/reference_metrics'
    method_root = translate(option('--run-dir'))
    plan = plan_for(method_root.parent)
    assert (len(plan['alphas']) == 48) == dense
    pairs = selected(plan)
    assert all(complete_raws(method_root.parent, method_root.name, pair) for pair in pairs)
    manifest = json.loads((translate('/benchmark') / 'run_manifest.json').read_text())
    if manifest['configuration']['EXPORT_GLB']:
        assert all(complete_glbs(method_root.parent, method_root.name, pair) for pair in pairs)
    output = translate(option('--output-dir'))
    write_json(output / ('fake_shard_' + option('--shard-index') + '.json'), [pair['pair_id'] for pair in pairs])
elif command[2] == 'merge':
    dense = '--metric-mode' in command
    if dense:
        assert option('--metric-mode') == 'weighted-endpoints-only'
        assert not any(value.startswith('--fid-') for value in command)
    else:
        assert option('--fid-aggregation') == 'both'
    output = translate(option('--output-dir'))
    expected = int(option('--expected-num-pairs'))
    assignments = [json.loads(path.read_text()) for path in output.glob('fake_shard_*.json')]
    ids = [i for group in assignments for i in group]
    assert sorted(ids) == list(range(expected)), (ids, expected)
    assert len(ids) == len(set(ids))
    write_json(output / 'metrics_batch.json', dict(num_pairs=expected, missing_pairs=[],
               aggregate_across_pairs=dense_metrics() if dense else METRICS))
elif command[2] == 'aggregate':
    root = translate(option('--run-root'))
    plan = plan_for(root)
    data = dict(num_pairs=plan['num_pairs'], methods={}, k_intermediate=len(plan['alphas']),
                alphas=plan['alphas'], sequence_alphas=plan['sequence_alphas'],
                num_sequence_frames=plan['num_sequence_frames'], metric_mode=plan['metric_mode'])
    for method in methods():
        data['methods'][method] = json.loads((root / 'metrics' / method / 'metrics_batch.json').read_text())
    write_json(root / 'final_metrics.json', data)
    with (root / 'final_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'metric', 'mean', 'std', 'median'])
        writer.writeheader()
        writer.writerows(dict(method=method, metric=metric, **value) for method in methods()
                         for metric, value in data['methods'][method]['aggregate_across_pairs'].items())
    write_json(root / 'generation_timings.json', {})
    (root / 'generation_timings.csv').write_text('method,seconds\nmorphflow,1\n')
else:
    raise AssertionError('Unexpected container command: ' + repr(command))
record('done')
'''


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="threeway_l40s_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        (self.project / "slurm").mkdir(parents=True)
        for filename in ("benchmark_threeway_interp3d_l40s.slurm", "threeway_ateneo.py", "prepare_threeway_interp3d_l40s.sh",
                         "prepare_gaussian_rasterizer.py"):
            shutil.copyfile(str(SLURM_DIR / filename), str(self.project / "slurm" / filename))
        (self.project / "benchmarks").mkdir()
        for filename in ("threeway_textured_benchmark.py", "eval_textured_benchmark.py", "weighted_endpoint_report.py"):
            (self.project / "benchmarks" / filename).write_text("# mocked GPU work\n")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.executable("module", "#!/bin/sh\nexit 0\n")
        self.executable("singularity", "#!" + sys.executable + "\n" + FAKE_SINGULARITY.split("\n", 1)[1])
        self.executable("sleep", "#!/bin/sh\nexec /bin/sleep 0.02\n")
        self.executable("nvidia-smi", "#!/bin/sh\nprintf 'GPU-test-0\\nGPU-test-1\\nGPU-test-2\\nGPU-test-3\\nGPU-test-4\\nGPU-test-5\\nGPU-test-6\\nGPU-test-7\\n'\n")
        self.dataset = self.root / "data" / "morphing_dataset_v4"
        self.dataset.mkdir(parents=True)
        (self.root / "data" / "flux_outputs").mkdir()
        # Twenty metadata rows describe exactly ten unique, directed pairs.
        entries = [dict(src_1="source_" + str(i), src_2="target_" + str(i), alpha=alpha)
                   for i in range(10) for alpha in (.3, .7)]
        (self.dataset / "metadata_test.json").write_text(json.dumps(entries))
        (self.dataset / "split_assets.json").write_text(json.dumps({"test": ["test fixture"]}))
        self.ss = self.root / "ss-choice.pt"
        self.slat = self.root / "slat-choice.pt"
        self.ss.write_bytes(b"SS selected checkpoint")
        self.slat.write_bytes(b"SLat selected checkpoint")
        self.run_root = self.root / "runs" / "test_run"
        self.journal = self.root / "container_calls.jsonl"
        self.addCleanup(self.kill_remaining_mock_workers)
        self.env = dict(os.environ)
        # HPC login shells export a module() function which takes precedence
        # over PATH executables. Keep the mock independent of the host modules.
        for key in list(self.env):
            if key.startswith("BASH_FUNC_") or key in {"BASH_ENV", "ENV"}:
                self.env.pop(key)
        self.env.pop("CONFIG_FILE", None)
        self.env.pop("EXPORT_GLB", None)
        self.env.update({
            "PATH": str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
            "HOST_PYTHON": sys.executable, "PROJECT_DIR": str(self.project),
            "SIF": str(self.root / "container.sif"), "BENCH_ROOT": str(self.root / "bench"),
            "BENCH_ARCHIVE": str(self.root / "archive"), "FINAL_ARCHIVE": str(self.root / "reports"),
            "TRELLIS_BASE": str(self.root / "trellis"), "DATA_PARENT": str(self.root / "data"),
            "CHECKPOINT_PATH": str(self.ss), "SLAT_CHECKPOINT_PATH": str(self.slat),
            "SHARED_CACHE": str(self.root / "cache"), "PROTOCOL_PYTHON": str(self.root / "protocol"),
            "TOOLCHAIN_ROOT_HOST": str(self.root / "toolchain"),
            "ORIGINAL_ENV": str(self.root / "main_env"), "MORPHANY_ENV": str(self.root / "morphany_env"),
            "INTERP_ENV": str(self.root / "interp_env"), "TRELLIS_REPO_HOST": str(self.root / "TRELLIS"),
            "MORPHANY_REPO": str(self.root / "MorphAny3D"), "INTERP_REPO": str(self.root / "Interp3D"),
            "NODE_LOCAL_PARENT": str(self.root / "nvme"), "NVME_MIN_GB": "0",
            "RUN_NAME": "test_run", "RUN_ROOT": str(self.run_root),
            "CFG_VALUES": "2 5", "SLAT_CFG_VALUES": "1.5", "N": "0", "SEED": "42", "NGPUS": "8",
            "CPU_THREADS_PER_WORKER": "1", "SLURM_JOB_ID": "90210", "SLURM_JOB_NUM_NODES": "1",
            "SLURM_CPUS_PER_TASK": "64", "CUDA_VISIBLE_DEVICES": ",".join("GPU-test-" + str(i) for i in range(8)),
            "FAKE_CONTAINER_JOURNAL": str(self.journal),
            "K": "5", "STEPS": "25", "SLAT_STEPS": "25", "BETA_SHAPE": "5", "NUM_VIEWS": "64",
            "FID_NUM_VIEWS": "1000", "LPIPS_NET": "vgg", "FID_AGGREGATION": "both", "RENDER_BACKEND": "gaussian",
            "MIXED_PRECISION": "bf16", "TEXTURE_SIZE": "1024", "SIMPLIFY": "0.95", "FILL_HOLES": "1",
            "SOURCE_IMAGE_FILENAME": "", "METADATA": "metadata_test.json",
            "DATASET_NAME": "morphing_dataset_v4", "SOURCE_IMAGES_NAME": "flux_outputs",
            "WEIGHTED_DENSE": "0",
        })
        self.env.pop("FAKE_FAIL_GENERATION", None)
        Path(self.env["SIF"]).write_bytes(b"fake container")
        for key in ("ORIGINAL_ENV", "MORPHANY_ENV", "INTERP_ENV", "TOOLCHAIN_ROOT_HOST", "PROTOCOL_PYTHON"):
            Path(self.env[key]).mkdir(parents=True)
        for key in ("TRELLIS_REPO_HOST", "MORPHANY_REPO", "INTERP_REPO"):
            repo = Path(self.env[key]) / "trellis"
            repo.mkdir(parents=True)
            (repo / "__init__.py").write_text("# isolated fork fixture\n")
        for directory in ("hf-home", "torch-home", "u2net"):
            (Path(self.env["SHARED_CACHE"]) / directory).mkdir(parents=True)

    def executable(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(0o755)

    def run_launcher(self, mode="--run", timeout=30):
        command = ["bash", str(self.project / "slurm" / "benchmark_threeway_interp3d_l40s.slurm"), mode]
        started = time.monotonic()
        process = subprocess.Popen(command, cwd=str(self.project), env=self.env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            self.fail("Launcher hung instead of finishing/terminating workers:\n" + stdout + "\n" + stderr)
        return process.returncode, stdout, stderr, time.monotonic() - started

    def calls(self, event="start"):
        if not self.journal.is_file():
            return []
        return [row for row in map(json.loads, self.journal.read_text().splitlines()) if row["event"] == event]

    def kill_remaining_mock_workers(self):
        # If the real launcher regresses and leaves grandchildren running, this
        # test must report the regression without leaving its own fake workers.
        for row in self.calls():
            pid = row["pid"]
            try:
                cmdline = Path("/proc") / str(pid) / "cmdline"
                if str(self.bin / "singularity").encode() in cmdline.read_bytes():
                    os.kill(pid, signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError):
                pass

    @staticmethod
    def operation(call):
        command = call["command"]
        return command[2] if len(command) > 2 and command[:2] != ["python", "-c"] else None

    def assert_success(self, result):
        self.assertEqual(result[0], 0, result[1] + "\n" + result[2])

    def assert_gaussian_runtime_prepared(self, calls):
        preparation = [row for row in calls if row['command'][0] == '--check']
        self.assertEqual(len(preparation), 1)
        binding = preparation[0]['bindings']['/prepare_gaussian_rasterizer.py']
        self.assertEqual(Path(binding).read_bytes(),
                         (self.project / 'slurm/prepare_gaussian_rasterizer.py').read_bytes())
        workers = [row for row in calls if row not in preparation]
        self.assertTrue(workers)
        environments_seen = set()
        for row in workers:
            environment = row['bindings']['/opt/trellis/env']
            environments_seen.add(environment)
            self.assertEqual(row['binding_modes']['/opt/trellis/env'], 'ro')
            self.assertEqual(row['binding_modes']['/workspace/MethodRepo'], 'ro')
            enabled = environment == self.env['MORPHANY_ENV']
            self.assertEqual(row['metric_overlay'], str(int(enabled)))
            self.assertEqual(row['gaussian_overlay_enabled'], enabled)
            self.assertEqual('/opt/protocol-python' in row['bindings'], enabled)
            if enabled:
                self.assertEqual(row['bindings']['/opt/protocol-python'], self.env['PROTOCOL_PYTHON'])
                self.assertEqual(row['binding_modes']['/opt/protocol-python'], 'ro')
            operation = self.operation(row)
            if operation in ('morphflow', 'morphany3d', 'interp3d'):
                expected_env, fork = {
                    'morphflow': ('ORIGINAL_ENV', 'TRELLIS'),
                    'morphany3d': ('MORPHANY_ENV', 'MorphAny3D'),
                    'interp3d': ('INTERP_ENV', 'Interp3D'),
                }[operation]
                self.assertEqual(environment, self.env[expected_env])
                self.assertEqual(Path(row['bindings']['/workspace/MethodRepo']).name, fork)
            if operation in ('shard', 'merge', 'texture', 'texture-status', 'aggregate', 'check-fid-weights'):
                self.assertTrue(enabled, operation)
        self.assertEqual(environments_seen, {self.env[name] for name in
                         ('ORIGINAL_ENV', 'MORPHANY_ENV', 'INTERP_ENV')})
        rasterizer_checks = [row for row in workers
                             if row['command'][:2] == ['python', '-c'] and
                             'runpy.run_path' in row['command'][2] and
                             'smoke_gpu' in row['command'][2]]
        self.assertEqual(len(rasterizer_checks), 1)
        self.assertEqual(rasterizer_checks[0]['command'][-1], 'gpu')
        self.assertEqual(rasterizer_checks[0]['gpu'], 'GPU-test-0')

    def test_full_test_eight_gpu_two_cfg_grid_preserves_protocol_and_all_glbs(self):
        self.env['EXPORT_GLB'] = '1'
        self.assert_success(self.run_launcher())
        calls = self.calls()
        self.assert_gaussian_runtime_prepared(calls)
        grouped = {}
        for call in calls:
            grouped.setdefault(self.operation(call), []).append(call)
        self.assertEqual(len(grouped["morphflow"]), 16)
        self.assertEqual(len(grouped["morphany3d"]), 8)
        self.assertEqual(len(grouped["interp3d"]), 8)
        for operation in ("morphany3d", "interp3d"):
            workers = grouped[operation]
            self.assertEqual({row["command"][row["command"].index("--shard-index") + 1] for row in workers}, set(map(str, range(8))))
            self.assertEqual({row["gpu"] for row in workers}, {"GPU-test-" + str(i) for i in range(8)})
        self.assertEqual(len(grouped["texture"]), 24)  # eight shared + eight per CFG cell
        self.assertEqual(len(grouped["shard"]), 32)  # two baselines + two MorphFlow cells
        self.assertEqual(len(grouped["merge"]), 4)
        self.assertEqual(len(grouped["aggregate"]), 2)
        self.assertEqual(len(grouped["texture-status"]), 3)
        shard_phases = {}
        for row in calls:
            command = row["command"]
            if "--shard-index" in command:
                phase = Path(row["bindings"]["/runtime-home"]).name.rsplit("_g", 1)[0]
                shard_phases.setdefault(phase, []).append(row)
                index = command[command.index("--shard-index") + 1]
                self.assertEqual(command[command.index("--num-shards") + 1], "8")
                self.assertEqual(row["gpu"], "GPU-test-" + index)
        for phase, workers in shard_phases.items():
            self.assertEqual(len(workers), 8, phase)
            self.assertEqual({row["command"][row["command"].index("--shard-index") + 1] for row in workers}, set(map(str, range(8))), phase)
        for row in grouped["merge"]:
            command = row["command"]
            self.assertEqual(command[command.index("--expected-num-pairs") + 1], "10")
            self.assertEqual(command[command.index("--fid-aggregation") + 1], "both")
        manifest = json.loads((self.run_root / "run_manifest.json").read_text())
        self.assertEqual((manifest["metadata_entries"], manifest["unique_test_pairs"], manifest["evaluated_pairs"]), (20, 10, 10))
        self.assertEqual((self.run_root / "checkpoints/ss/checkpoint.pt").read_bytes(), self.ss.read_bytes())
        self.assertEqual((self.run_root / "checkpoints/slat/checkpoint.pt").read_bytes(), self.slat.read_bytes())
        for cfg in ("2", "5"):
            cell = self.run_root / ("cfg_ss_" + cfg + "_slat_1.5")
            self.assertEqual(len(list((cell / "morphflow").glob("pair_*/alpha_*/pred_final.glb"))), 50)
            for name in ("plan.json", "morphany3d", "interp3d", "metrics/morphany3d", "metrics/interp3d"):
                self.assertTrue((cell / name).is_symlink())
                self.assertFalse(os.path.isabs(os.readlink(str(cell / name))))
        for method in ("morphany3d", "interp3d"):
            self.assertEqual(len(list((self.run_root / "shared" / method).glob("pair_*/alpha_*/pred_final.glb"))), 50)
        grid = json.loads((self.run_root / "cfg_grid_metrics.json").read_text())
        self.assertEqual((grid["completed_configurations"], grid["expected_configurations"]), (2, 2))
        with (self.run_root / "cfg_grid_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row["configuration"] for row in rows}, {"cfg_ss_2_slat_1.5", "cfg_ss_5_slat_1.5"})
        self.assertTrue((Path(self.env["FINAL_ARCHIVE"]) / "cfg_grid_metrics.csv").is_file())
        report_calls = [row for row in calls if len(row['command']) > 1 and Path(row['command'][1]).name == 'weighted_endpoint_report.py']
        self.assertEqual(len(report_calls), 2)
        self.assertTrue((self.run_root / 'weighted_endpoint_lpips/evolution.png').is_file())
        self.assertTrue((Path(self.env['FINAL_ARCHIVE']) / 'weighted_endpoint_lpips/evolution.pdf').is_file())
        self.assertIn('weighted_endpoint_lpips_mean', {row['metric'] for row in rows})

    def test_dense_weighted_fifty_frames_adds_separate_workload_without_repeating_baselines_per_cfg(self):
        self.env['EXPORT_GLB'] = '1'
        self.env['WEIGHTED_DENSE'] = '1'
        self.assert_success(self.run_launcher(timeout=45))
        calls = self.calls()
        self.assert_gaussian_runtime_prepared(calls)
        grouped = {}
        for call in calls:
            grouped.setdefault(self.operation(call), []).append(call)
        self.assertEqual(len(grouped['plan']), 2)
        plans = {row['command'][row['command'].index('--run-root')+1]: row['command']
                 for row in grouped['plan']}
        ordinary = plans['/benchmark/shared']
        dense = plans['/benchmark/weighted_50/shared']
        self.assertEqual(ordinary[ordinary.index('--k')+1], '5')
        self.assertEqual(ordinary[ordinary.index('--alpha-schedule')+1], 'beta')
        self.assertEqual(dense[dense.index('--k')+1], '48')
        self.assertEqual(dense[dense.index('--alpha-schedule')+1], 'uniform')
        self.assertEqual(dense[dense.index('--protocol')+1], 'weighted-endpoints-50')
        self.assertEqual(len(grouped['morphflow']), 32)  # 2 workloads × 2 CFG × 8 GPUs
        self.assertEqual(len(grouped['texture']), 48)
        self.assertEqual(len(grouped['shard']), 64)
        self.assertEqual(len(grouped['merge']), 8)
        self.assertEqual(len(grouped['aggregate']), 4)
        for method in ('morphany3d', 'interp3d'):
            workers = grouped[method]
            self.assertEqual(len(workers), 16)  # Each baseline generated once per workload.
            by_root = {}
            for row in workers:
                command = row['command']
                by_root.setdefault(command[command.index('--run-root')+1], []).append(row)
            self.assertEqual(set(by_root), {'/benchmark/shared', '/benchmark/weighted_50/shared'})
            for workload_workers in by_root.values():
                self.assertEqual(len(workload_workers), 8)
                self.assertEqual({row['gpu'] for row in workload_workers},
                                 {'GPU-test-'+str(index) for index in range(8)})
        dense_root = self.run_root / 'weighted_50'
        plan = json.loads((dense_root / 'shared/plan.json').read_text())
        self.assertEqual(len(plan['alphas']), 48)
        self.assertEqual(len(plan['sequence_alphas']), 50)
        self.assertEqual(plan['sequence_alphas'][0], 1)
        self.assertEqual(plan['sequence_alphas'][-1], 0)
        self.assertAlmostEqual(plan['alphas'][0], 48/49)
        self.assertAlmostEqual(plan['alphas'][-1], 1/49)
        self.assertEqual((dense_root / 'shared/references_raw').resolve(),
                         (self.run_root / 'shared/references_raw').resolve())
        self.assertEqual((dense_root / 'shared/references_textured').resolve(),
                         (self.run_root / 'shared/references_textured').resolve())
        for root, k in ((self.run_root, 5), (dense_root, 48)):
            for cfg in ('2', '5'):
                cell = root / ('cfg_ss_'+cfg+'_slat_1.5')
                self.assertEqual(len(list((cell/'morphflow').glob('pair_*/alpha_*/pred_final.glb'))), 10*k)
                for name in ('plan.json', 'morphany3d', 'interp3d', 'metrics/morphany3d', 'metrics/interp3d'):
                    self.assertTrue((cell/name).is_symlink())
                    self.assertFalse(os.path.isabs(os.readlink(str(cell/name))))
            for method in ('morphany3d', 'interp3d'):
                self.assertEqual(len(list((root/'shared'/method).glob('pair_*/alpha_*/pred_final.glb'))), 10*k)
            grid = json.loads((root/'cfg_grid_metrics.json').read_text())
            self.assertEqual((grid['completed_configurations'], grid['expected_configurations']), (2, 2))
        for row in grouped['shard'] + grouped['merge']:
            command = row['command']
            location = command[command.index('--output-dir')+1]
            if location.startswith('/benchmark/weighted_50/'):
                self.assertEqual(command[command.index('--metric-mode')+1], 'weighted-endpoints-only')
                self.assertFalse(any(argument.startswith('--fid-') for argument in command))
            else:
                self.assertNotIn('--metric-mode', command)
                self.assertEqual(command[command.index('--fid-aggregation')+1], 'both')
        with (dense_root/'cfg_grid_metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row['metric'] for row in rows},
                         {'weighted_endpoint_lpips_mean', 'weighted_endpoint_lpips_auc',
                          'weighted_endpoint_lpips_full_mean', 'weighted_endpoint_lpips_full_auc'})
        report_calls = [row for row in calls if len(row['command']) > 1 and
                        Path(row['command'][1]).name == 'weighted_endpoint_report.py']
        self.assertEqual(len(report_calls), 4)
        dense_archive = Path(self.env['FINAL_ARCHIVE'])/'weighted_50'
        self.assertTrue((dense_archive/'cfg_grid_metrics.csv').is_file())
        self.assertTrue((dense_archive/'weighted_endpoint_lpips/evolution.pdf').is_file())
        for cfg in ('2', '5'):
            archived_cell = dense_archive/('cfg_ss_'+cfg+'_slat_1.5')
            self.assertTrue((archived_cell/'final_metrics.json').is_file())
            self.assertTrue((archived_cell/'weighted_endpoint_lpips/evolution.png').is_file())

    def assert_raw_only_workloads(self, dense):
        self.env['WEIGHTED_DENSE'] = str(int(dense))
        # Leave EXPORT_GLB unset to exercise the production default.
        self.assertNotIn('EXPORT_GLB', self.env)
        self.assert_success(self.run_launcher(timeout=45))
        calls = self.calls()
        self.assert_gaussian_runtime_prepared(calls)
        operations = [self.operation(row) for row in calls]
        self.assertNotIn('texture', operations)
        self.assertNotIn('texture-status', operations)
        self.assertEqual(operations.count('morphflow'), 16 * (1 + int(dense)))
        for method in ('morphany3d', 'interp3d'):
            self.assertEqual(operations.count(method), 8 * (1 + int(dense)))
        self.assertEqual(operations.count('shard'), 32 * (1 + int(dense)))
        self.assertEqual(operations.count('merge'), 4 * (1 + int(dense)))
        self.assertEqual(operations.count('aggregate'), 2 * (1 + int(dense)))
        manifest = json.loads((self.run_root / 'run_manifest.json').read_text())
        self.assertEqual(manifest['configuration']['EXPORT_GLB'], 0)
        self.assertEqual(manifest['asset_export'], {
            'glb_enabled': False,
            'raw_mesh_and_gaussian': True,
            'glb_can_be_baked_later': True,
        })
        self.assertEqual(list(self.run_root.rglob('*.glb')), [])
        self.assertEqual(list(self.run_root.rglob('texture_status.json')), [])
        archive = Path(self.env['FINAL_ARCHIVE'])
        self.assertEqual(list(archive.rglob('*.glb')), [])
        self.assertEqual(list(archive.rglob('texture_status.json')), [])
        roots = [(self.run_root, archive, 5)]
        if dense:
            roots.append((self.run_root / 'weighted_50', archive / 'weighted_50', 48))
        for root, saved, k in roots:
            for cfg in ('2', '5'):
                cell = root / ('cfg_ss_' + cfg + '_slat_1.5')
                raw_paths = list((cell / 'morphflow').glob('pair_*/alpha_*/mesh_raw.pt'))
                self.assertEqual(len(raw_paths), 10 * k)
                self.assertTrue(all(p.with_suffix('.appearance.json').is_file() for p in raw_paths))
                for method in ('morphany3d', 'interp3d'):
                    self.assertTrue((cell / method).is_symlink())
                    self.assertEqual((cell / method).resolve(), root / 'shared' / method)
                    raw_paths = list((cell / method).glob('pair_*/alpha_*/mesh_raw.pt'))
                    self.assertEqual(len(raw_paths), 10 * k)
                    self.assertTrue(all(p.with_suffix('.appearance.json').is_file() for p in raw_paths))
                self.assertTrue((saved / cell.name / 'final_metrics.json').is_file())
            references = list((root / 'shared/references_raw').glob('pair_*/*_raw.pt'))
            self.assertEqual(len(references), 20)
            self.assertTrue(all(p.with_suffix('.appearance.json').is_file() for p in references))
            grid = json.loads((root / 'cfg_grid_metrics.json').read_text())
            self.assertEqual((grid['completed_configurations'], grid['expected_configurations']), (2, 2))
            with (saved / 'cfg_grid_metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertIn('weighted_endpoint_lpips_mean', {row['metric'] for row in rows})
            if k == 5:
                self.assertIn('fid_endpoint_manifold', {row['metric'] for row in rows})
                self.assertIn('fid_endpoint_manifold_mean_per_state', {row['metric'] for row in rows})
            else:
                self.assertTrue(all(row['metric'].startswith('weighted_endpoint_') for row in rows))
            self.assertTrue((saved / 'weighted_endpoint_lpips/evolution.pdf').is_file())

    def test_default_raw_only_runs_standard_metrics_without_any_glb_export(self):
        self.assert_raw_only_workloads(dense=False)

    def test_default_raw_only_runs_both_protocols_with_shared_baselines_and_all_raws(self):
        self.assert_raw_only_workloads(dense=True)

    def test_insufficient_cuda_allocation_fails_before_generation(self):
        self.env["CUDA_VISIBLE_DEVICES"] = "GPU-test-0,GPU-test-1"
        rc, out, err, _ = self.run_launcher()
        self.assertNotEqual(rc, 0)
        self.assertIn("fewer than NGPUS", err)
        self.assertFalse(any(self.operation(row) in {"morphflow", "morphany3d", "interp3d"} for row in self.calls()))

    def test_cpu_check_ignores_nodevfiles_cuda_sentinel(self):
        self.env["CUDA_VISIBLE_DEVICES"] = "NoDevFiles"
        self.env.pop("SLURM_JOB_ID")
        result = self.run_launcher("--check")
        self.assert_success(result)
        self.assertIn("CHECK PASSED: 10 pairs", result[1])
        calls = self.calls()
        preflights = [row for row in calls if row["command"][:2] == ["python", "-c"]]
        self.assertEqual(len(preflights), 5)
        self.assertTrue(all(row["gpu"] == "" for row in preflights))
        self.assertFalse(any(self.operation(row) in {"morphflow", "morphany3d", "interp3d", "texture", "shard"} for row in calls))

    def test_failed_worker_stops_next_phases_and_terminates_siblings(self):
        self.env["FAKE_FAIL_GENERATION"] = "morphflow"
        rc, out, err, elapsed = self.run_launcher(timeout=15)
        self.assertNotEqual(rc, 0)
        self.assertIn("shard 0 failed", err)
        self.assertLess(elapsed, 8.0)
        calls = self.calls()
        workers = [row for row in calls if self.operation(row) == "morphflow"]
        self.assertEqual(len(workers), 8)
        self.assertEqual(len([row for row in self.calls("terminated") if self.operation(row) == "morphflow"]), 7)
        self.assertFalse(self.calls("unexpected_survivor"))
        self.assertFalse(any(self.operation(row) in {"morphany3d", "interp3d", "texture", "shard", "merge", "aggregate"} for row in calls))
        self.assertFalse((self.run_root / "cfg_grid_metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
