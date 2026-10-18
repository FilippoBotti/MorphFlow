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
for position, value in enumerate(argv):
    if value == '--bind':
        host, guest, mode = argv[position + 1].rsplit(':', 2)
        binds[guest] = host
bash = argv.index('bash')
command = argv[argv.index('--', bash) + 1:]
# Discard the actual container bootstrap, which belongs inside Singularity.
sys.stdin.read()

def record(event, **extra):
    value = dict(event=event, command=command, bindings=binds,
                 gpu=os.environ.get('SINGULARITYENV_CUDA_VISIBLE_DEVICES'), pid=os.getpid())
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
    return all((root / method / pair['pair_dir'] / ('alpha_' + str(i)) / 'pred_final.glb').is_file() for i in range(5))

METRICS = {
    'fid_endpoint_manifold': {'mean': 12., 'std': 1., 'median': 12.},
    'fid_endpoint_manifold_mean_per_state': {'mean': 15., 'std': 2., 'median': 15.},
    'fid_endpoint_manifold_pooled': {'mean': 7., 'std': 0., 'median': 7.},
    'ppl_full_normalized_reference_endpoints': {'mean': 3., 'std': .1, 'median': 3.},
    'pdv_full': {'mean': .001, 'std': .0001, 'median': .001},
}

if command[0] == '--check':
    # The separate protocol preparation helper is itself invoked by the launcher.
    assert '/protocol-overlay-parent' in binds and '/opt/trellis/env' in binds
    print('Fake offline protocol preparation verified')
elif command[:2] == ['python', '-c']:
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
elif command[2] == 'check-fid-weights':
    print('/opt/trellis/torch-home/hub/checkpoints/fake-fid.pth')
elif command[2] == 'plan':
    root = translate(option('--run-root'))
    manifest = json.loads((root.parent / 'run_manifest.json').read_text())
    assert int(option('--num-pairs')) == manifest['configuration']['N']
    assert option('--k') == '5' and option('--alpha-schedule') == 'beta' and option('--beta-shape') == '5'
    assert translate(option('--metadata')).read_bytes() == (root.parent / 'input_snapshot/metadata_test.json').read_bytes()
    count = manifest['evaluated_pairs']
    pairs = [dict(pair_id=i, pair_dir='pair_{:04d}_a_to_b'.format(i)) for i in range(count)]
    write_json(root / 'plan.json', dict(num_pairs=count, pairs=pairs, alphas=[.7, .6, .5, .4, .3]))
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
        (output / 'raw.done').write_text('five appearance-aware intermediates')
    timing = root / 'timings' / (method + '_shard_' + option('--shard-index') + '.jsonl')
    timing.parent.mkdir(parents=True, exist_ok=True)
    timing.write_text('{}\n')
elif command[2] == 'texture':
    root = translate(option('--run-root'))
    for pair in selected(plan_for(root)):
        for method in methods():
            assert (root / method / pair['pair_dir'] / 'raw.done').is_file()
            for index in range(5):
                output = root / method / pair['pair_dir'] / ('alpha_' + str(index)) / 'pred_final.glb'
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b'fake baked GLB')
elif command[2] == 'texture-status':
    root = translate(option('--run-root'))
    plan = plan_for(root)
    assert all(complete_glbs(root, method, pair) for method in methods() for pair in plan['pairs'])
    write_json(root / 'texture_status.json', dict(complete=True, methods=methods()))
elif command[2] == 'shard':
    assert option('--fid-aggregation') == 'both'
    assert option('--num-views') == '64' and option('--fid-num-views') == '1000'
    assert option('--camera-distribution') == 'hammersley' and option('--render-backend') == 'gaussian'
    assert option('--lpips-net') == 'vgg'
    assert option('--reference-cache-root') == '/benchmark/shared/reference_metrics'
    method_root = translate(option('--run-dir'))
    plan = plan_for(method_root.parent)
    pairs = selected(plan)
    assert all(complete_glbs(method_root.parent, method_root.name, pair) for pair in pairs)
    output = translate(option('--output-dir'))
    write_json(output / ('fake_shard_' + option('--shard-index') + '.json'), [pair['pair_id'] for pair in pairs])
elif command[2] == 'merge':
    assert option('--fid-aggregation') == 'both'
    output = translate(option('--output-dir'))
    expected = int(option('--expected-num-pairs'))
    assignments = [json.loads(path.read_text()) for path in output.glob('fake_shard_*.json')]
    ids = [i for group in assignments for i in group]
    assert sorted(ids) == list(range(expected)), (ids, expected)
    assert len(ids) == len(set(ids))
    write_json(output / 'metrics_batch.json', dict(num_pairs=expected, missing_pairs=[], aggregate_across_pairs=METRICS))
elif command[2] == 'aggregate':
    root = translate(option('--run-root'))
    plan = plan_for(root)
    data = dict(num_pairs=plan['num_pairs'], methods={})
    for method in methods():
        data['methods'][method] = json.loads((root / 'metrics' / method / 'metrics_batch.json').read_text())
    write_json(root / 'final_metrics.json', data)
    with (root / 'final_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'metric', 'mean', 'std', 'median'])
        writer.writeheader()
        writer.writerows(dict(method=method, metric=metric, **value) for method in methods() for metric, value in METRICS.items())
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
        for filename in ("benchmark_threeway_interp3d_l40s.slurm", "threeway_ateneo.py", "prepare_threeway_interp3d_l40s.sh"):
            shutil.copyfile(str(SLURM_DIR / filename), str(self.project / "slurm" / filename))
        (self.project / "benchmarks").mkdir()
        for filename in ("threeway_textured_benchmark.py", "eval_textured_benchmark.py"):
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

    def test_full_test_eight_gpu_two_cfg_grid_preserves_protocol_and_all_glbs(self):
        self.assert_success(self.run_launcher())
        calls = self.calls()
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
