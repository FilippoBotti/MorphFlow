#!/usr/bin/env python3
"""Resume a frozen benchmark with a narrowly scoped mesh-kernel fallback.

On Ateneo copy this file AND mesh_channel_blocks.py together OUTSIDE the
checkout; do not git-pull into that run's checkout. Native Leonardo resumes use
only the already frozen source files.
The original launcher, source snapshots and manifest stay untouched.
The external launcher and worker override are recorded under runtime_repairs/.
Only the Python standard library is needed to prepare/submit on the login node.
"""

import argparse
import functools
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone


SCRIPT = "/workspace/MorphFlow/benchmarks/threeway_textured_benchmark.py"
WORKER = "/opt/morphflow-mesh-repair/resume_threeway_mesh.py"
LAUNCHER = "MorphFlow/slurm/benchmark_threeway_interp3d_l40s.slurm"
NATIVE_LAUNCHER = "MorphFlow/slurm/benchmark_interp3ddata_leonardo.slurm"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_int32_error(error):
    return isinstance(error, RuntimeError) and "data exceed int32 range" in str(error)


def record_fallback(payload):
    payload = dict(payload, utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid())
    print("MESH KERNEL FALLBACK: " + json.dumps(payload), file=sys.stderr, flush=True)
    directory = os.environ.get("MORPHFLOW_MESH_REPAIR_LOG_DIR")
    if directory:
        path = Path(directory) / ("mesh_fallbacks_%s.jsonl" % os.getpid())
        with path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")


@functools.lru_cache(maxsize=1)
def load_channel_blocker():
    path = Path(__file__).with_name("mesh_channel_blocks.py")
    spec = importlib.util.spec_from_file_location("mesh_channel_blocks_runtime", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.channel_block_subm


def decode_with_mesh_fallback(original, mesh_decoder, sparse_tensor_cls, feats, coords, device):
    try:
        return original(mesh_decoder, sparse_tensor_cls, feats, coords, device)
    except RuntimeError as error:
        if not native_int32_error(error):
            raise
        reason = str(error)
    # Leave the except block before retrying: its traceback owns large GPU
    # activations. Keeping it alive can turn an index-limit failure into an OOM.
    import torch
    import spconv
    import spconv.constants as constants
    from spconv.pytorch import ConvAlgo
    from spconv.pytorch.conv import SparseConvolution

    if not getattr(constants, "ALL_WEIGHT_IS_KRSC", False):
        raise RuntimeError(
            "Mesh int32 fallback requires ALL_WEIGHT_IS_KRSC=True to preserve the "
            "weight layout. This older spconv needs a separately validated fix. " + reason
        )
    layers = []
    for module in mesh_decoder.modules():
        if not isinstance(module, SparseConvolution) or module.algo != ConvAlgo.Native:
            continue
        kernel = module.kernel_size
        volume = kernel if isinstance(kernel, int) else math.prod(kernel)
        if volume > 1:
            if volume > 32:
                raise RuntimeError("Mesh fallback does not support kernel volume >32. " + reason)
            layers.append(module)
    if not layers:
        raise RuntimeError("Mesh int32 failure has no compatible Native convolutions to retry. " + reason)
    event = {
        "reason": reason, "spconv": getattr(spconv, "__version__", "unknown"),
        "sparse_input_rows": int(feats.shape[0]), "layers": len(layers),
        "from": "Native", "to": "Native_FP32_channel_blocks", "precision_changed": False,
    }
    if "--run-root" in sys.argv:
        index = sys.argv.index("--run-root")
        if index + 1 < len(sys.argv):
            event["cell"] = sys.argv[index + 1]
    record_fallback(dict(event, status="retry_started"))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    restores = []
    block_convolution = load_channel_blocker()
    try:
        for module in layers:
            had_override = "forward" in module.__dict__
            previous_override = module.__dict__.get("forward")
            previous_forward = module.forward
            restores.append((module, had_override, previous_override))

            def bounded_forward(sparse_input, module=module, previous_forward=previous_forward):
                return block_convolution(
                    module, sparse_input, original_forward=previous_forward,
                    on_split=lambda details: record_fallback(dict(event, status="channel_split", **details)),
                )

            module.forward = bounded_forward
        # The original function constructs a NEW SparseTensor, so the retry does
        # not retain sparse state from the failed decoder invocation.
        try:
            result = original(mesh_decoder, sparse_tensor_cls, feats, coords, device)
        except Exception as retry_error:
            record_fallback(dict(event, status="retry_failed", retry_error=str(retry_error)))
            raise
        record_fallback(dict(event, status="retry_succeeded"))
        return result
    finally:
        for module, had_override, previous_override in restores:
            if had_override:
                module.forward = previous_override
            else:
                del module.forward


def worker_main():
    script = os.environ.get("MORPHFLOW_MESH_REPAIR_SCRIPT", SCRIPT)
    sys.path.insert(0, str(Path(script).parent))
    sys.argv[0] = script
    namespace = runpy.run_path(script, run_name="threeway_mesh_repair_worker")
    main = namespace["main"]
    # run_path returns a copy; patch the actual globals referenced by functions.
    original = main.__globals__["decode_slat_mesh"]
    main.__globals__["decode_slat_mesh"] = functools.partial(decode_with_mesh_fallback, original)
    main()


def native_aliases(root):
    snapshot = root / "source_snapshot"
    return {
        "PROJECT_DIR": snapshot / "MorphFlow", "TRELLIS_REPO_HOST": snapshot / "TRELLIS",
        "MORPHANY_REPO": snapshot / "MorphAny3D", "INTERP_REPO": snapshot / "Interp3D",
        "CHECKPOINT_PATH": root / "checkpoints/ss/checkpoint.pt",
        "SLAT_CHECKPOINT_PATH": root / "checkpoints/slat/checkpoint.pt",
    }


def verify_frozen_run(run_root):
    """Keep original integrity checks, resolving native source identities to pins.

    The live checkout may now include repair tools. We execute the frozen helper
    and let it verify all its ORIGINAL source, checkpoint and dataset hashes.
    Only the six path aliases are translated back to their manifest identities;
    all other requested settings must still exactly match the manifest.
    """
    root = Path(run_root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest["configuration"].get("EXECUTION_BACKEND") != "native":
        raise ValueError("Frozen native verification requires a native manifest")
    namespace = runpy.run_path(str(root / "source_snapshot/MorphFlow/slurm/threeway_ateneo.py"))
    prepare = namespace["prepare"]
    globals_ = prepare.__globals__
    requested = globals_["requested_configuration"]

    def requested_from_pins(environment):
        actual = requested(environment)
        for key, pin in native_aliases(root).items():
            if actual[key] != str(pin.resolve()):
                raise ValueError("Native repair must use frozen input: " + key)
            actual[key] = manifest["configuration"][key]
        return actual

    def frozen_sources(project, repositories):
        return {name: root / "source_snapshot" / name for name in manifest["sources"]}

    globals_["requested_configuration"] = requested_from_pins
    globals_["source_files"] = frozen_sources
    prepare(argparse.Namespace(run_root=str(root)))


def prepare_resume(run_root):
    root = Path(run_root).resolve()
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported run manifest schema")
    config = manifest["configuration"]
    backend = config.get("EXECUTION_BACKEND", "container")
    if backend not in ("container", "native"):
        raise ValueError("Unsupported execution backend")
    # Adding a repair tool inside PROJECT_DIR changes its source fingerprint.
    if backend == "container" and Path(config["PROJECT_DIR"]).resolve() in Path(__file__).resolve().parents:
        raise ValueError("Copy this tool outside PROJECT_DIR before using it; do not git-pull to resume")
    source = NATIVE_LAUNCHER if backend == "native" else LAUNCHER
    frozen_launcher = root / "source_snapshot" / source
    if sha256(frozen_launcher) != manifest["sources"][source]:
        raise ValueError("Frozen launcher differs from its manifest; refusing to patch it")
    text = frozen_launcher.read_text()
    if backend == "native":
        script_line = 'SCRIPT="$ACTIVE_PROJECT/benchmarks/threeway_textured_benchmark.py"'
        prepare_line = 'run_python "$ORIGINAL_ENV" "$TRELLIS_REPO_HOST" "$HELPER" prepare --run-root "$RUN_ROOT"'
        phase_line = '        phase="${prefix}$(basename "$cell")"\n'
        if any(text.count(line) != 1 for line in (script_line, prepare_line, phase_line)):
            raise ValueError("Unexpected native launcher layout; original run has not been modified")
        text = text.replace(script_line, 'SCRIPT="$MORPHFLOW_MESH_REPAIR_DIR/resume_threeway_mesh.py"', 1)
        text = text.replace(prepare_line,
                            'run_python "$ORIGINAL_ENV" "$TRELLIS_REPO_HOST" '
                            '"$MORPHFLOW_MESH_REPAIR_DIR/resume_threeway_mesh.py" '
                            '--verify-frozen --run-root "$RUN_ROOT"', 1)
        smoke_before = 'run_phase canonical_endpoints '
        if smoke_before in text:
            text = text.replace(smoke_before,
                                'run_phase mesh_channel_blocks_smoke "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" '
                                '"$MORPHFLOW_MESH_REPAIR_DIR/mesh_channel_blocks.py"\n' + smoke_before, 1)
        text = text.replace(phase_line, phase_line +
                            '        if [[ -s "$cell/final_metrics.json" && -s "$cell/final_metrics.csv" ]]; then\n'
                            '            echo "RESUME: $phase already completed; retaining its existing results"\n'
                            '            baselines_done=1\n'
                            '            continue\n'
                            '        fi\n', 1)
    else:
        script_line = "SCRIPT_IN=" + SCRIPT
        bind_line = '        --bind "$SNAPSHOT/MorphFlow:/workspace/MorphFlow:ro" \\\n'
        phase_line = '    PHASE_NAME="${phase_prefix}${CELL_NAME}"\n'
        if any(text.count(line) != 1 for line in (script_line, bind_line, phase_line)):
            raise ValueError("Unexpected launcher layout; original run has not been modified")
        text = text.replace(script_line, "SCRIPT_IN=" + WORKER, 1)
        text = text.replace(bind_line,
                            '        --bind "$MORPHFLOW_MESH_REPAIR_DIR:/opt/morphflow-mesh-repair:ro" \\\n' + bind_line, 1)
        text = text.replace(phase_line, phase_line +
                            '    if [[ -s "$CELL_HOST/final_metrics.json" && -s "$CELL_HOST/final_metrics.csv" ]]; then\n'
                            '        echo "RESUME: $PHASE_NAME already completed; retaining its existing results"\n'
                            '        BASELINES_DONE=1\n'
                            '        continue\n'
                            '    fi\n', 1)
        smoke_before = 'run_evaluation_grid interp3d /benchmark "" all'
        if smoke_before in text:
            text = text.replace(smoke_before,
                                'run_container mesh_channel_blocks_smoke "$ORIGINAL_ENV" "$SNAPSHOT/TRELLIS" '
                                '"${GPU_TOKENS[0]}" python /opt/morphflow-mesh-repair/mesh_channel_blocks.py\n' + smoke_before, 1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    repair = root / "runtime_repairs" / ("mesh_native_int32_" + stamp)
    repair.mkdir(parents=True)
    helper = repair / "resume_threeway_mesh.py"
    shutil.copyfile(__file__, helper)
    blocks = repair / "mesh_channel_blocks.py"
    shutil.copyfile(Path(__file__).with_name("mesh_channel_blocks.py"), blocks)
    launcher = repair / "resume.slurm"
    launcher.write_text(text)
    environment = {}
    for key, value in config.items():
        if key == "METADATA_HOST":
            environment["METADATA"] = str(value)
        elif isinstance(value, list):
            environment[key] = " ".join(str(item) for item in value)
        else:
            environment[key] = str(value)
    environment.update({
        "RUN_ROOT": str(root), "RUN_NAME": root.name,
        "MORPHFLOW_MESH_REPAIR_DIR": str(repair),
    })
    if backend == "native":
        environment.update({key: str(path) for key, path in native_aliases(root).items()})
        environment["MORPHFLOW_MESH_REPAIR_SCRIPT"] = str(root / "source_snapshot/MorphFlow/benchmarks/threeway_textured_benchmark.py")
        environment["MORPHFLOW_MESH_REPAIR_LOG_DIR"] = str(repair)
        base = Path(config["PROJECT_DIR"]).parents[1]
        environment["FINAL_ARCHIVE"] = str(base / "outputs/threeway_benchmark" / root.name)
        log_prefix = str(base / "logs/interp3ddata_threeway_")
    else:
        data = Path(config["DATA_ROOT_HOST"])
        images = Path(config["SOURCE_IMAGES_ROOT_HOST"])
        if data.parent != images.parent:
            raise ValueError("Expected the original Ateneo dataset and images under the same parent")
        environment.update({
            "BENCH_ROOT": str(root.parent.parent),
            "DATA_PARENT": str(data.parent), "DATASET_NAME": data.name,
            "SOURCE_IMAGES_NAME": images.name,
            "SINGULARITYENV_MORPHFLOW_MESH_REPAIR_LOG_DIR": "/benchmark/" + repair.relative_to(root).as_posix(),
        })
        log_prefix = config["PROJECT_DIR"] + "/threeway-l40s-threeway_v4_interp3d-"
    record = {
        "repair": "mesh_native_int32_retry_only", "launcher": str(launcher),
        "repair_dir": str(repair), "environment": environment,
        "backend": backend, "log_prefix": log_prefix,
        "original_manifest_sha256": sha256(manifest_path),
        "original_launcher_sha256": sha256(frozen_launcher),
        "repair_launcher_sha256": sha256(launcher), "repair_worker_sha256": sha256(helper),
        "channel_blocks_sha256": sha256(blocks),
        "scope": "Only mesh decoding retries Native int32 failures with bounded FP32 channel blocks; "
                 "original snapshots, checkpoints, sampling, Gaussian decoder and protocol are preserved",
        "gpu_validation": "Must be confirmed by the resumed GPU job; no failed pairs are skipped",
    }
    (repair / "repair.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--run-root", type=Path)
    selector.add_argument("--job-id", type=int, help="Find the run from its phase-log directory")
    parser.add_argument("--bench-root", type=Path,
                        default=Path("/hpc/scratch/marco.barezzi/morph-benchmark"))
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--verify-frozen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.run_root
    if root is None:
        matches = {path.parent.parent for path in
                   (args.bench_root / "threeway-runs").glob("*/logs/job_%s_*" % args.job_id)}
        if len(matches) != 1:
            parser.error("Expected exactly one run for that job; pass --run-root explicitly")
        root = matches.pop()
    if args.verify_frozen:
        verify_frozen_run(root)
        return
    # The normal launcher holds the same lock throughout the entire job.
    import fcntl
    with (root / ".benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = prepare_resume(root)
    print("Prepared mesh-only recovery: " + record["repair_dir"], flush=True)
    if args.submit:
        subprocess.check_call(["bash", "-n", record["launcher"]])
        environment = os.environ.copy()
        # Reconstruct the original configuration instead of inheriting a stale
        # shell CONFIG_FILE or RUN_ROOT from a different benchmark.
        environment.pop("CONFIG_FILE", None)
        environment.update(record["environment"])
        result = subprocess.run(["sbatch", "--parsable", "--export=ALL", record["launcher"]],
                                cwd=record["environment"]["PROJECT_DIR"], env=environment,
                                check=True, universal_newlines=True, stdout=subprocess.PIPE)
        job = result.stdout.strip()
        record["submitted_job"] = job
        (Path(record["repair_dir"]) / "repair.json").write_text(json.dumps(record, indent=2) + "\n")
        print("Submitted: " + job)
        number = job.split(";", 1)[0]
        prefix = record["log_prefix"] + number
        print("tail -n 60 -F '%s.out' '%s.err'" % (prefix, prefix))


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        worker_main()
    else:
        main()
