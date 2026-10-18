"""Run layout, provenance and summaries for the native Leonardo benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SOURCE_FILES = (
    "benchmarks/threeway_textured_benchmark.py",
    "benchmarks/eval_textured_benchmark.py",
    "eval_perceptual_sequence.py",
    "slurm/threeway_leonardo.py",
    "slurm/benchmark_threeway_textured_leonardo.slurm",
)
REPOSITORIES = ("TRELLIS_REPO", "MORPHANY_REPO", "INTERP_REPO", "PROJECT_DIR")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repository, *arguments):
    return subprocess.check_output(["git", "-C", str(repository), *arguments])


def untracked_files(repository):
    names = git_output(repository, "ls-files", "--others", "--exclude-standard", "-z")
    # Python creates these at import time. They are derived runtime caches,
    # rather than source inputs, including in submodules without a .gitignore.
    return [Path(os.fsdecode(name)) for name in names.split(b"\0") if name
            and "__pycache__" not in Path(os.fsdecode(name)).parts
            and Path(os.fsdecode(name)).suffix not in (".pyc", ".pyo")]


def working_tree_diff(repository):
    return git_output(repository, "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--submodule=diff")


def repository_state(repository):
    repository = Path(repository).resolve()
    diff = working_tree_diff(repository)
    untracked = {}
    for relative in untracked_files(repository):
        path = repository / relative
        details = {}
        if path.is_symlink():
            details["symlink_target"] = os.readlink(path)
        if path.is_file():
            details["sha256"] = sha256_file(path)
        untracked[str(relative)] = details
    submodules = {}
    for entry in git_output(repository, "ls-files", "--stage", "-z").split(b"\0"):
        if not entry or not entry.startswith(b"160000 "):
            continue
        _, relative = entry.split(b"\t", 1)
        relative = os.fsdecode(relative)
        path = repository / relative
        if not (path / ".git").exists():
            submodules[relative] = {"initialized": False}
        else:
            submodules[relative] = repository_state(path)
    return {
        "path": str(repository),
        "revision": git_output(repository, "rev-parse", "HEAD").decode().strip(),
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_files": untracked,
        "submodules": submodules,
    }


def metadata_path():
    path = Path(os.environ["METADATA"])
    return path if path.is_absolute() else Path(os.environ["ROOT_DIR"]) / path


def provenance():
    project = Path(os.environ["PROJECT_DIR"])
    inputs = {}
    for name, path in (("metadata", metadata_path()),
                       ("split_assets", Path(os.environ["ROOT_DIR"]) / "split_assets.json")):
        inputs[name] = {"path": str(path.resolve()), "sha256": sha256_file(path) if path.is_file() else None}
    return {
        "source_sha256": {name: sha256_file(project / name) for name in SOURCE_FILES},
        "dataset_inputs": inputs,
        "repositories": {key: repository_state(os.environ[key]) for key in REPOSITORIES},
    }


def snapshot_repository(repository, destination):
    repository = Path(repository)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "working_tree.patch").write_bytes(working_tree_diff(repository))
    for relative in untracked_files(repository):
        source, target = repository / relative, destination / "untracked" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(os.readlink(source), target_is_directory=source.is_dir())
        elif source.is_file():
            shutil.copy2(source, target)
    for entry in git_output(repository, "ls-files", "--stage", "-z").split(b"\0"):
        if entry.startswith(b"160000 "):
            _, relative = entry.split(b"\t", 1)
            relative = Path(os.fsdecode(relative))
            if (repository / relative / ".git").exists():
                snapshot_repository(repository / relative, destination / "submodules" / relative)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def link(destination, source):
    destination, source = Path(destination), Path(source).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source:
            raise ValueError(f"Unexpected link: {destination} -> {destination.resolve()}")
    elif destination.exists():
        raise ValueError(f"Refusing to replace existing output: {destination}")
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())


def checkpoint_metadata(path):
    import torch
    from eval_validation_latents import checkpoint_args, detect_flow_target

    checkpoint = torch.load(path, mmap=True, map_location="cpu", weights_only=False)
    config = checkpoint_args(checkpoint)
    return {
        "path": str(Path(path).resolve()),
        "bytes": Path(path).stat().st_size,
        "mtime_ns": Path(path).stat().st_mtime_ns,
        "flow_target": detect_flow_target(checkpoint),
        "epoch": checkpoint.get("epoch"),
        "step": checkpoint.get("step"),
        "val_loss": checkpoint.get("val_loss"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "dataset_root": config.get("root_dir"),
    }


def configuration():
    keys = (
        "N", "K", "SEED", "STEPS", "SLAT_STEPS", "CFG_VALUES", "SLAT_CFG_VALUES",
        "ALPHA_SCHEDULE", "BETA_SHAPE", "NUM_VIEWS", "FID_NUM_VIEWS", "FID_MIN_SAMPLES",
        "RENDER_RESOLUTION", "LPIPS_NET", "RENDER_BACKEND", "MIXED_PRECISION",
        "ROOT_DIR", "SOURCE_IMAGES_ROOT", "SOURCE_IMAGE_FILENAME", "METADATA",
        "CHECKPOINT_PATH", "SLAT_CHECKPOINT_PATH", "RUN_ROOT", "TRELLIS_REPO",
        "MORPHANY_REPO", "INTERP_REPO", "MORPHANY_ENV", "INTERP_ENV", "BAKE_TEXTURES",
        "PROJECT_DIR", "MORPHFLOW_ENV", "BLENDER_BIN",
    )
    return {key: os.environ.get(key, "") for key in keys}


def check(args):
    import importlib

    for key in ("N", "K", "STEPS", "SLAT_STEPS", "NUM_VIEWS", "FID_NUM_VIEWS", "FID_MIN_SAMPLES"):
        if int(os.environ[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if int(os.environ["K"]) < 2:
        raise ValueError("K must be at least 2 for two distinct intermediate FID states")
    if int(os.environ["SEED"]) < 0:
        raise ValueError("SEED must be nonnegative")
    if os.environ["ALPHA_SCHEDULE"] not in ("beta", "uniform"):
        raise ValueError("ALPHA_SCHEDULE must be beta or uniform")
    if os.environ["RENDER_BACKEND"] not in ("gaussian", "blender"):
        raise ValueError("RENDER_BACKEND must be gaussian or blender")
    if not math.isfinite(float(os.environ["BETA_SHAPE"])) or float(os.environ["BETA_SHAPE"]) <= 0:
        raise ValueError("BETA_SHAPE must be finite and positive")
    for key in ("CFG_VALUES", "SLAT_CFG_VALUES"):
        values = os.environ[key].split()
        if not values or len(set(map(float, values))) != len(values):
            raise ValueError(f"{key} is empty or contains duplicate values")
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in values):
            raise ValueError(f"Invalid {key}: {values}")
    checkpoints = [checkpoint_metadata(os.environ[key]) for key in ("CHECKPOINT_PATH", "SLAT_CHECKPOINT_PATH")]
    if [c["flow_target"] for c in checkpoints] != ["ss", "slat"]:
        raise ValueError("Expected SS and SLat checkpoints, in that order")
    entries = json.loads(metadata_path().read_text())
    pairs = {(entry["src_1"], entry["src_2"]) for entry in entries}
    if len(pairs) < int(os.environ["N"]):
        raise ValueError(f"Requested {os.environ['N']} pairs but test metadata contains {len(pairs)}")
    from huggingface_hub import hf_hub_download
    for name in ("ss_dec_conv3d_16l8_fp16", "slat_dec_mesh_swin8_B_64l8m256c_fp16", "slat_dec_gs_swin8_B_64l8gs32_fp16"):
        for extension in ("json", "safetensors"):
            hf_hub_download("microsoft/TRELLIS-image-large", f"ckpts/{name}.{extension}", local_files_only=True)
    for module in ("torch", "torchvision", "scipy", "lpips"):
        importlib.import_module(module)
    print(json.dumps({"checkpoint_metadata": checkpoints, "test_entries": len(entries), "unique_test_pairs": len(pairs)}, indent=2))


def initialize(args):
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    requested = configuration()
    signature = hashlib.sha256(json.dumps(requested, sort_keys=True).encode()).hexdigest()
    manifest = run_root / "run_manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["configuration_sha256"] != signature:
            differences = {key: [old["configuration"].get(key), value] for key, value in requested.items()
                           if old["configuration"].get(key) != value}
            raise ValueError(f"Run configuration changed; use a new RUN_ROOT: {differences}")
        if old.get("schema_version") != 2 or "provenance" not in old:
            raise ValueError("Run manifest lacks reproducible input/source fingerprints; use a new RUN_ROOT")
        current_provenance = provenance()
        if current_provenance != old["provenance"]:
            changed = [key for key, value in current_provenance.items() if value != old["provenance"].get(key)]
            raise ValueError(f"Benchmark sources, repository state or dataset inputs changed ({', '.join(changed)}); use a new RUN_ROOT")
        if len(old.get("checkpoints", [])) != 2:
            raise ValueError("Run manifest lacks both checkpoint identities; use a new RUN_ROOT")
        for key, saved in zip(("CHECKPOINT_PATH", "SLAT_CHECKPOINT_PATH"), old["checkpoints"]):
            current = checkpoint_metadata(os.environ[key])
            if current != saved:
                raise ValueError(f"Checkpoint changed since benchmark started: {current['path']}; use a new RUN_ROOT")
        print(f"Resuming {run_root}")
        return
    project = Path(os.environ["PROJECT_DIR"])
    fingerprints = provenance()
    checkpoints = [checkpoint_metadata(os.environ[key]) for key in ("CHECKPOINT_PATH", "SLAT_CHECKPOINT_PATH")]
    # Publish the complete snapshot directory first, then atomically publish
    # its manifest. A killed initialization cannot expose a partial manifest.
    sources = Path(tempfile.mkdtemp(prefix=".source_snapshot.", dir=run_root))
    try:
        for name in SOURCE_FILES:
            destination = sources / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project / name, destination)
            if sha256_file(destination) != fingerprints["source_sha256"][name]:
                raise ValueError(f"Source changed during initialization: {name}; retry before generating outputs")
        for key in REPOSITORIES:
            snapshot_repository(os.environ[key], sources / "repositories" / key)
        for name, details in fingerprints["dataset_inputs"].items():
            if details["sha256"] is not None:
                destination = sources / "dataset_inputs" / f"{name}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(details["path"], destination)
                if sha256_file(destination) != details["sha256"]:
                    raise ValueError(f"Dataset input changed during initialization: {details['path']}")
        if provenance() != fingerprints:
            raise ValueError("Sources or dataset inputs changed during initialization; retry before generating outputs")
        destination = run_root / "source_snapshot"
        if destination.exists():
            # Preserve an orphan from interrupted initialization for inspection.
            orphan = run_root / ("source_snapshot.uncommitted." + sources.name.rsplit(".", 1)[-1])
            destination.rename(orphan)
        sources.replace(destination)
    except BaseException:
        shutil.rmtree(sources, ignore_errors=True)
        raise
    write_json(manifest, {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": requested, "configuration_sha256": signature,
        "checkpoints": checkpoints,
        "provenance": fingerprints,
        "source_sha256": fingerprints["source_sha256"],
        "git_revisions": {key: value["revision"] for key, value in fingerprints["repositories"].items()},
        "source_snapshot": "source_snapshot",
        "dataset_note": "Held-out v4 test pairs; this is not the original Interp3D paper dataset.",
        "baseline_reuse": "MorphAny3D and Interp3D are generated/evaluated once and shared across all MorphFlow CFG cells.",
    })


def cell(args):
    run_root = Path(args.run_root).resolve()
    shared = run_root / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    destination = run_root / f"cfg_ss_{args.cfg}_slat_{args.slat_cfg}"
    destination.mkdir(exist_ok=True)
    plan = shared / "plan.json"
    link(destination / "plan.json", plan)
    for name in ("references_raw", "references_textured", "morphany3d", "interp3d"):
        (shared / name).mkdir(exist_ok=True)
        link(destination / name, shared / name)
    for name in ("morphany3d", "interp3d"):
        (shared / "metrics" / name).mkdir(parents=True, exist_ok=True)
        link(destination / "metrics" / name, shared / "metrics" / name)
    print(destination)


def summarize(args):
    run_root = Path(args.run_root).resolve()
    cells = []
    rows = []
    for path in sorted(run_root.glob("cfg_ss_*/final_metrics.json")):
        metrics = json.loads(path.read_text())
        cells.append({"configuration": path.parent.name, "metrics": metrics})
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            with csv_path.open() as handle:
                rows.extend({"configuration": path.parent.name, **row} for row in csv.DictReader(handle))
    write_json(run_root / "cfg_grid_metrics.json", {"completed_configurations": len(cells), "cells": cells})
    if rows:
        with (run_root / "cfg_grid_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Completed CFG cells: {len(cells)}; metrics: {run_root / 'cfg_grid_metrics.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "initialize", "cell", "summarize"))
    parser.add_argument("--run-root", default=os.environ.get("RUN_ROOT", ""))
    parser.add_argument("--cfg")
    parser.add_argument("--slat-cfg")
    args = parser.parse_args()
    {"check": check, "initialize": initialize, "cell": cell, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
