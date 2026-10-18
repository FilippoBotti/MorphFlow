"""Portable host-side layout and immutable inputs for the eight-L40S benchmark.

Only the Python standard library is required on the login/compute host.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL = {
    "K": "5", "STEPS": "25", "SLAT_STEPS": "25", "BETA_SHAPE": "5",
    "NUM_VIEWS": "64", "FID_NUM_VIEWS": "1000", "LPIPS_NET": "vgg",
    "FID_AGGREGATION": "both", "RENDER_BACKEND": "gaussian",
}
IGNORED = {".git", "__pycache__", ".pytest_cache", "outputs", "logs", "wandb", "envs", ".venv"}
SOURCE_SUFFIXES = {".py", ".json", ".sh", ".slurm", ".sbatch", ".txt", ".yaml", ".yml", ".toml", ".md"}
REPOS = {"TRELLIS": "TRELLIS_REPO_HOST", "MorphAny3D": "MORPHANY_REPO", "Interp3D": "INTERP_REPO"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2)
            f.write("\n")
        os.replace(name, str(path))
    finally:
        if os.path.exists(name):
            os.unlink(name)


def cfg_values(value):
    values = [float(v) for v in value.replace(",", " ").split()]
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("CFG lists must contain finite nonnegative numbers")
    if len(set(values)) != len(values):
        raise ValueError("CFG lists must not contain duplicate numerical values")
    return [format(v, ".12g") for v in values]


def requested_configuration(env):
    for key, expected in PROTOCOL.items():
        if env.get(key, expected) != expected:
            raise ValueError("{}={} changes the paper protocol; expected {}".format(key, env[key], expected))
    n = int(env.get("N", "0"))
    if n < 0 or int(env.get("SEED", "42")) < 0:
        raise ValueError("N and SEED must be nonnegative (N=0 means all test pairs)")
    if env.get("MIXED_PRECISION", "bf16") not in {"bf16", "fp16", "no"}:
        raise ValueError("MIXED_PRECISION must be bf16, fp16 or no")
    texture_size = int(env.get("TEXTURE_SIZE", "1024"))
    simplify = float(env.get("SIMPLIFY", "0.95"))
    if texture_size < 1 or not math.isfinite(simplify) or not 0 <= simplify < 1:
        raise ValueError("Invalid texture size or simplify ratio")
    if env.get("FILL_HOLES", "1") not in {"0", "1"}:
        raise ValueError("FILL_HOLES must be 0 or 1")
    configuration = dict(PROTOCOL)
    configuration.update({
        "N": n, "SEED": int(env.get("SEED", "42")),
        "CFG_VALUES": cfg_values(env.get("CFG_VALUES", "2 3 5")),
        "SLAT_CFG_VALUES": cfg_values(env.get("SLAT_CFG_VALUES", "1.5 2 2.5")),
        "MIXED_PRECISION": env.get("MIXED_PRECISION", "bf16"),
        "TEXTURE_SIZE": texture_size, "SIMPLIFY": simplify,
        "FILL_HOLES": int(env.get("FILL_HOLES", "1")),
        "SOURCE_IMAGE_FILENAME": env.get("SOURCE_IMAGE_FILENAME", ""),
    })
    for key in ("PROJECT_DIR", "DATA_ROOT_HOST", "SOURCE_IMAGES_ROOT_HOST", "CHECKPOINT_PATH",
                "SLAT_CHECKPOINT_PATH", "ORIGINAL_ENV", "MORPHANY_ENV", "INTERP_ENV", "SIF",
                "TRELLIS_REPO_HOST", "MORPHANY_REPO", "INTERP_REPO", "PROTOCOL_PYTHON"):
        configuration[key] = str(Path(env[key]).resolve())
    metadata = Path(env.get("METADATA", "metadata_test.json"))
    if not metadata.is_absolute():
        metadata = Path(configuration["DATA_ROOT_HOST"]) / metadata
    configuration["METADATA_HOST"] = str(metadata.resolve())
    return configuration


def metadata_pairs(metadata):
    entries = json.loads(Path(metadata).read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Test metadata must be a nonempty list")
    pairs = set()
    for row in entries:
        if not isinstance(row, dict) or not row.get("src_1") or not row.get("src_2"):
            raise ValueError("Every test entry must contain src_1 and src_2")
        pairs.add((str(row["src_1"]), str(row["src_2"])))
    return len(entries), len(pairs)


def source_files(project, repositories):
    result = {}
    project = Path(project)
    for parent, directories, names in os.walk(str(project)):
        directories[:] = sorted(d for d in directories if d not in IGNORED)
        for name in sorted(names):
            path = Path(parent) / name
            if path.suffix in SOURCE_SUFFIXES and path.is_file():
                result["MorphFlow/" + path.relative_to(project).as_posix()] = path
    for name, repo in repositories.items():
        trellis = Path(repo) / "trellis"
        if not trellis.is_dir():
            raise FileNotFoundError("Missing method fork: {}".format(trellis))
        for path in sorted(trellis.rglob("*")):
            relative = path.relative_to(repo)
            if any(p in IGNORED for p in relative.parts) or not path.is_file():
                continue
            result[name + "/" + relative.as_posix()] = path
    return result


def copy_checkpoint(source, destination):
    source, destination = Path(source), Path(destination)
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint.", dir=str(destination.parent))
    h = hashlib.sha256()
    try:
        with source.open("rb") as src, os.fdopen(fd, "wb") as dst:
            for chunk in iter(lambda: src.read(4 * 1024 * 1024), b""):
                h.update(chunk)
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        after = source.stat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if identity(before) != identity(after):
            raise ValueError("Checkpoint changed during snapshot; retry: {}".format(source))
        os.replace(temporary, str(destination))
        return {"original_path": str(source.resolve()), "snapshot": str(destination),
                "bytes": after.st_size, "sha256": h.hexdigest()}
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare(args):
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = requested_configuration(os.environ)
    entries, pairs = metadata_pairs(config["METADATA_HOST"])
    if config["N"] > pairs:
        raise ValueError("Requested {} pairs, but metadata has {}".format(config["N"], pairs))
    actual = pairs if config["N"] == 0 else config["N"]
    repos = {name: config[key] for name, key in REPOS.items()}
    files = source_files(config["PROJECT_DIR"], repos)
    sources = {name: digest(path) for name, path in files.items()}
    inputs = {"metadata": digest(config["METADATA_HOST"])}
    splits = {name: Path(config["DATA_ROOT_HOST"]) / name
              for name in ("split_assets.json", "splits.json")}
    for name, split in splits.items():
        if split.is_file():
            inputs[name] = digest(split)
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text())
        if old.get("schema_version") != 1 or old["configuration"] != config:
            raise ValueError("Configuration changed: choose a new RUN_NAME/RUN_ROOT")
        if old["sources"] != sources or old["dataset_inputs"] != inputs:
            raise ValueError("Code or test metadata changed: choose a new RUN_NAME/RUN_ROOT")
        for item in old["checkpoints"].values():
            if digest(item["snapshot"]) != item["sha256"]:
                raise ValueError("Pinned checkpoint snapshot changed: {}".format(item["snapshot"]))
        for name, expected in sources.items():
            if digest(root / "source_snapshot" / name) != expected:
                raise ValueError("Source snapshot changed: {}".format(name))
        if digest(root / "input_snapshot/metadata_test.json") != inputs["metadata"]:
            raise ValueError("Frozen test metadata changed")
        for name in splits:
            if name in inputs and digest(root / "input_snapshot" / name) != inputs[name]:
                raise ValueError("Frozen split changed: {}".format(name))
        print("Resume verified; using the existing immutable checkpoint snapshots.")
        return
    checkpoints = {
        "ss": copy_checkpoint(config["CHECKPOINT_PATH"], root / "checkpoints/ss/checkpoint.pt"),
        "slat": copy_checkpoint(config["SLAT_CHECKPOINT_PATH"], root / "checkpoints/slat/checkpoint.pt"),
    }
    snapshot = root / "source_snapshot"
    for name, source in files.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(source), str(target))
        if digest(target) != sources[name]:
            raise ValueError("Source changed during snapshot: {}".format(source))
    (root / "input_snapshot").mkdir(exist_ok=True)
    shutil.copyfile(config["METADATA_HOST"], str(root / "input_snapshot/metadata_test.json"))
    if digest(root / "input_snapshot/metadata_test.json") != inputs["metadata"]:
        raise ValueError("Test metadata changed during snapshot")
    for name, split in splits.items():
        if name in inputs:
            target = root / "input_snapshot" / name
            shutil.copyfile(str(split), str(target))
            if digest(target) != inputs[name]:
                raise ValueError("Split changed during snapshot: {}".format(split))
    for name in ("shared/references_raw", "shared/references_textured", "shared/morphany3d", "shared/interp3d", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config, "metadata_entries": entries, "unique_test_pairs": pairs,
        "evaluated_pairs": actual, "checkpoints": checkpoints, "sources": sources,
        "dataset_inputs": inputs,
        "fid_definitions": {
            "fid_endpoint_manifold": "mean_pairs FID(concat(R_source,R_target), concat(G_selected_1,G_selected_2))",
            "fid_endpoint_manifold_mean_per_state": "mean_pairs mean_selected_states FID(concat(R_source,R_target), G_state)",
            "fid_endpoint_manifold_pooled": "FID pooled over all pairs; diagnostic only",
        },
        "scope": "Interp3D-style counts and metric definitions on the selected held-out test set; both explicit FID interpretations.",
    })
    print("Prepared {} unique test pairs, {} CFG cells; all intermediate GLBs required.".format(
        actual, len(config["CFG_VALUES"]) * len(config["SLAT_CFG_VALUES"])))


def relative_link(destination, source):
    destination, source = Path(destination), Path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = os.path.relpath(str(source), str(destination.parent))
    if destination.is_symlink():
        if os.readlink(str(destination)) != expected:
            raise ValueError("Unexpected symlink: {}".format(destination))
    elif destination.exists():
        raise ValueError("Refusing to replace output: {}".format(destination))
    else:
        destination.symlink_to(expected)


def cell(args):
    root = Path(args.run_root).resolve()
    cfg, slat = cfg_values(args.cfg), cfg_values(args.slat_cfg)
    if len(cfg) != 1 or len(slat) != 1:
        raise ValueError("A cell needs one SS and one SLat CFG")
    manifest = json.loads((root / "run_manifest.json").read_text())
    if cfg[0] not in manifest["configuration"]["CFG_VALUES"] or slat[0] not in manifest["configuration"]["SLAT_CFG_VALUES"]:
        raise ValueError("CFG not present in the immutable run manifest")
    target = root / ("cfg_ss_{}_slat_{}".format(cfg[0], slat[0]))
    target.mkdir(exist_ok=True)
    shared = root / "shared"
    relative_link(target / "plan.json", shared / "plan.json")
    for name in ("references_raw", "references_textured", "morphany3d", "interp3d"):
        (shared / name).mkdir(exist_ok=True)
        relative_link(target / name, shared / name)
    for method in ("morphany3d", "interp3d"):
        (shared / "metrics" / method).mkdir(parents=True, exist_ok=True)
        relative_link(target / "metrics" / method, shared / "metrics" / method)
    for path in (shared / "timings").glob("*.jsonl"):
        relative_link(target / "timings" / path.name, path)
    print(target)


def summarize(args):
    root = Path(args.run_root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    expected_pairs = manifest["evaluated_pairs"]
    expected_cells = len(manifest["configuration"]["CFG_VALUES"]) * len(manifest["configuration"]["SLAT_CFG_VALUES"])
    cells, rows = [], []
    for path in sorted(root.glob("cfg_ss_*/final_metrics.json")):
        data = json.loads(path.read_text())
        if data["num_pairs"] != expected_pairs:
            raise ValueError("Incomplete cell: {}".format(path))
        for method in ("morphflow", "morphany3d", "interp3d"):
            metrics = data["methods"][method]
            if metrics["num_pairs"] != expected_pairs or metrics.get("missing_pairs"):
                raise ValueError("Incomplete method: {} {}".format(path, method))
            for key in ("fid_endpoint_manifold", "fid_endpoint_manifold_mean_per_state", "ppl_full_normalized_reference_endpoints", "pdv_full"):
                if not math.isfinite(metrics["aggregate_across_pairs"][key]["mean"]):
                    raise ValueError("Nonfinite metric {} in {}".format(key, path))
        cells.append({"configuration": path.parent.name, "metrics": data})
        with path.with_suffix(".csv").open() as f:
            rows.extend(dict(row, configuration=path.parent.name) for row in csv.DictReader(f))
    if args.require_complete and len(cells) != expected_cells:
        raise ValueError("Expected {} CFG cells, found {}".format(expected_cells, len(cells)))
    write_json(root / "cfg_grid_metrics.json", {"completed_configurations": len(cells), "expected_configurations": expected_cells, "cells": cells})
    if rows:
        temporary = root / ".cfg_grid_metrics.csv.tmp"
        with temporary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["configuration"] + [k for k in rows[0] if k != "configuration"])
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(root / "cfg_grid_metrics.csv")
    print("Completed CFG cells: {}/{}".format(len(cells), expected_cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "cell", "summarize", "count", "grid"))
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--cfg")
    parser.add_argument("--slat-cfg")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.command in {"count", "grid"}:
        manifest = json.loads((Path(args.run_root) / "run_manifest.json").read_text())
        if args.command == "count":
            print(manifest["evaluated_pairs"])
        else:
            for ss in manifest["configuration"]["CFG_VALUES"]:
                for slat in manifest["configuration"]["SLAT_CFG_VALUES"]:
                    print(ss, slat)
    else:
        {"prepare": prepare, "cell": cell, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
