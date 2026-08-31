"""Compute multi-view FID/KID from an existing MorphFlow evaluation folder.

Expected sample layout (the one produced by eval_validation_latents.py):

    <eval_dir>/<sample>/src1.glb
    <eval_dir>/<sample>/src2.glb
    <eval_dir>/<sample>/target.glb
    <eval_dir>/<sample>/pred_final.glb

The image distributions are:

* reference: src1/src2 assets found in the evaluation samples;
* target: MorphAny3D target.glb meshes;
* result: MorphFlow pred_final.glb meshes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RENDER_SCRIPT = PROJECT_DIR / "dataset_toolkits/blender_script/render_batch.py"
REQUIRED_MESHES = ("src1.glb", "src2.glb", "target.glb", "pred_final.glb")
SET_NAMES = ("reference", "target", "result")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an existing MorphFlow eval folder and compute Clean-FID/KID."
    )
    parser.add_argument(
        "--eval_dir",
        required=True,
        help="Evaluation run containing one subdirectory per sample.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Defaults to <eval_dir>/fid_kid.",
    )
    parser.add_argument("--stage", choices=["all", "prepare", "metrics"], default="all")
    parser.add_argument(
        "--reference_mode",
        choices=["unique", "occurrences"],
        default="unique",
        help="Deduplicate source assets by metadata name (or exact GLB content), or keep every occurrence.",
    )
    parser.add_argument(
        "--skip_incomplete",
        action="store_true",
        help="Skip sample folders missing one of the four required GLBs instead of failing.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used by KID subsets.")

    parser.add_argument(
        "--blender_bin",
        default=os.environ.get("BLENDER_BIN", "blender"),
        help="Blender executable or absolute path.",
    )
    parser.add_argument("--render_script", default=str(DEFAULT_RENDER_SCRIPT))
    parser.add_argument("--num_views", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--pitch_degrees", type=float, default=15.0)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--fov_degrees", type=float, default=40.0)
    parser.add_argument("--render_engine", default="CYCLES")
    parser.add_argument("--render_batch_size", type=int, default=64)
    parser.add_argument(
        "--use_materials",
        action="store_true",
        help="Use GLB materials. By default every set uses the same neutral geometry material.",
    )
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--metrics_device", default="auto")
    parser.add_argument("--fid_batch_size", type=int, default=32)
    parser.add_argument("--fid_num_workers", type=int, default=4)
    parser.add_argument("--kid_subsets", type=int, default=100)
    parser.add_argument("--kid_subset_size", type=int, default=1000)
    return parser.parse_args()


def safe_slug(value: Any, max_len: int = 64) -> str:
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value))
    return value[:max_len].strip("_") or "sample"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_sample_metadata(sample_dir: Path) -> Dict[str, Any]:
    path = sample_dir / "metrics.json"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def discover_sample_dirs(eval_dir: Path, skip_incomplete: bool) -> List[Path]:
    candidates = {
        path.parent
        for mesh_name in REQUIRED_MESHES
        for path in eval_dir.rglob(mesh_name)
    }
    if not candidates:
        raise RuntimeError(f"No evaluation samples containing {REQUIRED_MESHES} found in {eval_dir}")

    complete: List[Path] = []
    incomplete = []
    for sample_dir in sorted(candidates, key=lambda path: str(path.relative_to(eval_dir))):
        missing = [name for name in REQUIRED_MESHES if not (sample_dir / name).is_file()]
        if missing:
            incomplete.append((sample_dir, missing))
        else:
            complete.append(sample_dir)

    if incomplete and not skip_incomplete:
        details = "\n".join(
            f"  {path}: missing {', '.join(missing)}" for path, missing in incomplete[:20]
        )
        raise RuntimeError(
            f"Found {len(incomplete)} incomplete evaluation sample(s):\n{details}\n"
            "Fix them or pass --skip_incomplete explicitly."
        )
    if incomplete:
        print(f"WARNING: skipped {len(incomplete)} incomplete evaluation sample(s).")
    if not complete:
        raise RuntimeError("No complete evaluation samples remain.")
    return complete


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def reference_identity(role: str, mesh_path: Path, metadata: Mapping[str, Any]) -> str:
    name = metadata.get(role)
    if isinstance(name, str) and name.strip():
        return f"asset:{name.strip()}"
    return f"sha256:{sha256(mesh_path)}"


def build_manifest(
    eval_dir: Path,
    sample_dirs: Sequence[Path],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    references: List[Dict[str, Any]] = []
    morphs: List[Dict[str, Any]] = []
    seen_references = set()

    for sample_id, sample_dir in enumerate(sample_dirs):
        metadata = read_sample_metadata(sample_dir)
        sample_key = f"morph_{sample_id:06d}_{safe_slug(sample_dir.name, 48)}"
        morphs.append(
            {
                "sample_id": sample_id,
                "sample_key": sample_key,
                "sample_dir": str(sample_dir.resolve()),
                "src1": metadata.get("src1"),
                "src2": metadata.get("src2"),
                "target": metadata.get("target"),
                "alpha": metadata.get("alpha"),
                "dataset_index": metadata.get("dataset_idx"),
                "target_mesh": str((sample_dir / "target.glb").resolve()),
                "result_mesh": str((sample_dir / "pred_final.glb").resolve()),
            }
        )

        for role in ("src1", "src2"):
            mesh_path = sample_dir / f"{role}.glb"
            identity = reference_identity(role, mesh_path, metadata)
            if args.reference_mode == "unique" and identity in seen_references:
                continue
            seen_references.add(identity)
            reference_id = len(references)
            asset = metadata.get(role)
            references.append(
                {
                    "reference_id": reference_id,
                    "sample_key": f"reference_{reference_id:06d}_{safe_slug(asset or role, 48)}",
                    "asset": asset,
                    "source_role": role,
                    "source_sample_id": sample_id,
                    "identity": identity,
                    "mesh": str(mesh_path.resolve()),
                }
            )

    render_protocol = {
        "num_views": args.num_views,
        "resolution": args.resolution,
        "yaw": "uniform over [0, 2pi)",
        "pitch_degrees": args.pitch_degrees,
        "radius": args.radius,
        "fov_degrees": args.fov_degrees,
        "engine": args.render_engine,
        "normalization": "each mesh centered and scaled to a unit cube",
        "background": "white RGB after alpha compositing",
        "appearance": "GLB materials" if args.use_materials else "shared neutral geometry material",
    }
    source_artifacts = [
        artifact_identity(sample_dir / mesh_name)
        for sample_dir in sample_dirs
        for mesh_name in REQUIRED_MESHES
    ]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "selected",
        "source_eval_dir": str(eval_dir),
        "source_artifacts": source_artifacts,
        "protocol": {
            "reference": (
                "unique original src1/src2 assets in the eval folder"
                if args.reference_mode == "unique"
                else "every src1/src2 occurrence in the eval folder"
            ),
            "target": "every target.glb in the eval folder",
            "result": "every pred_final.glb in the eval folder",
            "comparisons": {
                "result_vs_reference": "student plausibility",
                "target_vs_reference": "MorphAny3D teacher plausibility",
                "result_vs_target": "student-teacher distribution similarity",
            },
            "render": render_protocol,
        },
        "selection": {
            "reference_mode": args.reference_mode,
            "num_samples": len(morphs),
            "num_references": len(references),
            "skip_incomplete": bool(args.skip_incomplete),
        },
        "references": references,
        "morphs": morphs,
    }


def assert_resume_is_compatible(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    path = output_dir / "manifest.json"
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        previous = json.load(handle)
    keys = ("source_eval_dir", "source_artifacts", "protocol", "selection")
    changed = [key for key in keys if previous.get(key) != manifest.get(key)]
    if changed:
        raise RuntimeError(
            "The output directory belongs to an incompatible evaluation "
            f"(changed: {', '.join(changed)}). Use another --output_dir or pass --overwrite."
        )


def write_samples_csv(path: Path, manifest: Mapping[str, Any]) -> None:
    fields = (
        "set", "sample_key", "mesh", "sample_dir", "asset", "source_role",
        "src1", "src2", "target", "alpha", "dataset_index",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for reference in manifest["references"]:
            writer.writerow(
                {
                    "set": "reference",
                    "sample_key": reference["sample_key"],
                    "mesh": reference["mesh"],
                    "asset": reference.get("asset"),
                    "source_role": reference["source_role"],
                }
            )
        for morph in manifest["morphs"]:
            common = {
                key: morph.get(key)
                for key in (
                    "sample_key", "sample_dir", "src1", "src2", "target",
                    "alpha", "dataset_index",
                )
            }
            writer.writerow({**common, "set": "target", "mesh": morph["target_mesh"]})
            writer.writerow({**common, "set": "result", "mesh": morph["result_mesh"]})


def uniform_views(args: argparse.Namespace) -> List[Dict[str, float]]:
    if args.num_views < 1 or args.resolution < 1:
        raise ValueError("--num_views and --resolution must be >= 1")
    if args.radius <= 0:
        raise ValueError("--radius must be > 0")
    if not 0 < args.fov_degrees < 180:
        raise ValueError("--fov_degrees must be between 0 and 180")
    return [
        {
            "yaw": 2.0 * math.pi * index / args.num_views,
            "pitch": math.radians(args.pitch_degrees),
            "radius": args.radius,
            "fov": math.radians(args.fov_degrees),
        }
        for index in range(args.num_views)
    ]


def mesh_records(manifest: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    for reference in manifest["references"]:
        yield {
            "set": "reference",
            "sample_key": reference["sample_key"],
            "mesh": reference["mesh"],
        }
    for morph in manifest["morphs"]:
        yield {"set": "target", "sample_key": morph["sample_key"], "mesh": morph["target_mesh"]}
        yield {"set": "result", "sample_key": morph["sample_key"], "mesh": morph["result_mesh"]}


def resolve_executable(value: str) -> str:
    expanded = Path(value).expanduser()
    if expanded.is_file():
        return str(expanded.resolve())
    found = shutil.which(value)
    if found:
        return found
    raise FileNotFoundError(
        f"Blender executable not found: {value!r}. Pass --blender_bin or set BLENDER_BIN."
    )


def render_meshes(args: argparse.Namespace, output_dir: Path, manifest: Dict[str, Any]) -> None:
    blender = resolve_executable(args.blender_bin)
    render_script = Path(args.render_script).expanduser().resolve()
    if not render_script.is_file():
        raise FileNotFoundError(f"Blender batch render script not found: {render_script}")
    if args.render_batch_size < 1:
        raise ValueError("--render_batch_size must be >= 1")

    jobs: List[Dict[str, str]] = []
    for record in mesh_records(manifest):
        render_dir = output_dir / "renders" / record["set"] / record["sample_key"]
        expected = [render_dir / f"{index:03d}.png" for index in range(args.num_views)]
        if args.overwrite or not all(path.is_file() for path in expected):
            jobs.append(
                {
                    "object": str(Path(record["mesh"]).resolve()),
                    "output_folder": str(render_dir.resolve()),
                }
            )

    views = json.dumps(uniform_views(args), separators=(",", ":"))
    jobs_dir = output_dir / "render_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(jobs), args.render_batch_size):
        chunk = jobs[start : start + args.render_batch_size]
        batch_id = start // args.render_batch_size
        jobs_path = jobs_dir / f"jobs_{batch_id:04d}.json"
        write_json(jobs_path, chunk)
        command = [
            blender, "-b", "--python", str(render_script), "--",
            "--manifest", str(jobs_path), "--views", views,
            "--resolution", str(args.resolution), "--engine", args.render_engine,
        ]
        if not args.use_materials:
            command.append("--geo_mode")
        print(
            f"[render {batch_id + 1}/{math.ceil(len(jobs) / args.render_batch_size)}] "
            f"meshes={len(chunk)}",
            flush=True,
        )
        subprocess.run(command, check=True)
    if not jobs:
        print("All raw renders already exist; skipping Blender.")

    flatten_renders(args, output_dir, manifest)
    manifest["status"] = "images_ready"
    write_json(output_dir / "manifest.json", manifest)


def composite_on_white(source: Path, destination: Path) -> None:
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        Image.alpha_composite(background, rgba).convert("RGB").save(destination, "PNG")


def expected_image_names(manifest: Mapping[str, Any], set_name: str) -> List[str]:
    num_views = int(manifest["protocol"]["render"]["num_views"])
    records = manifest["references"] if set_name == "reference" else manifest["morphs"]
    return [
        f"{record['sample_key']}_v{view_index:03d}.png"
        for record in records
        for view_index in range(num_views)
    ]


def flatten_renders(args: argparse.Namespace, output_dir: Path, manifest: Mapping[str, Any]) -> None:
    for set_name in SET_NAMES:
        image_dir = output_dir / "images" / set_name
        image_dir.mkdir(parents=True, exist_ok=True)
        expected_names = set(expected_image_names(manifest, set_name))
        for stale_path in image_dir.glob("*.png"):
            if stale_path.name not in expected_names:
                stale_path.unlink()

        records = manifest["references"] if set_name == "reference" else manifest["morphs"]
        for record in records:
            sample_key = record["sample_key"]
            raw_dir = output_dir / "renders" / set_name / sample_key
            for view_index in range(args.num_views):
                source = raw_dir / f"{view_index:03d}.png"
                destination = image_dir / f"{sample_key}_v{view_index:03d}.png"
                if not source.is_file():
                    raise FileNotFoundError(f"Expected Blender render not found: {source}")
                if args.overwrite or not destination.is_file():
                    composite_on_white(source, destination)


def load_manifest(output_dir: Path) -> Dict[str, Any]:
    path = output_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}. Run --stage prepare first.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_image_sets(output_dir: Path, manifest: Mapping[str, Any]) -> Dict[str, List[Path]]:
    sets: Dict[str, List[Path]] = {}
    for set_name in SET_NAMES:
        folder = output_dir / "images" / set_name
        expected = set(expected_image_names(manifest, set_name))
        actual = {path.name for path in folder.glob("*.png")} if folder.is_dir() else set()
        if expected != actual:
            raise RuntimeError(
                f"Image set {set_name!r} differs from the manifest: "
                f"missing={len(expected - actual)}, extra={len(actual - expected)}."
            )
        paths = sorted(folder.glob("*.png"))
        if len(paths) < 2:
            raise RuntimeError(f"FID/KID require at least two images in {folder}.")
        sets[set_name] = paths
    return sets


def compute_metrics(args: argparse.Namespace, output_dir: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    from importlib.metadata import version

    import numpy as np
    import torch

    try:
        from cleanfid import fid
    except ImportError as exc:
        raise RuntimeError(
            "Install Clean-FID with `pip install -r requirements-eval.txt`."
        ) from exc

    if args.fid_batch_size < 1 or args.fid_num_workers < 0:
        raise ValueError("Invalid Clean-FID batch size or worker count.")
    if args.kid_subsets < 1 or args.kid_subset_size < 2:
        raise ValueError("--kid_subsets must be >= 1 and --kid_subset_size >= 2.")

    image_sets = validate_image_sets(output_dir, manifest)
    device_name = args.metrics_device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    for set_name, paths in image_sets.items():
        if len(paths) < 2048:
            print(
                f"WARNING: {set_name} has {len(paths)} renders (<2048); "
                "FID is finite-sample biased, so report KID too."
            )

    feature_model = fid.build_feature_extractor("clean", device, use_dataparallel=False)
    features = {}
    for set_name in SET_NAMES:
        folder = output_dir / "images" / set_name
        features[set_name] = fid.get_folder_features(
            str(folder), model=feature_model, num_workers=args.fid_num_workers,
            batch_size=args.fid_batch_size, device=device, mode="clean",
            description=set_name, verbose=True,
        )

    specifications = (
        ("result_vs_reference", "result", "reference", "student plausibility"),
        ("target_vs_reference", "target", "reference", "MorphAny3D teacher plausibility"),
        ("result_vs_target", "result", "target", "student-teacher distribution similarity"),
    )
    rows = []
    comparisons = {}
    for index, (name, set_a, set_b, meaning) in enumerate(specifications):
        np.random.seed(args.seed + index)
        fid_value = float(fid.fid_from_feats(features[set_a], features[set_b]))
        kid_value = float(
            fid.kernel_distance(
                features[set_a], features[set_b],
                num_subsets=args.kid_subsets, max_subset_size=args.kid_subset_size,
            )
        )
        row = {
            "comparison": name, "meaning": meaning, "set_a": set_a, "set_b": set_b,
            "num_images_a": int(features[set_a].shape[0]),
            "num_images_b": int(features[set_b].shape[0]),
            "fid": fid_value, "kid": kid_value, "kid_x1000": kid_value * 1000.0,
        }
        rows.append(row)
        comparisons[name] = row
        print(f"{name}: FID={fid_value:.6f} KID={kid_value:.8f}")

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": "clean-fid",
        "clean_fid_version": version("clean-fid"),
        "mode": "clean",
        "feature_extractor": "InceptionV3",
        "kid_seed": args.seed,
        "kid_subsets": args.kid_subsets,
        "kid_subset_size": args.kid_subset_size,
        "sets": {
            name: {
                "num_objects": len(
                    manifest["references"] if name == "reference" else manifest["morphs"]
                ),
                "num_images": len(image_sets[name]),
            }
            for name in SET_NAMES
        },
        "comparisons": comparisons,
        "plausibility_gap": {
            "fid": comparisons["result_vs_reference"]["fid"]
            - comparisons["target_vs_reference"]["fid"],
            "kid": comparisons["result_vs_reference"]["kid"]
            - comparisons["target_vs_reference"]["kid"],
            "interpretation": "student minus teacher; negative means lower distance to references",
        },
    }
    write_json(output_dir / "metrics.json", result)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else eval_dir / "fid_kid"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("all", "prepare"):
        sample_dirs = discover_sample_dirs(eval_dir, args.skip_incomplete)
        manifest = build_manifest(eval_dir, sample_dirs, args)
        if not args.overwrite:
            assert_resume_is_compatible(output_dir, manifest)
        write_json(output_dir / "manifest.json", manifest)
        write_samples_csv(output_dir / "samples.csv", manifest)
        print(
            f"Discovered {len(manifest['morphs'])} complete samples and "
            f"{len(manifest['references'])} references in {eval_dir}"
        )
        render_meshes(args, output_dir, manifest)
    else:
        manifest = load_manifest(output_dir)

    if args.stage in ("all", "metrics"):
        compute_metrics(args, output_dir, manifest)
        print(f"Metrics written to {output_dir / 'metrics.json'}")
    else:
        print(f"Prepared image sets in {output_dir / 'images'}")


if __name__ == "__main__":
    main()
