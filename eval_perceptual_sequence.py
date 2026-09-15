#!/usr/bin/env python3
"""Perceptual metrics for complete MorphFlow 3D morphing sequences.

Supports both MorphFlow alpha-sweep layouts::

    # single pair (generate_alpha_steps.py)
    <run_dir>/src1.glb
    <run_dir>/src2.glb
    <run_dir>/summary.json
    <run_dir>/alpha_*/pred_final.glb

    # validation/test batch (generate_alpha_steps_validation.py)
    <run_dir>/pair_0000_.../src1.glb
    <run_dir>/pair_0000_.../src2.glb
    <run_dir>/pair_0000_.../sequence.json
    <run_dir>/pair_0000_.../alpha_*/pred_final.glb
    <run_dir>/pair_0001_.../...

For batch runs, metrics are computed independently for every pair and then
aggregated with equal weight across pairs.

The script renders every generated morph state from the SAME camera views and
computes perceptual sequence metrics used in image/textured-3D morphing work:

* adjacent LPIPS mean/std/max            (Interp3D-style local smoothness)
* PPL sum                                (DiffMorpher / 3D morphing definition)
* normalized PPL = path length/endpoints (Interp3D adaptation)
* PDV = variance of adjacent LPIPS       (DiffMorpher / MorphAny3D)

It also reports endpoint fidelity and alpha-speed-normalized diagnostics. The
latter are useful diagnostics, not canonical paper metrics.

Lower is better for all metrics reported by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
# When copied into the MorphFlow repository this resolves to its existing
# batch renderer. Override with --render_script if running elsewhere.
DEFAULT_RENDER_SCRIPT = PROJECT_DIR / "dataset_toolkits/blender_script/render_batch.py"


@dataclass(frozen=True)
class Frame:
    index: int
    alpha: float
    mesh: Path
    key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a MorphFlow alpha sweep and compute LPIPS/PPL/PDV over the complete sequence."
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help=(
            "Either one alpha-sweep directory, or a validation/test batch root "
            "containing pair_* subdirectories."
        ),
    )
    parser.add_argument("--output_dir", default=None, help="Defaults to <run_dir>/perceptual_sequence")
    parser.add_argument("--stage", choices=["all", "prepare", "metrics"], default="all")
    parser.add_argument(
        "--pair_glob",
        default="pair_*",
        help="Glob used to discover pair directories in batch mode (default: pair_*).",
    )
    parser.add_argument(
        "--skip_invalid_pairs",
        action="store_true",
        help="In batch mode, skip pairs that cannot provide at least two valid generated frames.",
    )

    parser.add_argument("--blender_bin", default=os.environ.get("BLENDER_BIN", "blender"))
    parser.add_argument("--render_script", default=str(DEFAULT_RENDER_SCRIPT))
    parser.add_argument("--num_views", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--pitch_degrees", type=float, default=15.0)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--fov_degrees", type=float, default=40.0)
    parser.add_argument("--object_up_axis", choices=["X", "Y", "Z"], default="Y")
    parser.add_argument("--render_engine", default="CYCLES")
    parser.add_argument(
        "--appearance",
        choices=["materials", "geometry"],
        default="materials",
        help="Use GLB materials (recommended for textured morphing) or a shared neutral geometry material.",
    )
    parser.add_argument("--render_batch_size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--device", default="auto", help="LPIPS device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--lpips_net", choices=["alex", "vgg", "squeeze"], default="vgg")
    parser.add_argument("--lpips_batch_size", type=int, default=8)
    parser.add_argument(
        "--strict_uniform_alphas",
        action="store_true",
        help="Fail if alpha spacing is not uniform. Recommended for benchmark comparisons.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def alpha_key(alpha: float) -> str:
    return f"a{alpha:.6f}".replace("-", "m").replace(".", "p")


def _resolve_recorded_mesh(run_dir: Path, mesh_value: Any) -> Path | None:
    """Resolve a path stored in JSON, including runs moved between machines."""
    if not mesh_value:
        return None
    mesh = Path(str(mesh_value)).expanduser()
    if not mesh.is_absolute():
        candidate = run_dir / mesh
        if candidate.is_file():
            return candidate.resolve()
    elif mesh.is_file():
        return mesh.resolve()

    # Absolute paths in summary/sequence JSON often point to the machine that
    # generated the run. Recover from the alpha directory basename if possible.
    parts = mesh.parts
    for i, part in enumerate(parts):
        if part.startswith("alpha_"):
            candidate = run_dir / part / "pred_final.glb"
            if candidate.is_file():
                return candidate.resolve()
    return None


def discover_sequence_dirs(run_dir: Path, pair_glob: str) -> List[Path]:
    """Return one or more directories that each represent a complete morph pair."""
    if (run_dir / "src1.glb").is_file() and (run_dir / "src2.glb").is_file():
        return [run_dir]

    pair_dirs: List[Path] = []
    for path in sorted(run_dir.glob(pair_glob)):
        if not path.is_dir():
            continue
        # The validation generator writes src1/src2 per pair. sequence.json or
        # alpha_* metadata are accepted as evidence that this is a sequence dir.
        has_sources = (path / "src1.glb").is_file() and (path / "src2.glb").is_file()
        has_sequence = (path / "sequence.json").is_file() or any(path.glob("alpha_*/metadata.json"))
        if has_sources and has_sequence:
            pair_dirs.append(path.resolve())

    if pair_dirs:
        return pair_dirs

    raise FileNotFoundError(
        f"Could not find a single sequence or any {pair_glob!r} sequence directories in {run_dir}. "
        "Expected either <run_dir>/src1.glb + src2.glb, or "
        "<run_dir>/pair_*/src1.glb + src2.glb + alpha_*/pred_final.glb."
    )


def discover_frames(run_dir: Path) -> Tuple[Path, Path, List[Frame], Dict[str, Any]]:
    src1 = run_dir / "src1.glb"
    src2 = run_dir / "src2.glb"
    if not src1.is_file() or not src2.is_file():
        raise FileNotFoundError(f"Expected src1.glb and src2.glb in sequence directory {run_dir}")

    summary_path = run_dir / "summary.json"
    summary: Dict[str, Any] = read_json(summary_path) if summary_path.is_file() else {}

    candidates: List[Tuple[float, Path]] = []

    # 1) Batch generator's sequence.json is the most explicit description.
    sequence_path = run_dir / "sequence.json"
    if sequence_path.is_file():
        sequence = read_json(sequence_path)
        if isinstance(sequence, list):
            for row in sequence:
                if not isinstance(row, Mapping) or row.get("kind") != "prediction":
                    continue
                if "alpha" not in row or row.get("saved", True) is False:
                    continue
                mesh = _resolve_recorded_mesh(run_dir, row.get("path"))
                if mesh is not None:
                    candidates.append((float(row["alpha"]), mesh))

    # 2) summary.json: single-pair generator uses outputs; batch pair summary uses steps.
    if not candidates and isinstance(summary, dict):
        rows = summary.get("outputs")
        if not isinstance(rows, list):
            rows = summary.get("steps")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, Mapping) or "alpha" not in row:
                    continue
                if row.get("pred_final_saved", True) is False:
                    continue
                alpha = float(row["alpha"])
                mesh = _resolve_recorded_mesh(run_dir, row.get("pred_final"))
                if mesh is None:
                    # Both generators have existed with 4- and 6-decimal alpha folders.
                    names = [
                        f"alpha_{alpha:.6f}".replace(".", "p"),
                        f"alpha_{alpha:.4f}".replace(".", "p"),
                    ]
                    for step_name in names:
                        candidate = run_dir / step_name / "pred_final.glb"
                        if candidate.is_file():
                            mesh = candidate.resolve()
                            break
                if mesh is not None:
                    candidates.append((alpha, mesh))

    # 3) Filesystem fallback: authoritative after moving/copying a run.
    if not candidates:
        for metadata_path in sorted(run_dir.glob("alpha_*/metadata.json")):
            row = read_json(metadata_path)
            if not isinstance(row, Mapping) or "alpha" not in row:
                continue
            if row.get("pred_final_saved", True) is False:
                continue
            mesh = metadata_path.parent / "pred_final.glb"
            if mesh.is_file():
                candidates.append((float(row["alpha"]), mesh.resolve()))

    # Deduplicate alpha/path records that can be present in more than one manifest.
    unique: Dict[Tuple[float, str], Tuple[float, Path]] = {}
    for alpha, mesh in candidates:
        unique[(float(alpha), str(mesh))] = (float(alpha), mesh)
    candidates = list(unique.values())

    if len(candidates) < 2:
        raise RuntimeError(
            f"Need at least two valid pred_final.glb frames in {run_dir}; found {len(candidates)}."
        )

    # MorphFlow convention: alpha=1 -> src1, alpha=0 -> src2.
    candidates.sort(key=lambda item: item[0], reverse=True)
    frames = [
        Frame(index=i, alpha=alpha, mesh=mesh, key=f"frame_{i:03d}_{alpha_key(alpha)}")
        for i, (alpha, mesh) in enumerate(candidates)
    ]
    return src1.resolve(), src2.resolve(), frames, summary


def alpha_diagnostics(frames: Sequence[Frame]) -> Dict[str, Any]:
    alphas = np.asarray([f.alpha for f in frames], dtype=np.float64)
    deltas = np.abs(np.diff(alphas))
    if len(deltas) == 0:
        return {"uniform": True, "deltas": []}
    mean = float(deltas.mean())
    uniform = bool(np.allclose(deltas, mean, rtol=1e-4, atol=1e-6))
    return {
        "uniform": uniform,
        "deltas": [float(x) for x in deltas],
        "mean_delta": mean,
        "min_delta": float(deltas.min()),
        "max_delta": float(deltas.max()),
    }


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


def resolve_executable(value: str) -> str:
    expanded = Path(value).expanduser()
    if expanded.is_file():
        return str(expanded.resolve())
    found = shutil.which(value)
    if found:
        return found
    raise FileNotFoundError(f"Blender executable not found: {value!r}")


def composite_on_white(source: Path, destination: Path) -> None:
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        Image.alpha_composite(background, rgba).convert("RGB").save(destination, "PNG")


def render_records(
    args: argparse.Namespace,
    output_dir: Path,
    src1: Path,
    src2: Path,
    frames: Sequence[Frame],
) -> None:
    blender = resolve_executable(args.blender_bin)
    render_script = Path(args.render_script).expanduser().resolve()
    if not render_script.is_file():
        raise FileNotFoundError(
            f"Render script not found: {render_script}. "
            "If this file is outside MorphFlow, pass --render_script <MorphFlow>/dataset_toolkits/blender_script/render_batch.py"
        )
    if args.render_batch_size < 1:
        raise ValueError("--render_batch_size must be >= 1")

    records: List[Tuple[str, Path]] = [("src1", src1)]
    records.extend((frame.key, frame.mesh) for frame in frames)
    records.append(("src2", src2))

    jobs: List[Dict[str, str]] = []
    for key, mesh in records:
        raw_dir = output_dir / "renders" / key
        expected = [raw_dir / f"{view:03d}.png" for view in range(args.num_views)]
        if args.overwrite or not all(path.is_file() for path in expected):
            jobs.append({"object": str(mesh), "output_folder": str(raw_dir.resolve())})

    jobs_dir = output_dir / "render_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    views_json = json.dumps(uniform_views(args), separators=(",", ":"))

    for start in range(0, len(jobs), args.render_batch_size):
        chunk = jobs[start : start + args.render_batch_size]
        batch_id = start // args.render_batch_size
        manifest = jobs_dir / f"jobs_{batch_id:04d}.json"
        write_json(manifest, chunk)
        command = [
            blender,
            "-b",
            "--python-exit-code",
            "1",
            "--python",
            str(render_script),
            "--",
            "--manifest",
            str(manifest),
            "--views",
            views_json,
            "--resolution",
            str(args.resolution),
            "--engine",
            args.render_engine,
            "--object_up_axis",
            args.object_up_axis,
        ]
        if args.appearance == "geometry":
            command.append("--geo_mode")
        subprocess.run(command, check=True)

    # Convert RGBA renders to consistent RGB-on-white images used by LPIPS.
    image_root = output_dir / "images"
    for key, _mesh in records:
        for view in range(args.num_views):
            source = output_dir / "renders" / key / f"{view:03d}.png"
            destination = image_root / key / f"{view:03d}.png"
            if not source.is_file():
                raise FileNotFoundError(f"Expected Blender render not found: {source}")
            if args.overwrite or not destination.is_file():
                composite_on_white(source, destination)


def resolve_device(value: str):
    import torch

    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lpips_image(path: Path):
    import torch
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor.mul(2.0).sub(1.0)


def batched_lpips(
    model,
    pairs: Sequence[Tuple[Path, Path]],
    device,
    batch_size: int,
) -> List[float]:
    import torch

    if batch_size < 1:
        raise ValueError("--lpips_batch_size must be >= 1")
    out: List[float] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            a = torch.stack([load_lpips_image(x) for x, _ in chunk]).to(device)
            b = torch.stack([load_lpips_image(y) for _, y in chunk]).to(device)
            scores = model(a, b).reshape(-1).detach().float().cpu().numpy()
            out.extend(float(x) for x in scores)
    return out


def image_path(output_dir: Path, key: str, view: int) -> Path:
    path = output_dir / "images" / key / f"{view:03d}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Rendered image missing: {path}. Run --stage prepare first.")
    return path


def safe_ratio(num: float, den: float) -> float:
    if den <= 1e-12:
        return float("nan")
    return float(num / den)


def compute_metrics(
    args: argparse.Namespace,
    output_dir: Path,
    frames: Sequence[Frame],
    model=None,
    device=None,
) -> Dict[str, Any]:
    if model is None or device is None:
        import lpips
        device = resolve_device(args.device)
        model = lpips.LPIPS(net=args.lpips_net, spatial=False).to(device).eval()

    # Build all LPIPS pairs once, then run them in batches.
    pair_specs: List[Tuple[str, int, int, float, float, Path, Path]] = []
    # kind, view, transition_idx, alpha_from, alpha_to, image_a, image_b
    for view in range(args.num_views):
        for i in range(len(frames) - 1):
            a, b = frames[i], frames[i + 1]
            pair_specs.append(
                (
                    "adjacent",
                    view,
                    i,
                    a.alpha,
                    b.alpha,
                    image_path(output_dir, a.key, view),
                    image_path(output_dir, b.key, view),
                )
            )
        pair_specs.append(
            (
                "generated_endpoints",
                view,
                -1,
                frames[0].alpha,
                frames[-1].alpha,
                image_path(output_dir, frames[0].key, view),
                image_path(output_dir, frames[-1].key, view),
            )
        )
        pair_specs.append(
            (
                "reference_endpoints",
                view,
                -1,
                1.0,
                0.0,
                image_path(output_dir, "src1", view),
                image_path(output_dir, "src2", view),
            )
        )
        pair_specs.append(
            (
                "src1_fidelity",
                view,
                -1,
                1.0,
                frames[0].alpha,
                image_path(output_dir, "src1", view),
                image_path(output_dir, frames[0].key, view),
            )
        )
        pair_specs.append(
            (
                "src2_fidelity",
                view,
                -1,
                frames[-1].alpha,
                0.0,
                image_path(output_dir, frames[-1].key, view),
                image_path(output_dir, "src2", view),
            )
        )

    values = batched_lpips(
        model,
        [(spec[-2], spec[-1]) for spec in pair_specs],
        device,
        args.lpips_batch_size,
    )

    transition_rows: List[Dict[str, Any]] = []
    per_view: List[Dict[str, Any]] = []
    grouped: Dict[int, Dict[str, Any]] = {v: {} for v in range(args.num_views)}
    for spec, value in zip(pair_specs, values):
        kind, view, transition_idx, alpha_from, alpha_to, _image_a, _image_b = spec
        if kind == "adjacent":
            delta_alpha = abs(alpha_from - alpha_to)
            transition_rows.append(
                {
                    "view": view,
                    "transition": transition_idx,
                    "alpha_from": alpha_from,
                    "alpha_to": alpha_to,
                    "delta_alpha": delta_alpha,
                    "lpips": value,
                    # Diagnostic that makes nonuniform alpha schedules easier to inspect.
                    "lpips_per_alpha": safe_ratio(value, delta_alpha),
                }
            )
            grouped[view].setdefault("adjacent", []).append(value)
            grouped[view].setdefault("adjacent_per_alpha", []).append(safe_ratio(value, delta_alpha))
        else:
            grouped[view][kind] = value

    for view in range(args.num_views):
        g = grouped[view]
        adjacent = np.asarray(g["adjacent"], dtype=np.float64)
        adjacent_per_alpha = np.asarray(g["adjacent_per_alpha"], dtype=np.float64)
        path_sum = float(adjacent.sum())
        generated_endpoint = float(g["generated_endpoints"])
        reference_endpoint = float(g["reference_endpoints"])
        src1_fidelity = float(g["src1_fidelity"])
        src2_fidelity = float(g["src2_fidelity"])

        # Canonical-style metrics.
        row = {
            "view": view,
            "lpips_adjacent_mean": float(adjacent.mean()),
            "lpips_adjacent_std": float(adjacent.std(ddof=0)),
            "lpips_adjacent_max": float(adjacent.max()),
            "ppl_sum": path_sum,
            # Interp3D adaptation: cumulative perceptual path length divided by
            # the perceptual distance between trajectory endpoints.
            # Interp3D adaptation: cumulative perceptual path length divided by
            # the perceptual distance between the true source/target endpoint renders.
            "ppl_normalized_reference_endpoints": safe_ratio(path_sum, reference_endpoint),
            # Same normalization using the first/last generated frames. Useful when
            # the sweep does not include alpha=1/0, but not the paper-faithful Interp3D denominator.
            "ppl_normalized_generated_endpoints": safe_ratio(path_sum, generated_endpoint),
            # PDV as introduced in DiffMorpher: variance of consecutive
            # perceptual distances along the sequence.
            "pdv": float(adjacent.var(ddof=0)),
            "generated_endpoint_lpips": generated_endpoint,
            "reference_endpoint_lpips": reference_endpoint,
            "src1_endpoint_fidelity_lpips": src1_fidelity,
            "src2_endpoint_fidelity_lpips": src2_fidelity,
            "endpoint_fidelity_lpips_mean": 0.5 * (src1_fidelity + src2_fidelity),
            # Reference-anchored diagnostic: includes any failure to hit the
            # true source/target endpoints. Not a canonical paper metric.
            "ppl_reference_anchored": safe_ratio(
                src1_fidelity + path_sum + src2_fidelity,
                reference_endpoint,
            ),
            # Diagnostics for nonuniform alpha schedules.
            "lpips_per_alpha_mean": float(np.nanmean(adjacent_per_alpha)),
            "pdv_per_alpha": float(np.nanvar(adjacent_per_alpha, ddof=0)),
        }
        per_view.append(row)

    def aggregate_field(field: str) -> Dict[str, float]:
        x = np.asarray([float(row[field]) for row in per_view], dtype=np.float64)
        finite = x[np.isfinite(x)]
        if len(finite) == 0:
            return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
        return {
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=0)),
            "median": float(np.median(finite)),
        }

    metric_fields = [field for field in per_view[0] if field != "view"]
    aggregate = {field: aggregate_field(field) for field in metric_fields}

    with (output_dir / "transitions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transition_rows[0].keys()))
        writer.writeheader()
        writer.writerows(transition_rows)

    with (output_dir / "per_view.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_view[0].keys()))
        writer.writeheader()
        writer.writerows(per_view)

    result = {
        "metric_direction": "lower_is_better",
        "lpips_backbone": args.lpips_net,
        "device": str(device),
        "num_views": args.num_views,
        "num_generated_frames": len(frames),
        "sequence_order": "src1 -> src2 (alpha descending)",
        "alphas": [frame.alpha for frame in frames],
        "definitions": {
            "lpips_adjacent_mean": "mean LPIPS over consecutive generated morph frames",
            "ppl_sum": "sum of LPIPS over consecutive generated morph frames",
            "ppl_normalized_reference_endpoints": "ppl_sum / LPIPS(src1, src2), matching the Interp3D source-target normalization",
            "ppl_normalized_generated_endpoints": "ppl_sum / LPIPS(first generated frame, last generated frame); diagnostic",
            "pdv": "population variance of consecutive LPIPS distances",
            "endpoint_fidelity_lpips_mean": "mean LPIPS between generated trajectory endpoints and true src1/src2 renders",
            "ppl_reference_anchored": "(src1 endpoint error + path + src2 endpoint error) / LPIPS(src1, src2); diagnostic",
            "lpips_per_alpha_mean": "mean adjacent LPIPS divided by alpha step; diagnostic",
            "pdv_per_alpha": "variance of LPIPS/alpha-step; diagnostic",
        },
        "aggregate_across_views": aggregate,
    }
    write_json(output_dir / "metrics.json", result)
    return result


def print_summary(result: Mapping[str, Any]) -> None:
    agg = result["aggregate_across_views"]
    keys = [
        "lpips_adjacent_mean",
        "ppl_sum",
        "ppl_normalized_reference_endpoints",
        "pdv",
        "endpoint_fidelity_lpips_mean",
        "ppl_reference_anchored",
    ]
    print("\n===== PERCEPTUAL SEQUENCE METRICS =====")
    print(f"frames: {result['num_generated_frames']} | views: {result['num_views']} | LPIPS: {result['lpips_backbone']}")
    for key in keys:
        stats = agg[key]
        print(f"{key:38s} {stats['mean']:.6f} +/- {stats['std']:.6f}")
    print("=======================================")


def _aggregate_pair_results(pair_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not pair_results:
        return {}

    metric_names = list(pair_results[0]["aggregate_across_views"].keys())
    out: Dict[str, Any] = {}
    for metric in metric_names:
        values = np.asarray(
            [float(result["aggregate_across_views"][metric]["mean"]) for result in pair_results],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            out[metric] = {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
        else:
            out[metric] = {
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=0)),
                "median": float(np.median(finite)),
            }
    return out


def _write_batch_csv(path: Path, pair_rows: Sequence[Mapping[str, Any]]) -> None:
    if not pair_rows:
        return
    fields = list(pair_rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_rows)


def process_sequence(
    args: argparse.Namespace,
    sequence_dir: Path,
    output_dir: Path,
    lpips_model=None,
    lpips_device=None,
) -> Dict[str, Any] | None:
    src1, src2, frames, source_summary = discover_frames(sequence_dir)
    alpha_info = alpha_diagnostics(frames)
    if args.strict_uniform_alphas and not alpha_info["uniform"]:
        raise ValueError(
            f"Alpha schedule is not uniform in {sequence_dir.name}. "
            f"Observed deltas: {alpha_info['deltas']}"
        )
    if not alpha_info["uniform"]:
        print(
            f"WARNING [{sequence_dir.name}]: nonuniform alpha spacing detected. "
            "Canonical PPL/PDV are schedule-dependent; compare methods only with identical alpha schedules."
        )

    manifest = {
        "sequence_dir": str(sequence_dir),
        "source_summary": source_summary,
        "src1": str(src1),
        "src2": str(src2),
        "frames": [
            {"index": frame.index, "alpha": frame.alpha, "mesh": str(frame.mesh), "key": frame.key}
            for frame in frames
        ],
        "alpha_diagnostics": alpha_info,
        "render_protocol": {
            "num_views": args.num_views,
            "resolution": args.resolution,
            "pitch_degrees": args.pitch_degrees,
            "radius": args.radius,
            "fov_degrees": args.fov_degrees,
            "object_up_axis": args.object_up_axis,
            "render_engine": args.render_engine,
            "appearance": args.appearance,
            "camera_policy": "identical indexed camera views for every morph frame and endpoint",
            "background": "white RGB after alpha compositing",
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    if args.stage in ("all", "prepare"):
        render_records(args, output_dir, src1, src2, frames)
    if args.stage in ("all", "metrics"):
        result = compute_metrics(
            args,
            output_dir,
            frames,
            model=lpips_model,
            device=lpips_device,
        )
        result["sequence_dir"] = str(sequence_dir)
        result["sequence_name"] = sequence_dir.name
        write_json(output_dir / "metrics.json", result)
        return result
    return None


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / "perceptual_sequence"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    sequence_dirs = discover_sequence_dirs(run_dir, args.pair_glob)
    batch_mode = not (len(sequence_dirs) == 1 and sequence_dirs[0] == run_dir)
    mode = "batch" if batch_mode else "single-pair"
    print(f"Detected {mode} layout: {len(sequence_dirs)} sequence(s)")

    lpips_model = None
    lpips_device = None
    if args.stage in ("all", "metrics"):
        import lpips
        lpips_device = resolve_device(args.device)
        print(f"Loading LPIPS/{args.lpips_net} once on {lpips_device}...")
        lpips_model = lpips.LPIPS(net=args.lpips_net, spatial=False).to(lpips_device).eval()

    pair_results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for index, sequence_dir in enumerate(sequence_dirs):
        pair_output = output_root / sequence_dir.name if batch_mode else output_root
        pair_output.mkdir(parents=True, exist_ok=True)
        print(f"\n[{index + 1}/{len(sequence_dirs)}] {sequence_dir.name}")
        try:
            result = process_sequence(
                args,
                sequence_dir,
                pair_output,
                lpips_model=lpips_model,
                lpips_device=lpips_device,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            if not (batch_mode and args.skip_invalid_pairs):
                raise
            print(f"WARNING: skipping {sequence_dir.name}: {exc}")
            skipped.append({"sequence": sequence_dir.name, "reason": str(exc)})
            continue

        if result is not None:
            pair_results.append(result)
            print_summary(result)

    if batch_mode:
        batch_manifest = {
            "run_dir": str(run_dir),
            "pair_glob": args.pair_glob,
            "num_discovered_pairs": len(sequence_dirs),
            "num_completed_pairs": len(pair_results) if args.stage in ("all", "metrics") else len(sequence_dirs) - len(skipped),
            "skipped_pairs": skipped,
            "pair_outputs": [str((output_root / d.name).resolve()) for d in sequence_dirs],
        }
        write_json(output_root / "batch_manifest.json", batch_manifest)

        if pair_results:
            pair_rows: List[Dict[str, Any]] = []
            for result in pair_results:
                row: Dict[str, Any] = {
                    "sequence": result["sequence_name"],
                    "num_generated_frames": result["num_generated_frames"],
                    "num_views": result["num_views"],
                }
                for metric, stats in result["aggregate_across_views"].items():
                    row[metric] = stats["mean"]
                pair_rows.append(row)

            aggregate = _aggregate_pair_results(pair_results)
            batch_result = {
                "metric_direction": "lower_is_better",
                "aggregation": "each morph pair has equal weight; per-pair value is first averaged across camera views",
                "num_pairs": len(pair_results),
                "num_views_per_pair": args.num_views,
                "lpips_backbone": args.lpips_net,
                "pairs": pair_rows,
                "aggregate_across_pairs": aggregate,
                "skipped_pairs": skipped,
            }
            write_json(output_root / "metrics_batch.json", batch_result)
            _write_batch_csv(output_root / "per_pair.csv", pair_rows)

            print("\n===== BATCH PERCEPTUAL SEQUENCE METRICS =====")
            print(f"pairs: {len(pair_results)} | views/pair: {args.num_views} | LPIPS: {args.lpips_net}")
            for key in (
                "lpips_adjacent_mean",
                "ppl_sum",
                "ppl_normalized_reference_endpoints",
                "pdv",
                "endpoint_fidelity_lpips_mean",
                "ppl_reference_anchored",
            ):
                stats = aggregate[key]
                print(f"{key:38s} {stats['mean']:.6f} +/- {stats['std']:.6f}")
            print("==============================================")
            print(f"batch metrics: {output_root / 'metrics_batch.json'}")
            print(f"per-pair CSV:  {output_root / 'per_pair.csv'}")


if __name__ == "__main__":
    main()
