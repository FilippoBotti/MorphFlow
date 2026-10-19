#!/usr/bin/env python3
"""Appearance-aware evaluation for the three-way textured-GLB benchmark.

This script wraps MorphFlow's eval_perceptual_sequence.py and adds:
  * full-sequence LPIPS/PPL/PDV including src1/src2 endpoint transitions;
  * both concatenated-state and mean-per-state trajectory FID interpretations;
  * Gaussian rendering with official TRELLIS spherical Hammersley cameras;
  * deterministic merge of shard outputs.

The default backend renders genuine Gaussian appearance decoded from each SLat,
as in official Interp3D. Optional Blender rendering compares final textured GLBs.
The benchmark uses MorphFlow held-out test pairs, rather than Interp3DData.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np

import eval_perceptual_sequence as base
from threeway_textured_benchmark import alpha_dir, load_raw_gaussian, raw_asset_ready


FID_METRICS = {
    "concatenated": "fid_endpoint_manifold",
    "mean-per-state": "fid_endpoint_manifold_mean_per_state",
}
FID_FORMULAS = {
    "fid_endpoint_manifold": "mean_pairs FID(concat(src1_views, src2_views), concat(selected_intermediate_views))",
    "fid_endpoint_manifold_mean_per_state": "mean_pairs mean_selected_states FID(concat(src1_views, src2_views), views_of_one_selected_state)",
    "fid_endpoint_manifold_pooled": "FID(concat_all_pairs(src1_views, src2_views), concat_all_pairs(selected_intermediate_views))",
}


def fid_aggregation_modes(mode: str) -> List[str]:
    if mode == "both":
        return ["concatenated", "mean-per-state"]
    if mode not in FID_METRICS:
        raise ValueError(f"Unknown FID aggregation: {mode}")
    return [mode]


def required_fid_metrics(mode: str) -> List[str]:
    return [FID_METRICS[name] for name in fid_aggregation_modes(mode)]


def completed_fid_metrics(result: Mapping[str, Any], mode: str) -> bool:
    """Do not resume an old, single-interpretation result as a dual-FID result."""
    aggregate = result.get("aggregate_across_views", {})
    protocol = result.get("fid_protocol", {})
    for metric in required_fid_metrics(mode):
        try:
            if not math.isfinite(float(aggregate[metric]["mean"])):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    if "mean-per-state" in fid_aggregation_modes(mode):
        states = protocol.get("selected_states", [])
        if protocol.get("schema_version", 0) < 2 or not states:
            return False
        try:
            if not all(int(row["generated_samples"]) >= 2 and math.isfinite(float(row["fid"])) for row in states):
                return False
            if not np.isclose(np.mean([row["fid"] for row in states]), aggregate[FID_METRICS["mean-per-state"]]["mean"]):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


WEIGHTED_METRICS = (
    "weighted_endpoint_lpips_mean", "weighted_endpoint_lpips_auc",
    "weighted_endpoint_lpips_full_mean", "weighted_endpoint_lpips_full_auc",
)


def completed_weighted_metrics(result: Mapping[str, Any]) -> bool:
    """Require the endpoint-inclusive curve and all explicitly named summaries."""
    try:
        if result.get("metric_mode") != "weighted-endpoints-only" or not result.get("weighted_endpoint_includes_reference_endpoints"):
            return False
        n = int(result["num_generated_frames"])
        curve = result["weighted_endpoint_curve"]
        alphas = np.asarray(result["alphas"], dtype=np.float64)
        if n < 2 or result["num_sequence_frames"] != n + 2 or len(curve) != n + 2 or len(alphas) != n + 2:
            return False
        if not np.isfinite(alphas).all() or not np.all(np.diff(alphas) < 0) or alphas[0] != 1 or alphas[-1] != 0:
            return False
        values = np.asarray([row["weighted_lpips_mean"] for row in curve], dtype=np.float64)
        if not np.isfinite(values).all() or not np.allclose([row["alpha"] for row in curve], alphas):
            return False
        if any(int(row["num_views"]) != result["num_views"] for row in curve) or not np.allclose(values[[0, -1]], 0):
            return False
        expected = (values[1:-1].mean(), np.trapz(values[1:-1][::-1], alphas[1:-1][::-1]),
                    values.mean(), np.trapz(values[::-1], alphas[::-1]))
        for metric, value in zip(WEIGHTED_METRICS, expected):
            stats = result["aggregate_across_views"][metric]
            if not all(math.isfinite(float(stats[field])) for field in ("mean", "std", "median")) or not np.isclose(stats["mean"], value):
                return False
    except (KeyError, TypeError, ValueError, IndexError):
        return False
    return True


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp.replace(path)


def stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
    }


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 1e-12 else float("nan")


def pair_id_from_name(name: str, fallback: int) -> int:
    match = re.match(r"pair_(\d+)", name)
    return int(match.group(1)) if match else int(fallback)


def select_sequence_dirs(run_dir: Path, pair_glob: str, shard_index: int, num_shards: int) -> List[Path]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index out of range")
    raw_dirs = sorted(path for path in run_dir.glob(pair_glob) if path.is_dir() and any(path.glob("alpha_*/mesh_raw.pt")))
    all_dirs = raw_dirs or base.discover_sequence_dirs(run_dir, pair_glob)
    selected: List[Path] = []
    for idx, path in enumerate(all_dirs):
        pair_id = pair_id_from_name(path.name, idx)
        if pair_id % num_shards == shard_index:
            selected.append(path)
    return selected


def discover_raw_frames(sequence_dir: Path) -> Tuple[Path, Path, List[Any], Dict[str, Any]]:
    run_root = sequence_dir.parent.parent
    plan = base.read_json(run_root / "plan.json")
    matches = [pair for pair in plan["pairs"] if pair["pair_dir"] == sequence_dir.name]
    if len(matches) != 1:
        raise ValueError(f"Sequence {sequence_dir.name} does not uniquely match plan.json")
    reference_root = run_root / "references_raw" / sequence_dir.name
    src1, src2 = reference_root / "src1_raw.pt", reference_root / "src2_raw.pt"
    frames = [base.Frame(index=i, alpha=float(a), mesh=sequence_dir / alpha_dir(a) / "mesh_raw.pt", key=base.alpha_key(float(a))) for i, a in enumerate(plan["alphas"])]
    for path in (src1, src2, *(frame.mesh for frame in frames)):
        if not raw_asset_ready(path):
            raise FileNotFoundError(f"Incomplete raw Gaussian asset: {path}")
    return src1, src2, frames, {"plan": plan, "pair": matches[0]}


def cache_identity(path: Path, args: argparse.Namespace, num_views: int) -> str:
    stat = path.stat()
    identity = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "views": num_views, "resolution": args.resolution, "radius": args.radius,
                "fov": args.fov_degrees, "cameras": args.camera_distribution,
                "background": "white", "renderer": "TRELLIS GaussianRenderer kernel_size=0.1 use_mip_gaussian=True", "feature_backbone": "pytorch-fid TensorFlow-compatible pool3 2048D"}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def evaluation_signature(args: argparse.Namespace, src1: Path, src2: Path, frames: Sequence[Any]) -> str:
    payload = {"assets": [cache_identity(path, args, args.num_views) for path in (src1, src2, *(frame.mesh for frame in frames))],
               "alphas": [frame.alpha for frame in frames], "lpips": args.lpips_net,
               "fid_num_views": args.fid_num_views, "fid_states": args.fid_num_intermediate_states,
               "fid_seed": args.fid_selection_seed, "render_backend": args.render_backend,
               "fid_aggregation": getattr(args, "fid_aggregation", "both"),
               "metric_mode": getattr(args, "metric_mode", "interp3d"),
               "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "metrics_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def gaussian_rgb_views(path: Path, args: argparse.Namespace, num_views: int):
    """Stream official TRELLIS spherical views, preserving SLat appearance."""
    import torch
    from trellis.utils import render_utils
    gaussian = load_raw_gaussian(path, "cuda")
    renderer = render_utils.get_renderer(gaussian, resolution=args.resolution, bg_color=(1.0, 1.0, 1.0))
    view_args = argparse.Namespace(**vars(args))
    view_args.num_views = num_views
    views = base.uniform_views(view_args)
    # Blender's yaw convention is rotated by pi/2 relative to TRELLIS's
    # [sin(yaw), cos(yaw), z] camera origins. uniform_views accounts for that.
    yaws = [math.pi / 2 - view["yaw"] for view in views]
    pitches = [view["pitch"] for view in views]
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitches, args.radius, args.fov_degrees)
    with torch.inference_mode():
        for extrinsic, intrinsic in zip(extrinsics, intrinsics):
            rendered = renderer.render(gaussian, extrinsic, intrinsic)["color"]
            yield np.clip(rendered.detach().float().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)


def prepare_gaussian_images(args: argparse.Namespace, pair_output: Path, src1: Path, src2: Path, frames: Sequence[Any]) -> None:
    from PIL import Image
    for key, raw_path in [("src1", src1)] + [(frame.key, frame.mesh) for frame in frames] + [("src2", src2)]:
        destination = pair_output / "images" / key
        identity = cache_identity(raw_path, args, args.num_views)
        if key in {"src1", "src2"} and args.reference_cache_root:
            destination = Path(args.reference_cache_root).resolve() / "images" / identity
        marker = destination / "identity.json"
        complete = marker.is_file() and base.read_json(marker).get("identity") == identity and all((destination / f"{i:03d}.png").is_file() for i in range(args.num_views))
        if not complete or args.overwrite:
            if args.stage == "metrics":
                raise FileNotFoundError(f"Missing matching cached Gaussian renders: {destination}")
            destination.mkdir(parents=True, exist_ok=True)
            for view, array in enumerate(gaussian_rgb_views(raw_path, args, args.num_views)):
                Image.fromarray(array).save(destination / f"{view:03d}.png")
            write_json(marker, {"identity": identity})
        local = pair_output / "images" / key
        if local != destination:
            local.parent.mkdir(parents=True, exist_ok=True)
            if local.is_symlink() and (os.path.isabs(os.readlink(local)) or local.resolve() != destination):
                local.unlink()
            if not local.exists():
                local.symlink_to(os.path.relpath(destination, local.parent), target_is_directory=True)


def gaussian_inception_features(raw_path: Path, args: argparse.Namespace, model: Any, transform: Any, device: Any, cache_root: Path) -> np.ndarray:
    import torch
    from PIL import Image
    identity = cache_identity(raw_path, args, args.fid_num_views)
    path = cache_root / f"{identity}.npz"
    if path.is_file() and not args.overwrite:
        with np.load(path) as payload:
            return np.asarray(payload["features"], dtype=np.float32)
    features, tensors = [], []
    with torch.inference_mode():
        for array in gaussian_rgb_views(raw_path, args, args.fid_num_views):
            tensors.append(transform(Image.fromarray(array)))
            if len(tensors) == args.fid_batch_size:
                out = inception_forward(model, torch.stack(tensors).to(device))
                features.append(out.detach().float().cpu().numpy().reshape(len(tensors), -1))
                tensors = []
        if tensors:
            out = inception_forward(model, torch.stack(tensors).to(device))
            features.append(out.detach().float().cpu().numpy().reshape(len(tensors), -1))
    result = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, features=result)
    temporary.replace(path)
    return result


def full_path_metrics(
    args: argparse.Namespace,
    pair_output: Path,
    frames: Sequence[Any],
    lpips_model: Any,
    lpips_device: Any,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute endpoint-inclusive LPIPS path metrics view by view."""
    keys = ["src1"] + [frame.key for frame in frames] + ["src2"]
    full_transition_rows: List[Dict[str, Any]] = []
    per_view_rows: List[Dict[str, Any]] = []

    for view in range(args.num_views):
        adjacent_pairs = [
            (
                base.image_path(pair_output, keys[i], view),
                base.image_path(pair_output, keys[i + 1], view),
            )
            for i in range(len(keys) - 1)
        ]
        endpoint_pair = [
            (
                base.image_path(pair_output, "src1", view),
                base.image_path(pair_output, "src2", view),
            )
        ]
        adjacent = base.batched_lpips(lpips_model, adjacent_pairs, lpips_device, args.lpips_batch_size)
        endpoint_ref = base.batched_lpips(lpips_model, endpoint_pair, lpips_device, args.lpips_batch_size)[0]

        path_sum = float(np.sum(adjacent))
        path_mean = float(np.mean(adjacent))
        path_var = float(np.var(np.asarray(adjacent, dtype=np.float64), ddof=0))
        path_norm = safe_ratio(path_sum, float(endpoint_ref))

        per_view_rows.append(
            {
                "view": view,
                "lpips_adjacent_full_mean": path_mean,
                "ppl_full_sum": path_sum,
                "ppl_full_normalized_reference_endpoints": path_norm,
                "pdv_full": path_var,
                "lpips_reference_endpoints": float(endpoint_ref),
            }
        )
        for transition_index, value in enumerate(adjacent):
            full_transition_rows.append(
                {
                    "view": view,
                    "transition": transition_index,
                    "from": keys[transition_index],
                    "to": keys[transition_index + 1],
                    "lpips": float(value),
                }
            )

    metric_names = [
        "lpips_adjacent_full_mean",
        "ppl_full_sum",
        "ppl_full_normalized_reference_endpoints",
        "pdv_full",
        "lpips_reference_endpoints",
    ]
    aggregate = {name: stats([float(row[name]) for row in per_view_rows]) for name in metric_names}
    return aggregate, full_transition_rows, per_view_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def inception_cache_path() -> Path:
    import torch
    from pytorch_fid.inception import FID_WEIGHTS_URL
    return Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(FID_WEIGHTS_URL).path).name


def load_inception(device: Any):
    from pytorch_fid.inception import InceptionV3
    from torchvision.transforms import ToTensor
    cache = inception_cache_path()
    if not cache.is_file():
        raise FileNotFoundError(f"Missing canonical FID Inception weights: {cache}; preload pytorch-fid weights into TORCH_HOME before the offline job")
    # pytorch-fid's network matches the TensorFlow FID weights and pooling;
    # torchvision ImageNet Inception is a different network/metric.
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True, normalize_input=True).eval().to(device)
    return model, ToTensor(), cache


def inception_forward(model: Any, batch: Any) -> Any:
    output = model(batch)
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.reshape(output.shape[0], -1)


def inception_features(
    model: Any,
    transform: Any,
    image_paths: Sequence[Path],
    device: Any,
    batch_size: int,
) -> np.ndarray:
    import torch
    from PIL import Image

    features: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            chunk = image_paths[start : start + batch_size]
            tensors = []
            for path in chunk:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            batch = torch.stack(tensors).to(device)
            out = inception_forward(model, batch)
            if hasattr(out, "logits"):
                out = out.logits
            out = out.reshape(out.shape[0], -1)
            features.append(out.detach().float().cpu().numpy())
    return np.concatenate(features, axis=0).astype(np.float32, copy=False)


def central_frame_indices(num_frames: int, count: int = 2) -> List[int]:
    if num_frames < 1:
        raise ValueError("Need at least one generated frame")
    count = min(max(1, count), num_frames)
    center = (num_frames - 1) / 2.0
    order = sorted(range(num_frames), key=lambda idx: (abs(idx - center), idx))
    return sorted(order[:count])


def random_frame_indices(num_frames: int, count: int, seed: int, pair_id: int) -> List[int]:
    if not 1 <= count <= num_frames:
        raise ValueError(f"Need {count} distinct FID states, but sequence has {num_frames}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, pair_id]))
    return sorted(int(index) for index in rng.choice(num_frames, size=count, replace=False))


def save_fid_features(
    args: argparse.Namespace,
    pair_output: Path,
    frames: Sequence[Any],
    inception_model: Any,
    inception_transform: Any,
    device: Any,
    raw_sources: Tuple[Path, Path] | None = None,
) -> Dict[str, Any]:
    center_indices = random_frame_indices(len(frames), args.fid_num_intermediate_states, args.fid_selection_seed, pair_id_from_name(pair_output.name, 0))
    if args.fid_num_views < 2:
        raise ValueError("FID needs at least two views per selected state")
    reference_paths: List[Path] = []
    for view in range(args.num_views):
        reference_paths.append(base.image_path(pair_output, "src1", view))
        reference_paths.append(base.image_path(pair_output, "src2", view))

    if raw_sources is not None:
        reference_cache = Path(args.reference_cache_root).resolve() / "inception" if args.reference_cache_root else pair_output / "feature_cache"
        reference = np.concatenate([gaussian_inception_features(path, args, inception_model, inception_transform, device, reference_cache) for path in raw_sources])
        generated = np.concatenate([gaussian_inception_features(frames[i].mesh, args, inception_model, inception_transform, device, pair_output / "feature_cache") for i in center_indices])
    else:
        if args.fid_num_views != args.num_views:
            raise ValueError("Blender backend currently requires --fid-num-views == --num-views; use gaussian for the 1000-view paper protocol")
        reference = inception_features(inception_model, inception_transform, reference_paths, device, args.fid_batch_size)
        # Both backends now save contiguous state blocks and explicit row labels.
        # Historical Blender archives were interleaved by view; the reader below
        # recognizes that legacy layout when the explicit labels are absent.
        generated_paths = [base.image_path(pair_output, frames[index].key, view)
                           for index in center_indices for view in range(args.num_views)]
        generated = inception_features(inception_model, inception_transform, generated_paths, device, args.fid_batch_size)
    if len(reference) != 2 * args.fid_num_views or len(generated) != len(center_indices) * args.fid_num_views:
        raise ValueError(f"Unexpected FID feature counts: reference={len(reference)}, generated={len(generated)}")
    feature_path = pair_output / "fid_features.npz"
    temporary = feature_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        reference=reference,
        generated=generated,
        center_indices=np.asarray(center_indices, dtype=np.int32),
        center_alphas=np.asarray([frames[i].alpha for i in center_indices], dtype=np.float32),
        generated_state_indices=np.repeat(np.asarray(center_indices, dtype=np.int32), args.fid_num_views),
        schema_version=np.asarray(2, dtype=np.int32),
    )
    temporary.replace(feature_path)
    return {
        "feature_file": str(feature_path),
        "reference_samples": int(reference.shape[0]),
        "generated_samples": int(generated.shape[0]),
        "feature_dim": int(reference.shape[1]),
        "selected_indices": center_indices,
        "selected_alphas": [float(frames[i].alpha) for i in center_indices],
        "selection": "seeded_random_without_replacement",
        "selection_seed": args.fid_selection_seed,
        "views_per_state": args.fid_num_views,
        "feature_layout": "contiguous_by_selected_state",
    }


def split_generated_state_features(features: Mapping[str, Any], protocol: Mapping[str, Any],
                                   render_backend: str = "gaussian") -> List[Tuple[int, float, np.ndarray]]:
    """Read explicit row labels, or reconstruct the two historical NPZ layouts."""
    generated = np.asarray(features["generated"])
    reference = np.asarray(features["reference"])
    indices = np.asarray(features["center_indices"] if "center_indices" in features else protocol.get("selected_indices", []))
    alphas = np.asarray(features["center_alphas"] if "center_alphas" in features else protocol.get("selected_alphas", []))
    if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer) or len(set(indices.tolist())) != len(indices) or np.any(indices < 0):
        raise ValueError("FID feature archive needs distinct nonnegative selected frame indices")
    if alphas.shape != indices.shape or not np.isfinite(alphas).all():
        raise ValueError("FID selected frame alphas must match the selected indices")
    views = int(protocol["views_per_state"])
    if views < 2 or generated.ndim != 2 or reference.ndim != 2 or reference.shape[1] != generated.shape[1]:
        raise ValueError("FID needs matching feature dimensions and >= 2 views per state")
    if len(reference) != 2 * views or len(generated) != len(indices) * views:
        raise ValueError(f"FID feature counts do not match two endpoints and {len(indices)} selected states at {views} views/state")
    if not np.isfinite(reference).all() or not np.isfinite(generated).all():
        raise ValueError("FID features must be finite")
    if "generated_state_indices" in features:
        labels = np.asarray(features["generated_state_indices"])
        if labels.shape != (len(generated),) or not np.issubdtype(labels.dtype, np.integer) or set(labels.tolist()) != set(indices.tolist()):
            raise ValueError("FID generated-state row labels do not match selected frame indices")
    else:
        layout = protocol.get("feature_layout")
        if layout is None:
            if render_backend not in {"gaussian", "blender"}:
                raise ValueError(f"Cannot infer legacy FID feature layout for backend {render_backend}")
            layout = "interleaved_by_view" if render_backend == "blender" else "contiguous_by_selected_state"
        if layout == "contiguous_by_selected_state":
            labels = np.repeat(indices, views)
        elif layout == "interleaved_by_view":
            labels = np.tile(indices, views)
        else:
            raise ValueError(f"Unknown FID feature layout: {layout}")
    states = [(int(index), float(alpha), generated[labels == index]) for index, alpha in zip(indices, alphas)]
    if any(len(state) != views for _, _, state in states):
        raise ValueError("FID row labels must assign exactly views_per_state samples to every selected state")
    return states


def compute_fid_interpretations(features: Mapping[str, Any], protocol: Mapping[str, Any],
                                mode: str = "both", render_backend: str = "gaussian") -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Compute both interpretations from the same rendered features, once."""
    modes = fid_aggregation_modes(mode)
    reference_stats = feature_statistics(features["reference"])
    metrics: Dict[str, Dict[str, float]] = {}
    details: Dict[str, Any] = {
        "schema_version": 2,
        "aggregation_modes": modes,
        "formulas": {FID_METRICS[name]: FID_FORMULAS[FID_METRICS[name]] for name in modes},
        "reference_distribution": "concat(src1_views, src2_views), shared by both interpretations",
        "pair_weighting": "arithmetic mean across pairs; each pair has equal weight",
    }
    if "concatenated" in modes:
        fid = frechet_distance_from_statistics(reference_stats, feature_statistics(features["generated"]))
        details["trajectory_fid"] = fid
        metrics[FID_METRICS["concatenated"]] = {"mean": fid, "std": float("nan"), "median": fid}
    if "mean-per-state" in modes:
        states = split_generated_state_features(features, protocol, render_backend)
        state_rows = [{"frame_index": index, "alpha": alpha, "generated_samples": len(values),
                       "fid": frechet_distance_from_statistics(reference_stats, feature_statistics(values))}
                      for index, alpha, values in states]
        mean = float(np.mean([row["fid"] for row in state_rows]))
        details["selected_states"] = state_rows
        details["mean_per_state_fid"] = mean
        details["state_weighting"] = "arithmetic mean across selected states within each pair"
        metrics[FID_METRICS["mean-per-state"]] = {"mean": mean, "std": float("nan"), "median": mean}
    return metrics, details


def evaluate_shard(args: argparse.Namespace) -> None:
    import torch
    import lpips

    run_dir = Path(args.run_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sequence_dirs = select_sequence_dirs(run_dir, args.pair_glob, args.shard_index, args.num_shards)
    print(
        f"Eval shard {args.shard_index}/{args.num_shards} | method_run={run_dir} | "
        f"pairs={len(sequence_dirs)} | views={args.num_views} | resolution={args.resolution}",
        flush=True,
    )

    lpips_device = base.resolve_device(args.device)
    lpips_model = lpips.LPIPS(net=args.lpips_net, spatial=False).to(lpips_device).eval()
    weighted_only = getattr(args, "metric_mode", "interp3d") == "weighted-endpoints-only"
    if weighted_only:
        print("Weighted endpoint LPIPS only: canonical sources + intermediates; no PPL/PDV/FID", flush=True)
    else:
        inception_model, inception_transform, inception_weights = load_inception(lpips_device)
        print(f"FID Inception weights: {inception_weights}", flush=True)

    completed: List[str] = []
    skipped: List[Dict[str, str]] = []
    for index, sequence_dir in enumerate(sequence_dirs):
        pair_output = output_root / sequence_dir.name
        pair_output.mkdir(parents=True, exist_ok=True)
        print(f"[{index + 1}/{len(sequence_dirs)}] {sequence_dir.name}", flush=True)
        try:
            if args.render_backend == "gaussian":
                src1, src2, frames, raw_info = discover_raw_frames(sequence_dir)
            else:
                src1, src2, frames, _ = base.discover_frames(sequence_dir)
            alpha_info = base.alpha_diagnostics(frames)
            if args.strict_uniform_alphas and not alpha_info["uniform"]:
                raise ValueError(f"Non-uniform alpha schedule: {alpha_info['deltas']}")
            signature = evaluation_signature(args, src1, src2, frames)
            metrics_path = pair_output / "metrics.json"
            if not args.overwrite and args.stage != "prepare" and metrics_path.is_file() and (weighted_only or (pair_output / "fid_features.npz").is_file()):
                previous = base.read_json(metrics_path)
                complete = completed_weighted_metrics(previous) if weighted_only else completed_fid_metrics(previous, args.fid_aggregation)
                if previous.get("evaluation_signature") == signature and complete:
                    completed.append(sequence_dir.name)
                    print("  reuse completed pair metrics", flush=True)
                    continue

            if args.render_backend == "gaussian":
                prepare_gaussian_images(args, pair_output, src1, src2, frames)
                write_json(pair_output / "manifest.json", {"render_backend": "gaussian", "camera_distribution": args.camera_distribution,
                           "num_views": args.num_views, "fid_num_views": args.fid_num_views,
                           "resolution": args.resolution, "protocol": raw_info["plan"].get("evaluation_protocol", {}),
                           "alpha_schedule": raw_info["plan"].get("alpha_schedule", "uniform"),
                           "source": "canonical dataset endpoints, shared across all methods"})
                if args.stage == "prepare":
                    completed.append(sequence_dir.name)
                    continue
                compute = base.compute_weighted_endpoint_metrics if weighted_only else base.compute_metrics
                result = compute(args, pair_output, frames, model=lpips_model, device=lpips_device)
                result["sequence_name"] = sequence_dir.name
                result["render_backend"] = "gaussian"
            else:
                result = base.process_sequence(args, sequence_dir, pair_output, lpips_model=lpips_model, lpips_device=lpips_device)
                if args.stage == "prepare":
                    completed.append(sequence_dir.name)
                    continue
                if result is None:
                    result = base.read_json(pair_output / "metrics.json")
                result["render_backend"] = "blender"

            if weighted_only:
                result["evaluation_signature"] = signature
                write_json(pair_output / "metrics.json", result)
                completed.append(sequence_dir.name)
                print(f"  weighted LPIPS (intermediate/full)={result['aggregate_across_views']['weighted_endpoint_lpips_mean']['mean']:.6f}/{result['aggregate_across_views']['weighted_endpoint_lpips_full_mean']['mean']:.6f}", flush=True)
                continue

            full_aggregate, full_transitions, full_per_view = full_path_metrics(
                args, pair_output, frames, lpips_model, lpips_device
            )
            result["aggregate_across_views"].update(full_aggregate)
            result.setdefault("definitions", {}).update(
                {
                    "lpips_adjacent_full_mean": "mean LPIPS over the complete src1 -> generated frames -> src2 path",
                    "ppl_full_sum": "sum of LPIPS over the complete src1 -> generated frames -> src2 path",
                    "ppl_full_normalized_reference_endpoints": "ppl_full_sum / LPIPS(src1, src2)",
                    "pdv_full": "population variance of complete-path consecutive LPIPS distances",
                }
            )
            result["full_path_includes_reference_endpoints"] = True
            result["evaluation_signature"] = signature
            result["fid_protocol"] = save_fid_features(
                args,
                pair_output,
                frames,
                inception_model,
                inception_transform,
                lpips_device,
                raw_sources=(src1, src2) if args.render_backend == "gaussian" else None,
            )
            result["fid_protocol"].update(
                {
                    "scope": "per trajectory; both concatenated-state and mean-per-state interpretations; pooled FID reported separately as diagnostic",
                    "reference_distribution": "src1 + src2 renders",
                    "generated_distribution": "two seeded random intermediate states",
                    "views_per_state": int(args.fid_num_views),
                    "backbone": "pytorch-fid TensorFlow-compatible Inception-v3 pool3 2048D",
                }
            )
            with np.load(pair_output / "fid_features.npz") as features:
                fid_metrics, fid_details = compute_fid_interpretations(features, result["fid_protocol"], args.fid_aggregation, args.render_backend)
            result["fid_protocol"].update(fid_details)
            result["aggregate_across_views"].update(fid_metrics)
            result["definitions"].update(fid_details["formulas"])
            write_csv(pair_output / "full_transitions.csv", full_transitions)
            write_csv(pair_output / "full_per_view.csv", full_per_view)
            write_json(pair_output / "metrics.json", result)
            completed.append(sequence_dir.name)
            print(
                f"  full PPL={full_aggregate['ppl_full_sum']['mean']:.6f} | "
                f"full PDV={full_aggregate['pdv_full']['mean']:.6f}",
                flush=True,
            )
        except Exception as exc:
            if not args.skip_invalid_pairs:
                raise
            skipped.append({"sequence": sequence_dir.name, "reason": repr(exc)})
            print(f"WARNING skip {sequence_dir.name}: {exc!r}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_json(
        output_root / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json",
        {
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "completed": completed,
            "skipped": skipped,
            "num_views": args.num_views,
            "resolution": args.resolution,
            "scheduled": [path.name for path in sequence_dirs],
            "render_backend": args.render_backend,
            "fid_num_views": args.fid_num_views,
            "fid_aggregation": args.fid_aggregation,
            "metric_mode": getattr(args, "metric_mode", "interp3d"),
        },
    )


def feature_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] < 1 or len(features) < 2:
        raise ValueError("FID needs a feature matrix with at least two samples")
    if not np.isfinite(features).all():
        raise ValueError("FID features must be finite")
    return features.mean(axis=0), np.atleast_2d(np.cov(features, rowvar=False))


def frechet_distance_from_statistics(reference_stats: Tuple[np.ndarray, np.ndarray],
                                     generated_stats: Tuple[np.ndarray, np.ndarray]) -> float:
    from scipy import linalg

    mu1, sigma1 = reference_stats
    mu2, sigma2 = generated_stats
    if mu1.shape != mu2.shape or sigma1.shape != sigma2.shape:
        raise ValueError("FID needs matching feature dimensions")
    eps = 1e-6
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0], dtype=np.float64) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diag(covmean).imag, 0.0, atol=1e-3):
            raise ValueError("FID covariance square-root has a significant imaginary component")
        covmean = covmean.real
    value = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(max(0.0, np.real(value)))


def frechet_distance(reference: np.ndarray, generated: np.ndarray) -> float:
    return frechet_distance_from_statistics(feature_statistics(reference), feature_statistics(generated))


def merge_outputs(args: argparse.Namespace) -> None:
    if getattr(args, "metric_mode", "interp3d") == "weighted-endpoints-only":
        merge_weighted_outputs(args)
        return
    # Namespace callers predating this option retain the historical behavior.
    # The public CLI defaults to both and never silently accepts legacy-only
    # pair files as a complete dual-interpretation evaluation.
    aggregation = getattr(args, "fid_aggregation", "concatenated")
    requested_metrics = required_fid_metrics(aggregation)
    output_root = Path(args.output_dir).resolve()
    pair_dirs = sorted(path for path in output_root.glob(args.pair_glob) if path.is_dir())
    if not pair_dirs:
        raise FileNotFoundError(f"No pair outputs found in {output_root}")

    pair_results: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    reference_features: List[np.ndarray] = []
    generated_features: List[np.ndarray] = []
    missing: List[str] = []

    for pair_dir in pair_dirs:
        metrics_path = pair_dir / "metrics.json"
        features_path = pair_dir / "fid_features.npz"
        if not metrics_path.is_file() or not features_path.is_file():
            missing.append(pair_dir.name)
            continue
        result = base.read_json(metrics_path)
        if not completed_fid_metrics(result, aggregation):
            raise RuntimeError(f"{pair_dir.name} lacks complete FID aggregation '{aggregation}'; rerun shard evaluation to refresh cached features/metrics")
        result["aggregate_across_views"] = {name: value for name, value in result["aggregate_across_views"].items()
                                            if name not in FID_METRICS.values() or name in requested_metrics}
        if pair_results and set(result["aggregate_across_views"]) != set(pair_results[0]["aggregate_across_views"]):
            raise RuntimeError(f"Inconsistent metric fields across pairs: {pair_dir.name}")
        if pair_results:
            first = pair_results[0]
            if (result["num_views"], result["lpips_backbone"], result["fid_protocol"]["views_per_state"], result.get("render_backend", "blender")) != (
                    first["num_views"], first["lpips_backbone"], first["fid_protocol"]["views_per_state"], first.get("render_backend", "blender")):
                raise RuntimeError(f"Inconsistent evaluation protocol across pairs: {pair_dir.name}")
            if "mean-per-state" in fid_aggregation_modes(aggregation) and len(result["fid_protocol"]["selected_states"]) != len(first["fid_protocol"]["selected_states"]):
                raise RuntimeError(f"Inconsistent number of selected FID states across pairs: {pair_dir.name}")
        result["sequence_name"] = pair_dir.name
        pair_results.append(result)
        row: Dict[str, Any] = {
            "sequence": pair_dir.name,
            "num_generated_frames": result["num_generated_frames"],
            "num_views": result["num_views"],
        }
        for metric, metric_stats in result["aggregate_across_views"].items():
            row[metric] = metric_stats["mean"]
        pair_rows.append(row)
        with np.load(features_path) as payload:
            if "mean-per-state" in fid_aggregation_modes(aggregation):
                split_generated_state_features(payload, result["fid_protocol"], result.get("render_backend", "blender"))
            reference_features.append(np.asarray(payload["reference"], dtype=np.float32))
            generated_features.append(np.asarray(payload["generated"], dtype=np.float32))

    if args.expected_num_pairs is not None and len(pair_results) != args.expected_num_pairs:
        raise RuntimeError(f"Expected {args.expected_num_pairs} evaluated pairs, found {len(pair_results)}")
    if missing and not args.allow_incomplete:
        raise RuntimeError(f"Incomplete evaluation: {len(missing)} pair(s) missing metrics/features: {missing[:20]}")
    if not pair_results:
        raise RuntimeError("No complete pair results to merge")

    aggregate = base._aggregate_pair_results(pair_results)
    reference = np.concatenate(reference_features, axis=0)
    generated = np.concatenate(generated_features, axis=0)

    fid_reason = None
    if min(len(reference), len(generated)) < args.fid_min_samples:
        fid = float("nan")
        fid_reason = (
            f"Skipped: need >= {args.fid_min_samples} pooled samples per distribution; "
            f"got reference={len(reference)}, generated={len(generated)}"
        )
    else:
        fid = frechet_distance(reference, generated)

    aggregate["fid_endpoint_manifold_pooled"] = {
        "mean": float(fid),
        "std": float("nan"),
        "median": float(fid),
    }

    global_fid = {
        metric: {
            "value": aggregate[metric]["mean"],
            "std": aggregate[metric]["std"],
            "formula": FID_FORMULAS[metric],
            "aggregation": "equal-weight arithmetic mean across pairs" + (" after equal-weight arithmetic mean across selected states within each pair" if metric == FID_METRICS["mean-per-state"] else " after concatenating selected-state features within each pair"),
            "views_per_state": pair_results[0]["fid_protocol"]["views_per_state"],
            "intermediate_selection": "seeded random states without replacement, identical across methods",
            "reference_distribution": "both endpoints concatenated within each pair",
            "backbone": "pytorch-fid TensorFlow-compatible Inception-v3 pool3 2048D",
        }
        for metric in requested_metrics
    }
    global_fid["fid_endpoint_manifold_pooled"] = {
        "value": float(fid),
        "formula": FID_FORMULAS["fid_endpoint_manifold_pooled"],
        "role": "diagnostic; different aggregation from either per-pair interpretation",
        "reference_samples": int(reference.shape[0]),
        "generated_samples": int(generated.shape[0]),
        "feature_dim": int(reference.shape[1]),
        "reference_distribution": "all src1/src2 renders pooled across pairs",
        "generated_distribution": "seeded random intermediate states per pair pooled across pairs",
        "backbone": "pytorch-fid TensorFlow-compatible Inception-v3 pool3 2048D",
        "reason": fid_reason,
    }

    batch_result = {
        "metric_direction": "lower_is_better",
        "aggregation": "equal pair weight after averaging views; per-pair FID interpretations are separately named; pooled FID is a diagnostic",
        "fid_aggregation": aggregation,
        "num_pairs": len(pair_results),
        "num_views_per_pair": pair_results[0]["num_views"],
        "lpips_backbone": pair_results[0]["lpips_backbone"],
        "render_backend": pair_results[0].get("render_backend", "blender"),
        "pairs": pair_rows,
        "aggregate_across_pairs": aggregate,
        "global_metrics": global_fid,
        "per_pair_fid_protocol": {result["sequence_name"]: result["fid_protocol"] for result in pair_results},
        "missing_pairs": missing,
    }
    write_json(output_root / "metrics_batch.json", batch_result)
    write_csv(output_root / "per_pair.csv", pair_rows)
    print(f"Merged {len(pair_results)} pairs -> {output_root / 'metrics_batch.json'}")
    print("; ".join(f"{metric} = {aggregate[metric]['mean']}" for metric in requested_metrics) + f"; pooled diagnostic FID = {fid}")


def merge_weighted_outputs(args: argparse.Namespace) -> None:
    """Merge dense LPIPS curves without opening feature files or loading FID."""
    output_root = Path(args.output_dir).resolve()
    pair_dirs = sorted(path for path in output_root.glob(args.pair_glob) if path.is_dir())
    if not pair_dirs:
        raise FileNotFoundError(f"No pair outputs found in {output_root}")
    results, pair_rows, missing = [], [], []
    for pair_dir in pair_dirs:
        path = pair_dir / "metrics.json"
        if not path.is_file():
            missing.append(pair_dir.name)
            continue
        result = base.read_json(path)
        if not completed_weighted_metrics(result):
            raise RuntimeError(f"Incomplete endpoint-inclusive weighted LPIPS metrics: {pair_dir.name}")
        if results:
            first = results[0]
            for field in ("num_views", "lpips_backbone", "num_generated_frames", "num_sequence_frames", "alphas", "render_backend"):
                if result.get(field) != first.get(field):
                    raise RuntimeError(f"Inconsistent weighted LPIPS protocol ({field}): {pair_dir.name}")
        result["sequence_name"] = pair_dir.name
        results.append(result)
        row = {"sequence": pair_dir.name, "num_generated_frames": result["num_generated_frames"],
               "num_sequence_frames": result["num_sequence_frames"], "num_views": result["num_views"]}
        row.update({metric: result["aggregate_across_views"][metric]["mean"] for metric in WEIGHTED_METRICS})
        pair_rows.append(row)
    if args.expected_num_pairs is not None and len(results) != args.expected_num_pairs:
        raise RuntimeError(f"Expected {args.expected_num_pairs} evaluated pairs, found {len(results)}")
    if missing and not args.allow_incomplete:
        raise RuntimeError(f"Incomplete weighted LPIPS evaluation: missing metrics for {missing[:20]}")
    if not results:
        raise RuntimeError("No complete pair results to merge")
    first = results[0]
    result = {
        "metric_direction": "lower_is_better", "metric_mode": "weighted-endpoints-only",
        "aggregation": "equal pair weight after averaging identical views; legacy mean/AUC use intermediates; full mean/AUC include canonical endpoints",
        "num_pairs": len(results), "num_views_per_pair": first["num_views"],
        "num_generated_frames": first["num_generated_frames"], "num_sequence_frames": first["num_sequence_frames"],
        "lpips_backbone": first["lpips_backbone"], "render_backend": first.get("render_backend", "blender"),
        "alphas": first["alphas"], "weighted_endpoint_includes_reference_endpoints": True,
        "definitions": first["definitions"], "pairs": pair_rows,
        "aggregate_across_pairs": base._aggregate_pair_results(results), "global_metrics": {}, "missing_pairs": missing,
    }
    write_json(output_root / "metrics_batch.json", result)
    write_csv(output_root / "per_pair.csv", pair_rows)
    print(f"Merged {len(results)} weighted LPIPS pairs, without FID -> {output_root / 'metrics_batch.json'}")


def check_fid_weights(_args: argparse.Namespace) -> None:
    path = inception_cache_path()
    if not path.is_file():
        raise FileNotFoundError(path)
    print(path)


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric-mode", choices=["interp3d", "weighted-endpoints-only"], default="interp3d",
                        help="Preserve Interp3D metrics, or compute source-weighted LPIPS only (with canonical endpoint curves and no FID/PPL)")
    parser.add_argument("--pair-glob", default="pair_*")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--stage", choices=["all", "prepare", "metrics"], default="all")
    parser.add_argument("--skip-invalid-pairs", action="store_true")
    parser.add_argument("--blender-bin", default=os.environ.get("BLENDER_BIN", "blender"))
    parser.add_argument("--render-script", default=str(base.DEFAULT_RENDER_SCRIPT))
    parser.add_argument("--num-views", dest="num_views", type=int, default=64)
    parser.add_argument("--render-backend", choices=["gaussian", "blender"], default="gaussian")
    parser.add_argument("--camera-distribution", choices=["hammersley", "orbit"], default="hammersley")
    parser.add_argument("--reference-cache-root", default="", help="Shared cache for canonical reference images and Inception features across all methods/CFGs")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--pitch-degrees", dest="pitch_degrees", type=float, default=15.0)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--fov-degrees", dest="fov_degrees", type=float, default=40.0)
    parser.add_argument("--object-up-axis", dest="object_up_axis", choices=["X", "Y", "Z"], default="Y")
    parser.add_argument("--render-engine", dest="render_engine", default="CYCLES")
    parser.add_argument("--appearance", choices=["materials", "geometry"], default="materials")
    parser.add_argument("--render-batch-size", dest="render_batch_size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips-net", dest="lpips_net", choices=["alex", "vgg", "squeeze"], default="vgg")
    parser.add_argument("--lpips-batch-size", dest="lpips_batch_size", type=int, default=8)
    parser.add_argument("--strict-uniform-alphas", dest="strict_uniform_alphas", action="store_true")
    parser.add_argument("--fid-batch-size", type=int, default=16)
    parser.add_argument("--fid-num-intermediate-states", type=int, default=2)
    parser.add_argument("--fid-num-views", type=int, default=1000)
    parser.add_argument("--fid-selection-seed", type=int, default=42)
    parser.add_argument("--fid-aggregation", choices=["both", "concatenated", "mean-per-state"], default="both",
                        help="FID of concatenated selected states, mean of separate state FIDs, or both (same cached render features)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Textured GLB full-path perceptual evaluation + both per-pair FID interpretations")
    sub = parser.add_subparsers(dest="command", required=True)

    shard = sub.add_parser("shard", help="Render/evaluate one pair shard")
    add_eval_args(shard)

    merge = sub.add_parser("merge", help="Merge pair outputs and compute pooled FID")
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--metric-mode", choices=["interp3d", "weighted-endpoints-only"], default="interp3d")
    merge.add_argument("--pair-glob", default="pair_*")
    merge.add_argument("--fid-min-samples", type=int, default=512)
    merge.add_argument("--allow-incomplete", action="store_true")
    merge.add_argument("--expected-num-pairs", type=int)
    merge.add_argument("--fid-aggregation", choices=["both", "concatenated", "mean-per-state"], default="both",
                       help="Required per-pair FID interpretations; use concatenated to merge historical results")

    sub.add_parser("check-fid-weights", help="Fail unless cached Inception weights are available")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "shard":
        evaluate_shard(args)
    elif args.command == "merge":
        merge_outputs(args)
    elif args.command == "check-fid-weights":
        check_fid_weights(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
