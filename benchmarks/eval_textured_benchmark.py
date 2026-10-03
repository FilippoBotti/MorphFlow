#!/usr/bin/env python3
"""Appearance-aware evaluation for the three-way textured-GLB benchmark.

This script wraps MorphFlow's eval_perceptual_sequence.py and adds:
  * full-sequence LPIPS/PPL/PDV including src1/src2 endpoint transitions;
  * pooled endpoint-manifold FID using two central generated states;
  * pair sharding for multi-GPU execution on one node;
  * deterministic merge of shard outputs.

All methods are evaluated from their final textured GLBs using the exact same
Blender camera protocol and material rendering path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np

import eval_perceptual_sequence as base


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
    all_dirs = base.discover_sequence_dirs(run_dir, pair_glob)
    selected: List[Path] = []
    for idx, path in enumerate(all_dirs):
        pair_id = pair_id_from_name(path.name, idx)
        if pair_id % num_shards == shard_index:
            selected.append(path)
    return selected


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
    from torchvision.models import Inception_V3_Weights

    weights = Inception_V3_Weights.IMAGENET1K_V1
    filename = Path(urlparse(weights.url).path).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def load_inception(device: Any):
    import torch
    import torch.nn as nn
    from torchvision.models import Inception_V3_Weights, inception_v3

    cache = inception_cache_path()
    if not cache.is_file():
        raise FileNotFoundError(
            f"Missing cached Inception-v3 weights: {cache}. "
            "Preload torchvision Inception_V3_Weights.IMAGENET1K_V1 into TORCH_HOME before the offline job."
        )
    weights = Inception_V3_Weights.IMAGENET1K_V1
    # Load the cached official torchvision state dict explicitly so the offline
    # benchmark never attempts a network download and preprocessing happens
    # exactly once via weights.transforms().
    model = inception_v3(
        weights=None,
        aux_logits=True,
        transform_input=False,
        init_weights=False,
    )
    state = torch.load(cache, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.fc = nn.Identity()
    model = model.eval().to(device)
    transform = weights.transforms()
    return model, transform, cache


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
            out = model(batch)
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


def save_fid_features(
    args: argparse.Namespace,
    pair_output: Path,
    frames: Sequence[Any],
    inception_model: Any,
    inception_transform: Any,
    device: Any,
) -> Dict[str, Any]:
    center_indices = central_frame_indices(len(frames), args.fid_num_intermediate_states)
    reference_paths: List[Path] = []
    generated_paths: List[Path] = []
    for view in range(args.num_views):
        reference_paths.append(base.image_path(pair_output, "src1", view))
        reference_paths.append(base.image_path(pair_output, "src2", view))
        for index in center_indices:
            generated_paths.append(base.image_path(pair_output, frames[index].key, view))

    reference = inception_features(
        inception_model, inception_transform, reference_paths, device, args.fid_batch_size
    )
    generated = inception_features(
        inception_model, inception_transform, generated_paths, device, args.fid_batch_size
    )
    feature_path = pair_output / "fid_features.npz"
    np.savez_compressed(
        feature_path,
        reference=reference,
        generated=generated,
        center_indices=np.asarray(center_indices, dtype=np.int32),
        center_alphas=np.asarray([frames[i].alpha for i in center_indices], dtype=np.float32),
    )
    return {
        "feature_file": str(feature_path),
        "reference_samples": int(reference.shape[0]),
        "generated_samples": int(generated.shape[0]),
        "feature_dim": int(reference.shape[1]),
        "center_indices": center_indices,
        "center_alphas": [float(frames[i].alpha) for i in center_indices],
    }


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
    inception_model, inception_transform, inception_weights = load_inception(lpips_device)
    print(f"FID Inception weights: {inception_weights}", flush=True)

    completed: List[str] = []
    skipped: List[Dict[str, str]] = []
    for index, sequence_dir in enumerate(sequence_dirs):
        pair_output = output_root / sequence_dir.name
        pair_output.mkdir(parents=True, exist_ok=True)
        print(f"[{index + 1}/{len(sequence_dirs)}] {sequence_dir.name}", flush=True)
        try:
            src1, src2, frames, _ = base.discover_frames(sequence_dir)
            alpha_info = base.alpha_diagnostics(frames)
            if args.strict_uniform_alphas and not alpha_info["uniform"]:
                raise ValueError(f"Non-uniform alpha schedule: {alpha_info['deltas']}")

            # Reuse the canonical renderer + original internal-only metrics.
            result = base.process_sequence(
                args,
                sequence_dir,
                pair_output,
                lpips_model=lpips_model,
                lpips_device=lpips_device,
            )
            if result is None:
                result = base.read_json(pair_output / "metrics.json")

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
            result["fid_protocol"] = save_fid_features(
                args,
                pair_output,
                frames,
                inception_model,
                inception_transform,
                lpips_device,
            )
            result["fid_protocol"].update(
                {
                    "scope": "pooled across all benchmark pairs during merge",
                    "reference_distribution": "src1 + src2 renders",
                    "generated_distribution": "two states nearest alpha=0.5",
                    "views_per_state": int(args.num_views),
                    "backbone": "torchvision Inception-v3 IMAGENET1K_V1 pre-logit 2048D",
                }
            )
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
        },
    )


def frechet_distance(reference: np.ndarray, generated: np.ndarray) -> float:
    from scipy import linalg

    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    mu1 = reference.mean(axis=0)
    mu2 = generated.mean(axis=0)
    sigma1 = np.cov(reference, rowvar=False)
    sigma2 = np.cov(generated, rowvar=False)
    eps = 1e-6
    sigma1 = sigma1 + np.eye(sigma1.shape[0], dtype=np.float64) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0], dtype=np.float64) * eps
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diag(covmean).imag, 0.0, atol=1e-3):
            raise ValueError("FID covariance square-root has a significant imaginary component")
        covmean = covmean.real
    value = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(np.real(value))


def merge_outputs(args: argparse.Namespace) -> None:
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
        payload = np.load(features_path)
        reference_features.append(np.asarray(payload["reference"], dtype=np.float32))
        generated_features.append(np.asarray(payload["generated"], dtype=np.float32))

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

    # Keep the final aggregator simple. This is a pooled batch-level metric,
    # not a mean of per-pair FIDs; std is intentionally NaN.
    aggregate["fid_endpoint_manifold"] = {
        "mean": float(fid),
        "std": float("nan"),
        "median": float(fid),
    }

    batch_result = {
        "metric_direction": "lower_is_better",
        "aggregation": "pair metrics: equal pair weight after averaging views; FID: pooled feature distributions across all pairs",
        "num_pairs": len(pair_results),
        "num_views_per_pair": pair_results[0]["num_views"],
        "lpips_backbone": pair_results[0]["lpips_backbone"],
        "pairs": pair_rows,
        "aggregate_across_pairs": aggregate,
        "global_metrics": {
            "fid_endpoint_manifold": {
                "value": float(fid),
                "reference_samples": int(reference.shape[0]),
                "generated_samples": int(generated.shape[0]),
                "feature_dim": int(reference.shape[1]),
                "reference_distribution": "all src1/src2 renders pooled across pairs",
                "generated_distribution": "two center intermediate states per pair pooled across pairs",
                "backbone": "torchvision Inception-v3 IMAGENET1K_V1 pre-logit 2048D",
                "reason": fid_reason,
            }
        },
        "missing_pairs": missing,
    }
    write_json(output_root / "metrics_batch.json", batch_result)
    write_csv(output_root / "per_pair.csv", pair_rows)
    print(f"Merged {len(pair_results)} pairs -> {output_root / 'metrics_batch.json'}")
    print(f"FID endpoint manifold = {fid}")


def check_fid_weights(_args: argparse.Namespace) -> None:
    path = inception_cache_path()
    if not path.is_file():
        raise FileNotFoundError(path)
    print(path)


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pair-glob", default="pair_*")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--stage", choices=["all", "prepare", "metrics"], default="all")
    parser.add_argument("--skip-invalid-pairs", action="store_true")
    parser.add_argument("--blender-bin", default=os.environ.get("BLENDER_BIN", "blender"))
    parser.add_argument("--render-script", default=str(base.DEFAULT_RENDER_SCRIPT))
    parser.add_argument("--num-views", dest="num_views", type=int, default=64)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Textured GLB full-path perceptual evaluation + pooled FID")
    sub = parser.add_subparsers(dest="command", required=True)

    shard = sub.add_parser("shard", help="Render/evaluate one pair shard")
    add_eval_args(shard)

    merge = sub.add_parser("merge", help="Merge pair outputs and compute pooled FID")
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--pair-glob", default="pair_*")
    merge.add_argument("--fid-min-samples", type=int, default=512)
    merge.add_argument("--allow-incomplete", action="store_true")

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
