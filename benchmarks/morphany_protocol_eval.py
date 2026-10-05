#!/usr/bin/env python3
"""MorphAny3D-style post-evaluation for the three-way textured benchmark.

This script reuses renders already produced by eval_textured_benchmark.py.
It computes a pooled FID with:
  reference = src1 + src2 renders for every pair/view
  generated = every state of the 50-frame morph sequence, including src1/src2

For the intended protocol:
  50 pairs x 12 views x 2 endpoints = 1,200 reference samples
  50 pairs x 12 views x 50 states   = 30,000 generated samples

PPL and PDV are not recomputed here: the existing evaluator's endpoint-inclusive
ppl_full_sum and pdv_full already implement sum/variance of consecutive LPIPS
on the complete source -> intermediates -> target trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

import eval_textured_benchmark as textured_eval


DEFAULT_METHODS = ("morphflow", "morphany3d", "interp3d")
FID_METRIC = "fid_morphany_all_frames"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scalar_mean(aggregate: Dict[str, Any], name: str) -> float:
    value = aggregate[name]
    if isinstance(value, dict):
        value = value["mean"]
    return float(value)


def collect_image_paths(
    metrics_dir: Path,
    num_views: int,
    expected_pairs: int,
    expected_sequence_states: int,
) -> tuple[List[Path], List[Path], List[Dict[str, Any]]]:
    pair_dirs = sorted(path for path in metrics_dir.glob("pair_*") if path.is_dir())
    if len(pair_dirs) != expected_pairs:
        raise RuntimeError(
            f"{metrics_dir}: expected {expected_pairs} pair directories, found {len(pair_dirs)}"
        )

    reference_paths: List[Path] = []
    generated_paths: List[Path] = []
    pair_meta: List[Dict[str, Any]] = []

    for pair_dir in pair_dirs:
        manifest_path = pair_dir / "manifest.json"
        metrics_path = pair_dir / "metrics.json"
        if not manifest_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Incomplete pair evaluation: {pair_dir}")

        manifest = read_json(manifest_path)
        frames = list(manifest.get("frames", []))
        generated_keys = ["src1"] + [str(frame["key"]) for frame in frames] + ["src2"]
        if len(generated_keys) != expected_sequence_states:
            raise RuntimeError(
                f"{pair_dir.name}: expected {expected_sequence_states} total states, "
                f"found {len(generated_keys)} ({len(frames)} intermediates)"
            )

        for view in range(num_views):
            view_name = f"{view:03d}.png"
            src1 = pair_dir / "images" / "src1" / view_name
            src2 = pair_dir / "images" / "src2" / view_name
            if not src1.is_file() or not src2.is_file():
                raise FileNotFoundError(f"Missing endpoint render in {pair_dir}, view={view}")
            reference_paths.extend((src1, src2))

            for key in generated_keys:
                path = pair_dir / "images" / key / view_name
                if not path.is_file():
                    raise FileNotFoundError(f"Missing generated render: {path}")
                generated_paths.append(path)

        pair_meta.append(
            {
                "pair": pair_dir.name,
                "num_intermediates": len(frames),
                "num_sequence_states": len(generated_keys),
                "num_views": num_views,
            }
        )

    return reference_paths, generated_paths, pair_meta


def compute_method(
    method: str,
    run_root: Path,
    model: Any,
    transform: Any,
    device: Any,
    batch_size: int,
    num_views: int,
    expected_pairs: int,
    expected_sequence_states: int,
    inception_weights: Path,
) -> Dict[str, Any]:
    metrics_dir = run_root / "metrics" / method
    batch_path = metrics_dir / "metrics_batch.json"
    if not batch_path.is_file():
        raise FileNotFoundError(batch_path)

    batch = read_json(batch_path)
    if int(batch.get("num_pairs", -1)) != expected_pairs:
        raise RuntimeError(
            f"{method}: metrics_batch num_pairs={batch.get('num_pairs')} != {expected_pairs}"
        )
    if int(batch.get("num_views_per_pair", -1)) != num_views:
        raise RuntimeError(
            f"{method}: num_views_per_pair={batch.get('num_views_per_pair')} != {num_views}"
        )

    reference_paths, generated_paths, pair_meta = collect_image_paths(
        metrics_dir,
        num_views=num_views,
        expected_pairs=expected_pairs,
        expected_sequence_states=expected_sequence_states,
    )

    expected_reference = expected_pairs * num_views * 2
    expected_generated = expected_pairs * num_views * expected_sequence_states
    if len(reference_paths) != expected_reference:
        raise RuntimeError(
            f"{method}: reference count {len(reference_paths)} != expected {expected_reference}"
        )
    if len(generated_paths) != expected_generated:
        raise RuntimeError(
            f"{method}: generated count {len(generated_paths)} != expected {expected_generated}"
        )

    print(
        f"[{method}] Inception features: reference={len(reference_paths)} "
        f"generated={len(generated_paths)} views={num_views} states={expected_sequence_states}",
        flush=True,
    )

    reference = textured_eval.inception_features(
        model, transform, reference_paths, device, batch_size
    )
    generated = textured_eval.inception_features(
        model, transform, generated_paths, device, batch_size
    )
    fid = textured_eval.frechet_distance(reference, generated)

    protocol = {
        "value": float(fid),
        "reference_samples": int(reference.shape[0]),
        "generated_samples": int(generated.shape[0]),
        "feature_dim": int(reference.shape[1]),
        "num_pairs": expected_pairs,
        "views_per_state": num_views,
        "sequence_states_per_pair": expected_sequence_states,
        "reference_distribution": "src1 + src2 renders pooled across all pairs/views",
        "generated_distribution": (
            "all morph-sequence renders pooled across all pairs/views, "
            "including src1 and src2"
        ),
        "backbone": "torchvision Inception-v3 IMAGENET1K_V1 pre-logit 2048D",
        "weights_cache": str(inception_weights),
    }

    aggregate = batch.setdefault("aggregate_across_pairs", {})
    aggregate[FID_METRIC] = {
        "mean": float(fid),
        "std": float("nan"),
        "median": float(fid),
    }
    batch.setdefault("global_metrics", {})[FID_METRIC] = protocol
    batch["morphany_protocol"] = {
        "total_sequence_frames": expected_sequence_states,
        "num_intermediate_frames": expected_sequence_states - 2,
        "num_views": num_views,
        "num_pairs": expected_pairs,
        "ppl_metric": "ppl_full_sum",
        "pdv_metric": "pdv_full",
        "fid_metric": FID_METRIC,
    }
    write_json(batch_path, batch)

    aggregate = batch["aggregate_across_pairs"]
    result = {
        "method": method,
        "num_pairs": expected_pairs,
        "num_views": num_views,
        "sequence_states": expected_sequence_states,
        "ppl": scalar_mean(aggregate, "ppl_full_sum"),
        "pdv": scalar_mean(aggregate, "pdv_full"),
        "lpips_adjacent_mean": scalar_mean(aggregate, "lpips_adjacent_full_mean"),
        "fid": float(fid),
        "reference_fid_samples": int(reference.shape[0]),
        "generated_fid_samples": int(generated.shape[0]),
    }
    print(
        f"[{method}] PPL={result['ppl']:.6f} PDV={result['pdv']:.8f} "
        f"FID={result['fid']:.6f}",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MorphAny3D-style 50-frame / 12-view / 50-pair evaluation post-pass"
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--num-views", type=int, default=12)
    parser.add_argument("--expected-pairs", type=int, default=50)
    parser.add_argument("--expected-sequence-states", type=int, default=50)
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    run_root = Path(args.run_root).resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)

    device = torch.device(args.device)
    model, transform, inception_weights = textured_eval.load_inception(device)
    print(f"Inception weights: {inception_weights}", flush=True)

    rows: List[Dict[str, Any]] = []
    for method in args.methods:
        rows.append(
            compute_method(
                method=method,
                run_root=run_root,
                model=model,
                transform=transform,
                device=device,
                batch_size=args.fid_batch_size,
                num_views=args.num_views,
                expected_pairs=args.expected_pairs,
                expected_sequence_states=args.expected_sequence_states,
                inception_weights=inception_weights,
            )
        )

    summary = {
        "protocol": {
            "name": "MorphAny3D-style requested protocol",
            "num_pairs": args.expected_pairs,
            "total_sequence_frames": args.expected_sequence_states,
            "num_intermediate_frames": args.expected_sequence_states - 2,
            "num_views": args.num_views,
            "fid_reference_samples_expected": args.expected_pairs * args.num_views * 2,
            "fid_generated_samples_expected": (
                args.expected_pairs * args.num_views * args.expected_sequence_states
            ),
            "ppl_source": "aggregate_across_pairs.ppl_full_sum.mean",
            "pdv_source": "aggregate_across_pairs.pdv_full.mean",
            "fid_source": FID_METRIC,
        },
        "methods": {row["method"]: row for row in rows},
    }
    write_json(run_root / "morphany_protocol_metrics.json", summary)
    write_csv(run_root / "morphany_protocol_metrics.csv", rows)

    print("\nFinal comparison", flush=True)
    print("method,ppl,pdv,fid", flush=True)
    for row in rows:
        print(
            f"{row['method']},{row['ppl']:.6f},{row['pdv']:.8f},{row['fid']:.6f}",
            flush=True,
        )
    print(f"Wrote {run_root / 'morphany_protocol_metrics.json'}", flush=True)
    print(f"Wrote {run_root / 'morphany_protocol_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
