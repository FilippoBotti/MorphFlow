#!/usr/bin/env python3
"""
Generate a split-disjoint, alpha-coherent MorphAny3D distillation dataset.

Core guarantees
---------------
1. Split by source object before pair generation. Train/val/test pairs are built
   only within their own asset split, so no source object can appear in more than
   one split.
2. Alpha symmetry is respected. Metadata alpha always means:

       target = alpha * src_1 + (1 - alpha) * src_2

   where src_1/src_2 are canonical sorted asset names.
3. MorphAny3D's discrete morphing trajectory is respected. Requested alpha is
   mapped to the official frame grid alpha_i = 1 - i / (morphing_num - 1).
4. Per pair, the script chooses forward or reverse generation for each requested
   alpha to minimize the number of sequential TFSA steps that must actually be run.
5. Only useful files are saved:

       assets/<asset>/ss_latent.pt
       assets/<asset>/structured_latent.pt
       assets/<asset>/occupancy.pt
       targets/<src_1>+<src_2>/alpha_<alpha>/ss_latent.pt
       targets/<src_1>+<src_2>/alpha_<alpha>/structured_latent.pt
       targets/<src_1>+<src_2>/alpha_<alpha>/occupancy.pt
       metadata.json, metadata_train.json, metadata_val.json, metadata_test.json
       splits.json

Example
-------
python generate_morphany3d_split_dataset.py \
  --assets-dir /home/filippo/datasets/3d/flux_outputs \
  --output-dir /home/filippo/datasets/3d/morphing_dataset_flux_v2 \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --pairs-per-asset 50 \
  --alphas 0.23 0.64 0.88 \
  --alpha-denom 100 \
  --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Set these before importing TRELLIS / torch modules.
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
SPLIT_ORDER = ("train", "val", "test")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def safe_name(value: str) -> str:
    value = Path(value).stem
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    return value.strip("_") or "asset"


def alpha_slug(alpha: float, decimals: int = 6) -> str:
    text = f"{alpha:.{decimals}f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


def latent_dir_ready(path: Path) -> bool:
    return all((path / name).is_file() for name in ("ss_latent.pt", "structured_latent.pt", "occupancy.pt"))


def structured_payload(slat) -> Dict[str, torch.Tensor]:
    return {
        "feats": slat.feats.detach().cpu(),
        "coords": slat.coords.detach().cpu().to(torch.int32),
    }


def occupancy_from_ss_decoder(pipeline, ss_latent: torch.Tensor) -> torch.Tensor:
    decoder = pipeline.models["sparse_structure_decoder"]
    with torch.no_grad():
        logits = decoder(ss_latent)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    occ = (logits > 0).detach().cpu()
    if occ.ndim == 5 and occ.shape[0] == 1 and occ.shape[1] == 1:
        return occ[0, 0].contiguous()
    if occ.ndim == 5 and occ.shape[0] == 1:
        return occ[0].contiguous()
    return occ.contiguous()


def save_latent_triplet(pipeline, out_dir: Path, ss_latent: torch.Tensor, slat, extra_manifest: Optional[dict] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(ss_latent.detach().cpu(), out_dir / "ss_latent.pt")
    atomic_torch_save(structured_payload(slat), out_dir / "structured_latent.pt")
    atomic_torch_save(occupancy_from_ss_decoder(pipeline, ss_latent), out_dir / "occupancy.pt")
    if extra_manifest is not None:
        atomic_json_save(extra_manifest, out_dir / "manifest.json")


def list_image_paths(assets_dir: Path) -> Dict[str, Path]:
    paths = sorted(p for p in assets_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images in {assets_dir}, found {len(paths)}")
    mapping: Dict[str, Path] = {}
    collisions: Dict[str, List[str]] = {}
    for path in paths:
        name = safe_name(path.name)
        if name in mapping:
            collisions.setdefault(name, [str(mapping[name])]).append(str(path))
        mapping[name] = path
    if collisions:
        details = "; ".join(f"{k}: {v}" for k, v in list(collisions.items())[:10])
        raise ValueError(f"Image names collide after sanitization: {details}")
    return mapping


def split_assets(
    names: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    strategy: str = "random",
) -> Dict[str, List[str]]:
    ratios = {"train": float(train_ratio), "val": float(val_ratio), "test": float(test_ratio)}
    if any(v < 0 for v in ratios.values()):
        raise ValueError(f"Split ratios must be non-negative: {ratios}")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("At least one split ratio must be > 0")

    names = list(sorted(names))
    if strategy == "random":
        rng = random.Random(seed)
        rng.shuffle(names)
    elif strategy == "sorted":
        pass
    else:
        raise ValueError("--split-strategy must be random or sorted")

    n = len(names)
    exact = {k: ratios[k] / total * n for k in SPLIT_ORDER}
    counts = {k: int(math.floor(exact[k])) for k in SPLIT_ORDER}
    remainder = n - sum(counts.values())
    order = sorted(SPLIT_ORDER, key=lambda k: (exact[k] - counts[k], ratios[k]), reverse=True)
    for key in order[:remainder]:
        counts[key] += 1

    # Avoid pathological one-object splits when possible. A split with one asset cannot create pairs.
    for key in ("val", "test", "train"):
        if ratios[key] > 0 and counts[key] == 1 and n >= 4:
            donor = max((k for k in SPLIT_ORDER if counts[k] > 2), key=lambda k: counts[k], default=None)
            if donor:
                counts[donor] -= 1
                counts[key] += 1

    splits: Dict[str, List[str]] = {}
    start = 0
    for key in SPLIT_ORDER:
        count = counts[key]
        splits[key] = sorted(names[start : start + count])
        start += count

    seen = {}
    for split, assets in splits.items():
        for asset in assets:
            if asset in seen:
                raise RuntimeError(f"Internal split error: {asset} in both {seen[asset]} and {split}")
            seen[asset] = split
    return splits


def compute_degree(names: Sequence[str], pairs: Iterable[Tuple[str, str]]) -> Dict[str, int]:
    degree = {name: 0 for name in names}
    for a, b in pairs:
        if a in degree:
            degree[a] += 1
        if b in degree:
            degree[b] += 1
    return degree


def build_pairs_with_cap(names: Sequence[str], pairs_per_asset: int, seed: int) -> List[Tuple[str, str]]:
    names = sorted(names)
    if len(names) < 2:
        return []
    max_possible = len(names) - 1
    if pairs_per_asset <= 0 or pairs_per_asset >= max_possible:
        return list(combinations(names, 2))

    rng = random.Random(seed)
    target = min(pairs_per_asset, max_possible)
    degree = {name: 0 for name in names}
    used = set()
    pairs: List[Tuple[str, str]] = []
    shuffled = names[:]
    rng.shuffle(shuffled)

    made_progress = True
    while made_progress:
        made_progress = False
        for a in shuffled:
            if degree[a] >= target:
                continue
            candidates = [b for b in names if b != a and degree[b] < target and tuple(sorted((a, b))) not in used]
            rng.shuffle(candidates)
            for b in candidates:
                if degree[a] >= target:
                    break
                if degree[b] >= target:
                    continue
                pair = tuple(sorted((a, b)))
                if pair in used:
                    continue
                used.add(pair)
                degree[a] += 1
                degree[b] += 1
                pairs.append(pair)
                made_progress = True
    return sorted(pairs)


@torch.no_grad()
def encode_image_condition(pipeline, image_path: Path) -> dict:
    image = Image.open(image_path)
    processed = pipeline.preprocess_image(image)
    return pipeline.get_cond([processed])


@torch.no_grad()
def sample_endpoint_latents(pipeline, cond: dict, seed: int, sparse_sampler_params: dict, slat_sampler_params: dict):
    """Equivalent to pipeline.run(image), but returns source ss latent and SLAT without mesh/GS/RF decoding."""
    seed_everything(seed)

    flow_model = pipeline.models["sparse_structure_flow_model"]
    reso = flow_model.resolution
    noise = torch.randn(1, flow_model.in_channels, reso, reso, reso, device=pipeline.device)

    sampler_params = {**pipeline.sparse_structure_sampler_params, **sparse_sampler_params}
    ss_latent = pipeline.sparse_structure_sampler.sample(
        flow_model,
        noise,
        **cond,
        **sampler_params,
        verbose=True,
    ).samples

    decoder = pipeline.models["sparse_structure_decoder"]
    voxels = decoder(ss_latent) > 0
    coords = torch.argwhere(voxels)[:, [0, 2, 3, 4]].int()
    slat = pipeline.sample_slat(cond, coords, slat_sampler_params)
    return ss_latent, slat


@dataclass(frozen=True)
class PairRequest:
    split: str
    src1_name: str
    src2_name: str
    requested_alpha_src1: float
    canon_a: str
    canon_b: str
    canon_a_path: Path
    canon_b_path: Path
    requested_alpha_a: float


@dataclass(frozen=True)
class PlannedTarget:
    request: PairRequest
    direction: str  # forward A->B, reverse B->A, endpoint_a, endpoint_b.
    morphing_idx: int
    alpha_dir: float
    effective_alpha_a: float

    @property
    def split(self) -> str:
        return self.request.split

    @property
    def pair_name(self) -> str:
        return f"{self.request.canon_a}+{self.request.canon_b}"

    @property
    def target_name(self) -> str:
        return f"{self.pair_name}/alpha_{alpha_slug(self.effective_alpha_a)}"


def canonicalize_request(split: str, a_name: str, b_name: str, alpha_a_input: float, name_to_path: Dict[str, Path]) -> PairRequest:
    if not (0.0 <= alpha_a_input <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha_a_input}")
    if a_name == b_name:
        raise ValueError(f"Pair uses the same asset twice: {a_name}")

    if a_name <= b_name:
        canon_a, canon_b = a_name, b_name
        alpha_a = float(alpha_a_input)
    else:
        canon_a, canon_b = b_name, a_name
        alpha_a = 1.0 - float(alpha_a_input)

    return PairRequest(
        split=split,
        src1_name=a_name,
        src2_name=b_name,
        requested_alpha_src1=float(alpha_a_input),
        canon_a=canon_a,
        canon_b=canon_b,
        canon_a_path=name_to_path[canon_a],
        canon_b_path=name_to_path[canon_b],
        requested_alpha_a=float(alpha_a),
    )


def direction_index_for_alpha(alpha_dir: float, morphing_num: int, snap: bool, tol: float) -> Tuple[int, float]:
    denom = morphing_num - 1
    raw_idx = (1.0 - alpha_dir) * denom
    idx = int(round(raw_idx))
    idx = max(0, min(denom, idx))
    effective = 1.0 - idx / denom
    if abs(effective - alpha_dir) > tol and not snap:
        raise ValueError(
            f"alpha={alpha_dir:.10f} is not on the MorphAny3D grid for morphing_num={morphing_num}. "
            f"Nearest effective alpha is {effective:.10f} at idx={idx}. "
            "Use --snap-alpha 1 or choose a finer --alpha-denom."
        )
    return idx, effective


def optimize_direction_plan_for_pair(
    requests: Sequence[PairRequest],
    morphing_num: int,
    snap: bool,
    tol: float,
) -> List[PlannedTarget]:
    """Minimize TFSA trajectory length for one canonical pair."""
    rows = []
    for req in requests:
        a = req.requested_alpha_a
        f_idx, f_eff_dir_alpha = direction_index_for_alpha(a, morphing_num, snap, tol)
        r_idx, r_eff_dir_alpha = direction_index_for_alpha(1.0 - a, morphing_num, snap, tol)
        rows.append((req, f_idx, f_eff_dir_alpha, f_eff_dir_alpha, r_idx, r_eff_dir_alpha, 1.0 - r_eff_dir_alpha))

    denom = morphing_num - 1
    interior = [r for r in rows if r[1] not in (0, denom) and r[4] not in (0, denom)]
    endpoints = [r for r in rows if r not in interior]

    planned: List[PlannedTarget] = []
    if interior:
        candidate_f = sorted({0} | {r[1] for r in interior})
        candidate_r = sorted({0} | {r[4] for r in interior})
        best: Optional[Tuple[int, int, int, int]] = None
        for max_f in candidate_f:
            for max_r in candidate_r:
                feasible = all((f_idx <= max_f) or (r_idx <= max_r) for _, f_idx, _, _, r_idx, _, _ in interior)
                if not feasible:
                    continue
                num_dirs = int(max_f > 0) + int(max_r > 0)
                score = max_f + max_r
                candidate = (score, num_dirs, max_f, max_r)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            raise RuntimeError("Unable to find a feasible direction plan")
        _, _, best_max_f, best_max_r = best

        for req, f_idx, f_eff_dir_alpha, f_eff_a, r_idx, r_eff_dir_alpha, r_eff_a in interior:
            can_f = f_idx <= best_max_f
            can_r = r_idx <= best_max_r
            if can_f and (not can_r or f_idx <= r_idx):
                planned.append(PlannedTarget(req, "forward", f_idx, f_eff_dir_alpha, f_eff_a))
            elif can_r:
                planned.append(PlannedTarget(req, "reverse", r_idx, r_eff_dir_alpha, r_eff_a))
            else:
                raise RuntimeError("Internal direction planning error")

    for req, f_idx, f_eff_dir_alpha, f_eff_a, r_idx, r_eff_dir_alpha, r_eff_a in endpoints:
        if req.requested_alpha_a >= 0.5:
            planned.append(PlannedTarget(req, "endpoint_a", 0, 1.0, 1.0))
        else:
            planned.append(PlannedTarget(req, "endpoint_b", 0, 1.0, 0.0))

    planned.sort(key=lambda x: (x.split, x.pair_name, x.effective_alpha_a, x.direction, x.morphing_idx))
    return planned


def plan_all_targets(
    requests: Sequence[PairRequest],
    morphing_num: int,
    snap: bool,
    tol: float,
) -> List[PlannedTarget]:
    grouped: Dict[Tuple[str, str, str], List[PairRequest]] = {}
    for req in requests:
        grouped.setdefault((req.split, req.canon_a, req.canon_b), []).append(req)
    planned: List[PlannedTarget] = []
    for key in sorted(grouped):
        # Deduplicate same alpha after canonicalization.
        dedup: Dict[str, PairRequest] = {}
        for req in grouped[key]:
            dedup[f"{req.requested_alpha_a:.10f}"] = req
        planned.extend(optimize_direction_plan_for_pair(list(dedup.values()), morphing_num, snap, tol))
    return planned


def cleanup_old_index(cache_dir: Path, old_idx: int) -> None:
    if old_idx <= 0 or not cache_dir.exists():
        return
    patterns = [
        f"ss_sa_morphing{old_idx}_*",
        f"slat_sa_morphing{old_idx}_*",
        f"feat_coords_morphing{old_idx}.pt",
        f"coords_morphing{old_idx}.pt",
    ]
    for pattern in patterns:
        for path in cache_dir.glob(pattern):
            path.unlink(missing_ok=True)


@torch.no_grad()
def run_one_morphany3d_step(
    pipeline,
    src_cond: dict,
    tar_cond: dict,
    work_cache: Path,
    seed: int,
    morphing_idx: int,
    alpha_dir: float,
    tfsa_alpha: float,
    sparse_sampler_params: dict,
    slat_sampler_params: dict,
    ss_mca_flag: bool = True,
    slat_mca_flag: bool = True,
    ss_tfsa_flag: bool = True,
    slat_tfsa_flag: bool = True,
):
    seed_everything(seed)
    work_cache.mkdir(parents=True, exist_ok=True)

    morphing_params = {
        "save_cache_path": str(work_cache),
        "init_morphing_flag": False,
        "ss_mca_flag": bool(ss_mca_flag),
        "slat_mca_flag": bool(slat_mca_flag),
        "ss_tfsa_flag": bool(ss_tfsa_flag),
        "slat_tfsa_flag": bool(slat_tfsa_flag),
        "oc_flag": False,
        "tar_cond": tar_cond["cond"],
        "alpha": float(alpha_dir),
        "morphing_idx": int(morphing_idx),
        "tfsa_cache_idx": int(morphing_idx - 1),
        "tfsa_alpha": float(tfsa_alpha),
    }

    coords, _voxels, ss_latent = pipeline.sample_sparse_structure_morphing(
        src_cond,
        1,
        sparse_sampler_params,
        morphing_params,
    )
    slat = pipeline.sample_slat_morphing(
        src_cond,
        coords,
        slat_sampler_params,
        morphing_params,
    )
    return ss_latent, slat


def build_metadata_entry(planned: PlannedTarget) -> dict:
    req = planned.request
    target_rel = f"targets/{planned.target_name}"
    return {
        "split": planned.split,
        "src_1": req.canon_a,
        "src_2": req.canon_b,
        "target": planned.target_name,
        "alpha": float(planned.effective_alpha_a),
        "alpha_requested": float(req.requested_alpha_a),
        "alpha_definition": "alpha is the fraction of src_1; target = alpha*src_1 + (1-alpha)*src_2",
        "src1_dir": f"assets/{req.canon_a}",
        "src2_dir": f"assets/{req.canon_b}",
        "target_dir": target_rel,
        "src1_ss_latent": f"assets/{req.canon_a}/ss_latent.pt",
        "src2_ss_latent": f"assets/{req.canon_b}/ss_latent.pt",
        "target_ss_latent": f"{target_rel}/ss_latent.pt",
        "src1_structured_latent": f"assets/{req.canon_a}/structured_latent.pt",
        "src2_structured_latent": f"assets/{req.canon_b}/structured_latent.pt",
        "target_structured_latent": f"{target_rel}/structured_latent.pt",
        "src1_occupancy": f"assets/{req.canon_a}/occupancy.pt",
        "src2_occupancy": f"assets/{req.canon_b}/occupancy.pt",
        "target_occupancy": f"{target_rel}/occupancy.pt",
        "generated_direction": planned.direction,
        "direction_alpha": float(planned.alpha_dir),
        "morphing_idx": int(planned.morphing_idx),
    }


def ensure_asset_latents(
    pipeline,
    asset_name: str,
    asset_path: Path,
    output_dir: Path,
    seed: int,
    cond_cache: Dict[str, dict],
    sparse_sampler_params: dict,
    slat_sampler_params: dict,
) -> dict:
    out_dir = output_dir / "assets" / asset_name
    if latent_dir_ready(out_dir):
        if asset_name not in cond_cache:
            cond_cache[asset_name] = encode_image_condition(pipeline, asset_path)
        return cond_cache[asset_name]

    print(f"[asset] generating {asset_name}")
    cond = encode_image_condition(pipeline, asset_path)
    cond_cache[asset_name] = cond
    ss_latent, slat = sample_endpoint_latents(
        pipeline,
        cond,
        seed=seed,
        sparse_sampler_params=sparse_sampler_params,
        slat_sampler_params=slat_sampler_params,
    )
    save_latent_triplet(
        pipeline,
        out_dir,
        ss_latent,
        slat,
        extra_manifest={"name": asset_name, "image": str(asset_path), "seed": seed, "kind": "source_asset"},
    )
    torch.cuda.empty_cache()
    return cond


def group_by_pair_and_direction(planned: Sequence[PlannedTarget]) -> Dict[Tuple[str, str, str], List[PlannedTarget]]:
    grouped: Dict[Tuple[str, str, str], List[PlannedTarget]] = {}
    for item in planned:
        grouped.setdefault((item.split, item.pair_name, item.direction), []).append(item)
    for key in grouped:
        grouped[key].sort(key=lambda x: x.morphing_idx)
    return grouped


def parse_sampler_params(json_text: Optional[str]) -> dict:
    if not json_text:
        return {}
    return json.loads(json_text)


def write_metadata_files(output_dir: Path, planned: Sequence[PlannedTarget], splits: Dict[str, List[str]], config: dict) -> None:
    ready_entries = []
    missing = 0
    for item in planned:
        out_dir = output_dir / "targets" / item.target_name
        if latent_dir_ready(out_dir):
            ready_entries.append(build_metadata_entry(item))
        else:
            missing += 1

    ready_entries.sort(key=lambda e: (e["split"], e["src_1"], e["src_2"], float(e["alpha"])))
    atomic_json_save(ready_entries, output_dir / "metadata.json")
    for split in SPLIT_ORDER:
        split_entries = [e for e in ready_entries if e["split"] == split]
        atomic_json_save(split_entries, output_dir / f"metadata_{split}.json")

    split_asset_sets = {k: sorted(v) for k, v in splits.items()}
    # Plain split file consumed by morph_dataset_coherent.py.
    atomic_json_save(split_asset_sets, output_dir / "split_assets.json")
    split_payload = {
        "assets": split_asset_sets,
        "counts": {k: len(v) for k, v in split_asset_sets.items()},
        "metadata_counts": {k: sum(1 for e in ready_entries if e["split"] == k) for k in SPLIT_ORDER},
        "missing_planned_targets": missing,
        "config": config,
    }
    atomic_json_save(split_payload, output_dir / "splits.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate split-disjoint MorphAny3D alpha dataset.")
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-strategy", choices=["random", "sorted"], default="random")
    parser.add_argument("--pairs-per-asset", type=int, default=50, help="<=0 means all unordered pairs inside each split.")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.23, 0.64, 0.88])

    parser.add_argument("--alpha-denom", type=int, default=100, help="Grid denominator; 100 gives alpha step 0.01 and morphing_num=101.")
    parser.add_argument("--morphing-num", type=int, default=None, help="Overrides --alpha-denom by setting MorphAny3D grid size directly.")
    parser.add_argument("--snap-alpha", type=int, choices=[0, 1], default=0)
    parser.add_argument("--alpha-tol", type=float, default=1e-7)
    parser.add_argument("--tfsa-alpha", type=float, default=0.8)
    parser.add_argument("--disable-tfsa", type=int, choices=[0, 1], default=0)
    parser.add_argument("--sparse-sampler-params", type=str, default=None, help='JSON, e.g. \'{"steps": 25}\'')
    parser.add_argument("--slat-sampler-params", type=str, default=None, help='JSON, e.g. \'{"steps": 25}\'')
    parser.add_argument("--dry-run", action="store_true", help="Only build splits and print the plan; do not import TRELLIS or generate latents.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    if args.alpha_denom <= 0:
        raise ValueError("--alpha-denom must be > 0")
    morphing_num = int(args.morphing_num or (args.alpha_denom + 1))
    if morphing_num < 3:
        raise ValueError("morphing_num must be >= 3")

    sparse_sampler_params = parse_sampler_params(args.sparse_sampler_params)
    slat_sampler_params = parse_sampler_params(args.slat_sampler_params)

    name_to_path = list_image_paths(args.assets_dir)
    splits = split_assets(
        names=sorted(name_to_path),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        strategy=args.split_strategy,
    )

    split_pairs: Dict[str, List[Tuple[str, str]]] = {}
    requests: List[PairRequest] = []
    for split in SPLIT_ORDER:
        pairs = build_pairs_with_cap(splits[split], args.pairs_per_asset, seed=args.seed + {'train': 101, 'val': 202, 'test': 303}[split])
        split_pairs[split] = pairs
        for a_name, b_name in pairs:
            for alpha in args.alphas:
                requests.append(canonicalize_request(split, a_name, b_name, float(alpha), name_to_path))

    planned = plan_all_targets(
        requests,
        morphing_num=morphing_num,
        snap=bool(args.snap_alpha),
        tol=float(args.alpha_tol),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_root = args.output_dir / ".tmp_morphany3d_cache"
    work_root.mkdir(parents=True, exist_ok=True)

    print("=== MorphAny3D split-disjoint dataset ===")
    print(f"assets_dir: {args.assets_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"assets total: {len(name_to_path)}")
    print(f"morphing_num: {morphing_num}  grid_step: {1.0 / (morphing_num - 1):.8f}")
    print(f"alphas requested: {args.alphas}")
    for split in SPLIT_ORDER:
        print(f"split {split:5s}: assets={len(splits[split])} pairs={len(split_pairs[split])} samples={len(split_pairs[split]) * len(args.alphas)}")

    for (split, pair_name, direction), items in group_by_pair_and_direction(planned).items():
        if direction.startswith("endpoint"):
            continue
        print(
            f"plan {split:5s} {pair_name:40s} {direction:7s}: "
            f"targets={len(items)} max_idx={max(x.morphing_idx for x in items)} "
            f"alphas={[round(x.effective_alpha_a, 6) for x in items]}"
        )

    config = {
        "assets_dir": str(args.assets_dir),
        "seed": args.seed,
        "split_strategy": args.split_strategy,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "pairs_per_asset": args.pairs_per_asset,
        "alphas": [float(x) for x in args.alphas],
        "alpha_denom": args.alpha_denom,
        "morphing_num": morphing_num,
        "snap_alpha": bool(args.snap_alpha),
        "tfsa_alpha": args.tfsa_alpha,
        "tfsa_enabled": args.disable_tfsa == 0,
    }

    if args.dry_run:
        dry_payload = {
            "assets": {k: sorted(v) for k, v in splits.items()},
            "counts": {k: len(v) for k, v in splits.items()},
            "planned_pairs": {k: len(v) for k, v in split_pairs.items()},
            "planned_samples": {k: len(split_pairs[k]) * len(args.alphas) for k in SPLIT_ORDER},
            "config": config,
        }
        atomic_json_save(dry_payload, args.output_dir / "splits.dry_run.json")
        print(f"Dry run written: {args.output_dir / 'splits.dry_run.json'}")
        return

    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model_id)
    pipeline.cuda()

    cond_cache: Dict[str, dict] = {}
    required_assets = sorted({p.request.canon_a for p in planned} | {p.request.canon_b for p in planned})
    for asset_name in required_assets:
        ensure_asset_latents(
            pipeline,
            asset_name,
            name_to_path[asset_name],
            args.output_dir,
            args.seed,
            cond_cache,
            sparse_sampler_params,
            slat_sampler_params,
        )

    # Endpoint targets are physical copies to keep the loader simple.
    for item in [p for p in planned if p.direction.startswith("endpoint")]:
        src_asset = item.request.canon_a if item.direction == "endpoint_a" else item.request.canon_b
        src_dir = args.output_dir / "assets" / src_asset
        tgt_dir = args.output_dir / "targets" / item.target_name
        if not latent_dir_ready(tgt_dir):
            tgt_dir.mkdir(parents=True, exist_ok=True)
            for fname in ("ss_latent.pt", "structured_latent.pt", "occupancy.pt"):
                shutil.copy2(src_dir / fname, tgt_dir / fname)
            atomic_json_save(
                {"kind": "endpoint_copy", "copied_from": f"assets/{src_asset}", "alpha": item.effective_alpha_a, "split": item.split},
                tgt_dir / "manifest.json",
            )

    grouped = group_by_pair_and_direction([p for p in planned if p.direction in ("forward", "reverse")])
    ss_tfsa_flag = args.disable_tfsa == 0
    slat_tfsa_flag = args.disable_tfsa == 0

    for (split, pair_name, direction), items in grouped.items():
        missing_items = [p for p in items if not latent_dir_ready(args.output_dir / "targets" / p.target_name)]
        if not missing_items:
            print(f"[pair] {split} {pair_name} {direction}: skipped, all targets ready")
            continue

        req = items[0].request
        if direction == "forward":
            src_name, tar_name = req.canon_a, req.canon_b
        else:
            src_name, tar_name = req.canon_b, req.canon_a

        src_cond = cond_cache[src_name]
        tar_cond = cond_cache[tar_name]
        work_cache = work_root / f"{split}__{pair_name}__{direction}"
        shutil.rmtree(work_cache, ignore_errors=True)
        work_cache.mkdir(parents=True, exist_ok=True)

        items_by_idx: Dict[int, List[PlannedTarget]] = {}
        for p in missing_items:
            items_by_idx.setdefault(p.morphing_idx, []).append(p)
        max_idx = max(items_by_idx)
        print(f"[pair] {split} {pair_name} {direction}: running idx 1..{max_idx}, saving {len(missing_items)} target(s)")

        for idx in range(1, max_idx + 1):
            alpha_dir = 1.0 - idx / (morphing_num - 1)
            ss_latent, slat = run_one_morphany3d_step(
                pipeline,
                src_cond=src_cond,
                tar_cond=tar_cond,
                work_cache=work_cache,
                seed=args.seed,
                morphing_idx=idx,
                alpha_dir=alpha_dir,
                tfsa_alpha=args.tfsa_alpha,
                sparse_sampler_params=sparse_sampler_params,
                slat_sampler_params=slat_sampler_params,
                ss_mca_flag=True,
                slat_mca_flag=True,
                ss_tfsa_flag=ss_tfsa_flag,
                slat_tfsa_flag=slat_tfsa_flag,
            )

            if idx in items_by_idx:
                for target in items_by_idx[idx]:
                    out_dir = args.output_dir / "targets" / target.target_name
                    save_latent_triplet(
                        pipeline,
                        out_dir,
                        ss_latent,
                        slat,
                        extra_manifest={
                            "kind": "morph_target",
                            "split": split,
                            "pair": target.pair_name,
                            "direction": direction,
                            "morphing_idx": idx,
                            "morphing_num": morphing_num,
                            "direction_alpha": alpha_dir,
                            "canonical_alpha_src_1": target.effective_alpha_a,
                            "requested_alpha_src_1": target.request.requested_alpha_a,
                            "seed": args.seed,
                            "tfsa_alpha": args.tfsa_alpha,
                            "tfsa_enabled": args.disable_tfsa == 0,
                        },
                    )
                    print(f"  saved {split}/{target.target_name}  idx={idx} dir_alpha={alpha_dir:.6f}")

            cleanup_old_index(work_cache, idx - 1)
            torch.cuda.empty_cache()

        shutil.rmtree(work_cache, ignore_errors=True)

    write_metadata_files(args.output_dir, planned, splits, config)
    shutil.rmtree(work_root, ignore_errors=True)
    print(f"Done. Metadata: {args.output_dir / 'metadata.json'}")
    print(f"Train metadata: {args.output_dir / 'metadata_train.json'}")
    print(f"Val metadata:   {args.output_dir / 'metadata_val.json'}")
    print(f"Test metadata:  {args.output_dir / 'metadata_test.json'}")


if __name__ == "__main__":
    main()
