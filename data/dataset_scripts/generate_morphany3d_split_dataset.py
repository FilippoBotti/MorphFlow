"""
Generate a split-disjoint, alpha-coherent MorphAny3D distillation dataset.

Guarantees
----------
1. Split by source object before pair generation.
   Train/val/test pairs are built only inside their own asset split.
2. No object leakage between train, val and test.
3. For every planned pair, generate a 5-step MorphAny3D sequence.
4. Keep exactly 3 deterministic-random targets from those 5 steps.
5. For every object, generate at most K pairings inside the same split.
6. Alpha symmetry is respected:

       target = alpha * src_1 + (1 - alpha) * src_2

   where src_1/src_2 are canonical sorted asset names in metadata.
7. Alpha is the real MorphAny3D blending parameter passed to MCA/noise interpolation.
8. MCA and TFSA are both enabled by default for plausible intermediate targets.
9. Only useful files are saved:

       assets/<asset>/slat_feats.pt
       assets/<asset>/slat_coords.pt
       assets/<asset>/ss_latent.pt
       assets/<asset>/structured_latent.pt
       assets/<asset>/occupancy.pt

       targets/<src_1>+<src_2>/alpha_<alpha>/ss_latent.pt
       targets/<src_1>+<src_2>/alpha_<alpha>/slat_feats.pt
       targets/<src_1>+<src_2>/alpha_<alpha>/slat_coords.pt
       targets/<src_1>+<src_2>/alpha_<alpha>/structured_latent.pt
       targets/<src_1>+<src_2>/alpha_<alpha>/occupancy.pt

       metadata.json
       metadata_train.json
       metadata_val.json
       metadata_test.json
       split_assets.json
       splits.json
       pair_alphas.json
       pair_sequence_alphas.json
       pair_selected_indices.json
"""

from __future__ import annotations

import fcntl
import uuid
from contextlib import contextmanager

import argparse
import hashlib
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

# Set before importing TRELLIS / torch modules.
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
SPLIT_ORDER = ("train", "val", "test")
MORPH_SEQUENCE_STEPS = 5
TARGETS_PER_PAIR = 3
DEFAULT_ALPHA_JITTER = 0.04


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

def tree_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: tree_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tree_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(tree_to_device(v, device) for v in obj)
    return obj


def tree_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: tree_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tree_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(tree_to_cpu(v) for v in obj)
    return obj


def alpha_slug(alpha: float, decimals: int = 6) -> str:
    text = f"{alpha:.{decimals}f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    os.replace(tmp, path)

@contextmanager
def file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def stable_mod_key(text: str, modulo: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little", signed=False)
    return value % modulo


def latent_dir_ready(path: Path) -> bool:
    return all(
        (path / name).is_file()
        for name in (
            "slat_feats.pt",
            "slat_coords.pt",
            "ss_latent.pt",
            "structured_latent.pt",
            "occupancy.pt",
        )
    )


def structured_payload(slat) -> Dict[str, torch.Tensor]:
    return {
        "feats": slat.feats.detach().cpu(),
        "coords": slat.coords.detach().cpu().to(torch.int32),
    }


def normalize_occupancy_tensor(occ: torch.Tensor) -> torch.Tensor:
    occ = occ.detach().cpu()
    if occ.ndim == 5 and occ.shape[0] == 1 and occ.shape[1] == 1:
        return occ[0, 0].contiguous()
    if occ.ndim == 5 and occ.shape[0] == 1:
        return occ[0].contiguous()
    return occ.contiguous()


def occupancy_from_ss_decoder(pipeline, ss_latent: torch.Tensor) -> torch.Tensor:
    decoder = pipeline.models["sparse_structure_decoder"]
    with torch.no_grad():
        logits = decoder(ss_latent)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    return normalize_occupancy_tensor(logits > 0)


def save_latent_triplet(
    pipeline,
    out_dir: Path,
    ss_latent: torch.Tensor,
    slat,
    extra_manifest: Optional[dict] = None,
    occupancy: Optional[torch.Tensor] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    atomic_torch_save(slat.feats.detach().cpu(), out_dir / "slat_feats.pt")
    atomic_torch_save(slat.coords.detach().cpu().to(torch.int32), out_dir / "slat_coords.pt")
    atomic_torch_save(ss_latent.detach().cpu(), out_dir / "ss_latent.pt")
    atomic_torch_save(structured_payload(slat), out_dir / "structured_latent.pt")
    if occupancy is None:
        occupancy = occupancy_from_ss_decoder(pipeline, ss_latent)
    else:
        occupancy = normalize_occupancy_tensor(occupancy)
    atomic_torch_save(occupancy, out_dir / "occupancy.pt")

    if extra_manifest is not None:
        atomic_json_save(extra_manifest, out_dir / "manifest.json")


def list_image_paths(assets_dir: Path) -> Dict[str, Path]:
    paths = sorted(
        p for p in assets_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

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
    ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }

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
    order = sorted(
        SPLIT_ORDER,
        key=lambda k: (exact[k] - counts[k], ratios[k]),
        reverse=True,
    )

    for key in order[:remainder]:
        counts[key] += 1

    # A split with one object cannot create pairs.
    # When possible, avoid one-object splits by borrowing from the largest split.
    for key in ("val", "test", "train"):
        if ratios[key] > 0 and counts[key] == 1 and n >= 4:
            donors = [k for k in SPLIT_ORDER if counts[k] > 2]
            if donors:
                donor = max(donors, key=lambda k: counts[k])
                counts[donor] -= 1
                counts[key] += 1

    splits: Dict[str, List[str]] = {}
    start = 0

    for key in SPLIT_ORDER:
        count = counts[key]
        splits[key] = sorted(names[start:start + count])
        start += count

    seen = {}
    for split, assets in splits.items():
        for asset in assets:
            if asset in seen:
                raise RuntimeError(
                    f"Internal split error: {asset} in both {seen[asset]} and {split}"
                )
            seen[asset] = split

    return splits


def build_pairs_with_cap(
    names: Sequence[str],
    pairs_per_asset: int,
    seed: int,
) -> List[Tuple[str, str]]:
    """
    Build unordered pairs inside one split.

    pairs_per_asset <= 0 means all unordered pairs.
    Otherwise each object appears in at most pairs_per_asset pairs.
    """
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

            candidates = [
                b for b in names
                if b != a
                and degree[b] < target
                and tuple(sorted((a, b))) not in used
            ]

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


def stable_seed_from_parts(*parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def stable_pair_seed(seed: int, split: str, a_name: str, b_name: str) -> int:
    return stable_seed_from_parts(seed, split, a_name, b_name)


def random_grid_alphas_for_pair(
    seed: int,
    split: str,
    a_name: str,
    b_name: str,
    count: int,
    denom: int,
    alpha_min: float,
    alpha_max: float,
) -> List[float]:
    """
    Generate deterministic random alpha values for one pair.

    Values are sampled on the MorphAny3D grid:
        alpha = k / denom

    Endpoints are excluded by default because source assets are saved separately.
    """
    if count <= 0:
        return []

    candidates = alpha_grid_indices(denom, alpha_min, alpha_max)

    if len(candidates) < count:
        raise ValueError(
            f"Not enough alpha grid points between {alpha_min} and {alpha_max} "
            f"with denom={denom}: have {len(candidates)}, need {count}."
        )

    rng = random.Random(stable_pair_seed(seed, split, a_name, b_name))
    values = sorted(rng.sample(candidates, count))
    return [v / denom for v in values]


def jittered_sequence_alphas_for_pair(
    seed: int,
    split: str,
    a_name: str,
    b_name: str,
    steps: int,
    jitter: float,
    alpha_min: float,
    alpha_max: float,
) -> List[float]:
    """
    Build a monotonic MorphAny3D sequence from src_1 to src_2.

    The ideal 5-step sequence is evenly spaced at
    5/6, 4/6, 3/6, 2/6, 1/6. Each point receives a small
    deterministic jitter, so alpha remains the real blending parameter
    but samples are not perfectly aligned across pairs.
    """
    if steps <= 0:
        raise ValueError("sequence steps must be positive")
    if jitter < 0:
        raise ValueError("--alpha-jitter must be >= 0")
    if not (0.0 <= alpha_min < alpha_max <= 1.0):
        raise ValueError("Require 0 <= --alpha-min < --alpha-max <= 1")

    gap = 1.0 / float(steps + 1)
    max_jitter = gap * 0.45
    if jitter > max_jitter:
        raise ValueError(
            f"--alpha-jitter={jitter} is too large for {steps} sequence steps; "
            f"use <= {max_jitter:.6f} to preserve source-to-target order."
        )

    rng = random.Random(stable_seed_from_parts(seed, split, a_name, b_name, "sequence"))
    values: List[float] = []

    for idx in range(1, steps + 1):
        base = 1.0 - idx * gap
        lo = max(alpha_min, base - jitter)
        hi = min(alpha_max, base + jitter)

        if lo > hi:
            raise ValueError(
                f"alpha range [{alpha_min}, {alpha_max}] excludes sequence point "
                f"{base:.6f} for pair {a_name}+{b_name}"
            )

        alpha = rng.uniform(lo, hi) if jitter > 0 else base
        values.append(float(alpha))

    for left, right in zip(values, values[1:]):
        if left <= right:
            raise RuntimeError(
                f"Internal alpha sequence error for {a_name}+{b_name}: {values}"
            )

    return values


def selected_sequence_indices_for_pair(
    seed: int,
    split: str,
    a_name: str,
    b_name: str,
    steps: int,
    count: int,
) -> List[int]:
    if count > steps:
        raise ValueError(f"Cannot keep {count} targets from a {steps}-step sequence")
    rng = random.Random(stable_seed_from_parts(seed, split, a_name, b_name, "keep"))
    return sorted(rng.sample(range(1, steps + 1), count))


def alpha_grid_indices(
    denom: int,
    alpha_min: float,
    alpha_max: float,
) -> List[int]:
    if denom <= 1:
        raise ValueError("--alpha-denom must be > 1")

    if not (0.0 <= alpha_min < alpha_max <= 1.0):
        raise ValueError("Require 0 <= --alpha-min < --alpha-max <= 1")

    lo = max(1, int(math.ceil(alpha_min * denom - 1e-9)))
    hi = min(denom - 1, int(math.floor(alpha_max * denom + 1e-9)))
    return list(range(lo, hi + 1))


def split_counts_from_total(
    total_count: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    unit: int,
) -> Dict[str, int]:
    if total_count < 0:
        raise ValueError("--target-total-samples must be >= 0")
    if total_count % unit != 0:
        raise ValueError(
            f"--target-total-samples must be divisible by {unit}, because "
            f"the dataset keeps exactly {unit} targets per pair."
        )

    ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }
    if any(v < 0 for v in ratios.values()):
        raise ValueError(f"Split ratios must be non-negative: {ratios}")

    ratio_total = sum(ratios.values())
    if ratio_total <= 0:
        raise ValueError("At least one split ratio must be > 0")

    total_units = total_count // unit
    exact = {k: ratios[k] / ratio_total * total_units for k in SPLIT_ORDER}
    units = {k: int(math.floor(exact[k])) for k in SPLIT_ORDER}

    remainder = total_units - sum(units.values())
    order = sorted(
        SPLIT_ORDER,
        key=lambda k: (exact[k] - units[k], ratios[k]),
        reverse=True,
    )
    for key in order[:remainder]:
        units[key] += 1

    return {k: units[k] * unit for k in SPLIT_ORDER}


def target_sample_counts_from_args(args) -> Dict[str, Optional[int]]:
    explicit = {
        "train": args.target_train_samples,
        "val": args.target_val_samples,
        "test": args.target_test_samples,
    }

    if args.target_total_samples is None:
        return explicit

    if any(value is not None for value in explicit.values()):
        raise ValueError(
            "--target-total-samples cannot be combined with "
            "--target-train-samples/--target-val-samples/--target-test-samples"
        )

    return split_counts_from_total(
        total_count=args.target_total_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        unit=TARGETS_PER_PAIR,
    )


def has_exact_sample_targets(target_counts: Dict[str, Optional[int]]) -> bool:
    return any(value is not None for value in target_counts.values())


def choose_pairs_for_exact_sample_count(
    split: str,
    pairs: Sequence[Tuple[str, str]],
    target_count: int,
    samples_per_pair: int,
    seed: int,
) -> List[Tuple[Tuple[str, str], int]]:
    if target_count < 0:
        raise ValueError(f"target sample count for {split} must be >= 0")
    if target_count == 0:
        return []
    if samples_per_pair <= 0:
        raise ValueError("samples_per_pair must be > 0")
    if target_count % samples_per_pair != 0:
        raise ValueError(
            f"target sample count for {split} must be divisible by {samples_per_pair}, "
            f"because the generator keeps exactly {samples_per_pair} targets per pair."
        )

    pairs_needed = target_count // samples_per_pair
    total_capacity = len(pairs) * samples_per_pair
    if total_capacity < target_count:
        raise ValueError(
            f"Split {split} cannot produce {target_count} samples with the current assets/pair cap: "
            f"{len(pairs)} pairs * {samples_per_pair} target(s) = {total_capacity}. "
            "Increase --pairs-per-asset or add more assets."
        )

    shuffled = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    return [(pair, samples_per_pair) for pair in shuffled[:pairs_needed]]


@torch.no_grad()
def encode_image_condition(pipeline, image_path: Path) -> dict:
    with Image.open(image_path) as image:
        processed = pipeline.preprocess_image(image)

    cond = pipeline.get_cond([processed])
    cond = tree_to_cpu(cond)

    del processed
    torch.cuda.empty_cache()

    return cond

@torch.no_grad()
def sample_endpoint_latents(
    pipeline,
    cond: dict,
    seed: int,
    sparse_sampler_params: dict,
    slat_sampler_params: dict,
):
    """
    Equivalent to pipeline.run(image), but returns source ss latent and SLAT
    without decoding mesh / gaussian / radiance field.
    """
    cond = tree_to_device(cond, pipeline.device)

    seed_everything(seed)

    flow_model = pipeline.models["sparse_structure_flow_model"]
    reso = flow_model.resolution

    noise = torch.randn(
        1,
        flow_model.in_channels,
        reso,
        reso,
        reso,
        device=pipeline.device,
    )

    sampler_params = {
        **pipeline.sparse_structure_sampler_params,
        **sparse_sampler_params,
    }

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
    sequence_idx: int
    requested_alpha_src1: float
    requested_sequence_alphas_src1: Tuple[float, ...]
    canon_a: str
    canon_b: str
    canon_a_path: Path
    canon_b_path: Path
    requested_alpha_a: float
    sequence_alphas_a: Tuple[float, ...]


@dataclass(frozen=True)
class PlannedTarget:
    request: PairRequest
    direction: str
    morphing_idx: int
    alpha_dir: float
    effective_alpha_a: float
    sequence_alphas_dir: Tuple[float, ...]

    @property
    def split(self) -> str:
        return self.request.split

    @property
    def pair_name(self) -> str:
        return f"{self.request.canon_a}+{self.request.canon_b}"

    @property
    def target_name(self) -> str:
        return f"{self.pair_name}/alpha_{alpha_slug(self.effective_alpha_a)}"


def canonicalize_request(
    split: str,
    a_name: str,
    b_name: str,
    sequence_idx: int,
    alpha_a_input: float,
    sequence_alphas_input: Sequence[float],
    name_to_path: Dict[str, Path],
) -> PairRequest:
    if not (0.0 <= alpha_a_input <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha_a_input}")

    if a_name == b_name:
        raise ValueError(f"Pair uses the same asset twice: {a_name}")

    if a_name <= b_name:
        canon_a, canon_b = a_name, b_name
        alpha_a = float(alpha_a_input)
        sequence_alphas_a = tuple(float(alpha) for alpha in sequence_alphas_input)
    else:
        canon_a, canon_b = b_name, a_name
        alpha_a = 1.0 - float(alpha_a_input)
        sequence_alphas_a = tuple(1.0 - float(alpha) for alpha in sequence_alphas_input)

    return PairRequest(
        split=split,
        src1_name=a_name,
        src2_name=b_name,
        sequence_idx=int(sequence_idx),
        requested_alpha_src1=float(alpha_a_input),
        requested_sequence_alphas_src1=tuple(float(alpha) for alpha in sequence_alphas_input),
        canon_a=canon_a,
        canon_b=canon_b,
        canon_a_path=name_to_path[canon_a],
        canon_b_path=name_to_path[canon_b],
        requested_alpha_a=float(alpha_a),
        sequence_alphas_a=sequence_alphas_a,
    )


def direction_index_for_alpha(
    alpha_dir: float,
    morphing_num: int,
    snap: bool,
    tol: float,
) -> Tuple[int, float]:
    denom = morphing_num - 1
    raw_idx = (1.0 - alpha_dir) * denom
    idx = int(round(raw_idx))
    idx = max(0, min(denom, idx))

    effective = 1.0 - idx / denom

    if abs(effective - alpha_dir) > tol and not snap:
        raise ValueError(
            f"alpha={alpha_dir:.10f} is not on the MorphAny3D grid for "
            f"morphing_num={morphing_num}. Nearest effective alpha is "
            f"{effective:.10f} at idx={idx}. Use --snap-alpha 1 or choose "
            "a finer --alpha-denom."
        )

    return idx, effective


def optimize_direction_plan_for_pair(
    requests: Sequence[PairRequest],
    morphing_num: int,
    snap: bool,
    tol: float,
) -> List[PlannedTarget]:
    """
    For one canonical pair A+B, choose whether each target is cheaper to generate
    as A->B or B->A.

    MorphAny3D cache is sequential, so cost is max idx per direction.
    """
    rows = []

    for req in requests:
        a = req.requested_alpha_a

        f_idx, f_eff_dir_alpha = direction_index_for_alpha(a, morphing_num, snap, tol)
        r_idx, r_eff_dir_alpha = direction_index_for_alpha(1.0 - a, morphing_num, snap, tol)

        rows.append(
            (
                req,
                f_idx,
                f_eff_dir_alpha,
                f_eff_dir_alpha,
                r_idx,
                r_eff_dir_alpha,
                1.0 - r_eff_dir_alpha,
            )
        )

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
                feasible = all(
                    (f_idx <= max_f) or (r_idx <= max_r)
                    for _, f_idx, _, _, r_idx, _, _ in interior
                )

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
                planned.append(
                    PlannedTarget(
                        req,
                        "forward",
                        f_idx,
                        f_eff_dir_alpha,
                        f_eff_a,
                        req.sequence_alphas_a,
                    )
                )
            elif can_r:
                planned.append(
                    PlannedTarget(
                        req,
                        "reverse",
                        r_idx,
                        r_eff_dir_alpha,
                        r_eff_a,
                        tuple(1.0 - alpha for alpha in req.sequence_alphas_a),
                    )
                )
            else:
                raise RuntimeError("Internal direction planning error")

    for req, f_idx, f_eff_dir_alpha, f_eff_a, r_idx, r_eff_dir_alpha, r_eff_a in endpoints:
        if req.requested_alpha_a >= 0.5:
            planned.append(PlannedTarget(req, "endpoint_a", 0, 1.0, 1.0, req.sequence_alphas_a))
        else:
            planned.append(PlannedTarget(req, "endpoint_b", 0, 1.0, 0.0, req.sequence_alphas_a))

    planned.sort(
        key=lambda x: (
            x.split,
            x.pair_name,
            x.effective_alpha_a,
            x.direction,
            x.morphing_idx,
        )
    )

    return planned


def plan_all_targets(
    requests: Sequence[PairRequest],
    morphing_num: int,
    snap: bool,
    tol: float,
) -> List[PlannedTarget]:
    del morphing_num, snap, tol
    planned = [
        PlannedTarget(
            request=req,
            direction="forward",
            morphing_idx=req.sequence_idx,
            alpha_dir=req.requested_alpha_a,
            effective_alpha_a=req.requested_alpha_a,
            sequence_alphas_dir=req.sequence_alphas_a,
        )
        for req in requests
    ]
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


def cleanup_old_tfsa_cache(tfsa_cache: Optional[dict], old_idx: int) -> None:
    if old_idx <= 0 or not tfsa_cache:
        return

    for key in list(tfsa_cache):
        if isinstance(key, tuple) and len(key) >= 2 and key[1] == old_idx:
            del tfsa_cache[key]


@torch.no_grad()
def run_one_morphany3d_step(
    pipeline,
    src_cond: dict,
    tar_cond: dict,
    work_cache: Path,
    seed: int,
    morphing_idx: int,
    morphing_num: int,
    alpha_dir: float,
    tfsa_alpha: float,
    sparse_sampler_params: dict,
    slat_sampler_params: dict,
    ss_mca_flag: bool = True,
    slat_mca_flag: bool = True,
    slat_tfsa_flag: bool = True,
    ss_tfsa_flag: bool = True,
    tfsa_cache: Optional[dict] = None,
):
    src_cond = tree_to_device(src_cond, pipeline.device)
    tar_cond = tree_to_device(tar_cond, pipeline.device)

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
        "morphing_num": int(morphing_num),
        "tfsa_cache_idx": int(morphing_idx - 1),
        "tfsa_alpha": float(tfsa_alpha),
        "tfsa_cache": tfsa_cache,
    }

    coords, voxels, ss_latent = pipeline.sample_sparse_structure_morphing(
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

    return ss_latent, slat, voxels


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
        "alpha_sequence_index": int(planned.morphing_idx),
        "sequence_intermediates": MORPH_SEQUENCE_STEPS,
        "sequence_alphas": [float(alpha) for alpha in req.sequence_alphas_a],
        "src1_dir": f"assets/{req.canon_a}",
        "src2_dir": f"assets/{req.canon_b}",
        "target_dir": target_rel,
        "src1_slat_feats": f"assets/{req.canon_a}/slat_feats.pt",
        "src1_slat_coords": f"assets/{req.canon_a}/slat_coords.pt",
        "src2_slat_feats": f"assets/{req.canon_b}/slat_feats.pt",
        "src2_slat_coords": f"assets/{req.canon_b}/slat_coords.pt",
        "target_slat_feats": f"{target_rel}/slat_feats.pt",
        "target_slat_coords": f"{target_rel}/slat_coords.pt",
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
    lock_path = output_dir / ".locks" / f"asset_{asset_name}.lock"

    with file_lock(lock_path):
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
            extra_manifest={
                "name": asset_name,
                "image": str(asset_path),
                "seed": seed,
                "kind": "source_asset",
            },
        )
        del ss_latent, slat
        torch.cuda.empty_cache()
        return cond


def group_by_pair_and_direction(
    planned: Sequence[PlannedTarget],
) -> Dict[Tuple[str, str, str], List[PlannedTarget]]:
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


def write_metadata_files(
    output_dir: Path,
    planned: Sequence[PlannedTarget],
    splits: Dict[str, List[str]],
    config: dict,
) -> None:
    ready_entries = []
    missing = 0

    for item in planned:
        out_dir = output_dir / "targets" / item.target_name
        if latent_dir_ready(out_dir):
            ready_entries.append(build_metadata_entry(item))
        else:
            missing += 1

    ready_entries.sort(
        key=lambda e: (
            e["split"],
            e["src_1"],
            e["src_2"],
            float(e["alpha"]),
        )
    )

    atomic_json_save(ready_entries, output_dir / "metadata.json")

    for split in SPLIT_ORDER:
        split_entries = [e for e in ready_entries if e["split"] == split]
        atomic_json_save(split_entries, output_dir / f"metadata_{split}.json")

    split_asset_sets = {k: sorted(v) for k, v in splits.items()}

    atomic_json_save(split_asset_sets, output_dir / "split_assets.json")

    split_payload = {
        "assets": split_asset_sets,
        "counts": {k: len(v) for k, v in split_asset_sets.items()},
        "metadata_counts": {
            k: sum(1 for e in ready_entries if e["split"] == k)
            for k in SPLIT_ORDER
        },
        "missing_planned_targets": missing,
        "config": config,
    }

    atomic_json_save(split_payload, output_dir / "splits.json")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate split-disjoint MorphAny3D alpha dataset."
    )

    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-device", default=0)

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-strategy", choices=["random", "sorted"], default="random")

    parser.add_argument(
        "--pairs-per-asset",
        type=int,
        default=50,
        help="<=0 means all unordered pairs inside each split.",
    )

    parser.add_argument(
        "--target-train-samples",
        type=int,
        default=None,
        help="If set, plan exactly this many generated samples for the train split.",
    )
    parser.add_argument(
        "--target-val-samples",
        type=int,
        default=None,
        help="If set, plan exactly this many generated samples for the val split.",
    )
    parser.add_argument(
        "--target-test-samples",
        type=int,
        default=None,
        help="If set, plan exactly this many generated samples for the test split.",
    )
    parser.add_argument(
        "--target-total-samples",
        type=int,
        default=None,
        help=(
            "If set, split this total target count across train/val/test ratios. "
            "Must be divisible by 3 because each pair contributes exactly 3 targets."
        ),
    )

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Deprecated. Alpha is now generated as a jittered 5-step MorphAny3D "
            "sequence and 3 steps are kept per pair."
        ),
    )

    parser.add_argument(
        "--random-alphas-per-pair",
        type=int,
        default=3,
        help="Deprecated compatibility flag; must remain 3.",
    )

    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.01,
        help="Minimum allowed jittered alpha for src_1.",
    )

    parser.add_argument(
        "--alpha-max",
        type=float,
        default=0.99,
        help="Maximum allowed jittered alpha for src_1.",
    )

    parser.add_argument(
        "--alpha-denom",
        type=int,
        default=100,
        help="Deprecated compatibility flag; alpha is no longer snapped to a grid.",
    )

    parser.add_argument(
        "--morphing-num",
        type=int,
        default=None,
        help="Deprecated compatibility flag; with 5 intermediates the value must be 7.",
    )

    parser.add_argument("--snap-alpha", type=int, choices=[0, 1], default=0)
    parser.add_argument("--alpha-tol", type=float, default=1e-7)
    parser.add_argument(
        "--alpha-jitter",
        type=float,
        default=DEFAULT_ALPHA_JITTER,
        help=(
            "Per-pair jitter around the ideal 5-step alpha sequence "
            "(5/6, 4/6, 3/6, 2/6, 1/6)."
        ),
    )
    parser.add_argument("--tfsa-alpha", type=float, default=0.8)
    parser.add_argument("--disable-tfsa", type=int, choices=[0, 1], default=0)

    parser.add_argument(
        "--sparse-sampler-params",
        type=str,
        default=None,
        help='JSON, e.g. \'{"steps": 25}\'',
    )

    parser.add_argument(
        "--slat-sampler-params",
        type=str,
        default=None,
        help='JSON, e.g. \'{"steps": 25}\'',
    )
    parser.add_argument(
        "--work-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for temporary MorphAny3D TFSA caches. Defaults to "
            "MORPHANY3D_WORK_CACHE_DIR, then TMPDIR, then output_dir/.tmp_morphany3d_cache. "
            "Use node-local storage for speed."
        ),
    )
    parser.add_argument(
        "--tfsa-cache-mode",
        choices=["file", "memory"],
        default="file",
        help=(
            "Where MorphAny3D stores TFSA attention caches. 'file' uses the local "
            "work cache and is safer for multi-process jobs; 'memory' avoids "
            "torch.load/save but can use a lot of RAM."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build splits and print the plan; do not import TRELLIS or generate latents.",
    )
    parser.add_argument(
        "--print-plan-details",
        type=int,
        choices=[0, 1],
        default=0,
        help="Print per-pair direction plans. Disabled by default because exact datasets are large.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of independent generation batches. Use with Slurm array.",
    )

    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Current batch index in [0, num_batches-1]. Use with Slurm array.",
    )

    args = parser.parse_args(argv)
    if args.num_batches < 1:
        raise ValueError("--num-batches must be >= 1")

    if not (0 <= args.batch_index < args.num_batches):
        raise ValueError(
            f"--batch-index must be in [0, {args.num_batches - 1}], "
            f"got {args.batch_index}"
        )


    if args.alpha_denom <= 0:
        raise ValueError("--alpha-denom must be > 0")

    if args.alphas is not None:
        raise ValueError(
            "--alphas is no longer supported for dataset generation. "
            "Alpha targets now come from a 5-step jittered MorphAny3D sequence."
        )

    if args.random_alphas_per_pair != TARGETS_PER_PAIR:
        raise ValueError(
            f"--random-alphas-per-pair is fixed at {TARGETS_PER_PAIR}; "
            "the dataset keeps exactly 3 targets per pair."
        )

    if args.snap_alpha != 0:
        raise ValueError("--snap-alpha is deprecated; alpha is now a real jittered blend value.")

    morphing_num = MORPH_SEQUENCE_STEPS + 2
    if args.morphing_num is not None and int(args.morphing_num) != morphing_num:
        raise ValueError(
            f"--morphing-num must be {morphing_num}, because the generator "
            f"creates exactly {MORPH_SEQUENCE_STEPS} intermediate steps."
        )

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
    pair_alphas: Dict[str, Dict[str, List[float]]] = {split: {} for split in SPLIT_ORDER}
    pair_sequence_alphas: Dict[str, Dict[str, List[float]]] = {split: {} for split in SPLIT_ORDER}
    pair_selected_indices: Dict[str, Dict[str, List[int]]] = {split: {} for split in SPLIT_ORDER}
    requests: List[PairRequest] = []

    split_pair_seeds = {
        "train": args.seed + 101,
        "val": args.seed + 202,
        "test": args.seed + 303,
    }
    exact_target_counts = target_sample_counts_from_args(args)
    exact_mode = has_exact_sample_targets(exact_target_counts)

    for split in SPLIT_ORDER:
        pairs = build_pairs_with_cap(
            splits[split],
            args.pairs_per_asset,
            seed=split_pair_seeds[split],
        )

        target_count = exact_target_counts[split]
        if exact_mode and target_count is not None:
            pair_counts = choose_pairs_for_exact_sample_count(
                split=split,
                pairs=pairs,
                target_count=target_count,
                samples_per_pair=TARGETS_PER_PAIR,
                seed=split_pair_seeds[split] + 1009,
            )
        else:
            pair_counts = [(pair, TARGETS_PER_PAIR) for pair in pairs]

        split_pairs[split] = [pair for pair, _count in pair_counts]

        for (a_name, b_name), alpha_count in pair_counts:
            sequence_alphas = jittered_sequence_alphas_for_pair(
                seed=args.seed,
                split=split,
                a_name=a_name,
                b_name=b_name,
                steps=MORPH_SEQUENCE_STEPS,
                jitter=args.alpha_jitter,
                alpha_min=args.alpha_min,
                alpha_max=args.alpha_max,
            )
            selected_indices = selected_sequence_indices_for_pair(
                seed=args.seed,
                split=split,
                a_name=a_name,
                b_name=b_name,
                steps=MORPH_SEQUENCE_STEPS,
                count=alpha_count,
            )
            alphas_for_pair = [sequence_alphas[idx - 1] for idx in selected_indices]

            pair_key = f"{a_name}+{b_name}"
            pair_sequence_alphas[split][pair_key] = sequence_alphas
            pair_selected_indices[split][pair_key] = selected_indices
            pair_alphas[split][pair_key] = alphas_for_pair

            for sequence_idx, alpha in zip(selected_indices, alphas_for_pair):
                requests.append(
                    canonicalize_request(
                        split,
                        a_name,
                        b_name,
                        int(sequence_idx),
                        float(alpha),
                        sequence_alphas,
                        name_to_path,
                    )
                )

    all_planned = plan_all_targets(
        requests,
        morphing_num=morphing_num,
        snap=bool(args.snap_alpha),
        tol=float(args.alpha_tol),
    )

    if exact_mode:
        planned_counts = {
            split: sum(1 for item in all_planned if item.split == split)
            for split in SPLIT_ORDER
        }
        for split, requested in exact_target_counts.items():
            if requested is not None and planned_counts[split] != requested:
                raise RuntimeError(
                    f"Exact planning mismatch for {split}: requested {requested}, "
                    f"planned {planned_counts[split]}. Check target counts and pair planning."
                )

    if args.num_batches > 1:
        planned = [
            item for item in all_planned
            if stable_mod_key(f"{item.split}|{item.pair_name}", args.num_batches)
            == args.batch_index
        ]
    else:
        planned = all_planned

    print(
        f"Batch selection: batch_index={args.batch_index} "
        f"num_batches={args.num_batches} "
        f"selected_targets={len(planned)} "
        f"total_targets={len(all_planned)}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_base = args.work_cache_dir
    if cache_base is None:
        env_cache = os.environ.get("MORPHANY3D_WORK_CACHE_DIR") or os.environ.get("TMPDIR")
        cache_base = Path(env_cache) if env_cache else args.output_dir / ".tmp_morphany3d_cache"

    work_root = cache_base / f"morphflow_batch_{args.batch_index}_pid_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    work_root.mkdir(parents=True, exist_ok=True)

    print("=== MorphAny3D split-disjoint dataset ===")
    print(f"assets_dir: {args.assets_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"assets total: {len(name_to_path)}")
    print(f"morphing_num: {morphing_num}")
    print(f"sequence_intermediates: {MORPH_SEQUENCE_STEPS}")
    print(f"targets_per_pair: {TARGETS_PER_PAIR}")
    print(f"alpha_jitter: {args.alpha_jitter}")
    print(f"alpha_range: [{args.alpha_min}, {args.alpha_max}]")
    print(f"work_cache_root: {work_root}")

    if exact_mode:
        print(f"target sample counts: {exact_target_counts}")

    for split in SPLIT_ORDER:
        planned_samples = sum(len(v) for v in pair_alphas[split].values())
        print(
            f"split {split:5s}: "
            f"assets={len(splits[split])} "
            f"pairs={len(split_pairs[split])} "
            f"samples={planned_samples} "
            f"sequence_steps_generated={len(split_pairs[split]) * MORPH_SEQUENCE_STEPS}"
        )

    if args.print_plan_details == 1:
        for (split, pair_name, direction), items in group_by_pair_and_direction(planned).items():
            if direction.startswith("endpoint"):
                continue

            print(
                f"plan {split:5s} {pair_name:40s} {direction:7s}: "
                f"targets={len(items)} "
                f"max_idx={max(x.morphing_idx for x in items)} "
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
        "target_total_samples": args.target_total_samples,
        "alphas": None,
        "targets_per_pair": TARGETS_PER_PAIR,
        "sequence_intermediates": MORPH_SEQUENCE_STEPS,
        "alpha_jitter": args.alpha_jitter,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_generation": "jittered_5_step_sequence_keep_3",
        "target_sample_counts": exact_target_counts,
        "alpha_denom": None,
        "morphing_num": morphing_num,
        "snap_alpha": False,
        "tfsa_alpha": args.tfsa_alpha,
        "tfsa_enabled": args.disable_tfsa == 0,
        "tfsa_cache_mode": args.tfsa_cache_mode,
        "mca_enabled": True,
        "work_cache_dir": str(cache_base),
    }

    atomic_json_save(pair_alphas, args.output_dir / "pair_alphas.json")
    atomic_json_save(pair_sequence_alphas, args.output_dir / "pair_sequence_alphas.json")
    atomic_json_save(pair_selected_indices, args.output_dir / "pair_selected_indices.json")

    if args.dry_run:
        dry_payload = {
            "assets": {k: sorted(v) for k, v in splits.items()},
            "counts": {k: len(v) for k, v in splits.items()},
            "planned_pairs": {k: len(v) for k, v in split_pairs.items()},
            "planned_samples": {
                k: len([p for p in planned if p.split == k])
                for k in SPLIT_ORDER
            },
            "pair_alphas": pair_alphas,
            "pair_sequence_alphas": pair_sequence_alphas,
            "pair_selected_indices": pair_selected_indices,
            "config": config,
        }

        atomic_json_save(dry_payload, args.output_dir / "splits.dry_run.json")
        print(f"Dry run written: {args.output_dir / 'splits.dry_run.json'}")
        return

    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model_id)
    pipeline.cuda()

    cond_cache: Dict[str, dict] = {}

    required_assets = sorted(
        {p.request.canon_a for p in planned}
        | {p.request.canon_b for p in planned}
    )

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

    # Endpoint targets are copied only to keep the loader simple.
    for item in [p for p in planned if p.direction.startswith("endpoint")]:
        src_asset = item.request.canon_a if item.direction == "endpoint_a" else item.request.canon_b
        src_dir = args.output_dir / "assets" / src_asset
        tgt_dir = args.output_dir / "targets" / item.target_name

        if not latent_dir_ready(tgt_dir):
            tgt_dir.mkdir(parents=True, exist_ok=True)

            for fname in (
                "slat_feats.pt",
                "slat_coords.pt",
                "ss_latent.pt",
                "structured_latent.pt",
                "occupancy.pt",
            ):
                src_file = src_dir / fname
                if src_file.is_file():
                    shutil.copy2(src_file, tgt_dir / fname)

            atomic_json_save(
                {
                    "kind": "endpoint_copy",
                    "copied_from": f"assets/{src_asset}",
                    "alpha": item.effective_alpha_a,
                    "split": item.split,
                },
                tgt_dir / "manifest.json",
            )

    grouped = group_by_pair_and_direction(
        [p for p in planned if p.direction in ("forward", "reverse")]
    )

    ss_tfsa_flag = args.disable_tfsa == 0
    slat_tfsa_flag = args.disable_tfsa == 0

    for (split, pair_name, direction), items in grouped.items():
        missing_items = [
            p for p in items
            if not latent_dir_ready(args.output_dir / "targets" / p.target_name)
        ]

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

        sequence_alphas_dir = tuple(float(alpha) for alpha in items[0].sequence_alphas_dir)
        if len(sequence_alphas_dir) != MORPH_SEQUENCE_STEPS:
            raise RuntimeError(
                f"Expected {MORPH_SEQUENCE_STEPS} sequence alphas for {split}/{pair_name}, "
                f"got {len(sequence_alphas_dir)}"
            )

        max_idx = MORPH_SEQUENCE_STEPS

        print(
            f"[pair] {split} {pair_name} {direction}: "
            f"running full sequence idx 1..{max_idx}, saving {len(missing_items)} target(s)"
        )

        tfsa_cache = {} if args.disable_tfsa == 0 and args.tfsa_cache_mode == "memory" else None

        for idx in range(1, max_idx + 1):
            alpha_dir = sequence_alphas_dir[idx - 1]

            ss_latent, slat, voxels = run_one_morphany3d_step(
                pipeline,
                src_cond=src_cond,
                tar_cond=tar_cond,
                work_cache=work_cache,
                seed=args.seed,
                morphing_idx=idx,
                morphing_num=morphing_num,
                alpha_dir=alpha_dir,
                tfsa_alpha=args.tfsa_alpha,
                sparse_sampler_params=sparse_sampler_params,
                slat_sampler_params=slat_sampler_params,
                ss_mca_flag=True,
                slat_mca_flag=True,
                ss_tfsa_flag=ss_tfsa_flag,
                slat_tfsa_flag=slat_tfsa_flag,
                tfsa_cache=tfsa_cache,
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
                            "alpha_sequence_index": target.morphing_idx,
                            "sequence_intermediates": MORPH_SEQUENCE_STEPS,
                            "sequence_alphas_src_1": [
                                float(alpha) for alpha in target.request.sequence_alphas_a
                            ],
                            "seed": args.seed,
                            "tfsa_alpha": args.tfsa_alpha,
                            "tfsa_enabled": args.disable_tfsa == 0,
                            "tfsa_cache_mode": args.tfsa_cache_mode,
                            "mca_enabled": True,
                        },
                        occupancy=voxels,
                    )

                    print(
                        f"  saved {split}/{target.target_name} "
                        f"idx={idx} dir_alpha={alpha_dir:.6f}"
                    )

            del ss_latent, slat, voxels
            cleanup_old_index(work_cache, idx - 1)
            cleanup_old_tfsa_cache(tfsa_cache, idx - 1)

        shutil.rmtree(work_cache, ignore_errors=True)
        torch.cuda.empty_cache()

    # In batch/array mode, each task writes metadata by scanning all planned targets.
    # The last successful task will leave complete metadata.json.
    write_metadata_files(args.output_dir, all_planned, splits, config)
    shutil.rmtree(work_root, ignore_errors=True)

    print(f"Done. Metadata: {args.output_dir / 'metadata.json'}")
    print(f"Train metadata: {args.output_dir / 'metadata_train.json'}")
    print(f"Val metadata:   {args.output_dir / 'metadata_val.json'}")
    print(f"Test metadata:  {args.output_dir / 'metadata_test.json'}")


if __name__ == "__main__":
    main()
