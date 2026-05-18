#!/usr/bin/env python3
"""
Generate an alpha-coherent MorphAny3D distillation dataset.

Semantics
---------
For each unordered pair (A, B), metadata alpha always means:
    target = alpha * A + (1 - alpha) * B
where A and B are the canonical source names sorted lexicographically.

MorphAny3D does not generate arbitrary continuous alpha values in its official
morphing loop. It generates frames on the discrete grid:
    alpha_i = 1 - i / (morphing_num - 1),  i = 1 .. morphing_num - 2
and temporal self-attention reuses caches from i-1. This script emulates that
loop, stops at the last needed index, and deletes transient attention caches.

Saved files only:
  assets/<name>/ss_latent.pt
  assets/<name>/structured_latent.pt
  assets/<name>/occupancy.pt
  targets/<A>+<B>/alpha_<...>/ss_latent.pt
  targets/<A>+<B>/alpha_<...>/structured_latent.pt
  targets/<A>+<B>/alpha_<...>/occupancy.pt
  metadata.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Set these before importing TRELLIS / torch modules.
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from PIL import Image


def seed_everything(seed: int) -> None:
    import random

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


def resolve_image_path(assets_dir: Path, item: str) -> Path:
    path = Path(item)
    if path.is_file():
        return path
    candidate = assets_dir / item
    if candidate.is_file():
        return candidate
    if candidate.suffix == "":
        for suffix in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = assets_dir / f"{item}{suffix}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Cannot resolve image: {item!r} under {assets_dir}")


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
    # Store a clean binary grid when the output is [1, 1, D, H, W].
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


@torch.no_grad()
def encode_image_condition(pipeline, image_path: Path) -> dict:
    image = Image.open(image_path)
    processed = pipeline.preprocess_image(image)
    return pipeline.get_cond([processed])


@torch.no_grad()
def sample_endpoint_latents(pipeline, cond: dict, seed: int, sparse_sampler_params: dict, slat_sampler_params: dict):
    """Equivalent to pipeline.run(image) but returns ss latent, occupancy coords and SLAT, without decoding mesh/GS/RF."""
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
    # Original user-facing pair.
    src1_name: str
    src2_name: str
    src1_path: Path
    src2_path: Path
    requested_alpha_src1: float

    # Canonical pair used for storage / training metadata.
    canon_a: str
    canon_b: str
    canon_a_path: Path
    canon_b_path: Path
    requested_alpha_a: float


@dataclass(frozen=True)
class PlannedTarget:
    request: PairRequest
    direction: str  # "forward" means A->B, "reverse" means B->A.
    morphing_idx: int
    alpha_dir: float
    effective_alpha_a: float

    @property
    def pair_name(self) -> str:
        return f"{self.request.canon_a}+{self.request.canon_b}"

    @property
    def target_name(self) -> str:
        return f"{self.pair_name}/alpha_{alpha_slug(self.effective_alpha_a)}"


def canonicalize_request(assets_dir: Path, src1: str, src2: str, alpha_src1: float) -> PairRequest:
    if not (0.0 <= alpha_src1 <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha_src1}")

    p1 = resolve_image_path(assets_dir, src1)
    p2 = resolve_image_path(assets_dir, src2)
    n1 = safe_name(p1.name)
    n2 = safe_name(p2.name)
    if n1 == n2:
        raise ValueError(f"Pair uses the same asset twice: {src1}, {src2}")

    if n1 <= n2:
        a_name, b_name = n1, n2
        a_path, b_path = p1, p2
        alpha_a = alpha_src1
    else:
        a_name, b_name = n2, n1
        a_path, b_path = p2, p1
        alpha_a = 1.0 - alpha_src1

    return PairRequest(
        src1_name=n1,
        src2_name=n2,
        src1_path=p1,
        src2_path=p2,
        requested_alpha_src1=float(alpha_src1),
        canon_a=a_name,
        canon_b=b_name,
        canon_a_path=a_path,
        canon_b_path=b_path,
        requested_alpha_a=float(alpha_a),
    )


def direction_index_for_alpha(alpha_dir: float, morphing_num: int, snap: bool, tol: float) -> Tuple[int, float]:
    """Return MorphAny3D index i and effective direction alpha 1 - i/(N-1)."""
    denom = morphing_num - 1
    raw_idx = (1.0 - alpha_dir) * denom
    idx = int(round(raw_idx))
    idx = max(0, min(denom, idx))
    effective = 1.0 - idx / denom
    if abs(effective - alpha_dir) > tol and not snap:
        raise ValueError(
            f"alpha={alpha_dir:.10f} is not on the MorphAny3D grid for morphing_num={morphing_num}. "
            f"Nearest effective alpha is {effective:.10f} at idx={idx}. "
            "Use --snap-alpha 1 to store the snapped effective alpha, or choose a finer --alpha-denom."
        )
    return idx, effective


def optimize_direction_plan(requests: Sequence[PairRequest], morphing_num: int, snap: bool, tol: float) -> List[PlannedTarget]:
    """
    Choose A->B or B->A per alpha to minimize total sequential TFSA steps.

    Cost of a direction is max morphing_idx generated in that direction. A target
    with canonical alpha a can be generated either:
      forward A->B with direction alpha a and idx (1-a)*(N-1), or
      reverse B->A with direction alpha 1-a and idx a*(N-1).
    """
    if not requests:
        return []

    rows = []
    for req in requests:
        a = req.requested_alpha_a
        f_idx, f_eff_dir_alpha = direction_index_for_alpha(a, morphing_num, snap, tol)
        r_idx, r_eff_dir_alpha = direction_index_for_alpha(1.0 - a, morphing_num, snap, tol)
        # Convert effective direction alpha back to canonical-alpha-of-A.
        f_eff_a = f_eff_dir_alpha
        r_eff_a = 1.0 - r_eff_dir_alpha
        rows.append((req, f_idx, f_eff_dir_alpha, f_eff_a, r_idx, r_eff_dir_alpha, r_eff_a))

    # Endpoints do not need morphing path. Keep them as zero-cost planned items.
    interior_rows = [r for r in rows if r[1] not in (0, morphing_num - 1) and r[4] not in (0, morphing_num - 1)]
    endpoint_rows = [r for r in rows if r not in interior_rows]

    candidate_f = sorted({0} | {r[1] for r in interior_rows})
    candidate_r = sorted({0} | {r[4] for r in interior_rows})

    best: Optional[Tuple[int, int, int]] = None
    for max_f in candidate_f:
        for max_r in candidate_r:
            feasible = all((f_idx <= max_f) or (r_idx <= max_r) for _, f_idx, _, _, r_idx, _, _ in interior_rows)
            if not feasible:
                continue
            # Tiny tie-breaker: prefer fewer total directions if costs equal.
            num_dirs = int(max_f > 0) + int(max_r > 0)
            score = max_f + max_r
            candidate = (score, num_dirs, max_f, max_r)
            if best is None or candidate < best:
                best = candidate

    planned: List[PlannedTarget] = []

    if best is not None:
        _, _, best_max_f, best_max_r = best
        for req, f_idx, f_eff_dir_alpha, f_eff_a, r_idx, r_eff_dir_alpha, r_eff_a in interior_rows:
            can_f = f_idx <= best_max_f
            can_r = r_idx <= best_max_r
            if can_f and (not can_r or f_idx <= r_idx):
                planned.append(PlannedTarget(req, "forward", f_idx, f_eff_dir_alpha, f_eff_a))
            elif can_r:
                planned.append(PlannedTarget(req, "reverse", r_idx, r_eff_dir_alpha, r_eff_a))
            else:
                # Should not happen because best is feasible.
                raise RuntimeError("Internal planning error")

    for req, f_idx, f_eff_dir_alpha, f_eff_a, r_idx, r_eff_dir_alpha, r_eff_a in endpoint_rows:
        # alpha=1 => A itself, alpha=0 => B itself. Store as a target entry that points to the asset.
        if req.requested_alpha_a >= 0.5:
            planned.append(PlannedTarget(req, "endpoint_a", 0, 1.0, 1.0))
        else:
            planned.append(PlannedTarget(req, "endpoint_b", 0, 1.0, 0.0))

    # Deterministic output order.
    planned.sort(key=lambda x: (x.pair_name, x.effective_alpha_a, x.direction, x.morphing_idx))
    return planned


def cleanup_transient_cache(cache_dir: Path, keep_idx: Optional[int] = None) -> None:
    """Delete MorphAny3D TFSA/coords cache files, optionally keeping exactly one index for the next step."""
    if not cache_dir.exists():
        return

    keep_fragments = []
    if keep_idx is not None:
        keep_fragments = [
            f"morphing{keep_idx}_",
            f"morphing{keep_idx}.",
            f"morphing{keep_idx}",
        ]

    for path in cache_dir.iterdir():
        if not path.is_file():
            continue
        if keep_idx is not None and any(fragment in path.name for fragment in keep_fragments):
            continue
        path.unlink(missing_ok=True)


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
    """One official MorphAny3D morphing frame, but returns latents and does not decode videos/meshes."""
    seed_everything(seed)
    work_cache.mkdir(parents=True, exist_ok=True)

    morphing_params = {
        "save_cache_path": str(work_cache),
        "init_morphing_flag": False,
        "ss_mca_flag": bool(ss_mca_flag),
        "slat_mca_flag": bool(slat_mca_flag),
        "ss_tfsa_flag": bool(ss_tfsa_flag),
        "slat_tfsa_flag": bool(slat_tfsa_flag),
        # Keep this False for latent/occupancy coherence. Official OC rotates decoded voxels/caches,
        # not the saved z_s latent itself.
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


def load_pair_requests(args) -> List[Tuple[str, str, float]]:
    raw: List[Tuple[str, str, float]] = []

    if args.pair is not None:
        if len(args.pair) != 2:
            raise ValueError("--pair requires exactly two values: SRC1 SRC2")
        for alpha in args.alphas:
            raw.append((args.pair[0], args.pair[1], float(alpha)))

    if args.pairs_json is not None:
        with Path(args.pairs_json).open("r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if isinstance(item, dict):
                src1 = item.get("src1") or item.get("src_1") or item.get("a")
                src2 = item.get("src2") or item.get("src_2") or item.get("b")
                alphas = item.get("alphas")
                if alphas is None and "alpha" in item:
                    alphas = [item["alpha"]]
            elif isinstance(item, (list, tuple)) and len(item) == 3:
                src1, src2, alphas = item
                if not isinstance(alphas, (list, tuple)):
                    alphas = [alphas]
            else:
                raise ValueError(f"Unsupported pairs-json entry: {item!r}")
            if src1 is None or src2 is None or alphas is None:
                raise ValueError(f"Invalid pairs-json entry: {item!r}")
            for alpha in alphas:
                raw.append((str(src1), str(src2), float(alpha)))

    if not raw:
        raise ValueError("Provide either --pair SRC1 SRC2 --alphas ... or --pairs-json pairs.json")

    return raw


def load_existing_metadata(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"metadata must be a JSON list: {path}")
    return data


def metadata_key(entry: dict) -> Tuple[str, str, str]:
    return (
        str(entry.get("src_1")),
        str(entry.get("src_2")),
        f"{float(entry.get('alpha')):.10f}",
    )


def append_metadata(metadata_path: Path, new_entries: Iterable[dict]) -> None:
    existing = load_existing_metadata(metadata_path)
    seen = {metadata_key(e) for e in existing if "alpha" in e}
    changed = False
    for entry in new_entries:
        key = metadata_key(entry)
        if key in seen:
            continue
        existing.append(entry)
        seen.add(key)
        changed = True
    if changed or not metadata_path.exists():
        existing.sort(key=lambda e: (e.get("src_1", ""), e.get("src_2", ""), float(e.get("alpha", 0.0))))
        atomic_json_save(existing, metadata_path)


def build_metadata_entry(planned: PlannedTarget, output_dir: Path) -> dict:
    req = planned.request
    target_rel = f"targets/{planned.target_name}"
    return {
        "src_1": req.canon_a,
        "src_2": req.canon_b,
        "target": planned.target_name,
        "alpha": float(planned.effective_alpha_a),
        "alpha_requested": float(req.requested_alpha_a),
        "alpha_definition": "alpha is the fraction of src_1 in the canonical pair; target = alpha*src_1 + (1-alpha)*src_2",
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
        extra_manifest={
            "name": asset_name,
            "image": str(asset_path),
            "seed": seed,
            "kind": "source_asset",
        },
    )
    torch.cuda.empty_cache()
    return cond


def group_by_pair_and_direction(planned: Sequence[PlannedTarget]) -> Dict[Tuple[str, str], List[PlannedTarget]]:
    grouped: Dict[Tuple[str, str], List[PlannedTarget]] = {}
    for item in planned:
        grouped.setdefault((item.pair_name, item.direction), []).append(item)
    for key in grouped:
        grouped[key].sort(key=lambda x: x.morphing_idx)
    return grouped


def parse_sampler_params(json_text: Optional[str]) -> dict:
    if not json_text:
        return {}
    return json.loads(json_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a coherent MorphAny3D alpha dataset.")
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair", nargs=2, default=None, metavar=("SRC1", "SRC2"))
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.23, 0.64, 0.88])
    parser.add_argument("--pairs-json", type=Path, default=None)
    parser.add_argument("--model-id", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--alpha-denom", type=int, default=100, help="Grid denominator; 100 gives alpha step 0.01 and morphing_num=101.")
    parser.add_argument("--morphing-num", type=int, default=None, help="Overrides --alpha-denom by setting MorphAny3D grid size directly.")
    parser.add_argument("--snap-alpha", type=int, choices=[0, 1], default=0, help="If 1, snap requested alpha to nearest grid and store effective alpha.")
    parser.add_argument("--alpha-tol", type=float, default=1e-7)
    parser.add_argument("--tfsa-alpha", type=float, default=0.8)
    parser.add_argument("--disable-tfsa", type=int, choices=[0, 1], default=0, help="Use direct MCA alpha only; not the official TFSA path.")
    parser.add_argument("--sparse-sampler-params", type=str, default=None, help='JSON, e.g. \'{"steps": 25}\'')
    parser.add_argument("--slat-sampler-params", type=str, default=None, help='JSON, e.g. \'{"steps": 25}\'')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    if args.alpha_denom <= 0:
        raise ValueError("--alpha-denom must be > 0")
    morphing_num = int(args.morphing_num or (args.alpha_denom + 1))
    if morphing_num < 3:
        raise ValueError("morphing_num must be >= 3")

    sparse_sampler_params = parse_sampler_params(args.sparse_sampler_params)
    slat_sampler_params = parse_sampler_params(args.slat_sampler_params)

    raw_requests = load_pair_requests(args)
    requests = [canonicalize_request(args.assets_dir, src1, src2, alpha) for src1, src2, alpha in raw_requests]

    # Merge exact duplicates after symmetry normalization.
    dedup: Dict[Tuple[str, str, str], PairRequest] = {}
    for req in requests:
        key = (req.canon_a, req.canon_b, f"{req.requested_alpha_a:.10f}")
        dedup[key] = req
    requests = list(dedup.values())

    planned = optimize_direction_plan(
        requests,
        morphing_num=morphing_num,
        snap=bool(args.snap_alpha),
        tol=float(args.alpha_tol),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.json"
    work_root = args.output_dir / ".tmp_morphany3d_cache"
    work_root.mkdir(parents=True, exist_ok=True)

    print("=== MorphAny3D coherent dataset ===")
    print(f"assets_dir: {args.assets_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"morphing_num: {morphing_num}  grid_step: {1.0 / (morphing_num - 1):.8f}")
    print(f"requests after symmetry de-dup: {len(requests)}")
    for (pair_name, direction), items in group_by_pair_and_direction(planned).items():
        if direction.startswith("endpoint"):
            continue
        print(
            f"plan {pair_name:40s} {direction:7s}: "
            f"targets={len(items)} max_idx={max(x.morphing_idx for x in items)} "
            f"alphas={[round(x.effective_alpha_a, 6) for x in items]}"
        )

    # Import TRELLIS after env variables are finalized.
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model_id)
    pipeline.cuda()

    cond_cache: Dict[str, dict] = {}

    # Save all source endpoints required by the planned targets.
    for req in sorted({p.request for p in planned}, key=lambda r: (r.canon_a, r.canon_b, r.requested_alpha_a)):
        ensure_asset_latents(
            pipeline,
            req.canon_a,
            req.canon_a_path,
            args.output_dir,
            args.seed,
            cond_cache,
            sparse_sampler_params,
            slat_sampler_params,
        )
        ensure_asset_latents(
            pipeline,
            req.canon_b,
            req.canon_b_path,
            args.output_dir,
            args.seed,
            cond_cache,
            sparse_sampler_params,
            slat_sampler_params,
        )

    new_meta_entries: List[dict] = []

    # Endpoints: target is already an asset; copy only if you want a physical target dir.
    # To keep consumers simple, create a small target directory with the same three files.
    for item in [p for p in planned if p.direction.startswith("endpoint")]:
        src_asset = item.request.canon_a if item.direction == "endpoint_a" else item.request.canon_b
        src_dir = args.output_dir / "assets" / src_asset
        tgt_dir = args.output_dir / "targets" / item.target_name
        if not latent_dir_ready(tgt_dir):
            tgt_dir.mkdir(parents=True, exist_ok=True)
            for fname in ("ss_latent.pt", "structured_latent.pt", "occupancy.pt"):
                shutil.copy2(src_dir / fname, tgt_dir / fname)
            atomic_json_save(
                {
                    "kind": "endpoint_copy",
                    "copied_from": f"assets/{src_asset}",
                    "alpha": item.effective_alpha_a,
                },
                tgt_dir / "manifest.json",
            )
        new_meta_entries.append(build_metadata_entry(item, args.output_dir))

    grouped = group_by_pair_and_direction([p for p in planned if p.direction in ("forward", "reverse")])

    for (pair_name, direction), items in grouped.items():
        # Skip a direction entirely if all targets already exist.
        missing_items = [p for p in items if not latent_dir_ready(args.output_dir / "targets" / p.target_name)]
        if not missing_items:
            for p in items:
                new_meta_entries.append(build_metadata_entry(p, args.output_dir))
            print(f"[pair] {pair_name} {direction}: skipped, all targets ready")
            continue

        first_req = items[0].request
        if direction == "forward":
            src_name, tar_name = first_req.canon_a, first_req.canon_b
        else:
            src_name, tar_name = first_req.canon_b, first_req.canon_a

        src_cond = cond_cache[src_name]
        tar_cond = cond_cache[tar_name]
        work_cache = work_root / f"{pair_name}__{direction}"
        shutil.rmtree(work_cache, ignore_errors=True)
        work_cache.mkdir(parents=True, exist_ok=True)

        items_by_idx: Dict[int, List[PlannedTarget]] = {}
        for p in missing_items:
            items_by_idx.setdefault(p.morphing_idx, []).append(p)

        max_idx = max(items_by_idx)
        print(f"[pair] {pair_name} {direction}: running idx 1..{max_idx}, saving {len(missing_items)} target(s)")

        ss_tfsa_flag = args.disable_tfsa == 0
        slat_tfsa_flag = args.disable_tfsa == 0

        for idx in range(1, max_idx + 1):
            # Direction alpha must match this official MorphAny3D frame.
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
                    new_meta_entries.append(build_metadata_entry(target, args.output_dir))
                    print(f"  saved {target.target_name}  idx={idx} dir_alpha={alpha_dir:.6f}")

            # Keep only the current index because the next step may read it as tfsa_cache_idx.
            cleanup_old_index(work_cache, idx - 1)
            torch.cuda.empty_cache()

        shutil.rmtree(work_cache, ignore_errors=True)

        # Add metadata for already-ready targets in this direction too.
        for p in items:
            new_meta_entries.append(build_metadata_entry(p, args.output_dir))

    append_metadata(metadata_path, new_meta_entries)
    shutil.rmtree(work_root, ignore_errors=True)
    print(f"Done. Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
