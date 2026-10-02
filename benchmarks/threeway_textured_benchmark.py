#!/usr/bin/env python3
"""Three-way MorphFlow / MorphAny3D / Interp3D benchmark.

Design goals
------------
* fixed test-set plan shared by all methods;
* MorphFlow consumes canonical source latents;
* MorphAny3D and Interp3D consume the corresponding source images;
* K intermediate states per pair;
* generation saves raw TRELLIS meshes first;
* ONE common TRELLIS texture-baking path converts every raw mesh to a textured GLB;
* existing eval_perceptual_sequence.py evaluates final textured GLBs (materials mode);
* resumable outputs and append-only progress log.

This script intentionally has several modes because the three methods live in
separate isolated Conda/Singularity environments. The companion Slurm script
runs each mode in the correct environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


METHODS = ("morphflow", "morphany3d", "interp3d")
DEFAULT_N = 100
DEFAULT_K = 10
DEFAULT_SEED = 42


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str, run_root: Optional[Path] = None) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    if run_root is not None:
        run_root.mkdir(parents=True, exist_ok=True)
        with (run_root / "progress.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload)) + "\n")


def safe_slug(value: str, limit: int = 36) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    if not value:
        value = "asset"
    return value[:limit]


def pair_dir_name(pair_id: int, src1: str, src2: str) -> str:
    return f"pair_{pair_id:04d}_{safe_slug(src1)}_to_{safe_slug(src2)}"


def alpha_dir(alpha: float) -> str:
    return f"alpha_{float(alpha):.6f}".replace(".", "p")


def canonical_alphas(k: int) -> List[float]:
    if k < 1:
        raise ValueError("K must be >= 1")
    # Canonical MorphFlow convention: alpha=1 -> src1, alpha=0 -> src2.
    values = np.linspace(1.0, 0.0, k + 2, dtype=np.float64)[1:-1]
    return [float(round(x, 10)) for x in values.tolist()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def pair_seed(base_seed: int, pair_id: int) -> int:
    return int(base_seed + pair_id * 100_000)


def step_seed(base_seed: int, pair_id: int, step_idx: int, offset: int = 0) -> int:
    return int(base_seed + pair_id * 100_000 + offset + step_idx)


def load_plan(run_root: Path) -> Dict[str, Any]:
    path = run_root / "plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing plan: {path}")
    return read_json(path)


def save_raw_mesh(mesh: Any, path: Path, metadata: Optional[Mapping[str, Any]] = None) -> None:
    import torch

    if not getattr(mesh, "success", True):
        raise RuntimeError(f"Mesh extraction failed for {path}")
    attrs = getattr(mesh, "vertex_attrs", None)
    payload: Dict[str, Any] = {
        "vertices": mesh.vertices.detach().float().cpu(),
        "faces": mesh.faces.detach().long().cpu(),
        "vertex_attrs": attrs.detach().float().cpu() if attrs is not None else None,
        "res": int(getattr(mesh, "res", 64)),
        "metadata": dict(metadata or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def decode_slat_mesh(mesh_decoder: Any, sparse_tensor_cls: Any, feats: Any, coords: Any, device: Any) -> Any:
    import torch

    feats = feats.to(device=device, dtype=torch.float32)
    coords = coords.to(device=device, dtype=torch.int32)
    if coords.ndim != 2:
        raise ValueError(f"Unexpected coords shape: {tuple(coords.shape)}")
    if coords.shape[1] == 3:
        batch = torch.zeros((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)
        coords = torch.cat([batch, coords], dim=1)
    coords = coords.clone()
    coords[:, 0] = 0
    sparse = sparse_tensor_cls(feats=feats, coords=coords)
    return mesh_decoder(sparse)[0]


def command_plan(args: argparse.Namespace) -> None:
    from data.morph_dataset import MorphingDistillDataset

    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    plan_path = run_root / "plan.json"
    if plan_path.exists() and not args.overwrite:
        log(f"PLAN already exists, keeping it: {plan_path}", run_root)
        return

    dataset = MorphingDistillDataset(
        root=args.data_root,
        metadata_file=args.metadata,
        split="test",
        strict_split=True,
        load_occupancy=False,
        skip_missing=True,
        verbose=True,
        load_source_images=False,
        source_images_root=args.source_images_root,
        source_image_filename=args.source_image_filename,
    )

    unique: List[Tuple[int, Dict[str, Any]]] = []
    seen = set()
    for idx, entry in enumerate(dataset.metadata):
        key = (str(entry["src_1"]), str(entry["src_2"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append((idx, dict(entry)))

    if len(unique) < args.num_pairs:
        raise RuntimeError(f"Requested N={args.num_pairs}, but only {len(unique)} unique test pairs are available")

    rng = np.random.default_rng(args.seed)
    chosen_positions = rng.choice(len(unique), size=args.num_pairs, replace=False).tolist()
    chosen = [unique[int(pos)] for pos in chosen_positions]
    alphas = canonical_alphas(args.k)

    pairs = []
    for pair_id, (metadata_index, entry) in enumerate(chosen):
        src1 = str(entry["src_1"])
        src2 = str(entry["src_2"])
        src1_image = dataset._find_source_image_path(src1, entry, "src1")
        src2_image = dataset._find_source_image_path(src2, entry, "src2")
        if src1_image is None or src2_image is None:
            raise FileNotFoundError(
                f"Training-free input image missing for pair {src1} -> {src2}: "
                f"src1={src1_image}, src2={src2_image}"
            )
        pairs.append(
            {
                "pair_id": pair_id,
                "pair_dir": pair_dir_name(pair_id, src1, src2),
                "metadata_index": int(metadata_index),
                "src1": src1,
                "src2": src2,
                "src1_image": str(src1_image),
                "src2_image": str(src2_image),
                "pair_seed": pair_seed(args.seed, pair_id),
            }
        )

    payload = {
        "created_utc": utc_now(),
        "seed": int(args.seed),
        "num_pairs": int(args.num_pairs),
        "k_intermediate": int(args.k),
        "split": "test",
        "data_root": str(Path(args.data_root).resolve()),
        "metadata": str(args.metadata),
        "source_images_root": str(Path(args.source_images_root).resolve()),
        "alpha_convention": "alpha=1 -> src1; alpha=0 -> src2",
        "alphas": alphas,
        "interp3d_internal_t": [float(round(1.0 - a, 10)) for a in alphas],
        "pairs": pairs,
    }
    write_json(plan_path, payload)
    log(f"PLAN written: {plan_path} | pairs={len(pairs)} | K={args.k} | seed={args.seed}", run_root)


def command_morphflow(args: argparse.Namespace) -> None:
    import torch

    from data.morph_dataset import MorphingDistillDataset
    from eval_validation_latents import (
        build_model,
        checkpoint_args,
        checkpoint_requires_source_images,
        detect_flow_target,
        detect_model_type,
        load_checkpoint,
        load_decoders,
        preload_dino_if_needed,
        sample_slat_on_coords,
        sample_ss,
        ss_coords_from_latent,
    )
    from generate_alpha_steps import batch_for_alpha, load_asset

    run_root = Path(args.run_root).resolve()
    plan = load_plan(run_root)
    device = torch.device("cuda")

    ss_ckpt = load_checkpoint(args.checkpoint_path)
    slat_ckpt = load_checkpoint(args.slat_checkpoint_path)
    if detect_flow_target(ss_ckpt) != "ss":
        raise ValueError("--checkpoint-path must be the MorphFlow SS checkpoint")
    if detect_flow_target(slat_ckpt) != "slat":
        raise ValueError("--slat-checkpoint-path must be the MorphFlow SLat checkpoint")

    ss_model_type = detect_model_type(ss_ckpt, args.trellis_model)
    slat_model_type = detect_model_type(slat_ckpt, args.trellis_model)
    ss_model = build_model(ss_ckpt, ss_model_type, "ss").to(device).eval()
    slat_model = build_model(slat_ckpt, slat_model_type, "slat").to(device).eval()
    preload_dino_if_needed(ss_model, device)
    preload_dino_if_needed(slat_model, device)
    ss_decoder, mesh_decoder, sparse_cls = load_decoders("ss", device)

    needs_images = checkpoint_requires_source_images(slat_ckpt, "slat")
    dataset = MorphingDistillDataset(
        root=args.data_root,
        metadata_file=args.metadata,
        split="test",
        strict_split=True,
        load_occupancy=False,
        skip_missing=True,
        verbose=False,
        load_source_images=False,
        source_images_root=args.source_images_root,
        source_image_filename=args.source_image_filename,
    )

    method_root = run_root / "morphflow"
    refs_root = run_root / "references_raw"
    method_root.mkdir(parents=True, exist_ok=True)
    refs_root.mkdir(parents=True, exist_ok=True)

    log(
        "MorphFlow generation start | "
        f"ss_arch={checkpoint_args(ss_ckpt).get('ss_flow_arch', 'standard')} | "
        f"pairs={plan['num_pairs']} | K={plan['k_intermediate']}",
        run_root,
    )

    for pidx, pair in enumerate(plan["pairs"]):
        pair_id = int(pair["pair_id"])
        pair_out = method_root / pair["pair_dir"]
        pair_out.mkdir(parents=True, exist_ok=True)
        log(f"[MorphFlow] pair {pidx + 1}/{plan['num_pairs']} {pair['src1']} -> {pair['src2']}", run_root)

        entry = dataset.metadata[int(pair["metadata_index"])]
        src1 = load_asset(Path(args.data_root), pair["src1"])
        src2 = load_asset(Path(args.data_root), pair["src2"])
        if needs_images:
            src1["image"] = dataset._load_source_image(pair["src1"], entry, "src1")
            src2["image"] = dataset._load_source_image(pair["src2"], entry, "src2")

        ref_pair = refs_root / pair["pair_dir"]
        ref_pair.mkdir(parents=True, exist_ok=True)
        src1_raw = ref_pair / "src1_raw.pt"
        src2_raw = ref_pair / "src2_raw.pt"
        if not src1_raw.exists():
            mesh = decode_slat_mesh(mesh_decoder, sparse_cls, src1["feats"], src1["coords"], device)
            save_raw_mesh(mesh, src1_raw, {"kind": "reference", "asset": pair["src1"]})
        if not src2_raw.exists():
            mesh = decode_slat_mesh(mesh_decoder, sparse_cls, src2["feats"], src2["coords"], device)
            save_raw_mesh(mesh, src2_raw, {"kind": "reference", "asset": pair["src2"]})

        template_ss = torch.zeros_like(src1["ss_latent"])
        for step_idx, alpha in enumerate(plan["alphas"]):
            step_out = pair_out / alpha_dir(alpha)
            raw_path = step_out / "mesh_raw.pt"
            if raw_path.exists() and not args.overwrite:
                log(f"[MorphFlow] pair={pair_id:04d} step={step_idx + 1}/{plan['k_intermediate']} skip existing", run_root)
                continue

            ss_seed = step_seed(plan["seed"], pair_id, step_idx, 0)
            slat_seed = step_seed(plan["seed"], pair_id, step_idx, 10_000)
            seed_everything(ss_seed)
            batch = batch_for_alpha(src1, src2, float(alpha))
            pred_ss = sample_ss(
                ss_model,
                batch,
                template_ss.to(device=device, dtype=torch.float32),
                args.steps,
                device,
                args.cfg_scale,
                args.mixed_precision,
            )
            pred_coords = ss_coords_from_latent(ss_decoder, pred_ss, device, args.mixed_precision)

            seed_everything(slat_seed)
            pred_slat = sample_slat_on_coords(
                slat_model,
                batch,
                pred_coords,
                args.slat_steps,
                device,
                args.slat_cfg_scale,
                args.mixed_precision,
            )
            if pred_slat is None:
                raise RuntimeError(f"MorphFlow produced empty SLat coords for pair={pair_id}, alpha={alpha}")
            mesh = decode_slat_mesh(mesh_decoder, sparse_cls, pred_slat.feats, pred_slat.coords, device)
            save_raw_mesh(
                mesh,
                raw_path,
                {
                    "method": "morphflow",
                    "pair_id": pair_id,
                    "step": step_idx,
                    "alpha": float(alpha),
                    "ss_seed": ss_seed,
                    "slat_seed": slat_seed,
                },
            )
            append_jsonl(run_root / "generation.jsonl", {"method": "morphflow", "pair_id": pair_id, "step": step_idx, "alpha": float(alpha), "raw": str(raw_path)})
            log(f"[MorphFlow] pair={pair_id:04d} step={step_idx + 1}/{plan['k_intermediate']} alpha={alpha:.6f} saved", run_root)
            del pred_ss, pred_coords, pred_slat, mesh, batch

        torch.cuda.empty_cache()

    write_json(method_root / "generation_status.json", {"status": "complete", "finished_utc": utc_now()})
    log("MorphFlow generation complete", run_root)


def morphany_cache_ready(path: Path) -> bool:
    return all((path / name).is_file() for name in ("coords_zs_init.pt", "slat_init.pt", "coords.pt"))


def command_morphany3d(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline

    run_root = Path(args.run_root).resolve()
    plan = load_plan(run_root)
    method_root = run_root / "morphany3d"
    cache_root = run_root / "method_cache" / "morphany3d"
    method_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    log(f"MorphAny3D generation start | pairs={plan['num_pairs']} | K={plan['k_intermediate']}", run_root)

    for pidx, pair in enumerate(plan["pairs"]):
        pair_id = int(pair["pair_id"])
        pseed = int(pair["pair_seed"])
        pair_out = method_root / pair["pair_dir"]
        pair_out.mkdir(parents=True, exist_ok=True)
        src_cache = cache_root / pair["pair_dir"] / "src"
        tar_cache = cache_root / pair["pair_dir"] / "tar"
        src_cache.mkdir(parents=True, exist_ok=True)
        tar_cache.mkdir(parents=True, exist_ok=True)

        with Image.open(pair["src1_image"]) as im:
            src_img = im.copy()
        with Image.open(pair["src2_image"]) as im:
            tar_img = im.copy()

        log(f"[MorphAny3D] pair {pidx + 1}/{plan['num_pairs']} {pair['src1']} -> {pair['src2']}", run_root)

        base_params = {
            "init_morphing_flag": False,
            "ss_mca_flag": False,
            "slat_mca_flag": False,
            "ss_tfsa_flag": False,
            "slat_tfsa_flag": False,
            "oc_flag": False,
        }
        if not morphany_cache_ready(src_cache):
            params = dict(base_params, save_cache_path=str(src_cache))
            seed_everything(pseed)
            pipeline.run_morphing(src_img, tar_img, morphing_params=params, seed=pseed, formats=["mesh"])
            log(f"[MorphAny3D] pair={pair_id:04d} source cache ready", run_root)
        if not morphany_cache_ready(tar_cache):
            params = dict(base_params, save_cache_path=str(tar_cache))
            seed_everything(pseed)
            pipeline.run_morphing(tar_img, src_img, morphing_params=params, seed=pseed, formats=["mesh"])
            log(f"[MorphAny3D] pair={pair_id:04d} target cache ready", run_root)

        for step_idx, alpha in enumerate(plan["alphas"]):
            step_out = pair_out / alpha_dir(alpha)
            raw_path = step_out / "mesh_raw.pt"
            if raw_path.exists() and not args.overwrite:
                log(f"[MorphAny3D] pair={pair_id:04d} step={step_idx + 1}/{plan['k_intermediate']} skip existing", run_root)
                continue

            params = {
                "morphing_num": int(plan["k_intermediate"]) + 2,
                "src_load_cache_path": str(src_cache),
                "tar_load_cache_path": str(tar_cache),
                "save_cache_path": str(cache_root / pair["pair_dir"] / "work"),
                "init_morphing_flag": False,
                "ss_mca_flag": True,
                "slat_mca_flag": True,
                "ss_tfsa_flag": True,
                "slat_tfsa_flag": True,
                "oc_flag": False,
                "alpha": float(alpha),
                "morphing_idx": int(step_idx + 1),
                "tfsa_cache_idx": int(step_idx),
                "tfsa_alpha": 0.8,
            }
            Path(params["save_cache_path"]).mkdir(parents=True, exist_ok=True)
            seed_everything(pseed)
            outputs = pipeline.run_morphing(
                src_img,
                tar_img,
                morphing_params=params,
                seed=pseed,
                formats=["mesh"],
            )
            mesh = outputs["mesh"][0]
            save_raw_mesh(mesh, raw_path, {"method": "morphany3d", "pair_id": pair_id, "step": step_idx, "alpha": float(alpha), "seed": pseed})
            append_jsonl(run_root / "generation.jsonl", {"method": "morphany3d", "pair_id": pair_id, "step": step_idx, "alpha": float(alpha), "raw": str(raw_path)})
            log(f"[MorphAny3D] pair={pair_id:04d} step={step_idx + 1}/{plan['k_intermediate']} alpha={alpha:.6f} saved", run_root)
            del outputs, mesh

        torch.cuda.empty_cache()

    write_json(method_root / "generation_status.json", {"status": "complete", "finished_utc": utc_now()})
    log("MorphAny3D generation complete", run_root)


def command_interp3d(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image
    import trellis.pipelines.trellis_image_to_3d_morphing as interp_module
    from trellis.pipelines import TrellisImageTo3DPipeline

    run_root = Path(args.run_root).resolve()
    plan = load_plan(run_root)
    method_root = run_root / "interp3d"
    method_root.mkdir(parents=True, exist_ok=True)

    # For a fair K-step comparison, keep Interp3D's interpolation algorithm but
    # evaluate it at the same uniform source->target schedule as the other methods.
    def uniform_alpha_list(coords0: Any, coords1: Any, steps: int = 5):
        return torch.linspace(0.0, 1.0, steps, device=coords0.device)

    interp_module.adaptive_alpha_list_by_chamfer = uniform_alpha_list

    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    log(f"Interp3D generation start | uniform schedule | pairs={plan['num_pairs']} | K={plan['k_intermediate']}", run_root)

    for pidx, pair in enumerate(plan["pairs"]):
        pair_id = int(pair["pair_id"])
        pseed = int(pair["pair_seed"])
        pair_out = method_root / pair["pair_dir"]
        pair_out.mkdir(parents=True, exist_ok=True)

        all_exist = all((pair_out / alpha_dir(a) / "mesh_raw.pt").is_file() for a in plan["alphas"])
        if all_exist and not args.overwrite:
            log(f"[Interp3D] pair {pidx + 1}/{plan['num_pairs']} skip complete pair", run_root)
            continue

        with Image.open(pair["src1_image"]) as im:
            src_img = im.copy()
        with Image.open(pair["src2_image"]) as im:
            tar_img = im.copy()

        log(f"[Interp3D] pair {pidx + 1}/{plan['num_pairs']} {pair['src1']} -> {pair['src2']}", run_root)
        seed_everything(pseed)
        outputs, _ = pipeline.run_interpolation_morphing(
            src_img,
            tar_img,
            num_search_iterations=int(plan["k_intermediate"]) + 2,
            seed=pseed,
            formats=["mesh"],
        )
        expected = int(plan["k_intermediate"]) + 2
        if len(outputs) != expected:
            raise RuntimeError(
                f"Interp3D returned {len(outputs)} states, expected {expected}. "
                "Upstream interpolation behavior differs from this benchmark assumption."
            )
        middle = outputs[1:-1]
        if len(middle) != int(plan["k_intermediate"]):
            raise RuntimeError(f"Interp3D middle-state count mismatch: {len(middle)}")

        for step_idx, (alpha, item) in enumerate(zip(plan["alphas"], middle)):
            raw_path = pair_out / alpha_dir(alpha) / "mesh_raw.pt"
            if raw_path.exists() and not args.overwrite:
                continue
            mesh = item["mesh"][0]
            save_raw_mesh(mesh, raw_path, {"method": "interp3d", "pair_id": pair_id, "step": step_idx, "alpha": float(alpha), "seed": pseed})
            append_jsonl(run_root / "generation.jsonl", {"method": "interp3d", "pair_id": pair_id, "step": step_idx, "alpha": float(alpha), "raw": str(raw_path)})
            log(f"[Interp3D] pair={pair_id:04d} step={step_idx + 1}/{plan['k_intermediate']} alpha={alpha:.6f} saved", run_root)

        del outputs, middle
        torch.cuda.empty_cache()

    write_json(method_root / "generation_status.json", {"status": "complete", "finished_utc": utc_now(), "schedule": "uniform"})
    log("Interp3D generation complete", run_root)


def load_raw_as_trellis_mesh(path: Path, device: str = "cuda") -> Any:
    import torch
    from trellis.representations.mesh import MeshExtractResult

    payload = torch.load(path, map_location="cpu")
    vertices = payload["vertices"].to(device=device, dtype=torch.float32)
    faces = payload["faces"].to(device=device, dtype=torch.long)
    attrs = payload.get("vertex_attrs")
    if attrs is None:
        attrs = torch.full((vertices.shape[0], 6), 0.7, dtype=torch.float32)
    elif attrs.shape[1] < 6:
        pad = torch.zeros((attrs.shape[0], 6 - attrs.shape[1]), dtype=attrs.dtype)
        attrs = torch.cat([attrs, pad], dim=1)
    attrs = attrs.to(device=device, dtype=torch.float32)
    return MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=attrs, res=int(payload.get("res", 64)))


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def texture_one(raw_path: Path, glb_path: Path, args: argparse.Namespace, run_root: Path, label: str) -> None:
    if glb_path.is_file() and not args.overwrite:
        return
    from trellis.utils import postprocessing_utils

    mesh = load_raw_as_trellis_mesh(raw_path, "cuda")
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[TEXTURE] start {label}", run_root)
    textured = postprocessing_utils.to_glb(
        mesh,
        mesh,
        simplify=args.simplify,
        fill_holes=bool(args.fill_holes),
        texture_size=args.texture_size,
        verbose=bool(args.texture_verbose),
    )
    textured.export(glb_path)
    log(f"[TEXTURE] done  {label} -> {glb_path}", run_root)


def command_texture(args: argparse.Namespace) -> None:
    import torch
    import nvdiffrast.torch  # noqa: F401 - explicit preflight

    run_root = Path(args.run_root).resolve()
    plan = load_plan(run_root)
    refs_raw = run_root / "references_raw"
    refs_glb = run_root / "references_textured"
    refs_glb.mkdir(parents=True, exist_ok=True)

    total = int(plan["num_pairs"]) * (2 + len(METHODS) * int(plan["k_intermediate"]))
    done = 0
    log(f"Common texture-baking start | expected GLBs={total} | texture={args.texture_size}px | simplify={args.simplify}", run_root)

    for pair in plan["pairs"]:
        pair_ref_raw = refs_raw / pair["pair_dir"]
        pair_ref_glb = refs_glb / pair["pair_dir"]
        src1_glb = pair_ref_glb / "src1.glb"
        src2_glb = pair_ref_glb / "src2.glb"
        texture_one(pair_ref_raw / "src1_raw.pt", src1_glb, args, run_root, f"reference {pair['pair_id']:04d} src1")
        done += 1
        texture_one(pair_ref_raw / "src2_raw.pt", src2_glb, args, run_root, f"reference {pair['pair_id']:04d} src2")
        done += 1

        for method in METHODS:
            pair_out = run_root / method / pair["pair_dir"]
            if not pair_out.is_dir():
                raise FileNotFoundError(f"Missing generated pair directory: {pair_out}")
            hardlink_or_copy(src1_glb, pair_out / "src1.glb")
            hardlink_or_copy(src2_glb, pair_out / "src2.glb")

            sequence = [{"kind": "source", "alpha": 1.0, "path": str(pair_out / "src1.glb"), "name": pair["src1"]}]
            steps = []
            for step_idx, alpha in enumerate(plan["alphas"]):
                step_out = pair_out / alpha_dir(alpha)
                raw_path = step_out / "mesh_raw.pt"
                glb_path = step_out / "pred_final.glb"
                if not raw_path.is_file():
                    raise FileNotFoundError(f"Missing raw mesh: {raw_path}")
                texture_one(raw_path, glb_path, args, run_root, f"{method} pair={pair['pair_id']:04d} step={step_idx + 1}/{plan['k_intermediate']}")
                done += 1
                row = {
                    "pair_id": int(pair["pair_id"]),
                    "step": step_idx,
                    "alpha": float(alpha),
                    "pred_final": str(glb_path),
                    "pred_final_saved": True,
                    "raw_mesh": str(raw_path),
                }
                write_json(step_out / "metadata.json", row)
                steps.append(row)
                sequence.append({"kind": "prediction", "alpha": float(alpha), "path": str(glb_path), "saved": True})

            sequence.append({"kind": "source", "alpha": 0.0, "path": str(pair_out / "src2.glb"), "name": pair["src2"]})
            write_json(pair_out / "sequence.json", sequence)
            write_json(
                pair_out / "summary.json",
                {
                    "method": method,
                    "pair_id": int(pair["pair_id"]),
                    "src1": pair["src1"],
                    "src2": pair["src2"],
                    "alphas": plan["alphas"],
                    "num_generated_steps": int(plan["k_intermediate"]),
                    "steps": steps,
                    "sequence": sequence,
                    "appearance": "common TRELLIS mesh-color texture baking",
                },
            )
            log(f"[TEXTURE] manifests ready {method} pair={pair['pair_id']:04d} | progress={done}/{total}", run_root)

    write_json(
        run_root / "texture_status.json",
        {
            "status": "complete",
            "finished_utc": utc_now(),
            "texture_size": args.texture_size,
            "simplify": args.simplify,
            "fill_holes": bool(args.fill_holes),
            "appearance_source": "MeshExtractResult.vertex_attrs -> common TRELLIS to_glb(mesh, mesh)",
        },
    )
    torch.cuda.empty_cache()
    log("Common texture-baking complete", run_root)


def command_aggregate(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).resolve()
    plan = load_plan(run_root)
    metrics_root = run_root / "metrics"
    combined: Dict[str, Any] = {
        "created_utc": utc_now(),
        "seed": plan["seed"],
        "num_pairs": plan["num_pairs"],
        "k_intermediate": plan["k_intermediate"],
        "alphas": plan["alphas"],
        "metric_source": "eval_perceptual_sequence.py rendered from final textured GLBs with appearance=materials",
        "methods": {},
    }
    rows: List[Dict[str, Any]] = []
    headline = (
        "lpips_adjacent_mean",
        "ppl_sum",
        "ppl_normalized_reference_endpoints",
        "pdv",
        "endpoint_fidelity_lpips_mean",
        "ppl_reference_anchored",
    )

    for method in METHODS:
        path = metrics_root / method / "metrics_batch.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing batch metrics for {method}: {path}")
        data = read_json(path)
        combined["methods"][method] = data
        agg = data.get("aggregate_across_pairs", {})
        for metric in headline:
            stats = agg.get(metric, {})
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                    "median": stats.get("median"),
                    "num_pairs": data.get("num_pairs"),
                }
            )

    write_json(run_root / "final_metrics.json", combined)
    with (run_root / "final_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "metric", "mean", "std", "median", "num_pairs"])
        writer.writeheader()
        writer.writerows(rows)
    log(f"Final metrics written: {run_root / 'final_metrics.json'}", run_root)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Three-way textured-GLB benchmark")
    sub = p.add_subparsers(dest="command", required=True)

    def common_run(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--run-root", required=True)
        sp.add_argument("--overwrite", action="store_true")

    plan = sub.add_parser("plan")
    common_run(plan)
    plan.add_argument("--data-root", required=True)
    plan.add_argument("--metadata", default="metadata_test.json")
    plan.add_argument("--source-images-root", required=True)
    plan.add_argument("--source-image-filename", default="")
    plan.add_argument("--num-pairs", type=int, default=DEFAULT_N)
    plan.add_argument("--k", type=int, default=DEFAULT_K)
    plan.add_argument("--seed", type=int, default=DEFAULT_SEED)

    mf = sub.add_parser("morphflow")
    common_run(mf)
    mf.add_argument("--data-root", required=True)
    mf.add_argument("--metadata", default="metadata_test.json")
    mf.add_argument("--source-images-root", required=True)
    mf.add_argument("--source-image-filename", default="")
    mf.add_argument("--checkpoint-path", required=True)
    mf.add_argument("--slat-checkpoint-path", required=True)
    mf.add_argument("--steps", type=int, default=50)
    mf.add_argument("--slat-steps", type=int, default=50)
    mf.add_argument("--cfg-scale", type=float, default=3.0)
    mf.add_argument("--slat-cfg-scale", type=float, default=3.0)
    mf.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    mf.add_argument("--trellis-model", choices=["auto", "text_base", "image_large"], default="image_large")

    ma = sub.add_parser("morphany3d")
    common_run(ma)

    ip = sub.add_parser("interp3d")
    common_run(ip)

    tx = sub.add_parser("texture")
    common_run(tx)
    tx.add_argument("--texture-size", type=int, default=1024)
    tx.add_argument("--simplify", type=float, default=0.95)
    tx.add_argument("--fill-holes", type=int, choices=[0, 1], default=1)
    tx.add_argument("--texture-verbose", type=int, choices=[0, 1], default=0)

    ag = sub.add_parser("aggregate")
    common_run(ag)
    return p


def main() -> None:
    args = build_parser().parse_args()
    command = args.command
    if command == "plan":
        command_plan(args)
    elif command == "morphflow":
        command_morphflow(args)
    elif command == "morphany3d":
        command_morphany3d(args)
    elif command == "interp3d":
        command_interp3d(args)
    elif command == "texture":
        command_texture(args)
    elif command == "aggregate":
        command_aggregate(args)
    else:
        raise ValueError(command)


if __name__ == "__main__":
    main()
