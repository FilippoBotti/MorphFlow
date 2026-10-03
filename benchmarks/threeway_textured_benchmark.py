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
* eval_textured_benchmark.py wraps eval_perceptual_sequence.py and adds full-path metrics + FID;
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


def object_nbytes(obj: Any) -> int:
    """Best-effort tensor payload size without counting Python/container overhead."""
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, Mapping):
        return sum(object_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(object_nbytes(v) for v in obj)
    return 0


def process_rss_bytes() -> int:
    """Linux RSS from /proc; zero on non-Linux/failure."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def node_mem_available_bytes() -> int:
    """Linux MemAvailable from /proc; zero on failure."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def gib(value: int | float) -> float:
    return float(value) / float(1024 ** 3)


def cuda_sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def torch_load_cpu(path: Path) -> Any:
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class HybridTFSACache:
    """Bounded CPU-RAM cache with transparent spill to node-local storage.

    MorphAny3D's attention modules only require mapping-style ``in``, assignment,
    and ``get``.  Keeping this object alive across alpha steps avoids the
    thousands of shared-filesystem torch.save/torch.load operations performed by
    the upstream disk cache.  Only the immediately preceding morph state is kept
    after each alpha, because TFSA never needs anything older.
    """

    def __init__(self, max_ram_bytes: int, spill_dir: Path):
        self.max_ram_bytes = max(0, int(max_ram_bytes))
        self.spill_dir = Path(spill_dir)
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        self._ram: Dict[Any, Any] = {}
        self._ram_sizes: Dict[Any, int] = {}
        self._spill: Dict[Any, Tuple[Path, int]] = {}
        self.ram_bytes = 0
        self.spill_bytes = 0
        self.peak_ram_bytes = 0
        self.peak_spill_bytes = 0
        self.spill_bytes_written = 0
        self.spill_bytes_read = 0
        self._serial = 0

    def __contains__(self, key: Any) -> bool:
        return key in self._ram or key in self._spill

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self:
            return
        size = object_nbytes(value)
        if self.max_ram_bytes > 0 and self.ram_bytes + size <= self.max_ram_bytes:
            self._ram[key] = value
            self._ram_sizes[key] = size
            self.ram_bytes += size
            self.peak_ram_bytes = max(self.peak_ram_bytes, self.ram_bytes)
            return
        self._spill_value(key, value, size)

    def get(self, key: Any, default: Any = None) -> Any:
        if key in self._ram:
            return self._ram[key]
        item = self._spill.get(key)
        if item is None:
            return default
        path, _size = item
        try:
            on_disk = int(path.stat().st_size)
        except OSError:
            on_disk = 0
        value = torch_load_cpu(path)
        self.spill_bytes_read += on_disk
        return value

    def _spill_value(self, key: Any, value: Any, logical_size: Optional[int] = None) -> None:
        import torch
        self._serial += 1
        path = self.spill_dir / f"tfsa_{self._serial:08d}.pt"
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        torch.save(value, tmp)
        tmp.replace(path)
        size = int(path.stat().st_size)
        self._spill[key] = (path, size)
        self.spill_bytes += size
        self.spill_bytes_written += size
        self.peak_spill_bytes = max(self.peak_spill_bytes, self.spill_bytes)

    def _delete_key(self, key: Any) -> None:
        if key in self._ram:
            self.ram_bytes -= self._ram_sizes.pop(key, 0)
            del self._ram[key]
        item = self._spill.pop(key, None)
        if item is not None:
            path, size = item
            self.spill_bytes -= size
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _morph_idx(key: Any) -> Optional[int]:
        if isinstance(key, tuple) and len(key) >= 2 and key[0] in {"ss", "slat", "slat_coords"}:
            try:
                return int(key[1])
            except Exception:
                return None
        return None

    def prune_to_morphing_idx(self, morphing_idx: int, promote: bool = True) -> None:
        """Drop states older than the one needed by the next alpha."""
        keep = int(morphing_idx)
        for key in list(self._ram) + list(self._spill):
            idx = self._morph_idx(key)
            if idx is not None and idx != keep:
                self._delete_key(key)
        if promote and self.max_ram_bytes > 0:
            self._promote_spill_until_full()

    def _promote_spill_until_full(self) -> None:
        for key, (path, disk_size) in list(self._spill.items()):
            # File size is a conservative-enough precheck and avoids loading a
            # large spilled tensor only to discover that it cannot fit in RAM.
            if self.ram_bytes + disk_size > self.max_ram_bytes:
                continue
            value = torch_load_cpu(path)
            logical_size = object_nbytes(value)
            if self.ram_bytes + logical_size > self.max_ram_bytes:
                del value
                continue
            self._ram[key] = value
            self._ram_sizes[key] = logical_size
            self.ram_bytes += logical_size
            self.peak_ram_bytes = max(self.peak_ram_bytes, self.ram_bytes)
            self.spill_bytes_read += disk_size
            self.spill_bytes -= disk_size
            del self._spill[key]
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def spill_all_ram(self) -> None:
        """Emergency pressure relief: move every resident entry to local disk."""
        for key in list(self._ram):
            value = self._ram[key]
            logical = self._ram_sizes.get(key, object_nbytes(value))
            self._spill_value(key, value, logical)
            self.ram_bytes -= self._ram_sizes.pop(key, 0)
            del self._ram[key]

    def clear(self) -> None:
        for key in list(self._ram) + list(self._spill):
            self._delete_key(key)
        try:
            self.spill_dir.rmdir()
        except OSError:
            pass

    def stats(self) -> Dict[str, float]:
        return {
            "ram_gib": gib(self.ram_bytes),
            "peak_ram_gib": gib(self.peak_ram_bytes),
            "spill_gib": gib(self.spill_bytes),
            "peak_spill_gib": gib(self.peak_spill_bytes),
            "spill_written_gib": gib(self.spill_bytes_written),
            "spill_read_gib": gib(self.spill_bytes_read),
        }


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


def shard_pairs(plan: Mapping[str, Any], shard_index: int, num_shards: int) -> List[Dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"--shard-index must be in [0, {num_shards - 1}]")
    return [
        dict(pair)
        for pair in plan["pairs"]
        if int(pair["pair_id"]) % int(num_shards) == int(shard_index)
    ]


def shard_tag(args: argparse.Namespace) -> str:
    return f"shard_{int(args.shard_index):02d}_of_{int(args.num_shards):02d}"


def shard_jsonl(run_root: Path, args: argparse.Namespace) -> Path:
    return run_root / f"generation_{shard_tag(args)}.jsonl"


def timing_jsonl(run_root: Path, method: str, args: argparse.Namespace) -> Path:
    return run_root / "timings" / f"{method}_{shard_tag(args)}.jsonl"


def append_timing(run_root: Path, method: str, args: argparse.Namespace, payload: Mapping[str, Any]) -> None:
    row = dict(payload)
    row.setdefault("method", method)
    row.setdefault("shard_index", int(args.shard_index))
    row.setdefault("num_shards", int(args.num_shards))
    row.setdefault("recorded_utc", utc_now())
    append_jsonl(timing_jsonl(run_root, method, args), row)


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
    with torch.no_grad():
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

    # This benchmark is inference-only. sample_ss() is not decorated with
    # torch.no_grad() upstream, so disable autograd globally to avoid retaining
    # the iterative SS sampling graph and exhausting VRAM.
    torch.set_grad_enabled(False)

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
    pairs = shard_pairs(plan, args.shard_index, args.num_shards)
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
        f"pairs_this_shard={len(pairs)}/{plan['num_pairs']} | K={plan['k_intermediate']} | {shard_tag(args)}",
        run_root,
    )

    for pidx, pair in enumerate(pairs):
        pair_wall_start = time.perf_counter()
        reference_export_seconds = 0.0
        intermediate_compute_seconds = 0.0
        serialization_seconds = 0.0
        steps_generated = 0
        steps_skipped_output = 0
        peak_rss = process_rss_bytes()
        pair_id = int(pair["pair_id"])
        pair_out = method_root / pair["pair_dir"]
        pair_out.mkdir(parents=True, exist_ok=True)
        log(f"[MorphFlow {shard_tag(args)}] pair {pidx + 1}/{len(pairs)} {pair['src1']} -> {pair['src2']}", run_root)

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
        ref_start = time.perf_counter()
        if not src1_raw.exists():
            mesh = decode_slat_mesh(mesh_decoder, sparse_cls, src1["feats"], src1["coords"], device)
            save_raw_mesh(mesh, src1_raw, {"kind": "reference", "asset": pair["src1"]})
            del mesh
            torch.cuda.empty_cache()
        if not src2_raw.exists():
            mesh = decode_slat_mesh(mesh_decoder, sparse_cls, src2["feats"], src2["coords"], device)
            save_raw_mesh(mesh, src2_raw, {"kind": "reference", "asset": pair["src2"]})
            del mesh
            torch.cuda.empty_cache()
        cuda_sync()
        reference_export_seconds = time.perf_counter() - ref_start

        template_ss = torch.zeros_like(src1["ss_latent"])
        for step_idx, alpha in enumerate(plan["alphas"]):
            step_out = pair_out / alpha_dir(alpha)
            raw_path = step_out / "mesh_raw.pt"
            if raw_path.exists() and not args.overwrite:
                steps_skipped_output += 1
                log(f"[MorphFlow] pair={pair_id:04d} step={step_idx + 1}/{plan['k_intermediate']} skip existing", run_root)
                continue

            ss_seed = step_seed(plan["seed"], pair_id, step_idx, 0)
            slat_seed = step_seed(plan["seed"], pair_id, step_idx, 10_000)
            seed_everything(ss_seed)
            batch = batch_for_alpha(src1, src2, float(alpha))
            cuda_sync()
            compute_start = time.perf_counter()
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
            cuda_sync()
            step_compute = time.perf_counter() - compute_start
            intermediate_compute_seconds += step_compute
            steps_generated += 1
            serial_start = time.perf_counter()
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
            serialization_seconds += time.perf_counter() - serial_start
            append_jsonl(shard_jsonl(run_root, args), {"method": "morphflow", "pair_id": pair_id, "step": step_idx, "alpha": float(alpha), "raw": str(raw_path)})
            rss = process_rss_bytes()
            peak_rss = max(peak_rss, rss)
            log(
                f"[MorphFlow] pair={pair_id:04d} step={step_idx + 1}/{plan['k_intermediate']} alpha={alpha:.6f} saved "
                f"compute={step_compute:.2f}s rss={gib(rss):.2f}GiB",
                run_root,
            )
            del pred_ss, pred_coords, pred_slat, mesh, batch
            torch.cuda.empty_cache()

        cuda_sync()
        pair_total_wall_seconds = time.perf_counter() - pair_wall_start
        sequence_end_to_end_seconds = max(0.0, pair_total_wall_seconds - reference_export_seconds)
        timing_valid = steps_generated == int(plan["k_intermediate"]) and steps_skipped_output == 0
        append_timing(
            run_root,
            "morphflow",
            args,
            {
                "pair_id": pair_id,
                "pair_dir": pair["pair_dir"],
                "timing_valid": bool(timing_valid),
                "steps_generated": int(steps_generated),
                "steps_skipped_output": int(steps_skipped_output),
                "k_intermediate": int(plan["k_intermediate"]),
                "sequence_end_to_end_seconds": float(sequence_end_to_end_seconds),
                "pair_total_wall_seconds": float(pair_total_wall_seconds),
                "benchmark_reference_export_seconds": float(reference_export_seconds),
                "intermediate_compute_seconds": float(intermediate_compute_seconds),
                "serialization_seconds": float(serialization_seconds),
                "seconds_per_intermediate_compute": float(intermediate_compute_seconds / max(1, steps_generated)),
                "process_peak_rss_gib": gib(peak_rss),
                "node_min_mem_available_gib": gib(node_mem_available_bytes()),
            },
        )
        log(
            f"[MorphFlow] pair={pair_id:04d} sequence timing: end_to_end={sequence_end_to_end_seconds:.2f}s "
            f"compute={intermediate_compute_seconds:.2f}s serialization={serialization_seconds:.2f}s",
            run_root,
        )
        del src1, src2, template_ss
        torch.cuda.empty_cache()

    write_json(method_root / f"generation_status_{shard_tag(args)}.json", {"status": "complete", "finished_utc": utc_now(), "pairs": len(pairs), "shard_index": args.shard_index, "num_shards": args.num_shards})
    log(f"MorphFlow generation complete | {shard_tag(args)}", run_root)


def morphany_cache_ready(path: Path) -> bool:
    return all((path / name).is_file() for name in ("coords_zs_init.pt", "slat_init.pt", "coords.pt"))


def command_morphany3d(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline

    run_root = Path(args.run_root).resolve()
    plan = load_plan(run_root)
    pairs = shard_pairs(plan, args.shard_index, args.num_shards)
    method_root = run_root / "morphany3d"
    cache_root = run_root / "method_cache" / "morphany3d"
    method_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    local_cache_root = Path(args.local_cache_root).resolve() if args.local_cache_root else None
    if local_cache_root is not None:
        local_cache_root.mkdir(parents=True, exist_ok=True)

    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    log(
        "MorphAny3D generation start | "
        f"pairs_this_shard={len(pairs)}/{plan['num_pairs']} | K={plan['k_intermediate']} | {shard_tag(args)} | "
        f"tfsa_cache_mode={args.tfsa_cache_mode} | tfsa_ram_limit={args.tfsa_ram_gb:.2f} GiB | "
        f"min_node_free={args.tfsa_min_node_free_gb:.2f} GiB | local_cache={local_cache_root}",
        run_root,
    )

    required_cache_names = ("coords_zs_init.pt", "slat_init.pt", "coords.pt")

    for pidx, pair in enumerate(pairs):
        pair_wall_start = time.perf_counter()
        pair_id = int(pair["pair_id"])
        pseed = int(pair["pair_seed"])
        pair_out = method_root / pair["pair_dir"]
        pair_out.mkdir(parents=True, exist_ok=True)
        src_cache = cache_root / pair["pair_dir"] / "src"
        tar_cache = cache_root / pair["pair_dir"] / "tar"
        src_cache.mkdir(parents=True, exist_ok=True)
        tar_cache.mkdir(parents=True, exist_ok=True)

        raw_paths = [pair_out / alpha_dir(a) / "mesh_raw.pt" for a in plan["alphas"]]
        if all(path.is_file() for path in raw_paths) and not args.overwrite:
            log(f"[MorphAny3D {shard_tag(args)}] pair {pidx + 1}/{len(pairs)} skip complete pair", run_root)
            append_timing(
                run_root,
                "morphany3d",
                args,
                {
                    "pair_id": pair_id,
                    "pair_dir": pair["pair_dir"],
                    "timing_valid": False,
                    "reason": "complete_pair_already_existed",
                    "steps_generated": 0,
                    "steps_skipped_output": int(plan["k_intermediate"]),
                },
            )
            continue

        with Image.open(pair["src1_image"]) as im:
            src_img = im.copy()
        with Image.open(pair["src2_image"]) as im:
            tar_img = im.copy()

        log(f"[MorphAny3D {shard_tag(args)}] pair {pidx + 1}/{len(pairs)} {pair['src1']} -> {pair['src2']}", run_root)

        # Cache image preprocessing + DINO conditioning once per pair. Upstream
        # run_morphing recomputes both for every alpha although the inputs are
        # unchanged. Reusing them is numerically equivalent and removes a large
        # amount of repeated GPU/CPU work.
        conditioning_start = time.perf_counter()
        original_preprocess = pipeline.preprocess_image
        original_get_cond = pipeline.get_cond
        src_pre = original_preprocess(src_img)
        tar_pre = original_preprocess(tar_img)
        with torch.no_grad():
            src_cond_cached = original_get_cond([src_pre])
            tar_cond_cached = original_get_cond([tar_pre])
        cuda_sync()
        conditioning_seconds = time.perf_counter() - conditioning_start

        def cached_preprocess(image):
            return image

        def cached_get_cond(images):
            if len(images) == 1 and images[0] is src_pre:
                return src_cond_cached
            if len(images) == 1 and images[0] is tar_pre:
                return tar_cond_cached
            return original_get_cond(images)

        pipeline.preprocess_image = cached_preprocess
        pipeline.get_cond = cached_get_cond

        source_cache_hit = morphany_cache_ready(src_cache)
        target_cache_hit = morphany_cache_ready(tar_cache)
        cache_prep_seconds = 0.0
        intermediate_compute_seconds = 0.0
        serialization_seconds = 0.0
        steps_generated = 0
        steps_skipped_output = 0
        pressure_relief_triggered = False
        peak_rss = process_rss_bytes()
        min_node_available = node_mem_available_bytes() or 0
        tfsa_cache = None
        local_pair = None

        try:
            base_params = {
                "init_morphing_flag": False,
                "ss_mca_flag": False,
                "slat_mca_flag": False,
                "ss_tfsa_flag": False,
                "slat_tfsa_flag": False,
                "oc_flag": False,
            }

            # Source/target initialization caches are persistent for resumability,
            # but we do NOT decode an unused mesh here. formats=[] still executes
            # the samplers and writes coords_zs_init/slat_init/coords.
            prep_start = time.perf_counter()
            if not source_cache_hit:
                params = dict(base_params, save_cache_path=str(src_cache))
                seed_everything(pseed)
                pipeline.run_morphing(src_pre, tar_pre, morphing_params=params, seed=pseed, formats=[])
                cuda_sync()
                log(f"[MorphAny3D] pair={pair_id:04d} source cache ready", run_root)
            if not target_cache_hit:
                params = dict(base_params, save_cache_path=str(tar_cache))
                seed_everything(pseed)
                pipeline.run_morphing(tar_pre, src_pre, morphing_params=params, seed=pseed, formats=[])
                cuda_sync()
                log(f"[MorphAny3D] pair={pair_id:04d} target cache ready", run_root)

            # Stage the small reusable endpoint caches onto node-local storage so
            # the 10 alpha steps do not repeatedly torch.load them from shared FS.
            load_src_cache = src_cache
            load_tar_cache = tar_cache
            if local_cache_root is not None:
                local_pair = local_cache_root / pair["pair_dir"]
                if local_pair.exists():
                    shutil.rmtree(local_pair, ignore_errors=True)
                local_src = local_pair / "src"
                local_tar = local_pair / "tar"
                local_src.mkdir(parents=True, exist_ok=True)
                local_tar.mkdir(parents=True, exist_ok=True)
                for name in required_cache_names:
                    shutil.copy2(src_cache / name, local_src / name)
                    shutil.copy2(tar_cache / name, local_tar / name)
                load_src_cache = local_src
                load_tar_cache = local_tar
                work_dir = local_pair / "work"
            else:
                work_dir = cache_root / pair["pair_dir"] / "work"
            work_dir.mkdir(parents=True, exist_ok=True)
            cache_prep_seconds = time.perf_counter() - prep_start

            if args.tfsa_cache_mode == "hybrid":
                spill_dir = (local_pair / "tfsa-spill") if local_pair is not None else (work_dir / "tfsa-spill")
                tfsa_cache = HybridTFSACache(int(args.tfsa_ram_gb * (1024 ** 3)), spill_dir)

            for step_idx, alpha in enumerate(plan["alphas"]):
                step_out = pair_out / alpha_dir(alpha)
                raw_path = step_out / "mesh_raw.pt"
                output_exists = raw_path.exists() and not args.overwrite

                # If an incomplete pair is resumed, MorphAny3D must still replay
                # earlier alphas to rebuild the previous-state TFSA cache chain.
                # Existing raw meshes are left untouched; only the compute is replayed.
                params = {
                    "morphing_num": int(plan["k_intermediate"]) + 2,
                    "src_load_cache_path": str(load_src_cache),
                    "tar_load_cache_path": str(load_tar_cache),
                    "save_cache_path": str(work_dir),
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
                if tfsa_cache is not None:
                    params["tfsa_cache"] = tfsa_cache

                available = node_mem_available_bytes()
                if available:
                    if min_node_available == 0:
                        min_node_available = available
                    else:
                        min_node_available = min(min_node_available, available)
                if (
                    tfsa_cache is not None
                    and tfsa_cache.max_ram_bytes > 0
                    and available > 0
                    and available < int(args.tfsa_min_node_free_gb * (1024 ** 3))
                ):
                    tfsa_cache.spill_all_ram()
                    tfsa_cache.max_ram_bytes = 0
                    pressure_relief_triggered = True
                    log(
                        f"[MorphAny3D] pair={pair_id:04d} RAM pressure guard triggered: "
                        f"MemAvailable={gib(available):.2f} GiB; TFSA cache switched to local spill for this pair",
                        run_root,
                    )

                seed_everything(pseed)
                cuda_sync()
                compute_start = time.perf_counter()
                outputs = pipeline.run_morphing(
                    src_pre,
                    tar_pre,
                    morphing_params=params,
                    seed=pseed,
                    formats=["mesh"],
                )
                cuda_sync()
                step_compute = time.perf_counter() - compute_start
                intermediate_compute_seconds += step_compute
                steps_generated += 1

                mesh = outputs["mesh"][0]
                if output_exists:
                    steps_skipped_output += 1
                    log(
                        f"[MorphAny3D] pair={pair_id:04d} step={step_idx + 1}/{plan['k_intermediate']} "
                        f"alpha={alpha:.6f} recomputed for TFSA chain; raw output kept",
                        run_root,
                    )
                else:
                    serial_start = time.perf_counter()
                    save_raw_mesh(
                        mesh,
                        raw_path,
                        {
                            "method": "morphany3d",
                            "pair_id": pair_id,
                            "step": step_idx,
                            "alpha": float(alpha),
                            "seed": pseed,
                        },
                    )
                    serialization_seconds += time.perf_counter() - serial_start
                    append_jsonl(
                        shard_jsonl(run_root, args),
                        {
                            "method": "morphany3d",
                            "pair_id": pair_id,
                            "step": step_idx,
                            "alpha": float(alpha),
                            "raw": str(raw_path),
                        },
                    )
                    log(
                        f"[MorphAny3D] pair={pair_id:04d} step={step_idx + 1}/{plan['k_intermediate']} "
                        f"alpha={alpha:.6f} saved",
                        run_root,
                    )

                del outputs, mesh

                if tfsa_cache is not None:
                    tfsa_cache.prune_to_morphing_idx(step_idx + 1, promote=tfsa_cache.max_ram_bytes > 0)
                    stats = tfsa_cache.stats()
                else:
                    stats = {
                        "ram_gib": 0.0,
                        "peak_ram_gib": 0.0,
                        "spill_gib": 0.0,
                        "peak_spill_gib": 0.0,
                        "spill_written_gib": 0.0,
                        "spill_read_gib": 0.0,
                    }
                rss = process_rss_bytes()
                peak_rss = max(peak_rss, rss)
                available = node_mem_available_bytes()
                if available:
                    min_node_available = available if min_node_available == 0 else min(min_node_available, available)
                log(
                    f"[MorphAny3D] pair={pair_id:04d} step={step_idx + 1}/{plan['k_intermediate']} "
                    f"compute={step_compute:.2f}s tfsa_ram={stats['ram_gib']:.2f}GiB "
                    f"tfsa_spill={stats['spill_gib']:.2f}GiB rss={gib(rss):.2f}GiB "
                    f"node_mem_available={gib(available):.2f}GiB",
                    run_root,
                )

            cuda_sync()
            sequence_end_to_end_seconds = time.perf_counter() - pair_wall_start
            cache_stats = tfsa_cache.stats() if tfsa_cache is not None else {
                "peak_ram_gib": 0.0,
                "peak_spill_gib": 0.0,
                "spill_written_gib": 0.0,
                "spill_read_gib": 0.0,
            }
            timing_valid = steps_generated == int(plan["k_intermediate"]) and steps_skipped_output == 0
            append_timing(
                run_root,
                "morphany3d",
                args,
                {
                    "pair_id": pair_id,
                    "pair_dir": pair["pair_dir"],
                    "timing_valid": bool(timing_valid),
                    "steps_generated": int(steps_generated),
                    "steps_skipped_output": int(steps_skipped_output),
                    "k_intermediate": int(plan["k_intermediate"]),
                    "sequence_end_to_end_seconds": float(sequence_end_to_end_seconds),
                    "conditioning_seconds": float(conditioning_seconds),
                    "endpoint_cache_and_staging_seconds": float(cache_prep_seconds),
                    "intermediate_compute_seconds": float(intermediate_compute_seconds),
                    "serialization_seconds": float(serialization_seconds),
                    "seconds_per_intermediate_compute": float(intermediate_compute_seconds / max(1, steps_generated)),
                    "source_cache_hit": bool(source_cache_hit),
                    "target_cache_hit": bool(target_cache_hit),
                    "tfsa_cache_mode": args.tfsa_cache_mode,
                    "tfsa_ram_limit_gib": float(args.tfsa_ram_gb),
                    "tfsa_peak_ram_gib": float(cache_stats.get("peak_ram_gib", 0.0)),
                    "tfsa_peak_spill_gib": float(cache_stats.get("peak_spill_gib", 0.0)),
                    "tfsa_spill_written_gib": float(cache_stats.get("spill_written_gib", 0.0)),
                    "tfsa_spill_read_gib": float(cache_stats.get("spill_read_gib", 0.0)),
                    "process_peak_rss_gib": gib(peak_rss),
                    "node_min_mem_available_gib": gib(min_node_available),
                    "ram_pressure_guard_triggered": bool(pressure_relief_triggered),
                },
            )
            log(
                f"[MorphAny3D] pair={pair_id:04d} sequence timing: end_to_end={sequence_end_to_end_seconds:.2f}s "
                f"compute={intermediate_compute_seconds:.2f}s serialization={serialization_seconds:.2f}s "
                f"tfsa_peak_ram={cache_stats.get('peak_ram_gib', 0.0):.2f}GiB "
                f"tfsa_peak_spill={cache_stats.get('peak_spill_gib', 0.0):.2f}GiB",
                run_root,
            )
        finally:
            pipeline.preprocess_image = original_preprocess
            pipeline.get_cond = original_get_cond
            if tfsa_cache is not None:
                tfsa_cache.clear()
            if local_pair is not None:
                shutil.rmtree(local_pair, ignore_errors=True)
            del src_cond_cached, tar_cond_cached, src_pre, tar_pre, src_img, tar_img
            torch.cuda.empty_cache()

    write_json(
        method_root / f"generation_status_{shard_tag(args)}.json",
        {
            "status": "complete",
            "finished_utc": utc_now(),
            "pairs": len(pairs),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "tfsa_cache_mode": args.tfsa_cache_mode,
            "tfsa_ram_gb": args.tfsa_ram_gb,
            "tfsa_min_node_free_gb": args.tfsa_min_node_free_gb,
        },
    )
    log(f"MorphAny3D generation complete | {shard_tag(args)}", run_root)

def command_interp3d(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image
    import trellis.pipelines.trellis_image_to_3d_morphing as interp_module
    from trellis.pipelines import TrellisImageTo3DPipeline

    run_root = Path(args.run_root).resolve()
    plan = load_plan(run_root)
    pairs = shard_pairs(plan, args.shard_index, args.num_shards)
    method_root = run_root / "interp3d"
    method_root.mkdir(parents=True, exist_ok=True)

    # For a fair K-step comparison, keep Interp3D's interpolation algorithm but
    # evaluate it at the same uniform source->target schedule as the other methods.
    def uniform_alpha_list(coords0: Any, coords1: Any, steps: int = 5):
        return torch.linspace(0.0, 1.0, steps, device=coords0.device)

    interp_module.adaptive_alpha_list_by_chamfer = uniform_alpha_list

    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    log(f"Interp3D generation start | uniform schedule | pairs_this_shard={len(pairs)}/{plan['num_pairs']} | K={plan['k_intermediate']} | {shard_tag(args)}", run_root)

    for pidx, pair in enumerate(pairs):
        pair_wall_start = time.perf_counter()
        intermediate_compute_seconds = 0.0
        serialization_seconds = 0.0
        peak_rss = process_rss_bytes()
        pair_id = int(pair["pair_id"])
        pseed = int(pair["pair_seed"])
        pair_out = method_root / pair["pair_dir"]
        pair_out.mkdir(parents=True, exist_ok=True)

        all_exist = all((pair_out / alpha_dir(a) / "mesh_raw.pt").is_file() for a in plan["alphas"])
        if all_exist and not args.overwrite:
            log(f"[Interp3D {shard_tag(args)}] pair {pidx + 1}/{len(pairs)} skip complete pair", run_root)
            append_timing(
                run_root,
                "interp3d",
                args,
                {
                    "pair_id": pair_id,
                    "pair_dir": pair["pair_dir"],
                    "timing_valid": False,
                    "reason": "complete_pair_already_existed",
                    "steps_generated": 0,
                    "steps_skipped_output": int(plan["k_intermediate"]),
                },
            )
            continue

        with Image.open(pair["src1_image"]) as im:
            src_img = im.copy()
        with Image.open(pair["src2_image"]) as im:
            tar_img = im.copy()

        log(f"[Interp3D {shard_tag(args)}] pair {pidx + 1}/{len(pairs)} {pair['src1']} -> {pair['src2']}", run_root)
        seed_everything(pseed)
        cuda_sync()
        compute_start = time.perf_counter()
        outputs, _ = pipeline.run_interpolation_morphing(
            src_img,
            tar_img,
            num_search_iterations=int(plan["k_intermediate"]) + 2,
            seed=pseed,
            formats=["mesh"],
        )
        cuda_sync()
        intermediate_compute_seconds = time.perf_counter() - compute_start
        expected = int(plan["k_intermediate"]) + 2
        if len(outputs) != expected:
            raise RuntimeError(
                f"Interp3D returned {len(outputs)} states, expected {expected}. "
                "Upstream interpolation behavior differs from this benchmark assumption."
            )
        middle = outputs[1:-1]
        if len(middle) != int(plan["k_intermediate"]):
            raise RuntimeError(f"Interp3D middle-state count mismatch: {len(middle)}")

        steps_generated = 0
        steps_skipped_output = 0
        for step_idx, (alpha, item) in enumerate(zip(plan["alphas"], middle)):
            raw_path = pair_out / alpha_dir(alpha) / "mesh_raw.pt"
            if raw_path.exists() and not args.overwrite:
                steps_skipped_output += 1
                continue
            mesh = item["mesh"][0]
            serial_start = time.perf_counter()
            save_raw_mesh(mesh, raw_path, {"method": "interp3d", "pair_id": pair_id, "step": step_idx, "alpha": float(alpha), "seed": pseed})
            serialization_seconds += time.perf_counter() - serial_start
            steps_generated += 1
            append_jsonl(shard_jsonl(run_root, args), {"method": "interp3d", "pair_id": pair_id, "step": step_idx, "alpha": float(alpha), "raw": str(raw_path)})
            log(f"[Interp3D] pair={pair_id:04d} step={step_idx + 1}/{plan['k_intermediate']} alpha={alpha:.6f} saved", run_root)

        cuda_sync()
        sequence_end_to_end_seconds = time.perf_counter() - pair_wall_start
        rss = process_rss_bytes()
        peak_rss = max(peak_rss, rss)
        timing_valid = steps_generated == int(plan["k_intermediate"]) and steps_skipped_output == 0
        append_timing(
            run_root,
            "interp3d",
            args,
            {
                "pair_id": pair_id,
                "pair_dir": pair["pair_dir"],
                "timing_valid": bool(timing_valid),
                "steps_generated": int(steps_generated),
                "steps_skipped_output": int(steps_skipped_output),
                "k_intermediate": int(plan["k_intermediate"]),
                "sequence_end_to_end_seconds": float(sequence_end_to_end_seconds),
                "intermediate_compute_seconds": float(intermediate_compute_seconds),
                "serialization_seconds": float(serialization_seconds),
                "seconds_per_intermediate_compute": float(intermediate_compute_seconds / max(1, int(plan["k_intermediate"]))),
                "process_peak_rss_gib": gib(peak_rss),
                "node_min_mem_available_gib": gib(node_mem_available_bytes()),
            },
        )
        log(
            f"[Interp3D] pair={pair_id:04d} sequence timing: end_to_end={sequence_end_to_end_seconds:.2f}s "
            f"compute={intermediate_compute_seconds:.2f}s serialization={serialization_seconds:.2f}s",
            run_root,
        )
        del outputs, middle
        torch.cuda.empty_cache()

    write_json(method_root / f"generation_status_{shard_tag(args)}.json", {"status": "complete", "finished_utc": utc_now(), "schedule": "uniform", "pairs": len(pairs), "shard_index": args.shard_index, "num_shards": args.num_shards})
    log(f"Interp3D generation complete | {shard_tag(args)}", run_root)


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
    # to_glb keeps several large CUDA buffers alive through local references.
    # Release them immediately so thousands of GLBs can be baked in one process.
    del textured, mesh
    torch.cuda.empty_cache()


def command_texture(args: argparse.Namespace) -> None:
    import torch
    import nvdiffrast.torch  # noqa: F401 - explicit preflight

    run_root = Path(args.run_root).resolve()
    plan = load_plan(run_root)
    pairs = shard_pairs(plan, args.shard_index, args.num_shards)
    refs_raw = run_root / "references_raw"
    refs_glb = run_root / "references_textured"
    refs_glb.mkdir(parents=True, exist_ok=True)

    total = len(pairs) * (2 + len(METHODS) * int(plan["k_intermediate"]))
    done = 0
    log(f"Common texture-baking start | {shard_tag(args)} | pairs_this_shard={len(pairs)}/{plan['num_pairs']} | expected GLBs={total} | texture={args.texture_size}px | simplify={args.simplify}", run_root)

    for pair in pairs:
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
        run_root / f"texture_status_{shard_tag(args)}.json",
        {
            "status": "complete",
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
            "pairs": len(pairs),
            "finished_utc": utc_now(),
            "texture_size": args.texture_size,
            "simplify": args.simplify,
            "fill_holes": bool(args.fill_holes),
            "appearance_source": "MeshExtractResult.vertex_attrs -> common TRELLIS to_glb(mesh, mesh)",
        },
    )
    torch.cuda.empty_cache()
    log(f"Common texture-baking complete | {shard_tag(args)}", run_root)


def command_texture_status(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).resolve()
    plan = load_plan(run_root)
    missing: List[str] = []
    for pair in plan["pairs"]:
        ref = run_root / "references_textured" / pair["pair_dir"]
        for name in ("src1.glb", "src2.glb"):
            if not (ref / name).is_file():
                missing.append(str(ref / name))
        for method in METHODS:
            pair_root = run_root / method / pair["pair_dir"]
            for alpha in plan["alphas"]:
                path = pair_root / alpha_dir(alpha) / "pred_final.glb"
                if not path.is_file():
                    missing.append(str(path))
    if missing:
        sample = "\n".join(missing[:20])
        raise FileNotFoundError(f"Texture phase incomplete: {len(missing)} GLBs missing. First entries:\n{sample}")
    write_json(run_root / "texture_status.json", {
        "status": "complete",
        "finished_utc": utc_now(),
        "num_pairs": int(plan["num_pairs"]),
        "k_intermediate": int(plan["k_intermediate"]),
        "appearance_source": "MeshExtractResult.vertex_attrs -> common TRELLIS to_glb(mesh, mesh)",
    })
    log("Texture completeness check passed", run_root)


def read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(dict(json.loads(line)))
    return rows


def numeric_stats(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "median": None, "count": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "count": int(arr.size),
    }


def aggregate_generation_timings(run_root: Path) -> Dict[str, Any]:
    timing_root = run_root / "timings"
    result: Dict[str, Any] = {
        "definition": (
            "Per-sequence timing excludes one-time model loading. sequence_end_to_end_seconds includes "
            "method-specific pair preparation, intermediate generation, and raw-mesh serialization; "
            "MorphFlow benchmark-only reference export is excluded. intermediate_compute_seconds is "
            "CUDA-synchronized model compute only. Only fresh complete sequences (timing_valid=true) "
            "enter aggregate timing statistics."
        ),
        "methods": {},
    }
    fields = (
        "sequence_end_to_end_seconds",
        "intermediate_compute_seconds",
        "serialization_seconds",
        "seconds_per_intermediate_compute",
        "process_peak_rss_gib",
        "node_min_mem_available_gib",
    )
    for method in METHODS:
        rows: List[Dict[str, Any]] = []
        for path in sorted(timing_root.glob(f"{method}_shard_*.jsonl")):
            rows.extend(read_jsonl_rows(path))
        valid = [row for row in rows if bool(row.get("timing_valid"))]
        aggregate = {
            field: numeric_stats([row.get(field) for row in valid if row.get(field) is not None])
            for field in fields
        }
        if method == "morphany3d":
            for field in (
                "conditioning_seconds",
                "endpoint_cache_and_staging_seconds",
                "tfsa_peak_ram_gib",
                "tfsa_peak_spill_gib",
                "tfsa_spill_written_gib",
                "tfsa_spill_read_gib",
            ):
                aggregate[field] = numeric_stats([row.get(field) for row in valid if row.get(field) is not None])
        result["methods"][method] = {
            "num_records": len(rows),
            "num_valid_sequences": len(valid),
            "aggregate": aggregate,
            "pairs": rows,
        }
    write_json(run_root / "generation_timings.json", result)

    csv_fields = [
        "method",
        "pair_id",
        "pair_dir",
        "timing_valid",
        "sequence_end_to_end_seconds",
        "intermediate_compute_seconds",
        "serialization_seconds",
        "seconds_per_intermediate_compute",
        "process_peak_rss_gib",
        "node_min_mem_available_gib",
        "conditioning_seconds",
        "endpoint_cache_and_staging_seconds",
        "tfsa_cache_mode",
        "tfsa_ram_limit_gib",
        "tfsa_peak_ram_gib",
        "tfsa_peak_spill_gib",
        "tfsa_spill_written_gib",
        "tfsa_spill_read_gib",
        "ram_pressure_guard_triggered",
    ]
    with (run_root / "generation_timings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for method in METHODS:
            for row in result["methods"][method]["pairs"]:
                writer.writerow(row)
    return result


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
        "metric_source": "eval_textured_benchmark.py over final textured GLBs; common uniform alpha schedule; appearance=materials",
        "methods": {},
    }
    generation_timings = aggregate_generation_timings(run_root)
    combined["generation_timings"] = generation_timings
    rows: List[Dict[str, Any]] = []
    headline = (
        # Paper-style full trajectory, INCLUDING src1/src2 endpoints.
        "lpips_adjacent_full_mean",
        "ppl_full_sum",
        "ppl_full_normalized_reference_endpoints",
        "pdv_full",
        # Endpoint-aware diagnostics retained from the original benchmark.
        "endpoint_fidelity_lpips_mean",
        "ppl_reference_anchored",
        # Interp3D-style endpoint-manifold FID.
        "fid_endpoint_manifold",
        # Original internal-only diagnostics retained for completeness.
        "lpips_adjacent_mean",
        "ppl_sum",
        "ppl_normalized_reference_endpoints",
        "pdv",
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

        timing_agg = generation_timings.get("methods", {}).get(method, {}).get("aggregate", {})
        timing_n = generation_timings.get("methods", {}).get(method, {}).get("num_valid_sequences", 0)
        for output_metric, timing_field in (
            ("generation_sequence_end_to_end_seconds", "sequence_end_to_end_seconds"),
            ("generation_intermediate_compute_seconds", "intermediate_compute_seconds"),
            ("generation_seconds_per_intermediate_compute", "seconds_per_intermediate_compute"),
        ):
            stats = timing_agg.get(timing_field, {})
            rows.append(
                {
                    "method": method,
                    "metric": output_metric,
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                    "median": stats.get("median"),
                    "num_pairs": timing_n,
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
        sp.add_argument("--shard-index", type=int, default=0)
        sp.add_argument("--num-shards", type=int, default=1)

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
    ma.add_argument("--tfsa-cache-mode", choices=["hybrid", "disk"], default="hybrid")
    ma.add_argument(
        "--tfsa-ram-gb",
        type=float,
        default=4.0,
        help="Per-worker TFSA CPU-RAM budget in hybrid mode; overflow spills to node-local storage.",
    )
    ma.add_argument(
        "--tfsa-min-node-free-gb",
        type=float,
        default=20.0,
        help="If node MemAvailable drops below this threshold, disable RAM caching for the current pair.",
    )
    ma.add_argument(
        "--local-cache-root",
        default="",
        help="Node-local directory for staged endpoint caches and TFSA spill files.",
    )

    ip = sub.add_parser("interp3d")
    common_run(ip)

    tx = sub.add_parser("texture")
    common_run(tx)
    tx.add_argument("--texture-size", type=int, default=1024)
    tx.add_argument("--simplify", type=float, default=0.95)
    tx.add_argument("--fill-holes", type=int, choices=[0, 1], default=1)
    tx.add_argument("--texture-verbose", type=int, choices=[0, 1], default=0)

    ts = sub.add_parser("texture-status")
    common_run(ts)

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
    elif command == "texture-status":
        command_texture_status(args)
    elif command == "aggregate":
        command_aggregate(args)
    else:
        raise ValueError(command)


if __name__ == "__main__":
    main()
