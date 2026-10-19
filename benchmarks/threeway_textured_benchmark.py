#!/usr/bin/env python3
"""Three-way MorphFlow / MorphAny3D / Interp3D benchmark.

Design goals
------------
* fixed test-set plan shared by all methods;
* MorphFlow consumes canonical source latents;
* MorphAny3D and Interp3D consume the corresponding source images;
* K intermediate states per pair;
* generation saves raw TRELLIS meshes and their genuine Gaussian appearance;
* paper evaluation renders the Gaussian appearance from shared spherical views;
* optional common TRELLIS texture baking produces textured GLB artifacts;
* eval_textured_benchmark.py adds full-path metrics and trajectory FID;
* resumable outputs and append-only progress log.

This script intentionally has several modes because the three methods live in
separate isolated Conda/Singularity environments. The companion Slurm script
runs each mode in the correct environment.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import json
import os
import random
import re
import shutil
import struct
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


METHODS = ("morphflow", "morphany3d", "interp3d")
DEFAULT_N = 100
DEFAULT_K = 5
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
    """Bounded CPU-RAM cache with transparent spill to explicit scratch storage.

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
        # torch.save snapshots values at the time of the write. CPU tensors may
        # otherwise alias views modified by the next attention call. Compact
        # clones also make logical tensor bytes a real upper bound on storage.
        def snapshot(obj: Any) -> Any:
            import torch
            if torch.is_tensor(obj):
                return obj.detach().to(device="cpu").clone(memory_format=torch.contiguous_format)
            if isinstance(obj, dict):
                return {name: snapshot(item) for name, item in obj.items()}
            if isinstance(obj, list):
                return [snapshot(item) for item in obj]
            if isinstance(obj, tuple):
                return tuple(snapshot(item) for item in obj)
            return obj
        value = snapshot(value)
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
        if isinstance(key, str):
            match = re.search(r"_morphing(\d+)(?:_|\.)", key)
            return int(match.group(1)) if match else None
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


class _ModuleProxy:
    """Override selected module attributes without changing global torch/os."""

    def __init__(self, module: Any, **overrides: Any):
        self._module = module
        self.__dict__.update(overrides)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


class TFSAFileCacheAdapter:
    """Translate upstream TFSA file I/O into a bounded mapping cache.

    Only the two attention modules and morphing pipeline receive private module
    proxies. Endpoint caches, raw outputs, and every other torch/os user retain
    normal file semantics. The upstream attention calculations stay unchanged.
    """

    _name = re.compile(r"(?:(?:ss|slat)_sa_morphing\d+_step\d+_block\d+|(?:feat_)?coords_morphing\d+)\.pt\Z")

    def __init__(self, cache: HybridTFSACache, work_dir: Path, min_free_bytes: int = 0):
        import torch
        self.cache = cache
        self.work_dir = Path(work_dir).absolute()
        self.min_free_bytes = max(0, int(min_free_bytes))
        self.pressure_relief_triggered = False
        self._torch = torch
        self._exists = os.path.exists

    def _key(self, path: Any) -> Optional[str]:
        try:
            path = Path(os.fspath(path)).absolute()
        except TypeError:
            return None
        return path.name if path.parent == self.work_dir and self._name.fullmatch(path.name) else None

    def exists(self, path: Any) -> bool:
        key = self._key(path)
        return key in self.cache if key is not None else self._exists(path)

    def save(self, value: Any, path: Any, *args: Any, **kwargs: Any) -> None:
        key = self._key(path)
        if key is None:
            return self._torch.save(value, path, *args, **kwargs)
        if self.min_free_bytes and self.cache.max_ram_bytes:
            available = node_mem_available_bytes()
            if available and available < self.min_free_bytes:
                self.cache.spill_all_ram()
                self.cache.max_ram_bytes = 0
                self.pressure_relief_triggered = True
        self.cache[key] = value

    def load(self, path: Any, *args: Any, **kwargs: Any) -> Any:
        key = self._key(path)
        if key is None:
            return self._torch.load(path, *args, **kwargs)
        if key not in self.cache:
            raise FileNotFoundError(path)
        # Upstream TFSA always stores detached CPU tensors and loads them before
        # explicitly moving to CUDA; no alternate map_location is used here.
        return self.cache.get(key)

    @contextmanager
    def install(self, modules: Optional[Sequence[Any]] = None):
        if modules is None:
            import importlib
            modules = [importlib.import_module(name) for name in (
                "trellis.modules.attention.modules",
                "trellis.modules.sparse.attention.modules",
                "trellis.pipelines.trellis_image_to_3d",
            )]
        saved = [(module, module.torch, module.os) for module in modules]
        try:
            for module, original_torch, original_os in saved:
                module.torch = _ModuleProxy(original_torch, save=self.save, load=self.load)
                module.os = _ModuleProxy(original_os, path=_ModuleProxy(original_os.path, exists=self.exists))
            yield self
        finally:
            for module, original_torch, original_os in saved:
                module.torch, module.os = original_torch, original_os


def safe_slug(value: str, limit: int = 36) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    if not value:
        value = "asset"
    return value[:limit]


def pair_dir_name(pair_id: int, src1: str, src2: str) -> str:
    return f"pair_{pair_id:04d}_{safe_slug(src1)}_to_{safe_slug(src2)}"


def alpha_dir(alpha: float) -> str:
    return f"alpha_{float(alpha):.6f}".replace(".", "p")


def canonical_alphas(k: int, schedule: str = "beta", beta_shape: float = 5.0) -> List[float]:
    if k < 1:
        raise ValueError("K must be >= 1")
    # Canonical MorphFlow convention: alpha=1 -> src1, alpha=0 -> src2.
    values = np.linspace(0.0, 1.0, k + 2, dtype=np.float64)[1:-1]
    if schedule == "beta":
        from scipy.stats import beta
        if beta_shape <= 0:
            raise ValueError("--beta-shape must be positive")
        values = beta.ppf(values, beta_shape, beta_shape)
    elif schedule != "uniform":
        raise ValueError(f"Unknown alpha schedule: {schedule}")
    values = 1.0 - values
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


GAUSSIAN_TENSOR_FIELDS = ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")


def raw_asset_ready(path: Path) -> bool:
    # Old benchmark outputs contain geometry alone and cannot support textured
    # evaluation. The marker is written only after the full asset is durable.
    return path.is_file() and path.with_suffix(".appearance.json").is_file()


def save_raw_mesh(mesh: Any, path: Path, metadata: Optional[Mapping[str, Any]] = None, gaussian: Any = None) -> None:
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
        "format_version": 2,
    }
    if gaussian is None:
        raise ValueError("Textured evaluation requires the Gaussian appearance decoded from the same SLat")
    payload["gaussian"] = {
        "init_params": gaussian.init_params,
        "active_sh_degree": int(gaussian.active_sh_degree),
        "tensors": {
            name: getattr(gaussian, name).detach().float().cpu() if getattr(gaussian, name) is not None else None
            for name in GAUSSIAN_TENSOR_FIELDS
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)
    write_json(path.with_suffix(".appearance.json"), {"format_version": 2, "appearance": "trellis_gaussian"})


def load_raw_gaussian(path: Path, device: str = "cuda") -> Any:
    from trellis.representations import Gaussian
    payload = torch_load_cpu(path)
    if "gaussian" not in payload:
        raise ValueError(f"Missing genuine Gaussian appearance in {path}; regenerate this asset")
    state = payload["gaussian"]
    gaussian = Gaussian(**state["init_params"], device=device)
    for name in GAUSSIAN_TENSOR_FIELDS:
        value = state["tensors"][name]
        setattr(gaussian, name, value.to(device) if value is not None else None)
    gaussian.active_sh_degree = int(state["active_sh_degree"])
    return gaussian


def decode_slat_gaussian(decoder: Any, sparse_tensor_cls: Any, feats: Any, coords: Any, device: Any) -> Any:
    import torch
    coords = coords.to(device=device, dtype=torch.int32)
    if coords.shape[1] == 3:
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
    coords = coords.clone()
    coords[:, 0] = 0
    sparse = sparse_tensor_cls(feats=feats.to(device=device, dtype=torch.float32), coords=coords)
    with torch.no_grad():
        return decoder(sparse)[0]


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


def select_test_pairs(dataset: Any, num_pairs: int, seed: int) -> Tuple[List[Tuple[int, Dict[str, Any]]], int]:
    """Select unique test pairs without ever filtering unavailable benchmark inputs.

    Metadata may contain multiple target states for each source pair. They are
    not ground truth for this evaluation, so only source latents and source
    images are required. Indexes refer to the original unfiltered metadata.
    """
    if int(num_pairs) < 0:
        raise ValueError("--num-pairs must be >= 0; zero selects the entire test set")
    unique: List[Tuple[int, Dict[str, Any]]] = []
    seen = set()
    for index, entry in enumerate(dataset.metadata):
        required = ("src_1", "src_2", "target")
        missing_keys = [key for key in required if key not in entry]
        if missing_keys:
            raise ValueError(f"Invalid test metadata row {index}: missing {missing_keys}")
        if not dataset._entry_allowed_by_split(entry):
            raise ValueError(f"Metadata row {index} is outside the strict test split; use test-only metadata")
        key = (str(entry["src_1"]), str(entry["src_2"]))
        if key not in seen:
            seen.add(key)
            unique.append((index, dict(entry)))
    available = len(unique)
    requested = available if int(num_pairs) == 0 else int(num_pairs)
    if requested < 1 or requested > available:
        raise RuntimeError(f"Requested N={num_pairs}, but {available} unique test pairs are available")
    positions = np.random.default_rng(seed).choice(available, size=requested, replace=False).tolist()
    chosen = [unique[int(position)] for position in positions]
    failures: List[str] = []
    for index, entry in chosen:
        paths = dataset._paths_for_entry(entry)
        for key in ("src1_feats", "src1_coords", "src1_ss_latent", "src2_feats", "src2_coords", "src2_ss_latent"):
            if not paths[key].is_file():
                failures.append(f"metadata row {index}: missing {key}: {paths[key]}")
        for prefix, asset in (("src1", str(entry["src_1"])), ("src2", str(entry["src_2"]))):
            if dataset._find_source_image_path(asset, entry, prefix) is None:
                failures.append(f"metadata row {index}: missing {prefix} image for {asset}")
    if failures:
        raise FileNotFoundError(
            f"Selected test set has {len(failures)} missing inputs; no pairs were dropped.\n" + "\n".join(failures[:30])
        )
    return chosen, available


def command_plan(args: argparse.Namespace) -> None:
    from data.morph_dataset import MorphingDistillDataset

    protocol = getattr(args, "protocol", "interp3d")
    if protocol == "weighted-endpoints-50" and (args.k != 48 or args.alpha_schedule != "uniform"):
        raise ValueError("weighted-endpoints-50 requires --k 48 --alpha-schedule uniform (48 intermediates + 2 canonical sources)")
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    plan_path = run_root / "plan.json"
    if plan_path.exists() and not args.overwrite:
        existing = read_json(plan_path)
        if existing.get("protocol", "interp3d") != protocol:
            raise ValueError("Existing plan uses a different protocol; use a fresh run root")
        expected_requested_n = existing.get("requested_num_pairs", existing.get("num_pairs"))
        if int(args.num_pairs) != int(expected_requested_n):
            raise ValueError(f"Existing plan requested {expected_requested_n} pairs, received {args.num_pairs}; use a fresh run root")
        requested = {"seed": int(args.seed), "k_intermediate": int(args.k),
                     "data_root": str(Path(args.data_root).resolve()), "metadata": str(args.metadata),
                     "source_images_root": str(Path(args.source_images_root).resolve()),
                     "alphas": canonical_alphas(args.k, args.alpha_schedule, args.beta_shape)}
        if "source_image_filename" in existing:
            requested["source_image_filename"] = args.source_image_filename
        mismatches = {key: (existing.get(key), value) for key, value in requested.items() if existing.get(key) != value}
        if mismatches:
            raise ValueError(f"Existing plan differs from requested benchmark: {mismatches}; use a fresh run root")
        log(f"PLAN already exists, keeping it: {plan_path}", run_root)
        return

    dataset = MorphingDistillDataset(
        root=args.data_root,
        metadata_file=args.metadata,
        split="test",
        strict_split=True,
        load_occupancy=False,
        skip_missing=False,
        verbose=True,
        load_source_images=False,
        source_images_root=args.source_images_root,
        source_image_filename=args.source_image_filename,
    )

    chosen, available_pairs = select_test_pairs(dataset, args.num_pairs, args.seed)
    alphas = canonical_alphas(args.k, args.alpha_schedule, args.beta_shape)

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
        "num_pairs": len(chosen),
        "requested_num_pairs": int(args.num_pairs),
        "available_test_pairs": available_pairs,
        "metadata_index_basis": "unfiltered",
        "source_image_filename": args.source_image_filename,
        "k_intermediate": int(args.k),
        "protocol": protocol,
        "metric_mode": "weighted-endpoints-only" if protocol == "weighted-endpoints-50" else "interp3d",
        "num_sequence_frames": int(args.k) + 2,
        "split": "test",
        "data_root": str(Path(args.data_root).resolve()),
        "metadata": str(args.metadata),
        "source_images_root": str(Path(args.source_images_root).resolve()),
        "alpha_convention": "alpha=1 -> src1; alpha=0 -> src2",
        "alphas": alphas,
        "alpha_schedule": args.alpha_schedule,
        "beta_shape": args.beta_shape if args.alpha_schedule == "beta" else None,
        "evaluation_protocol": {
            "source": "https://arxiv.org/html/2601.14103v1#A2.SS2",
            "scope": "Interp3D metric definitions on the MorphFlow held-out test split; not Interp3DData table reproduction",
            "paper_sequence_states": 7,
            "paper_beta_shape": 5,
            "upstream_code_beta_shape": 6,
            "render_backend": "gaussian",
            "perceptual_views": 64,
            "fid_views_per_state": 1000,
            "fid_intermediate_selection": "two seeded random states without replacement, shared across methods",
            "morphflow_noise_policy": "fixed pair seed across alpha and CFG cells; separate SS and SLat noise streams",
        },
        "interp3d_internal_t": [float(round(1.0 - a, 10)) for a in alphas],
        "pairs": pairs,
    }
    if protocol == "weighted-endpoints-50":
        payload["sequence_alphas"] = [1.0] + alphas + [0.0]
        payload["evaluation_protocol"] = {
            "scope": "Additional weighted source-proximity evaluation on the same held-out MorphFlow test pairs; separate from the Interp3D paper protocol",
            "name": "weighted-endpoints-50", "metric_mode": "weighted-endpoints-only",
            "sequence_states": 50, "generated_intermediate_states": 48,
            "alpha_schedule": "uniform", "sequence_alphas": payload["sequence_alphas"],
            "perceptual_views": 64, "lpips_backbone": "vgg", "render_backend": "gaussian",
            "canonical_endpoints": "true source renders reused as sequence endpoints; weighted distance exactly zero at alpha 1 and 0",
            "fid": "not computed; official FID/PPL/PDV remain in the separate 7-state Interp3D run",
            "morphflow_noise_policy": "fixed pair seed across alpha and CFG cells; separate SS and SLat noise streams",
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
        detect_flow_target,
        detect_model_type,
        load_checkpoint,
        load_decoders,
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
    ss_decoder, mesh_decoder, sparse_cls = load_decoders("ss", device)
    from trellis.models import from_pretrained as trellis_from_pretrained
    gaussian_decoder = trellis_from_pretrained(
        "microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16"
    ).to(device).eval()

    dataset = MorphingDistillDataset(
        root=args.data_root,
        metadata_file=args.metadata,
        split="test",
        strict_split=True,
        load_occupancy=False,
        skip_missing=plan.get("metadata_index_basis") != "unfiltered",
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

        ref_pair = refs_root / pair["pair_dir"]
        ref_pair.mkdir(parents=True, exist_ok=True)
        src1_raw = ref_pair / "src1_raw.pt"
        src2_raw = ref_pair / "src2_raw.pt"
        ref_start = time.perf_counter()
        if not raw_asset_ready(src1_raw):
            mesh = decode_slat_mesh(mesh_decoder, sparse_cls, src1["feats"], src1["coords"], device)
            save_raw_mesh(mesh, src1_raw, {"kind": "reference", "asset": pair["src1"]}, gaussian=decode_slat_gaussian(gaussian_decoder, sparse_cls, src1["feats"], src1["coords"], device))
            del mesh
            torch.cuda.empty_cache()
        if not raw_asset_ready(src2_raw):
            mesh = decode_slat_mesh(mesh_decoder, sparse_cls, src2["feats"], src2["coords"], device)
            save_raw_mesh(mesh, src2_raw, {"kind": "reference", "asset": pair["src2"]}, gaussian=decode_slat_gaussian(gaussian_decoder, sparse_cls, src2["feats"], src2["coords"], device))
            del mesh
            torch.cuda.empty_cache()
        cuda_sync()
        reference_export_seconds = time.perf_counter() - ref_start

        template_ss = torch.zeros_like(src1["ss_latent"])
        for step_idx, alpha in enumerate(plan["alphas"]):
            step_out = pair_out / alpha_dir(alpha)
            raw_path = step_out / "mesh_raw.pt"
            if raw_asset_ready(raw_path) and not args.overwrite:
                steps_skipped_output += 1
                log(f"[MorphFlow] pair={pair_id:04d} step={step_idx + 1}/{plan['k_intermediate']} skip existing", run_root)
                continue

            # Morphing evaluates a trajectory. Independent noise at every alpha
            # measures resampling jitter and unfairly inflates LPIPS/PPL relative
            # to the two baselines, which both reuse their pair noise stream.
            ss_seed = pair_seed(plan["seed"], pair_id)
            slat_seed = ss_seed + 10_000
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
            gaussian = decode_slat_gaussian(gaussian_decoder, sparse_cls, pred_slat.feats, pred_slat.coords, device)
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
                gaussian=gaussian,
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
            del pred_ss, pred_coords, pred_slat, mesh, gaussian, batch
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


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += int(item.stat().st_size)
            except OSError:
                pass
    return total


def prune_morphany_file_cache(path: Path, keep_morphing_idx: int) -> int:
    """Keep only the upstream MorphAny3D TFSA state needed by the next alpha.

    The official implementation stores SS/SLat TFSA tensors as files whose names
    contain ``morphing<idx>``.  After alpha i has finished, only morphing i is
    required by alpha i+1, so older states can be deleted without changing the
    algorithm.  Returns the number of bytes removed.
    """
    removed = 0
    keep = int(keep_morphing_idx)
    if not path.exists():
        return 0
    for item in list(path.glob("*morphing*.pt")):
        match = re.search(r"morphing(\d+)", item.name)
        if match is None or int(match.group(1)) == keep:
            continue
        try:
            size = int(item.stat().st_size)
        except OSError:
            size = 0
        try:
            item.unlink()
            removed += size
        except FileNotFoundError:
            pass
    return removed


def copy_directory_files(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in src.iterdir():
        if not item.is_file():
            continue
        target = dst / item.name
        shutil.copy2(item, target)
        try:
            copied += int(target.stat().st_size)
        except OSError:
            pass
    return copied


def morphany_tfsa_spill_directory(args: argparse.Namespace, cache_root: Path, pair_dir: str) -> Path:
    """Use explicit spill storage only when requested, always scoped per pair.

    The final tfsa_spill component prevents clearing endpoint caches even if a
    caller deliberately selects the persistent method cache as its spill root.
    """
    if not pair_dir or Path(pair_dir).name != pair_dir or pair_dir in {".", ".."}:
        raise ValueError(f"Unsafe pair directory for TFSA cache: {pair_dir!r}")
    configured = getattr(args, "tfsa_spill_root", "")
    parent = Path(configured).resolve() if configured else Path(cache_root)
    return parent / pair_dir / "tfsa_spill"


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
    pipeline.sparse_structure_sampler_params["steps"] = args.steps
    pipeline.slat_sampler_params["steps"] = args.slat_steps
    log(
        "MorphAny3D generation start | "
        f"pairs_this_shard={len(pairs)}/{plan['num_pairs']} | K={plan['k_intermediate']} | {shard_tag(args)} | "
        f"tfsa_cache_mode={args.tfsa_cache_mode} | tfsa_ram_limit={args.tfsa_ram_gb:.2f} GiB | "
        f"min_node_free={args.tfsa_min_node_free_gb:.2f} GiB | local_cache={local_cache_root} | "
        f"tfsa_spill_root={getattr(args, 'tfsa_spill_root', '') or cache_root}",
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
        if all(raw_asset_ready(path) for path in raw_paths) and not args.overwrite:
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

            # Route only TFSA files into a genuinely bounded CPU-RAM cache.
            # Default overflow storage remains persistent. A launcher may choose
            # its explicitly provisioned node-local directory independently of
            # endpoint-cache staging; never infer storage capacity from /tmp.
            # The current and preceding alpha coexist until the alpha completes.
            spill_dir = morphany_tfsa_spill_directory(args, cache_root, pair["pair_dir"])
            shutil.rmtree(spill_dir, ignore_errors=True)
            ram_budget = int(args.tfsa_ram_gb * (1024 ** 3)) if args.tfsa_cache_mode == "hybrid" else 0
            tfsa_cache = HybridTFSACache(ram_budget, spill_dir)
            adapter = TFSAFileCacheAdapter(tfsa_cache, work_dir, int(args.tfsa_min_node_free_gb * (1024 ** 3)))
            active_work_dir = work_dir
            tfsa_pruned_bytes = 0

            for step_idx, alpha in enumerate(plan["alphas"]):
                step_out = pair_out / alpha_dir(alpha)
                raw_path = step_out / "mesh_raw.pt"
                output_exists = raw_asset_ready(raw_path) and not args.overwrite

                # If an incomplete pair is resumed, MorphAny3D must still replay
                # earlier alphas to rebuild the previous-state TFSA cache chain.
                # Existing raw meshes are left untouched; only the compute is replayed.
                params = {
                    "morphing_num": int(plan["k_intermediate"]) + 2,
                    "src_load_cache_path": str(load_src_cache),
                    "tar_load_cache_path": str(load_tar_cache),
                    "save_cache_path": str(active_work_dir),
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
                available = node_mem_available_bytes()
                if available:
                    if min_node_available == 0:
                        min_node_available = available
                    else:
                        min_node_available = min(min_node_available, available)
                seed_everything(pseed)
                cuda_sync()
                compute_start = time.perf_counter()
                with adapter.install():
                    outputs = pipeline.run_morphing(
                        src_pre,
                        tar_pre,
                        morphing_params=params,
                        seed=pseed,
                        formats=["mesh", "gaussian"],
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
                        gaussian=outputs["gaussian"][0],
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

                retained_before_prune = tfsa_cache.ram_bytes + tfsa_cache.spill_bytes
                tfsa_cache.prune_to_morphing_idx(step_idx + 1)
                retained_after_prune = tfsa_cache.ram_bytes + tfsa_cache.spill_bytes
                tfsa_pruned_bytes += max(0, retained_before_prune - retained_after_prune)
                stats = tfsa_cache.stats()
                pressure_relief_triggered = adapter.pressure_relief_triggered
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
            cache_stats = tfsa_cache.stats()
            cache_stats["pruned_gib"] = gib(tfsa_pruned_bytes)
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
                    "tfsa_pruned_gib": float(cache_stats.get("pruned_gib", 0.0)),
                    "tfsa_backend": "bounded_ram_with_scratch_spill" if args.tfsa_cache_mode == "hybrid" else "scratch_files",
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
            try:
                shutil.rmtree(cache_root / pair["pair_dir"] / "work_fallback", ignore_errors=True)
            except Exception:
                pass
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

    # Honor the explicit shared plan. The paper specifies Beta(5,5), while
    # current upstream hard-codes Beta(6,6); do not silently change coefficients.
    def shared_alpha_list(coords0: Any, coords1: Any, steps: int = 7):
        if steps != int(plan["k_intermediate"]) + 2:
            raise ValueError("Interp3D requested a schedule inconsistent with plan.json")
        return torch.tensor([0.0] + [1.0 - float(a) for a in plan["alphas"]] + [1.0], device=coords0.device)

    interp_module.adaptive_alpha_list_by_chamfer = shared_alpha_list

    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    pipeline.sparse_structure_sampler_params["steps"] = args.steps
    pipeline.slat_sampler_params["steps"] = args.slat_steps

    def reset_interp3d_attention_state():
        # Interp3D leaves its flow-model self-attention in "interpolate"
        # after a morph sequence. Reusing the same pipeline for the next
        # pair would therefore feed alpha=None through interpolate attention.
        for model_name in ("sparse_structure_flow_model", "slat_flow_model"):
            if model_name not in pipeline.models:
                continue

            model = pipeline.models[model_name]

            for attr, value in (
                ("mode", "normal"),
                ("alpha", 0.5),
                ("step_id0", 0),
                ("step_id1", 0),
                ("step_id2", 0),
            ):
                if hasattr(model, attr):
                    setattr(model, attr, value)

            for cache_name in ("kv_dict_img0", "kv_dict_img1"):
                cache = getattr(model, cache_name, None)
                if cache is not None and hasattr(cache, "clear"):
                    cache.clear()

            for block in getattr(model, "blocks", []):
                attn = getattr(block, "self_attn", None)
                if attn is None:
                    continue
                if hasattr(attn, "attn_phase"):
                    attn.attn_phase = "normal"
                if hasattr(attn, "alpha"):
                    attn.alpha = 0.5
                if hasattr(attn, "step_id"):
                    attn.step_id = 0
    log(f"Interp3D generation start | shared plan schedule | pairs_this_shard={len(pairs)}/{plan['num_pairs']} | K={plan['k_intermediate']} | {shard_tag(args)}", run_root)

    for pidx, pair in enumerate(pairs):
        reset_interp3d_attention_state()
        pair_wall_start = time.perf_counter()
        intermediate_compute_seconds = 0.0
        serialization_seconds = 0.0
        peak_rss = process_rss_bytes()
        pair_id = int(pair["pair_id"])
        pseed = int(pair["pair_seed"])
        pair_out = method_root / pair["pair_dir"]
        pair_out.mkdir(parents=True, exist_ok=True)

        all_exist = all(raw_asset_ready(pair_out / alpha_dir(a) / "mesh_raw.pt") for a in plan["alphas"])
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
            formats=["mesh", "gaussian"],
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
            if raw_asset_ready(raw_path) and not args.overwrite:
                steps_skipped_output += 1
                continue
            mesh = item["mesh"][0]
            serial_start = time.perf_counter()
            save_raw_mesh(mesh, raw_path, {"method": "interp3d", "pair_id": pair_id, "step": step_idx, "alpha": float(alpha), "seed": pseed}, gaussian=item["gaussian"][0])
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

    write_json(method_root / f"generation_status_{shard_tag(args)}.json", {"status": "complete", "finished_utc": utc_now(), "schedule": plan.get("alpha_schedule", "uniform"), "pairs": len(pairs), "shard_index": args.shard_index, "num_shards": args.num_shards})
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


def glb_validation_error(path: Path) -> Optional[str]:
    """Validate the GLB container, JSON scene, and referenced embedded buffer.

    A file existing after an interrupted export is not a completed artifact.
    Parsing the chunk boundaries also catches truncation beyond the header.
    This CPU-only check does not import TRELLIS or load geometry into memory.
    """
    try:
        with path.open("rb") as handle:
            actual_size = path.stat().st_size
            header = handle.read(12)
            if len(header) != 12:
                return "truncated GLB header"
            magic, version, declared_size = struct.unpack("<4sII", header)
            if magic != b"glTF" or version != 2:
                return "not a GLB version 2 file"
            if declared_size != actual_size or actual_size < 20:
                return f"GLB byte length mismatch: header={declared_size}, actual={actual_size}"
            offset, document, binary_size = 12, None, 0
            while offset < actual_size:
                chunk_header = handle.read(8)
                if len(chunk_header) != 8:
                    return "truncated GLB chunk header"
                chunk_size, chunk_type = struct.unpack("<II", chunk_header)
                offset += 8
                if chunk_size == 0 or chunk_size % 4 or offset + chunk_size > actual_size:
                    return "invalid or truncated GLB chunk"
                if document is None:
                    if chunk_type != 0x4E4F534A:
                        return "first GLB chunk is not JSON"
                    document = json.loads(handle.read(chunk_size).decode("utf-8"))
                else:
                    if chunk_type == 0x004E4942:
                        binary_size += chunk_size
                    handle.seek(chunk_size, os.SEEK_CUR)
                offset += chunk_size
            if not isinstance(document, dict) or document.get("asset", {}).get("version") != "2.0":
                return "missing glTF 2.0 asset metadata"
            if not document.get("meshes"):
                return "GLB contains no mesh"
            for buffer in document.get("buffers", []):
                if "uri" not in buffer and int(buffer.get("byteLength", 0)) > binary_size:
                    return "embedded buffer is truncated or missing"
    except (OSError, ValueError, TypeError, AttributeError, UnicodeError, struct.error) as exc:
        return str(exc)
    return None


def glb_ready(path: Path) -> bool:
    return glb_validation_error(path) is None


@contextmanager
def artifact_lock(path: Path) -> Iterable[None]:
    """Serialize shared reference baking even when CFG roots use symlinks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def temporary_glb_path(path: Path) -> Path:
    # Keep the .glb suffix because trimesh chooses its exporter from it.
    return path.with_name(f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.glb")


def atomic_export_glb(exporter: Callable[[Path], Any], path: Path, overwrite: bool = False) -> bool:
    """Export then validate and rename; return False only for a valid reuse."""
    with artifact_lock(path):
        if not overwrite and glb_ready(path):
            return False
        temporary = temporary_glb_path(path)
        try:
            exporter(temporary)
            error = glb_validation_error(temporary)
            if error:
                raise ValueError(f"Invalid GLB export for {path}: {error}")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return True


def hardlink_or_copy(src: Path, dst: Path, overwrite: bool = False) -> None:
    if not glb_ready(src):
        raise ValueError(f"Invalid source GLB {src}: {glb_validation_error(src)}")
    def transfer(temporary: Path) -> None:
        try:
            os.link(src, temporary)
        except OSError:
            shutil.copy2(src, temporary)
    atomic_export_glb(transfer, dst, overwrite=overwrite)


def texture_one(raw_path: Path, glb_path: Path, args: argparse.Namespace, run_root: Path, label: str) -> None:
    def export(temporary: Path) -> None:
        import torch
        from trellis.utils import postprocessing_utils
        if not raw_asset_ready(raw_path):
            raise FileNotFoundError(f"Missing complete raw mesh and Gaussian appearance: {raw_path}")
        log(f"[TEXTURE] start {label}", run_root)
        mesh = load_raw_as_trellis_mesh(raw_path, "cuda")
        gaussian = load_raw_gaussian(raw_path, "cuda")
        textured = postprocessing_utils.to_glb(
            gaussian,
            mesh,
            simplify=args.simplify,
            fill_holes=bool(args.fill_holes),
            texture_size=args.texture_size,
            verbose=bool(args.texture_verbose),
        )
        textured.export(temporary)
        # Release CUDA buffers between the thousands of exports in a full run.
        del textured, mesh, gaussian
        torch.cuda.empty_cache()
    if atomic_export_glb(export, glb_path, overwrite=bool(args.overwrite)):
        log(f"[TEXTURE] done  {label} -> {glb_path}", run_root)


def selected_methods(args: argparse.Namespace) -> Tuple[str, ...]:
    methods = tuple(getattr(args, "methods", None) or METHODS)
    if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        raise ValueError(f"--methods must contain distinct methods chosen from {METHODS}")
    return methods


def command_texture(args: argparse.Namespace) -> None:
    import torch
    import nvdiffrast.torch  # noqa: F401 - explicit preflight

    run_root = Path(args.run_root).resolve()
    plan = load_plan(run_root)
    methods = selected_methods(args)
    pairs = shard_pairs(plan, args.shard_index, args.num_shards)
    refs_raw = run_root / "references_raw"
    refs_glb = run_root / "references_textured"
    refs_glb.mkdir(parents=True, exist_ok=True)

    total = len(pairs) * (2 + len(methods) * int(plan["k_intermediate"]))
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

        for method in methods:
            pair_out = run_root / method / pair["pair_dir"]
            if not pair_out.is_dir():
                raise FileNotFoundError(f"Missing generated pair directory: {pair_out}")
            hardlink_or_copy(src1_glb, pair_out / "src1.glb", overwrite=bool(args.overwrite))
            hardlink_or_copy(src2_glb, pair_out / "src2.glb", overwrite=bool(args.overwrite))

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
                    "appearance": "common TRELLIS Gaussian appearance texture baking",
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
            "methods": list(methods),
            "finished_utc": utc_now(),
            "texture_size": args.texture_size,
            "simplify": args.simplify,
            "fill_holes": bool(args.fill_holes),
            "appearance_source": "SLat Gaussian appearance -> common TRELLIS to_glb(gaussian, mesh)",
        },
    )
    torch.cuda.empty_cache()
    log(f"Common texture-baking complete | {shard_tag(args)}", run_root)


def command_texture_status(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).resolve()
    plan = load_plan(run_root)
    methods = selected_methods(args)
    invalid: List[str] = []
    checked = 0
    def check(path: Path) -> None:
        nonlocal checked
        checked += 1
        error = glb_validation_error(path)
        if error:
            invalid.append(f"{path}: {error}")
    for pair in plan["pairs"]:
        ref = run_root / "references_textured" / pair["pair_dir"]
        for name in ("src1.glb", "src2.glb"):
            check(ref / name)
        for method in methods:
            pair_root = run_root / method / pair["pair_dir"]
            for name in ("src1.glb", "src2.glb"):
                check(pair_root / name)
            for alpha in plan["alphas"]:
                path = pair_root / alpha_dir(alpha) / "pred_final.glb"
                check(path)
    if invalid:
        sample = "\n".join(invalid[:20])
        raise FileNotFoundError(f"Texture phase incomplete: {len(invalid)}/{checked} GLBs missing or invalid. First entries:\n{sample}")
    status_name = "texture_status.json" if methods == METHODS else f"texture_status_{'_'.join(methods)}.json"
    write_json(run_root / status_name, {
        "status": "complete",
        "finished_utc": utc_now(),
        "num_pairs": int(plan["num_pairs"]),
        "k_intermediate": int(plan["k_intermediate"]),
        "methods": list(methods),
        "checked_glbs": checked,
        "appearance_source": "SLat Gaussian appearance -> common TRELLIS to_glb(gaussian, mesh)",
    })
    log(f"Texture completeness check passed | methods={','.join(methods)} | valid GLBs={checked}", run_root)


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
    weighted_only = plan.get("metric_mode") == "weighted-endpoints-only"
    metrics_root = run_root / "metrics"
    combined: Dict[str, Any] = {
        "created_utc": utc_now(),
        "seed": plan["seed"],
        "num_pairs": plan["num_pairs"],
        "k_intermediate": plan["k_intermediate"],
        "alphas": plan["alphas"],
        "metric_source": "eval_textured_benchmark.py; common explicit alpha schedule; genuine SLat Gaussian appearance",
        "evaluation_protocol": plan.get("evaluation_protocol", {}),
        "alpha_schedule": plan.get("alpha_schedule", "uniform"),
        "protocol": plan.get("protocol", "interp3d"),
        "metric_mode": plan.get("metric_mode", "interp3d"),
        "weighted_endpoint_lpips_definition": {
            "formula": "alpha * LPIPS(frame, src1) + (1-alpha) * LPIPS(frame, src2)",
            "alpha_convention": "alpha=1 selects src1; alpha=0 selects src2",
            "aggregation": "equal frame weight after averaging views, then equal pair weight",
            "auc": "trapezoidal integral over the observed alpha interval, without extrapolation",
            "interpretation": "source proximity; additional diagnostic, not a standalone smoothness or realism metric",
        },
        "methods": {},
    }
    if weighted_only:
        combined["num_sequence_frames"] = plan["k_intermediate"] + 2
        combined["sequence_alphas"] = [1.0] + plan["alphas"] + [0.0]
        combined["weighted_endpoint_lpips_definition"].update({
            "curve": "all 50 states: two actual canonical source endpoints plus 48 uniformly spaced intermediates",
            "mean": "weighted_endpoint_lpips_mean averages the 48 generated intermediates only",
            "auc": "weighted_endpoint_lpips_auc integrates over the observed intermediate interval, excluding canonical endpoints",
            "full_mean": "weighted_endpoint_lpips_full_mean averages all 50 states, including zero distance at canonical endpoints",
            "full_auc": "weighted_endpoint_lpips_full_auc integrates the complete curve on [0,1] including canonical endpoints",
        })
    generation_timings = aggregate_generation_timings(run_root)
    combined["generation_timings"] = generation_timings
    rows: List[Dict[str, Any]] = []
    headline = (
        "weighted_endpoint_lpips_mean",
        "weighted_endpoint_lpips_auc",
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
        "fid_endpoint_manifold_mean_per_state",
        "fid_endpoint_manifold_pooled",
        # Original internal-only diagnostics retained for completeness.
        "lpips_adjacent_mean",
        "ppl_sum",
        "ppl_normalized_reference_endpoints",
        "pdv",
    )
    if weighted_only:
        headline = ("weighted_endpoint_lpips_mean", "weighted_endpoint_lpips_auc",
                    "weighted_endpoint_lpips_full_mean", "weighted_endpoint_lpips_full_auc")

    for method in METHODS:
        path = metrics_root / method / "metrics_batch.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing batch metrics for {method}: {path}")
        data = read_json(path)
        if weighted_only and (data.get("metric_mode") != "weighted-endpoints-only" or
                              data.get("num_generated_frames") != 48 or data.get("num_sequence_frames") != 50 or
                              data.get("num_views_per_pair") != 64 or data.get("lpips_backbone") != "vgg" or
                              data.get("alphas") != [1.0] + plan["alphas"] + [0.0]):
            raise ValueError(f"Weighted 50-state evaluation is incomplete or uses a different protocol: {path}")
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
    plan.add_argument("--num-pairs", type=int, default=DEFAULT_N, help="Number of unique test pairs; 0 evaluates all test pairs without dropping missing inputs")
    plan.add_argument("--k", type=int, default=DEFAULT_K)
    plan.add_argument("--seed", type=int, default=DEFAULT_SEED)
    plan.add_argument("--alpha-schedule", choices=["beta", "uniform"], default="beta")
    plan.add_argument("--beta-shape", type=float, default=5.0)
    plan.add_argument("--protocol", choices=["interp3d", "weighted-endpoints-50"], default="interp3d",
                      help="Use a separate run root for the additional 50-state weighted LPIPS evaluation")

    mf = sub.add_parser("morphflow")
    common_run(mf)
    mf.add_argument("--data-root", required=True)
    mf.add_argument("--metadata", default="metadata_test.json")
    mf.add_argument("--source-images-root", required=True)
    mf.add_argument("--source-image-filename", default="")
    mf.add_argument("--checkpoint-path", required=True)
    mf.add_argument("--slat-checkpoint-path", required=True)
    mf.add_argument("--steps", type=int, default=25)
    mf.add_argument("--slat-steps", type=int, default=25)
    mf.add_argument("--cfg-scale", type=float, default=3.0)
    mf.add_argument("--slat-cfg-scale", type=float, default=3.0)
    mf.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    mf.add_argument("--trellis-model", choices=["auto", "text_base", "image_large"], default="image_large")

    ma = sub.add_parser("morphany3d")
    common_run(ma)
    ma.add_argument("--steps", type=int, default=25)
    ma.add_argument("--slat-steps", type=int, default=25)
    ma.add_argument("--tfsa-cache-mode", choices=["hybrid", "disk"], default="hybrid")
    ma.add_argument(
        "--tfsa-ram-gb",
        type=float,
        default=4.0,
        help="Per-worker TFSA CPU-RAM budget in hybrid mode; overflow uses --tfsa-spill-root or persistent method cache.",
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
        help="Node-local directory for staged endpoint caches and upstream work files; select overflow storage separately with --tfsa-spill-root.",
    )
    ma.add_argument(
        "--tfsa-spill-root",
        default="",
        help="Explicit TFSA overflow storage; uses a private per-pair subdirectory. Default: run-root/method_cache/morphany3d.",
    )

    ip = sub.add_parser("interp3d")
    common_run(ip)
    ip.add_argument("--steps", type=int, default=25)
    ip.add_argument("--slat-steps", type=int, default=25)

    tx = sub.add_parser("texture")
    common_run(tx)
    tx.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    tx.add_argument("--texture-size", type=int, default=1024)
    tx.add_argument("--simplify", type=float, default=0.95)
    tx.add_argument("--fill-holes", type=int, choices=[0, 1], default=1)
    tx.add_argument("--texture-verbose", type=int, choices=[0, 1], default=0)

    ts = sub.add_parser("texture-status")
    common_run(ts)
    ts.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))

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
