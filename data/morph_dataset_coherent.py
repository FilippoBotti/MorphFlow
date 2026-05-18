"""
Dataset/DataLoader for the split-safe MorphAny3D distillation dataset.

This loader is aligned with generate_morphany3d_split_dataset.py.

Expected modern layout:
    assets/<name>/structured_latent.pt          # dict: {"feats": Tensor, "coords": Tensor}
    assets/<name>/ss_latent.pt
    assets/<name>/occupancy.pt
    targets/<src_1>+<src_2>/alpha_<a>/structured_latent.pt
    targets/<src_1>+<src_2>/alpha_<a>/ss_latent.pt
    targets/<src_1>+<src_2>/alpha_<a>/occupancy.pt
    metadata.json or metadata_<split>.json
    splits.json

For backward compatibility, it can also read the older layout used by the
uploaded morph_dataset.py:
    assets/<name>/slat_feats.pt, slat_coords.pt
    pairs_2/<target>/mid_slat_feats.pt, mid_slat_coords.pt,
    pairs_2/<target>/mid_sparse_structure_latent.pt
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

VALID_SPLITS = ("train", "val", "test")


class MorphingDistillDataset(Dataset):
    """
    Load pre-computed MorphAny3D source/target latents.

    Args:
        root: Dataset root.
        metadata_file: Optional metadata JSON. If omitted and split is provided,
            metadata_<split>.json is preferred; otherwise metadata.json is used.
        split: Optional one of train/val/test. When set, only samples whose two
            source objects belong to that split are loaded.
        split_assets_file: Optional split_assets.json. Used to enforce object-level
            separation even if metadata contains mixed entries.
        strict_split: If True, raise/skip samples whose metadata conflicts with
            split_assets.json. If False, only metadata split is used.
        load_occupancy: If True, include src/target occupancy grids when present.
        skip_missing: If True, missing/corrupt entries are filtered at init time.
        exclude_assets/include_assets: Optional asset-name filters.
    """

    def __init__(
        self,
        root: str,
        metadata_file: Optional[str] = None,
        max_num_voxels: Optional[int] = 32768,
        split: Optional[str] = None,
        split_assets_file: Optional[str] = None,
        strict_split: bool = True,
        load_occupancy: bool = True,
        skip_missing: bool = True,
        verbose: bool = True,
        exclude_assets: Optional[Iterable[str]] = None,
        include_assets: Optional[Iterable[str]] = None,
    ):
        self.root = Path(root)
        self.source_root = self.root / "assets"
        self.legacy_pairs_root = self.root / "pairs_2"
        self.max_num_voxels = max_num_voxels
        self.split = split
        self.strict_split = bool(strict_split)
        self.load_occupancy = bool(load_occupancy)
        self.skip_missing = bool(skip_missing)
        self.verbose = bool(verbose)
        self.exclude_assets = set(exclude_assets or [])
        self.include_assets = set(include_assets or [])

        if split is not None and split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

        self.metadata_path = self._resolve_metadata_path(metadata_file, split)
        self.split_assets_path = self._resolve_split_assets_path(split_assets_file)
        self.asset_to_split = self._load_asset_to_split(self.split_assets_path)

        with self.metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if isinstance(metadata, dict):
            metadata = metadata.get("samples", metadata.get("metadata", []))
        if not isinstance(metadata, list):
            raise ValueError(f"Metadata must be a list or contain a samples list: {self.metadata_path}")

        self.metadata = self._filter_valid_metadata(metadata) if self.skip_missing else list(metadata)
        if len(self.metadata) == 0:
            raise RuntimeError("No valid samples found after filtering metadata.")

    def __len__(self) -> int:
        return len(self.metadata)

    def _resolve_metadata_path(self, metadata_file: Optional[str], split: Optional[str]) -> Path:
        if metadata_file is not None:
            path = Path(metadata_file)
            return path if path.is_absolute() else self.root / path
        if split is not None and (self.root / f"metadata_{split}.json").is_file():
            return self.root / f"metadata_{split}.json"
        return self.root / "metadata.json"

    def _resolve_split_assets_path(self, split_assets_file: Optional[str]) -> Optional[Path]:
        if split_assets_file is not None:
            path = Path(split_assets_file)
            return path if path.is_absolute() else self.root / path
        default = self.root / "split_assets.json"
        if default.is_file():
            return default
        legacy = self.root / "splits.json"
        return legacy if legacy.is_file() else None

    def _load_asset_to_split(self, path: Optional[Path]) -> Dict[str, str]:
        if path is None:
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"split_assets must be a JSON object: {path}")
        if "assets" in data and isinstance(data["assets"], dict):
            data = data["assets"]
        out: Dict[str, str] = {}
        for split, names in data.items():
            if split not in VALID_SPLITS:
                continue
            if not isinstance(names, list):
                raise ValueError(f"split_assets[{split!r}] must be a list")
            for name in names:
                name = str(name)
                if name in out and out[name] != split:
                    raise ValueError(f"Asset {name!r} appears in multiple splits: {out[name]!r}, {split!r}")
                out[name] = split
        return out

    def _torch_load_safe(self, path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(path, map_location="cpu")

    def _root_join(self, value: Optional[str], fallback: Path) -> Path:
        if value is None:
            return fallback
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _entry_dirs(self, entry: Dict[str, Any]) -> Tuple[Path, Path, Path]:
        src1_dir = self._root_join(entry.get("src1_dir"), self.source_root / str(entry["src_1"]))
        src2_dir = self._root_join(entry.get("src2_dir"), self.source_root / str(entry["src_2"]))
        target_dir = self._root_join(entry.get("target_dir"), self.root / "targets" / str(entry["target"]))
        return src1_dir, src2_dir, target_dir

    def _paths_for_entry(self, entry: Dict[str, Any]) -> Dict[str, Path]:
        src1_dir, src2_dir, target_dir = self._entry_dirs(entry)
        return {
            "src1_dir": src1_dir,
            "src2_dir": src2_dir,
            "target_dir": target_dir,
            "src1_structured": self._root_join(entry.get("src1_structured_latent"), src1_dir / "structured_latent.pt"),
            "src2_structured": self._root_join(entry.get("src2_structured_latent"), src2_dir / "structured_latent.pt"),
            "target_structured": self._root_join(entry.get("target_structured_latent"), target_dir / "structured_latent.pt"),
            "src1_ss_latent": self._root_join(entry.get("src1_ss_latent"), src1_dir / "ss_latent.pt"),
            "src2_ss_latent": self._root_join(entry.get("src2_ss_latent"), src2_dir / "ss_latent.pt"),
            "target_ss_latent": self._root_join(entry.get("target_ss_latent"), target_dir / "ss_latent.pt"),
            "src1_occupancy": self._root_join(entry.get("src1_occupancy"), src1_dir / "occupancy.pt"),
            "src2_occupancy": self._root_join(entry.get("src2_occupancy"), src2_dir / "occupancy.pt"),
            "target_occupancy": self._root_join(entry.get("target_occupancy"), target_dir / "occupancy.pt"),
            # Legacy fallback paths.
            "legacy_src1_feats": src1_dir / "slat_feats.pt",
            "legacy_src1_coords": src1_dir / "slat_coords.pt",
            "legacy_src1_ss_latent": src1_dir / "cache" / "coords_zs_init.pt",
            "legacy_src2_feats": src2_dir / "slat_feats.pt",
            "legacy_src2_coords": src2_dir / "slat_coords.pt",
            "legacy_src2_ss_latent": src2_dir / "cache" / "coords_zs_init.pt",
            "legacy_target_feats": target_dir / "mid_slat_feats.pt",
            "legacy_target_coords": target_dir / "mid_slat_coords.pt",
            "legacy_target_ss_latent": target_dir / "mid_sparse_structure_latent.pt",
        }

    def _structured_exists(self, paths: Dict[str, Path], key: str, legacy_feats: str, legacy_coords: str) -> bool:
        return paths[key].is_file() or (paths[legacy_feats].is_file() and paths[legacy_coords].is_file())

    def _ss_exists(self, paths: Dict[str, Path], key: str, legacy_key: str) -> bool:
        return paths[key].is_file() or paths[legacy_key].is_file()

    def _missing_paths_for_entry(self, entry: Dict[str, Any]) -> List[Path]:
        paths = self._paths_for_entry(entry)
        missing: List[Path] = []
        checks = [
            ("src1_structured", "legacy_src1_feats", "legacy_src1_coords"),
            ("src2_structured", "legacy_src2_feats", "legacy_src2_coords"),
            ("target_structured", "legacy_target_feats", "legacy_target_coords"),
        ]
        for key, legacy_feats, legacy_coords in checks:
            if not self._structured_exists(paths, key, legacy_feats, legacy_coords):
                missing.append(paths[key])
        ss_checks = [
            ("src1_ss_latent", "legacy_src1_ss_latent"),
            ("src2_ss_latent", "legacy_src2_ss_latent"),
            ("target_ss_latent", "legacy_target_ss_latent"),
        ]
        for key, legacy_key in ss_checks:
            if not self._ss_exists(paths, key, legacy_key):
                missing.append(paths[key])
        if self.load_occupancy:
            for key in ("src1_occupancy", "src2_occupancy", "target_occupancy"):
                # Occupancy did not exist in the old layout, so it is optional.
                pass
        return missing

    def _entry_allowed_by_split(self, entry: Dict[str, Any]) -> bool:
        src1 = str(entry.get("src_1"))
        src2 = str(entry.get("src_2"))

        if self.exclude_assets and (src1 in self.exclude_assets or src2 in self.exclude_assets):
            return False
        if self.include_assets and (src1 not in self.include_assets or src2 not in self.include_assets):
            return False

        if self.split is None:
            return True

        metadata_split = entry.get("split")
        if metadata_split is not None and metadata_split != self.split:
            return False

        if self.strict_split and self.asset_to_split:
            s1 = self.asset_to_split.get(src1)
            s2 = self.asset_to_split.get(src2)
            if s1 != self.split or s2 != self.split:
                return False
        return True

    def _filter_valid_metadata(self, metadata: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid: List[Dict[str, Any]] = []
        skipped_split = 0
        skipped_missing = 0
        examples: List[Tuple[str, List[Path]]] = []

        for entry in metadata:
            if not self._entry_allowed_by_split(entry):
                skipped_split += 1
                continue
            missing = self._missing_paths_for_entry(entry)
            if missing:
                skipped_missing += 1
                if len(examples) < 10:
                    examples.append((str(entry.get("target", "unknown")), missing))
                continue
            valid.append(dict(entry))

        if self.verbose:
            print(
                f"[MorphingDistillDataset] split={self.split} valid={len(valid)} | "
                f"skipped_split={skipped_split} | skipped_missing={skipped_missing}"
            )
            for target_name, missing in examples:
                print(f"[MorphingDistillDataset][skip] target={target_name}")
                for p in missing:
                    print(f"  missing: {p}")
        return valid

    def _load_structured(self, paths: Dict[str, Path], key: str, legacy_feats: str, legacy_coords: str) -> Tuple[torch.Tensor, torch.Tensor]:
        path = paths[key]
        if path.is_file():
            data = self._torch_load_safe(path)
            if isinstance(data, dict):
                feats = data.get("feats", data.get("features"))
                coords = data.get("coords")
            elif isinstance(data, (list, tuple)) and len(data) == 2:
                feats, coords = data
            elif hasattr(data, "feats") and hasattr(data, "coords"):
                feats, coords = data.feats, data.coords
            else:
                raise TypeError(f"Unsupported structured latent payload in {path}: {type(data)}")
            if feats is None or coords is None:
                raise KeyError(f"structured_latent.pt must contain feats and coords: {path}")
            return feats.float(), coords.int()

        feats = self._torch_load_safe(paths[legacy_feats]).float()
        coords = self._torch_load_safe(paths[legacy_coords]).int()
        return feats, coords

    def _load_ss_latent(self, paths: Dict[str, Path], key: str, legacy_key: str) -> torch.Tensor:
        path = paths[key] if paths[key].is_file() else paths[legacy_key]
        return self._torch_load_safe(path).float()

    def _load_optional_tensor(self, path: Path) -> Optional[torch.Tensor]:
        if not path.is_file():
            return None
        return self._torch_load_safe(path)

    def _load_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        paths = self._paths_for_entry(entry)
        src1_feats, src1_coords = self._load_structured(paths, "src1_structured", "legacy_src1_feats", "legacy_src1_coords")
        src2_feats, src2_coords = self._load_structured(paths, "src2_structured", "legacy_src2_feats", "legacy_src2_coords")
        target_feats, target_coords = self._load_structured(paths, "target_structured", "legacy_target_feats", "legacy_target_coords")

        sample: Dict[str, Any] = {
            "src1_feats": src1_feats,
            "src1_coords": src1_coords,
            "src1_ss_latent": self._load_ss_latent(paths, "src1_ss_latent", "legacy_src1_ss_latent"),
            "src2_feats": src2_feats,
            "src2_coords": src2_coords,
            "src2_ss_latent": self._load_ss_latent(paths, "src2_ss_latent", "legacy_src2_ss_latent"),
            "target_feats": target_feats,
            "target_coords": target_coords,
            "target_ss_latent": self._load_ss_latent(paths, "target_ss_latent", "legacy_target_ss_latent"),
            "alpha": torch.tensor(float(entry["alpha"]), dtype=torch.float32),
            "split": entry.get("split", self.split),
            "src1_name": str(entry["src_1"]),
            "src2_name": str(entry["src_2"]),
            "target_name": str(entry["target"]),
        }

        if self.load_occupancy:
            sample["src1_occupancy"] = self._load_optional_tensor(paths["src1_occupancy"])
            sample["src2_occupancy"] = self._load_optional_tensor(paths["src2_occupancy"])
            sample["target_occupancy"] = self._load_optional_tensor(paths["target_occupancy"])

        return sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        max_attempts = min(32, len(self.metadata))
        for attempt in range(max_attempts):
            real_idx = (idx + attempt) % len(self.metadata)
            entry = self.metadata[real_idx]
            try:
                return self._load_entry(entry)
            except (FileNotFoundError, RuntimeError, KeyError, TypeError) as exc:
                if self.verbose:
                    print(
                        f"[MorphingDistillDataset][runtime-skip] idx={real_idx} "
                        f"target={entry.get('target', 'unknown')} reason={exc}"
                    )
                continue
        raise RuntimeError("Unable to load a valid sample after multiple attempts.")


def _stack_optional(batch: Sequence[Dict[str, Any]], key: str):
    values = [sample.get(key) for sample in batch]
    if not values or any(v is None for v in values):
        return None
    try:
        return torch.stack(values, dim=0)
    except RuntimeError:
        return values


def morphing_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sparse collate:
      - concatenates feats/coords for src1, src2 and target
      - rewrites coords[:, 0] as the batch index
      - stacks dense sparse-structure latents
    """
    result: Dict[str, Any] = {}

    for prefix in ("src1", "src2", "target"):
        all_feats: List[torch.Tensor] = []
        all_coords: List[torch.Tensor] = []
        lengths: List[int] = []

        for i, sample in enumerate(batch):
            feats = sample[f"{prefix}_feats"]
            coords = sample[f"{prefix}_coords"].clone()
            coords[:, 0] = i
            all_feats.append(feats)
            all_coords.append(coords)
            lengths.append(int(feats.shape[0]))

        result[f"{prefix}_feats"] = torch.cat(all_feats, dim=0)
        result[f"{prefix}_coords"] = torch.cat(all_coords, dim=0)
        result[f"{prefix}_lengths"] = torch.tensor(lengths, dtype=torch.long)

    for key in ("src1_ss_latent", "src2_ss_latent", "target_ss_latent", "alpha"):
        result[key] = torch.stack([sample[key] for sample in batch], dim=0)

    for key in ("src1_occupancy", "src2_occupancy", "target_occupancy"):
        stacked = _stack_optional(batch, key)
        if stacked is not None:
            result[key] = stacked

    result["split"] = [sample.get("split") for sample in batch]
    result["src1_name"] = [sample["src1_name"] for sample in batch]
    result["src2_name"] = [sample["src2_name"] for sample in batch]
    result["target_name"] = [sample["target_name"] for sample in batch]
    return result


def build_dataloader(
    root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    max_num_voxels: Optional[int] = 32768,
    shuffle: bool = True,
    drop_last: bool = False,
    split: Optional[str] = None,
    metadata_file: Optional[str] = None,
    split_assets_file: Optional[str] = None,
    strict_split: bool = True,
    load_occupancy: bool = True,
    skip_missing: bool = True,
    exclude_assets: Optional[Iterable[str]] = None,
    include_assets: Optional[Iterable[str]] = None,
):
    dataset = MorphingDistillDataset(
        root=root,
        metadata_file=metadata_file,
        max_num_voxels=max_num_voxels,
        split=split,
        split_assets_file=split_assets_file,
        strict_split=strict_split,
        load_occupancy=load_occupancy,
        skip_missing=skip_missing,
        exclude_assets=exclude_assets,
        include_assets=include_assets,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=morphing_collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )
    return dataset, loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test a MorphAny3D distillation dataset loader.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="train", choices=VALID_SPLITS)
    parser.add_argument("--metadata-file", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    dataset, loader = build_dataloader(
        root=args.root,
        split=args.split,
        metadata_file=args.metadata_file,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
    )
    print(f"samples: {len(dataset)}")
    for batch in loader:
        print("Batch keys:", sorted(batch.keys()))
        print("src1_feats:", tuple(batch["src1_feats"].shape))
        print("src1_coords:", tuple(batch["src1_coords"].shape))
        print("src2_feats:", tuple(batch["src2_feats"].shape))
        print("src2_coords:", tuple(batch["src2_coords"].shape))
        print("target_feats:", tuple(batch["target_feats"].shape))
        print("target_coords:", tuple(batch["target_coords"].shape))
        print("target_ss_latent:", tuple(batch["target_ss_latent"].shape))
        print("alpha:", batch["alpha"])
        break
