"""
Dataset/DataLoader for the canonical split-safe MorphAny3D distillation dataset.

Expected layout under root:
    assets/<name>/slat_feats.pt
    assets/<name>/slat_coords.pt
    assets/<name>/ss_latent.pt
    assets/<name>/occupancy.pt

    targets/<src_1>+<src_2>/alpha_<a>/slat_feats.pt
    targets/<src_1>+<src_2>/alpha_<a>/slat_coords.pt
    targets/<src_1>+<src_2>/alpha_<a>/ss_latent.pt
    targets/<src_1>+<src_2>/alpha_<a>/occupancy.pt

    metadata.json or metadata_<split>.json
    split_assets.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


VALID_SPLITS = ("train", "val", "test")
REQUIRED_METADATA_KEYS = (
    "split",
    "src_1",
    "src_2",
    "target",
    "alpha",
    "src1_slat_feats",
    "src1_slat_coords",
    "src1_ss_latent",
    "src2_slat_feats",
    "src2_slat_coords",
    "src2_ss_latent",
    "target_slat_feats",
    "target_slat_coords",
    "target_ss_latent",
)


class MorphingDistillDataset(Dataset):
    """Load precomputed MorphAny3D source/target latents with asset-disjoint splits."""

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
        self.assets_root = self.root / "assets"
        self.targets_root = self.root / "targets"
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
        if not self.metadata:
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
        splits = self.root / "splits.json"
        return splits if splits.is_file() else None

    def _load_asset_to_split(self, path: Optional[Path]) -> Dict[str, str]:
        if path is None:
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"split assets file must be a JSON object: {path}")
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

    def _root_join(self, value: Optional[str], fallback: Path) -> Path:
        if value is None:
            return fallback
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _entry_dirs(self, entry: Dict[str, Any]) -> Tuple[Path, Path, Path]:
        src1 = str(entry["src_1"])
        src2 = str(entry["src_2"])
        target = str(entry["target"])
        src1_dir = self._root_join(entry.get("src1_dir"), self.assets_root / src1)
        src2_dir = self._root_join(entry.get("src2_dir"), self.assets_root / src2)
        target_dir = self._root_join(entry.get("target_dir"), self.targets_root / target)
        return src1_dir, src2_dir, target_dir

    def _paths_for_entry(self, entry: Dict[str, Any]) -> Dict[str, Path]:
        src1_dir, src2_dir, target_dir = self._entry_dirs(entry)
        return {
            "src1_feats": self._root_join(entry.get("src1_slat_feats"), src1_dir / "slat_feats.pt"),
            "src1_coords": self._root_join(entry.get("src1_slat_coords"), src1_dir / "slat_coords.pt"),
            "src1_ss_latent": self._root_join(entry.get("src1_ss_latent"), src1_dir / "ss_latent.pt"),
            "src1_occupancy": self._root_join(entry.get("src1_occupancy"), src1_dir / "occupancy.pt"),
            "src2_feats": self._root_join(entry.get("src2_slat_feats"), src2_dir / "slat_feats.pt"),
            "src2_coords": self._root_join(entry.get("src2_slat_coords"), src2_dir / "slat_coords.pt"),
            "src2_ss_latent": self._root_join(entry.get("src2_ss_latent"), src2_dir / "ss_latent.pt"),
            "src2_occupancy": self._root_join(entry.get("src2_occupancy"), src2_dir / "occupancy.pt"),
            "target_feats": self._root_join(entry.get("target_slat_feats"), target_dir / "slat_feats.pt"),
            "target_coords": self._root_join(entry.get("target_slat_coords"), target_dir / "slat_coords.pt"),
            "target_ss_latent": self._root_join(entry.get("target_ss_latent"), target_dir / "ss_latent.pt"),
            "target_occupancy": self._root_join(entry.get("target_occupancy"), target_dir / "occupancy.pt"),
        }

    def _torch_load_safe(self, path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

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
            return self.asset_to_split.get(src1) == self.split and self.asset_to_split.get(src2) == self.split
        return True

    def _missing_paths_for_entry(self, entry: Dict[str, Any]) -> List[Path]:
        paths = self._paths_for_entry(entry)
        required_keys = [
            "src1_feats",
            "src1_coords",
            "src1_ss_latent",
            "src2_feats",
            "src2_coords",
            "src2_ss_latent",
            "target_feats",
            "target_coords",
            "target_ss_latent",
        ]
        if self.load_occupancy:
            required_keys.extend(["src1_occupancy", "src2_occupancy", "target_occupancy"])
        return [paths[key] for key in required_keys if not paths[key].is_file()]

    def _filter_valid_metadata(self, metadata: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid: List[Dict[str, Any]] = []
        skipped_schema = 0
        skipped_split = 0
        skipped_missing = 0
        examples: List[Tuple[str, List[Path]]] = []
        schema_examples: List[Tuple[int, List[str]]] = []

        for idx, entry in enumerate(metadata):
            missing_keys = [key for key in REQUIRED_METADATA_KEYS if key not in entry]
            if missing_keys:
                skipped_schema += 1
                if len(schema_examples) < 10:
                    schema_examples.append((idx, missing_keys))
                continue

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
                f"skipped_schema={skipped_schema} | "
                f"skipped_split={skipped_split} | skipped_missing={skipped_missing}"
            )
            for idx, missing_keys in schema_examples:
                print(
                    f"[MorphingDistillDataset][schema-skip] metadata_index={idx} "
                    f"missing_keys={missing_keys}"
                )
            for target_name, missing in examples:
                print(f"[MorphingDistillDataset][skip] target={target_name}")
                for path in missing:
                    print(f"  missing: {path}")

        return valid

    def _load_slat(self, paths: Dict[str, Path], prefix: str) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self._torch_load_safe(paths[f"{prefix}_feats"]).float()
        coords = self._torch_load_safe(paths[f"{prefix}_coords"]).int()
        return feats, coords

    def _load_tensor(self, paths: Dict[str, Path], key: str) -> torch.Tensor:
        return self._torch_load_safe(paths[key])

    def _load_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        paths = self._paths_for_entry(entry)
        src1_feats, src1_coords = self._load_slat(paths, "src1")
        src2_feats, src2_coords = self._load_slat(paths, "src2")
        target_feats, target_coords = self._load_slat(paths, "target")

        sample: Dict[str, Any] = {
            "src1_feats": src1_feats,
            "src1_coords": src1_coords,
            "src1_ss_latent": self._load_tensor(paths, "src1_ss_latent").float(),
            "src2_feats": src2_feats,
            "src2_coords": src2_coords,
            "src2_ss_latent": self._load_tensor(paths, "src2_ss_latent").float(),
            "target_feats": target_feats,
            "target_coords": target_coords,
            "target_ss_latent": self._load_tensor(paths, "target_ss_latent").float(),
            "alpha": torch.tensor(float(entry["alpha"]), dtype=torch.float32),
            "split": entry.get("split", self.split),
            "src1_name": str(entry["src_1"]),
            "src2_name": str(entry["src_2"]),
            "target_name": str(entry["target"]),
        }

        if self.load_occupancy:
            sample["src1_occupancy"] = self._load_tensor(paths, "src1_occupancy")
            sample["src2_occupancy"] = self._load_tensor(paths, "src2_occupancy")
            sample["target_occupancy"] = self._load_tensor(paths, "target_occupancy")

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
    if not values or any(value is None for value in values):
        return None
    try:
        return torch.stack(values, dim=0)
    except RuntimeError:
        return values


def morphing_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
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
