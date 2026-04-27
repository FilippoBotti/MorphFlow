"""
Dataset and DataLoader for morphing distillation.

Each sample contains:
  - src_1 SLAT  (feats [N_s, 8], coords [N_s, 4])
  - src_2 SLAT  (feats [N_s, 8], coords [N_s, 4])
  - target SLAT (feats [N_m, 8], coords [N_m, 4])
  - target sparse structure latent
  - alpha
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader


class MorphingDistillDataset(Dataset):
    """
    Dataset that loads pre-computed SLATs produced by generate_morphing_dataset.py.

    Expected disk layout (relative to `root`):
        assets/<name>/slat_feats.pt, slat_coords.pt
        pairs_2/<pair_name>/mid_slat_feats.pt, mid_slat_coords.pt, mid_sparse_structure_latent.pt
        metadata.json

    Each entry in metadata.json is expected to contain at least:
        {
            "src_1": ...,
            "src_2": ...,
            "target": ...,
            "alpha": ...
        }
    """

    def __init__(
        self,
        root: str,
        metadata_file: str,
        max_num_voxels: int = 32768,
        split: str = None,
        skip_missing: bool = True,
        verbose: bool = True,
        exclude_assets=None,
    ):
        self.root = root
        self.source_root = os.path.join(root, "assets")
        self.pairs_root = os.path.join(root, "pairs_2")
        self.max_num_voxels = max_num_voxels
        self.split = split
        self.skip_missing = skip_missing
        self.verbose = verbose
        self.exclude_assets = set(exclude_assets or [])

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        if self.skip_missing:
            self.metadata = self._filter_valid_metadata(metadata)
        else:
            self.metadata = metadata

        if len(self.metadata) == 0:
            raise RuntimeError("No valid samples found in dataset after filtering missing files.")

    def __len__(self):
        return len(self.metadata)

    def _torch_load_safe(self, path: str):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _load_slat(self, slat_dir: str, prefix: str = "slat"):
        feats = self._torch_load_safe(os.path.join(slat_dir, f"{prefix}_feats.pt"))
        coords = self._torch_load_safe(os.path.join(slat_dir, f"{prefix}_coords.pt"))
        return feats.float(), coords.int()

    def _paths_for_entry(self, entry):
        src1_dir = os.path.join(self.source_root, entry["src_1"])
        src2_dir = os.path.join(self.source_root, entry["src_2"])
        target_dir = os.path.join(self.pairs_root, entry["target"])

        return {
            "src1_dir": src1_dir,
            "src2_dir": src2_dir,
            "target_dir": target_dir,
            "src1_feats": os.path.join(src1_dir, "slat_feats.pt"),
            "src1_coords": os.path.join(src1_dir, "slat_coords.pt"),
            "src2_feats": os.path.join(src2_dir, "slat_feats.pt"),
            "src2_coords": os.path.join(src2_dir, "slat_coords.pt"),
            "target_feats": os.path.join(target_dir, "mid_slat_feats.pt"),
            "target_coords": os.path.join(target_dir, "mid_slat_coords.pt"),
            "target_ss_latent": os.path.join(target_dir, "mid_sparse_structure_latent.pt"),
        }

    def _missing_paths_for_entry(self, entry):
        paths = self._paths_for_entry(entry)
        required = [
            paths["src1_feats"],
            paths["src1_coords"],
            paths["src2_feats"],
            paths["src2_coords"],
            paths["target_feats"],
            paths["target_coords"],
            paths["target_ss_latent"],
        ]
        return [p for p in required if not os.path.exists(p)]

    def _filter_valid_metadata(self, metadata):
        valid = []
        skipped = 0
        skipped_by_asset_filter = 0
        examples = []

        for entry in metadata:
            if self.exclude_assets and (
                entry.get("src_1") in self.exclude_assets or entry.get("src_2") in self.exclude_assets
            ):
                skipped_by_asset_filter += 1
                continue

            missing = self._missing_paths_for_entry(entry)
            if missing:
                skipped += 1
                if len(examples) < 10:
                    examples.append((entry["target"], missing))
                continue
            valid.append(entry)

        if self.verbose:
            print(
                f"[MorphingDistillDataset] valid samples: {len(valid)} | "
                f"skipped missing/corrupted candidates: {skipped} | "
                f"skipped by asset exclusion: {skipped_by_asset_filter}"
            )
            for target_name, missing in examples:
                print(f"[MorphingDistillDataset][skip] target={target_name}")
                for p in missing:
                    print(f"  missing: {p}")

        return valid

    def _load_entry(self, entry):
        paths = self._paths_for_entry(entry)

        src1_feats, src1_coords = self._load_slat(paths["src1_dir"], prefix="slat")
        src2_feats, src2_coords = self._load_slat(paths["src2_dir"], prefix="slat")
        target_feats, target_coords = self._load_slat(paths["target_dir"], prefix="mid_slat")
        target_sparse_structure_latent = self._torch_load_safe(paths["target_ss_latent"]).float()

        alpha = torch.tensor(entry["alpha"], dtype=torch.float32)

        sample = {
            "src1_feats": src1_feats,
            "src1_coords": src1_coords,
            "src2_feats": src2_feats,
            "src2_coords": src2_coords,
            "target_feats": target_feats,
            "target_coords": target_coords,
            "target_ss_latent": target_sparse_structure_latent,
            "alpha": alpha,
            "src1_name": entry["src_1"],
            "src2_name": entry["src_2"],
            "target_name": entry["target"],
        }
        return sample

    def __getitem__(self, idx):
        # Fallback robusto: se un file sparisce dopo la validazione iniziale,
        # prova i sample successivi invece di far crashare il training.
        max_attempts = min(32, len(self.metadata))

        for attempt in range(max_attempts):
            real_idx = (idx + attempt) % len(self.metadata)
            entry = self.metadata[real_idx]

            try:
                return self._load_entry(entry)
            except FileNotFoundError as e:
                if self.verbose:
                    print(
                        f"[MorphingDistillDataset][runtime-skip] "
                        f"idx={real_idx} target={entry.get('target', 'unknown')} "
                        f"reason={e}"
                    )
                continue

        raise RuntimeError(
            "Unable to load a valid sample after multiple attempts. "
            "Too many files are missing or inaccessible."
        )


def morphing_collate_fn(batch):
    """
    Sparse collate:
    - concatena feats/coords
    - scrive il batch index in coords[:, 0]
    """
    result = {}

    for prefix in ["src1", "src2", "target"]:
        all_feats = []
        all_coords = []
        lengths = []

        for i, sample in enumerate(batch):
            feats = sample[f"{prefix}_feats"]
            coords = sample[f"{prefix}_coords"].clone()
            coords[:, 0] = i

            all_feats.append(feats)
            all_coords.append(coords)
            lengths.append(feats.shape[0])

        result[f"{prefix}_feats"] = torch.cat(all_feats, dim=0)
        result[f"{prefix}_coords"] = torch.cat(all_coords, dim=0)
        result[f"{prefix}_lengths"] = torch.tensor(lengths, dtype=torch.long)

    result["target_ss_latent"] = torch.stack(
        [sample["target_ss_latent"] for sample in batch], dim=0
    )
    result["alpha"] = torch.stack(
        [sample["alpha"] for sample in batch], dim=0
    )
    result["src1_name"] = [sample["src1_name"] for sample in batch]
    result["src2_name"] = [sample["src2_name"] for sample in batch]
    result["target_name"] = [sample["target_name"] for sample in batch]

    return result


def build_dataloader(
    root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    max_num_voxels: int = 32768,
    shuffle: bool = True,
    drop_last: bool = False,
    split: str = None,
    metadata_file: str = None,
    skip_missing: bool = True,
    exclude_assets=None,
):
    dataset = MorphingDistillDataset(
        root=root,
        max_num_voxels=max_num_voxels,
        split=split,
        metadata_file=metadata_file,
        skip_missing=skip_missing,
        exclude_assets=exclude_assets,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=morphing_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )
    return dataset, loader


if __name__ == "__main__":
    root = "/home/filippo/datasets/3d/morphing_dataset_flux"
    meta_path = os.path.join(root, "metadata_2.json")

    dataset = MorphingDistillDataset(
        root=root,
        metadata_file=meta_path,
        skip_missing=True,
        verbose=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=morphing_collate_fn,
    )

    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("src1_feats shape:", batch["src1_feats"].shape)
        print("src1_coords shape:", batch["src1_coords"].shape)
        print("src2_feats shape:", batch["src2_feats"].shape)
        print("src2_coords shape:", batch["src2_coords"].shape)
        print("target_feats shape:", batch["target_feats"].shape)
        print("target_coords shape:", batch["target_coords"].shape)
        print("target_ss_latent shape:", batch["target_ss_latent"].shape)
        print("alpha shape:", batch["alpha"].shape)
        break