"""
Dataset and DataLoader for morphing distillation.

Each sample contains:
  - src_1 SLAT  (feats [N_s, 8], coords [N_s, 4])
  - src_2 SLAT  (feats [N_s, 8], coords [N_s, 4])
  - target SLAT  (feats [N_m, 8], coords [N_m, 4])   ← ground-truth from MorphAny3D
  - alpha     (float, typically 0.5)
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
        pairs/<src_1>+<src_2>/mid_slat_feats.pt, mid_slat_coords.pt
        metadata.json

    Each entry in metadata.json:
        { "src_1", "src_2", "target", "src_1_slat_dir", "src_2_slat_dir", "target_slat_dir", "alpha" }
    """

    def __init__(
        self,
        root: str,
        metadata_file: str,
        max_num_voxels: int = 32768,
        split: str = None,

    ):
        self.root = root
        self.source_root = os.path.join(root, "assets")
        self.pairs_root = os.path.join(root, "pairs_2")
        self.max_num_voxels = max_num_voxels
        self.split = split
    
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def _torch_load_safe(self, path: str):
        # Prefer safer loading on newer PyTorch, fallback for older versions.
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _load_slat(self, slat_dir: str, prefix: str = "slat"):
        feats = self._torch_load_safe(
            os.path.join(self.root, slat_dir, f"{prefix}_feats.pt")
        )
        coords = self._torch_load_safe(
            os.path.join(self.root, slat_dir, f"{prefix}_coords.pt")
        )

        return feats.float(), coords.int()


    def __getitem__(self, idx):
        entry = self.metadata[idx]

        src1_dir = os.path.join(self.source_root, entry["src_1"])
        src2_dir = os.path.join(self.source_root, entry["src_2"])
        target_dir = os.path.join(self.pairs_root, entry["target"])

        # Load source / target / midpoint SLATs
        src1_feats, src1_coords = self._load_slat(src1_dir)
        src2_feats, src2_coords = self._load_slat(src2_dir)
        target_feats, target_coords = self._load_slat(target_dir, prefix="mid_slat")
        target_sparse_structure_latent = self._torch_load_safe(os.path.join(self.root, target_dir, f"mid_sparse_structure_latent.pt"))

        alpha = torch.tensor(entry["alpha"], dtype=torch.float32)

        sample = {
            "src1_feats": src1_feats,      # [N_s1, C]
            "src1_coords": src1_coords,    # [N_s1, 4]
            "src2_feats": src2_feats,      # [N_s2, C]
            "src2_coords": src2_coords,    # [N_s2, 4]
            "target_feats": target_feats,  # [N_t, C]
            "target_coords": target_coords, # [N_t, 4],
            "target_ss_latent": target_sparse_structure_latent,
            "alpha": alpha
        }

        return sample

       

def morphing_collate_fn(batch):
    """
    Custom collate: sparse tensors have variable length so we
    concatenate them, adjusting the batch index in coords[:, 0].
    Alpha is stacked normally.
    """
    keys_sparse = ["src1", "src2", "target"]
    result = {}

    for prefix in keys_sparse:
        all_feats = []
        all_coords = []
        lengths = []
        for i, sample in enumerate(batch):
            feats = sample[f"{prefix}_feats"]
            coords = sample[f"{prefix}_coords"].clone()
            coords[:, 0] = i  # set batch index
            all_feats.append(feats)
            all_coords.append(coords)
            lengths.append(feats.shape[0])
        result[f"{prefix}_feats"] = torch.cat(all_feats, dim=0)
        result[f"{prefix}_coords"] = torch.cat(all_coords, dim=0)
        result[f"{prefix}_lengths"] = torch.tensor(lengths, dtype=torch.long)

    result["alpha"] = torch.stack([s["alpha"] for s in batch])

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
):
    dataset = MorphingDistillDataset(
        root=root,
        max_num_voxels=max_num_voxels,
        split=split,
        metadata_file=metadata_file,
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
    # Quick test to verify dataset loading
    root = "/home/filippo/datasets/3d/morphing_dataset_flux"
    meta_path = os.path.join(root, "metadata_2.json")
    md = MorphingDistillDataset(root=root, metadata_file=meta_path)
    
    dataloader = DataLoader(md, batch_size=1)

    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("src1_feats shape:", batch["src1_feats"].shape)
        print("src1 coords shape: ", batch["src1_coords"].shape)
        print("target_feats shape:", batch["target_feats"].shape)
        print("target ssl: ", batch['target_ss_latent'].shape)
        print("src2_feats shape:", batch["src2_feats"].shape)
        print("src1_coords shape:", batch["src1_coords"].shape)
        print("alpha shape:", batch["alpha"].shape)
        print("Sample alpha:", batch["alpha"])


        break

    