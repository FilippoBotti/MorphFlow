import os

os.environ['ATTN_BACKEND'] = 'xformers' 

import torch
from torch.utils.data import Dataset, DataLoader
from models import morph_flow
from data.morph_dataset import MorphingDistillDataset



x = torch.randn(1, 8, 16, 16, 16)
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

    diffusion = morph_flow.MorphFlow().cuda()
    y = diffusion(batch["target_ss_latent"].cuda(), batch["src1_feats"].cuda(), batch["src1_coords"].cuda(), batch["src2_feats"].cuda(), batch["src2_coords"].cuda(), batch["alpha"].cuda())
    print(y)
    exit()