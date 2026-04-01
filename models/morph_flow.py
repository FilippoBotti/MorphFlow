import torch
from torch import nn as nn
import torch.nn.functional as F

from models import sparse_structure_flow
from models import cond_encoder

class MorphFlow(nn.Module):
    def __init__(self, sigma_min=1e-5,):
        super().__init__()
        self.cond_encoder = cond_encoder.BlockPoolConditionEncoder()
        self.cond_fusion = cond_encoder.PairConditionFusion()

        self.sparse_structure_flow = sparse_structure_flow.SparseStructureFlowModel(
                resolution=16,
                in_channels=8,
                out_channels=8,
                model_channels=256,
                cond_channels=256,
                num_blocks=12,
                num_heads=16,
                mlp_ratio=4,
                patch_size=1,
                pe_mode="ape",
                qk_rms_norm=True,
                use_fp16=True
            )
        self.sigma_min = sigma_min
        
    def get_v(self, x_0, noise, t):
        return (1 - self.sigma_min) * noise - x_0
    
    def diffuse(self, x_0, t):
        noise = torch.randn_like(x_0)

        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t, noise
    
    def forward_flow(self, x_t, t, src_1_feats, src_2_feats, src_1_coords, src_2_coords, alpha):
        # condition
        cond1 = self.cond_encoder(src_1_feats, src_1_coords)
        cond2 = self.cond_encoder(src_2_feats, src_2_coords)
        cond = self.cond_fusion(cond1, cond2, alpha)

        # diffusion
        out = self.sparse_structure_flow(x_t, t, cond)

        return out

    def forward(self, x_0, src_1_feats, src_1_coords, src_2_feats, src_2_coords, alpha):
        B = x_0.shape[0]
        x_0 = x_0.squeeze(1)
        t = torch.rand(B).to(x_0.device).float()
        x_t, noise = self.diffuse(x_0, t)

        velocity = self.get_v(x_0, noise, t)

        pred = self.forward_flow(x_t, t, src_1_feats, src_2_feats, src_1_coords, src_2_coords, alpha)

        loss = F.mse_loss(pred, velocity)
        return loss