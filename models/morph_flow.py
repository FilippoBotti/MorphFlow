import torch
from torch import nn as nn
import torch.nn.functional as F

from models import sparse_structure_flow
from models import cond_encoder

class MorphFlow(nn.Module):
    def __init__(self, sigma_min=1e-5, model_type="text_base", separate_cond=False, use_checkpoint=False):
        super().__init__()
        
        self.separate_cond = separate_cond

        if model_type == "text_base":
            model_channels = 768
            num_blocks = 12
            num_heads = 12
        elif model_type == "image_large":
            model_channels = 1024
            num_blocks = 24
            num_heads = 16
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.cond_encoder = cond_encoder.BlockPoolConditionEncoder()
        self.cond_fusion = cond_encoder.PairConditionFusionV2(
            cond_dim=128,
            alpha_dim=64,
            hidden_dim=512,
            out_dim=model_channels, 
        )
        if self.separate_cond:
            self.separate_cond_proj = nn.Linear(128, model_channels)

        self.cfg_drop_prob = 0.0
        self.null_cond = nn.Parameter(
            torch.zeros(1, self.cond_encoder.num_blocks, model_channels)
        )

        self.sparse_structure_flow = sparse_structure_flow.SparseStructureFlowModel(
            resolution=16,
            in_channels=8,
            out_channels=8,
            model_channels=model_channels, 
            cond_channels=model_channels, 
            num_blocks=num_blocks,    
            num_heads=num_heads,  
            mlp_ratio=4,
            patch_size=1,
            pe_mode="ape",
            qk_rms_norm=True,
            use_fp16=False,
            use_checkpoint=use_checkpoint,
            separate_cond=separate_cond        
        )
        self.sigma_min = sigma_min
        
    def get_v(self, x_0, noise):
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
        
        if not self.separate_cond:
            cond = self.cond_fusion(cond1, cond2, alpha)
        else:
            cond1 = self.separate_cond_proj(cond1)
            cond2 = self.separate_cond_proj(cond2)
            cond = (cond1, cond2, alpha)

        if self.training and self.cfg_drop_prob > 0.0:
            B = cond1.shape[0] if self.separate_cond else cond.shape[0]
            drop_mask = torch.rand(B, device=cond1.device) < self.cfg_drop_prob

            null_cond = self.null_cond.expand(B, -1, -1).to(dtype=cond1.dtype)

            if not self.separate_cond:
                cond = torch.where(
                    drop_mask.view(B, 1, 1),
                    null_cond,
                    cond,
                )
            else:
                drop_mask = drop_mask.view(B, 1, 1)
                cond = (
                    torch.where(drop_mask, null_cond, cond1),
                    torch.where(drop_mask, null_cond, cond2),
                    alpha
                )

        # diffusion
        t_flow = t.float() * 1000.0
        out = self.sparse_structure_flow(x_t, t_flow, cond, alpha=alpha)

        return out
    
    def forward_flow_cfg(self, x_t, t, src_1_feats, src_2_feats, src_1_coords, src_2_coords, alpha, guidance_scale=1.0):
        if guidance_scale == 1.0:
            return self.forward_flow(
                x_t,
                t,
                src_1_feats,
                src_2_feats,
                src_1_coords,
                src_2_coords,
                alpha,
            )

        cond1 = self.cond_encoder(src_1_feats, src_1_coords)
        cond2 = self.cond_encoder(src_2_feats, src_2_coords)
        
        if not self.separate_cond:
            cond = self.cond_fusion(cond1, cond2, alpha)
            B = cond.shape[0]
            null_cond = self.null_cond.expand(B, -1, -1).to(dtype=cond.dtype)
        else:
            cond1 = self.separate_cond_proj(cond1)
            cond2 = self.separate_cond_proj(cond2)
            cond = (cond1, cond2, alpha)
            B = cond1.shape[0]
            null_cond_tensor = self.null_cond.expand(B, -1, -1).to(dtype=cond1.dtype)
            null_cond = (null_cond_tensor, null_cond_tensor, alpha)

        t_flow = t.float() * 1000.0

        v_cond = self.sparse_structure_flow(x_t, t_flow, cond, alpha=alpha)
        v_uncond = self.sparse_structure_flow(x_t, t_flow, null_cond, alpha=alpha)

        return v_uncond + guidance_scale * (v_cond - v_uncond)

    def forward(self, x_0, src_1_feats, src_1_coords, src_2_feats, src_2_coords, alpha):
        B = x_0.shape[0]
        x_0 = x_0.squeeze(1)
        t = torch.rand(B).to(x_0.device).float()
        x_t, noise = self.diffuse(x_0, t)

        velocity = self.get_v(x_0, noise)

        pred = self.forward_flow(x_t, t, src_1_feats, src_2_feats, src_1_coords, src_2_coords, alpha)

        loss = F.mse_loss(pred, velocity)
        return loss
