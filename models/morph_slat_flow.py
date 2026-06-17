from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from models import cond_encoder
from models import structured_latent_flow
from modules import sparse as sp


TRELLIS_SLAT_MEAN = (
    -2.1687545776367188,
    -0.004347046371549368,
    -0.13352349400520325,
    -0.08418072760105133,
    -0.5271206498146057,
    0.7238689064979553,
    -1.1414450407028198,
    1.2039363384246826,
)

TRELLIS_SLAT_STD = (
    2.377650737762451,
    2.386378288269043,
    2.124418020248413,
    2.1748552322387695,
    2.663944721221924,
    2.371192216873169,
    2.6217446327209473,
    2.684523105621338,
)


class MorphSLatFlow(nn.Module):
    """
    MorphFlow variant for TRELLIS structured-latent flow.

    It keeps the TRELLIS SLat denoiser objective, but replaces image/text
    conditioning with MorphFlow's pair conditioning:
    src1 SLat, src2 SLat, alpha -> context tokens.
    """

    def __init__(
        self,
        sigma_min: float = 1e-5,
        model_type: str = "text_base",
        separate_cond: bool = False,
        use_checkpoint: bool = False,
        separate_cond_gate: str = "alpha_residual",
        cond_resample_tokens: int = 0,
        cond_resample_depth: int = 1,
        cond_resample_heads: int = 8,
        normalize_flow_latents: bool = True,
        t_schedule: str = "logit_normal",
        t_logit_mean: float = 0.0,
        t_logit_std: float = 1.0,
    ):
        super().__init__()
        del cond_resample_tokens, cond_resample_depth, cond_resample_heads

        self.sigma_min = sigma_min
        self.separate_cond = separate_cond
        self.separate_cond_gate = separate_cond_gate
        self.normalize_flow_latents = normalize_flow_latents
        self.t_schedule = t_schedule
        self.t_logit_mean = t_logit_mean
        self.t_logit_std = t_logit_std

        if self.t_schedule not in ("uniform", "logit_normal"):
            raise ValueError(f"Unknown t_schedule: {self.t_schedule}")
        if self.t_logit_std <= 0.0:
            raise ValueError(f"t_logit_std must be > 0, got {self.t_logit_std}")

        if model_type == "text_base":
            model_channels = 768
            num_blocks = 12
            num_heads = 12
            io_block_channels = [128]
        elif model_type == "image_large":
            model_channels = 1024
            num_blocks = 24
            num_heads = 16
            io_block_channels = [128]
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

        self.slat_flow = structured_latent_flow.SLatFlowModel(
            resolution=64,
            in_channels=8,
            out_channels=8,
            model_channels=model_channels,
            cond_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=4,
            patch_size=2,
            num_io_res_blocks=2,
            io_block_channels=io_block_channels,
            pe_mode="ape",
            qk_rms_norm=True,
            use_fp16=False,
            use_checkpoint=use_checkpoint,
            separate_cond=separate_cond,
            separate_cond_gate=separate_cond_gate,
        )

        self.register_buffer(
            "slat_mean",
            torch.tensor(TRELLIS_SLAT_MEAN, dtype=torch.float32).view(1, -1),
            persistent=False,
        )
        self.register_buffer(
            "slat_std",
            torch.tensor(TRELLIS_SLAT_STD, dtype=torch.float32).view(1, -1),
            persistent=False,
        )

    def make_slat(self, feats: torch.Tensor, coords: torch.Tensor) -> sp.SparseTensor:
        return sp.SparseTensor(feats=feats, coords=coords)

    def normalize_slat(self, slat: sp.SparseTensor) -> sp.SparseTensor:
        if not self.normalize_flow_latents:
            return slat
        mean = self.slat_mean.to(device=slat.device, dtype=slat.dtype)
        std = self.slat_std.to(device=slat.device, dtype=slat.dtype)
        return slat.replace((slat.feats - mean) / std)

    def denormalize_slat(self, slat: sp.SparseTensor) -> sp.SparseTensor:
        if not self.normalize_flow_latents:
            return slat
        mean = self.slat_mean.to(device=slat.device, dtype=slat.dtype)
        std = self.slat_std.to(device=slat.device, dtype=slat.dtype)
        return slat.replace(slat.feats * std + mean)

    def get_v(self, x_0: sp.SparseTensor, noise: sp.SparseTensor) -> sp.SparseTensor:
        return (1 - self.sigma_min) * noise - x_0

    def diffuse(
        self,
        x_0: sp.SparseTensor,
        t: torch.Tensor,
        noise: Optional[sp.SparseTensor] = None,
    ) -> tuple[sp.SparseTensor, sp.SparseTensor]:
        if noise is None:
            noise = x_0.replace(torch.randn_like(x_0.feats))

        t = t.view(-1, 1)
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise
        return x_t, noise

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.t_schedule == "uniform":
            return torch.rand(batch_size, device=device, dtype=torch.float32)
        noise = torch.randn(batch_size, device=device, dtype=torch.float32)
        return torch.sigmoid(noise * self.t_logit_std + self.t_logit_mean)

    def _build_condition(
        self,
        src_1_feats: torch.Tensor,
        src_1_coords: torch.Tensor,
        src_2_feats: torch.Tensor,
        src_2_coords: torch.Tensor,
        alpha: torch.Tensor,
    ):
        cond1 = self.cond_encoder(src_1_feats, src_1_coords)
        cond2 = self.cond_encoder(src_2_feats, src_2_coords)

        if not self.separate_cond:
            cond = self.cond_fusion(cond1, cond2, alpha)
        else:
            cond1 = self.separate_cond_proj(cond1)
            cond2 = self.separate_cond_proj(cond2)
            cond = (cond1, cond2, alpha)

        if self.training and self.cfg_drop_prob > 0.0:
            batch_size = cond1.shape[0] if self.separate_cond else cond.shape[0]
            drop_mask = torch.rand(batch_size, device=cond1.device) < self.cfg_drop_prob
            null_cond = self.null_cond.expand(batch_size, -1, -1).to(dtype=cond1.dtype)

            if not self.separate_cond:
                cond = torch.where(drop_mask.view(batch_size, 1, 1), null_cond, cond)
            else:
                drop_mask = drop_mask.view(batch_size, 1, 1)
                cond = (
                    torch.where(drop_mask, null_cond, cond1),
                    torch.where(drop_mask, null_cond, cond2),
                    alpha,
                )

        return cond

    def forward_flow(
        self,
        x_t: sp.SparseTensor,
        t: torch.Tensor,
        src_1_feats: torch.Tensor,
        src_2_feats: torch.Tensor,
        src_1_coords: torch.Tensor,
        src_2_coords: torch.Tensor,
        alpha: torch.Tensor,
    ) -> sp.SparseTensor:
        cond = self._build_condition(
            src_1_feats,
            src_1_coords,
            src_2_feats,
            src_2_coords,
            alpha,
        )
        t_flow = t.float() * 1000.0
        return self.slat_flow(x_t, t_flow, cond, alpha=alpha)

    def forward_flow_cfg(
        self,
        x_t: sp.SparseTensor,
        t: torch.Tensor,
        src_1_feats: torch.Tensor,
        src_2_feats: torch.Tensor,
        src_1_coords: torch.Tensor,
        src_2_coords: torch.Tensor,
        alpha: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> sp.SparseTensor:
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
            batch_size = cond.shape[0]
            null_cond = self.null_cond.expand(batch_size, -1, -1).to(dtype=cond.dtype)
        else:
            cond1 = self.separate_cond_proj(cond1)
            cond2 = self.separate_cond_proj(cond2)
            cond = (cond1, cond2, alpha)
            batch_size = cond1.shape[0]
            null_tensor = self.null_cond.expand(batch_size, -1, -1).to(dtype=cond1.dtype)
            null_cond = (null_tensor, null_tensor, alpha)

        t_flow = t.float() * 1000.0
        v_cond = self.slat_flow(x_t, t_flow, cond, alpha=alpha)
        v_uncond = self.slat_flow(x_t, t_flow, null_cond, alpha=alpha)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    @staticmethod
    def batch_mean_mse(pred: sp.SparseTensor, target: sp.SparseTensor) -> torch.Tensor:
        if pred.shape[0] <= 1:
            return F.mse_loss(pred.feats, target.feats)
        return torch.stack(
            [
                F.mse_loss(pred.feats[pred.layout[i]], target.feats[target.layout[i]])
                for i in range(pred.shape[0])
            ]
        ).mean()

    def forward(
        self,
        target_feats: torch.Tensor,
        target_coords: torch.Tensor,
        src_1_feats: torch.Tensor,
        src_1_coords: torch.Tensor,
        src_2_feats: torch.Tensor,
        src_2_coords: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        x_0 = self.make_slat(target_feats, target_coords)
        x_0 = self.normalize_slat(x_0)

        batch_size = x_0.shape[0]
        t = self.sample_t(batch_size, x_0.device)
        x_t, noise = self.diffuse(x_0, t)
        velocity = self.get_v(x_0, noise)

        pred = self.forward_flow(
            x_t,
            t,
            src_1_feats,
            src_2_feats,
            src_1_coords,
            src_2_coords,
            alpha,
        )

        return self.batch_mean_mse(pred, velocity)
