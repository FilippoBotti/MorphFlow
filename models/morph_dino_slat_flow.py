from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.morph_slat_flow import MorphSLatFlow
from modules import sparse as sp


class MorphDinoSLatFlow(MorphSLatFlow):
    """
    SLat flow variant conditioned on frozen DINOv2 features from source images.

    The TRELLIS SLat flow is unchanged. The MorphFlow pair condition becomes:
        src1 image, src2 image -> frozen DINOv2 tokens -> cross-attention context.

    The DINO model is loaded lazily and intentionally kept outside state_dict,
    so checkpoints only contain the trainable projection and MorphFlow/TRELLIS
    parameters.
    """

    requires_source_images = True

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
        cond_encoder_type: str = "block",
        normalize_flow_latents: bool = True,
        normalize_cond_latents: bool = False,
        cond_token_norm: str = "none",
        cond_proj_norm: str = "none",
        t_schedule: str = "logit_normal",
        t_logit_mean: float = 0.0,
        t_logit_std: float = 1.0,
        dino_model: str = "dinov2_vitl14_reg",
        dino_dim: int = 1024,
        dino_layer_norm: bool = True,
    ):
        self.dino_resample_tokens = int(cond_resample_tokens)
        super().__init__(
            sigma_min=sigma_min,
            model_type=model_type,
            separate_cond=separate_cond,
            use_checkpoint=use_checkpoint,
            separate_cond_gate=separate_cond_gate,
            # DINO uses cond_resample_tokens to downsample image tokens in _encode_images.
            # Do not instantiate the SLat condition resampler from the base class here,
            # otherwise old DINO checkpoints would gain unused/missing parameters.
            cond_resample_tokens=0,
            cond_resample_depth=cond_resample_depth,
            cond_resample_heads=cond_resample_heads,
            cond_encoder_type=cond_encoder_type,
            normalize_flow_latents=normalize_flow_latents,
            normalize_cond_latents=normalize_cond_latents,
            cond_token_norm=cond_token_norm,
            cond_proj_norm=cond_proj_norm,
            t_schedule=t_schedule,
            t_logit_mean=t_logit_mean,
            t_logit_std=t_logit_std,
        )
        self.dino_model_name = dino_model
        self.dino_dim = int(dino_dim)
        self.dino_norm = nn.LayerNorm(self.dino_dim) if dino_layer_norm else nn.Identity()
        self.dino_proj = nn.Linear(self.dino_dim, self.model_channels)
        self.dino_out_norm = nn.LayerNorm(self.model_channels)
        self.register_buffer("dino_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("dino_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False)
        self.__dict__["_dino_model"] = None

        for module in (self.cond_encoder, self.cond_fusion):
            for param in module.parameters():
                param.requires_grad = False
        if hasattr(self, "separate_cond_proj"):
            for param in self.separate_cond_proj.parameters():
                param.requires_grad = False
        if hasattr(self, "cond_proj_layer_norm"):
            for param in self.cond_proj_layer_norm.parameters():
                param.requires_grad = False
        self.null_cond.requires_grad = False

    def _get_dino_model(self, device: torch.device):
        model = self.__dict__.get("_dino_model")
        if model is None:
            model = torch.hub.load(
                "facebookresearch/dinov2",
                self.dino_model_name,
                pretrained=True,
                trust_repo=True,
            )
            model.eval().requires_grad_(False)
            self.__dict__["_dino_model"] = model
        model = model.to(device)
        model.eval()
        return model

    def _resample_tokens(self, tokens: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if num_tokens <= 0 or tokens.shape[1] <= num_tokens:
            return tokens
        tokens = tokens.transpose(1, 2)
        tokens = F.adaptive_avg_pool1d(tokens, num_tokens)
        return tokens.transpose(1, 2).contiguous()

    @torch.no_grad()
    def _encode_images(self, images: torch.Tensor, name: str) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"{name} must have shape [B, 3, H, W], got {tuple(images.shape)}")
        images = images.float()
        images = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
        images = (images - self.dino_mean.to(images.device, images.dtype)) / self.dino_std.to(images.device, images.dtype)
        dino = self._get_dino_model(images.device)
        output = dino(images, is_training=True)
        if isinstance(output, dict) and "x_norm_patchtokens" in output:
            pieces = []
            cls_token = output.get("x_norm_clstoken")
            reg_tokens = output.get("x_norm_regtokens")
            patch_tokens = output["x_norm_patchtokens"]
            if cls_token is not None:
                pieces.append(cls_token[:, None, :])
            if reg_tokens is not None:
                pieces.append(reg_tokens)
            special_tokens = sum(piece.shape[1] for piece in pieces)
            patch_target = self.dino_resample_tokens - special_tokens if self.dino_resample_tokens > 0 else 0
            patch_tokens = self._resample_tokens(patch_tokens, max(1, patch_target))
            features = torch.cat([*pieces, patch_tokens], dim=1) if pieces else patch_tokens
        else:
            features = output["x_prenorm"]
            features = self._resample_tokens(features, self.dino_resample_tokens)
        return F.layer_norm(features.float(), features.shape[-1:])

    def _project_dino_tokens(self, tokens: torch.Tensor, name: str) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"{name} must have shape [B, tokens, dim], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.dino_dim:
            raise ValueError(
                f"{name} has dim={tokens.shape[-1]}, but MorphDinoSLatFlow was built with dino_dim={self.dino_dim}. "
                "Set --dino_dim to match the DINO model output."
            )
        tokens = tokens.to(dtype=self.dino_proj.weight.dtype)
        return self.dino_out_norm(self.dino_proj(self.dino_norm(tokens)))

    def _build_condition_from_images(
        self,
        src1_image: torch.Tensor,
        src2_image: torch.Tensor,
        alpha: torch.Tensor,
    ):
        src1_tokens = self._encode_images(src1_image, "src1_image")
        src2_tokens = self._encode_images(src2_image, "src2_image")
        cond1 = self._project_dino_tokens(src1_tokens, "src1_image_dino_tokens")
        cond2 = self._project_dino_tokens(src2_tokens, "src2_image_dino_tokens")

        if self.separate_cond:
            cond = (cond1, cond2, alpha)
        else:
            alpha_view = alpha.view(-1, 1, 1).to(dtype=cond1.dtype)
            cond = alpha_view * cond1 + (1.0 - alpha_view) * cond2

        if self.training and self.cfg_drop_prob > 0.0:
            batch_size = cond1.shape[0]
            drop_mask = torch.rand(batch_size, device=cond1.device) < self.cfg_drop_prob
            drop_mask = drop_mask.view(batch_size, 1, 1)
            if self.separate_cond:
                cond = (
                    torch.where(drop_mask, torch.zeros_like(cond1), cond1),
                    torch.where(drop_mask, torch.zeros_like(cond2), cond2),
                    alpha,
                )
            else:
                cond = torch.where(drop_mask, torch.zeros_like(cond), cond)

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
        src1_image: Optional[torch.Tensor] = None,
        src2_image: Optional[torch.Tensor] = None,
    ) -> sp.SparseTensor:
        del src_1_feats, src_2_feats, src_1_coords, src_2_coords
        if src1_image is None or src2_image is None:
            raise ValueError("MorphDinoSLatFlow requires src1_image and src2_image.")
        cond = self._build_condition_from_images(src1_image, src2_image, alpha)
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
        src1_image: Optional[torch.Tensor] = None,
        src2_image: Optional[torch.Tensor] = None,
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
                src1_image=src1_image,
                src2_image=src2_image,
            )
        if src1_image is None or src2_image is None:
            raise ValueError("MorphDinoSLatFlow requires src1_image and src2_image.")

        src1_tokens = self._encode_images(src1_image, "src1_image")
        src2_tokens = self._encode_images(src2_image, "src2_image")
        cond1 = self._project_dino_tokens(src1_tokens, "src1_image_dino_tokens")
        cond2 = self._project_dino_tokens(src2_tokens, "src2_image_dino_tokens")
        if self.separate_cond:
            cond = (cond1, cond2, alpha)
            null_cond = (torch.zeros_like(cond1), torch.zeros_like(cond2), alpha)
        else:
            alpha_view = alpha.view(-1, 1, 1).to(dtype=cond1.dtype)
            cond = alpha_view * cond1 + (1.0 - alpha_view) * cond2
            null_cond = torch.zeros_like(cond)

        t_flow = t.float() * 1000.0
        v_cond = self.slat_flow(x_t, t_flow, cond, alpha=alpha)
        v_uncond = self.slat_flow(x_t, t_flow, null_cond, alpha=alpha)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    def forward(
        self,
        target_feats: torch.Tensor,
        target_coords: torch.Tensor,
        src_1_feats: torch.Tensor,
        src_1_coords: torch.Tensor,
        src_2_feats: torch.Tensor,
        src_2_coords: torch.Tensor,
        alpha: torch.Tensor,
        src1_image: Optional[torch.Tensor] = None,
        src2_image: Optional[torch.Tensor] = None,
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
            src1_image=src1_image,
            src2_image=src2_image,
        )

        loss = self.batch_mean_mse(pred, velocity)
        self._update_forward_metrics(pred, velocity, loss)
        return loss
