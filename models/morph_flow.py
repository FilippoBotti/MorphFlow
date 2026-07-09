import torch
from torch import nn as nn
import torch.nn.functional as F

from models import sparse_structure_flow
from models import cond_encoder
from models.semantic_token_matching import SemanticTokenMatchingMixin


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


class MorphFlow(SemanticTokenMatchingMixin, nn.Module):
    def __init__(
        self,
        sigma_min=1e-5,
        model_type="text_base",
        separate_cond=False,
        use_checkpoint=False,
        separate_cond_gate="alpha_residual",
        cond_resample_tokens=0,
        cond_resample_depth=1,
        cond_resample_heads=8,
        cond_encoder_type="block",
        normalize_cond_latents=False,
        cond_token_norm="none",
        cond_proj_norm="none",
        cond_style_tokens=0,
        cond_use_occupancy=False,
        cond_hybrid_pool_stats=False,
        cond_residual_blocks_64=0,
        cond_residual_blocks_32=0,
        cond_residual_blocks_16=0,
        t_schedule="logit_normal",
        t_logit_mean=0.0,
        t_logit_std=1.0,
        use_semantic_token_matching=False,
        semantic_match_dim=128,
        semantic_match_temperature=0.1,
        semantic_match_max_align=0.25,
        semantic_match_alpha_weight=True,
        semantic_match_detach_scores=False,
        semantic_match_exclude_style_tokens=True,
        semantic_cycle_loss_weight=0.0,
        semantic_cycle_loss_prob=1.0,
        semantic_cycle_detach_targets=True,
        semantic_cycle_alpha_weight=True,
        semantic_match_log_stats=True,
    ):
        super().__init__()
        
        self.separate_cond = separate_cond
        self.separate_cond_gate = separate_cond_gate
        self.cond_resample_tokens = int(cond_resample_tokens)
        self.cond_resample_depth = int(cond_resample_depth)
        self.cond_resample_heads = int(cond_resample_heads)
        self.cond_encoder_type = cond_encoder_type
        self.normalize_cond_latents = bool(normalize_cond_latents)
        self.cond_token_norm = cond_token_norm
        self.cond_proj_norm = cond_proj_norm
        self.cond_style_tokens = int(cond_style_tokens)
        self.cond_use_occupancy = bool(cond_use_occupancy)
        self.cond_hybrid_pool_stats = bool(cond_hybrid_pool_stats)
        self.cond_residual_blocks_64 = int(cond_residual_blocks_64)
        self.cond_residual_blocks_32 = int(cond_residual_blocks_32)
        self.cond_residual_blocks_16 = int(cond_residual_blocks_16)
        self.t_schedule = t_schedule
        self.t_logit_mean = t_logit_mean
        self.t_logit_std = t_logit_std

        if self.cond_token_norm not in ("none", "layernorm", "adaln_alpha"):
            raise ValueError(
                "cond_token_norm must be one of {'none', 'layernorm', 'adaln_alpha'}, "
                f"got {self.cond_token_norm!r}"
            )
        if self.cond_proj_norm not in ("none", "layernorm"):
            raise ValueError(
                "cond_proj_norm must be one of {'none', 'layernorm'}, "
                f"got {self.cond_proj_norm!r}"
            )
        if self.t_schedule not in ("uniform", "logit_normal"):
            raise ValueError(f"Unknown t_schedule: {self.t_schedule}")
        if self.t_logit_std <= 0.0:
            raise ValueError(f"t_logit_std must be > 0, got {self.t_logit_std}")

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

        self.model_channels = model_channels

        self.cond_encoder = cond_encoder.build_condition_encoder(
            self.cond_encoder_type,
            style_tokens=self.cond_style_tokens,
            use_occupancy=self.cond_use_occupancy,
            hybrid_pool_stats=self.cond_hybrid_pool_stats,
            residual_blocks_64=self.cond_residual_blocks_64,
            residual_blocks_32=self.cond_residual_blocks_32,
            residual_blocks_16=self.cond_residual_blocks_16,
        )
        self.cond_sequence_tokens = int(getattr(self.cond_encoder, "num_output_tokens", self.cond_encoder.num_blocks))
        if self.cond_resample_tokens > 0:
            self.cond_resampler = cond_encoder.ConditionResampler(
                dim=128,
                num_tokens=self.cond_resample_tokens,
                depth=self.cond_resample_depth,
                heads=self.cond_resample_heads,
            )
            self.cond_sequence_tokens = self.cond_resample_tokens
        self.cond_fusion = cond_encoder.PairConditionFusionV2(
            cond_dim=128,
            alpha_dim=64,
            hidden_dim=512,
            out_dim=model_channels, 
        )
        if self.cond_token_norm in ("layernorm", "adaln_alpha"):
            self.cond_token_layer_norm = nn.LayerNorm(128)
        if self.cond_token_norm == "adaln_alpha":
            self.cond_alpha_mod = nn.Sequential(
                nn.Linear(1, 128),
                nn.SiLU(),
                nn.Linear(128, 256),
            )
            nn.init.zeros_(self.cond_alpha_mod[-1].weight)
            nn.init.zeros_(self.cond_alpha_mod[-1].bias)
        if self.separate_cond:
            self.separate_cond_proj = nn.Linear(128, model_channels)
            if self.cond_proj_norm == "layernorm":
                self.cond_proj_layer_norm = nn.LayerNorm(model_channels)


        semantic_style_tokens = 0
        if semantic_match_exclude_style_tokens and not hasattr(self, "cond_resampler"):
            semantic_style_tokens = int(getattr(self.cond_encoder, "style_tokens", 0))
        self._init_semantic_token_matching(
            use_semantic_token_matching=use_semantic_token_matching,
            semantic_match_dim=semantic_match_dim,
            semantic_match_temperature=semantic_match_temperature,
            semantic_match_max_align=semantic_match_max_align,
            semantic_match_alpha_weight=semantic_match_alpha_weight,
            semantic_match_detach_scores=semantic_match_detach_scores,
            semantic_match_style_tokens=semantic_style_tokens,
            semantic_cycle_loss_weight=semantic_cycle_loss_weight,
            semantic_cycle_loss_prob=semantic_cycle_loss_prob,
            semantic_cycle_detach_targets=semantic_cycle_detach_targets,
            semantic_cycle_alpha_weight=semantic_cycle_alpha_weight,
            semantic_match_log_stats=semantic_match_log_stats,
        )

        self.cfg_drop_prob = 0.0
        self.null_cond = nn.Parameter(
            torch.zeros(1, self.cond_sequence_tokens, model_channels)
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
            separate_cond=separate_cond,       
            separate_cond_gate=separate_cond_gate, 
        )
        self.sigma_min = sigma_min

        self.register_buffer(
            "cond_slat_mean",
            torch.tensor(TRELLIS_SLAT_MEAN, dtype=torch.float32).view(1, -1),
            persistent=False,
        )
        self.register_buffer(
            "cond_slat_std",
            torch.tensor(TRELLIS_SLAT_STD, dtype=torch.float32).view(1, -1),
            persistent=False,
        )

    def normalize_condition_feats(self, feats):
        if not self.normalize_cond_latents:
            return feats
        mean = self.cond_slat_mean.to(device=feats.device, dtype=feats.dtype)
        std = self.cond_slat_std.to(device=feats.device, dtype=feats.dtype)
        return (feats - mean) / std

    def encode_condition_tokens(self, feats, coords):
        cond = self.cond_encoder(feats, coords)
        if hasattr(self, "cond_resampler"):
            cond = self.cond_resampler(cond)
        return cond

    def normalize_condition_tokens(self, cond1, cond2, alpha):
        if self.cond_token_norm == "none":
            return cond1, cond2

        cond1 = self.cond_token_layer_norm(cond1)
        cond2 = self.cond_token_layer_norm(cond2)

        if self.cond_token_norm == "adaln_alpha":
            alpha = alpha.to(device=cond1.device, dtype=cond1.dtype).view(-1, 1)
            scale, shift = self.cond_alpha_mod(alpha).view(alpha.shape[0], 1, 2, 128).unbind(dim=2)
            cond1 = cond1 * (1.0 + scale) + shift
            cond2 = cond2 * (1.0 + scale) + shift

        return cond1, cond2

    def normalize_projected_condition_tokens(self, cond1, cond2):
        if self.cond_proj_norm == "none":
            return cond1, cond2
        return self.cond_proj_layer_norm(cond1), self.cond_proj_layer_norm(cond2)

    def _build_condition(
        self,
        src_1_feats,
        src_2_feats,
        src_1_coords,
        src_2_coords,
        alpha,
        apply_cfg_drop=True,
    ):
        src_1_feats = self.normalize_condition_feats(src_1_feats)
        src_2_feats = self.normalize_condition_feats(src_2_feats)
        cond1 = self.encode_condition_tokens(src_1_feats, src_1_coords)
        cond2 = self.encode_condition_tokens(src_2_feats, src_2_coords)
        cond1, cond2 = self.normalize_condition_tokens(cond1, cond2, alpha)
        cond1, cond2 = self._apply_semantic_token_matching(cond1, cond2, alpha)

        if not self.separate_cond:
            cond = self.cond_fusion(cond1, cond2, alpha)
        else:
            cond1 = self.separate_cond_proj(cond1)
            cond2 = self.separate_cond_proj(cond2)
            cond1, cond2 = self.normalize_projected_condition_tokens(cond1, cond2)
            cond = (cond1, cond2, alpha)

        if apply_cfg_drop and self.training and self.cfg_drop_prob > 0.0:
            B = cond1.shape[0] if self.separate_cond else cond.shape[0]
            drop_mask = torch.rand(B, device=cond1.device) < self.cfg_drop_prob
            null_cond = self.null_cond.expand(B, -1, -1).to(dtype=cond1.dtype)
            if not self.separate_cond:
                cond = torch.where(drop_mask.view(B, 1, 1), null_cond, cond)
            else:
                drop_mask = drop_mask.view(B, 1, 1)
                cond = (
                    torch.where(drop_mask, null_cond, cond1),
                    torch.where(drop_mask, null_cond, cond2),
                    alpha,
                )
        return cond

    def get_v(self, x_0, noise):
        return (1 - self.sigma_min) * noise - x_0
    
    def diffuse(self, x_0, t):
        noise = torch.randn_like(x_0)

        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t, noise

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.t_schedule == "uniform":
            return torch.rand(batch_size, device=device, dtype=torch.float32)
        noise = torch.randn(batch_size, device=device, dtype=torch.float32)
        return torch.sigmoid(noise * self.t_logit_std + self.t_logit_mean)
    
    def forward_flow(
        self,
        x_t,
        t,
        src_1_feats,
        src_2_feats,
        src_1_coords,
        src_2_coords,
        alpha,
        apply_cfg_drop=True,
    ):
        cond = self._build_condition(
            src_1_feats,
            src_2_feats,
            src_1_coords,
            src_2_coords,
            alpha,
            apply_cfg_drop=apply_cfg_drop,
        )
        t_flow = t.float() * 1000.0
        return self.sparse_structure_flow(x_t, t_flow, cond, alpha=alpha)

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

        cond = self._build_condition(
            src_1_feats,
            src_2_feats,
            src_1_coords,
            src_2_coords,
            alpha,
            apply_cfg_drop=False,
        )
        if not self.separate_cond:
            B = cond.shape[0]
            null_cond = self.null_cond.expand(B, -1, -1).to(dtype=cond.dtype)
        else:
            cond1, cond2, alpha_cond = cond
            B = cond1.shape[0]
            null_cond_tensor = self.null_cond.expand(B, -1, -1).to(dtype=cond1.dtype)
            null_cond = (null_cond_tensor, null_cond_tensor, alpha_cond)

        t_flow = t.float() * 1000.0
        v_cond = self.sparse_structure_flow(x_t, t_flow, cond, alpha=alpha)
        v_uncond = self.sparse_structure_flow(x_t, t_flow, null_cond, alpha=alpha)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    def _prepare_ss_latent(self, x_0):
        if x_0.ndim == 6:
            return x_0.squeeze(1)
        return x_0

    def flow_matching_loss(
        self,
        x_0,
        src_1_feats,
        src_1_coords,
        src_2_feats,
        src_2_coords,
        alpha,
        return_terms=False,
        apply_cfg_drop=True,
    ):
        B = x_0.shape[0]
        x_0 = self._prepare_ss_latent(x_0)
        t = self.sample_t(B, x_0.device)
        x_t, noise = self.diffuse(x_0, t)
        velocity = self.get_v(x_0, noise)

        self._begin_semantic_match_record(x_0.device)
        pred = self.forward_flow(
            x_t,
            t,
            src_1_feats,
            src_2_feats,
            src_1_coords,
            src_2_coords,
            alpha,
            apply_cfg_drop=apply_cfg_drop,
        )
        loss = F.mse_loss(pred, velocity)
        semantic_aux = self._semantic_match_aux_loss(x_0.device, loss.dtype)
        loss = loss + semantic_aux
        self.last_forward_metrics = self._semantic_match_metrics()

        if return_terms:
            return loss, x_t, t, pred
        return loss

    def pred_x0_from_velocity(self, x_t, t, pred_velocity):
        sigma_t = self.sigma_min + (1.0 - self.sigma_min) * t
        sigma_t = sigma_t.view(-1, *[1 for _ in range(x_t.ndim - 1)])
        return (1.0 - self.sigma_min) * x_t - sigma_t * pred_velocity

    def endpoint_loss(
        self,
        src_1_ss_latent,
        src_2_ss_latent,
        src_1_feats,
        src_1_coords,
        src_2_feats,
        src_2_coords,
    ):
        src_1_ss_latent = self._prepare_ss_latent(src_1_ss_latent)
        src_2_ss_latent = self._prepare_ss_latent(src_2_ss_latent)

        B = src_1_ss_latent.shape[0]
        endpoint_is_src1 = torch.rand(B, device=src_1_ss_latent.device) < 0.5
        alpha = endpoint_is_src1.to(dtype=torch.float32)

        view_shape = (B,) + (1,) * (src_1_ss_latent.ndim - 1)
        target_x0 = torch.where(endpoint_is_src1.view(view_shape), src_1_ss_latent, src_2_ss_latent)

        t = self.sample_t(B, target_x0.device)
        x_t, _ = self.diffuse(target_x0, t)
        pred_velocity = self.forward_flow(
            x_t,
            t,
            src_1_feats,
            src_2_feats,
            src_1_coords,
            src_2_coords,
            alpha,
            apply_cfg_drop=False,
        )
        pred_x0 = self.pred_x0_from_velocity(x_t, t, pred_velocity)
        return F.mse_loss(pred_x0, target_x0)

    def symmetry_loss(
        self,
        x_t,
        t,
        src_1_feats,
        src_1_coords,
        src_2_feats,
        src_2_coords,
        alpha,
        pred_forward=None,
    ):
        if pred_forward is None:
            pred_forward = self.forward_flow(
                x_t,
                t,
                src_1_feats,
                src_2_feats,
                src_1_coords,
                src_2_coords,
                alpha,
                apply_cfg_drop=False,
            )

        pred_swapped = self.forward_flow(
            x_t,
            t,
            src_2_feats,
            src_1_feats,
            src_2_coords,
            src_1_coords,
            1.0 - alpha,
            apply_cfg_drop=False,
        )
        return F.mse_loss(pred_forward, pred_swapped)

    def forward(
        self,
        x_0,
        src_1_feats,
        src_1_coords,
        src_2_feats,
        src_2_coords,
        alpha,
        endpoint_loss_weight=0.0,
        symmetry_loss_weight=0.0,
        endpoint_loss_prob=0.25,
        symmetry_loss_prob=1.0,
        src1_ss_latent=None,
        src2_ss_latent=None,
    ):
        endpoint_active = (
            endpoint_loss_weight > 0.0
            and src1_ss_latent is not None
            and src2_ss_latent is not None
            and torch.rand((), device=x_0.device).item() < endpoint_loss_prob
        )
        symmetry_active = symmetry_loss_weight > 0.0 and torch.rand((), device=x_0.device).item() < symmetry_loss_prob

        loss, x_t, t, pred = self.flow_matching_loss(
            x_0,
            src_1_feats,
            src_1_coords,
            src_2_feats,
            src_2_coords,
            alpha,
            return_terms=True,
            apply_cfg_drop=not symmetry_active,
        )

        endpoint_term = None
        if endpoint_active:
            endpoint_term = self.endpoint_loss(
                src1_ss_latent,
                src2_ss_latent,
                src_1_feats,
                src_1_coords,
                src_2_feats,
                src_2_coords,
            )
            loss = loss + endpoint_loss_weight * endpoint_term

        symmetry_term = None
        if symmetry_active:
            symmetry_term = self.symmetry_loss(
                x_t,
                t,
                src_1_feats,
                src_1_coords,
                src_2_feats,
                src_2_coords,
                alpha,
                pred_forward=pred,
            )
            loss = loss + symmetry_loss_weight * symmetry_term

        self.last_loss_terms = {
            "endpoint_active": endpoint_term is not None,
            "symmetry_active": symmetry_term is not None,
        }
        return loss
