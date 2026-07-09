import torch
import torch.nn.functional as F

from models.morph_flow import MorphFlow


class MorphResidualSSFlow(MorphFlow):
    """
    Sparse-structure flow trained in residual space over the source interpolation.

    Standard SS flow learns the target latent x0 directly. This variant learns:

        residual = target_ss - (alpha * src1_ss + (1 - alpha) * src2_ss)

    or, with residual_gate="alpha":

        target_ss = base + alpha * (1 - alpha) * residual

    In this dataset alpha is the fraction of src_1, so alpha=1 is src1 and
    alpha=0 is src2. The gated form makes alpha=0/1 decode exactly to the source
    latent after sampling, independent of the residual sampled by the flow.
    """

    requires_source_ss_latents = True

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
        residual_interp_gate="alpha",
        residual_interp_gate_min=1e-3,
        residual_endpoint_prob=0.0,
        residual_endpoint_weight=1.0,
        residual_endpoint_max_items=1,
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
        super().__init__(
            sigma_min=sigma_min,
            model_type=model_type,
            separate_cond=separate_cond,
            use_checkpoint=use_checkpoint,
            separate_cond_gate=separate_cond_gate,
            cond_resample_tokens=cond_resample_tokens,
            cond_resample_depth=cond_resample_depth,
            cond_resample_heads=cond_resample_heads,
            cond_encoder_type=cond_encoder_type,
            normalize_cond_latents=normalize_cond_latents,
            cond_token_norm=cond_token_norm,
            cond_proj_norm=cond_proj_norm,
            cond_style_tokens=cond_style_tokens,
            cond_use_occupancy=cond_use_occupancy,
            cond_hybrid_pool_stats=cond_hybrid_pool_stats,
            cond_residual_blocks_64=cond_residual_blocks_64,
            cond_residual_blocks_32=cond_residual_blocks_32,
            cond_residual_blocks_16=cond_residual_blocks_16,
            t_schedule=t_schedule,
            t_logit_mean=t_logit_mean,
            t_logit_std=t_logit_std,
            use_semantic_token_matching=use_semantic_token_matching,
            semantic_match_dim=semantic_match_dim,
            semantic_match_temperature=semantic_match_temperature,
            semantic_match_max_align=semantic_match_max_align,
            semantic_match_alpha_weight=semantic_match_alpha_weight,
            semantic_match_detach_scores=semantic_match_detach_scores,
            semantic_match_exclude_style_tokens=semantic_match_exclude_style_tokens,
            semantic_cycle_loss_weight=semantic_cycle_loss_weight,
            semantic_cycle_loss_prob=semantic_cycle_loss_prob,
            semantic_cycle_detach_targets=semantic_cycle_detach_targets,
            semantic_cycle_alpha_weight=semantic_cycle_alpha_weight,
            semantic_match_log_stats=semantic_match_log_stats,
        )
        if residual_interp_gate not in ("none", "alpha"):
            raise ValueError(
                "residual_interp_gate must be one of {'none', 'alpha'}, "
                f"got {residual_interp_gate!r}"
            )
        if residual_interp_gate_min <= 0.0:
            raise ValueError(f"residual_interp_gate_min must be > 0, got {residual_interp_gate_min}")
        if residual_endpoint_prob < 0.0 or residual_endpoint_prob > 1.0:
            raise ValueError(f"residual_endpoint_prob must be in [0, 1], got {residual_endpoint_prob}")
        if residual_endpoint_weight < 0.0:
            raise ValueError(f"residual_endpoint_weight must be >= 0, got {residual_endpoint_weight}")
        if residual_endpoint_max_items < 0:
            raise ValueError(f"residual_endpoint_max_items must be >= 0, got {residual_endpoint_max_items}")

        self.residual_interp_gate = residual_interp_gate
        self.residual_interp_gate_min = float(residual_interp_gate_min)
        self.residual_endpoint_prob = float(residual_endpoint_prob)
        self.residual_endpoint_weight = float(residual_endpoint_weight)
        self.residual_endpoint_max_items = int(residual_endpoint_max_items)

    def _alpha_view(self, alpha, x):
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, device=x.device, dtype=x.dtype)
        alpha = alpha.to(device=x.device, dtype=x.dtype)
        if alpha.ndim == 0:
            alpha = alpha.expand(x.shape[0])
        return alpha.view(-1, *[1 for _ in range(x.ndim - 1)])

    def interpolation_base(self, src1_ss_latent, src2_ss_latent, alpha):
        src1 = self._prepare_ss_latent(src1_ss_latent)
        src2 = self._prepare_ss_latent(src2_ss_latent)
        alpha_view = self._alpha_view(alpha, src1)
        return alpha_view * src1 + (1.0 - alpha_view) * src2

    def residual_gate(self, alpha, x, *, clamp=False):
        if self.residual_interp_gate == "none":
            return torch.ones_like(self._alpha_view(alpha, x))

        alpha_view = self._alpha_view(alpha, x)
        gate = alpha_view * (1.0 - alpha_view)
        if clamp:
            gate = gate.clamp_min(self.residual_interp_gate_min)
        return gate

    def ss_to_residual(self, ss_latent, src1_ss_latent, src2_ss_latent, alpha):
        ss_latent = self._prepare_ss_latent(ss_latent)
        base = self.interpolation_base(src1_ss_latent, src2_ss_latent, alpha)
        residual = ss_latent - base
        if self.residual_interp_gate == "alpha":
            residual = residual / self.residual_gate(alpha, ss_latent, clamp=True)
        return residual

    def residual_to_ss(self, residual_latent, src1_ss_latent, src2_ss_latent, alpha):
        residual_latent = self._prepare_ss_latent(residual_latent)
        base = self.interpolation_base(src1_ss_latent, src2_ss_latent, alpha)
        return base + self.residual_gate(alpha, residual_latent, clamp=False) * residual_latent

    def _select_sparse_batch(self, feats, coords, indices):
        selected_feats = []
        selected_coords = []
        for new_batch_idx, old_batch_idx in enumerate(indices.tolist()):
            mask = coords[:, 0] == old_batch_idx
            sample_coords = coords[mask].clone()
            sample_coords[:, 0] = new_batch_idx
            selected_feats.append(feats[mask])
            selected_coords.append(sample_coords)
        return torch.cat(selected_feats, dim=0), torch.cat(selected_coords, dim=0)

    def _select_endpoint_items(
        self,
        src_1_ss_latent,
        src_2_ss_latent,
        src_1_feats,
        src_1_coords,
        src_2_feats,
        src_2_coords,
    ):
        max_items = self.residual_endpoint_max_items
        B = src_1_ss_latent.shape[0]
        if max_items <= 0 or B <= max_items:
            return (
                src_1_ss_latent,
                src_2_ss_latent,
                src_1_feats,
                src_1_coords,
                src_2_feats,
                src_2_coords,
            )

        indices = torch.randperm(B, device=src_1_ss_latent.device)[:max_items]
        src_1_feats, src_1_coords = self._select_sparse_batch(src_1_feats, src_1_coords, indices)
        src_2_feats, src_2_coords = self._select_sparse_batch(src_2_feats, src_2_coords, indices)
        return (
            src_1_ss_latent.index_select(0, indices),
            src_2_ss_latent.index_select(0, indices),
            src_1_feats,
            src_1_coords,
            src_2_feats,
            src_2_coords,
        )

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
        src1_ss_latent=None,
        src2_ss_latent=None,
    ):
        if src1_ss_latent is None or src2_ss_latent is None:
            raise ValueError("MorphResidualSSFlow requires src1_ss_latent and src2_ss_latent.")

        B = x_0.shape[0]
        x_0 = self.ss_to_residual(x_0, src1_ss_latent, src2_ss_latent, alpha)
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
        (
            src_1_ss_latent,
            src_2_ss_latent,
            src_1_feats,
            src_1_coords,
            src_2_feats,
            src_2_coords,
        ) = self._select_endpoint_items(
            src_1_ss_latent,
            src_2_ss_latent,
            src_1_feats,
            src_1_coords,
            src_2_feats,
            src_2_coords,
        )

        B = src_1_ss_latent.shape[0]
        endpoint_is_src1 = torch.rand(B, device=src_1_ss_latent.device) < 0.5
        alpha = endpoint_is_src1.to(dtype=torch.float32)

        view_shape = (B,) + (1,) * (src_1_ss_latent.ndim - 1)
        target_ss = torch.where(endpoint_is_src1.view(view_shape), src_1_ss_latent, src_2_ss_latent)
        target_residual = self.ss_to_residual(target_ss, src_1_ss_latent, src_2_ss_latent, alpha)

        t = self.sample_t(B, target_residual.device)
        x_t, _ = self.diffuse(target_residual, t)
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
        pred_residual = self.pred_x0_from_velocity(x_t, t, pred_velocity)
        pred_ss = self.residual_to_ss(pred_residual, src_1_ss_latent, src_2_ss_latent, alpha)
        return F.mse_loss(pred_ss, target_ss)

    def residual_endpoint_flow_matching_loss(
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
        (
            src_1_ss_latent,
            src_2_ss_latent,
            src_1_feats,
            src_1_coords,
            src_2_feats,
            src_2_coords,
        ) = self._select_endpoint_items(
            src_1_ss_latent,
            src_2_ss_latent,
            src_1_feats,
            src_1_coords,
            src_2_feats,
            src_2_coords,
        )

        B = src_1_ss_latent.shape[0]
        endpoint_is_src1 = torch.rand(B, device=src_1_ss_latent.device) < 0.5
        alpha = endpoint_is_src1.to(dtype=torch.float32)

        # At alpha=0/1, target_ss == interpolation_base, so the residual target is exactly zero.
        target_residual = torch.zeros_like(src_1_ss_latent)
        t = self.sample_t(B, target_residual.device)
        x_t, noise = self.diffuse(target_residual, t)
        velocity = self.get_v(target_residual, noise)

        pred = self.forward_flow(
            x_t,
            t,
            src_1_feats,
            src_2_feats,
            src_1_coords,
            src_2_coords,
            alpha,
            apply_cfg_drop=False,
        )
        return F.mse_loss(pred, velocity)

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
        residual_endpoint_loss_weight=None,
        residual_endpoint_loss_prob=None,
    ):
        if src1_ss_latent is None or src2_ss_latent is None:
            raise ValueError("MorphResidualSSFlow.forward requires src1_ss_latent and src2_ss_latent.")

        residual_endpoint_override = (
            residual_endpoint_loss_weight is not None
            or residual_endpoint_loss_prob is not None
        )
        effective_residual_endpoint_weight = (
            self.residual_endpoint_weight
            if residual_endpoint_loss_weight is None
            else float(residual_endpoint_loss_weight)
        )
        effective_residual_endpoint_prob = (
            self.residual_endpoint_prob
            if residual_endpoint_loss_prob is None
            else float(residual_endpoint_loss_prob)
        )

        endpoint_active = (
            endpoint_loss_weight > 0.0
            and torch.rand((), device=x_0.device).item() < endpoint_loss_prob
        )
        symmetry_active = symmetry_loss_weight > 0.0 and torch.rand((), device=x_0.device).item() < symmetry_loss_prob
        residual_endpoint_active = (
            (self.training or residual_endpoint_override)
            and effective_residual_endpoint_weight > 0.0
            and effective_residual_endpoint_prob > 0.0
            and torch.rand((), device=x_0.device).item() < effective_residual_endpoint_prob
        )

        loss, x_t, t, pred = self.flow_matching_loss(
            x_0,
            src_1_feats,
            src_1_coords,
            src_2_feats,
            src_2_coords,
            alpha,
            return_terms=True,
            apply_cfg_drop=not symmetry_active,
            src1_ss_latent=src1_ss_latent,
            src2_ss_latent=src2_ss_latent,
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

        residual_endpoint_term = None
        if residual_endpoint_active:
            residual_endpoint_term = self.residual_endpoint_flow_matching_loss(
                src1_ss_latent,
                src2_ss_latent,
                src_1_feats,
                src_1_coords,
                src_2_feats,
                src_2_coords,
            )
            loss = loss + effective_residual_endpoint_weight * residual_endpoint_term

        self.last_loss_terms = {
            "endpoint_active": endpoint_term is not None,
            "symmetry_active": symmetry_term is not None,
            "residual_endpoint_active": residual_endpoint_term is not None,
        }
        return loss
