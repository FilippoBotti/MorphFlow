"""Frozen endpoint-conditioned TRELLIS SS projection prior.

The student first generates an SS latent with its ordinary differentiable
rollout. A small amount of noise is then added at a low TRELLIS timestep.
The original frozen TRELLIS image-conditioned SS flow predicts a local clean
projection using a real endpoint image as conditioning.

Both endpoint conditions are evaluated at the same noisy student sample:
    Lproj = alpha * Lsrc1 + (1 - alpha) * Lsrc2
Each endpoint has its own detached, trust-region-clipped projection target.

No unconditional RFDS residual and no TRELLIS CFG are used.
"""

import json
import math

import torch
from torch import nn
from torch.nn import functional as F


TRELLIS_PRIOR_REPO = "microsoft/TRELLIS-image-large"
TRELLIS_PRIOR_CHECKPOINT = "ckpts/ss_flow_img_dit_L_16l8_fp16"

TRELLIS_PRIOR_METRIC_NAMES = (
    "trellis_prior_loss",
    "trellis_prior_projection_loss",
    "trellis_prior_projection_delta_rms",
    "trellis_prior_projection_tangent_delta_rms",
    "trellis_prior_projection_radial_fraction",
    "trellis_prior_projection_raw_cosine_z",
    "trellis_prior_projection_delta_clipped_rms",
    "trellis_prior_projection_clip_fraction",
    "trellis_prior_projection_x0_rms",
    "trellis_prior_velocity_rms",
    "trellis_prior_t_mean",
    "trellis_prior_scale_loss",
    "trellis_prior_scale_active_fraction",
    "trellis_prior_scale_reference_rms",
    "trellis_prior_scale_ratio",
    "trellis_prior_guard_loss",
    "trellis_prior_guard_fraction",
    "trellis_prior_guard_low",
    "trellis_prior_guard_high",
    "trellis_prior_endpoint_rms",
    "trellis_prior_sample_endpoint_rms_ratio",
    "trellis_prior_delta_endpoint_rms_ratio",
    "trellis_prior_src1_fraction",
    "trellis_prior_projection_src1_loss",
    "trellis_prior_projection_src2_loss",
)


def _build_original_ss_flow(config_args, dtype=None):
    """Reuse local TRELLIS primitives, but remove every morph-specific path.

The inherited constructor builds native blocks when separate_cond=False. The
alpha embedder is removed, and this forward never evaluates alpha modulation.
Consequently the state dict must match the upstream checkpoint strictly.
"""
    from models.sparse_structure_flow import SparseStructureFlowModel
    from modules.spatial import patchify, unpatchify

    if "separate_cond" in config_args or "separate_cond_gate" in config_args:
        raise ValueError("The TRELLIS prior requires an original SS flow config, without morph conditioning")
    if config_args.get("pe_mode", "ape") != "ape":
        raise ValueError("The TRELLIS image SS prior requires the native APE configuration")
    if dtype not in (None, torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Prior dtype must be float32, float16, or bfloat16")

    class OriginalTrellisSSFlow(SparseStructureFlowModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, separate_cond=False)
            del self.alpha_embedder

        def forward(self, x, t, cond):
            expected = (x.shape[0], self.in_channels, *([self.resolution] * 3))
            if tuple(x.shape) != expected:
                raise ValueError(f"TRELLIS SS input shape {tuple(x.shape)} does not match {expected}")
            h = patchify(x, self.patch_size)
            h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
            h = self.input_layer(h) + self.pos_emb[None]
            t_emb = self.t_embedder(t)
            if self.share_mod:
                t_emb = self.adaLN_modulation(t_emb)
            t_emb, h, cond = t_emb.to(self.dtype), h.to(self.dtype), cond.to(self.dtype)
            for block in self.blocks:
                h = block(h, t_emb, cond)
            h = F.layer_norm(h.to(x.dtype), h.shape[-1:])
            h = self.out_layer(h)
            h = h.permute(0, 2, 1).view(
                h.shape[0], h.shape[2], *([self.resolution // self.patch_size] * 3)
            )
            return unpatchify(h, self.patch_size).contiguous()

    args = dict(config_args)
    # The frozen model never needs activation checkpointing. Keep the time,
    # input and output projections FP32, like upstream; cast only the torso.
    args["use_checkpoint"] = False
    if dtype is not None:
        args["use_fp16"] = False
    model = OriginalTrellisSSFlow(**args)
    if dtype is not None:
        # LayerNorm32 deliberately normalizes in FP32, and its affine weights
        # must stay FP32 too. As upstream's convert_to_fp16 does, convert only
        # the linear projections; RMSNorm parameters also retain FP32.
        for module in model.blocks.modules():
            if isinstance(module, nn.Linear):
                module.to(dtype=dtype)
        model.dtype = dtype
        model.use_fp16 = dtype == torch.float16
    return model


class TrellisSSPrior(nn.Module):
    """Frozen endpoint-conditioned local TRELLIS projection prior."""

    def __init__(
        self,
        flow,
        image_encoder,
        *,
        sigma_min=1e-5,
        t_min=0.05,
        t_max=0.20,
        projection_clip_ratio=0.10,
        tangent_projection=False,
        scale_anchor_weight=0.0,
        scale_anchor_low_ratio=0.75,
        scale_anchor_high_ratio=1.25,
        rms_guard_weight=1.0,
        rms_guard_low_ratio=0.25,
        rms_guard_high_ratio=2.0,
    ):
        super().__init__()

        if not math.isfinite(sigma_min) or not 0 <= sigma_min < 1:
            raise ValueError("sigma_min must be finite and in [0, 1)")

        if not (
            math.isfinite(t_min)
            and math.isfinite(t_max)
            and 0 < t_min < t_max < 1
        ):
            raise ValueError(
                "Prior timesteps must satisfy 0 < t_min < t_max < 1"
            )

        for name, value in (
            ("projection_clip_ratio", projection_clip_ratio),
            ("scale_anchor_weight", scale_anchor_weight),
            ("scale_anchor_low_ratio", scale_anchor_low_ratio),
            ("scale_anchor_high_ratio", scale_anchor_high_ratio),
            ("rms_guard_weight", rms_guard_weight),
            ("rms_guard_low_ratio", rms_guard_low_ratio),
            ("rms_guard_high_ratio", rms_guard_high_ratio),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")

        if not (0 < scale_anchor_low_ratio < 1.0 < scale_anchor_high_ratio):
            raise ValueError(
                "scale anchor ratios must satisfy 0 < low < 1 < high"
            )

        if rms_guard_low_ratio >= rms_guard_high_ratio:
            raise ValueError(
                "rms_guard_low_ratio must be smaller than rms_guard_high_ratio"
            )

        self.flow = flow.requires_grad_(False).eval()
        self.image_encoder = image_encoder.requires_grad_(False).eval()

        self.sigma_min = float(sigma_min)
        self.t_min = float(t_min)
        self.t_max = float(t_max)

        self.projection_clip_ratio = float(projection_clip_ratio)
        self.tangent_projection = bool(tangent_projection)
        self.scale_anchor_weight = float(scale_anchor_weight)
        self.scale_anchor_low_ratio = float(scale_anchor_low_ratio)
        self.scale_anchor_high_ratio = float(scale_anchor_high_ratio)

        self.rms_guard_weight = float(rms_guard_weight)
        self.rms_guard_low_ratio = float(rms_guard_low_ratio)
        self.rms_guard_high_ratio = float(rms_guard_high_ratio)

        # Exact preprocessing used by upstream TRELLIS ImageConditionedMixin.
        self.register_buffer(
            "image_mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )

        self.eval()

    @classmethod
    def from_pretrained(
        cls,
        *,
        device=None,
        dtype=None,
        dino_model="dinov2_vitl14_reg",
        **kwargs,
    ):
        """Load the native TRELLIS SS flow and its native DINO image encoder."""

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        config_path = hf_hub_download(
            repo_id=TRELLIS_PRIOR_REPO,
            filename=f"{TRELLIS_PRIOR_CHECKPOINT}.json",
        )

        checkpoint_path = hf_hub_download(
            repo_id=TRELLIS_PRIOR_REPO,
            filename=f"{TRELLIS_PRIOR_CHECKPOINT}.safetensors",
        )

        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)

        if config.get("name") != "SparseStructureFlowModel":
            raise ValueError(
                "Expected an original TRELLIS SparseStructureFlowModel checkpoint"
            )

        # Loading frozen supervisors must not perturb the student's RNG stream.
        with torch.random.fork_rng(devices=[]):
            flow = _build_original_ss_flow(
                config["args"],
                dtype=dtype,
            )

        flow.load_state_dict(
            load_file(checkpoint_path),
            strict=True,
        )

        with torch.random.fork_rng(devices=[]):
            image_encoder = torch.hub.load(
                "facebookresearch/dinov2:main",
                dino_model,
                pretrained=True,
                trust_repo=True,
            )

        prior = cls(
            flow,
            image_encoder,
            **kwargs,
        )

        if device is not None:
            prior = prior.to(device=device)

        return prior

    def train(self, mode=True):
        # Must remain frozen/eval even if somebody calls parent.train().
        super().train(False)
        self.flow.eval()
        self.image_encoder.eval()
        return self

    @staticmethod
    def _prepare_ss(z):
        if z.ndim == 6:
            z = z.squeeze(1)

        if z.ndim != 5:
            raise ValueError(
                f"Expected SS latent [B,C,H,W,D], got {tuple(z.shape)}"
            )

        return z

    @torch.no_grad()
    def encode_image(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "TRELLIS prior image must have shape [B,3,H,W]"
            )

        image = image.float()

        mean = self.image_mean.to(
            device=image.device,
            dtype=image.dtype,
        )
        std = self.image_std.to(
            device=image.device,
            dtype=image.dtype,
        )

        image = (image - mean) / std

        features = self.image_encoder(
            image,
            is_training=True,
        )["x_prenorm"]

        # Exact upstream TRELLIS normalization.
        return F.layer_norm(
            features,
            features.shape[-1:],
        )

    @staticmethod
    def _rms_per_item(x, batch_size):
        """RMS over all non-batch dimensions. Always returns shape [B]."""
        if x.shape[0] != batch_size:
            raise ValueError(
                f"RMS batch mismatch: expected B={batch_size}, "
                f"got tensor shape={tuple(x.shape)}"
            )

        rms = (
            x.float()
            .reshape(batch_size, -1)
            .square()
            .mean(dim=1)
            .clamp_min(1e-12)
            .sqrt()
        )

        if tuple(rms.shape) != (batch_size,):
            raise RuntimeError(
                f"Internal RMS shape error: expected {(batch_size,)}, "
                f"got {tuple(rms.shape)}"
            )

        return rms

    @staticmethod
    def _split_radial_tangent(delta, z, batch_size):
        """Split delta into radial (parallel to z) and tangent components."""
        delta_f = delta.float().reshape(batch_size, -1)
        z_f = z.float().reshape(batch_size, -1)
        dot = (delta_f * z_f).sum(dim=1)
        z_sq = z_f.square().sum(dim=1).clamp_min(1e-12)
        coeff = dot / z_sq
        shape = (batch_size,) + (1,) * (delta.ndim - 1)
        radial = z.float() * coeff.reshape(shape)
        tangent = delta.float() - radial
        radial_rms = radial.reshape(batch_size, -1).square().mean(dim=1).clamp_min(1e-12).sqrt()
        raw_rms = delta_f.square().mean(dim=1).clamp_min(1e-12).sqrt()
        radial_fraction = radial_rms / raw_rms.clamp_min(1e-12)
        delta_norm = delta_f.square().sum(dim=1).clamp_min(1e-12).sqrt()
        raw_cosine_z = dot / (delta_norm * z_sq.sqrt()).clamp_min(1e-12)
        return tangent.contiguous(), radial_fraction, raw_cosine_z

    def forward(
        self,
        z,
        *,
        src1_image,
        src2_image,
        alpha,
        src1_ss_latent,
        src2_ss_latent,
        tau=None,
        noise=None,
        return_loss_terms=False,
    ):
        z = self._prepare_ss(z)

        if z.shape[0] < 1 or not z.is_floating_point():
            raise ValueError(
                "TRELLIS prior expects a nonempty floating SS tensor"
            )

        batch_size = z.shape[0]

        src1_image = src1_image[:batch_size]
        src2_image = src2_image[:batch_size]

        if src1_image.shape[0] != batch_size or src2_image.shape[0] != batch_size:
            raise ValueError(
                "Not enough endpoint images for the prior batch"
            )

        src1_ss = self._prepare_ss(
            src1_ss_latent[:batch_size]
        ).detach().float()

        src2_ss = self._prepare_ss(
            src2_ss_latent[:batch_size]
        ).detach().float()

        if (
            src1_ss.shape != z.shape
            or src2_ss.shape != z.shape
        ):
            raise ValueError(
                "Endpoint SS shape mismatch for TRELLIS projection prior: "
                f"student={tuple(z.shape)}, "
                f"src1={tuple(src1_ss.shape)}, "
                f"src2={tuple(src2_ss.shape)}"
            )

        alpha = (
            alpha[:batch_size]
            .detach()
            .float()
            .reshape(-1)
            .clamp(0.0, 1.0)
        )

        if tuple(alpha.shape) != (batch_size,):
            raise ValueError(
                "alpha must contain one value per prior sample"
            )

        with torch.no_grad(), torch.autocast(
            device_type=z.device.type,
            enabled=False,
        ):
            detached_z = z.detach().float()

            if tau is None:
                tau = (
                    torch.rand(
                        batch_size,
                        device=z.device,

                    )
                    * (self.t_max - self.t_min)
                    + self.t_min
                )
            else:
                tau = torch.as_tensor(
                    tau,
                    device=z.device,
                    dtype=torch.float32,
                ).detach()

                if tau.numel() == 1:
                    tau = tau.expand(batch_size)

                if tuple(tau.shape) != (batch_size,):
                    raise ValueError(
                        "tau must be scalar or have shape [B]"
                    )

                if (
                    not torch.isfinite(tau).all()
                    or not (
                        (tau >= self.t_min)
                        & (tau <= self.t_max)
                    ).all()
                ):
                    raise ValueError(
                        "tau must be finite and inside configured prior range"
                    )

            if noise is None:
                noise = torch.randn_like(
                    detached_z,
                    dtype=torch.float32,
                )
            else:
                if noise.shape != detached_z.shape:
                    raise ValueError(
                        "noise must have the same shape as z"
                    )

                noise = noise.detach().to(
                    device=z.device,
                    dtype=torch.float32,
                )

            if not torch.isfinite(noise).all():
                raise FloatingPointError(
                    "Non-finite prior noise"
                )

            time = tau.view(
                batch_size, 1, 1, 1, 1
            )

            sigma_t = (
                self.sigma_min
                + (1.0 - self.sigma_min) * time
            )

            # Local perturbation around the STUDENT sample.
            x_t = (
                (1.0 - time) * detached_z
                + sigma_t * noise
            )

            src1_rms = self._rms_per_item(src1_ss, batch_size)
            src2_rms = self._rms_per_item(src2_ss, batch_size)
            endpoint_rms = alpha * src1_rms + (1 - alpha) * src2_rms
            targets = []
            diagnostics = []
            # Sequential frozen forwards keep peak teacher memory unchanged.
            # Both use the SAME x_t/tau; clipping precedes endpoint weighting.
            for image, source_rms in ((src1_image, src1_rms), (src2_image, src2_rms)):
                cond = self.encode_image(image).float()
                velocity = self.flow(x_t, tau * 1000.0, cond).float()
                if velocity.shape != detached_z.shape:
                    raise ValueError("TRELLIS velocity shape must match student SS")
                if not torch.isfinite(velocity).all():
                    raise FloatingPointError("Non-finite endpoint-conditioned TRELLIS velocity")
                projected_x0 = (1 - self.sigma_min) * x_t - sigma_t * velocity
                if not torch.isfinite(projected_x0).all():
                    raise FloatingPointError("Non-finite TRELLIS x0 projection")
                raw_delta = projected_x0 - detached_z
                raw_rms = self._rms_per_item(raw_delta, batch_size)
                tangent, radial_fraction, cosine_z = self._split_radial_tangent(
                    raw_delta, detached_z, batch_size,
                )
                tangent_rms = self._rms_per_item(tangent, batch_size)
                delta = tangent if self.tangent_projection else raw_delta
                delta_rms = tangent_rms if self.tangent_projection else raw_rms
                scale = torch.ones_like(source_rms)
                if self.projection_clip_ratio > 0:
                    scale = (self.projection_clip_ratio * source_rms
                             / delta_rms.clamp_min(1e-12)).clamp(max=1.0)
                delta = delta * scale.view(batch_size, 1, 1, 1, 1)
                clipped_rms = self._rms_per_item(delta, batch_size)
                targets.append(detached_z + delta)
                diagnostics.append({
                    "raw": raw_rms, "tangent": tangent_rms,
                    "radial": radial_fraction, "cosine": cosine_z,
                    "clipped": clipped_rms, "clip": (scale < 0.999999).float(),
                    "x0": self._rms_per_item(projected_x0, batch_size),
                    "velocity": self._rms_per_item(velocity, batch_size),
                    "relative_delta": clipped_rms / source_rms.clamp_min(1e-8),
                })
            weighted = {key: alpha * diagnostics[0][key] + (1 - alpha) * diagnostics[1][key]
                        for key in diagnostics[0]}

            # Reference scale follows the morph endpoints but does not prescribe
            # geometry. alpha=1 -> src1, alpha=0 -> src2. Log interpolation is
            # symmetric for multiplicative scale changes.
            scale_reference_rms = torch.exp(
                alpha * torch.log(src1_rms.clamp_min(1e-8))
                + (1.0 - alpha) * torch.log(src2_rms.clamp_min(1e-8))
            )

            # Deliberately loose anti-collapse interval.
            # No geometry is imposed inside this range.
            guard_low = (
                self.rms_guard_low_ratio
                * torch.minimum(
                    src1_rms,
                    src2_rms,
                )
            )

            guard_high = (
                self.rms_guard_high_ratio
                * torch.maximum(
                    src1_rms,
                    src2_rms,
                )
            )

        endpoint_losses = [
            0.5 * (z.float() - target).square().reshape(batch_size, -1).mean(dim=1)
            for target in targets
        ]
        projection_loss = (alpha * endpoint_losses[0] + (1 - alpha) * endpoint_losses[1]).mean()

        # Differentiable student RMS.
        # Explicit reshape guarantees exactly one scalar per sample.
        z_rms_per_item = self._rms_per_item(
            z,
            batch_size,
        )

        if tuple(z_rms_per_item.shape) != (batch_size,):
            raise RuntimeError(
                "z_rms_per_item must have shape [B], got "
                f"{tuple(z_rms_per_item.shape)}"
            )

        # Proactive scale anchor with a wide dead-zone. It is zero inside
        # [low_ratio, high_ratio] relative to the endpoint log-RMS reference.
        scale_ratio_per_item = (
            z_rms_per_item
            / scale_reference_rms.clamp_min(1e-8)
        )
        scale_log_ratio = torch.log(
            scale_ratio_per_item.clamp_min(1e-8)
        )
        scale_low_log = math.log(self.scale_anchor_low_ratio)
        scale_high_log = math.log(self.scale_anchor_high_ratio)
        scale_low_violation = F.relu(
            scale_log_ratio.new_tensor(scale_low_log) - scale_log_ratio
        )
        scale_high_violation = F.relu(
            scale_log_ratio - scale_log_ratio.new_tensor(scale_high_log)
        )
        scale_loss = (
            scale_low_violation.square()
            + scale_high_violation.square()
        ).mean()
        scale_active = (
            (scale_low_violation > 0)
            | (scale_high_violation > 0)
        )
        scale_active_fraction = scale_active.float().mean()

        low_violation = F.relu(
            guard_low
            - z_rms_per_item
        )

        high_violation = F.relu(
            z_rms_per_item
            - guard_high
        )

        guard_loss = (
            low_violation.square()
            + high_violation.square()
        ).mean()

        # Fraction of prior samples currently outside the deliberately
        # loose endpoint-relative RMS interval.
        guard_active = (
            (low_violation > 0)
            | (high_violation > 0)
        )

        guard_fraction = (
            guard_active.float().mean()
        )

        # Expectation of each endpoint-relative diagnostic, matching Lproj.
        sample_endpoint_ratio = (
            alpha * z_rms_per_item / src1_rms.clamp_min(1e-8)
            + (1 - alpha) * z_rms_per_item / src2_rms.clamp_min(1e-8)
        ).mean()
        delta_endpoint_ratio = weighted["relative_delta"].mean()

        loss = (
            projection_loss
            + self.scale_anchor_weight * scale_loss
            + self.rms_guard_weight * guard_loss
        )

        metrics = {
            "trellis_prior_loss":
                loss.detach(),

            "trellis_prior_projection_loss":
                projection_loss.detach(),

            "trellis_prior_projection_delta_rms":
                weighted["raw"].mean().detach(),

            "trellis_prior_projection_tangent_delta_rms":
                weighted["tangent"].mean().detach(),

            "trellis_prior_projection_radial_fraction":
                weighted["radial"].mean().detach(),

            "trellis_prior_projection_raw_cosine_z":
                weighted["cosine"].mean().detach(),

            "trellis_prior_projection_delta_clipped_rms":
                weighted["clipped"].mean().detach(),

            "trellis_prior_projection_clip_fraction":
                weighted["clip"].mean().detach(),

            "trellis_prior_projection_x0_rms":
                weighted["x0"].mean().detach(),

            "trellis_prior_velocity_rms":
                weighted["velocity"].mean().detach(),

            "trellis_prior_t_mean":
                tau.mean().detach(),

            "trellis_prior_scale_loss":
                scale_loss.detach(),

            "trellis_prior_scale_active_fraction":
                scale_active_fraction.detach(),

            "trellis_prior_scale_reference_rms":
                scale_reference_rms.mean().detach(),

            "trellis_prior_scale_ratio":
                scale_ratio_per_item.mean().detach(),

            "trellis_prior_guard_loss":
                guard_loss.detach(),

            "trellis_prior_guard_fraction":
                guard_fraction.detach(),

            "trellis_prior_guard_low":
                guard_low.mean().detach(),

            "trellis_prior_guard_high":
                guard_high.mean().detach(),

            "trellis_prior_endpoint_rms":
                endpoint_rms.mean().detach(),

            "trellis_prior_sample_endpoint_rms_ratio":
                sample_endpoint_ratio.detach(),

            "trellis_prior_delta_endpoint_rms_ratio":
                delta_endpoint_ratio.detach(),

            "trellis_prior_src1_fraction":
                alpha.mean().detach(),

            "trellis_prior_projection_src1_loss": endpoint_losses[0].mean().detach(),
            "trellis_prior_projection_src2_loss": endpoint_losses[1].mean().detach(),
        }

        if return_loss_terms:
            return loss, metrics, {
                "projection": projection_loss,
                "scale": self.scale_anchor_weight * scale_loss,
                "guard": self.rms_guard_weight * guard_loss,
            }
        return loss, metrics
