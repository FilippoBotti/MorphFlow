"""Frozen, native TRELLIS image SS prior for rectified-flow distillation.

The prior uses zero DINO conditioning, as in the upstream image pipeline:
https://github.com/microsoft/TRELLIS/blob/main/trellis/pipelines/trellis_image_to_3d.py
Its SS forward is the original (non-morphing) flow:
https://github.com/microsoft/TRELLIS/blob/main/trellis/models/sparse_structure_flow.py

Only PyTorch is imported until a pretrained model is requested. RFDS unit tests
therefore do not need TRELLIS, sparse-convolution libraries, or GPU attention.
"""

import json
import math

import torch
from torch import nn
from torch.nn import functional as F


TRELLIS_PRIOR_REPO = "microsoft/TRELLIS-image-large"
TRELLIS_PRIOR_CHECKPOINT = "ckpts/ss_flow_img_dit_L_16l8_fp16"
# DINOv2 ViT-L/14 with registers: 37*37 patch tokens, CLS, four registers.
TRELLIS_IMAGE_COND_TOKENS = 1374
TRELLIS_PRIOR_METRIC_NAMES = (
    "trellis_prior_loss",
    "trellis_prior_residual_rms",
    "trellis_prior_gradient_rms",
    "trellis_prior_t_mean",
    "trellis_prior_velocity_rms",
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
    """Inject a stop-gradient RFDS residual into a generated SS latent.

For independent t ~ Uniform(t_min, t_max), use the TRELLIS convention
``x_t = (1-t) z + (sigma_min + (1-sigma_min)t) noise`` and unit weight w(t).
The desired gradient is ``v_prior - ((1-sigma_min) noise - z)``. The returned
surrogate is mean-reduced over all latent elements, matching the FM loss scale;
its derivative is that residual divided by ``z.numel()``. Its scalar value is
a diagnostic, not a likelihood or an independent quality metric.

``grad_clip`` optionally bounds each sample's residual RMS before injection.
Neither the prior, the noising operation, nor the residual is differentiated.
"""

    def __init__(
        self,
        flow,
        *,
        sigma_min=1e-5,
        t_min=0.05,
        t_max=0.95,
        grad_clip=0.0,
        cond_tokens=TRELLIS_IMAGE_COND_TOKENS,
    ):
        super().__init__()
        if not math.isfinite(sigma_min) or not 0 <= sigma_min < 1:
            raise ValueError("sigma_min must be finite and in [0, 1)")
        if not (math.isfinite(t_min) and math.isfinite(t_max) and 0 < t_min < t_max < 1):
            raise ValueError("Prior timesteps must satisfy 0 < t_min < t_max < 1")
        if not math.isfinite(grad_clip) or grad_clip < 0:
            raise ValueError("grad_clip must be finite and nonnegative")
        if not isinstance(cond_tokens, int) or isinstance(cond_tokens, bool) or cond_tokens < 1:
            raise ValueError("cond_tokens must be a positive integer")
        self.flow = flow.requires_grad_(False).eval()
        self.sigma_min = float(sigma_min)
        self.t_min, self.t_max = float(t_min), float(t_max)
        self.grad_clip = float(grad_clip)
        first_parameter = next(flow.parameters(), None)
        device = first_parameter.device if first_parameter is not None else None
        self.register_buffer(
            "null_cond",
            torch.zeros(1, cond_tokens, flow.cond_channels, device=device),
            persistent=False,
        )
        self.eval()

    @classmethod
    def from_pretrained(cls, *, device=None, dtype=None, **kwargs):
        """Strictly load the original image-large JSON and safetensors assets.

        Hugging Face's usual cache and HF_HUB_OFFLINE settings are respected.
        This loads only the SS flow, without DINO, decoders or the SLat model.
        ``dtype`` controls the frozen transformer torso; None preserves the
        checkpoint's native FP16 configuration.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        config_path = hf_hub_download(
            repo_id=TRELLIS_PRIOR_REPO, filename=f"{TRELLIS_PRIOR_CHECKPOINT}.json"
        )
        checkpoint_path = hf_hub_download(
            repo_id=TRELLIS_PRIOR_REPO, filename=f"{TRELLIS_PRIOR_CHECKPOINT}.safetensors"
        )
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("name") != "SparseStructureFlowModel":
            raise ValueError("Expected an original TRELLIS SparseStructureFlowModel checkpoint")
        # Loading the prior must not change the student's training RNG stream.
        with torch.random.fork_rng(devices=[]):
            flow = _build_original_ss_flow(config["args"], dtype=dtype)
        flow.load_state_dict(load_file(checkpoint_path), strict=True)
        prior = cls(flow, **kwargs)
        return prior.to(device=device) if device is not None else prior

    def train(self, mode=True):
        # A parent's train() call must never reactivate dropout or prior grads.
        super().train(False)
        return self

    def forward(self, z, *, tau=None, noise=None):
        if z.ndim != 5 or z.shape[0] < 1 or not z.is_floating_point():
            raise ValueError("The TRELLIS prior expects a nonempty floating SS tensor [B,C,H,W,D]")
        with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
            detached_z = z.detach().float()
            batch_size = z.shape[0]
            if tau is None:
                tau = torch.rand(batch_size, device=z.device) * (self.t_max - self.t_min) + self.t_min
            else:
                tau = torch.as_tensor(tau, device=z.device, dtype=torch.float32).detach()
                if tau.numel() == 1:
                    tau = tau.expand(batch_size)
                if tuple(tau.shape) != (batch_size,):
                    raise ValueError("tau must be a scalar or a [B] tensor")
                if not torch.isfinite(tau).all() or not ((tau >= self.t_min) & (tau <= self.t_max)).all():
                    raise ValueError("tau must be finite and inside [t_min, t_max]")
            if noise is None:
                noise = torch.randn_like(detached_z)
            else:
                if noise.shape != z.shape:
                    raise ValueError("noise must have the same shape as z")
                noise = noise.detach().to(device=z.device, dtype=torch.float32)
            time = tau.view(batch_size, 1, 1, 1, 1)
            sigma_t = self.sigma_min + (1 - self.sigma_min) * time
            x_t = (1 - time) * detached_z + sigma_t * noise
            native_null = self.null_cond.expand(batch_size, -1, -1)
            velocity = self.flow(x_t, tau * 1000.0, native_null).float()
            if velocity.shape != z.shape:
                raise ValueError("The TRELLIS prior velocity shape must match z")
            residual = velocity - ((1 - self.sigma_min) * noise - detached_z)
            if not torch.isfinite(residual).all():
                raise FloatingPointError("Non-finite TRELLIS RFDS residual; inspect student samples and prior precision")
            residual_rms = residual.square().mean().sqrt()
            gradient = residual
            if self.grad_clip > 0:
                sample_rms = residual.flatten(1).square().mean(1).sqrt()
                scale = (self.grad_clip / sample_rms.clamp_min(1e-12)).clamp(max=1)
                gradient = gradient * scale.view(batch_size, 1, 1, 1, 1)
            target = detached_z - gradient

        # The detached target supplies exactly the RFDS gradient, including its
        # sign, without a prior Jacobian or a (1-t) chain-rule factor.
        loss = 0.5 * F.mse_loss(z.float(), target, reduction="mean")
        metrics = {
            "trellis_prior_loss": loss.detach(),
            "trellis_prior_residual_rms": residual_rms,
            "trellis_prior_gradient_rms": gradient.square().mean().sqrt(),
            "trellis_prior_t_mean": tau.mean(),
            "trellis_prior_velocity_rms": velocity.square().mean().sqrt(),
        }
        return loss, metrics
