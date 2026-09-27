"""RFDS gradient direction, isolation, native conditioning and strict loading.

Run on CPU: ATTN_BACKEND=sdpa python -m unittest discover -s tests -p 'test_trellis_ss_prior.py' -v
"""

import json
import importlib.util
import os
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.trellis_ss_prior import (
    TRELLIS_IMAGE_COND_TOKENS,
    TRELLIS_PRIOR_CHECKPOINT,
    TRELLIS_PRIOR_METRIC_NAMES,
    TRELLIS_PRIOR_REPO,
    TrellisSSPrior,
    _build_original_ss_flow,
)


class PointMassPrior(nn.Module):
    """Exact RF velocity for a data distribution concentrated at one latent.

    Denoising must move every student sample toward this point, making the
    RFDS sign and its lack of an extra (1-t) factor independently checkable.
    """

    cond_channels = 4

    def __init__(self, center=0.5, sigma_min=0.1):
        super().__init__()
        self.center = nn.Parameter(torch.tensor(float(center)))
        self.sigma_min = sigma_min
        self.calls = []

    def forward(self, x, t, cond):
        self.calls.append((x.detach().clone(), t.detach().clone(), cond.detach().clone(), torch.is_grad_enabled()))
        tau = t.reshape(-1, 1, 1, 1, 1) / 1000
        sigma_t = self.sigma_min + (1 - self.sigma_min) * tau
        inferred_noise = (x - (1 - tau) * self.center) / sigma_t
        return (1 - self.sigma_min) * inferred_noise - self.center


class TrellisSSPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_point_mass_gradient_has_correct_sign_and_normalization(self):
        flow = PointMassPrior()
        prior = TrellisSSPrior(flow, sigma_min=flow.sigma_min)
        z = torch.tensor([-0.5, 2.0, 0.0, 1.5]).reshape(2, 1, 1, 1, 2).requires_grad_()
        tau = torch.tensor([0.2, 0.8], requires_grad=True)
        noise = torch.randn_like(z, requires_grad=True)
        loss, metrics = prior(z, tau=tau, noise=noise)
        loss.backward()
        sigma_t = 0.1 + 0.9 * tau.detach().reshape(2, 1, 1, 1, 1)
        expected = (z.detach() - 0.5) / sigma_t / z.numel()
        torch.testing.assert_close(z.grad, expected)
        self.assertIsNone(flow.center.grad)
        self.assertIsNone(noise.grad)
        self.assertIsNone(tau.grad)
        self.assertFalse(flow.calls[0][-1])
        self.assertLess(float((z.detach() - 0.1 * z.grad - 0.5).square().sum()), float((z.detach() - 0.5).square().sum()))
        self.assertEqual(tuple(metrics), TRELLIS_PRIOR_METRIC_NAMES)
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))

    def test_noising_timestep_scaling_and_native_zero_conditioning(self):
        flow = PointMassPrior()
        prior = TrellisSSPrior(flow, sigma_min=0.1)
        z = torch.ones(2, 1, 1, 1, 2, requires_grad=True)
        noise = torch.full_like(z, -2)
        prior(z, tau=torch.tensor([0.2, 0.8]), noise=noise)
        noisy, timesteps, cond, _ = flow.calls[-1]
        torch.testing.assert_close(noisy[0], torch.full_like(z[0], 0.24))
        torch.testing.assert_close(noisy[1], torch.full_like(z[1], -1.44))
        torch.testing.assert_close(timesteps, torch.tensor([200.0, 800.0]))
        self.assertEqual(cond.shape, (2, TRELLIS_IMAGE_COND_TOKENS, flow.cond_channels))
        self.assertEqual(float(cond.abs().sum()), 0)
        self.assertNotIn("null_cond", prior.state_dict())

    def test_independent_uniform_timesteps_and_noise_are_sampled_per_call(self):
        flow = PointMassPrior()
        prior = TrellisSSPrior(flow, t_min=0.2, t_max=0.6)
        z = torch.zeros(64, 1, 1, 1, 1, requires_grad=True)
        torch.manual_seed(32)
        prior(z)
        prior(z)
        first_x, first_t, _, _ = flow.calls[0]
        second_x, second_t, _, _ = flow.calls[1]
        self.assertTrue(((first_t >= 200) & (first_t <= 600)).all())
        self.assertGreater(float(first_t.std()), 50)
        self.assertFalse(torch.equal(first_t, second_t))
        self.assertFalse(torch.equal(first_x, second_x))

    def test_rms_clipping_is_per_sample_and_preserves_direction(self):
        flow = PointMassPrior(center=0, sigma_min=0)
        prior = TrellisSSPrior(flow, sigma_min=0, grad_clip=1)
        z = torch.tensor([0.1, 0.2, 4.0, 8.0]).reshape(2, 1, 1, 1, 2).requires_grad_()
        loss, metrics = prior(z, tau=0.5, noise=torch.zeros_like(z))
        loss.backward()
        injected = z.grad * z.numel()
        torch.testing.assert_close(injected[0], 2 * z.detach()[0])
        torch.testing.assert_close(injected[1].square().mean().sqrt(), torch.tensor(1.0))
        torch.testing.assert_close(injected[1, ..., 1] / injected[1, ..., 0], torch.tensor([[[2.0]]]))
        self.assertGreater(float(metrics["trellis_prior_residual_rms"]), float(metrics["trellis_prior_gradient_rms"]))

    def test_parent_training_keeps_prior_frozen_and_eval(self):
        prior = TrellisSSPrior(PointMassPrior())
        parent = nn.ModuleDict({"prior": prior, "student": nn.Linear(1, 1)})
        parent.train()
        self.assertTrue(parent["student"].training)
        self.assertFalse(prior.training)
        self.assertFalse(prior.flow.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in prior.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in parent["student"].parameters()))

    def test_bfloat_student_receives_gradient_with_fp32_residual(self):
        prior = TrellisSSPrior(PointMassPrior())
        z = torch.ones(1, 1, 1, 1, 2, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, metrics = prior(z, tau=0.5, noise=torch.zeros_like(z))
        loss.backward()
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(metrics["trellis_prior_residual_rms"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(float(z.grad.abs().sum()), 0)

    def test_configuration_and_bad_samples_fail_clearly(self):
        for kwargs in (
            {"t_min": 0}, {"t_max": 1}, {"t_min": 0.8, "t_max": 0.2},
            {"t_min": float("nan")}, {"sigma_min": 1}, {"sigma_min": -0.1},
            {"grad_clip": -1}, {"grad_clip": float("inf")}, {"cond_tokens": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrellisSSPrior(PointMassPrior(), **kwargs)
        prior = TrellisSSPrior(PointMassPrior())
        z = torch.ones(2, 1, 1, 1, 1, requires_grad=True)
        for tau in (0, 1, float("nan"), torch.ones(3), torch.ones(2, 1)):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                prior(z, tau=tau)
        with self.assertRaisesRegex(ValueError, "same shape"):
            prior(z, noise=torch.ones(1))
        with self.assertRaisesRegex(FloatingPointError, "Non-finite"):
            prior(z, noise=torch.full_like(z, float("nan")))

    def test_pretrained_loading_uses_both_original_assets_and_strict_keys(self):
        hub = ModuleType("huggingface_hub")
        safetensors = ModuleType("safetensors")
        safe_torch = ModuleType("safetensors.torch")
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "prior.json"
            config.write_text(json.dumps({"name": "SparseStructureFlowModel", "args": {}}))

            def download(**kwargs):
                calls.append(kwargs)
                return str(config) if kwargs["filename"].endswith(".json") else "prior.safetensors"

            hub.hf_hub_download = download
            safe_torch.load_file = lambda _: {"center": torch.tensor(2.5)}
            replacements = {"huggingface_hub": hub, "safetensors": safetensors, "safetensors.torch": safe_torch}
            with patch.dict("sys.modules", replacements), patch(
                "models.trellis_ss_prior._build_original_ss_flow", side_effect=lambda *args, **kwargs: PointMassPrior()
            ):
                prior = TrellisSSPrior.from_pretrained(device="cpu", dtype=torch.float32)
                self.assertEqual(float(prior.flow.center), 2.5)
                self.assertFalse(prior.flow.center.requires_grad)
                self.assertEqual(calls, [
                    {"repo_id": TRELLIS_PRIOR_REPO, "filename": f"{TRELLIS_PRIOR_CHECKPOINT}.json"},
                    {"repo_id": TRELLIS_PRIOR_REPO, "filename": f"{TRELLIS_PRIOR_CHECKPOINT}.safetensors"},
                ])
                safe_torch.load_file = lambda _: {"center": torch.tensor(2.5), "alpha_embedder.weight": torch.ones(1)}
                with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
                    TrellisSSPrior.from_pretrained()
                safe_torch.load_file = lambda _: {}
                with self.assertRaisesRegex(RuntimeError, "Missing key"):
                    TrellisSSPrior.from_pretrained()

    @unittest.skipUnless(importlib.util.find_spec("spconv"), "Local SS flow utility imports require spconv")
    def test_native_forward_matches_zero_alpha_base_without_morph_parameters(self):
        # This tiny dense model uses real attention/transformer code on CPU.
        # No spconv, sparse latents, DINO or external TRELLIS package is imported.
        with patch.dict(os.environ, {"ATTN_BACKEND": "sdpa"}):
            config = dict(
                resolution=2, in_channels=2, out_channels=2, model_channels=12,
                cond_channels=4, num_blocks=2, num_heads=2, patch_size=1,
                mlp_ratio=2, qk_rms_norm=True,
            )
            native = _build_original_ss_flow(config, dtype=torch.float32)
            from models.sparse_structure_flow import SparseStructureFlowModel
            base = SparseStructureFlowModel(**config)
        # Make the output nonzero so this compares the actual transformer path.
        nn.init.normal_(native.out_layer.weight, std=0.05)
        missing, unexpected = base.load_state_dict(native.state_dict(), strict=False)
        self.assertTrue(missing)
        self.assertTrue(all(name.startswith("alpha_embedder.") for name in missing))
        self.assertEqual(unexpected, [])
        self.assertFalse(any("alpha" in key or "lora" in key or "cross_attn2" in key for key in native.state_dict()))
        z, t, cond = torch.randn(1, 2, 2, 2, 2), torch.tensor([700.0]), torch.randn(1, 3, 4)
        with torch.no_grad():
            torch.testing.assert_close(native(z, t, cond), base(z, t, cond), rtol=0, atol=0)
        self.assertGreater(float(native(z, t, cond).abs().sum()), 0)

    @unittest.skipUnless(importlib.util.find_spec("spconv"), "Local SS flow utility imports require spconv")
    def test_native_bfloat_torso_keeps_normalization_fp32_and_runs_without_autocast(self):
        with patch.dict(os.environ, {"ATTN_BACKEND": "sdpa"}):
            native = _build_original_ss_flow(dict(
                resolution=2, in_channels=2, out_channels=2, model_channels=12,
                cond_channels=4, num_blocks=1, num_heads=2, patch_size=1,
                mlp_ratio=2, qk_rms_norm=True,
            ), dtype=torch.bfloat16)
        self.assertEqual(native.blocks[0].cross_attn.to_q.weight.dtype, torch.bfloat16)
        self.assertEqual(native.blocks[0].norm2.weight.dtype, torch.float32)
        self.assertEqual(native.blocks[0].self_attn.q_rms_norm.gamma.dtype, torch.float32)
        self.assertEqual(native.t_embedder.mlp[0].weight.dtype, torch.float32)
        self.assertEqual(native.out_layer.weight.dtype, torch.float32)
        nn.init.normal_(native.out_layer.weight, std=0.05)
        prior = TrellisSSPrior(native, cond_tokens=3)
        z = torch.randn(1, 2, 2, 2, 2, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, _ = prior(z)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(float(z.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
