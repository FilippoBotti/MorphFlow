"""Tests for the endpoint-conditioned TRELLIS SS projection prior."""

import unittest

import torch
from torch import nn

from models.trellis_ss_prior import (
    TRELLIS_PRIOR_METRIC_NAMES,
    TrellisSSPrior,
)


class FakeImageEncoder(nn.Module):
    """Tiny deterministic replacement for DINO."""

    def forward(self, image, is_training=True):
        m = image.mean(dim=(1, 2, 3))

        token = torch.stack(
            (
                m,
                -m,
                1.0 + m,
                -1.0 - m,
            ),
            dim=-1,
        )

        return {
            "x_prenorm": token[:, None, :].repeat(1, 2, 1)
        }


class ConditionalPointMassFlow(nn.Module):
    """Exact RF field for a conditional point-mass clean distribution."""

    cond_channels = 4

    def __init__(self, sigma_min=0.1, offset=0.0):
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.offset = float(offset)
        self.calls = []

    @staticmethod
    def center_from_cond(cond):
        return (
            cond[:, :, 0]
            .mean(dim=1)
            .view(-1, 1, 1, 1, 1)
        )

    def forward(self, x, t, cond):
        self.calls.append(
            (
                x.detach().clone(),
                t.detach().clone(),
                cond.detach().clone(),
                torch.is_grad_enabled(),
            )
        )

        center = self.center_from_cond(cond) + self.offset

        tau = t.view(-1, 1, 1, 1, 1) / 1000.0

        sigma_t = (
            self.sigma_min
            + (1.0 - self.sigma_min) * tau
        )

        inferred_noise = (
            x - (1.0 - tau) * center
        ) / sigma_t

        return (
            (1.0 - self.sigma_min) * inferred_noise
            - center
        )


def make_inputs(batch=2):
    z = torch.tensor(
        [0.30, -0.40],
        dtype=torch.float32,
    )[:batch].view(batch, 1, 1, 1, 1)

    src1_image = torch.ones(batch, 3, 2, 2)
    src2_image = torch.zeros(batch, 3, 2, 2)

    src1_ss = torch.ones(batch, 1, 1, 1, 1)
    src2_ss = torch.full(
        (batch, 1, 1, 1, 1),
        2.0,
    )

    return (
        z,
        src1_image,
        src2_image,
        src1_ss,
        src2_ss,
    )


class TrellisProjectionPriorTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_prior(
        self,
        *,
        flow=None,
        projection_clip_ratio=0.0,
        rms_guard_weight=0.0,
        t_min=0.05,
        t_max=0.20,
    ):
        if flow is None:
            flow = ConditionalPointMassFlow()

        return TrellisSSPrior(
            flow,
            FakeImageEncoder(),
            sigma_min=flow.sigma_min,
            t_min=t_min,
            t_max=t_max,
            projection_clip_ratio=projection_clip_ratio,
            rms_guard_weight=rms_guard_weight,
        )

    def test_projection_gradient_points_to_conditional_clean_x0(self):
        flow = ConditionalPointMassFlow()

        prior = self.make_prior(flow=flow)

        (
            z,
            src1_image,
            src2_image,
            src1_ss,
            src2_ss,
        ) = make_inputs()

        z = z.requires_grad_()

        with torch.no_grad():
            cond = prior.encode_image(src1_image)
            expected_center = flow.center_from_cond(cond)

        noise = torch.randn_like(z)

        loss, metrics = prior(
            z,
            src1_image=src1_image,
            src2_image=src2_image,
            alpha=torch.ones(2),
            src1_ss_latent=src1_ss,
            src2_ss_latent=src2_ss,
            tau=0.10,
            noise=noise,
        )

        loss.backward()

        # 0.5 * mean((z - stopgrad(target))^2)
        expected_grad = (
            z.detach() - expected_center
        ) / z.numel()

        torch.testing.assert_close(
            z.grad,
            expected_grad,
            rtol=1e-5,
            atol=1e-6,
        )

        self.assertEqual(
            tuple(metrics.keys()),
            TRELLIS_PRIOR_METRIC_NAMES,
        )

        self.assertAlmostEqual(
            float(metrics["trellis_prior_src1_fraction"]),
            1.0,
        )

        # Frozen TRELLIS forward must not build a graph.
        self.assertFalse(flow.calls[-1][-1])

    def test_alpha_one_selects_src1(self):
        flow = ConditionalPointMassFlow()
        prior = self.make_prior(flow=flow)

        (
            z,
            src1_image,
            src2_image,
            src1_ss,
            src2_ss,
        ) = make_inputs()

        prior(
            z.requires_grad_(),
            src1_image=src1_image,
            src2_image=src2_image,
            alpha=torch.ones(2),
            src1_ss_latent=src1_ss,
            src2_ss_latent=src2_ss,
            tau=0.10,
            noise=torch.zeros_like(z),
        )

        self.assertEqual(len(flow.calls), 2)
        used_cond = flow.calls[0][2]
        expected_cond = prior.encode_image(src1_image)

        torch.testing.assert_close(
            used_cond,
            expected_cond,
        )

    def test_alpha_zero_selects_src2(self):
        flow = ConditionalPointMassFlow()
        prior = self.make_prior(flow=flow)

        (
            z,
            src1_image,
            src2_image,
            src1_ss,
            src2_ss,
        ) = make_inputs()

        prior(
            z.requires_grad_(),
            src1_image=src1_image,
            src2_image=src2_image,
            alpha=torch.zeros(2),
            src1_ss_latent=src1_ss,
            src2_ss_latent=src2_ss,
            tau=0.10,
            noise=torch.zeros_like(z),
        )

        used_cond = flow.calls[-1][2]
        expected_cond = prior.encode_image(src2_image)

        torch.testing.assert_close(
            used_cond,
            expected_cond,
        )

    def test_dual_projection_loss_and_gradient_with_independent_clipping(self):
        for clip in (0.0, 0.10):
            flow = ConditionalPointMassFlow()
            prior = self.make_prior(flow=flow, projection_clip_ratio=clip)
            z, image1, image2, endpoint1, endpoint2 = make_inputs()
            z.requires_grad_()
            alpha = torch.tensor([0.2, 0.8])
            noise = torch.randn_like(z)
            targets = []
            for image, endpoint in ((image1, endpoint1), (image2, endpoint2)):
                target = flow.center_from_cond(prior.encode_image(image))
                delta = target - z.detach()
                if clip:
                    radius = clip * endpoint.abs()
                    delta = delta.clamp(min=-radius, max=radius)
                targets.append(z.detach() + delta)
            loss, metrics, terms = prior(
                z, src1_image=image1, src2_image=image2, alpha=alpha,
                src1_ss_latent=endpoint1, src2_ss_latent=endpoint2,
                tau=torch.tensor([0.1, 0.15]), noise=noise, return_loss_terms=True,
            )
            weight = alpha.view(-1, 1, 1, 1, 1)
            expected = 0.5 * (weight * (z - targets[0]).square()
                              + (1 - weight) * (z - targets[1]).square()).mean()
            torch.testing.assert_close(loss, expected)
            torch.testing.assert_close(loss, sum(terms.values()))
            loss.backward()
            torch.testing.assert_close(z.grad, (z.detach() - weight * targets[0]
                                               - (1 - weight) * targets[1]) / z.numel())
            self.assertEqual(len(flow.calls), 2)
            torch.testing.assert_close(flow.calls[0][0], flow.calls[1][0])
            torch.testing.assert_close(flow.calls[0][1], flow.calls[1][1])
            self.assertTrue(all(not call[-1] for call in flow.calls))
            self.assertTrue(all(not value.requires_grad for value in metrics.values()))

    def test_low_noise_timestep_range(self):
        batch = 64

        flow = ConditionalPointMassFlow()

        prior = self.make_prior(
            flow=flow,
            t_min=0.05,
            t_max=0.20,
        )

        z = torch.randn(
            batch,
            1,
            1,
            1,
            1,
            requires_grad=True,
        )

        image = torch.rand(batch, 3, 2, 2)
        endpoint = torch.ones_like(z)

        prior(
            z,
            src1_image=image,
            src2_image=image,
            alpha=torch.full((batch,), 0.5),
            src1_ss_latent=endpoint,
            src2_ss_latent=endpoint,
        )

        timesteps = flow.calls[-1][1]

        self.assertTrue((timesteps >= 50.0).all())
        self.assertTrue((timesteps <= 200.0).all())

        # Ensure timestep is independently sampled rather than constant.
        self.assertGreater(
            float(timesteps.std()),
            1.0,
        )

    def test_projection_trust_region(self):
        flow = ConditionalPointMassFlow(
            offset=5.0
        )

        prior = self.make_prior(
            flow=flow,
            projection_clip_ratio=0.10,
        )

        z = torch.zeros(
            1,
            1,
            1,
            1,
            1,
            requires_grad=True,
        )

        image = torch.ones(1, 3, 2, 2)

        # RMS endpoint = 2.
        endpoint = torch.full_like(z, 2.0)

        _, metrics = prior(
            z,
            src1_image=image,
            src2_image=image,
            alpha=torch.ones(1),
            src1_ss_latent=endpoint,
            src2_ss_latent=endpoint,
            tau=0.10,
            noise=torch.zeros_like(z),
        )

        # clip_ratio=.1 => max delta RMS=.2
        self.assertAlmostEqual(
            float(
                metrics[
                    "trellis_prior_projection_delta_clipped_rms"
                ]
            ),
            0.2,
            places=5,
        )

        self.assertAlmostEqual(
            float(
                metrics[
                    "trellis_prior_projection_clip_fraction"
                ]
            ),
            1.0,
        )

    def test_rms_guard_prevents_zero_collapse(self):
        flow = ConditionalPointMassFlow()

        prior = TrellisSSPrior(
            flow,
            FakeImageEncoder(),
            sigma_min=flow.sigma_min,
            t_min=0.05,
            t_max=0.20,
            projection_clip_ratio=0.10,
            rms_guard_weight=1.0,
            rms_guard_low_ratio=0.25,
            rms_guard_high_ratio=2.0,
        )

        # Student sample intentionally much smaller than real endpoints.
        z = torch.full(
            (1, 1, 1, 1, 1),
            0.05,
            requires_grad=True,
        )

        image = torch.ones(1, 3, 2, 2)

        src1_ss = torch.ones_like(z)
        src2_ss = torch.full_like(z, 2.0)

        loss, metrics = prior(
            z,
            src1_image=image,
            src2_image=image,
            alpha=torch.ones(1),
            src1_ss_latent=src1_ss,
            src2_ss_latent=src2_ss,
            tau=0.10,
            noise=torch.zeros_like(z),
        )

        self.assertGreater(
            float(
                metrics[
                    "trellis_prior_guard_loss"
                ]
            ),
            0.0,
        )

        loss.backward()

        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(
            float(z.grad.abs().sum()),
            0.0,
        )

    def test_guard_is_zero_inside_valid_scale_range(self):
        flow = ConditionalPointMassFlow()

        prior = TrellisSSPrior(
            flow,
            FakeImageEncoder(),
            sigma_min=flow.sigma_min,
            projection_clip_ratio=0.0,
            rms_guard_weight=1.0,
            rms_guard_low_ratio=0.25,
            rms_guard_high_ratio=2.0,
        )

        z = torch.ones(
            1,
            1,
            1,
            1,
            1,
            requires_grad=True,
        )

        image = torch.ones(1, 3, 2, 2)

        src1_ss = torch.ones_like(z)
        src2_ss = torch.full_like(z, 2.0)

        _, metrics = prior(
            z,
            src1_image=image,
            src2_image=image,
            alpha=torch.ones(1),
            src1_ss_latent=src1_ss,
            src2_ss_latent=src2_ss,
            tau=0.10,
            noise=torch.zeros_like(z),
        )

        self.assertEqual(
            float(
                metrics[
                    "trellis_prior_guard_loss"
                ]
            ),
            0.0,
        )

    def test_parent_train_keeps_both_supervisors_frozen(self):
        prior = self.make_prior()

        parent = nn.ModuleDict(
            {
                "prior": prior,
                "student": nn.Linear(1, 1),
            }
        )

        parent.train()

        self.assertFalse(prior.training)
        self.assertFalse(prior.flow.training)
        self.assertFalse(prior.image_encoder.training)

        self.assertTrue(
            all(
                not p.requires_grad
                for p in prior.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
