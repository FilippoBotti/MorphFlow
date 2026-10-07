"""CPU checks for the two-endpoint sparse projection and its actual gradients."""
import unittest

import torch
from torch import nn

from modules import sparse as sp
from models.trellis_slat_prior import TrellisSLatPrior, TRELLIS_PRIOR_METRIC_NAMES


class ImageEncoder(nn.Module):
    def forward(self, image, is_training=True):
        m = image.mean((1, 2, 3))
        return {"x_prenorm": torch.stack((m, -m, 1 + m, -1 - m), -1)[:, None]}


class PointMassFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.2))
        self.calls = []

    def forward(self, x, t, cond):
        self.calls.append((x.feats.clone(), t.clone(), cond.clone(), torch.is_grad_enabled()))
        ids = x.coords[:, 0].long()
        center = cond[:, 0, 0][ids, None] + self.offset
        tau = t[ids, None] / 1000
        sigma = 0.1 + 0.9 * tau
        noise = (x.feats - (1 - tau) * center) / sigma
        return x.replace(0.9 * noise - center)


class SLatDualPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_both_endpoints_unequal_token_counts_and_stop_gradient(self):
        # Different sparse lengths expose mistaken batch/global alpha weighting.
        coords = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 2, 0, 0]], dtype=torch.int32)
        ids = coords[:, 0].long()
        for alpha in (torch.tensor([0.2, 0.8]), torch.ones(2), torch.zeros(2)):
            for clip in (0.0, 0.1):
                feats = torch.tensor([[0.3, 0.5], [-0.4, 0.7], [0.8, 0.1], [-0.2, 0.6]], requires_grad=True)
                z = sp.SparseTensor(feats, coords)
                src1 = z.replace(torch.ones_like(feats))
                src2 = z.replace(torch.full_like(feats, 2.0))
                image1, image2 = torch.ones(2, 3, 2, 2), torch.zeros(2, 3, 2, 2)
                flow = PointMassFlow()
                prior = TrellisSLatPrior(flow, ImageEncoder(), sigma_min=0.1,
                                        projection_clip_ratio=clip, stat_anchor_weight=0, rms_guard_weight=0)
                targets = []
                for image, radius in ((image1, clip), (image2, 2 * clip)):
                    center = prior.encode_image(image)[:, 0, 0][ids, None] + flow.offset.detach()
                    delta = center - feats.detach()
                    if clip:
                        rms = torch.stack([delta[ids == i].square().mean().sqrt() for i in range(2)])
                        delta = delta * (radius / rms).clamp(max=1)[ids, None]
                    targets.append(feats.detach() + delta)
                noise = torch.randn_like(feats)
                loss, metrics, terms = prior(z, src1_image=image1, src2_image=image2,
                                            src1_slat=src1, src2_slat=src2, alpha=alpha,
                                            tau=torch.tensor([0.1, 0.15]), noise=noise,
                                            return_loss_terms=True)
                a = alpha[ids, None]
                expected = 0.5 * (a * (feats - targets[0]).square()
                                  + (1 - a) * (feats - targets[1]).square()).mean()
                torch.testing.assert_close(loss, expected)
                torch.testing.assert_close(loss, sum(terms.values()))
                loss.backward()
                torch.testing.assert_close(feats.grad, (feats.detach() - a * targets[0]
                                                       - (1 - a) * targets[1]) / feats.numel())
                self.assertEqual(len(flow.calls), 2)
                torch.testing.assert_close(flow.calls[0][0], flow.calls[1][0])
                torch.testing.assert_close(flow.calls[0][1], flow.calls[1][1])
                self.assertTrue(all(not c[-1] for c in flow.calls))
                self.assertIsNone(flow.offset.grad)
                self.assertFalse(flow.offset.requires_grad)
                self.assertEqual(tuple(metrics), TRELLIS_PRIOR_METRIC_NAMES)
                self.assertTrue(all(not v.requires_grad for v in metrics.values()))

    def test_scale_and_guard_applied_once_and_endpoint_swap_symmetric(self):
        coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32)
        z = sp.SparseTensor(torch.full((2, 2), 0.05, requires_grad=True), coords)
        src1, src2 = z.replace(torch.ones(2, 2)), z.replace(torch.full((2, 2), 2.0))
        image1, image2 = torch.ones(1, 3, 2, 2), torch.zeros(1, 3, 2, 2)
        prior = TrellisSLatPrior(PointMassFlow(), ImageEncoder(), sigma_min=0.1)
        kwargs = dict(tau=0.1, noise=torch.zeros(2, 2), return_loss_terms=True)
        first = prior(z, src1_image=image1, src2_image=image2, alpha=torch.tensor([0.3]),
                      src1_slat=src1, src2_slat=src2, **kwargs)
        swapped = prior(z, src1_image=image2, src2_image=image1, alpha=torch.tensor([0.7]),
                        src1_slat=src2, src2_slat=src1, **kwargs)
        torch.testing.assert_close(first[0], swapped[0])
        torch.testing.assert_close(first[0], sum(first[2].values()))
        self.assertGreater(float(first[2]["scale"]), 0)
        self.assertGreater(float(first[2]["guard"]), 0)


if __name__ == "__main__":
    unittest.main()
