"""Small CPU tests; no TRELLIS weights, DINO, spconv or GPU are loaded."""
import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from modules.prior_gradient_balance import BalanceConfig, PriorGradientBalancer


class FakeAccelerator:
    scaler = None

    def backward(self, loss, **kwargs):
        loss.backward(**kwargs)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3, 4)
        self.b = nn.Linear(4, 2)
        self.unused = nn.Parameter(torch.tensor(3.0))

    def forward(self, x, target, checkpointed=False, active_guard=False):
        # Separate FM and rollout forwards sharing the same parameter leaves.
        def evaluate(v):
            return self.b(torch.tanh(self.a(v)))
        fm = (evaluate(x) - target).square().mean()
        z = checkpoint(evaluate, x + 0.3, use_reentrant=False) if checkpointed else evaluate(x + 0.3)
        # Detached pseudo-teacher target; not an endpoint RMS target.
        projection = 0.5 * (z - (z.detach() + 0.12)).square().mean()
        guard = torch.relu((10.0 if active_guard else 0.0) - z.square().mean().clamp_min(1e-12).sqrt()).square()
        return dict(fm=fm, projection=projection, guard=guard)


def setup(config=None):
    torch.manual_seed(12)
    net = ToyModel()
    opt = torch.optim.AdamW([
        dict(params=list(net.a.parameters()), name='condition'),
        dict(params=list(net.b.parameters()), name='lora'),
        dict(params=[net.unused], name='flow_adapter')], lr=1e-3)
    bal = PriorGradientBalancer(opt, config or BalanceConfig(warmup_steps=0, ema=0), 70000)
    return net, opt, bal


class BalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ratio_and_weighted_gradient_match_ordinary_backward(self):
        net, opt, bal = setup()
        reference = copy.deepcopy(net)
        x, target = torch.randn(2, 3), torch.randn(2, 2)
        parts = net(x, target)
        loss, metrics = bal.backward_parts(parts, FakeAccelerator(), net, 70000)
        weight = metrics['trellis_prior_weight']
        ref_parts = reference(x, target)
        (ref_parts['fm'] + weight * ref_parts['projection']).backward()
        self.assertAlmostEqual(metrics['balance/ratio_projection_fm'], 0.5, places=6)
        self.assertFalse(loss.requires_grad)
        for p, r in zip(net.parameters(), reference.parameters()):
            if r.grad is None:
                self.assertIsNone(p.grad)
            else:
                torch.testing.assert_close(p.grad, r.grad, atol=2e-6, rtol=2e-5)

    def test_guard_has_independent_coefficient(self):
        net, opt, bal = setup()
        reference = copy.deepcopy(net)
        x, target = torch.randn(2, 3), torch.randn(2, 2)
        parts = net(x, target, active_guard=True)
        loss, metrics = bal.backward_parts(parts, FakeAccelerator(), net, 70000)
        ref_parts = reference(x, target, active_guard=True)
        (ref_parts['fm'] + metrics['trellis_prior_weight'] * ref_parts['projection']
         + metrics['balance/guard_weight'] * ref_parts['guard']).backward()
        self.assertGreater(metrics['balance/ratio_guard_fm'], 0)
        self.assertAlmostEqual(metrics['balance/ratio_projection_fm'], 0.5, places=6)
        for p, r in zip(net.parameters(), reference.parameters()):
            if r.grad is not None:
                torch.testing.assert_close(p.grad, r.grad, atol=2e-6, rtol=2e-5)

    def test_phase_warmup_uses_new_phase_not_global_step(self):
        net, opt, bal = setup(BalanceConfig(warmup_steps=500, ema=0))
        parts = net(torch.randn(2, 3), torch.randn(2, 2))
        _, metrics = bal.backward_parts(parts, FakeAccelerator(), net, 70000)
        self.assertAlmostEqual(metrics['balance/ramp'], 1/500)
        self.assertEqual(metrics['balance/phase_step'], 1)
        self.assertTrue(bal.active(70000, 2))
        self.assertFalse(bal.active(70001, 2))
        self.assertTrue(bal.active(70002, 2))

    def test_checkpointed_bfloat_graphs_with_guard(self):
        net, opt, bal = setup()
        with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
            parts = net(torch.randn(2, 3), torch.randn(2, 2), checkpointed=True, active_guard=True)
        _, metrics = bal.backward_parts(parts, FakeAccelerator(), net, 70010)
        self.assertAlmostEqual(metrics['balance/ratio_projection_fm'], 0.5, places=6)
        for p in bal.params:
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_ratio_cap_and_weak_weight_limit_are_visible(self):
        net, opt, bal = setup(BalanceConfig(start_weight=1e5, warmup_steps=500, max_ratio=1.0))
        _, metrics = bal.backward_parts(net(torch.randn(2, 3), torch.randn(2, 2)), FakeAccelerator(), net, 70000)
        self.assertLessEqual(metrics['balance/ratio_projection_fm'], 1.000001)
        self.assertEqual(metrics['balance/ratio_cap_hit'], 1.0)
        net, opt, bal = setup(BalanceConfig(min_weight=1e-6, max_weight=1e-6, warmup_steps=0))
        _, metrics = bal.backward_parts(net(torch.randn(2, 3), torch.randn(2, 2)), FakeAccelerator(), net, 70000)
        self.assertEqual(metrics['balance/lambda_max_hit'], 1.0)
        self.assertEqual(metrics['balance/weak_streak'], 1.0)

    def test_persistent_weak_signal_aborts(self):
        net, opt, bal = setup(BalanceConfig(min_weight=1e-6, max_weight=1e-6,
                                          warmup_steps=0, weak_patience=1))
        with self.assertRaisesRegex(RuntimeError, 'remains too weak'):
            bal.backward_parts(net(torch.randn(2, 3), torch.randn(2, 2)),
                               FakeAccelerator(), net, 70000)

    def test_controller_state_resume(self):
        net, opt, bal = setup()
        bal.backward_parts(net(torch.randn(2, 3), torch.randn(2, 2)), FakeAccelerator(), net, 70000)
        net2, opt2, bal2 = setup()
        bal2.load_state_dict(bal.state_dict())
        self.assertEqual(bal.state_dict(), bal2.state_dict())

    def test_zero_gradient_fails_before_optimizer_step(self):
        net, opt, bal = setup()
        parts = net(torch.randn(2, 3), torch.randn(2, 2))
        parts['projection'] = parts['projection'] * 0
        with self.assertRaisesRegex(RuntimeError, 'connectivity'):
            bal.backward_parts(parts, FakeAccelerator(), net, 70000)

    def test_shape_grid_and_clipped_target_mean_reduction(self):
        # Real SS grid size, distinct endpoint scale per item. Targets detached.
        z = nn.Parameter(torch.randn(2, 8, 16, 16, 16))
        opt = torch.optim.SGD([dict(params=[z], name='condition')], lr=0.1)
        bal = PriorGradientBalancer(opt, BalanceConfig(warmup_steps=0, ema=0), 70000)
        raw = torch.randn_like(z)
        raw_rms = raw.flatten(1).square().mean(1).sqrt()
        max_rms = torch.tensor([0.6, 0.8]) * 0.15
        delta = raw * (max_rms / raw_rms).clamp(max=1).view(2,1,1,1,1)
        parts = dict(fm=z.square().mean(),
                     projection=0.5*(z-(z.detach()+delta)).square().mean(),
                     guard=z.square().mean()*0)
        _, metrics = bal.backward_parts(parts, FakeAccelerator(), nn.Identity(), 70000)
        self.assertAlmostEqual(metrics['balance/ratio_projection_fm'], 0.5, places=6)
        self.assertEqual(z.grad.shape, z.shape)


if __name__ == '__main__':
    unittest.main()
