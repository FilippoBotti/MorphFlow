"""CPU regressions for differentiable generation and in-forward prior losses."""

import importlib.util
from datetime import timedelta
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import models
from modules.flow_rollout import differentiable_ss_rollout


class TinyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.3))
        self.adapter = nn.Parameter(torch.tensor(0.2))
        self.dropout = nn.Dropout(0.9)
        self.use_checkpoint = True
        self.calls = []

    def forward(self, x, t, condition, alpha):
        self.calls.append((torch.is_grad_enabled(), self.training, self.use_checkpoint, t.detach().clone()))
        if isinstance(condition, tuple):
            src1, src2, _ = condition
            cond = alpha[:, None, None] * src1 + (1 - alpha[:, None, None]) * src2
        else:
            cond = condition
        pooled = self.dropout(cond).mean(dim=(1, 2)).view(-1, 1, 1, 1, 1)
        return self.gate * x + self.adapter * pooled


class FlowRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_checkpoint_matches_gradients_for_tensor_and_tuple_conditions(self):
        for separate in (False, True):
            for grad_steps in (0, 1, 3):
                results = []
                for checkpoint_enabled in (False, True):
                    torch.manual_seed(18)
                    encoder = nn.Linear(3, 4)
                    flow = TinyFlow().train()
                    # A child with a distinct original mode must stay distinct.
                    flow.dropout.eval()
                    src1 = encoder(torch.randn(2, 3, 3))
                    src2 = encoder(torch.randn(2, 3, 3))
                    alpha = torch.tensor([0.25, 0.75])
                    cond = (src1, src2, alpha) if separate else src1 + src2
                    sample = differentiable_ss_rollout(
                        flow, torch.randn(2, 1, 2, 2, 2), cond, alpha,
                        steps=3, grad_steps=grad_steps, use_checkpoint=checkpoint_enabled,
                    )
                    prefix = 0 if grad_steps == 0 else 3 - grad_steps
                    self.assertEqual([call[0] for call in flow.calls], [False] * prefix + [True] * (3 - prefix))
                    for index, (_, training, checkpointing, timestep) in enumerate(flow.calls):
                        self.assertFalse(training)
                        self.assertFalse(checkpointing)
                        torch.testing.assert_close(timestep, torch.full((2,), 1000 * (1 - index / 3)))
                    sample.square().mean().backward()
                    self.assertTrue(flow.training)
                    self.assertFalse(flow.dropout.training)
                    self.assertTrue(flow.use_checkpoint)
                    self.assertTrue(all(not call[1] and not call[2] for call in flow.calls))
                    parameters = (*encoder.parameters(), *flow.parameters())
                    for parameter in parameters:
                        self.assertIsNotNone(parameter.grad)
                        self.assertGreater(float(parameter.grad.abs().sum()), 0)
                    results.append((sample.detach(), *(p.grad.clone() for p in parameters)))
                for expected, actual in zip(*results):
                    torch.testing.assert_close(actual, expected)

    def test_linear_euler_sign_and_truncated_gradient(self):
        class ConstantFlow(nn.Module):
            def forward(self, x, t, condition, alpha):
                return condition.expand_as(x)

        for grad_steps, expected_grad in ((0, -1.0), (1, -0.25), (2, -0.5), (4, -1.0)):
            condition = torch.tensor(2.0, requires_grad=True)
            noise = torch.full((1, 1, 1, 1, 1), 5.0, requires_grad=True)
            result = differentiable_ss_rollout(
                ConstantFlow(), noise, condition, torch.ones(1), steps=4, grad_steps=grad_steps,
            )
            result.sum().backward()
            torch.testing.assert_close(result, torch.full_like(noise, 3.0))
            torch.testing.assert_close(condition.grad, torch.tensor(expected_grad))
            self.assertIsNone(noise.grad)

    def test_invalid_rollout_configuration(self):
        for steps, grad_steps in ((0, 0), (2, -1), (2, 3), (2.5, 1), (3, 1.5)):
            with self.subTest(steps=steps, grad_steps=grad_steps), self.assertRaises(ValueError):
                differentiable_ss_rollout(
                    TinyFlow(), torch.zeros(1, 1, 1, 1, 1), torch.zeros(1, 1, 1),
                    torch.ones(1), steps=steps, grad_steps=grad_steps,
                )


def load_morphflow_without_optional_kernels():
    """Exercise real MorphFlow orchestration with a small CPU architecture."""
    path = Path(__file__).resolve().parents[1] / "models" / "morph_flow.py"
    spec = importlib.util.spec_from_file_location("_morphflow_rollout_probe", path)
    module = importlib.util.module_from_spec(spec)
    # The class methods below are real; only its unused heavy constructor's
    # dependencies are stubbed, so tests need neither spconv nor flash-attn.
    with patch.object(models, "sparse_structure_flow", SimpleNamespace(), create=True), \
            patch.object(models, "cond_encoder", SimpleNamespace(), create=True):
        spec.loader.exec_module(module)
    return module.MorphFlow


class MorphFlowPriorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        morphflow = load_morphflow_without_optional_kernels()

        class TinyMorphFlow(morphflow):
            def __init__(self):
                nn.Module.__init__(self)
                self.encoder = nn.Linear(2, 4)
                self.sparse_structure_flow = TinyFlow()
                self.sigma_min = 1e-5
                self.t_schedule = "uniform"
                self.condition_calls = []
                self._init_semantic_token_matching(use_semantic_token_matching=False)

            def _build_condition(self, f1, f2, c1, c2, alpha, apply_cfg_drop=True):
                self.condition_calls.append((self.training, apply_cfg_drop, torch.is_grad_enabled(), len(alpha)))
                if not apply_cfg_drop:
                    assert not self._semantic_match_record_aux
                    assert not self._semantic_match_record_usage
                    assert not self.semantic_match_log_stats
                def pool(feats, coords):
                    return torch.stack([self.encoder(feats[coords[:, 0] == item]).mean(0)
                                        for item in range(len(alpha))]).unsqueeze(1)
                return pool(f1, c1), pool(f2, c2), alpha

        cls.model_type = TinyMorphFlow

    def test_prior_is_in_main_forward_and_preserves_state_dict_and_fm_metrics(self):
        from models.trellis_ss_prior import TRELLIS_PRIOR_METRIC_NAMES

        model = self.model_type().train()
        state_keys = tuple(model.state_dict())
        coords = torch.tensor([[0, 1, 2, 3], [0, 2, 2, 3], [1, 3, 2, 1]])
        values = (
            torch.full((2, 1, 2, 2, 2), 1000.0),
            torch.randn(3, 2), coords, torch.randn(3, 2), coords,
            torch.tensor([0.25, 0.75]),
        )
        samples = []
        def prior(sample):
            samples.append(sample.detach().clone())
            loss = sample.square().mean()
            return loss, {key: loss.detach() for key in TRELLIS_PRIOR_METRIC_NAMES}

        torch.manual_seed(14)
        ordinary_loss = model(*values)
        ordinary_metrics = dict(model.last_forward_metrics)
        model.condition_calls.clear()
        torch.manual_seed(14)
        loss = model(*values, trellis_prior=prior, trellis_prior_weight=0.1,
                     trellis_prior_rollout_steps=3, trellis_prior_grad_steps=1)
        self.assertEqual(model.condition_calls, [(True, True, True, 2), (False, False, True, 1)])
        self.assertEqual(tuple(model.state_dict()), state_keys)
        self.assertEqual(set(model.last_forward_metrics), set(ordinary_metrics))
        torch.testing.assert_close(model.last_forward_metrics["base_mse"], ordinary_metrics["base_mse"])
        torch.testing.assert_close(loss, ordinary_loss + 0.1 * samples[0].square().mean())
        self.assertLess(float(samples[0].abs().max()), 10)
        self.assertEqual(samples[0].shape[0], 1)
        self.assertTrue(all(not value.requires_grad for value in model.last_forward_metrics.values()))
        loss.backward()
        self.assertGreater(float(model.encoder.weight.grad.abs().sum()), 0)
        model.eval()
        with torch.no_grad():
            model(*values)
        self.assertEqual(set(model.last_forward_metrics), set(ordinary_metrics))
        self.assertEqual(float(model.last_forward_metrics["trellis_prior_active"]), 0)

    def test_full_batch_selection_and_semantic_state_restoration(self):
        model = self.model_type().train()
        model._semantic_match_record_aux = True
        model._semantic_match_record_usage = True
        sentinel = torch.tensor(7.0)
        model._semantic_match_cycle_terms = [sentinel]
        model._semantic_match_last_metrics = {"sentinel": sentinel}
        coords = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0]])
        target = torch.zeros(2, 1, 2, 2, 2)
        sample = model.sample_ss_for_prior(
            target, torch.randn(2, 2), coords, torch.randn(2, 2), coords,
            torch.tensor([0.25, 0.75]), steps=2, grad_steps=1, max_items=0,
        )
        self.assertEqual(len(sample), 2)
        self.assertTrue(model._semantic_match_record_aux)
        self.assertTrue(model._semantic_match_record_usage)
        self.assertTrue(model.semantic_match_log_stats)
        self.assertIs(model._semantic_match_cycle_terms[0], sentinel)
        self.assertIs(model._semantic_match_last_metrics["sentinel"], sentinel)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires Gloo")
    def test_ddp_with_alternating_prior_and_checkpointed_rollouts(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(prior_ddp_regression, args=(str(Path(directory) / "store"),), nprocs=2, join=True)


def prior_ddp_regression(rank, store_path):
    dist.init_process_group(
        "gloo", init_method=Path(store_path).as_uri(), rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        MorphFlowPriorIntegrationTests.setUpClass()
        student = MorphFlowPriorIntegrationTests.model_type().train()
        model = DistributedDataParallel(student, find_unused_parameters=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        coords = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0]])
        for iteration in range(4):
            # Repeat graph changes over consecutive DDP forwards, including
            # opposite activation on the ranks to check metric schema safety.
            active = iteration == 0 or (iteration == 2 and rank == 0) or iteration == 3
            optimizer.zero_grad()
            loss = model(
                torch.randn(2, 1, 2, 2, 2), torch.randn(2, 2), coords,
                torch.randn(2, 2), coords, torch.tensor([0.25, 0.75]),
                trellis_prior=lambda sample: (sample.square().mean(), {}),
                trellis_prior_weight=0.1 if active else 0.0,
                trellis_prior_rollout_steps=3, trellis_prior_grad_steps=1,
            )
            loss.backward()
            for parameter in student.parameters():
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
            metrics = torch.stack([student.last_forward_metrics[key]
                                   for key in sorted(student.last_forward_metrics)])
            dist.all_reduce(metrics)
            optimizer.step()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
