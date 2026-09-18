"""Regression checks for per-rank stochastic semantic loss and DDP logging.

Run in an environment with PyTorch (CPU/Gloo is sufficient):
    python -m unittest discover -s tests -p 'test_training_metrics.py' -v
"""

from datetime import timedelta
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from models.semantic_token_matching import SEMANTIC_METRIC_NAMES, SemanticTokenMatchingMixin
from modules.training_metrics import collect_reduced_forward_metrics


class SemanticTrainingProbe(SemanticTokenMatchingMixin, torch.nn.Module):
    """Exercise the real matcher, stochastic gate, auxiliary loss and gradients."""

    def __init__(self, log_stats=True, enabled=True, usage_weight=0.0, usage_cap=4.0, cycle_weight=0.01):
        super().__init__()
        self._init_semantic_token_matching(
            use_semantic_token_matching=enabled,
            semantic_match_dim=8,
            semantic_cycle_loss_weight=cycle_weight if enabled else 0.0,
            semantic_cycle_loss_prob=0.25,
            semantic_match_log_stats=log_stats,
            semantic_usage_loss_weight=usage_weight,
            semantic_usage_cap=usage_cap,
        )

    def forward(self, src1, src2, alpha, draw):
        # Force opposite outcomes on the two ranks without changing the 0.25
        # loss probability or synchronizing its random activation in production.
        with patch("models.semantic_token_matching.torch.rand", return_value=torch.tensor(draw)):
            self._begin_semantic_match_record(src1.device)
        out1, out2 = self._apply_semantic_token_matching(src1, src2, alpha)
        base_loss = out1.square().mean() + out2.square().mean()
        auxiliary = self._semantic_match_aux_loss(src1.device)
        loss = base_loss + auxiliary
        self.last_forward_metrics = self._semantic_match_metrics()
        self.last_forward_metrics.update({
            "base_mse": base_loss.detach(),
            "semantic_aux_loss_weighted": auxiliary.detach(),
            "total_loss": loss.detach(),
        })
        return loss


class CPUAccelerator:
    """The unwrap/reduce interface used by the helper, with real Gloo collectives."""

    device = torch.device("cpu")

    def __init__(self, distributed=False):
        self.distributed = distributed
        self.calls = []

    @staticmethod
    def unwrap_model(model):
        return model.module if isinstance(model, DistributedDataParallel) else model

    def reduce(self, values, reduction):
        assert reduction == "mean"
        assert not values.requires_grad
        self.calls.append(tuple(values.shape))
        result = values.clone()
        if self.distributed:
            dist.all_reduce(result)
            result /= dist.get_world_size()
        return result


def distributed_regression(rank, store_path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(store_path).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=20))
    try:
        accelerator = CPUAccelerator(distributed=True)
        for log_stats, usage_weight in ((True, 0.0), (False, 0.0), (True, 0.01), (False, 0.01)):
            torch.manual_seed(100 + rank)
            model = DistributedDataParallel(SemanticTrainingProbe(log_stats=log_stats, usage_weight=usage_weight, usage_cap=1.0),
                                           find_unused_parameters=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            src1, src2 = torch.randn(2, 4, 128), torch.randn(2, 5, 128)
            alpha = torch.tensor([0.25, 0.75])
            # Exercise active/inactive, inactive/active, both inactive, both active.
            for step in range(4):
                active = (rank == step) if step < 2 else (step == 3)
                optimizer.zero_grad(set_to_none=True)
                loss = model(src1, src2, alpha, 0.1 if active else 0.9)
                loss.backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise AssertionError("Missing or non-finite gradients")
                optimizer.step()
                metrics = model.module.last_forward_metrics
                # Deliberately scramble insertion order on one rank.
                if rank == 1:
                    model.module.last_forward_metrics = dict(reversed(list(metrics.items())))

                local = {key: float(value) for key, value in metrics.items()}
                gathered = [None, None]
                dist.all_gather_object(gathered, local)
                if set(gathered[0]) != set(gathered[1]):
                    raise AssertionError("Rank-dependent metric keys would desynchronize NCCL")
                previous_calls = len(accelerator.calls)
                reduced = collect_reduced_forward_metrics(accelerator, model)
                if len(accelerator.calls) != previous_calls + 1:
                    raise AssertionError("Expected exactly one collective for logging")
                for key, actual in reduced.items():
                    expected = (gathered[0][key] + gathered[1][key]) / 2
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(reduced["semantic_cycle_active"],
                                           0.5 if step < 2 else float(step == 3))
                torch.testing.assert_close(reduced["semantic_usage_active"], float(usage_weight > 0))
                torch.testing.assert_close(reduced["semantic_aux_loss_weighted"],
                                           reduced["semantic_cycle_loss_weighted"] + reduced["semantic_usage_loss_weighted"])
            # Check that DDP parameters remain synchronized after all steps.
            weights = torch.cat([p.detach().flatten() for p in model.parameters()])
            copies = [torch.empty_like(weights), torch.empty_like(weights)]
            dist.all_gather(copies, weights)
            torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
            del model, optimizer
    finally:
        dist.destroy_process_group()


class TrainingMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_schema_and_zero_values_are_stable_across_activation_and_logging_modes(self):
        src1, src2 = torch.randn(2, 4, 128), torch.randn(2, 5, 128)
        alpha = torch.tensor([0.25, 0.75])
        for log_stats in (True, False):
            with self.subTest(log_stats=log_stats):
                model = SemanticTrainingProbe(log_stats=log_stats)
                for active in (True, False, True):
                    model(src1, src2, alpha, 0.1 if active else 0.9)
                    metrics = model._semantic_match_metrics()
                    self.assertEqual(tuple(metrics), SEMANTIC_METRIC_NAMES)
                    self.assertTrue(all(v.ndim == 0 and not v.requires_grad for v in metrics.values()))
                    self.assertEqual(float(metrics["semantic_cycle_active"]), float(active))
                    if active:
                        self.assertGreater(float(metrics["semantic_cycle_loss"]), 0)
                        torch.testing.assert_close(metrics["semantic_cycle_loss_weighted"],
                                                   metrics["semantic_cycle_loss"] * 0.01)
                    else:
                        self.assertEqual(float(metrics["semantic_cycle_loss"]), 0)
                        self.assertEqual(float(metrics["semantic_cycle_loss_weighted"]), 0)
                        if not log_stats:
                            self.assertTrue(all(float(value) == 0 for value in metrics.values()))
                    recorded = model.last_forward_metrics
                    torch.testing.assert_close(recorded["total_loss"],
                                               recorded["base_mse"] + recorded["semantic_aux_loss_weighted"])

    def test_disabled_matcher_keeps_empty_semantic_metrics(self):
        self.assertEqual(SemanticTrainingProbe(enabled=False)._semantic_match_metrics(), {})

    def test_packed_reduction_sorts_names_detaches_and_accepts_scalar_values(self):
        accelerator = CPUAccelerator()
        grad_value = torch.tensor(3.0, requires_grad=True)
        model = SimpleNamespace(last_forward_metrics={"z": grad_value, "a": 2.0, "b": torch.tensor([4.0])})
        reduced = collect_reduced_forward_metrics(accelerator, model)
        self.assertEqual(list(reduced), ["a", "b", "z"])
        self.assertEqual(reduced, {"a": 2.0, "b": 4.0, "z": 3.0})
        self.assertEqual(accelerator.calls, [(3,)])
        self.assertTrue(grad_value.requires_grad)
        self.assertIsNone(grad_value.grad)
        for value in (None, {}):
            model.last_forward_metrics = value
            self.assertEqual(collect_reduced_forward_metrics(accelerator, model), {})
        self.assertEqual(accelerator.calls, [(3,)])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo required")
    def test_two_process_ddp_with_opposite_loss_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_regression, args=(str(Path(directory) / "store"),),
                     nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
