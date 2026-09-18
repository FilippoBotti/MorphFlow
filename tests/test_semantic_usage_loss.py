"""Capped hubness-loss mathematics, gradient flow, and checkpoint compatibility."""

import unittest
from unittest.mock import patch

import torch

from models.semantic_token_matching import SEMANTIC_METRIC_NAMES, SemanticTokenMatcher
from test_training_metrics import SemanticTrainingProbe


class SemanticUsageLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_nonuniform_usage_below_cap_and_at_cap_is_not_penalized(self):
        matcher = SemanticTokenMatcher(dim=8, usage_cap=4.0)
        # Uneven usage [2.8, 0.8, 0.4, 0] is allowed, as requested.
        uneven = torch.tensor([[[0.7, 0.2, 0.1, 0.0]]]).expand(1, 4, 4)
        at_cap = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).expand(1, 4, 4)
        self.assertEqual(float(matcher._usage_loss(uneven, at_cap)), 0.0)

    def test_rectangular_attention_normalizes_by_destination_count(self):
        matcher = SemanticTokenMatcher(dim=8, usage_cap=4.0)
        # Three queries collapse on one of eight destinations: usage=8, not 3.
        a12 = torch.zeros(1, 3, 8)
        a12[:, :, 0] = 1.0
        a21 = torch.full((1, 8, 3), 1.0 / 3)
        # 0.5 * ((8-4)^2 / 8 + 0) = 1.
        torch.testing.assert_close(matcher._usage_loss(a12, a21), torch.tensor(1.0))
        torch.testing.assert_close(matcher._usage_loss(a21, a12), torch.tensor(1.0))
        balanced = torch.full((1, 3, 8), 1.0 / 8)
        self.assertEqual(float(matcher._usage_loss(balanced, a21)), 0.0)

    def test_zero_gradient_below_cap_and_positive_penalty_gradient_above_cap(self):
        matcher = SemanticTokenMatcher(dim=8, usage_cap=4.0)
        a12 = torch.tensor([[[0.75] + [0.25 / 7] * 7]], requires_grad=True)
        a21 = torch.full((1, 8, 1), 1.0, requires_grad=True)
        loss = matcher._usage_loss(a12, a21)
        loss.backward()
        self.assertGreater(float(loss), 0)
        self.assertGreater(float(a12.grad[0, 0, 0]), 0)
        self.assertEqual(float(a12.grad[0, 0, 1:].abs().sum()), 0)
        self.assertEqual(float(a21.grad.abs().sum()), 0)

    def test_hubness_gradient_reaches_qk_with_and_without_detached_score_inputs(self):
        for detach_scores in (False, True):
            with self.subTest(detach_scores=detach_scores):
                torch.manual_seed(47)
                matcher = SemanticTokenMatcher(dim=8, match_dim=4, usage_cap=1.0, detach_scores=detach_scores)
                src1 = torch.randn(2, 3, 8, requires_grad=True)
                src2 = torch.randn(2, 5, 8, requires_grad=True)
                _, _, aux = matcher(src1, src2, torch.tensor([0.0, 1.0]), return_aux=True, compute_usage=True)
                # Usage loss has no alpha multiplier: it regularizes correspondence even at endpoints.
                self.assertGreater(float(aux["usage_loss"]), 0)
                aux["usage_loss"].backward()
                for projection in (matcher.q_proj, matcher.k_proj):
                    self.assertIsNotNone(projection.weight.grad)
                    self.assertTrue(torch.isfinite(projection.weight.grad).all())
                    self.assertGreater(float(projection.weight.grad.abs().sum()), 0)
                for source in (src1, src2):
                    if detach_scores:
                        self.assertIsNone(source.grad)
                    else:
                        self.assertGreater(float(source.grad.abs().sum()), 0)

    def test_style_tokens_do_not_change_the_usage_penalty(self):
        torch.manual_seed(47)
        matcher = SemanticTokenMatcher(dim=8, match_dim=4, usage_cap=1.0, style_tokens=1)
        src1, src2 = torch.randn(2, 4, 8), torch.randn(2, 6, 8)
        alpha = torch.tensor([0.5, 0.5])
        _, _, first = matcher(src1, src2, alpha, return_aux=True, compute_usage=True)
        src1[:, -1] = 1e6
        src2[:, -1] = -1e6
        out1, out2, second = matcher(src1, src2, alpha, return_aux=True, compute_usage=True)
        torch.testing.assert_close(first["usage_loss"], second["usage_loss"])
        torch.testing.assert_close(out1[:, -1], src1[:, -1])
        torch.testing.assert_close(out2[:, -1], src2[:, -1])

    def test_usage_only_training_and_validation_with_stats_disabled(self):
        torch.manual_seed(47)
        model = SemanticTrainingProbe(log_stats=False, usage_weight=0.01, usage_cap=1.0, cycle_weight=0.0)
        src1, src2 = torch.randn(2, 4, 128), torch.randn(2, 5, 128)
        alpha = torch.tensor([0.25, 0.75])
        for training in (True, False):
            with self.subTest(training=training):
                model.train(training)
                loss = model(src1, src2, alpha, draw=0.9)
                metrics = model.last_forward_metrics
                self.assertEqual(tuple(model._semantic_match_metrics()), SEMANTIC_METRIC_NAMES)
                self.assertEqual(float(metrics["semantic_cycle_active"]), 0)
                self.assertEqual(float(metrics["semantic_usage_active"]), 1)
                self.assertGreater(float(metrics["semantic_usage_loss"]), 0)
                torch.testing.assert_close(metrics["semantic_usage_loss_weighted"], metrics["semantic_usage_loss"] * 0.01)
                torch.testing.assert_close(loss.detach(), metrics["base_mse"] + metrics["semantic_usage_loss_weighted"])
                self.assertTrue(all(not value.requires_grad for value in metrics.values()))

    def test_disabled_loss_and_sampling_do_not_compute_usage_or_change_state_dict(self):
        torch.manual_seed(47)
        baseline = SemanticTrainingProbe(cycle_weight=0)
        enabled = SemanticTrainingProbe(cycle_weight=0, usage_weight=0.01, usage_cap=1.0)
        enabled.load_state_dict(baseline.state_dict(), strict=True)
        self.assertEqual(list(baseline.state_dict()), list(enabled.state_dict()))
        src1, src2 = torch.randn(2, 4, 128), torch.randn(2, 5, 128)
        alpha = torch.tensor([0.25, 0.75])
        with patch.object(baseline.semantic_matcher, "_usage_loss", side_effect=AssertionError("disabled")):
            loss = baseline(src1, src2, alpha, 0.1)
        torch.testing.assert_close(loss.detach(), baseline.last_forward_metrics["base_mse"])
        # Samplers call forward_flow -> _apply_semantic_token_matching directly.
        with patch.object(enabled.semantic_matcher, "_usage_loss", side_effect=AssertionError("sampling")):
            enabled.eval()
            out_enabled = enabled._apply_semantic_token_matching(src1, src2, alpha)
        out_baseline = baseline._apply_semantic_token_matching(src1, src2, alpha)
        for expected, actual in zip(out_baseline, out_enabled):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_configuration_is_rejected(self):
        for weight in (-0.01, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "usage_loss_weight"):
                SemanticTrainingProbe(usage_weight=weight)
        for cap in (0.0, 0.9, float("nan"), float("inf")):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "usage_cap"):
                SemanticTrainingProbe(usage_cap=cap)
        with self.assertRaisesRegex(ValueError, "requires use_semantic_token_matching"):
            SemanticTrainingProbe(enabled=False, usage_weight=0.01)


if __name__ == "__main__":
    unittest.main()
