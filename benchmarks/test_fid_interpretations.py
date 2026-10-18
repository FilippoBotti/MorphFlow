"""CPU checks for the two scientifically distinct per-trajectory FID definitions."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import eval_textured_benchmark as evaluation


class DualFIDTests(unittest.TestCase):
    def features(self, shift=3.0):
        cloud = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        states = [cloud + [-shift, 0.], cloud + [shift, 0.]]
        features = {
            "reference": np.concatenate([cloud, cloud]),
            "generated": np.concatenate(states),
            "center_indices": np.array([0, 3]),
            "center_alphas": np.array([0.7, 0.4]),
            "generated_state_indices": np.repeat([0, 3], 4),
        }
        protocol = {"views_per_state": 4, "selected_indices": [0, 3], "selected_alphas": [0.7, 0.4]}
        return features, protocol, states

    def test_mixture_fid_is_not_mean_of_separate_state_fids(self):
        features, protocol, states = self.features()
        metrics, detail = evaluation.compute_fid_interpretations(features, protocol)
        concat = metrics["fid_endpoint_manifold"]["mean"]
        mean_state = metrics["fid_endpoint_manifold_mean_per_state"]["mean"]
        self.assertAlmostEqual(concat, evaluation.frechet_distance(features["reference"], np.concatenate(states)))
        self.assertAlmostEqual(mean_state, np.mean([evaluation.frechet_distance(features["reference"], state) for state in states]))
        self.assertGreater(mean_state, concat + 1.0)
        self.assertEqual([row["frame_index"] for row in detail["selected_states"]], [0, 3])
        self.assertEqual([row["generated_samples"] for row in detail["selected_states"]], [4, 4])
        self.assertIn("mean_pairs mean_selected_states", detail["formulas"]["fid_endpoint_manifold_mean_per_state"])

    def test_identical_states_make_both_interpretations_zero(self):
        # All samples have exactly zero covariance, so differing sample counts
        # cannot introduce Bessel-correction differences in the covariance.
        features, protocol, _ = self.features()
        features["reference"][:] = [2., 3.]
        features["generated"][:] = [2., 3.]
        metrics, _ = evaluation.compute_fid_interpretations(features, protocol)
        for metric in metrics.values():
            self.assertAlmostEqual(metric["mean"], 0.0)

    def test_explicit_labels_override_backend_and_row_order(self):
        features, protocol, states = self.features()
        order = [7, 0, 5, 2, 1, 6, 3, 4]
        features["generated"] = features["generated"][order]
        features["generated_state_indices"] = features["generated_state_indices"][order]
        actual = evaluation.split_generated_state_features(features, protocol, "blender")
        for (_, _, rows), expected in zip(actual, states):
            self.assertEqual(sorted(map(tuple, rows)), sorted(map(tuple, expected)))

    def test_legacy_gaussian_contiguous_and_blender_interleaved_layouts(self):
        features, protocol, states = self.features()
        del features["generated_state_indices"]
        for (_, _, rows), expected in zip(evaluation.split_generated_state_features(features, protocol, "gaussian"), states):
            np.testing.assert_array_equal(rows, expected)
        features["generated"] = np.stack(states, axis=1).reshape(-1, 2)
        for (_, _, rows), expected in zip(evaluation.split_generated_state_features(features, protocol, "blender"), states):
            np.testing.assert_array_equal(rows, expected)

    def test_invalid_splits_are_rejected(self):
        changes = [
            {"center_indices": np.array([0, 0])},
            {"center_indices": np.array([0., 3.])},
            {"center_alphas": np.array([0.7])},
            {"center_alphas": np.array([0.7, np.nan])},
            {"generated_state_indices": np.array([0, 0, 0, 0, 0, 3, 3, 3])},
            {"generated_state_indices": np.array([0, 0, 0, 0, 3, 3, 3, 9])},
            {"reference": np.zeros((7, 2))},
            {"generated": np.zeros((8, 3))},
            {"generated": np.full((8, 2), np.nan)},
        ]
        for change in changes:
            with self.subTest(change=list(change)):
                features, protocol, _ = self.features()
                features.update(change)
                with self.assertRaises(ValueError):
                    evaluation.compute_fid_interpretations(features, protocol)

    def test_aggregation_options_and_old_result_resume_validation(self):
        features, protocol, _ = self.features()
        for mode, expected in [
                ("concatenated", {"fid_endpoint_manifold"}),
                ("mean-per-state", {"fid_endpoint_manifold_mean_per_state"}),
                ("both", {"fid_endpoint_manifold", "fid_endpoint_manifold_mean_per_state"})]:
            metrics, details = evaluation.compute_fid_interpretations(features, protocol, mode)
            self.assertEqual(set(metrics), expected)
            result = {"aggregate_across_views": metrics, "fid_protocol": details}
            self.assertTrue(evaluation.completed_fid_metrics(result, mode))
        old = {"aggregate_across_views": {"fid_endpoint_manifold": {"mean": 5.0}}}
        self.assertTrue(evaluation.completed_fid_metrics(old, "concatenated"))
        self.assertFalse(evaluation.completed_fid_metrics(old, "both"))
        self.assertFalse(evaluation.completed_fid_metrics(old, "mean-per-state"))

    def test_merge_retains_each_definition_per_pair_and_equal_pair_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {metric: [] for metric in evaluation.FID_METRICS.values()}
            for pair, shift in enumerate([1.0, 3.0]):
                output = root / f"pair_{pair:04d}_a_to_b"
                output.mkdir()
                features, protocol, _ = self.features(shift)
                metrics, details = evaluation.compute_fid_interpretations(features, protocol)
                for metric in expected:
                    expected[metric].append(metrics[metric]["mean"])
                result = {"num_generated_frames": 5, "num_views": 64, "lpips_backbone": "vgg",
                          "fid_protocol": {**protocol, **details}, "render_backend": "gaussian",
                          "aggregate_across_views": metrics}
                (output / "metrics.json").write_text(json.dumps(result))
                np.savez(output / "fid_features.npz", **features)
            args = argparse.Namespace(output_dir=str(root), pair_glob="pair_*", expected_num_pairs=2,
                                      allow_incomplete=False, fid_min_samples=2, fid_aggregation="both")
            evaluation.merge_outputs(args)
            batch = json.loads((root / "metrics_batch.json").read_text())
            for metric, values in expected.items():
                self.assertAlmostEqual(batch["aggregate_across_pairs"][metric]["mean"], np.mean(values))
                self.assertAlmostEqual(batch["global_metrics"][metric]["value"], np.mean(values))
                self.assertEqual(batch["global_metrics"][metric]["formula"], evaluation.FID_FORMULAS[metric])
                self.assertIn(metric, batch["pairs"][0])
            self.assertIn("fid_endpoint_manifold_pooled", batch["global_metrics"])
            self.assertEqual(len(batch["per_pair_fid_protocol"]), 2)
            self.assertEqual(len(next(iter(batch["per_pair_fid_protocol"].values()))["selected_states"]), 2)
            args.expected_num_pairs = 3
            with self.assertRaisesRegex(RuntimeError, "Expected 3"):
                evaluation.merge_outputs(args)

    def test_cli_defaults_to_both_and_signature_tracks_aggregation(self):
        parser = evaluation.build_parser()
        self.assertEqual(parser.parse_args(["shard", "--run-dir", "/unused", "--output-dir", "/unused"]).fid_aggregation, "both")
        self.assertEqual(parser.parse_args(["merge", "--output-dir", "/unused"]).fid_aggregation, "both")
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "asset.pt"
            asset.write_bytes(b"feature signature fixture")
            args = SimpleNamespace(num_views=64, resolution=512, radius=2, fov_degrees=40,
                                   camera_distribution="hammersley", lpips_net="vgg", fid_num_views=1000,
                                   fid_num_intermediate_states=2, fid_selection_seed=42,
                                   render_backend="gaussian", fid_aggregation="concatenated")
            frames = [SimpleNamespace(mesh=asset, alpha=0.5)]
            original = evaluation.evaluation_signature(args, asset, asset, frames)
            args.fid_aggregation = "both"
            self.assertNotEqual(original, evaluation.evaluation_signature(args, asset, asset, frames))

    def test_missing_new_fields_cannot_silently_merge_old_outputs_as_both(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "pair_0000_a_to_b"
            output.mkdir()
            features, _, _ = self.features()
            result = {"num_generated_frames": 5, "num_views": 64, "lpips_backbone": "vgg",
                      "fid_protocol": {"views_per_state": 4}, "render_backend": "gaussian",
                      "aggregate_across_views": {"fid_endpoint_manifold": {"mean": 2.0}}}
            (output / "metrics.json").write_text(json.dumps(result))
            np.savez(output / "fid_features.npz", **features)
            args = argparse.Namespace(output_dir=str(root), pair_glob="pair_*", expected_num_pairs=1,
                                      allow_incomplete=False, fid_min_samples=2, fid_aggregation="both")
            with self.assertRaisesRegex(RuntimeError, "rerun shard"):
                evaluation.merge_outputs(args)
            args.fid_aggregation = "concatenated"
            with mock.patch.object(evaluation, "frechet_distance", return_value=99.):
                evaluation.merge_outputs(args)
            batch = json.loads((root / "metrics_batch.json").read_text())
            self.assertEqual(batch["global_metrics"]["fid_endpoint_manifold"]["value"], 2.0)
            self.assertNotIn("fid_endpoint_manifold_mean_per_state", batch["global_metrics"])


if __name__ == "__main__":
    unittest.main()
