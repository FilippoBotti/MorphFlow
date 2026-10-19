"""Targeted CPU checks for the separate 50-state weighted LPIPS protocol."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import eval_perceptual_sequence as base
import eval_textured_benchmark as evaluation
import threeway_textured_benchmark as benchmark


class DenseWeightedTests(unittest.TestCase):
    @staticmethod
    def frames():
        return [base.Frame(i, alpha, Path(f"raw_{i}.pt"), f"middle_{i}")
                for i, alpha in enumerate(benchmark.canonical_alphas(48, "uniform"))]

    def synthetic_metrics(self, root, factor=1.0):
        frames = self.frames()
        alpha_by_key = {frame.key: frame.alpha for frame in frames}
        alpha_by_key.update(src1=1.0, src2=0.0)
        args = SimpleNamespace(num_views=64, lpips_net="vgg", lpips_batch_size=8)

        def distance(_model, pairs, _device, _batch_size):
            # Different camera factors verify equal-view averaging as well as
            # per-view variation; no generated-to-generated pairs are allowed.
            values = []
            for first, second in pairs:
                self.assertIn(second.parent.name, ("src1", "src2"))
                view = int(first.stem)
                values.append(abs(alpha_by_key[first.parent.name] - alpha_by_key[second.parent.name]) * factor * (1 + view / 64))
            return values

        root.mkdir(parents=True, exist_ok=True)
        with mock.patch.object(base, "batched_lpips", side_effect=distance) as lpips, \
                mock.patch.object(base, "plot_weighted_endpoints"), \
                mock.patch.object(base, "image_path", side_effect=lambda directory, key, view: directory / "images" / key / f"{view:03d}.png"):
            result = base.compute_weighted_endpoint_metrics(args, root, frames, model=object(), device="cpu")
        self.assertEqual(len(lpips.call_args.args[1]), 64 * (48 * 2 + 1))
        result.update(render_backend="gaussian", sequence_name=root.name)
        evaluation.write_json(root / "metrics.json", result)
        return result

    def test_50_state_curve_real_source_endpoints_and_distinct_intermediate_full_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.synthetic_metrics(root)
            self.assertTrue(evaluation.completed_weighted_metrics(result))
            self.assertEqual(result["num_generated_frames"], 48)
            self.assertEqual(result["num_sequence_frames"], 50)
            curve = result["weighted_endpoint_curve"]
            self.assertEqual(len(curve), 50)
            self.assertEqual([curve[0]["weighted_lpips_mean"], curve[-1]["weighted_lpips_mean"]], [0, 0])
            self.assertGreater(curve[0]["lpips_src2_mean"], 1)
            self.assertEqual(curve[0]["lpips_src2_mean"], curve[-1]["lpips_src1_mean"])
            alpha = np.asarray(result["alphas"])
            view_factor = np.mean(1 + np.arange(64) / 64)
            expected = 2 * alpha * (1 - alpha) * view_factor
            np.testing.assert_allclose([row["weighted_lpips_mean"] for row in curve], expected)
            agg = result["aggregate_across_views"]
            self.assertAlmostEqual(agg["weighted_endpoint_lpips_mean"]["mean"], expected[1:-1].mean())
            self.assertAlmostEqual(agg["weighted_endpoint_lpips_full_mean"]["mean"], expected.mean())
            self.assertAlmostEqual(agg["weighted_endpoint_lpips_full_auc"]["mean"], np.trapz(expected[::-1], alpha[::-1]))
            self.assertGreater(agg["weighted_endpoint_lpips_full_auc"]["mean"], agg["weighted_endpoint_lpips_auc"]["mean"])
            self.assertEqual(set(agg), set(evaluation.WEIGHTED_METRICS))
            self.assertFalse((root / "transitions.csv").exists())
            self.assertFalse((root / "fid_features.npz").exists())
            with (root / "weighted_endpoint_lpips_per_view.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 50 * 64)
            broken = json.loads(json.dumps(result))
            broken["aggregate_across_views"]["weighted_endpoint_lpips_full_mean"]["mean"] += .1
            self.assertFalse(evaluation.completed_weighted_metrics(broken))

    def test_uniform_50_state_plan_and_resume_reject_other_protocols(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            image.write_bytes(b"fake input")
            dataset = SimpleNamespace(_find_source_image_path=lambda *unused: image)
            args = benchmark.build_parser().parse_args([
                "plan", "--run-root", str(root / "run"), "--data-root", str(root),
                "--source-images-root", str(root), "--num-pairs", "0", "--k", "48",
                "--alpha-schedule", "uniform", "--protocol", "weighted-endpoints-50"])
            with mock.patch("data.morph_dataset.MorphingDistillDataset", return_value=dataset), \
                    mock.patch.object(benchmark, "select_test_pairs", return_value=([(0, {"src_1": "a", "src_2": "b"})], 1)):
                benchmark.command_plan(args)
                benchmark.command_plan(args)
                payload = benchmark.read_json(root / "run" / "plan.json")
                self.assertEqual(payload["metric_mode"], "weighted-endpoints-only")
                self.assertEqual(payload["num_sequence_frames"], 50)
                self.assertEqual(payload["evaluation_protocol"]["perceptual_views"], 64)
                self.assertNotIn("paper_sequence_states", payload["evaluation_protocol"])
                np.testing.assert_allclose(payload["sequence_alphas"], np.linspace(1, 0, 50), atol=1e-10)
                args.alpha_schedule = "beta"
                with self.assertRaisesRegex(ValueError, "requires --k 48 --alpha-schedule uniform"):
                    benchmark.command_plan(args)
                args.alpha_schedule = "uniform"
                args.protocol = "interp3d"
                with self.assertRaisesRegex(ValueError, "different protocol"):
                    benchmark.command_plan(args)

    def test_weighted_merge_equal_pair_weight_without_fid_or_inception(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.synthetic_metrics(root / "pair_0000_a_to_b", 1)
            second = self.synthetic_metrics(root / "pair_0001_c_to_d", 2)
            args = evaluation.build_parser().parse_args([
                "merge", "--output-dir", str(root), "--metric-mode", "weighted-endpoints-only", "--expected-num-pairs", "2"])
            with mock.patch.object(evaluation, "load_inception", side_effect=AssertionError("Unexpected Inception")), \
                    mock.patch.object(evaluation, "frechet_distance", side_effect=AssertionError("Unexpected FID")):
                evaluation.merge_outputs(args)
            merged = evaluation.base.read_json(root / "metrics_batch.json")
            self.assertEqual(merged["global_metrics"], {})
            self.assertEqual(merged["num_sequence_frames"], 50)
            for metric in evaluation.WEIGHTED_METRICS:
                expected = (first["aggregate_across_views"][metric]["mean"] + second["aggregate_across_views"][metric]["mean"]) / 2
                self.assertAlmostEqual(merged["aggregate_across_pairs"][metric]["mean"], expected)
            args.expected_num_pairs = 3
            with self.assertRaisesRegex(RuntimeError, "Expected 3"):
                evaluation.merge_outputs(args)

    def test_aggregate_dense_protocol_has_only_four_weighted_headline_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.synthetic_metrics(root / "pair_0000_a_to_b")
            batch = {
                "metric_mode": "weighted-endpoints-only", "num_generated_frames": 48, "num_sequence_frames": 50,
                "num_pairs": 1, "num_views_per_pair": 64, "lpips_backbone": "vgg", "alphas": result["alphas"],
                "aggregate_across_pairs": result["aggregate_across_views"],
            }
            benchmark.write_json(root / "plan.json", {
                "seed": 42, "num_pairs": 1, "k_intermediate": 48, "alphas": result["intermediate_alphas"],
                "protocol": "weighted-endpoints-50", "metric_mode": "weighted-endpoints-only", "alpha_schedule": "uniform",
            })
            for method in benchmark.METHODS:
                benchmark.write_json(root / "metrics" / method / "metrics_batch.json", batch)
            with mock.patch.object(benchmark, "aggregate_generation_timings", return_value={}):
                benchmark.command_aggregate(SimpleNamespace(run_root=str(root)))
            final = benchmark.read_json(root / "final_metrics.json")
            self.assertEqual(final["metric_mode"], "weighted-endpoints-only")
            self.assertEqual(final["sequence_alphas"], result["alphas"])
            self.assertEqual(final["num_sequence_frames"], 50)
            with (root / "final_metrics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            weighted = [row for row in rows if row["metric"].startswith("weighted_")]
            self.assertEqual(len(weighted), 3 * 4)
            self.assertEqual({row["metric"] for row in weighted}, set(evaluation.WEIGHTED_METRICS))
            self.assertFalse(any(row["metric"].startswith(("fid", "ppl", "lpips_adjacent")) for row in rows))

    def test_metric_mode_changes_evaluation_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"{index}.pt" for index in range(3)]
            for path in paths:
                path.write_bytes(b"stable raw asset")
            args = evaluation.build_parser().parse_args(["shard", "--run-dir", str(root), "--output-dir", str(root / "metrics")])
            frames = [base.Frame(0, .5, paths[2], "middle")]
            original = evaluation.evaluation_signature(args, paths[0], paths[1], frames)
            args.metric_mode = "weighted-endpoints-only"
            self.assertNotEqual(original, evaluation.evaluation_signature(args, paths[0], paths[1], frames))

    def test_shard_skips_inception_full_path_and_fid_and_resumes_without_feature_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "method" / "pair_0000_a_to_b"
            sequence.mkdir(parents=True)
            output = root / "metrics"
            args = evaluation.build_parser().parse_args([
                "shard", "--run-dir", str(sequence.parent), "--output-dir", str(output),
                "--metric-mode", "weighted-endpoints-only", "--device", "cpu"])
            fake_lpips = SimpleNamespace(LPIPS=lambda **unused: SimpleNamespace(to=lambda _device: SimpleNamespace(eval=lambda: object())))
            frames = self.frames()

            original = base.compute_weighted_endpoint_metrics
            with mock.patch.dict(sys.modules, {"lpips": fake_lpips}), \
                    mock.patch.object(evaluation, "select_sequence_dirs", return_value=[sequence]), \
                    mock.patch.object(evaluation, "discover_raw_frames", return_value=(root / "src1_raw.pt", root / "src2_raw.pt", frames, {"plan": {}})), \
                    mock.patch.object(evaluation, "evaluation_signature", return_value="dense-signature"), \
                    mock.patch.object(evaluation, "prepare_gaussian_images"), \
                    mock.patch.object(base, "compute_weighted_endpoint_metrics",
                                      side_effect=lambda _args, destination, _frames, **unused: self._synthetic_with_compute(destination, original)) as calculate, \
                    mock.patch.object(evaluation, "load_inception", side_effect=AssertionError("Unexpected Inception")), \
                    mock.patch.object(evaluation, "full_path_metrics", side_effect=AssertionError("Unexpected PPL")), \
                    mock.patch.object(evaluation, "save_fid_features", side_effect=AssertionError("Unexpected FID")), \
                    mock.patch("torch.cuda.is_available", return_value=False):
                evaluation.evaluate_shard(args)
                evaluation.evaluate_shard(args)
                self.assertEqual(calculate.call_count, 1)
            self.assertFalse((output / sequence.name / "fid_features.npz").exists())

    def _synthetic_with_compute(self, destination, original):
        # Restore just this function while using the common synthetic fixture.
        with mock.patch.object(base, "compute_weighted_endpoint_metrics", original):
            return self.synthetic_metrics(destination)


if __name__ == "__main__":
    unittest.main()
