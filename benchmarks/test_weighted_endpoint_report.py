"""Focused cached-data reporting checks; no Torch/model/renders/FID imports."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weighted_endpoint_report as report


ALPHAS = [.75, .625, .5, .375, .25]
PAIRS = ("pair_0000_a_to_b", "pair_0001_c_to_d")


def fixture(root, cells=("cfg_ss_2_slat_1.5", "cfg_ss_3_slat_2"), alphas=ALPHAS, include_endpoints=False):
    for cell in cells:
        methods = {}
        for method in report.METHODS:
            methods[method] = {"num_pairs": 2, "num_views_per_pair": 64,
                               "pairs": [{"sequence": pair} for pair in PAIRS]}
            for pair, scale in zip(PAIRS, (1., 3.)):
                directory = (root / cell / "metrics" / method / pair)
                if method != "morphflow":
                    target = root / "shared" / "metrics" / method / pair
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    directory.symlink_to(target)
                    directory = target
                directory.mkdir(parents=True, exist_ok=True)
                rows = []
                sequence_alphas = [1.0, *alphas, 0.0] if include_endpoints else alphas
                for alpha in sequence_alphas:
                    d1, d2 = (1-alpha)*scale, alpha*scale
                    rows.append({"alpha": alpha, "num_views": 64,
                                 "lpips_src1_mean": d1, "lpips_src2_mean": d2,
                                 "src1_weighted_mean": alpha*d1,
                                 "src2_weighted_mean": (1-alpha)*d2,
                                 "weighted_lpips_mean": alpha*d1+(1-alpha)*d2,
                                 "weighted_lpips_std_views": .123})
                report.write_csv(directory / "weighted_endpoint_lpips.csv", rows)
                report.write_json(directory / "metrics.json", {"num_views": 64, "weighted_endpoint_curve": rows})
        report.write_json(root / cell / "final_metrics.json", {
            "num_pairs": 2, "k_intermediate": len(alphas), "alphas": alphas, "methods": methods,
            "metric_mode": "weighted-endpoints-only" if include_endpoints else "full"})


class WeightedEndpointReportTests(unittest.TestCase):
    def test_real_plot_equal_pair_statistics_auc_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            output = report.report(root)
            overview = report.read_json(output / "summary.json")
            self.assertEqual(len(overview["summaries"]), 4)  # 2 CFG + 2 baselines
            self.assertEqual(len(overview["curves"]), 20)
            first = overview["summaries"][0]
            self.assertEqual(first["num_total_frames"], 7)
            self.assertEqual(overview["trajectory"]["num_intermediates"], 5)
            self.assertAlmostEqual(first["weighted_endpoint_lpips_mean"], .875)
            self.assertAlmostEqual(first["weighted_endpoint_lpips_mean_std_pairs"], .4375)
            self.assertAlmostEqual(first["weighted_endpoint_lpips_auc"], .453125)
            center = overview["curves"][2]
            self.assertEqual(center["progress"], .5)
            self.assertEqual(center["weighted_lpips_mean"], 1.)
            self.assertEqual(center["weighted_lpips_std_pairs"], .5)
            self.assertEqual([row["progress"] for row in overview["curves"][:5]], sorted(1-a for a in ALPHAS))
            self.assertEqual((output / "evolution.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual((output / "evolution.pdf").read_bytes()[:4], b"%PDF")
            self.assertFalse((root / "cfg_ss_2_slat_1.5" / "weighted_endpoint_lpips" / "pairs").exists())
            # Completed outputs reuse cached JSON; neither curve reads nor plots repeat.
            with mock.patch.object(report, "read_curve", side_effect=AssertionError("Should reuse cache")), \
                    mock.patch.object(report, "plot_curves", side_effect=AssertionError("Should reuse plots")):
                report.report(root)

    def test_dense_fifty_frame_trajectory_keeps_endpoint_and_intermediate_scores_separate(self):
        alphas = [index / 49 for index in range(48, 0, -1)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "weighted_50"
            fixture(root, ("cfg_ss_2_slat_1.5",), alphas=alphas, include_endpoints=True)
            output = report.report(root)
            overview = report.read_json(output / "summary.json")
            self.assertEqual(len(overview["curves"]), 3*50)
            self.assertEqual(overview["trajectory"]["num_total_frames"], 50)
            self.assertEqual(overview["trajectory"]["num_intermediates"], 48)
            self.assertFalse(overview["trajectory"]["endpoints_in_mean"])
            self.assertTrue(overview["trajectory"]["full_mean_includes_endpoints"])
            self.assertEqual(overview["trajectory"]["num_plotted_frames"], 50)
            for summary in overview["summaries"]:
                self.assertEqual(summary["num_intermediates"], 48)
                self.assertEqual(summary["num_total_frames"], 50)
                expected_mean = sum(4*a*(1-a) for a in alphas) / 48
                self.assertAlmostEqual(summary["weighted_endpoint_lpips_mean"], expected_mean)
                self.assertAlmostEqual(summary["weighted_endpoint_lpips_full_mean"], expected_mean*48/50)
            rows = overview["curves"][:50]
            self.assertEqual(rows[0]["progress"], 0)
            self.assertEqual(rows[-1]["progress"], 1)
            self.assertEqual(rows[0]["weighted_lpips_mean"], 0)
            self.assertEqual(rows[-1]["weighted_lpips_mean"], 0)
            self.assertEqual(rows[0]["lpips_src2_mean"], 2)
            self.assertEqual(rows[-1]["lpips_src1_mean"], 2)
            expected_auc = sum((right["progress"]-left["progress"]) *
                               (right["weighted_lpips_mean"]+left["weighted_lpips_mean"])/2
                               for left, right in zip(rows, rows[1:]))
            self.assertAlmostEqual(overview["summaries"][0]["weighted_endpoint_lpips_full_auc"], expected_auc)
            self.assertLess(overview["summaries"][0]["weighted_endpoint_lpips_auc"], expected_auc)
            self.assertEqual((output / "evolution.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual((output / "evolution.pdf").read_bytes()[:4], b"%PDF")
            cell_output = root / "cfg_ss_2_slat_1.5" / "weighted_endpoint_lpips"
            with (cell_output / "pairs.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3*2*50)

    def test_dense_curve_requires_true_reference_endpoint_and_complete_fifty_frames(self):
        alphas = [index / 49 for index in range(48, 0, -1)]
        rows = []
        for alpha in [1., *alphas, 0.]:
            rows.append({"alpha": alpha, "num_views": 64,
                         "lpips_src1_mean": 1-alpha, "lpips_src2_mean": alpha,
                         "src1_weighted_mean": alpha*(1-alpha),
                         "src2_weighted_mean": (1-alpha)*alpha,
                         "weighted_lpips_mean": 2*alpha*(1-alpha)})
        self.assertEqual(len(report.validate_curve(rows, alphas, "dense", True)), 50)
        with self.assertRaisesRegex(ValueError, "alpha points"):
            report.validate_curve(rows[1:-1], alphas, "dense", True)
        wrong_source = [dict(row) for row in rows]
        wrong_source[0].update(lpips_src1_mean=.1, src1_weighted_mean=.1, weighted_lpips_mean=.1)
        with self.assertRaisesRegex(ValueError, "Canonical endpoint"):
            report.validate_curve(wrong_source, alphas, "dense", True)

    def test_json_fallback_and_optional_pair_comparison_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, ("cfg_ss_2_slat_1.5",))
            curve = root / "cfg_ss_2_slat_1.5" / "metrics" / "morphflow" / PAIRS[0] / "weighted_endpoint_lpips.csv"
            curve.unlink()
            with mock.patch.object(report, "plot_curves", wraps=report.plot_curves) as plots:
                report.report(root, per_pair_plots=True)
            self.assertEqual(plots.call_count, 4)  # cell + 2 individual pairs + global
            image = root / "cfg_ss_2_slat_1.5" / "weighted_endpoint_lpips" / "pairs" / (PAIRS[0]+".png")
            self.assertGreater(image.stat().st_size, 1000)

    def test_reject_pair_identity_mismatch_and_partial_cell_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            partial = root / "cfg_ss_5_slat_2.5"
            partial.mkdir()
            path = root / "cfg_ss_3_slat_2" / "final_metrics.json"
            data = report.read_json(path)
            data["methods"]["interp3d"]["pairs"][0]["sequence"] = "pair_9999_other_to_other"
            report.write_json(path, data)
            with self.assertRaisesRegex(ValueError, "pair identities"):
                report.report(root)

    def test_reject_view_count_alpha_weight_arithmetic_and_nonfinite(self):
        rows = [{"alpha": alpha, "num_views": 64, "lpips_src1_mean": 1.,
                 "lpips_src2_mean": 1., "src1_weighted_mean": alpha,
                 "src2_weighted_mean": 1-alpha, "weighted_lpips_mean": 1.}
                for alpha in ALPHAS]
        self.assertEqual(len(report.validate_curve(rows, ALPHAS, "fixture")), 5)
        variants = (("num_views", 63, "64 views"), ("alpha", .9, "arithmetic"),
                    ("weighted_lpips_mean", 8, "arithmetic"),
                    ("lpips_src1_mean", float("nan"), "Non-finite"))
        for key, value, message in variants:
            with self.subTest(key=key):
                changed = [dict(row) for row in rows]
                changed[0][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    report.validate_curve(changed, ALPHAS, "fixture")
        with self.assertRaisesRegex(ValueError, "alpha points"):
            report.validate_curve(rows, [.8, .625, .5, .375, .25], "fixture")

    def test_reject_changed_baseline_between_cfg_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            source = root / "cfg_ss_3_slat_2" / "metrics" / "interp3d" / PAIRS[0]
            source.unlink()
            source.mkdir()
            rows = []
            for alpha in ALPHAS:
                rows.append({"alpha": alpha, "num_views": 64, "lpips_src1_mean": 2.,
                             "lpips_src2_mean": 2., "src1_weighted_mean": 2*alpha,
                             "src2_weighted_mean": 2*(1-alpha), "weighted_lpips_mean": 2.})
            report.write_csv(source / "weighted_endpoint_lpips.csv", rows)
            with mock.patch.object(report, "plot_curves"):
                with self.assertRaisesRegex(ValueError, "Different interp3d baseline"):
                    report.report(root)


if __name__ == "__main__":
    unittest.main()
