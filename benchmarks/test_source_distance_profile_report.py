import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name('source_distance_profile_report.py')
spec = importlib.util.spec_from_file_location('profile_report', SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def rows_for(function, ts=(.1, .3, .5, .7, .9), distance=2):
    rows = []
    for t in ts:
        d1, d2 = function(t)
        rows.append(dict(alpha=1-t, progress=t, lpips_src1=d1*distance,
                         lpips_src2=d2*distance))
    return rows


class ProfileTests(unittest.TestCase):
    def test_linear_profile_zero_and_scale_invariance(self):
        for distance in [.1, 1, 100]:
            score, _ = report.score_curve(rows_for(lambda t:(t, 1-t), distance=distance), distance)
            self.assertLess(score['profile_rmse'], 1e-14)
            self.assertEqual(score['wrong_source_fraction'], 0)
        function = lambda t:(.2+.8*t, 1-.8*t)
        a, _ = report.score_curve(rows_for(function), 2)
        b, _ = report.score_curve(rows_for(function, distance=.1), .1)
        for k in a:
            self.assertAlmostEqual(a[k], b[k])

    def test_frozen_hard_switch_and_unrelated_outputs_penalized(self):
        for function in [lambda t:(0, 1), lambda t:(int(t>.5), int(t<=.5)), lambda t:(3,3)]:
            score, _ = report.score_curve(rows_for(function), 2)
            self.assertGreater(score['profile_rmse'], .2)
        score, _ = report.score_curve(rows_for(lambda t:(0,1)), 2)
        self.assertEqual(score['wrong_source_fraction'], .5)
        tied, _ = report.score_curve(rows_for(lambda t:(1,1)), 2)
        self.assertEqual(tied['wrong_source_fraction'], .5)

    def test_nonuniform_integration_is_not_arithmetic_frame_mean(self):
        score, points = report.score_curve(rows_for(lambda t:(0,1), ts=(.1,.2,.9)), 2)
        # Frozen src1 gives instantaneous profile error t^2.
        expected = ((.2-.1)*(.1**2+.2**2)/2 + (.9-.2)*(.2**2+.9**2)/2)/.8
        self.assertAlmostEqual(score['profile_rmse'], math.sqrt(expected))
        self.assertNotAlmostEqual(score['profile_rmse'], math.sqrt(sum(t*t for t in [.1,.2,.9])/3))
        self.assertEqual(len(points), 3)

    def test_auc_frozen_source_analytic_area_and_normalization(self):
        score, _ = report.score_curve(rows_for(lambda t:(0,1), ts=(.1,.2,.9)), 2)
        # Magnitudes are t for the profile and 2t for progression.
        self.assertAlmostEqual(score['profile_auc'], (.9**2-.1**2)/2)
        self.assertAlmostEqual(score['profile_auc_normalized'], .5)
        self.assertAlmostEqual(score['progress_abs_auc'], .8)
        self.assertAlmostEqual(score['progress_abs_auc_normalized'], 1)
        self.assertLess(score['profile_auc_normalized'], score['profile_rmse'])

    def test_absolute_auc_does_not_cancel_opposite_errors(self):
        score, _ = report.score_curve(rows_for(lambda t:(.5,.5), ts=(.1,.5,.9)), 2)
        self.assertAlmostEqual(score['progress_abs_auc'], .32)
        self.assertAlmostEqual(score['progress_abs_auc_normalized'], .4)
        self.assertAlmostEqual(score['profile_auc'], .16)
        # The signed contrast area would be zero for this curve.
        self.assertAlmostEqual(report.integral([.1,.5,.9], [.8,0,-.8]), 0)

    def test_auc_mean_curve_area_equals_mean_pair_area(self):
        points, scores = [], []
        for pair, function in [('a', lambda t:(t+.2,1-t+.2)),
                               ('b', lambda t:(t+.4,1-t+.4))]:
            score, rows = report.score_curve(rows_for(function), 2)
            scores.append(score)
            points.extend(dict(protocol='interp3d', configuration='cfg', method='morphflow',
                               pair=pair, **r) for r in rows)
        curves = report.aggregate_curves(points)
        t = [r['progress'] for r in curves]
        for metric, field in [('profile_auc', 'profile_error_magnitude_mean'),
                              ('progress_abs_auc', 'progress_abs_error_mean')]:
            area = report.integral(t, [r[field] for r in curves])
            self.assertAlmostEqual(area, sum(s[metric] for s in scores)/2)
        # sqrt(mean squared error) is a different curve from mean magnitude.
        self.assertGreater(curves[0]['profile_point_rmse'], curves[0]['profile_error_magnitude_mean'])

    def test_decomposition_and_copied_endpoints_do_not_change_score(self):
        rows = rows_for(lambda t:(.2+.7*t, .9-.6*t))
        a, _ = report.score_curve(rows, 2)
        b, _ = report.score_curve(rows + rows_for(lambda t:(t,1-t), ts=(0,1)), 2)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a['profile_rmse']**2,
                               (a['progress_contrast_rmse']**2+a['distance_sum_rmse']**2)/4)

    def test_source_swap_and_time_reversal_preserve_scores(self):
        rows = rows_for(lambda t:(.2+.7*t, .9-.6*t))
        reversed_rows = [dict(alpha=1-r['alpha'], progress=1-r['progress'],
                              lpips_src1=r['lpips_src2'], lpips_src2=r['lpips_src1']) for r in rows]
        a, _ = report.score_curve(rows, 2)
        b, _ = report.score_curve(reversed_rows, 2)
        for key in a:
            self.assertAlmostEqual(a[key], b[key])

    def test_invalid_inputs_fail(self):
        rows = rows_for(lambda t:(t,1-t))
        for distance in [0,1e-12,float('nan')]:
            with self.assertRaises(ValueError):
                report.score_curve(rows, distance)
        for malformed in [rows+[rows[0]], [dict(rows[0], progress=.8), *rows[1:]],
                          [dict(rows[0], lpips_src1=float('nan')), *rows[1:]]]:
            with self.assertRaises(ValueError):
                report.score_curve(malformed, 2)

    def test_paired_bootstrap_matches_by_identity(self):
        result = report.bootstrap_difference({'a':1,'b':2}, {'b':3,'a':2}, 100, 42)
        self.assertEqual(result['mean_difference'], -1)
        self.assertEqual(result['ci95_low'], -1)
        self.assertEqual(result['ci95_high'], -1)
        self.assertEqual(result['wins'], 2)
        self.assertEqual(result, report.bootstrap_difference({'b':2,'a':1}, {'a':2,'b':3}, 100, 42))
        with self.assertRaises(ValueError):
            report.bootstrap_difference({'a':1}, {'b':1}, 100, 42)


def make_fixture(root, dense=False):
    cell = root/'cfg_ss_3_slat_1'
    curve_path = cell/'weighted_endpoint_lpips/pairs.csv'
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    alphas = [.9,.5,.1]
    final = dict(num_pairs=2, alphas=alphas,
                 metric_mode='weighted-endpoints-only' if dense else 'interp3d', methods={})
    all_rows = []
    for method in report.METHODS:
        final['methods'][method] = dict(num_views_per_pair=64, pairs=[])
        for pair in ['pair_0000_a_to_b','pair_0001_c_to_d']:
            final['methods'][method]['pairs'].append(dict(sequence=pair,reference_endpoint_lpips=2))
            function = (lambda t:(t,1-t)) if method=='morphflow' else (lambda t:(0,1))
            rows = rows_for(function, ts=[1-a for a in alphas])
            if dense:
                rows += rows_for(lambda t:(t,1-t), ts=[0,1])
            for i, row in enumerate(rows):
                all_rows.append(dict(configuration=cell.name, method=method, pair=pair,
                                     step=i+1, num_views=64, **row))
    with curve_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (cell/'final_metrics.json').write_text(json.dumps(final))
    return curve_path


class ReportIntegrationTests(unittest.TestCase):
    def test_cli_both_protocols_from_archived_reports_without_model_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_fixture(root)
            make_fixture(root/'weighted_50', dense=True)
            result = subprocess.run([sys.executable,str(SCRIPT),'--run-root',str(root),
                                     '--bootstrap-samples','100'],stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,universal_newlines=True)
            self.assertEqual(result.returncode,0,result.stderr)
            out=root/'source_distance_profile'
            summary=report.read_csv(out/'summary.csv')
            self.assertEqual(len(summary),2*3*len(report.METRICS))
            self.assertTrue(all(float(r['mean'])<1e-14 for r in summary if r['method']=='morphflow'))
            points=report.read_csv(out/'per_point.csv')
            self.assertTrue(all(0<float(r['alpha'])<1 for r in points))
            self.assertEqual(json.loads((out/'definitions.json').read_text())['bootstrap']['unit'], 'paired trajectories')
            curves=report.read_csv(out/'curves.csv')
            pair_rows=report.read_csv(out/'per_pair.csv')
            for row in pair_rows:
                span=float(row['progress_span'])
                for metric in ['profile_auc', 'progress_abs_auc']:
                    self.assertAlmostEqual(float(row[metric]), span*float(row[metric+'_normalized']))
            ideal=[r for r in curves if r['method']=='morphflow']
            self.assertTrue(all(abs(float(r['actual_contrast_mean'])-float(r['target_contrast']))<1e-12 for r in ideal))

    def test_incomplete_curve_or_different_reference_sources_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=make_fixture(root)
            rows=report.read_csv(path)
            report.write_csv(path, rows[:-1])
            with self.assertRaisesRegex(ValueError,'alpha coverage'):
                report.read_cell(path.parent.parent,{})
            make_fixture(root)
            final_path=path.parent.parent/'final_metrics.json'
            final=json.loads(final_path.read_text())
            final['methods']['interp3d']['pairs'][0]['reference_endpoint_lpips']=3
            final_path.write_text(json.dumps(final))
            with self.assertRaisesRegex(ValueError,'different source distances'):
                report.read_cell(path.parent.parent,{})


if __name__ == '__main__':
    unittest.main()
