#!/usr/bin/env python3
"""Score archived endpoint-distance curves. Standard library only; no inference.

Example:
  python3 benchmarks/source_distance_profile_report.py --run-root /path/to/run

Let t=1-alpha and D=LPIPS(src1,src2). The ideal normalized distances are
(t, 1-t). profile_rmse is the root of the integrated mean squared error of
the two observed normalized distances, on the observed intermediate interval.
The separate contrast and distance-sum errors expose how this score is formed.
profile_auc integrates instantaneous profile error magnitude; progress_abs_auc
integrates absolute contrast error. Their _normalized variants divide by the
observed interval length. Integrate per pair before averaging across pairs.
wrong_source_fraction counts which side of the transition prefers the wrong
source, without assuming linear distance magnitudes (midpoint excluded).

This proposed diagnostic assumes linear ideal endpoint distances, not a known
ground-truth morph. It does not measure visual realism or adjacent-frame LPIPS.
Distances have already been averaged across camera views before scoring.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


METHODS = ("morphflow", "morphany3d", "interp3d")
METRICS = ("profile_rmse", "progress_contrast_rmse", "distance_sum_rmse",
           "wrong_source_fraction", "profile_auc", "profile_auc_normalized",
           "progress_abs_auc", "progress_abs_auc_normalized")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No output rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {label}")
    return value


def integral(points, values):
    """Trapezoidal area on the observed interval; no endpoint extrapolation."""
    if len(points) < 2 or len(points) != len(values):
        raise ValueError("Need at least two matching observations")
    widths = [b-a for a, b in zip(points, points[1:])]
    if any(w <= 0 for w in widths):
        raise ValueError("Duplicate or unordered progress observations")
    return sum(w*(a+b)/2 for w, a, b in zip(widths, values, values[1:]))


def average_integral(points, values):
    """Trapezoidal mean on the observed interval; no endpoint extrapolation."""
    return integral(points, values) / (points[-1]-points[0])


def score_curve(rows, source_distance):
    """Return one equally weighted pair score and intermediate-only point scores."""
    if not math.isfinite(source_distance) or source_distance <= 1e-8:
        raise ValueError("Source LPIPS must exceed 1e-8; normalization is undefined otherwise")
    points = []
    seen = set()
    for row in rows:
        alpha = finite(row['alpha'], 'alpha')
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha outside [0, 1]")
        if alpha in seen:
            raise ValueError("Duplicate alpha")
        seen.add(alpha)
        progress = 1-alpha
        if not math.isclose(finite(row['progress'], 'progress'), progress, abs_tol=1e-8):
            raise ValueError("Progress must equal 1-alpha")
        d1, d2 = (finite(row[key], key) for key in ('lpips_src1', 'lpips_src2'))
        if min(d1, d2) < -1e-8:
            raise ValueError("Negative LPIPS distance")
        if alpha in (0, 1):
            # Canonical endpoints are copied inputs, not generated observations.
            continue
        r1, r2 = d1/source_distance, d2/source_distance
        contrast_error = (r1-r2)-(2*progress-1)
        sum_error = r1+r2-1
        squared_error = ((r1-progress)**2+(r2-(1-progress))**2)/2
        if not math.isclose(squared_error, (contrast_error**2+sum_error**2)/4, abs_tol=1e-10):
            raise ValueError("Profile decomposition failed")
        wrong = None
        if not math.isclose(progress, .5, abs_tol=1e-8):
            # At ties neither source is preferred: half an error, not a free win.
            wrong = .5 if math.isclose(d1, d2, abs_tol=1e-10) else float((d1 > d2) != (progress > .5))
        points.append(dict(progress=progress, alpha=alpha, lpips_src1=d1,
                           lpips_src2=d2, source_lpips=source_distance,
                           profile_squared_error=squared_error,
                           profile_error_magnitude=math.sqrt(squared_error),
                           progress_contrast_error=contrast_error,
                           progress_abs_error=abs(contrast_error),
                           distance_sum_error=sum_error,
                           wrong_source=wrong))
    points.sort(key=lambda r: r['progress'])
    t = [r['progress'] for r in points]
    wrong = [r['wrong_source'] for r in points if r['wrong_source'] is not None]
    if not wrong:
        raise ValueError("Need observations on either side of the midpoint")
    scores = {
        'profile_rmse': math.sqrt(average_integral(t, [r['profile_squared_error'] for r in points])),
        'progress_contrast_rmse': math.sqrt(average_integral(t, [r['progress_contrast_error']**2 for r in points])),
        'distance_sum_rmse': math.sqrt(average_integral(t, [r['distance_sum_error']**2 for r in points])),
        'wrong_source_fraction': statistics.mean(wrong),
    }
    for metric, field in [('profile_auc', 'profile_error_magnitude'),
                          ('progress_abs_auc', 'progress_abs_error')]:
        scores[metric] = integral(t, [r[field] for r in points])
        scores[metric + '_normalized'] = scores[metric] / (t[-1]-t[0])
    return scores, points


def bootstrap_difference(first, second, samples, seed):
    if first.keys() != second.keys():
        raise ValueError("Paired comparison requires identical pair identities")
    differences = [first[p]-second[p] for p in sorted(first)]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(differences, k=len(differences)))/len(differences)
                   for _ in range(samples))
    return dict(mean_difference=statistics.mean(differences),
                ci95_low=draws[int(.025*(samples-1))],
                ci95_high=draws[int(.975*(samples-1))],
                wins=sum(d < -1e-12 for d in differences),
                ties=sum(abs(d) <= 1e-12 for d in differences),
                num_pairs=len(differences))


def read_cell(cell, references):
    final_path = cell / 'final_metrics.json'
    pairs_path = cell / 'weighted_endpoint_lpips/pairs.csv'
    final = json.loads(final_path.read_text())
    grouped = defaultdict(list)
    for row in read_csv(pairs_path):
        if row['configuration'] != cell.name or row['method'] not in METHODS:
            raise ValueError(f"Invalid configuration/method in {pairs_path}")
        grouped[row['method'], row['pair']].append(row)
    expected_alphas = sorted(float(a) for a in final['alphas'])
    results, curves = [], []
    methods = final['methods']
    pair_sets = [{p['sequence'] for p in methods[m]['pairs']} for m in METHODS]
    if not all(p == pair_sets[0] and len(p) == int(final['num_pairs']) for p in pair_sets):
        raise ValueError(f"Incomplete pair coverage in {final_path}")
    expected_keys = {(m, p) for m in METHODS for p in pair_sets[0]}
    if grouped.keys() != expected_keys:
        raise ValueError(f"Curve identities differ from final metrics in {cell}")
    dense = final.get('metric_mode') == 'weighted-endpoints-only'
    for method in METHODS:
        pair_meta = {p['sequence']: p for p in methods[method]['pairs']}
        for pair in sorted(pair_sets[0]):
            rows = grouped[method, pair]
            if any(int(r['num_views']) != int(methods[method]['num_views_per_pair']) for r in rows):
                raise ValueError(f"Mismatched camera counts: {cell}/{method}/{pair}")
            actual = sorted(float(r['alpha']) for r in rows)
            expected = sorted([0., 1., *expected_alphas]) if dense else expected_alphas
            if len(actual) != len(expected) or any(not math.isclose(a,b,abs_tol=1e-9) for a,b in zip(actual,expected)):
                raise ValueError(f"Incomplete alpha coverage: {cell}/{method}/{pair}")
            if dense:
                start = next(r for r in rows if float(r['alpha']) == 1)
                end = next(r for r in rows if float(r['alpha']) == 0)
                distance = finite(start['lpips_src2'], 'source LPIPS')
                if not math.isclose(distance, finite(end['lpips_src1'], 'source LPIPS'), rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError("Source distance differs at opposite endpoints")
                if abs(float(start['lpips_src1'])) > 1e-8 or abs(float(end['lpips_src2'])) > 1e-8:
                    raise ValueError("Canonical source endpoint is not zero")
            else:
                distance = finite(pair_meta[pair]['reference_endpoint_lpips'], 'reference endpoint LPIPS')
            if pair in references and not math.isclose(distance, references[pair], rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError(f"Methods/CFG cells use different source distances for {pair}")
            references[pair] = distance
            scores, points = score_curve(rows, distance)
            info = dict(configuration=cell.name, method=method, pair=pair,
                        num_views=int(methods[method]['num_views_per_pair']))
            results.append(dict(**info, num_intermediates=len(points),
                                progress_min=points[0]['progress'], progress_max=points[-1]['progress'],
                                progress_span=points[-1]['progress']-points[0]['progress'],
                                source_lpips=distance, **scores))
            curves.extend(dict(**info, **r) for r in points)
    return results, curves, [final_path, pairs_path]


def analyze(root, protocol, selected, samples, seed):
    cells = sorted(p.parent for p in root.glob('cfg_ss_*/final_metrics.json'))
    if selected:
        missing = set(selected)-{c.name for c in cells}
        if missing:
            raise ValueError(f"Missing completed configurations in {root}: {sorted(missing)}")
        cells = [c for c in cells if c.name in selected]
    if not cells:
        raise ValueError(f"No completed CFG reports in {root}")
    pair_rows, point_rows, files, references = [], [], [], {}
    expected_pairs, expected_schedule = None, None
    for cell in cells:
        pairs, points, inputs = read_cell(cell, references)
        pair_set = {r['pair'] for r in pairs}
        schedule = tuple(sorted({r['progress'] for r in points}))
        if expected_pairs is not None and (pair_set != expected_pairs or schedule != expected_schedule):
            raise ValueError("CFG cells have different pair coverage or alpha domains")
        expected_pairs, expected_schedule = pair_set, schedule
        pair_rows.extend(dict(protocol=protocol, **r) for r in pairs)
        point_rows.extend(dict(protocol=protocol, **r) for r in points)
        files.extend(inputs)
    summary, comparisons = [], []
    for cell in cells:
        current = [r for r in pair_rows if r['configuration'] == cell.name]
        for metric in METRICS:
            maps = {m: {r['pair']: r[metric] for r in current if r['method'] == m} for m in METHODS}
            for method, values in maps.items():
                summary.append(dict(protocol=protocol, configuration=cell.name, method=method,
                                    metric=metric, mean=statistics.mean(values.values()),
                                    std_pairs=statistics.pstdev(values.values()),
                                    median=statistics.median(values.values()), num_pairs=len(values)))
            for method in METHODS[1:]:
                comparisons.append(dict(protocol=protocol, configuration=cell.name,
                                        baseline=method, metric=metric,
                                        **bootstrap_difference(maps['morphflow'], maps[method], samples, seed)))
    # Shared baselines should not change when MorphFlow CFG changes.
    for method in METHODS[1:]:
        for metric in METRICS:
            reference = None
            for cell in cells:
                values = {r['pair']: r[metric] for r in pair_rows if r['configuration']==cell.name and r['method']==method}
                if reference is not None and any(not math.isclose(values[p], reference[p], abs_tol=1e-9) for p in values):
                    raise ValueError(f"Baseline {method} changes across CFG cells")
                reference = values
    return pair_rows, point_rows, summary, comparisons, files


def aggregate_curves(points):
    groups = defaultdict(list)
    for row in points:
        groups[row['protocol'], row['configuration'], row['method'], row['progress']].append(row)
    result = []
    for (protocol, config, method, progress), rows in sorted(groups.items()):
        target = 2*progress-1
        contrast = [r['progress_contrast_error']+target for r in rows]
        result.append(dict(protocol=protocol, configuration=config, method=method,
                           progress=progress, target_contrast=target,
                           actual_contrast_mean=statistics.mean(contrast),
                           actual_contrast_std_pairs=statistics.pstdev(contrast),
                           profile_point_rmse=math.sqrt(statistics.mean(r['profile_squared_error'] for r in rows)),
                           profile_error_magnitude_mean=statistics.mean(r['profile_error_magnitude'] for r in rows),
                           progress_abs_error_mean=statistics.mean(r['progress_abs_error'] for r in rows),
                           distance_sum_error_mean=statistics.mean(r['distance_sum_error'] for r in rows),
                           num_pairs=len(rows)))
    return result


def plot_curves(out, curves, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for protocol in sorted({r['protocol'] for r in summary}):
        candidates = [r for r in summary if r['protocol']==protocol and r['method']=='morphflow' and r['metric']=='profile_rmse']
        cfg = min(candidates, key=lambda r:r['mean'])['configuration']
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        for method in METHODS:
            rows = sorted([r for r in curves if r['protocol']==protocol and r['configuration']==cfg and r['method']==method],
                          key=lambda r:r['progress'])
            x = [r['progress'] for r in rows]
            axes[0].plot(x, [r['actual_contrast_mean'] for r in rows], label=method)
            axes[1].plot(x, [r['profile_point_rmse'] for r in rows], label=method)
        axes[0].plot(x, [2*t-1 for t in x], 'k--', label='assumed linear target')
        axes[0].set_ylabel('(LPIPS to src1 - LPIPS to src2) / source LPIPS')
        axes[1].set_ylabel('Pointwise distance-profile RMSE')
        axes[1].set_xlabel('Progress = 1-alpha')
        for ax in axes:
            ax.grid(alpha=.25)
            ax.legend()
        fig.suptitle(f'{protocol}: {cfg}; best MorphFlow profile RMSE in this grid\n'
                     'View-averaged LPIPS; intermediates only; linear target is an assumption')
        fig.tight_layout(rect=(0, 0, 1, .92))
        for extension in ['png', 'pdf']:
            fig.savefig(out/f'{protocol}_best_profile.{extension}', dpi=180)
        plt.close(fig)

        candidates = [r for r in summary if r['protocol']==protocol and r['method']=='morphflow'
                      and r['metric']=='profile_auc_normalized']
        cfg = min(candidates, key=lambda r:r['mean'])['configuration']
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        for method in METHODS:
            rows = sorted([r for r in curves if r['protocol']==protocol and r['configuration']==cfg
                           and r['method']==method], key=lambda r:r['progress'])
            x = [r['progress'] for r in rows]
            for ax, field in zip(axes, ['profile_error_magnitude_mean', 'progress_abs_error_mean']):
                ax.plot(x, [r[field] for r in rows], label=method)
                ax.fill_between(x, 0, [r[field] for r in rows], alpha=.08)
        axes[0].set_ylabel('Mean instantaneous profile error magnitude')
        axes[1].set_ylabel('Mean absolute progression contrast error')
        axes[1].set_xlabel('Progress = 1-alpha; observed intermediates only')
        for ax in axes:
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.25)
            ax.legend()
        fig.suptitle(f'{protocol}: {cfg}; best MorphFlow normalized profile AUC in this grid\n'
                     'Area = mean pair AUC; area / observed span = normalized AUC')
        fig.tight_layout(rect=(0, 0, 1, .92))
        for extension in ['png', 'pdf']:
            fig.savefig(out/f'{protocol}_best_auc.{extension}', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Default: RUN/source_distance_profile')
    parser.add_argument('--protocol', choices=['both', 'interp3d', 'weighted50'], default='both')
    parser.add_argument('--configuration', action='append', help='CFG directory name; repeat to select several')
    parser.add_argument('--bootstrap-samples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plots', action='store_true', help='Also save comparison PNG/PDF; requires matplotlib')
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error('--bootstrap-samples must be at least 100')
    if args.plots:
        try:
            import matplotlib
        except ImportError:
            parser.error('--plots requires matplotlib; omit --plots for dependency-free CSV/JSON reports')
    root = args.run_root.resolve()
    protocols = [('interp3d', root), ('weighted50', root/'weighted_50')]
    if args.protocol != 'both':
        protocols = [p for p in protocols if p[0] == args.protocol]
    outputs = [[], [], [], []]
    inputs = []
    for name, path in protocols:
        print(f'Analyzing {name}: {path}', flush=True)
        result = analyze(path, name, args.configuration, args.bootstrap_samples, args.seed)
        for target, rows in zip(outputs, result[:4]):
            target.extend(rows)
        inputs.extend(result[4])
    out = (args.output or root/'source_distance_profile').resolve()
    if out == root or any(out == p or p in out.parents for p in inputs):
        parser.error('Output must be a separate report directory')
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(['per_pair', 'per_point', 'summary', 'paired_comparisons'], outputs):
        write_csv(out/f'{name}.csv', rows)
    curves = aggregate_curves(outputs[1])
    write_csv(out/'curves.csv', curves)
    definitions = {
        'metric_status': 'proposed diagnostic, not a standard or ground-truth morph metric',
        'notation': 't=1-alpha; D=LPIPS(src1,src2); r1=LPIPS(frame,src1)/D; r2=LPIPS(frame,src2)/D',
        'profile_rmse': 'sqrt(integral(0.5*((r1-t)^2+(r2-(1-t))^2),dt)/(tmax-tmin))',
        'progress_contrast_rmse': 'sqrt(integral(((r1-r2)-(2*t-1))^2,dt)/(tmax-tmin))',
        'distance_sum_rmse': 'sqrt(integral((r1+r2-1)^2,dt)/(tmax-tmin))',
        'profile_auc': 'integral(sqrt(0.5*((r1-t)^2+(r2-(1-t))^2)),dt)',
        'profile_auc_normalized': 'profile_auc/(tmax-tmin); temporal mean of instantaneous profile error magnitude',
        'progress_abs_auc': 'integral(abs((r1-r2)-(2*t-1)),dt)',
        'progress_abs_auc_normalized': 'progress_abs_auc/(tmax-tmin); temporal mean absolute progression contrast error',
        'decomposition': 'per pair: profile_rmse^2=(progress_contrast_rmse^2+distance_sum_rmse^2)/4',
        'wrong_source_fraction': 'fraction of observed non-midpoint frames closer to the wrong source; ties count 0.5',
        'direction': 'lower is better under the stated linear-distance assumption',
        'aggregation': 'LPIPS is already averaged across views; score each pair first, then equal weight across pairs',
        'integration': 'trapezoidal integration of sampled integrands over observed intermediates only; canonical endpoints excluded. RMSEs and normalized AUCs divide by observed interval length; raw AUCs do not.',
        'auc_curves': {'profile': 'profile_error_magnitude_mean in curves.csv',
                       'progression': 'progress_abs_error_mean in curves.csv',
                       'identity': 'integral of either mean curve equals mean per-pair raw AUC on the shared observed grid',
                       'rmse_curve': 'profile_point_rmse is sqrt(mean pair squared error); its area does not equal profile_auc'},
        'bootstrap': {'samples': args.bootstrap_samples, 'seed': args.seed,
                      'unit': 'paired trajectories', 'difference': 'MorphFlow minus baseline',
                      'interval': '95% percentile; exploratory, no multiple-CFG selection correction'},
        'limitations': ['Ideal endpoint distances are assumed linear in t; LPIPS need not follow that profile for a valid morph.',
                        'The distance sum need not equal D; its signed deviation is not a geometric triangle-inequality proof.',
                        'No realism, semantic correctness or adjacent-frame smoothness is measured.',
                        'No direct comparison of absolute scores across the narrow beta and full uniform alpha domains.',
                        'Tiny source distances <=1e-8 raise an error rather than produce unstable normalized scores.',
                        'Copied canonical endpoints are excluded; they do not demonstrate generated endpoint fidelity.'],
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    }
    (out/'definitions.json').write_text(json.dumps(definitions, indent=2, allow_nan=False)+'\n')
    if args.plots:
        plot_curves(out, curves, outputs[2])
    for name, _ in protocols:
        candidates = [r for r in outputs[2] if r['protocol']==name and r['method']=='morphflow' and r['metric']=='profile_rmse']
        best = min(candidates, key=lambda r:r['mean'])
        print(f"{name}: best MorphFlow {best['configuration']}: profile RMSE={best['mean']:.6f}")
    print(f'Reports: {out}')


if __name__ == '__main__':
    main()
