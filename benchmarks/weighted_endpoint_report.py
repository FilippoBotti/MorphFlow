#!/usr/bin/env python3
"""Plot cached alpha-weighted endpoint LPIPS; never runs inference or metrics.

Only completed ``cfg_ss_*/final_metrics.json`` cells are considered. Each
trajectory contributes equally after its camera views have already been averaged.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path


METHODS = ("morphflow", "morphany3d", "interp3d")
VALUES = ("lpips_src1", "lpips_src2", "src1_weighted", "src2_weighted", "weighted_lpips")
FORMULA = "alpha * LPIPS(frame, src1) + (1-alpha) * LPIPS(frame, src2)"
NOTE = ("Equal pair weights after averaging 64 identical camera views. Progress=1-alpha "
        "runs from src1 to src2. Only observed intermediate points are plotted; no "
        "artificial endpoints or extrapolation. Bands/std describe population standard "
        "deviation across pairs, not across views or a confidence interval. This metric "
        "measures source proximity, not smoothness or realism.")
SCHEMA = 2


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def same_alphas(left, right):
    return len(left) == len(right) and all(math.isclose(a, b, abs_tol=1e-9, rel_tol=0)
                                         for a, b in zip(left, right))


def trajectory_spec(alphas, include_endpoints=False):
    """Describe what is scored without implying unobserved endpoint LPIPS values."""
    return {"num_total_frames": len(alphas) + 2,
            "num_intermediates": len(alphas), "num_endpoint_sources": 2,
            "num_plotted_frames": len(alphas) + (2 if include_endpoints else 0),
            "scored_frames": "intermediates_and_canonical_reference_endpoints" if include_endpoints else "intermediates_only",
            "endpoints_in_mean": False,
            "intermediate_progress_interval": [1-max(alphas), 1-min(alphas)],
            "auc_domain": "observed_intermediate_interval_without_extrapolation",
            **({"full_mean_includes_endpoints": True, "full_auc_domain": [0.0, 1.0],
                "endpoint_origin": "canonical input source frames, not inferred zero-valued points"}
               if include_endpoints else {})}


def interpretation(include_endpoints):
    if not include_endpoints:
        return NOTE
    return ("Equal pair weights after averaging 64 identical camera views. Progress=1-alpha "
            "runs from src1 to src2. All observed frames are plotted, including the two canonical "
            "input source frames. The original mean/AUC score intermediates only; full_mean/full_auc "
            "also score the two endpoint frames, whose self-source weighted LPIPS is zero. "
            "Bands/std describe population standard deviation across pairs, not across views or "
            "a confidence interval. This metric measures source proximity, not smoothness or realism.")


def validate_curve(rows, expected_alphas, source, include_endpoints=False):
    """Recheck cached arithmetic, camera count and alpha identities without LPIPS."""
    normalized = []
    for row in rows:
        alpha = float(row["alpha"])
        count = float(row["num_views"])
        numbers = {name: float(row[name + "_mean"]) for name in VALUES}
        if not all(math.isfinite(value) for value in (alpha, count, *numbers.values())):
            raise ValueError(f"Non-finite endpoint curve: {source}")
        if count != 64:
            raise ValueError(f"Expected 64 views, found {count}: {source}")
        if not (0 <= alpha <= 1 if include_endpoints else 0 < alpha < 1):
            raise ValueError(f"Expected intermediate alpha strictly between 0 and 1: {source}")
        for actual, expected in (
            (numbers["src1_weighted"], alpha * numbers["lpips_src1"]),
            (numbers["src2_weighted"], (1-alpha) * numbers["lpips_src2"]),
            (numbers["weighted_lpips"], numbers["src1_weighted"] + numbers["src2_weighted"]),
        ):
            if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError(f"Inconsistent alpha-weighted LPIPS arithmetic: {source}")
        normalized.append({"alpha": alpha, "progress": 1-alpha, "num_views": 64, **numbers})
    normalized.sort(key=lambda row: row["progress"])
    complete_alphas = [1.0, *expected_alphas, 0.0] if include_endpoints else expected_alphas
    if not same_alphas([row["alpha"] for row in normalized], complete_alphas):
        raise ValueError(f"Mismatched or duplicate alpha points: {source}")
    if include_endpoints:
        for endpoint in (normalized[0], normalized[-1]):
            self_distance = endpoint["lpips_src1"] if endpoint["alpha"] == 1 else endpoint["lpips_src2"]
            if not math.isclose(self_distance, 0, abs_tol=1e-8):
                raise ValueError(f"Canonical endpoint must match its input source: {source}")
    return normalized


def cell_spec(path):
    final = read_json(path)
    name = path.parent.name
    count = int(final["num_pairs"])
    alphas = sorted((float(value) for value in final["alphas"]), reverse=True)
    if count < 1 or len(alphas) < 2 or len(set(alphas)) != len(alphas):
        raise ValueError(f"Invalid pair count or alpha schedule: {path}")
    if len(alphas) != int(final["k_intermediate"]):
        raise ValueError(f"Intermediate count differs from alpha schedule: {path}")
    include_endpoints = final.get("metric_mode") == "weighted-endpoints-only"
    if include_endpoints and len(alphas) != 48:
        raise ValueError(f"Dense weighted protocol requires 48 intermediate alphas: {path}")
    pair_names = None
    sources = {}
    for method in METHODS:
        meta = final["methods"][method]
        pairs = sorted(row["sequence"] for row in meta["pairs"])
        if len(pairs) != count or len(set(pairs)) != count or int(meta["num_pairs"]) != count:
            raise ValueError(f"Mismatched pair counts in {name}/{method}")
        if int(meta["num_views_per_pair"]) != 64:
            raise ValueError(f"Expected 64 views in {name}/{method}")
        if pair_names is not None and pairs != pair_names:
            raise ValueError(f"Mismatched pair identities in {name}/{method}")
        pair_names = pairs
        sources[method] = {}
        for pair in pairs:
            if Path(pair).name != pair or not pair.startswith("pair_"):
                raise ValueError(f"Invalid pair identity: {pair}")
            directory = path.parent / "metrics" / method / pair
            source = directory / "weighted_endpoint_lpips.csv"
            if not source.is_file():
                source = directory / "metrics.json"
            if not source.is_file():
                raise FileNotFoundError(f"Missing cached weighted endpoint curve: {directory}")
            sources[method][pair] = source.resolve()
    return {"name": name, "directory": path.parent, "alphas": alphas,
            "include_endpoints": include_endpoints,
            "pair_names": pair_names, "sources": sources, "path": path}


def source_signature(spec):
    paths = [spec["path"], *(source for method in METHODS for source in spec["sources"][method].values())]
    inputs = []
    for path in paths:
        info = path.stat()
        inputs.append((str(path), info.st_size, info.st_mtime_ns))
    return digest({"schema": SCHEMA, "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "files": inputs})


def read_curve(path, expected_alphas, cache, include_endpoints=False):
    if path in cache:
        return cache[path]
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        data = read_json(path)
        if int(data["num_views"]) != 64:
            raise ValueError(f"Expected 64 views: {path}")
        rows = data.get("weighted_endpoint_curve")
        if not rows:
            raise ValueError(f"Cached weighted_endpoint_curve absent; refresh endpoint metrics once: {path}")
    result = validate_curve(rows, expected_alphas, path, include_endpoints)
    cache[path] = result
    return result


def aggregate_method(configuration, method, trajectories, include_endpoints=False):
    pairs = sorted(trajectories)
    curves = []
    for step, first in enumerate(trajectories[pairs[0]], 1):
        row = {"configuration": configuration, "method": method, "step": step,
               "alpha": first["alpha"], "progress": first["progress"],
               "num_pairs": len(pairs), "num_views": 64}
        for key in VALUES:
            values = [trajectories[pair][step-1][key] for pair in pairs]
            row[key + "_mean"] = statistics.mean(values)
            row[key + "_std_pairs"] = statistics.pstdev(values)
        curves.append(row)
    statistics_by_metric = {"weighted_endpoint_lpips_mean": [], "weighted_endpoint_lpips_auc": []}
    if include_endpoints:
        statistics_by_metric.update(weighted_endpoint_lpips_full_mean=[], weighted_endpoint_lpips_full_auc=[])
    for pair in pairs:
        trajectory = trajectories[pair]
        intermediate = [row for row in trajectory if 0 < row["alpha"] < 1]
        def area(rows):
            return sum((right["progress"]-left["progress"]) *
                       (right["weighted_lpips"]+left["weighted_lpips"])/2
                       for left, right in zip(rows, rows[1:]))
        statistics_by_metric["weighted_endpoint_lpips_mean"].append(statistics.mean(row["weighted_lpips"] for row in intermediate))
        statistics_by_metric["weighted_endpoint_lpips_auc"].append(area(intermediate))
        if include_endpoints:
            statistics_by_metric["weighted_endpoint_lpips_full_mean"].append(statistics.mean(row["weighted_lpips"] for row in trajectory))
            statistics_by_metric["weighted_endpoint_lpips_full_auc"].append(area(trajectory))
    intermediate_curves = [row for row in curves if 0 < row["alpha"] < 1]
    summary = {"configuration": configuration, "method": method, "num_pairs": len(pairs),
               "num_views": 64, "num_intermediates": len(intermediate_curves),
               "num_total_frames": len(intermediate_curves) + 2,
               "alpha_min": intermediate_curves[-1]["alpha"], "alpha_max": intermediate_curves[0]["alpha"]}
    for name, values in statistics_by_metric.items():
        summary[name] = statistics.mean(values)
        summary[name + "_std_pairs"] = statistics.pstdev(values)
        summary[name + "_median_pairs"] = statistics.median(values)
    return curves, summary


def plot_curves(curves, output, title, overview=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 10 if overview else 9), sharex=True,
                             constrained_layout=True)
    groups = {}
    for row in curves:
        groups.setdefault((row["configuration"], row["method"]), []).append(row)
    palette = plt.get_cmap("tab20")
    for index, ((configuration, method), rows) in enumerate(groups.items()):
        rows = sorted(rows, key=lambda row: row["progress"])
        progress = [row["progress"] for row in rows]
        color = palette(index % 20)
        label = (f"MorphFlow {configuration.removeprefix('cfg_ss_').replace('_slat_', ' / ')}"
                 if method == "morphflow" else {"morphany3d": "MorphAny3D", "interp3d": "Interp3D"}[method])
        values = [row["weighted_lpips_mean"] for row in rows]
        axes[0].plot(progress, values, "o-", color=color, label=label, markersize=4)
        if not overview:
            sd = [row["weighted_lpips_std_pairs"] for row in rows]
            axes[0].fill_between(progress, [max(0, x-y) for x, y in zip(values, sd)],
                                 [x+y for x, y in zip(values, sd)], color=color, alpha=.12)
        for ax, first, second in ((axes[1], "src1_weighted_mean", "src2_weighted_mean"),
                                  (axes[2], "lpips_src1_mean", "lpips_src2_mean")):
            ax.plot(progress, [row[first] for row in rows], "o-", color=color, markersize=3)
            ax.plot(progress, [row[second] for row in rows], "s--", color=color, markersize=3)
    axes[0].set_title(title + ("\nBands: ±1 SD across pairs" if not overview else ""))
    axes[0].set_ylabel("Weighted endpoint LPIPS")
    axes[0].legend(fontsize=8, ncol=3 if overview else 1)
    axes[1].set_ylabel("Weighted components")
    axes[1].set_title("Solid: alpha × LPIPS(src1); dashed: (1-alpha) × LPIPS(src2)", fontsize=9)
    axes[2].set_ylabel("Unweighted LPIPS")
    axes[2].set_title("Solid: LPIPS(src1); dashed: LPIPS(src2)", fontsize=9)
    has_endpoints = any(row["alpha"] in (0.0, 1.0) for row in curves)
    frame_note = "canonical source endpoints included" if has_endpoints else "observed intermediates only"
    axes[2].set_xlabel("Progress = 1-alpha (0 = src1, 1 = src2); " + frame_note)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        destination = output.with_suffix("." + extension)
        temporary = destination.with_name(destination.stem + ".tmp." + str(os.getpid()) + destination.suffix)
        fig.savefig(temporary, dpi=160)
        temporary.replace(destination)
    plt.close(fig)


def cell_report(spec, cache, per_pair_plots=False):
    output = spec["directory"] / "weighted_endpoint_lpips"
    signature = source_signature(spec)
    completed = output / "summary.json"
    products = ("curves.csv", "pairs.csv", "summary.csv", "evolution.png", "evolution.pdf")
    if completed.is_file() and all((output / name).is_file() for name in products):
        previous = read_json(completed)
        if (previous.get("input_signature") == signature and
                (not per_pair_plots or previous.get("per_pair_plots"))):
            return previous
    all_curves, summaries, pair_rows, identities = [], [], [], {}
    trajectories = {}
    for method in METHODS:
        trajectories[method] = {
            pair: read_curve(path, spec["alphas"], cache, spec["include_endpoints"])
            for pair, path in spec["sources"][method].items()
        }
        curves, summary = aggregate_method(spec["name"], method, trajectories[method], spec["include_endpoints"])
        all_curves.extend(curves)
        summaries.append(summary)
        identities[method] = digest(trajectories[method])
        for pair, rows in trajectories[method].items():
            for step, row in enumerate(rows, 1):
                pair_rows.append({"configuration": spec["name"], "method": method,
                                  "pair": pair, "step": step, **row})
    write_csv(output / "curves.csv", all_curves)
    write_csv(output / "pairs.csv", pair_rows)
    write_csv(output / "summary.csv", summaries)
    trajectory = trajectory_spec(spec["alphas"], spec["include_endpoints"])
    title = (f"{spec['name']} — {len(spec['pair_names'])} pairs; "
             f"{trajectory['num_intermediates']} intermediates + 2 sources")
    plot_curves(all_curves, output / "evolution", title)
    if per_pair_plots:
        for pair in spec["pair_names"]:
            curves = []
            for method in METHODS:
                values, _ = aggregate_method(spec["name"], method, {pair: trajectories[method][pair]}, spec["include_endpoints"])
                curves.extend(values)
            plot_curves(curves, output / "pairs" / pair, pair)
    report = {"schema_version": SCHEMA, "formula": FORMULA, "interpretation": interpretation(spec["include_endpoints"]),
              "input_signature": signature, "configuration": spec["name"],
              "alphas": spec["alphas"], "pair_names": spec["pair_names"],
              "num_pairs": len(spec["pair_names"]), "num_views": 64,
              "trajectory": trajectory,
              "per_pair_plots": per_pair_plots, "method_content_sha256": identities,
              "summaries": summaries, "curves": all_curves}
    write_json(completed, report)
    print(f"Weighted endpoint LPIPS: {output}")
    return report


def report(run_root, per_pair_plots=False):
    run_root = Path(run_root).resolve()
    paths = sorted(run_root.glob("cfg_ss_*/final_metrics.json"))
    if not paths:
        raise FileNotFoundError(f"No completed CFG cells in {run_root}")
    specs = [cell_spec(path) for path in paths]
    first = specs[0]
    for spec in specs[1:]:
        if spec["pair_names"] != first["pair_names"]:
            raise ValueError(f"Mismatched pair identities across CFG cells: {spec['name']}")
        if not same_alphas(spec["alphas"], first["alphas"]):
            raise ValueError(f"Mismatched alpha schedule across CFG cells: {spec['name']}")
        if spec["include_endpoints"] != first["include_endpoints"]:
            raise ValueError(f"Mismatched endpoint scoring across CFG cells: {spec['name']}")
    cache = {}
    reports = [cell_report(spec, cache, per_pair_plots) for spec in specs]
    for item in reports[1:]:
        for baseline in METHODS[1:]:
            if item["method_content_sha256"][baseline] != reports[0]["method_content_sha256"][baseline]:
                raise ValueError(f"Different {baseline} baseline curves across completed CFG cells")
    curves, summaries = [], []
    for index, item in enumerate(reports):
        # The shared baselines contribute once, not once for every MorphFlow CFG.
        for target, key in ((curves, "curves"), (summaries, "summaries")):
            for row in item[key]:
                if row["method"] == "morphflow" or index == 0:
                    target.append({**row, "configuration": row["configuration"] if row["method"] == "morphflow" else "shared"})
    output = run_root / "weighted_endpoint_lpips"
    signature = digest([item["input_signature"] for item in reports])
    previous = read_json(output / "summary.json") if (output / "summary.json").is_file() else {}
    if (previous.get("input_signature") != signature or
            not all((output / name).is_file() for name in ("curves.csv", "summary.csv", "evolution.png", "evolution.pdf"))):
        write_csv(output / "curves.csv", curves)
        write_csv(output / "summary.csv", summaries)
        trajectory = trajectory_spec(first["alphas"], first["include_endpoints"])
        title = ("Weighted endpoint LPIPS — all completed CFG cells\n"
                 f"{trajectory['num_intermediates']} intermediates + 2 sources")
        plot_curves(curves, output / "evolution", title, overview=True)
        write_json(output / "summary.json", {
            "schema_version": SCHEMA, "formula": FORMULA, "interpretation": interpretation(first["include_endpoints"]),
            "input_signature": signature, "completed_configurations": [spec["name"] for spec in specs],
            "num_pairs": len(first["pair_names"]), "num_views": 64, "alphas": first["alphas"],
            "trajectory": trajectory,
            "baseline_reuse": "Shared baseline curves included once; each MorphFlow CFG remains separate",
            "summaries": summaries, "curves": curves})
    print(f"Weighted endpoint LPIPS overview: {output / 'evolution.png'}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--per-pair-plots", action="store_true",
                        help="Also draw three-model comparison plots for every pair/CFG (optional, many files)")
    args = parser.parse_args()
    try:
        report(args.run_root, args.per_pair_plots)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(1, f"Weighted endpoint report: {error}\n")


if __name__ == "__main__":
    main()
