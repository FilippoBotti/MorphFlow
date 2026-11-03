"""Measure MorphFlow sensitivity to GLB-reencoded source SLat conditioning.

Primary comparisons use fresh dataset-conditioned outputs versus GLB-conditioned
outputs at identical seeds and alpha. Appearance is rendered from the actual
Gaussian decoder representation, without transferring original input textures.
All geometry stays in the shared internal Z-up unit-cube frame.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def alpha_dir(alpha):
    return ("alpha_%.6f" % float(alpha)).replace(".", "p")


def _as_numpy(value, integer=False):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if not integer:
            value = value.float()
        value = value.numpy()
    return np.asarray(value, dtype=np.int64 if integer else np.float32)


def _coords_keys(coords):
    import numpy as np
    coords = _as_numpy(coords, integer=True)
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError("Expected sparse coordinates [N,3] or [N,4]")
    xyz = coords[:, -3:]
    if np.any(xyz < 0) or np.any(xyz >= 64):
        raise ValueError("Sparse coordinates must be within the 64-cubed grid")
    batch = coords[:, 0] if coords.shape[1] == 4 else np.zeros(len(coords), dtype=np.int64)
    return batch * 64**3 + xyz @ np.array([4096, 64, 1], dtype=np.int64)


def sparse_diagnostics(reference, encoded):
    """Align sparse feature rows by coordinate before comparing their values."""
    import numpy as np
    ka, kb = _coords_keys(reference["coords"]), _coords_keys(encoded["coords"])
    if len(np.unique(ka)) != len(ka) or len(np.unique(kb)) != len(kb):
        raise ValueError("Sparse tensor contains duplicate coordinates")
    common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    union = np.union1d(ka, kb)
    result = {"reference_voxels": len(ka), "encoded_voxels": len(kb),
              "common_voxels": len(common), "coordinate_iou": float(len(common) / max(1, len(union))),
              "coordinate_sets_equal": bool(np.array_equal(np.sort(ka), np.sort(kb))),
              "noise_caveat": "Identical SLat RNG seed does not imply identical noise at a spatial coordinate when sparse row sets/order differ."}
    if len(common) and "feats" in reference and "feats" in encoded:
        a, b = _as_numpy(reference["feats"])[ia].astype(np.float64), _as_numpy(encoded["feats"])[ib].astype(np.float64)
        if a.shape != b.shape:
            raise ValueError("Sparse features have incompatible channel dimensions")
        result["feature_rmse_common_coords"] = float(np.sqrt(np.mean((a - b) ** 2)))
        cosine = np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
        result["feature_cosine_common_coords"] = float(np.mean(cosine))
    return result


def dense_diagnostics(reference, encoded):
    import numpy as np
    a, b = _as_numpy(reference).astype(np.float64), _as_numpy(encoded).astype(np.float64)
    result = {"reference_shape": list(a.shape), "encoded_shape": list(b.shape)}
    if a.shape != b.shape:
        result["status"] = "shape_mismatch"
        return result
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite dense latent values")
    result.update(status="available", latent_rmse=float(np.sqrt(np.mean((a - b) ** 2))),
                  latent_max_abs=float(np.max(np.abs(a - b))), latent_exact_equal=bool(np.array_equal(a, b)))
    return result


def photometric_diagnostics(reference_renders, encoded_renders):
    """Fit one affine RGB transform across all eight views as a diagnostic only.

    This fit uses the evaluation target. It is deliberately not presented as a
    deployable image correction or a second independent reconstruction score.
    Raw uncorrected appearance metrics remain the primary comparison.
    """
    import numpy as np
    from trellis_codec_metrics import appearance_metrics

    n = 0
    sum_x = np.zeros(3, np.float64)
    sum_y = np.zeros(3, np.float64)
    sum_xx = np.zeros(3, np.float64)
    sum_xy = np.zeros(3, np.float64)
    for target, generated in zip(reference_renders, encoded_renders):
        mask = (target["alpha"] > 0.5) & (generated["alpha"] > 0.5)
        x = generated["rgb"][mask].astype(np.float64)
        y = target["rgb"][mask].astype(np.float64)
        n += len(x)
        sum_x += x.sum(axis=0)
        sum_y += y.sum(axis=0)
        sum_xx += (x * x).sum(axis=0)
        sum_xy += (x * y).sum(axis=0)
    if not n:
        return {"status": "no_foreground_intersection"}
    mean_x, mean_y = sum_x / n, sum_y / n
    variance = sum_xx / n - mean_x**2
    covariance = sum_xy / n - mean_x * mean_y
    gain = np.ones(3, np.float64)
    valid = variance > 1e-12
    gain[valid] = np.maximum(0, covariance[valid] / variance[valid])
    bias = mean_y - gain * mean_x
    corrected = [{**r, "rgb": np.clip(r["rgb"] * gain + bias, 0, 1).astype(np.float32)} for r in encoded_renders]
    scores = appearance_metrics(reference_renders, corrected)
    return {"status": "available", "gain_rgb": gain.tolist(), "bias_rgb": bias.tolist(),
            "fit_foreground_pixels": int(n),
            "fit_protocol": "Single per-channel affine transform across all evaluation views, fitted on foreground intersection; target-assisted diagnostic only.",
            "psnr_foreground_union": scores["psnr_foreground_union"], "psnr_white": scores["psnr_white"]}


def _load_raw(path):
    import torch
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not all(k in value for k in ("vertices", "faces", "gaussian")):
        raise ValueError(f"Expected raw mesh plus genuine Gaussian appearance: {path}")
    if not len(value["vertices"]) or not len(value["faces"]):
        raise ValueError(f"Empty raw mesh: {path}")
    return value


def _raw_parts(raw):
    return [{"vertices": _as_numpy(raw["vertices"]), "faces": _as_numpy(raw["faces"], integer=True)}]


class _RawRenderer:
    def __init__(self, resolution=384):
        import numpy as np
        import torch
        import trellis
        from trellis.renderers import GaussianRenderer, MeshRenderer
        from trellis.representations import Gaussian, MeshExtractResult
        from trellis_source_texture import camera_matrices
        from trellis_codec_metrics import heldout_cameras

        if not torch.cuda.is_available():
            raise RuntimeError("Raw Gaussian appearance metrics require an allocated CUDA GPU")
        self.torch, self.np = torch, np
        self.Gaussian, self.MeshExtractResult = Gaussian, MeshExtractResult
        self.trellis_import = str(Path(trellis.__file__).resolve())
        self.resolution = int(resolution)
        self.cameras = heldout_cameras()
        self.matrices = [(torch.from_numpy(e).cuda(), torch.from_numpy(i).cuda())
                         for e, i in map(camera_matrices, self.cameras)]
        self.gaussian_renderer = GaussianRenderer({"resolution": resolution, "near": 0.1,
                                                   "far": 10.0, "ssaa": 1, "bg_color": (1, 1, 1)})
        self.mesh_renderer = MeshRenderer({"resolution": resolution, "near": 0.1, "far": 10.0, "ssaa": 1})

    def render(self, raw):
        torch, np = self.torch, self.np
        state = raw["gaussian"]
        gaussian = self.Gaussian(**state["init_params"], device="cuda")
        for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
            value = state["tensors"][name]
            setattr(gaussian, name, value.to(device="cuda", dtype=torch.float32) if value is not None else None)
        gaussian.active_sh_degree = int(state["active_sh_degree"])
        mesh = self.MeshExtractResult(vertices=raw["vertices"].float().cuda(), faces=raw["faces"].long().cuda(),
                                     vertex_attrs=None, res=int(raw.get("res", 256)))
        output = []
        with torch.no_grad():
            for camera, (extrinsics, intrinsics) in zip(self.cameras, self.matrices):
                color = self.gaussian_renderer.render(gaussian, extrinsics, intrinsics)["color"]
                alpha = self.mesh_renderer.render(mesh, extrinsics, intrinsics, return_types=["mask"])["mask"]
                rgb = color.detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                mask = alpha.detach().float().clamp(0, 1).cpu().numpy().reshape(self.resolution, self.resolution)
                if not np.isfinite(rgb).all():
                    raise ValueError("Gaussian renderer produced nonfinite RGB")
                output.append({"rgb": rgb, "alpha": mask, "camera": dict(camera)})
        del mesh, gaussian
        return output


def _save_view(view, path):
    import numpy as np
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.dstack((view["rgb"], view["alpha"]))
    Image.fromarray(np.clip(rgba * 255, 0, 255).astype(np.uint8)).save(path)


def _raw_reproducibility(reference, historical):
    import numpy as np
    result = {"metadata_reference": reference.get("metadata", {}), "metadata_historical": historical.get("metadata", {})}
    equal = True
    for key in ("vertices", "faces", "vertex_attrs"):
        a, b = reference.get(key), historical.get(key)
        if a is None or b is None:
            matched = a is None and b is None
        else:
            a, b = _as_numpy(a, integer=key == "faces"), _as_numpy(b, integer=key == "faces")
            matched = a.shape == b.shape and np.array_equal(a, b)
            if a.shape == b.shape and a.size:
                result[key + "_max_abs"] = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        result[key + "_exact_equal"] = bool(matched)
        equal &= matched
    gs_equal = True
    for key, a in reference["gaussian"]["tensors"].items():
        b = historical["gaussian"]["tensors"].get(key)
        if a is None or b is None:
            matched = a is None and b is None
        else:
            a, b = _as_numpy(a), _as_numpy(b)
            matched = a.shape == b.shape and np.array_equal(a, b)
        gs_equal &= matched
    result["gaussian_tensors_exact_equal"] = bool(gs_equal)
    result["all_raw_tensors_exact_equal"] = bool(equal and gs_equal)
    return result


def _path(root, value):
    value = Path(value)
    return value if value.is_absolute() else root / value


def _historical_path(root, manifest, pair, alpha, directory):
    for key in ("baseline_raw", "historical_raw"):
        mapping = pair.get(key)
        if isinstance(mapping, dict):
            for name in (directory, str(alpha), "%.6f" % alpha):
                if name in mapping:
                    return _path(root, mapping[name])
    baseline = manifest.get("baseline_run") or manifest.get("baseline_run_root") or manifest.get("historical_run")
    if isinstance(baseline, dict):
        baseline = baseline.get("root") or baseline.get("path")
    if baseline:
        return _path(root, baseline) / "cfg_ss_5_slat_1.5/morphflow" / pair["pair_dir"] / directory / "mesh_raw.pt"
    return None


def _latent_diagnostics(reference_dir, encoded_dir):
    import torch
    output = {"structure": {"status": "not_available"}, "slat": {"status": "not_available"}}
    slat_packs = []
    for folder in (reference_dir, encoded_dir):
        path = folder / "slat.pt"
        slat_packs.append(torch.load(path, map_location="cpu", weights_only=True) if path.is_file() else None)
    if all(pack is not None and all(k in pack for k in ("feats", "coords")) for pack in slat_packs):
        output["slat"] = {"status": "available", **sparse_diagnostics(*slat_packs)}
    ss_packs = []
    for folder in (reference_dir, encoded_dir):
        path = next((folder / name for name in ("ss.pt", "ss_latent.pt", "structure.pt") if (folder / name).is_file()), None)
        ss_packs.append(torch.load(path, map_location="cpu", weights_only=True) if path is not None else None)
    if all(pack is not None for pack in ss_packs):
        def dense(pack):
            if not isinstance(pack, dict):
                return pack
            return next((pack[k] for k in ("latent", "ss_latent", "pred_ss", "ss") if k in pack), None)
        a, b = map(dense, ss_packs)
        if a is not None and b is not None:
            output["structure"] = dense_diagnostics(a, b)
        if all(isinstance(pack, dict) and "coords" in pack for pack in ss_packs):
            coords = sparse_diagnostics(ss_packs[0], ss_packs[1])
            output["structure"]["occupancy_iou"] = coords["coordinate_iou"]
            output["structure"]["reference_occupied_voxels"] = coords["reference_voxels"]
            output["structure"]["encoded_occupied_voxels"] = coords["encoded_voxels"]
    return output


def _mean_groups(rows):
    result = {}
    for group in ("geometry", "appearance", "structure", "slat"):
        dictionaries = [row.get(group, {}) for row in rows]
        keys = sorted({key for value in dictionaries for key, number in value.items()
                       if isinstance(number, (int, float)) and not isinstance(number, bool)})
        result[group] = {key: sum(value[key] for value in dictionaries if key in value) /
                        sum(key in value for value in dictionaries) for key in keys}
    result["num_alphas"] = len(rows)
    return result


def _trajectory(rows, endpoints, path):
    import numpy as np
    from PIL import Image, ImageDraw
    labels = ["src1"] + ["alpha %.6f" % row["alpha"] for row in rows] + ["src2"]
    images = {branch: [endpoints[0]] + [row["_trajectory_views"][branch] for row in rows] + [endpoints[1]]
              for branch in ("dataset_control", "glb_encoded")}
    tile, left, gap, caption = 256, 190, 8, 26
    canvas = Image.new("RGB", (left + (tile + gap) * len(labels) + gap, 2 * (tile + caption + gap) + gap), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (branch, title) in enumerate((("dataset_control", "SLat dal dataset"), ("glb_encoded", "SLat dal GLB"))):
        y = gap + row_index * (tile + caption + gap)
        draw.text((8, y + 18), title, fill="black")
        for column, (view, label) in enumerate(zip(images[branch], labels)):
            x = left + gap + column * (tile + gap)
            image = Image.fromarray(np.clip(view["rgb"] * 255, 0, 255).astype(np.uint8)).resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y))
            draw.text((x, y + tile + 4), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def evaluate_pair(runroot, index):
    """Write per-alpha comparisons and a two-row trajectory, returning metrics."""
    from trellis_codec_metrics import geometry_metrics, appearance_metrics, _save_sheet

    root = Path(runroot).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    pair = manifest["pairs"][int(index)]
    name = pair["pair_dir"]
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("pair_dir must be one directory name")
    out = root / "pairs" / name
    renderer = _RawRenderer(resolution=int(manifest.get("metric_resolution", 384)))
    rows = []
    alphas = manifest.get("alphas", pair.get("alphas"))
    if not alphas:
        raise ValueError("Manifest must contain alpha values")
    for alpha in alphas:
        alpha = float(alpha)
        directory = alpha_dir(alpha)
        reference_dir, encoded_dir = out / "dataset_control" / directory, out / "glb_encoded" / directory
        reference_path, encoded_path = reference_dir / "mesh_raw.pt", encoded_dir / "mesh_raw.pt"
        reference, encoded = _load_raw(reference_path), _load_raw(encoded_path)
        reference_renders, encoded_renders = renderer.render(reference), renderer.render(encoded)
        geometry = geometry_metrics(_raw_parts(reference), _raw_parts(encoded), sample_count=20000, seed=42)
        appearance = appearance_metrics(reference_renders, encoded_renders)
        appearance["render_mode"] = "Official Gaussian decoder appearance, white background; triangle-mesh masks; shared Z-up cameras; no texture transfer."
        calibrated = photometric_diagnostics(reference_renders, encoded_renders)
        appearance["photometric_diagnostic"] = calibrated
        if calibrated["status"] == "available":
            appearance["psnr_foreground_union_gain_bias_corrected"] = calibrated["psnr_foreground_union"]
            appearance["psnr_white_gain_bias_corrected"] = calibrated["psnr_white"]
        destination = out / "comparison" / directory
        destination.mkdir(parents=True, exist_ok=True)
        for view_index, (a, b) in enumerate(zip(reference_renders, encoded_renders)):
            _save_view(a, destination / ("dataset_control_view_%02d.png" % view_index))
            _save_view(b, destination / ("glb_encoded_view_%02d.png" % view_index))
        _save_sheet([("SLat dataset", reference_renders), ("SLat da GLB", encoded_renders)], destination / "comparison.png")
        latent = _latent_diagnostics(reference_dir, encoded_dir)
        row = {"alpha": alpha, "alpha_dir": directory, "geometry": geometry, "appearance": appearance,
               **latent, "paths": {"reference_raw": str(reference_path), "encoded_raw": str(encoded_path),
                   "comparison_png": str(destination / "comparison.png"),
                   "reference_view_07": str(destination / "dataset_control_view_07.png"),
                   "encoded_view_07": str(destination / "glb_encoded_view_07.png")}}
        historical_path = _historical_path(root, manifest, pair, alpha, directory)
        if historical_path is not None and historical_path.is_file():
            historical = _load_raw(historical_path)
            repro = _raw_reproducibility(reference, historical)
            repro["historical_raw"] = str(historical_path)
            if not repro["all_raw_tensors_exact_equal"]:
                historical_renders = renderer.render(historical)
                repro["appearance"] = appearance_metrics(historical_renders, reference_renders)
                repro["geometry"] = geometry_metrics(_raw_parts(historical), _raw_parts(reference), sample_count=20000, seed=42)
                row["historical_vs_glb_encoded"] = {
                    "appearance": appearance_metrics(historical_renders, encoded_renders),
                    "geometry": geometry_metrics(_raw_parts(historical), _raw_parts(encoded), sample_count=20000, seed=42)}
                for view_index, view in enumerate(historical_renders):
                    _save_view(view, destination / ("historical_view_%02d.png" % view_index))
                _save_sheet([("Risultato dataset gia salvato", historical_renders),
                             ("Controllo dataset rigenerato", reference_renders),
                             ("SLat da GLB", encoded_renders)], destination / "historical_comparison.png")
                row["paths"]["historical_comparison_png"] = str(destination / "historical_comparison.png")
                del historical_renders
            else:
                row["historical_vs_glb_encoded"] = {"appearance": appearance, "geometry": geometry}
            row["historical_reproducibility"] = repro
            del historical
        _write_json(destination / "metrics.json", row)
        row["_trajectory_views"] = {"dataset_control": reference_renders[7], "glb_encoded": encoded_renders[7]}
        rows.append(row)
        del reference, encoded, reference_renders, encoded_renders
        gc.collect()
        print("[conditioning metrics] %s alpha=%.6f PSNR=%.3f IoU=%.4f" % (
            name, alpha, appearance["psnr_foreground_union"], appearance["silhouette_iou"]), flush=True)
    endpoints = []
    for label in ("src1", "src2"):
        mapping = pair.get("source_raw", {})
        source_path = mapping.get(label)
        if source_path is None:
            raise ValueError("Manifest pair.source_raw must provide src1/src2 raw Gaussian reference paths")
        raw = _load_raw(_path(root, source_path))
        view = renderer.render(raw)[7]
        endpoints.append(view)
        _save_view(view, out / "comparison" / (label + "_view_07.png"))
        del raw
    trajectory = out / "comparison.png"
    _trajectory(rows, endpoints, trajectory)
    for row in rows:
        del row["_trajectory_views"]
    result = {"pair_id": pair.get("pair_id", int(index)), "pair_dir": name,
              "src1": pair.get("src1"), "src2": pair.get("src2"),
              "primary_reference": "fresh dataset_control", "compared_method": "glb_encoded",
              "appearance_uses_original_texture": False, "geometry_alignment": "none; shared internal Z-up frame",
              "render_backend": "official TRELLIS GaussianRenderer", "trellis_import": renderer.trellis_import,
              "evaluation_cameras": renderer.cameras, "resolution": renderer.resolution,
              "alphas": rows, "aggregate": _mean_groups(rows), "trajectory_png": str(trajectory),
              "endpoint_views": {label: str(out / "comparison" / (label + "_view_07.png")) for label in ("src1", "src2")}}
    _write_json(out / "metrics.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runroot", type=Path)
    parser.add_argument("index", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_pair(args.runroot, args.index), indent=2))


if __name__ == "__main__":
    main()
