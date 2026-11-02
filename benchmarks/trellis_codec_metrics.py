"""Deterministic, aligned GLB round-trip measurements.

Torch and nvdiffrast are imported only by the renderer.  Geometry is measured in
the supplied common coordinate frame: the reconstruction is never individually
centred or scaled.  Distances/thresholds are in the unit-cube frame used by
TRELLIS, not a fraction of the (possibly incorrect) reconstruction bounding box.
"""

import json
import math
from pathlib import Path

import numpy as np


def _frame_matrix(frame):
    if frame == "z_up":
        return np.eye(3)
    if frame == "y_up":
        # Inverse of official postprocessing_utils.to_glb's row-vector rotation.
        return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    raise ValueError("frame must be 'y_up' or 'z_up'")


def _transform(vertices, frame="y_up", normalization=None):
    vertices = np.asarray(vertices, dtype=np.float64) @ _frame_matrix(frame)
    if normalization is None:
        return vertices
    if isinstance(normalization, dict) and "matrix" not in normalization:
        # This matches Blender's normalization after conversion to Blender Z-up.
        return (vertices * float(normalization.get("scale", 1.0))
                + np.asarray(normalization.get("offset", [0, 0, 0]), dtype=np.float64))
    matrix = np.asarray(normalization.get("matrix") if isinstance(normalization, dict)
                        else normalization, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("normalization matrix must have shape (4, 4)")
    return (np.c_[vertices, np.ones(len(vertices))] @ matrix.T)[:, :3]


def load_glb_parts(path, frame="y_up", normalization=None):
    """Load every transformed mesh instance, retaining per-primitive materials."""
    import trimesh

    scene = trimesh.load(str(path), force="scene", process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        matrix, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name]
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            continue
        vertices = trimesh.transform_points(np.asarray(mesh.vertices), matrix)
        vertices = _transform(vertices, frame, normalization)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("GLB contains non-triangular mesh: " + str(path))
        if not np.isfinite(vertices).all():
            raise ValueError("GLB contains non-finite coordinates: " + str(path))
        parts.append({"vertices": vertices, "faces": faces, "visual": mesh.visual})
    if not parts:
        raise ValueError("GLB contains no triangle meshes: " + str(path))
    return parts


def _merged(parts):
    vertices, faces, offset = [], [], 0
    for part in parts:
        vertices.append(part["vertices"])
        faces.append(part["faces"].astype(np.int64) + offset)
        offset += len(part["vertices"])
    return np.concatenate(vertices), np.concatenate(faces)


def _sample_surface(vertices, faces, count, seed):
    """Area-weighted barycentric samples independent of trimesh RNG versions."""
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError("Mesh surface has no finite nonzero area")
    rng = np.random.default_rng(seed)
    tri = triangles[rng.choice(len(triangles), size=count, p=area / area.sum())]
    r = rng.random((count, 2))
    s = np.sqrt(r[:, 0])
    return (tri[:, 0] * (1 - s[:, None])
            + tri[:, 1] * (s * (1 - r[:, 1]))[:, None]
            + tri[:, 2] * (s * r[:, 1])[:, None])


def geometry_metrics(source_parts, recon_parts, sample_count=20000, seed=42):
    """Chamfer means use Euclidean (unsquared) distances in a shared frame."""
    from scipy.spatial import cKDTree

    sv, sf = _merged(source_parts)
    rv, rf = _merged(recon_parts)
    source = _sample_surface(sv, sf, sample_count, seed)
    recon = _sample_surface(rv, rf, sample_count, seed + 1)
    # Independent samples have a sampling floor; retain both directed distances.
    source_to_recon = cKDTree(recon).query(source, workers=1)[0]
    recon_to_source = cKDTree(source).query(recon, workers=1)[0]
    result = {
        "sample_count_per_mesh": int(sample_count), "sample_seed": int(seed),
        "distance_units": "common TRELLIS unit cube; no per-mesh rescaling",
        "chamfer_l1": float((source_to_recon.mean() + recon_to_source.mean()) / 2),
        "chamfer_l1_definition": "half the sum of directed mean Euclidean distances",
        "source_to_recon_mean": float(source_to_recon.mean()),
        "recon_to_source_mean": float(recon_to_source.mean()),
        "symmetric_distance_p95": float(max(np.percentile(source_to_recon, 95),
                                              np.percentile(recon_to_source, 95))),
        "source_bounds": np.stack((sv.min(0), sv.max(0))).tolist(),
        "recon_bounds": np.stack((rv.min(0), rv.max(0))).tolist(),
        "source_vertices": int(len(sv)), "source_faces": int(len(sf)),
        "recon_vertices": int(len(rv)), "recon_faces": int(len(rf)),
    }
    for threshold in (0.01, 0.02):
        precision = float(np.mean(recon_to_source < threshold))
        recall = float(np.mean(source_to_recon < threshold))
        key = "%.2f" % threshold
        result["fscore_" + key] = 2 * precision * recall / max(precision + recall, 1e-15)
        result["precision_" + key] = precision
        result["recall_" + key] = recall
    return result


def heldout_cameras():
    """Eight explicit Z-up views, with yaw offsets and two elevations."""
    return [{"yaw_degrees": 22.5 + 90 * i, "pitch_degrees": pitch,
             "radius": 2.0, "fov_degrees": 40.0}
            for pitch in (12.0, 32.0) for i in range(4)]


def _view_projection(camera):
    yaw, pitch = np.deg2rad([camera["yaw_degrees"], camera["pitch_degrees"]])
    eye = camera["radius"] * np.array([np.cos(yaw) * np.cos(pitch),
                                       np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    backward = eye / np.linalg.norm(eye)
    right = np.cross([0, 0, 1], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    view = np.eye(4)
    view[:3, :3] = np.stack((right, up, backward))
    view[:3, 3] = -view[:3, :3] @ eye
    near, far = 0.1, 10.0
    f = 1 / np.tan(np.deg2rad(camera["fov_degrees"]) / 2)
    projection = np.zeros((4, 4))
    projection[0, 0] = projection[1, 1] = f
    projection[2, 2] = -(far + near) / (far - near)
    projection[2, 3] = -2 * far * near / (far - near)
    projection[3, 2] = -1
    return projection @ view


def _unit_color(value, size=4):
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if len(value) < 3:
        return np.ones(size, dtype=np.float32)
    if value.max(initial=0) > 1:
        value = value / 255.0
    if len(value) == 3 and size == 4:
        value = np.r_[value, 1.0]
    return value[:size]


def _visual_arrays(part):
    visual = part["visual"]
    count = len(part["vertices"])
    color = np.tile(np.array([0.7, 0.7, 0.7, 1.0], np.float32), (count, 1))
    uv = texture = None
    factor = np.ones(4, np.float32)
    alpha_cutoff = 0.0
    if getattr(visual, "kind", None) == "texture":
        material = getattr(visual, "material", None)
        image = getattr(material, "baseColorTexture", None)
        if image is None:
            image = getattr(material, "image", None)
        raw_factor = getattr(material, "baseColorFactor", None)
        if raw_factor is None:
            raw_factor = getattr(material, "diffuse", None)
        if raw_factor is not None:
            factor = _unit_color(raw_factor)
        if image is not None and getattr(visual, "uv", None) is not None:
            texture = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
            # nvdiffrast texture row zero is v=0; PIL row zero is the top.
            texture = np.ascontiguousarray(texture[::-1])
            uv = np.ascontiguousarray(visual.uv, dtype=np.float32)
        else:
            color[:] = factor
            factor = np.ones(4, np.float32)
        if getattr(material, "alphaMode", None) == "MASK":
            cutoff = getattr(material, "alphaCutoff", 0.5)
            alpha_cutoff = 0.5 if cutoff is None else float(cutoff)
    elif getattr(visual, "kind", None) in ("vertex", "face"):
        raw = np.asarray(visual.vertex_colors)
        color = raw.astype(np.float32) / (255.0 if raw.dtype.kind in "ui" or raw.max() > 1 else 1.0)
        if color.shape[1] == 3:
            color = np.c_[color, np.ones(len(color), dtype=np.float32)]
    return color, uv, texture, factor, alpha_cutoff


def render_glb(path, frame="y_up", normalization=None, cameras=None,
               resolution=384, device="cuda", context=None):
    """Render flat, unlit RGBA with bilinear base-color texture sampling.

    No independent centring, lighting, tone mapping, or image auto-crop is used.
    Returned RGB is composited over white; alpha is the geometric/material mask.
    """
    import torch
    import nvdiffrast.torch as dr

    if not torch.cuda.is_available():
        raise RuntimeError("GLB appearance measurements require an allocated CUDA GPU")
    cameras = heldout_cameras() if cameras is None else cameras
    context = dr.RasterizeCudaContext(device=device) if context is None else context
    parts = load_glb_parts(path, frame, normalization)
    prepared = []
    for part in parts:
        color, uv, texture, factor, cutoff = _visual_arrays(part)
        prepared.append({
            "vertices": torch.tensor(np.c_[part["vertices"], np.ones(len(part["vertices"]))],
                                     device=device, dtype=torch.float32),
            "faces": torch.tensor(part["faces"], device=device, dtype=torch.int32),
            "colors": torch.tensor(color, device=device, dtype=torch.float32),
            "uv": None if uv is None else torch.tensor(uv, device=device),
            "texture": None if texture is None else torch.tensor(texture[None], device=device),
            "factor": torch.tensor(factor, device=device), "cutoff": cutoff,
        })
    rendered = []
    with torch.no_grad():
        for camera in cameras:
            matrix = torch.tensor(_view_projection(camera), device=device, dtype=torch.float32)
            rgb = torch.ones((resolution, resolution, 3), device=device)
            alpha = torch.zeros((resolution, resolution), device=device)
            depth = torch.full((resolution, resolution), float("inf"), device=device)
            for part in prepared:
                clip = (part["vertices"] @ matrix.T)[None]
                rast, _ = dr.rasterize(context, clip, part["faces"], (resolution, resolution))
                inside = rast[0, :, :, 3] > 0
                if part["texture"] is not None:
                    uv, _ = dr.interpolate(part["uv"][None], rast, part["faces"])
                    rgba = dr.texture(part["texture"], uv, filter_mode="linear", boundary_mode="wrap")[0]
                    rgba = rgba * part["factor"]
                else:
                    rgba, _ = dr.interpolate(part["colors"][None], rast, part["faces"])
                    rgba = rgba[0]
                a = rgba[:, :, 3].clamp(0, 1)
                if part["cutoff"]:
                    a = (a >= part["cutoff"]).float()
                visible = inside & (a > 0) & (rast[0, :, :, 2] < depth)
                composite = rgba[:, :, :3].clamp(0, 1) * a[:, :, None] + (1 - a[:, :, None])
                rgb[visible] = composite[visible]
                alpha[visible] = a[visible]
                depth[visible] = rast[0, :, :, 2][visible]
            rendered.append({"rgb": rgb.flip(0).cpu().numpy(),
                             "alpha": alpha.flip(0).cpu().numpy(), "camera": dict(camera)})
    return rendered


def appearance_metrics(source_renders, recon_renders):
    rows = []
    mse_all, mse_foreground = [], []
    for source, recon in zip(source_renders, recon_renders):
        error = np.square(source["rgb"].astype(np.float64) - recon["rgb"].astype(np.float64))
        a, b = source["alpha"] > 0.5, recon["alpha"] > 0.5
        union = a | b
        mse = float(error.mean())
        foreground = float(error[union].mean()) if union.any() else 0.0
        iou = float((a & b).sum() / union.sum()) if union.any() else 1.0
        rows.append({"camera": source["camera"], "silhouette_iou": iou,
                     "mse_white": mse, "mse_foreground_union": foreground,
                     "psnr_white": -10 * math.log10(max(mse, 1e-12)),
                     "psnr_foreground_union": -10 * math.log10(max(foreground, 1e-12))})
        mse_all.append(mse)
        mse_foreground.append(foreground)
    return {"render_mode": "flat unlit sRGB base color, white composite, bilinear textures",
            "psnr_cap_db": 120,
            "psnr_white": -10 * math.log10(max(float(np.mean(mse_all)), 1e-12)),
            "psnr_foreground_union": -10 * math.log10(max(float(np.mean(mse_foreground)), 1e-12)),
            "silhouette_iou": float(np.mean([r["silhouette_iou"] for r in rows])),
            "views": rows}


def _save_sheet(rows, path, thumb_size=256):
    from PIL import Image, ImageDraw

    columns = max(len(renders) for _, renders in rows)
    margin, label_width, caption = 8, 210, 24
    width = label_width + columns * (thumb_size + margin) + margin
    height = margin + len(rows) * (thumb_size + caption + margin)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, renders) in enumerate(rows):
        y = margin + i * (thumb_size + caption + margin)
        # Wrap long labels without depending on a platform-specific font.
        text = str(label)
        wrapped = "\n".join(text[j:j + 28] for j in range(0, len(text), 28))
        draw.text((margin, y + 10), wrapped, fill="black")
        for j, rendered in enumerate(renders):
            rgb = np.clip(rendered["rgb"] * 255, 0, 255).astype(np.uint8)
            tile = Image.fromarray(rgb).resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            x = label_width + margin + j * (thumb_size + margin)
            canvas.paste(tile, (x, y))
            camera = rendered["camera"]
            draw.text((x, y + thumb_size + 3), "yaw %g / pitch %g" % (
                camera["yaw_degrees"], camera["pitch_degrees"]), fill="black")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def render_montage(entries, out_path, normalization=None, resolution=384, cameras=None):
    """Render labeled GLBs into rows with identical fixed cameras.

    Each entry is a dict with ``label``, ``path``, optional ``frame`` (y_up) and
    optional ``normalization``.  Explicit normalization belongs to that entry;
    the shared default is mainly convenient for externally supplied originals.
    """
    rows = []
    for entry in entries:
        if not isinstance(entry, dict):
            entry = {"label": entry[0], "path": entry[1]}
        renders = render_glb(entry["path"], frame=entry.get("frame", "y_up"),
                             normalization=entry.get("normalization", normalization),
                             resolution=resolution, cameras=cameras)
        rows.append((entry["label"], renders))
    return _save_sheet(rows, out_path)


def compare_glbs(source_glb, recon_glb, out_dir, normalization=None,
                 source_frame="y_up", recon_frame="y_up", render=True,
                 resolution=384, sample_count=20000, seed=42):
    """Save CPU geometry, optional CUDA appearance metrics, and contact sheet.

    Source normalization is applied after source_frame -> Z-up conversion.
    Reconstruction coordinates are interpreted as already normalized.  Use the
    scale/offset from encoding/transforms.json, including a rotated GLB's source
    frame, rather than independently fitting either geometry to its bounding box.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_parts = load_glb_parts(source_glb, source_frame, normalization)
    recon_parts = load_glb_parts(recon_glb, recon_frame)
    metrics = {"source_glb": str(Path(source_glb).resolve()),
               "recon_glb": str(Path(recon_glb).resolve()),
               "source_frame": source_frame, "recon_frame": recon_frame,
               "source_normalization": normalization,
               "geometry": geometry_metrics(source_parts, recon_parts, sample_count, seed)}
    if render:
        source_renders = render_glb(source_glb, source_frame, normalization, resolution=resolution)
        recon_renders = render_glb(recon_glb, recon_frame, resolution=resolution)
        metrics["appearance"] = appearance_metrics(source_renders, recon_renders)
        metrics["contact_sheet"] = _save_sheet(
            [("Source", source_renders), ("Reconstruction", recon_renders)],
            out_dir / "comparison.png")
        from PIL import Image
        for index, (source, recon) in enumerate(zip(source_renders, recon_renders)):
            for label, view in (("source", source), ("recon", recon)):
                rgba = np.dstack((view["rgb"], view["alpha"]))
                Image.fromarray(np.clip(rgba * 255, 0, 255).astype(np.uint8)).save(
                    out_dir / ("%s_view_%02d.png" % (label, index)))
    else:
        metrics["appearance"] = {"status": "not_requested"}
    # Convert ndarray normalization metadata for JSON without changing values.
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2,
                                                    default=lambda value: np.asarray(value).tolist()) + "\n")
    return metrics
