"""Optional source-appearance transfer onto TRELLIS reconstructed geometry.

This auxiliary export needs the original GLB. It preserves decoded geometry and
UVs, and fits its texture from independent source views with TRELLIS's official
texture optimizer. It is not an appearance reconstruction from the latent alone.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time


VERSION = 1


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fit_cameras(nviews, trellis_root, seed=42):
    """Seeded training views; the benchmark's eight evaluation views stay held out."""
    import numpy as np

    utils_path = Path(trellis_root) / "dataset_toolkits/utils.py"
    spec = importlib.util.spec_from_file_location("_source_texture_official_dataset_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    offset = tuple(np.random.RandomState(seed).rand(2))
    cameras = []
    for index in range(nviews):
        yaw, pitch = module.sphere_hammersley_sequence(index, nviews, offset)
        cameras.append({"yaw_degrees": float(np.degrees(yaw) % 360),
                        "pitch_degrees": float(np.degrees(pitch)),
                        "radius": 2.0, "fov_degrees": 40.0})
    return cameras


def camera_matrices(camera):
    """OpenCV extrinsics/intrinsics matching render_glb's OpenGL camera exactly."""
    import numpy as np

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
    extrinsics = np.diag([1, -1, -1, 1]) @ view
    focal = 0.5 / np.tan(np.deg2rad(camera["fov_degrees"]) / 2)
    intrinsics = np.array([[focal, 0, 0.5], [0, focal, 0.5], [0, 0, 1]])
    return extrinsics.astype(np.float32), intrinsics.astype(np.float32)


def transfer_source_texture(source_glb, decoded_textured_glb, out_glb, normalization,
                            trellis_root=None, texture_size=2048, nviews=100,
                            resolution=1024, seed=42):
    """Fit source unlit sRGB appearance onto an unchanged decoded UV mesh.

    ``normalization`` is the original GLB's official rendering scale/offset in
    internal Z-up coordinates. The decoded mesh already occupies that frame.
    Training masks intersect both silhouettes and exclude their outermost pixel
    to limit background contamination where reconstruction geometry differs.
    """
    source_glb = Path(source_glb).expanduser().resolve()
    decoded_textured_glb = Path(decoded_textured_glb).expanduser().resolve()
    out_glb = Path(out_glb).expanduser().resolve()
    if not source_glb.is_file() or not decoded_textured_glb.is_file():
        raise FileNotFoundError("Both original and decoded textured GLBs must exist")
    if out_glb in (source_glb, decoded_textured_glb):
        raise ValueError("Source-texture export must use a separate output GLB")
    texture_size, nviews, resolution, seed = map(int, (texture_size, nviews, resolution, seed))
    if min(texture_size, nviews, resolution) < 1:
        raise ValueError("texture_size, nviews and resolution must be positive")
    # Normalize to JSON types before comparing resumable cache identities.
    normalization = json.loads(json.dumps(normalization))
    identity = {"version": VERSION, "source_glb": str(source_glb),
                "source_sha256": _sha256(source_glb),
                "decoded_glb": str(decoded_textured_glb),
                "decoded_sha256": _sha256(decoded_textured_glb),
                "normalization": normalization, "texture_size": texture_size,
                "fit_views_requested": nviews, "fit_resolution": resolution, "seed": seed,
                "optimizer": "official bake_texture opt", "lambda_tv": 0.01,
                "mask": "source and reconstruction alpha>0.5; erode one pixel",
                "observation": "unlit sRGB source base-color; white composite masked out"}
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = out_glb.with_suffix(".json")
    if metadata_path.is_file():
        old = json.loads(metadata_path.read_text())
        if old["identity"] != identity:
            raise ValueError(f"Source-texture cache belongs to different inputs/settings: {metadata_path}")
        if old.get("status") == "complete" and out_glb.is_file() and out_glb.stat().st_size:
            return old
    elif out_glb.exists():
        raise FileExistsError(f"Output exists without source-texture provenance: {out_glb}")
    _write_json(metadata_path, {"identity": identity, "status": "running"})

    started = time.monotonic()
    import numpy as np
    import torch
    import trimesh
    import cv2
    from PIL import Image
    from trellis_asset_codec import OfficialAssetCodec

    # Enforce the same official package identity as the latent codec. This does
    # not load an encoder or a decoder, and reuses already imported modules.
    codec = OfficialAssetCodec(trellis_root=trellis_root, texture_size=texture_size, seed=seed)
    codec._init()
    from trellis.utils import postprocessing_utils
    benchmark_dir = Path(__file__).resolve().parent / "benchmarks"
    sys.path.insert(0, str(benchmark_dir))
    from trellis_codec_metrics import render_glb, load_glb_parts, heldout_cameras

    cameras = fit_cameras(nviews, codec.trellis_root, seed)
    evaluation_cameras = heldout_cameras()
    for camera in cameras:
        for evaluation in evaluation_cameras:
            yaw_delta = abs((camera["yaw_degrees"] - evaluation["yaw_degrees"] + 180) % 360 - 180)
            if yaw_delta < 1e-5 and abs(camera["pitch_degrees"] - evaluation["pitch_degrees"]) < 1e-5:
                raise ValueError("A source-texture training camera coincides with a held-out evaluation camera")

    parts = load_glb_parts(decoded_textured_glb, frame="y_up")
    if len(parts) != 1:
        raise ValueError("Decoded GLB must use one shared texture atlas, as produced by the official codec")
    vertices, faces, uvs, offset = [], [], [], 0
    for part in parts:
        uv = getattr(part["visual"], "uv", None)
        if uv is None or len(uv) != len(part["vertices"]):
            raise ValueError("Decoded GLB must contain the official texture atlas UVs")
        vertices.append(np.asarray(part["vertices"], dtype=np.float32))
        faces.append(np.asarray(part["faces"], dtype=np.int32) + offset)
        uvs.append(np.asarray(uv, dtype=np.float32))
        offset += len(part["vertices"])
    vertices, faces, uvs = map(np.concatenate, (vertices, faces, uvs))
    if not np.isfinite(uvs).all():
        raise ValueError("Decoded UV coordinates are nonfinite")

    print(f"[source texture] Rendering {nviews} independent source/reconstruction views at {resolution}px", flush=True)
    source_renders = render_glb(source_glb, normalization=normalization, cameras=cameras, resolution=resolution)
    recon_renders = render_glb(decoded_textured_glb, cameras=cameras, resolution=resolution)
    observations, masks, extrinsics, intrinsics, used_cameras, discarded = [], [], [], [], [], []
    mask_pixels = []
    for index, (source, recon, camera) in enumerate(zip(source_renders, recon_renders, cameras)):
        mask = ((source["alpha"] > 0.5) & (recon["alpha"] > 0.5)).astype(np.uint8)
        mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        if not mask.any():
            discarded.append(index)
            continue
        # Official bake_texture divides every observation by 255. Pass float
        # 0..255 values to retain the renderer's interpolated sRGB precision.
        observations.append(np.ascontiguousarray(np.clip(source["rgb"], 0, 1) * 255, dtype=np.float32))
        masks.append(mask)
        extrinsic, intrinsic = camera_matrices(camera)
        extrinsics.append(extrinsic)
        intrinsics.append(intrinsic)
        used_cameras.append(camera)
        mask_pixels.append(int(mask.sum()))
    del source_renders, recon_renders
    if not observations:
        raise ValueError("Source and reconstructed surfaces do not overlap in any fitting view")
    rendering_finished = time.monotonic()
    print(f"[source texture] Optimizing {texture_size}px texture from {len(observations)} views; geometry and UVs retained", flush=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with torch.inference_mode(False), torch.enable_grad():
        texture = postprocessing_utils.bake_texture(
            vertices, faces, uvs, observations, masks, extrinsics, intrinsics,
            texture_size=texture_size, near=0.1, far=10.0, mode="opt",
            lambda_tv=0.01, verbose=False)
    baking_finished = time.monotonic()
    # Row-vector rotation used by the official to_glb; all geometry is unchanged.
    rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=1.0, baseColorTexture=Image.fromarray(texture),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8))
    mesh = trimesh.Trimesh(vertices @ rotation, faces,
                           visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
                           process=False)
    temporary = out_glb.with_name(out_glb.stem + ".tmp.glb")
    mesh.export(temporary)
    temporary.replace(out_glb)
    result = {"identity": identity, "status": "complete", "output_glb": str(out_glb),
              "output_sha256": _sha256(out_glb),
              "role": "auxiliary source-assisted appearance; original GLB required; not latent-only decoding",
              "geometry_preserved": True, "uvs_reused": True,
              "vertices": int(len(vertices)), "faces": int(len(faces)),
              "fit_cameras": used_cameras, "fit_views_used": len(used_cameras),
              "discarded_empty_view_indices": discarded, "fit_mask_pixels": mask_pixels,
              "evaluation_views_used_for_fitting": False,
              "trellis_import": codec._trellis_file,
              "seconds": {"prepare_and_render": rendering_finished - started,
                          "bake": baking_finished - rendering_finished,
                          "total": time.monotonic() - started}}
    _write_json(metadata_path, result)
    print(f"[source texture] Complete ({time.monotonic() - started:.1f}s): {out_glb}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_glb", type=Path)
    parser.add_argument("decoded_textured_glb", type=Path)
    parser.add_argument("out_glb", type=Path)
    parser.add_argument("--normalization-json", required=True, type=Path,
                        help="Official transforms.json or an object containing scale/offset or normalization")
    parser.add_argument("--trellis-root", type=Path)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--nviews", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    normalization = json.loads(args.normalization_json.read_text())
    normalization = normalization.get("normalization", normalization)
    if "scale" in normalization:
        normalization = {"scale": normalization["scale"], "offset": normalization.get("offset", [0, 0, 0])}
    elif "matrix" not in normalization:
        parser.error("--normalization-json must provide scale/offset or a matrix")
    transfer_source_texture(args.source_glb, args.decoded_textured_glb, args.out_glb,
                            normalization, trellis_root=args.trellis_root,
                            texture_size=args.texture_size, nviews=args.nviews,
                            resolution=args.resolution, seed=args.seed)


if __name__ == "__main__":
    main()
