#!/usr/bin/env python3
"""
Encode one GLB asset into TRELLIS dataset latents using 150 multiview renders.

Outputs MorphFlow-ready files:
  ss_latent.pt
  slat_feats.pt
  slat_coords.pt
  structured_latent.pt
  occupancy.pt
  manifest.json

Run from an environment where the TRELLIS dataset toolkits work.
Example:
  python encode_glb_trellis_slat150.py \
    --glb /path/to/asset.glb \
    --trellis-root /path/to/TRELLIS \
    --out-dir /path/to/MorphFlow_dataset/assets/asset_name \
    --num-views 150 \
    --blender /path/to/blender
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Iterable, Optional


DEFAULT_FEATURE_MODEL = "dinov2_vitl14_reg"
DEFAULT_SS_ENCODER = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
DEFAULT_SLAT_ENCODER = "microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16"
OFFICIAL_BLENDER_PATH = Path("/tmp/blender-3.0.1-linux-x64/blender")


SINGLE_GLB_ADAPTER = r'''
import os
import argparse
import pandas as pd
from tqdm import tqdm


def add_args(parser: argparse.ArgumentParser):
    pass


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    records = []
    rows = metadata.to_dict('records')
    for metadatum in tqdm(rows, desc=desc):
        sha256 = str(metadatum['sha256'])
        local_path = str(metadatum['local_path'])
        file_path = local_path if os.path.isabs(local_path) else os.path.join(output_dir, local_path)
        try:
            record = func(file_path, sha256)
            if record is not None:
                records.append(record)
        except Exception as exc:
            print(f"Error processing object {sha256}: {exc}", flush=True)
    return pd.DataFrame.from_records(records)
'''.lstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode a single .glb into TRELLIS SS/SLAT latents via 150-view rendering + DINO features."
    )
    parser.add_argument("--glb", required=True, type=Path, help="Input .glb/.gltf path.")
    parser.add_argument(
        "--trellis-root",
        required=True,
        type=Path,
        help="Path to the cloned microsoft/TRELLIS repository. Must contain dataset_toolkits/.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Final output directory. The MorphFlow-ready .pt files are written here.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="TRELLIS intermediate directory. Default: <out-dir>/_trellis_work.",
    )
    parser.add_argument(
        "--asset-name",
        default=None,
        help="Name recorded in manifest.json. Default: sanitized GLB stem.",
    )
    parser.add_argument(
        "--instance-id",
        default=None,
        help="Internal id used by TRELLIS toolkit. Default: <asset-name>_<sha12>.",
    )
    parser.add_argument("--num-views", type=int, default=150, help="Number of multiview renders. Default: 150.")
    parser.add_argument(
        "--render-progress-interval",
        type=float,
        default=2.0,
        help="Seconds between render image-count progress updates. Default: 2.0.",
    )
    parser.add_argument(
        "--render-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel Blender processes for rendering views of this one asset. "
            "Default: 1 uses the official TRELLIS render.py path. Try 2-4 on one L40S."
        ),
    )
    parser.add_argument("--feature-model", default=DEFAULT_FEATURE_MODEL, help="DINO feature model.")
    parser.add_argument("--ss-encoder", default=DEFAULT_SS_ENCODER, help="TRELLIS sparse-structure encoder.")
    parser.add_argument("--slat-encoder", default=DEFAULT_SLAT_ENCODER, help="TRELLIS SLat encoder.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for DINO feature extraction.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch TRELLIS toolkit scripts. Default: current Python.",
    )
    parser.add_argument(
        "--blender",
        type=Path,
        default=None,
        help=(
            "Optional Blender executable. If set, the script symlinks it to the path hardcoded by "
            "TRELLIS render.py: /tmp/blender-3.0.1-linux-x64/blender."
        ),
    )
    parser.add_argument(
        "--copy-glb",
        action="store_true",
        help="Copy the GLB into work-dir instead of symlinking it. Symlink is the default.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite existing final .pt outputs. Intermediate TRELLIS files are still reused when present.",
    )
    parser.add_argument(
        "--only-convert",
        action="store_true",
        help="Skip rendering/voxel/DINO/encoding and only convert existing TRELLIS .npz files in work-dir.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._-")
    return value or "asset"


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_render_images(render_dir: Path) -> int:
    if not render_dir.exists():
        return 0
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    return sum(1 for path in render_dir.rglob("*") if path.is_file() and path.suffix.lower() in image_suffixes)


def monitor_render_progress(
    work_dir: Path,
    instance_id: str,
    expected_views: int,
    stop_event: threading.Event,
    interval: float,
) -> None:
    render_dir = work_dir / "renders" / instance_id
    last_count = -1
    while not stop_event.is_set():
        count = count_render_images(render_dir)
        if count != last_count:
            pct = 100.0 * min(count, expected_views) / max(expected_views, 1)
            print(
                f"[render progress] {count}/{expected_views} images written ({pct:.1f}%) -> {render_dir}",
                flush=True,
            )
            last_count = count
        stop_event.wait(max(interval, 0.25))

    count = count_render_images(render_dir)
    pct = 100.0 * min(count, expected_views) / max(expected_views, 1)
    print(
        f"[render progress] final: {count}/{expected_views} images written ({pct:.1f}%) -> {render_dir}",
        flush=True,
    )


def monitor_parallel_render_progress(
    chunk_root: Path,
    expected_views: int,
    stop_event: threading.Event,
    interval: float,
) -> None:
    last_count = -1
    while not stop_event.is_set():
        count = 0
        if chunk_root.exists():
            for chunk_dir in chunk_root.glob("chunk_*"):
                if chunk_dir.is_dir():
                    count += count_render_images(chunk_dir)
        if count != last_count:
            pct = 100.0 * min(count, expected_views) / max(expected_views, 1)
            print(
                f"[render progress] {count}/{expected_views} images written ({pct:.1f}%) -> {chunk_root}",
                flush=True,
            )
            last_count = count
        stop_event.wait(max(interval, 0.25))

    count = 0
    if chunk_root.exists():
        for chunk_dir in chunk_root.glob("chunk_*"):
            if chunk_dir.is_dir():
                count += count_render_images(chunk_dir)
    pct = 100.0 * min(count, expected_views) / max(expected_views, 1)
    print(
        f"[render progress] final: {count}/{expected_views} images written ({pct:.1f}%) -> {chunk_root}",
        flush=True,
    )


def run(
    cmd: Iterable[str],
    cwd: Path,
    env: dict[str, str],
    progress_monitor: Optional[Callable[[threading.Event], None]] = None,
) -> None:
    cmd = [str(part) for part in cmd]
    run_env = env.copy()
    run_env.setdefault("PYTHONUNBUFFERED", "1")

    print("\n$ " + " ".join(cmd), flush=True)

    stop_event = threading.Event()
    monitor_thread = None
    if progress_monitor is not None:
        monitor_thread = threading.Thread(target=progress_monitor, args=(stop_event,), daemon=True)
        monitor_thread.start()

    try:
        subprocess.run(cmd, cwd=str(cwd), env=run_env, check=True)
    finally:
        if monitor_thread is not None:
            stop_event.set()
            monitor_thread.join(timeout=5.0)


def check_trellis_root(trellis_root: Path) -> None:
    required = [
        trellis_root / "dataset_toolkits" / "render.py",
        trellis_root / "dataset_toolkits" / "voxelize.py",
        trellis_root / "dataset_toolkits" / "extract_feature.py",
        trellis_root / "dataset_toolkits" / "encode_ss_latent.py",
        trellis_root / "dataset_toolkits" / "encode_latent.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("TRELLIS root is missing required toolkit files:\n" + "\n".join(missing))


def install_single_glb_adapter(trellis_root: Path) -> Path:
    datasets_dir = trellis_root / "dataset_toolkits" / "datasets"
    if not datasets_dir.is_dir():
        raise FileNotFoundError(f"TRELLIS dataset modules directory not found: {datasets_dir}")
    adapter_path = datasets_dir / "SingleGLB.py"
    old = adapter_path.read_text(encoding="utf-8") if adapter_path.exists() else None
    if old != SINGLE_GLB_ADAPTER:
        adapter_path.write_text(SINGLE_GLB_ADAPTER, encoding="utf-8")
        print(f"Wrote TRELLIS dataset adapter: {adapter_path}", flush=True)
    return adapter_path


def ensure_blender(blender: Optional[Path]) -> None:
    candidate = blender
    if candidate is None:
        found = shutil.which("blender")
        candidate = Path(found) if found else None

    if OFFICIAL_BLENDER_PATH.exists() and os.access(OFFICIAL_BLENDER_PATH, os.X_OK):
        return

    if candidate is None:
        print(
            "WARNING: Blender executable not found. TRELLIS render.py may try to install Blender under /tmp "
            "and may call sudo. On HPC, pass --blender /path/to/blender if available.",
            flush=True,
        )
        return

    candidate = candidate.expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Blender executable does not exist: {candidate}")

    OFFICIAL_BLENDER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OFFICIAL_BLENDER_PATH.exists() or OFFICIAL_BLENDER_PATH.is_symlink():
        OFFICIAL_BLENDER_PATH.unlink()
    OFFICIAL_BLENDER_PATH.symlink_to(candidate)
    print(f"Linked Blender for TRELLIS render.py: {OFFICIAL_BLENDER_PATH} -> {candidate}", flush=True)


def prepare_single_asset_dataset(
    glb: Path,
    work_dir: Path,
    asset_name: str,
    instance_id: Optional[str],
    copy_glb: bool,
) -> tuple[str, Path, Path, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = work_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    digest = file_sha256(glb)
    internal_id = instance_id or f"{asset_name}_{digest[:12]}"
    internal_id = sanitize_name(internal_id)

    suffix = glb.suffix.lower() or ".glb"
    staged = raw_dir / f"{internal_id}{suffix}"
    if not staged.exists():
        if copy_glb:
            shutil.copy2(glb, staged)
            print(f"Copied GLB to: {staged}", flush=True)
        else:
            try:
                staged.symlink_to(glb.resolve())
                print(f"Symlinked GLB to: {staged}", flush=True)
            except OSError:
                shutil.copy2(glb, staged)
                print(f"Symlink failed; copied GLB to: {staged}", flush=True)

    rel_path = staged.relative_to(work_dir).as_posix()
    metadata_path = work_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sha256", "local_path", "file_identifier", "aesthetic_score"])
        writer.writeheader()
        writer.writerow(
            {
                "sha256": internal_id,
                "local_path": rel_path,
                "file_identifier": glb.name,
                "aesthetic_score": 10.0,
            }
        )

    instances_path = work_dir / "instances.txt"
    instances_path.write_text(internal_id + "\n", encoding="utf-8")
    print(f"Prepared single-asset metadata: {metadata_path}", flush=True)
    return internal_id, metadata_path, instances_path, staged


def build_trellis_render_views(trellis_root: Path, num_views: int) -> list[dict]:
    """Build the same type of camera-view list expected by TRELLIS blender_script/render.py."""
    import numpy as np

    toolkit_dir = trellis_root / "dataset_toolkits"
    toolkit_str = str(toolkit_dir)
    if toolkit_str not in sys.path:
        sys.path.insert(0, toolkit_str)
    from utils import sphere_hammersley_sequence

    offset = (np.random.rand(), np.random.rand())
    views = []
    for i in range(num_views):
        yaw, pitch = sphere_hammersley_sequence(i, num_views, offset)
        views.append(
            {
                "yaw": float(yaw),
                "pitch": float(pitch),
                "radius": 2.0,
                "fov": float(40 / 180 * np.pi),
            }
        )
    return views


def split_indexed_views(views: list[dict], workers: int) -> list[list[tuple[int, dict]]]:
    workers = max(1, min(int(workers), len(views)))
    indexed = list(enumerate(views))
    chunks = [[] for _ in range(workers)]
    for idx, item in enumerate(indexed):
        chunks[idx % workers].append(item)
    return [chunk for chunk in chunks if chunk]


def tail_file(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def write_render_status_csv(work_dir: Path, instance_id: str) -> None:
    status_path = work_dir / "rendered_0.csv"
    with status_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sha256", "rendered"])
        writer.writeheader()
        writer.writerow({"sha256": instance_id, "rendered": True})


def render_single_asset_parallel(
    trellis_root: Path,
    work_dir: Path,
    instance_id: str,
    staged_asset: Path,
    num_views: int,
    render_workers: int,
    env: dict[str, str],
    progress_interval: float,
) -> None:
    """Render one object by splitting the TRELLIS view list across several Blender processes."""
    render_dir = work_dir / "renders" / instance_id
    transforms_path = render_dir / "transforms.json"
    mesh_path = render_dir / "mesh.ply"
    if transforms_path.exists() and mesh_path.exists() and count_render_images(render_dir) >= num_views:
        print(f"Skipping render: existing complete render folder found at {render_dir}", flush=True)
        write_render_status_csv(work_dir, instance_id)
        return

    if not OFFICIAL_BLENDER_PATH.exists():
        raise FileNotFoundError(
            f"Parallel rendering calls Blender directly, but {OFFICIAL_BLENDER_PATH} does not exist. "
            "Pass --blender /path/to/blender so the script can link it first."
        )

    if render_dir.exists():
        shutil.rmtree(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)

    chunk_root = work_dir / "_render_chunks" / instance_id
    if chunk_root.exists():
        shutil.rmtree(chunk_root)
    chunk_root.mkdir(parents=True, exist_ok=True)

    views = build_trellis_render_views(trellis_root, num_views)
    chunks = split_indexed_views(views, render_workers)
    print(
        f"Parallel render: {num_views} views split across {len(chunks)} Blender processes. Logs: {chunk_root}",
        flush=True,
    )

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_parallel_render_progress,
        args=(chunk_root, num_views, stop_event, progress_interval),
        daemon=True,
    )
    monitor_thread.start()

    processes = []
    try:
        for chunk_idx, chunk in enumerate(chunks):
            chunk_dir = chunk_root / f"chunk_{chunk_idx:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            log_path = chunk_root / f"chunk_{chunk_idx:03d}.log"
            chunk_views = [view for _, view in chunk]
            cmd = [
                str(OFFICIAL_BLENDER_PATH),
                "-b",
                "-P",
                str(trellis_root / "dataset_toolkits" / "blender_script" / "render.py"),
                "--",
                "--views",
                json.dumps(chunk_views),
                "--object",
                str(staged_asset),
                "--resolution",
                "512",
                "--output_folder",
                str(chunk_dir),
                "--engine",
                "CYCLES",
            ]
            if chunk_idx == 0:
                cmd.append("--save_mesh")

            log_handle = log_path.open("w", encoding="utf-8")
            print(
                f"[render worker {chunk_idx}] {len(chunk)} views -> {chunk_dir}",
                flush=True,
            )
            proc = subprocess.Popen(
                cmd,
                cwd=str(trellis_root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((proc, log_handle, log_path, chunk_dir, chunk))

        failures = []
        for proc, log_handle, log_path, chunk_dir, chunk in processes:
            code = proc.wait()
            log_handle.close()
            if code != 0:
                failures.append((code, log_path))
        if failures:
            message_parts = []
            for code, log_path in failures:
                message_parts.append(f"{log_path} exited with code {code}:\n{tail_file(log_path)}")
            raise RuntimeError("One or more Blender render workers failed:\n\n" + "\n\n".join(message_parts))
    finally:
        stop_event.set()
        monitor_thread.join(timeout=5.0)
        for proc, log_handle, _, _, _ in processes:
            if proc.poll() is None:
                proc.terminate()
            try:
                log_handle.close()
            except Exception:
                pass

    merged = None
    frames: list[Optional[dict]] = [None] * num_views
    for chunk_idx, (_, _, log_path, chunk_dir, chunk) in enumerate(processes):
        chunk_transforms_path = chunk_dir / "transforms.json"
        if not chunk_transforms_path.exists():
            raise FileNotFoundError(
                f"Missing transforms.json for render chunk {chunk_idx}: {chunk_transforms_path}\n"
                f"Log tail:\n{tail_file(log_path)}"
            )
        chunk_metadata = json.loads(chunk_transforms_path.read_text(encoding="utf-8"))
        if merged is None:
            merged = {k: v for k, v in chunk_metadata.items() if k != "frames"}

        chunk_frames = chunk_metadata.get("frames", [])
        if len(chunk_frames) != len(chunk):
            raise ValueError(
                f"Chunk {chunk_idx} wrote {len(chunk_frames)} frame metadata entries for {len(chunk)} views."
            )
        for local_idx, (global_idx, _) in enumerate(chunk):
            src_image = chunk_dir / f"{local_idx:03d}.png"
            dst_image = render_dir / f"{global_idx:03d}.png"
            if not src_image.exists():
                raise FileNotFoundError(f"Missing rendered image: {src_image}")
            shutil.copy2(src_image, dst_image)
            frame = dict(chunk_frames[local_idx])
            frame["file_path"] = f"{global_idx:03d}.png"
            frames[global_idx] = frame

    missing = [idx for idx, frame in enumerate(frames) if frame is None]
    if missing:
        raise RuntimeError(f"Missing merged frame metadata for view indices: {missing[:20]}")

    mesh_src = chunk_root / "chunk_000" / "mesh.ply"
    if not mesh_src.exists():
        raise FileNotFoundError(f"Parallel render did not produce mesh.ply in first chunk: {mesh_src}")
    shutil.copy2(mesh_src, mesh_path)

    assert merged is not None
    merged["frames"] = frames
    transforms_path.write_text(json.dumps(merged, indent=4), encoding="utf-8")
    write_render_status_csv(work_dir, instance_id)
    print(f"Parallel render complete: {render_dir}", flush=True)


def merge_status_csvs(work_dir: Path, prefixes: Iterable[str]) -> None:
    """Optional convenience: merge TRELLIS status CSVs back into metadata.csv."""
    try:
        import pandas as pd
    except Exception:
        return

    metadata_path = work_dir / "metadata.csv"
    if not metadata_path.exists():
        return
    metadata = pd.read_csv(metadata_path, dtype={"sha256": str})
    for prefix in prefixes:
        for status_path in sorted(work_dir.glob(f"{prefix}*.csv")):
            status = pd.read_csv(status_path, dtype={"sha256": str})
            if status.empty or "sha256" not in status.columns:
                continue
            metadata = metadata.merge(status, on="sha256", how="left", suffixes=("", "__new"))
            for col in list(metadata.columns):
                if col.endswith("__new"):
                    base = col[:-5]
                    if base in metadata.columns:
                        metadata[base] = metadata[col].combine_first(metadata[base])
                    else:
                        metadata[base] = metadata[col]
                    metadata = metadata.drop(columns=[col])
    metadata.to_csv(metadata_path, index=False)


def build_occupancy_from_voxel_ply(voxel_ply: Path, resolution: int = 64):
    import numpy as np
    import torch

    positions = None
    try:
        import utils3d

        positions = utils3d.io.read_ply(str(voxel_ply))[0]
    except Exception:
        pass

    if positions is None:
        try:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(str(voxel_ply))
            positions = np.asarray(pcd.points)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read voxel PLY {voxel_ply}. Install utils3d or open3d in this environment."
            ) from exc

    positions = torch.from_numpy(np.asarray(positions)).float()
    coords = ((positions + 0.5) * resolution).long()
    coords = torch.clamp(coords, 0, resolution - 1)
    occupancy = torch.zeros(resolution, resolution, resolution, dtype=torch.bool)
    occupancy[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return occupancy


def convert_to_morphflow_format(
    work_dir: Path,
    out_dir: Path,
    asset_name: str,
    instance_id: str,
    glb: Path,
    num_views: int,
    feature_model: str,
    ss_encoder: str,
    slat_encoder: str,
    trellis_root: Path,
    overwrite: bool,
) -> None:
    import numpy as np
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        out_dir / "ss_latent.pt",
        out_dir / "slat_feats.pt",
        out_dir / "slat_coords.pt",
        out_dir / "structured_latent.pt",
        out_dir / "occupancy.pt",
        out_dir / "manifest.json",
    ]
    if not overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "Output files already exist. Use --overwrite-output to replace them:\n"
                + "\n".join(str(path) for path in existing)
            )

    ss_latent_name = ss_encoder.rstrip("/").split("/")[-1]
    slat_latent_name = f"{feature_model}_{slat_encoder.rstrip('/').split('/')[-1]}"

    ss_npz_path = work_dir / "ss_latents" / ss_latent_name / f"{instance_id}.npz"
    slat_npz_path = work_dir / "latents" / slat_latent_name / f"{instance_id}.npz"
    voxel_ply_path = work_dir / "voxels" / f"{instance_id}.ply"

    missing = [path for path in [ss_npz_path, slat_npz_path, voxel_ply_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing TRELLIS intermediate outputs:\n" + "\n".join(str(path) for path in missing))

    ss_npz = np.load(ss_npz_path)
    if "mean" not in ss_npz:
        raise KeyError(f"Expected key 'mean' in {ss_npz_path}. Keys: {list(ss_npz.keys())}")
    ss_latent = torch.from_numpy(ss_npz["mean"]).float()
    if ss_latent.ndim == 4:
        ss_latent = ss_latent.unsqueeze(0)

    slat_npz = np.load(slat_npz_path)
    for key in ["feats", "coords"]:
        if key not in slat_npz:
            raise KeyError(f"Expected key {key!r} in {slat_npz_path}. Keys: {list(slat_npz.keys())}")
    slat_feats = torch.from_numpy(slat_npz["feats"]).float()
    slat_coords = torch.from_numpy(slat_npz["coords"]).to(torch.int32)
    if slat_coords.ndim != 2:
        raise ValueError(f"Unexpected slat coords shape: {tuple(slat_coords.shape)}")
    if slat_coords.shape[1] == 3:
        batch_col = torch.zeros((slat_coords.shape[0], 1), dtype=torch.int32)
        slat_coords = torch.cat([batch_col, slat_coords], dim=1)
    elif slat_coords.shape[1] != 4:
        raise ValueError(f"Expected slat coords shape [N,3] or [N,4], got {tuple(slat_coords.shape)}")

    occupancy = build_occupancy_from_voxel_ply(voxel_ply_path, resolution=64)

    torch.save(ss_latent.cpu(), out_dir / "ss_latent.pt")
    torch.save(slat_feats.cpu(), out_dir / "slat_feats.pt")
    torch.save(slat_coords.cpu(), out_dir / "slat_coords.pt")
    torch.save({"feats": slat_feats.cpu(), "coords": slat_coords.cpu()}, out_dir / "structured_latent.pt")
    torch.save(occupancy.cpu(), out_dir / "occupancy.pt")

    manifest = {
        "asset_name": asset_name,
        "instance_id": instance_id,
        "source_glb": str(glb.resolve()),
        "trellis_root": str(trellis_root.resolve()),
        "work_dir": str(work_dir.resolve()),
        "num_views": num_views,
        "feature_model": feature_model,
        "ss_encoder": ss_encoder,
        "slat_encoder": slat_encoder,
        "trellis_intermediates": {
            "ss_npz": str(ss_npz_path),
            "slat_npz": str(slat_npz_path),
            "voxel_ply": str(voxel_ply_path),
        },
        "outputs": {
            "ss_latent": "ss_latent.pt",
            "slat_feats": "slat_feats.pt",
            "slat_coords": "slat_coords.pt",
            "structured_latent": "structured_latent.pt",
            "occupancy": "occupancy.pt",
        },
        "shapes": {
            "ss_latent": list(ss_latent.shape),
            "slat_feats": list(slat_feats.shape),
            "slat_coords": list(slat_coords.shape),
            "occupancy": list(occupancy.shape),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone. MorphFlow-ready asset files:", flush=True)
    for path in outputs:
        print(f"  {path}", flush=True)
    print("\nShapes:", flush=True)
    print(f"  ss_latent:  {tuple(ss_latent.shape)}", flush=True)
    print(f"  slat_feats: {tuple(slat_feats.shape)}", flush=True)
    print(f"  slat_coords:{tuple(slat_coords.shape)}", flush=True)
    print(f"  occupancy:  {tuple(occupancy.shape)}", flush=True)


def main() -> None:
    args = parse_args()
    glb = args.glb.expanduser().resolve()
    trellis_root = args.trellis_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    work_dir = (args.work_dir.expanduser().resolve() if args.work_dir else (out_dir / "_trellis_work").resolve())
    asset_name = sanitize_name(args.asset_name or glb.stem)

    if not glb.exists():
        raise FileNotFoundError(f"Input GLB/GLTF not found: {glb}")
    if glb.suffix.lower() not in {".glb", ".gltf", ".blend", ".obj", ".fbx"}:
        print(f"WARNING: input suffix is {glb.suffix!r}; TRELLIS render.py must be able to load it.", flush=True)

    check_trellis_root(trellis_root)
    install_single_glb_adapter(trellis_root)
    ensure_blender(args.blender)

    instance_id, metadata_path, instances_path, staged_asset = prepare_single_asset_dataset(
        glb=glb,
        work_dir=work_dir,
        asset_name=asset_name,
        instance_id=args.instance_id,
        copy_glb=args.copy_glb,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(trellis_root),
            str(trellis_root / "dataset_toolkits"),
            env.get("PYTHONPATH", ""),
        ]
    )

    if not args.only_convert:
        if args.render_workers > 1:
            render_single_asset_parallel(
                trellis_root=trellis_root,
                work_dir=work_dir,
                instance_id=instance_id,
                staged_asset=staged_asset,
                num_views=args.num_views,
                render_workers=args.render_workers,
                env=env,
                progress_interval=args.render_progress_interval,
            )
        else:
            run(
                [
                    args.python,
                    "dataset_toolkits/render.py",
                    "SingleGLB",
                    "--output_dir",
                    str(work_dir),
                    "--instances",
                    str(instances_path),
                    "--num_views",
                    str(args.num_views),
                    "--max_workers",
                    "1",
                ],
                cwd=trellis_root,
                env=env,
                progress_monitor=lambda stop_event: monitor_render_progress(
                    work_dir=work_dir,
                    instance_id=instance_id,
                    expected_views=args.num_views,
                    stop_event=stop_event,
                    interval=args.render_progress_interval,
                ),
            )
        merge_status_csvs(work_dir, ["rendered_"])

        run(
            [
                args.python,
                "dataset_toolkits/voxelize.py",
                "SingleGLB",
                "--output_dir",
                str(work_dir),
                "--instances",
                str(instances_path),
                "--max_workers",
                "1",
            ],
            cwd=trellis_root,
            env=env,
        )
        merge_status_csvs(work_dir, ["voxelized_"])

        run(
            [
                args.python,
                "dataset_toolkits/extract_feature.py",
                "--output_dir",
                str(work_dir),
                "--instances",
                str(instances_path),
                "--model",
                args.feature_model,
                "--batch_size",
                str(args.batch_size),
            ],
            cwd=trellis_root,
            env=env,
        )
        merge_status_csvs(work_dir, [f"feature_{args.feature_model}_"])

        run(
            [
                args.python,
                "dataset_toolkits/encode_ss_latent.py",
                "--output_dir",
                str(work_dir),
                "--instances",
                str(instances_path),
                "--enc_pretrained",
                args.ss_encoder,
            ],
            cwd=trellis_root,
            env=env,
        )
        ss_latent_name = args.ss_encoder.rstrip("/").split("/")[-1]
        merge_status_csvs(work_dir, [f"ss_latent_{ss_latent_name}_"])

        run(
            [
                args.python,
                "dataset_toolkits/encode_latent.py",
                "--output_dir",
                str(work_dir),
                "--instances",
                str(instances_path),
                "--feat_model",
                args.feature_model,
                "--enc_pretrained",
                args.slat_encoder,
            ],
            cwd=trellis_root,
            env=env,
        )
        slat_latent_name = f"{args.feature_model}_{args.slat_encoder.rstrip('/').split('/')[-1]}"
        merge_status_csvs(work_dir, [f"latent_{slat_latent_name}_"])

    convert_to_morphflow_format(
        work_dir=work_dir,
        out_dir=out_dir,
        asset_name=asset_name,
        instance_id=instance_id,
        glb=glb,
        num_views=args.num_views,
        feature_model=args.feature_model,
        ss_encoder=args.ss_encoder,
        slat_encoder=args.slat_encoder,
        trellis_root=trellis_root,
        overwrite=args.overwrite_output,
    )


if __name__ == "__main__":
    main()
