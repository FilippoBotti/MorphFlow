"""Generate unseen FLUX pairs and aligned MorphAny3D / MorphFlow sequences.

Planning uses only the Python standard library. GPU phases run in separate
processes so teacher weights and TFSA caches are released before student inference.
See slurm/README_teacher_student_comparison.md for usage and the output layout.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import uuid


ALPHA_DEFINITION = "alpha * src_1 + (1 - alpha) * src_2; alpha=1 is src_1"
BUNDLE_FILES = (
    "ss_latent.pt", "slat_feats.pt", "slat_coords.pt", "structured_latent.pt",
    "occupancy.pt", "mesh.glb", "manifest.json",
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_pair(a, b):
    if not all(isinstance(x, str) and x and Path(x).name == x for x in (a, b)):
        raise ValueError(f"Invalid dataset asset names: {a!r}, {b!r}")
    return tuple(sorted((a, b)))


def discover_images(root):
    """Match the asset naming convention in the existing dataset generator."""
    images = {}
    for path in sorted(Path(root).iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        name = re.sub(r"[^A-Za-z0-9_.=-]+", "_", path.stem).strip("_") or "asset"
        if name in {".", ".."} or name in images:
            raise ValueError(f"Ambiguous asset name {name!r}: {images.get(name)}, {path}")
        images[name] = path.resolve()
    if len(images) < 2:
        raise ValueError(f"Need at least two images in {root}")
    return images


def dataset_pairs(root):
    """Exclude all splits, plus planned/partially generated dataset pairs."""
    root = Path(root)
    metadata_files = sorted(root.glob("metadata*.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata*.json in {root}; cannot guarantee unseen pairs")
    excluded, provenance = set(), []
    for path in metadata_files:
        payload = read_json(path)
        if isinstance(payload, dict):
            payload = payload.get("samples", payload.get("metadata"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a metadata list in {path}")
        for row in payload:
            if not isinstance(row, dict) or not {"src_1", "src_2"} <= row.keys():
                raise ValueError(f"Missing src_1/src_2 in {path}")
            excluded.add(canonical_pair(row["src_1"], row["src_2"]))
        provenance.append({"path": str(path.resolve()), "sha256": sha256(path)})

    def add_pair_key(key):
        parts = key.split("+")
        if len(parts) != 2:
            raise ValueError(f"Invalid dataset pair key: {key!r}")
        excluded.add(canonical_pair(*parts))

    for filename in ("pair_alphas.json", "pair_sequence_alphas.json", "pair_selected_indices.json"):
        path = root / filename
        if not path.is_file():
            continue
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a pair registry in {path}")
        for key, value in payload.items():
            if key in ("train", "val", "test") and isinstance(value, dict):
                for pair_key in value:
                    add_pair_key(pair_key)
            else:
                add_pair_key(key)
        provenance.append({"path": str(path.resolve()), "sha256": sha256(path)})
    for path in sorted((root / "targets").glob("*")):
        if path.is_dir() and "+" in path.name:
            add_pair_key(path.name)
    return excluded, provenance


def pair_rank(i, j, count):
    return i * (2 * count - i - 1) // 2 + j - i - 1


def unrank_pair(rank, count):
    lo, hi = 0, count - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if pair_rank(mid, mid + 1, count) <= rank:
            lo = mid
        else:
            hi = mid
    return lo, lo + 1 + rank - pair_rank(lo, lo + 1, count)


def select_pairs(names, excluded, num_pairs, seed):
    """Sample the exact complement without materializing O(images**2) pairs."""
    names = sorted(names)
    positions = {name: i for i, name in enumerate(names)}
    blocked = sorted({
        pair_rank(positions[a], positions[b], len(names))
        for a, b in (canonical_pair(*pair) for pair in excluded)
        if a in positions and b in positions and a != b
    })
    available = len(names) * (len(names) - 1) // 2 - len(blocked)
    if num_pairs < 1 or num_pairs > available:
        raise ValueError(f"Requested {num_pairs} pairs, but only {available} unseen pairs are available")
    pairs = []
    for ordinal in random.Random(seed).sample(range(available), num_pairs):
        lo, hi = ordinal, ordinal + len(blocked)
        while lo < hi:
            mid = (lo + hi) // 2
            if mid - bisect_right(blocked, mid) < ordinal:
                lo = mid + 1
            else:
                hi = mid
        i, j = unrank_pair(lo, len(names))
        pairs.append((names[i], names[j]))
    return pairs, available


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", required=True, help="Flat FLUX image directory")
    parser.add_argument("--dataset-dir", required=True, help="Existing dataset to exclude (all splits)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-path", required=True, help="Trained SS checkpoint")
    parser.add_argument("--slat-checkpoint-path", required=True)
    parser.add_argument("--num-pairs", type=positive_int, required=True)
    parser.add_argument("--num-intermediates", type=positive_int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--steps", type=positive_int, default=50, help="Student SS solver steps")
    parser.add_argument("--slat-steps", type=positive_int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--slat-cfg-scale", type=float, default=3.0)
    parser.add_argument("--teacher-ss-steps", type=positive_int, default=25)
    parser.add_argument("--teacher-slat-steps", type=positive_int, default=25)
    parser.add_argument("--teacher-ss-cfg", type=float, default=7.5)
    parser.add_argument("--teacher-slat-cfg", type=float, default=3.0)
    parser.add_argument("--tfsa-alpha", type=float, default=0.8)
    parser.add_argument("--tfsa-cache-mode", choices=("file", "memory"), default="file")
    parser.add_argument("--max-work-cache-gb", type=float, default=60.0)
    parser.add_argument("--mixed-precision", choices=("auto", "no", "fp16", "bf16"), default="auto")
    parser.add_argument("--work-cache-dir", help="Temporary TFSA directory; prefer node-local storage")
    parser.add_argument("--stage", choices=("all", "teacher", "student"), default="all")
    parser.add_argument("--resume", action="store_true", help="Resume only an identical saved plan")
    parser.add_argument("--dry-run", action="store_true", help="Write plan.dry_run.json without ML dependencies")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    for key in ("cfg_scale", "slat_cfg_scale", "teacher_ss_cfg", "teacher_slat_cfg", "max_work_cache_gb", "tfsa_alpha"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and nonnegative")
    if args.tfsa_alpha > 1:
        parser.error("--tfsa-alpha must be <= 1")
    for key in ("assets_dir", "dataset_dir", "output_dir", "checkpoint_path", "slat_checkpoint_path"):
        setattr(args, key, str(Path(getattr(args, key)).expanduser().resolve()))
    output = Path(args.output_dir)
    for source in (Path(args.dataset_dir), Path(args.assets_dir)):
        if output == source or source in output.parents or output in source.parents:
            parser.error("--output-dir must be separate from the dataset and FLUX directories")
    return args


def build_plan(args):
    images = discover_images(args.assets_dir)
    excluded, provenance = dataset_pairs(args.dataset_dir)
    selected, available = select_pairs(images, excluded, args.num_pairs, args.seed)
    config = {key: value for key, value in vars(args).items()
              if key not in {"stage", "resume", "dry_run", "work_cache_dir", "output_dir"}}
    checkpoints = {}
    for key in ("checkpoint_path", "slat_checkpoint_path"):
        path = Path(getattr(args, key))
        if not args.dry_run and not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        # Checkpoints can be many GB; record size and mtime without reading twice.
        checkpoints[key] = ({"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                            if path.is_file() else None)
    sources = {}
    for name in sorted({name for pair in selected for name in pair}):
        path = images[name]
        sources[name] = {"image": str(path), "sha256": sha256(path),
                         "input_file": "input" + path.suffix.lower()}
    # Descending alpha is the teacher's temporal src1 -> src2 direction.
    alphas = [i / (args.num_intermediates + 1) for i in range(args.num_intermediates, 0, -1)]
    return {
        "schema_version": 1, "config": config, "checkpoint_identity": checkpoints,
        "alpha_definition": ALPHA_DEFINITION, "alphas": alphas,
        "excluded_pair_count": len(excluded), "available_pair_count": available,
        "exclusion_files": provenance, "sources": sources,
        "pairs": [{"id": f"pair_{index:06d}", "src_1": a, "src_2": b}
                  for index, (a, b) in enumerate(selected)],
        "seeds": {"teacher_and_sources": args.seed,
                  "student": "fixed across alphas; (seed + 2*pair_index) % 2**32 for SS, +1 for SLat"},
    }


def step_name(index):
    return f"alpha_{index:04d}"


def relative_link(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source, destination.parent)
    if destination.is_symlink():
        if os.readlink(destination) != target:
            raise RuntimeError(f"Unexpected link at {destination}")
    elif destination.exists():
        raise FileExistsError(f"Refusing to replace {destination}")
    else:
        destination.symlink_to(target, target_is_directory=source.is_dir())


def bundle_ready(path):
    path = Path(path)
    try:
        sizes = read_json(path / "complete.json")["files"]
        return all(name in sizes for name in BUNDLE_FILES) and all(
            (path / name).is_file() and (path / name).stat().st_size == size and size > 0
            for name, size in sizes.items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def save_bundle(path, ss, slat, ss_decoder, mesh_decoder, sparse_cls, device, manifest, occupancy=None, image=None):
    import torch
    from data.dataset_scripts.generate_morphany3d_split_dataset import save_latent_triplet
    from eval_validation_latents import save_slat_glb

    path.mkdir(parents=True, exist_ok=True)
    (path / "complete.json").unlink(missing_ok=True)
    if slat is None or slat.feats.shape[0] == 0:
        raise RuntimeError(f"Empty SLat at {path}")
    if not torch.isfinite(ss).all() or not torch.isfinite(slat.feats).all():
        raise RuntimeError(f"Non-finite latents at {path}")
    adapter = SimpleNamespace(models={"sparse_structure_decoder": ss_decoder})
    save_latent_triplet(adapter, path, ss, slat, extra_manifest=manifest, occupancy=occupancy)
    temp_mesh = path / f".mesh.{uuid.uuid4().hex}.glb"
    try:
        if not save_slat_glb(mesh_decoder, sparse_cls, slat.feats, slat.coords, temp_mesh, device, "no"):
            raise RuntimeError(f"TRELLIS mesh extraction failed at {path}")
        temp_mesh.replace(path / "mesh.glb")
    finally:
        temp_mesh.unlink(missing_ok=True)
    files = list(BUNDLE_FILES)
    if image is not None:
        filename = "input" + Path(image).suffix.lower()
        shutil.copy2(image, path / filename)
        files.append(filename)
    write_json(path / "complete.json", {"files": {name: (path / name).stat().st_size for name in files}})


def prepare_layout(root, plan, version):
    base = root / version
    relative_link(root / "assets", base / "assets")
    for pair in plan["pairs"]:
        folder = base / pair["id"]
        for role, key in (("src1", "src_1"), ("src2", "src_2")):
            asset = root / "assets" / pair[key]
            relative_link(asset, folder / role)
            relative_link(asset / "mesh.glb", folder / f"{role}.glb")


def publish_version(root, plan, version):
    """Publish sequence and dataset-compatible metadata only after full validation."""
    rows = []
    for pair in plan["pairs"]:
        sequence = []
        for index, alpha in enumerate(plan["alphas"], 1):
            target = f"{pair['id']}/{step_name(index)}"
            folder = root / version / target
            if not bundle_ready(folder):
                raise RuntimeError(f"Incomplete {version} bundle: {folder}")
            relative_link(folder / "mesh.glb", folder / "pred_final.glb")
            row = {**pair, "split": "comparison", "alpha": alpha,
                   "alpha_definition": ALPHA_DEFINITION, "target": target,
                   "target_dir": target, "src1_dir": f"assets/{pair['src_1']}",
                   "src2_dir": f"assets/{pair['src_2']}",
                   "src1_image": f"assets/{pair['src_1']}/{plan['sources'][pair['src_1']]['input_file']}",
                   "src2_image": f"assets/{pair['src_2']}/{plan['sources'][pair['src_2']]['input_file']}"}
            for prefix, directory in (("src1", row["src1_dir"]), ("src2", row["src2_dir"]), ("target", target)):
                for field in ("slat_feats", "slat_coords", "ss_latent", "structured_latent", "occupancy"):
                    row[f"{prefix}_{field}"] = f"{directory}/{field}.pt"
            rows.append(row)
            write_json(folder / "metadata.json", row)
            sequence.append({"kind": "prediction", "alpha": alpha, "saved": True,
                             "path": f"{step_name(index)}/pred_final.glb"})
        write_json(root / version / pair["id"] / "sequence.json", sequence)
    write_json(root / version / "metadata.json", rows)


def teacher_phase(root, plan, cache_dir):
    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.modules.sparse.basic import SparseTensor
    from data.dataset_scripts import generate_morphany3d_split_dataset as teacher
    from eval_validation_latents import patch_trellis_mesh_dtype

    args = SimpleNamespace(**plan["config"])
    for method in ("sample_sparse_structure_morphing", "sample_slat_morphing"):
        if not hasattr(TrellisImageTo3DPipeline, method):
            raise RuntimeError("TRELLIS must come from the MorphAny3D checkout; missing " + method)
    prepare_layout(root, plan, "teacher")
    required = [root / "assets" / name for name in plan["sources"]] + [
        root / "teacher" / pair["id"] / step_name(i)
        for pair in plan["pairs"] for i in range(1, len(plan["alphas"]) + 1)
    ]
    if all(bundle_ready(path) for path in required):
        publish_version(root, plan, "teacher")
        return
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model_id)
    for key in ("slat_decoder_gs", "slat_decoder_rf"):
        pipeline.models.pop(key, None)
    pipeline.cuda()
    patch_trellis_mesh_dtype()
    mesh_decoder = pipeline.models["slat_decoder_mesh"]
    if hasattr(mesh_decoder, "convert_to_fp32"):
        mesh_decoder.convert_to_fp32()
    mesh_decoder.float()
    if hasattr(mesh_decoder, "dtype"):
        mesh_decoder.dtype = torch.float32
    ss_decoder = pipeline.models["sparse_structure_decoder"]
    sparse_params = {"steps": args.teacher_ss_steps, "cfg_strength": args.teacher_ss_cfg}
    slat_params = {"steps": args.teacher_slat_steps, "cfg_strength": args.teacher_slat_cfg}
    with torch.no_grad():
        for name, source in plan["sources"].items():
            folder = root / "assets" / name
            if bundle_ready(folder):
                continue
            print(f"[source] {name}", flush=True)
            cond = teacher.encode_image_condition(pipeline, Path(source["image"]))
            ss, slat = teacher.sample_endpoint_latents(pipeline, cond, args.seed, sparse_params, slat_params)
            save_bundle(folder, ss, slat, ss_decoder, mesh_decoder, SparseTensor, pipeline.device,
                        {"kind": "shared_source", "name": name, "seed": args.seed,
                         "original_image": source["image"], "image_sha256": source["sha256"]},
                        image=source["image"])
            del ss, slat, cond
            torch.cuda.empty_cache()

        cache_root = Path(cache_dir or os.environ.get("TMPDIR", str(root / ".cache")))
        cache_root.mkdir(parents=True, exist_ok=True)
        for pair in plan["pairs"]:
            folders = [root / "teacher" / pair["id"] / step_name(i)
                       for i in range(1, len(plan["alphas"]) + 1)]
            if all(bundle_ready(folder) for folder in folders):
                continue
            conditions = [teacher.encode_image_condition(pipeline, Path(plan["sources"][pair[key]]["image"]))
                          for key in ("src_1", "src_2")]
            # Restart the full temporal chain after interruption: TFSA needs the previous frame.
            with tempfile.TemporaryDirectory(prefix="comparison_tfsa_", dir=cache_root) as temp:
                work = Path(temp)
                tfsa_cache = {} if args.tfsa_cache_mode == "memory" else None
                cache_bytes = [0]
                limit = int(args.max_work_cache_gb * 1024**3) if args.max_work_cache_gb else None
                for index, (alpha, folder) in enumerate(zip(plan["alphas"], folders), 1):
                    print(f"[teacher] {pair['id']} {index}/{len(folders)} alpha={alpha:.9f}", flush=True)
                    ss, slat, occupancy = teacher.run_one_morphany3d_step(
                        pipeline, conditions[0], conditions[1], work, args.seed,
                        index, len(folders) + 2, alpha, args.tfsa_alpha, sparse_params, slat_params,
                        tfsa_cache=tfsa_cache, max_tfsa_cache_bytes=limit, tfsa_cache_bytes=cache_bytes,
                    )
                    if not bundle_ready(folder):
                        save_bundle(folder, ss, slat, ss_decoder, mesh_decoder, SparseTensor, pipeline.device,
                                    {**pair, "kind": "teacher", "alpha": alpha, "seed": args.seed,
                                     "morphing_idx": index, "morphing_num": len(folders) + 2,
                                     "alpha_definition": ALPHA_DEFINITION}, occupancy=occupancy)
                    del ss, slat, occupancy
                    teacher.cleanup_old_index(work, index - 1)
                    teacher.cleanup_old_tfsa_cache(tfsa_cache, index - 1, cache_bytes)
                    if args.tfsa_cache_mode == "file":
                        cache_bytes[0] = teacher.directory_size_bytes(work)
                    if limit is not None and cache_bytes[0] > limit:
                        raise RuntimeError(f"TFSA cache exceeded {args.max_work_cache_gb} GiB")
            del conditions, tfsa_cache
            torch.cuda.empty_cache()
    publish_version(root, plan, "teacher")


def student_phase(root, plan):
    import torch
    from data.dataset_scripts.generate_morphany3d_split_dataset import seed_everything
    from data.morph_dataset import MorphingDistillDataset
    import eval_validation_latents as inference
    from generate_alpha_steps import load_asset, batch_for_alpha

    args = SimpleNamespace(**plan["config"])
    for name in plan["sources"]:
        if not bundle_ready(root / "assets" / name):
            raise RuntimeError("Run the teacher stage first; missing source " + name)
    publish_version(root, plan, "teacher")
    prepare_layout(root, plan, "student")
    required = [root / "student" / pair["id"] / step_name(i)
                for pair in plan["pairs"] for i in range(1, len(plan["alphas"]) + 1)]
    if all(bundle_ready(path) for path in required):
        publish_version(root, plan, "student")
        return
    device = torch.device("cuda")
    precision = inference.resolve_mixed_precision(args.mixed_precision, device)
    models, needs_images = [], False
    for target, checkpoint_path in (("ss", args.checkpoint_path), ("slat", args.slat_checkpoint_path)):
        checkpoint = inference.load_checkpoint(checkpoint_path)
        if inference.detect_flow_target(checkpoint) != target:
            raise ValueError(f"Expected a {target} checkpoint: {checkpoint_path}")
        model_type = inference.detect_model_type(checkpoint, "auto")
        if model_type != "image_large":
            raise ValueError(f"The image teacher requires image_large checkpoints; got {model_type}")
        model = inference.build_model(checkpoint, model_type, target).to(device).eval()
        inference.preload_dino_if_needed(model, device)
        needs_images |= inference.checkpoint_requires_source_images(checkpoint, target)
        models.append(model)
        del checkpoint
    ss_model, slat_model = models
    ss_decoder, mesh_decoder, sparse_cls = inference.load_decoders("ss", device)
    dataset = MorphingDistillDataset(root=str(root / "teacher"), metadata_file="metadata.json",
                                    split=None, strict_split=False, skip_missing=False, verbose=False)
    with torch.no_grad():
        for pair_index, pair in enumerate(plan["pairs"]):
            src1, src2 = [load_asset(root, pair[key]) for key in ("src_1", "src_2")]
            if needs_images:
                entry = {"src_1": pair["src_1"], "src_2": pair["src_2"]}
                for role, src in (("src1", src1), ("src2", src2)):
                    entry[f"{role}_image"] = f"assets/{src['name']}/{plan['sources'][src['name']]['input_file']}"
                    src["image"] = dataset._load_source_image(src["name"], entry, role)
            ss_seed = (args.seed + 2 * pair_index) % 2**32
            slat_seed = (ss_seed + 1) % 2**32
            for index, alpha in enumerate(plan["alphas"], 1):
                folder = root / "student" / pair["id"] / step_name(index)
                if bundle_ready(folder):
                    continue
                print(f"[student] {pair['id']} {index}/{len(plan['alphas'])} alpha={alpha:.9f}", flush=True)
                batch = batch_for_alpha(src1, src2, alpha)
                seed_everything(ss_seed)
                ss = inference.sample_ss(ss_model, batch, torch.zeros_like(src1["ss_latent"]).to(device),
                                         args.steps, device, args.cfg_scale, precision)
                # The student's SLat uses ONLY coordinates predicted by its own SS stage.
                occupancy = inference.ss_logits_raw(ss_decoder, ss, device, "no") > 0
                coords = torch.argwhere(occupancy)[:, [0, 2, 3, 4]].int()
                seed_everything(slat_seed)
                slat = inference.sample_slat_on_coords(slat_model, batch, coords, args.slat_steps,
                                                      device, args.slat_cfg_scale, precision)
                save_bundle(folder, ss, slat, ss_decoder, mesh_decoder, sparse_cls, device,
                            {**pair, "kind": "student", "alpha": alpha, "ss_seed": ss_seed,
                             "slat_seed": slat_seed, "mixed_precision": precision,
                             "alpha_definition": ALPHA_DEFINITION}, occupancy=occupancy)
                del ss, slat, occupancy, coords, batch
            torch.cuda.empty_cache()
    publish_version(root, plan, "student")


def publish_comparison(root, plan):
    """Make the existing FID/KID evaluator's four-mesh layout without data copies."""
    rows = []
    for name in plan["sources"]:
        if not bundle_ready(root / "assets" / name):
            raise RuntimeError(f"Missing matched source asset: {name}")
    for pair in plan["pairs"]:
        for index, alpha in enumerate(plan["alphas"], 1):
            teacher = root / "teacher" / pair["id"] / step_name(index)
            student = root / "student" / pair["id"] / step_name(index)
            if not all(bundle_ready(path) for path in (teacher, student)):
                raise RuntimeError(f"Missing matched teacher/student sample: {pair['id']} / {index}")
            folder = root / "comparison" / f"{pair['id']}_{step_name(index)}"
            for source, name in ((root / "assets" / pair["src_1"] / "mesh.glb", "src1.glb"),
                                 (root / "assets" / pair["src_2"] / "mesh.glb", "src2.glb"),
                                 (teacher / "mesh.glb", "target.glb"), (student / "mesh.glb", "pred_final.glb")):
                relative_link(source, folder / name)
            row = {**pair, "src1": pair["src_1"], "src2": pair["src_2"], "alpha": alpha,
                   "teacher_dir": str(teacher.relative_to(root)), "student_dir": str(student.relative_to(root)),
                   "comparison_dir": str(folder.relative_to(root)), "alpha_definition": ALPHA_DEFINITION}
            write_json(folder / "metrics.json", row)
            rows.append(row)
    write_json(root / "comparison_manifest.json", {"status": "complete", "num_pairs": len(plan["pairs"]),
               "num_intermediates": len(plan["alphas"]), "num_matched_samples": len(rows), "samples": rows})


def worker(phase, plan_path, cache_dir):
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")
    if os.environ.get("TRELLIS_REPO"):
        sys.path.insert(0, os.environ["TRELLIS_REPO"])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU generation requires CUDA and the TRELLIS/MorphAny3D environment")
    plan_path = Path(plan_path)
    plan = read_json(plan_path)
    if phase == "teacher":
        teacher_phase(plan_path.parent, plan, cache_dir)
    elif phase == "student":
        student_phase(plan_path.parent, plan)
    else:
        raise ValueError(f"Invalid worker phase: {phase}")


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another comparison process is using {root}") from None
        if args.dry_run:
            write_json(root / "plan.dry_run.json", plan)
            print(f"Dry run: {len(plan['pairs'])} unseen pairs, {len(plan['alphas'])} intermediates each")
            print(f"Available pairs: {plan['available_pair_count']}; excluded: {plan['excluded_pair_count']}")
            print(f"Alphas (src1 -> src2): {plan['alphas']}")
            print(f"Plan: {root / 'plan.dry_run.json'}")
            return
        plan_path = root / "plan.json"
        if plan_path.exists():
            if not args.resume:
                raise FileExistsError(f"{plan_path} already exists; use --resume or a new output directory")
            if read_json(plan_path) != plan:
                raise ValueError("Resume plan differs: inputs, checkpoints, exclusions or settings changed. Use a new output directory.")
        else:
            unexpected = [p for p in root.iterdir() if p.name not in {".run.lock", "plan.dry_run.json"}]
            if unexpected:
                raise FileExistsError(f"Output directory is not empty: {root}")
            write_json(plan_path, plan)
        phases = ("teacher", "student") if args.stage == "all" else (args.stage,)
        (root / "comparison_manifest.json").unlink(missing_ok=True)
        write_json(root / "status.json", {"status": "running", "stage": args.stage})
        try:
            for phase in phases:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "_worker", phase,
                                str(plan_path), args.work_cache_dir or ""], check=True)
            if "student" in phases:
                publish_comparison(root, plan)
            write_json(root / "status.json", {"status": "complete" if "student" in phases else "teacher_complete"})
        except Exception as exc:
            write_json(root / "status.json", {"status": "failed", "stage": args.stage, "error": str(exc)})
            raise
        print(f"Finished: {root}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "_worker":
        worker(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()
