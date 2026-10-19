#!/usr/bin/env python3
"""Index official Interp3DData image pairs and cache canonical TRELLIS endpoints.

--index-only reads PNGs and writes evaluation metadata using CPU only. --prepare
also samples each unique endpoint once with the installed Interp3D TRELLIS fork.
It requires one allocated GPU and never generates interpolation targets or GLBs.
Endpoint images, latents, seeds and sampling settings are shared by every method,
CFG cell and evaluation protocol. The prepared dataset is immutable and resumable.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import uuid


DIFFICULTIES = ("easy", "mid", "hard")
MODEL_ID = "microsoft/TRELLIS-image-large"
SOURCE_URL = "https://drive.google.com/file/d/1Y-J_7KwEk8VFy_v19cF3wCgbTW8Pqff0/view"
REQUIRED_LATENTS = ("ss_latent.pt", "slat_feats.pt", "slat_coords.pt")
SCHEMA_VERSION = 1


class PreparationError(ValueError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                             encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise PreparationError("Prepared dataset input/settings changed: {}. "
                                   "Use a new --output-root.".format(path))
        return
    atomic_json(path, value)


def endpoint_seed(seed, asset_name):
    value = "interp3ddata-endpoint|{}|{}".format(seed, asset_name).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2 ** 31 - 1)


def discover_pairs(input_root):
    """Preserve the authors' difficulty/case/0->1 ordering, including digit names."""
    root = Path(input_root).expanduser().resolve()
    if (root / "all_cases_morphing").is_dir():
        root = root / "all_cases_morphing"
    if not root.is_dir():
        raise PreparationError("Missing extracted Interp3DData directory: {}".format(root))
    pairs = []
    for difficulty in DIFFICULTIES:
        directory = root / difficulty
        if not directory.is_dir():
            raise PreparationError("Missing difficulty directory: {}".format(directory))
        cases = {}
        for path in sorted(directory.iterdir()):
            if path.name.startswith("."):
                continue
            if path.suffix.lower() != ".png":
                raise PreparationError("Unexpected entry in {}: {}".format(directory, path.name))
            if not path.is_file() or path.is_symlink():
                raise PreparationError("Expected a regular source PNG: {}".format(path))
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)([01])\.png", path.name)
            if not match:
                raise PreparationError("Source image must end in 0.png or 1.png: {}".format(path))
            case, endpoint = match[1], int(match[2])
            cases.setdefault(case, {})[endpoint] = path
        if not cases:
            raise PreparationError("No source image pairs in {}".format(directory))
        for case in sorted(cases):
            if set(cases[case]) != {0, 1}:
                raise PreparationError("Incomplete source pair {}/{}: expected 0.png and 1.png".format(
                    difficulty, case))
            pairs.append({"difficulty": difficulty, "case": case,
                          "paths": [cases[case][0], cases[case][1]]})
    return root, pairs


def generation_settings(seed, steps, slat_steps, cfg_scale, slat_cfg_scale, model_id):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PreparationError("--seed must be a nonnegative integer")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (steps, slat_steps)):
        raise PreparationError("Endpoint step counts must be positive integers")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in (cfg_scale, slat_cfg_scale)):
        raise PreparationError("Endpoint CFG values must be finite and nonnegative")
    return {"model_id": str(model_id), "seed": seed, "steps": steps, "slat_steps": slat_steps,
            "cfg_scale": float(cfg_scale), "slat_cfg_scale": float(slat_cfg_scale),
            "seed_policy": "SHA256(interp3ddata-endpoint|base_seed|asset_name), first 8 bytes little-endian modulo 2**31-1",
            "preprocessing": "Interp3D TrellisImageTo3DPipeline.preprocess_image on the original source PNG",
            "slat_storage": "denormalized decoder-space features returned by pipeline.sample_slat",
            "sampler_defaults": "Other sampler parameters retain the pinned installed TRELLIS pretrained configuration"}


def index_dataset(input_root, output_root, seed=42, steps=25, slat_steps=25,
                  cfg_scale=5.0, slat_cfg_scale=5.0, model_id=MODEL_ID):
    """Create stable evaluation-only metadata; intermediate target assets are absent."""
    from PIL import Image

    raw_root, discovered = discover_pairs(input_root)
    output_root = Path(output_root).expanduser().resolve()
    settings = generation_settings(seed, steps, slat_steps, cfg_scale, slat_cfg_scale, model_id)
    assets, pairs, metadata = [], [], []
    for pair_id, pair in enumerate(discovered):
        asset_names = []
        for endpoint, path in enumerate(pair["paths"]):
            name = pair["difficulty"] + "__" + path.stem
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise PreparationError("Source is not a PNG: {}".format(path))
                size, mode = list(image.size), image.mode
                image.verify()
            asset_names.append(name)
            assets.append({"name": name, "difficulty": pair["difficulty"], "case": pair["case"],
                           "endpoint": endpoint, "source_path": path.relative_to(raw_root).as_posix(),
                           "image": "images/" + name + ".png", "sha256": digest(path),
                           "bytes": path.stat().st_size, "size": size, "mode": mode,
                           "seed": endpoint_seed(seed, name)})
        identifier = pair["difficulty"] + "__" + pair["case"]
        pairs.append({"pair_id": pair_id, "case_id": identifier, "difficulty": pair["difficulty"],
                      "case": pair["case"], "src_1": asset_names[0], "src_2": asset_names[1]})
        # The evaluator uses source inputs only; this identifier is not a target GT.
        metadata.append({"split": "test", "src_1": asset_names[0], "src_2": asset_names[1],
                         "target": "evaluation_only/" + identifier, "alpha": 0.5,
                         "src1_image": "images/" + asset_names[0] + ".png",
                         "src2_image": "images/" + asset_names[1] + ".png",
                         "difficulty": pair["difficulty"], "case": pair["case"],
                         "case_id": identifier, "evaluation_only": True})
    if len({asset["name"] for asset in assets}) != len(assets):
        raise PreparationError("Source asset name collision in input dataset")
    counts = {difficulty: sum(pair["difficulty"] == difficulty for pair in pairs)
              for difficulty in DIFFICULTIES}
    provenance = {"schema_version": SCHEMA_VERSION, "dataset": "Interp3DData", "source_url": SOURCE_URL,
                  "input_root": str(raw_root), "pair_order": "difficulty easy,mid,hard; case lexicographic; source0 -> source1",
                  "alpha_convention": "alpha=1 -> source0/src1; alpha=0 -> source1/src2",
                  "num_pairs": len(pairs), "num_assets": len(assets), "difficulty_pairs": counts,
                  "endpoint_generation": settings, "assets": assets, "pairs": pairs,
                  "evaluation_note": "Official Interp3DData source image pairs; canonical endpoints are cached once from TRELLIS, not intermediate ground truth"}
    provenance_path = output_root / "dataset_provenance.json"
    if provenance_path.exists() and read_json(provenance_path) != provenance:
        raise PreparationError("Images, source pairs, endpoint seeds or sampling parameters changed. "
                               "Use a new --output-root instead of reusing {}.".format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        source = raw_root / asset["source_path"]
        destination = output_root / asset["image"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.is_file() or digest(destination) != asset["sha256"]:
                raise PreparationError("Prepared source PNG changed: {}".format(destination))
        else:
            temporary = destination.with_name("." + destination.name + "." + uuid.uuid4().hex + ".tmp")
            try:
                shutil.copyfile(source, temporary)
                if digest(temporary) != asset["sha256"]:
                    raise PreparationError("Source image changed during copy: {}".format(source))
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    write_immutable_json(output_root / "metadata_test.json", metadata)
    write_immutable_json(output_root / "split_assets.json", {"train": [], "val": [], "test": [asset["name"] for asset in assets]})
    write_immutable_json(provenance_path, provenance)
    print("INDEX READY: {} pairs {}; {} source images; {}".format(len(pairs), counts, len(assets), output_root), flush=True)
    return provenance


def asset_ready(output_root, asset, settings):
    root = Path(output_root) / "assets" / asset["name"]
    marker = root / "endpoint_ready.json"
    if not marker.is_file():
        return False
    try:
        record = read_json(marker)
        if record.get("schema_version") != SCHEMA_VERSION or record.get("asset") != asset or record.get("settings") != settings:
            raise PreparationError("Endpoint cache identity/settings changed: {}".format(marker))
        for name in REQUIRED_LATENTS:
            path = root / name
            expected = record["files"][name]
            if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
                return False
        return True
    except (KeyError, TypeError, json.JSONDecodeError):
        return False


def atomic_tensor(path, tensor, torch):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(tensor, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_endpoints(output_root, pipeline_factory=None):
    """GPU endpoint sampling, with a test seam for a tiny CPU fake pipeline."""
    from PIL import Image

    output_root = Path(output_root).expanduser().resolve()
    provenance = read_json(output_root / "dataset_provenance.json")
    settings = provenance["endpoint_generation"]
    pending = [asset for asset in provenance["assets"] if not asset_ready(output_root, asset, settings)]
    if not pending:
        print("ENDPOINTS READY: {} cached assets; no sampling needed".format(provenance["num_assets"]), flush=True)
        return
    import torch

    if pipeline_factory is None:
        from data.dataset_scripts.generate_interp3d_v4 import prepare_pipeline
        pipeline_factory = prepare_pipeline
    pipeline = pipeline_factory(settings["model_id"])
    ss_params = {"steps": settings["steps"], "cfg_strength": settings["cfg_scale"]}
    slat_params = {"steps": settings["slat_steps"], "cfg_strength": settings["slat_cfg_scale"]}
    actual_samplers = {"ss": {**pipeline.sparse_structure_sampler_params, **ss_params},
                       "slat": {**pipeline.slat_sampler_params, **slat_params}}
    with torch.inference_mode():
        for index, asset in enumerate(pending):
            path = output_root / asset["image"]
            if digest(path) != asset["sha256"]:
                raise PreparationError("Prepared input image changed: {}".format(path))
            torch.manual_seed(asset["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(asset["seed"])
            with Image.open(path) as image:
                processed = pipeline.preprocess_image(image.copy())
            conditioning = pipeline.get_cond([processed])
            endpoint = pipeline.sample_sparse_structure(conditioning, None, 1, ss_params)
            if not isinstance(endpoint, (tuple, list)) or len(endpoint) != 2:
                raise PreparationError("Use the Interp3D TRELLIS fork: sample_sparse_structure must return (coords, ss_latent)")
            coords, ss = endpoint
            slat = pipeline.sample_slat(conditioning, coords, slat_params)
            if (ss.ndim != 5 or ss.shape[0] != 1 or slat.feats.ndim != 2 or not len(slat.feats)
                    or slat.coords.ndim != 2 or slat.coords.shape != (slat.feats.shape[0], 4)
                    or not torch.isfinite(ss).all() or not torch.isfinite(slat.feats).all()
                    or torch.any(slat.coords[:, 0] != 0)):
                raise PreparationError("Invalid/empty/non-finite endpoint latents for {}".format(asset["name"]))
            root = output_root / "assets" / asset["name"]
            root.mkdir(parents=True, exist_ok=True)
            marker = root / "endpoint_ready.json"
            marker.unlink(missing_ok=True)
            atomic_tensor(root / "ss_latent.pt", ss.detach().cpu(), torch)
            atomic_tensor(root / "slat_feats.pt", slat.feats.detach().cpu(), torch)
            atomic_tensor(root / "slat_coords.pt", slat.coords.detach().cpu().to(torch.int32), torch)
            files = {name: {"bytes": (root / name).stat().st_size, "sha256": digest(root / name)}
                     for name in REQUIRED_LATENTS}
            atomic_json(marker, {"schema_version": SCHEMA_VERSION, "asset": asset, "settings": settings,
                                 "sampler_params": actual_samplers, "files": files})
            print("ENDPOINT {}/{} READY {} seed={}".format(index + 1, len(pending), asset["name"], asset["seed"]), flush=True)
            del conditioning, endpoint, coords, ss, slat, processed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print("ENDPOINTS READY: {} canonical assets".format(provenance["num_assets"]), flush=True)


@contextmanager
def preparation_lock(output_root):
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".dataset_prepare.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PreparationError("Another process is preparing {}".format(output_root)) from error
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", required=True, help="Extracted Interp3DData directory, containing all_cases_morphing or easy/mid/hard")
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parents[3] / "datasets/interp3ddata_prepared"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--index-only", action="store_true", help="CPU: validate original pairs/images and write stable evaluation metadata")
    mode.add_argument("--prepare", action="store_true", help="Index and sample missing canonical endpoints on one allocated GPU with the Interp3D env/fork")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--slat-steps", type=int, default=25)
    parser.add_argument("--cfg-scale", type=float, default=5.0, help="Canonical endpoint SS CFG; independent of MorphFlow evaluation CFG grid; pretrained default 5")
    parser.add_argument("--slat-cfg-scale", type=float, default=5.0, help="Canonical endpoint SLat CFG; pretrained default 5")
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args(argv)
    try:
        with preparation_lock(args.output_root):
            index_dataset(args.input_root, args.output_root, args.seed, args.steps, args.slat_steps,
                          args.cfg_scale, args.slat_cfg_scale, args.model_id)
            if args.prepare:
                prepare_endpoints(args.output_root)
    except (PreparationError, OSError, ValueError) as error:
        parser.exit(1, "ERROR: {}\n".format(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
