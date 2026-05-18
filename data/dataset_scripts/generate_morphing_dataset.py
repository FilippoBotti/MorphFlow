"""
Generate a morphing dataset in resumable stages.

Stages:
- assets: generate per-asset SLAT + cache
- pairs: generate midpoint SLAT for valid unordered pairs
- all: run assets then pairs

Key guarantees:
- Resume-safe: already completed items are skipped.
- Pair cap-safe: an asset never exceeds PAIRS_PER_ASSET completed pairs,
  including pairs already present on disk from previous/interrupted runs.
"""

import argparse
import json
import os
import random
from glob import glob

# Keep defaults compatible with previous scripts but allow override from shell.
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils.morphing_utils import seed_everything

DEFAULT_MODEL_ID = "microsoft/TRELLIS-image-large"
DEFAULT_SEED = 0
DEFAULT_ASSETS_DIR = "/home/filippo/datasets/3d/flux_outputs"
DEFAULT_OUTPUT_DIR = "/home/filippo/datasets/3d/morphing_dataset_flux"
DEFAULT_MORPHING_ALPHA = 0.5
DEFAULT_MORPHING_NUM = 3
DEFAULT_PAIRS_PER_ASSET = 50
REQUIRED_CACHE_FILES = ("slat_init.pt", "coords.pt", "coords_zs_init.pt")


def load_image_names(assets_dir):
    image_names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(assets_dir)
        if f.lower().endswith(".png")
    )
    if len(image_names) < 2:
        raise ValueError(f"Need at least 2 png assets in {assets_dir}, found {len(image_names)}")
    return image_names


def asset_paths(output_dir, name):
    asset_dir = os.path.join(output_dir, "assets", name)
    cache_dir = os.path.join(asset_dir, "cache")
    return {
        "asset_dir": asset_dir,
        "cache_dir": cache_dir,
        "feats": os.path.join(asset_dir, "slat_feats.pt"),
        "coords": os.path.join(asset_dir, "slat_coords.pt"),
    }


def is_asset_ready(output_dir, name):
    p = asset_paths(output_dir, name)
    if not (os.path.exists(p["feats"]) and os.path.exists(p["coords"])):
        return False
    return all(os.path.exists(os.path.join(p["cache_dir"], f)) for f in REQUIRED_CACHE_FILES)


def pair_paths(output_dir, src_name, tar_name):
    pair_key = tuple(sorted((src_name, tar_name)))
    pair_name = f"{pair_key[0]}+{pair_key[1]}"
    pair_dir = os.path.join(output_dir, "pairs", pair_name)
    return {
        "pair_key": pair_key,
        "pair_name": pair_name,
        "pair_dir": pair_dir,
        "cache_dir": os.path.join(pair_dir, "cache"),
        "feats": os.path.join(pair_dir, "mid_slat_feats.pt"),
        "coords": os.path.join(pair_dir, "mid_slat_coords.pt"),
    }


def is_pair_ready(output_dir, src_name, tar_name):
    p = pair_paths(output_dir, src_name, tar_name)
    return os.path.exists(p["feats"]) and os.path.exists(p["coords"])


def parse_completed_pairs(output_dir, valid_names):
    valid_names = set(valid_names)
    pairs_root = os.path.join(output_dir, "pairs")
    completed_pairs = set()

    if not os.path.isdir(pairs_root):
        return completed_pairs

    for pair_name in os.listdir(pairs_root):
        if "+" not in pair_name:
            continue
        src_name, tar_name = pair_name.split("+", 1)
        if src_name not in valid_names or tar_name not in valid_names or src_name == tar_name:
            continue
        p = pair_paths(output_dir, src_name, tar_name)
        if os.path.exists(p["feats"]) and os.path.exists(p["coords"]):
            completed_pairs.add(p["pair_key"])

    return completed_pairs


def compute_degree(image_names, pairs):
    degree = {name: 0 for name in image_names}
    for src_name, tar_name in pairs:
        if src_name in degree:
            degree[src_name] += 1
        if tar_name in degree:
            degree[tar_name] += 1
    return degree


def build_additional_pairs(image_names, pairs_per_asset, existing_pairs, seed):
    """
    Build only the missing unordered pairs while respecting current completed degree.

    existing_pairs are treated as fixed/committed, so no asset can exceed
    pairs_per_asset after adding new pairs.
    """
    rng = random.Random(seed)
    max_possible_per_asset = max(0, len(image_names) - 1)
    target_pairs_per_asset = min(pairs_per_asset, max_possible_per_asset)

    used_pairs = set(existing_pairs)
    degree = compute_degree(image_names, used_pairs)
    new_pairs = []

    shuffled_assets = image_names[:]
    rng.shuffle(shuffled_assets)

    made_progress = True
    while made_progress:
        made_progress = False

        for src_name in shuffled_assets:
            if degree[src_name] >= target_pairs_per_asset:
                continue

            candidates = [
                tar_name
                for tar_name in image_names
                if tar_name != src_name
                and degree[tar_name] < target_pairs_per_asset
                and tuple(sorted((src_name, tar_name))) not in used_pairs
            ]
            rng.shuffle(candidates)

            for tar_name in candidates:
                if degree[src_name] >= target_pairs_per_asset:
                    break
                if degree[tar_name] >= target_pairs_per_asset:
                    continue

                pair_key = tuple(sorted((src_name, tar_name)))
                if pair_key in used_pairs:
                    continue

                used_pairs.add(pair_key)
                degree[src_name] += 1
                degree[tar_name] += 1
                new_pairs.append(pair_key)
                made_progress = True

    return new_pairs, target_pairs_per_asset


def generate_asset_slat(pipeline, img, partner_img, seed, cache_dir):
    seed_everything(seed)
    with torch.no_grad():
        src_processed = pipeline.preprocess_image(img)
        src_cond = pipeline.get_cond([src_processed])
        tar_processed = pipeline.preprocess_image(partner_img)
        tar_cond = pipeline.get_cond([tar_processed])

        morphing_params = {
            "save_cache_path": cache_dir,
            "init_morphing_flag": False,
            "ss_mca_flag": False,
            "slat_mca_flag": False,
            "ss_tfsa_flag": False,
            "slat_tfsa_flag": False,
            "oc_flag": False,
            "tar_cond": tar_cond["cond"],
        }

        torch.manual_seed(seed)
        coords, _voxels, _z_s = pipeline.sample_sparse_structure_morphing(src_cond, 1, {}, morphing_params)
        slat = pipeline.sample_slat_morphing(src_cond, coords, {}, morphing_params)
    return slat


def generate_morphed_slat(
    pipeline,
    src_img,
    tar_img,
    src_cache,
    tar_cache,
    pair_cache,
    seed,
    alpha,
    morphing_num,
):
    seed_everything(seed)
    with torch.no_grad():
        src_processed = pipeline.preprocess_image(src_img)
        src_cond = pipeline.get_cond([src_processed])
        tar_processed = pipeline.preprocess_image(tar_img)
        tar_cond = pipeline.get_cond([tar_processed])

        morphing_params = {
            "save_cache_path": pair_cache,
            "src_load_cache_path": src_cache,
            "tar_load_cache_path": tar_cache,
            "init_morphing_flag": False,
            "ss_mca_flag": True,
            "slat_mca_flag": True,
            "ss_tfsa_flag": True,
            "slat_tfsa_flag": True,
            "oc_flag": True,
            "tar_cond": tar_cond["cond"],
            "alpha": alpha,
            "morphing_idx": 1,
            "tfsa_cache_idx": 0,
            "tfsa_alpha": 0.8,
            "morphing_num": morphing_num,
        }

        torch.manual_seed(seed)
        coords, _voxels, _z_s = pipeline.sample_sparse_structure_morphing(src_cond, 1, {}, morphing_params)
        slat = pipeline.sample_slat_morphing(src_cond, coords, {}, morphing_params)
    return slat


def write_metadata(output_dir, assets_dir, pairs, alpha):
    metadata = []
    for src_name, tar_name in sorted(pairs):
        pair_name = f"{src_name}+{tar_name}"
        metadata.append(
            {
                "src": src_name,
                "tar": tar_name,
                "src_image": os.path.join(assets_dir, f"{src_name}.png"),
                "tar_image": os.path.join(assets_dir, f"{tar_name}.png"),
                "src_slat_dir": f"assets/{src_name}",
                "tar_slat_dir": f"assets/{tar_name}",
                "mid_slat_dir": f"pairs/{pair_name}",
                "alpha": alpha,
            }
        )

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return meta_path, len(metadata)


def run_assets_stage(pipeline, image_names, assets_dir, output_dir, seed):
    print("\n=== Stage assets: Generating per-asset SLAT + cache ===")
    for i, name in enumerate(image_names):
        p = asset_paths(output_dir, name)
        os.makedirs(p["cache_dir"], exist_ok=True)

        if is_asset_ready(output_dir, name):
            print(f"  [{i + 1}/{len(image_names)}] {name} - skipped (ready)")
            continue

        img = Image.open(os.path.join(assets_dir, f"{name}.png"))
        partner_name = image_names[(i + 1) % len(image_names)]
        partner_img = Image.open(os.path.join(assets_dir, f"{partner_name}.png"))

        slat = generate_asset_slat(pipeline, img, partner_img, seed, p["cache_dir"])
        torch.save(slat.feats.cpu(), p["feats"])
        torch.save(slat.coords.cpu(), p["coords"])

        print(
            f"  [{i + 1}/{len(image_names)}] {name} "
            f"feats={list(slat.feats.shape)} coords={list(slat.coords.shape)}"
        )


def run_pairs_stage(
    pipeline,
    image_names,
    assets_dir,
    output_dir,
    seed,
    pairs_per_asset,
    morphing_alpha,
    morphing_num,
):
    existing_pairs = parse_completed_pairs(output_dir, image_names)
    degree = compute_degree(image_names, existing_pairs)

    new_pairs, target_pairs_per_asset = build_additional_pairs(
        image_names=image_names,
        pairs_per_asset=pairs_per_asset,
        existing_pairs=existing_pairs,
        seed=seed,
    )

    saturated_assets = sum(1 for v in degree.values() if v >= target_pairs_per_asset)
    print("\n=== Stage pairs: Generating midpoint pair SLATs ===")
    print(
        "Existing completed pairs: "
        f"{len(existing_pairs)} | Planned new pairs: {len(new_pairs)} | "
        f"Cap per asset: {target_pairs_per_asset} | Assets at cap: {saturated_assets}/{len(image_names)}"
    )

    for idx, (src_name, tar_name) in enumerate(new_pairs):
        p = pair_paths(output_dir, src_name, tar_name)
        os.makedirs(p["cache_dir"], exist_ok=True)

        if is_pair_ready(output_dir, src_name, tar_name):
            print(f"  [{idx + 1}/{len(new_pairs)}] {p['pair_name']} - skipped (ready)")
            continue

        if not is_asset_ready(output_dir, src_name) or not is_asset_ready(output_dir, tar_name):
            print(f"  [{idx + 1}/{len(new_pairs)}] {p['pair_name']} - skipped (missing asset cache/slat)")
            continue

        src_img = Image.open(os.path.join(assets_dir, f"{src_name}.png"))
        tar_img = Image.open(os.path.join(assets_dir, f"{tar_name}.png"))
        src_cache = asset_paths(output_dir, src_name)["cache_dir"]
        tar_cache = asset_paths(output_dir, tar_name)["cache_dir"]

        slat = generate_morphed_slat(
            pipeline=pipeline,
            src_img=src_img,
            tar_img=tar_img,
            src_cache=src_cache,
            tar_cache=tar_cache,
            pair_cache=p["cache_dir"],
            seed=seed,
            alpha=morphing_alpha,
            morphing_num=morphing_num,
        )

        torch.save(slat.feats.cpu(), p["feats"])
        torch.save(slat.coords.cpu(), p["coords"])

        for cache_pt in glob(os.path.join(p["cache_dir"], "*.pt")):
            os.remove(cache_pt)

        print(
            f"  [{idx + 1}/{len(new_pairs)}] {p['pair_name']} "
            f"feats={list(slat.feats.shape)} coords={list(slat.coords.shape)}"
        )

    completed_pairs = parse_completed_pairs(output_dir, image_names)
    meta_path, count = write_metadata(
        output_dir=output_dir,
        assets_dir=assets_dir,
        pairs=completed_pairs,
        alpha=morphing_alpha,
    )
    print(f"\nMetadata updated: {count} completed pairs -> {meta_path}")


def build_parser(default_stage="all"):
    parser = argparse.ArgumentParser(description="Generate TRELLIS morphing dataset in resumable stages")
    parser.add_argument("--stage", choices=["assets", "pairs", "all"], default=default_stage)
    parser.add_argument("--cuda-device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--assets-dir", default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--morphing-alpha", type=float, default=DEFAULT_MORPHING_ALPHA)
    parser.add_argument("--morphing-num", type=int, default=DEFAULT_MORPHING_NUM)
    parser.add_argument("--pairs-per-asset", type=int, default=DEFAULT_PAIRS_PER_ASSET)
    return parser


def main(default_stage="all"):
    parser = build_parser(default_stage=default_stage)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model_id)
    pipeline.cuda()

    image_names = load_image_names(args.assets_dir)
    print(f"Found {len(image_names)} assets in {args.assets_dir}")

    if args.stage in ("assets", "all"):
        run_assets_stage(
            pipeline=pipeline,
            image_names=image_names,
            assets_dir=args.assets_dir,
            output_dir=args.output_dir,
            seed=args.seed,
        )

    if args.stage in ("pairs", "all"):
        run_pairs_stage(
            pipeline=pipeline,
            image_names=image_names,
            assets_dir=args.assets_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            pairs_per_asset=args.pairs_per_asset,
            morphing_alpha=args.morphing_alpha,
            morphing_num=args.morphing_num,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
