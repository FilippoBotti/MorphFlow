"""
Create a validation split from the tail of the asset list.

Goal:
- pick 200 validation samples from entries that use the last assets in sorted order,
  so validation data differs from the usual early-asset training region.

Outputs:
- validation metadata json (default: metadata_val_200_tail.json)
- optional train metadata json with validation assets removed
"""

import argparse
import json
import os


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_sorted_assets(assets_root):
    names = sorted(
        name
        for name in os.listdir(assets_root)
        if os.path.isdir(os.path.join(assets_root, name))
    )
    if len(names) < 2:
        raise RuntimeError(f"Need at least 2 asset folders in {assets_root}, found {len(names)}")
    return names


def valid_entry(entry):
    return all(k in entry for k in ("src_1", "src_2", "target", "alpha"))


def has_target_files(root_dir, target_name):
    target_dir = os.path.join(root_dir, "pairs_2", target_name)
    required = (
        os.path.join(target_dir, "mid_slat_feats.pt"),
        os.path.join(target_dir, "mid_slat_coords.pt"),
        os.path.join(target_dir, "mid_sparse_structure_latent.pt"),
    )
    return all(os.path.exists(p) for p in required)


def count_tail_candidates(metadata, tail_assets, root_dir, check_files):
    tail = set(tail_assets)
    count = 0
    for entry in metadata:
        if entry["src_1"] not in tail or entry["src_2"] not in tail:
            continue
        if check_files and not has_target_files(root_dir, entry["target"]):
            continue
        count += 1
    return count


def choose_tail_assets(metadata, sorted_assets, root_dir, val_size, check_files):
    # Pick the smallest tail that already contains enough validation candidates.
    for k in range(2, len(sorted_assets) + 1):
        tail_assets = sorted_assets[-k:]
        num = count_tail_candidates(metadata, tail_assets, root_dir, check_files)
        if num >= val_size:
            return tail_assets

    raise RuntimeError(
        "Unable to find enough validation candidates from tail assets. "
        f"Requested val_size={val_size}. Try lowering --val-size or disabling --check-files."
    )


def build_validation_entries(metadata, sorted_assets, tail_assets, root_dir, val_size, check_files):
    tail_set = set(tail_assets)
    rank = {name: idx for idx, name in enumerate(sorted_assets)}

    candidates = []
    for entry in metadata:
        if entry["src_1"] not in tail_set or entry["src_2"] not in tail_set:
            continue
        if check_files and not has_target_files(root_dir, entry["target"]):
            continue
        candidates.append(entry)

    # Prioritize pairs made of the latest assets.
    candidates.sort(
        key=lambda e: (
            rank.get(e["src_1"], -1) + rank.get(e["src_2"], -1),
            rank.get(e["src_1"], -1),
            rank.get(e["src_2"], -1),
            e["target"],
        ),
        reverse=True,
    )

    if len(candidates) < val_size:
        raise RuntimeError(
            f"Not enough tail candidates after filtering: {len(candidates)} < {val_size}"
        )

    return candidates[:val_size]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate validation split from tail assets")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/home/filippo/datasets/3d/morphing_dataset_flux",
    )
    parser.add_argument("--input-metadata", type=str, default="metadata_2.json")
    parser.add_argument("--val-output-metadata", type=str, default="metadata_val_200_tail.json")
    parser.add_argument("--train-output-metadata", type=str, default="metadata_train_wo_val_assets.json")
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument(
        "--exclude-val-assets-from-train",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, write train metadata excluding all entries that use validation assets.",
    )
    parser.add_argument(
        "--check-files",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, keep only entries whose target files exist under pairs_2.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.val_size <= 0:
        raise ValueError("--val-size must be > 0")

    input_meta_path = os.path.join(args.root_dir, args.input_metadata)
    val_meta_path = os.path.join(args.root_dir, args.val_output_metadata)
    train_meta_path = os.path.join(args.root_dir, args.train_output_metadata)

    metadata = load_json(input_meta_path)
    metadata = [entry for entry in metadata if valid_entry(entry)]

    sorted_assets = load_sorted_assets(os.path.join(args.root_dir, "assets"))
    tail_assets = choose_tail_assets(
        metadata=metadata,
        sorted_assets=sorted_assets,
        root_dir=args.root_dir,
        val_size=args.val_size,
        check_files=bool(args.check_files),
    )

    val_entries = build_validation_entries(
        metadata=metadata,
        sorted_assets=sorted_assets,
        tail_assets=tail_assets,
        root_dir=args.root_dir,
        val_size=args.val_size,
        check_files=bool(args.check_files),
    )
    save_json(val_meta_path, val_entries)

    val_assets = sorted({entry["src_1"] for entry in val_entries} | {entry["src_2"] for entry in val_entries})
    train_entries = metadata
    if args.exclude_val_assets_from_train == 1:
        train_entries = [
            entry for entry in metadata
            if entry["src_1"] not in val_assets and entry["src_2"] not in val_assets
        ]
        save_json(train_meta_path, train_entries)

    print("Validation split generated successfully.")
    print(f"Input metadata: {input_meta_path}")
    print(f"Validation metadata: {val_meta_path}")
    print(f"Validation samples: {len(val_entries)}")
    print(f"Tail assets used: {len(tail_assets)}")
    print(f"Tail span: {tail_assets[0]} -> {tail_assets[-1]}")

    if args.exclude_val_assets_from_train == 1:
        print(f"Train metadata (val assets excluded): {train_meta_path}")
        print(f"Train samples after exclusion: {len(train_entries)}")

    print("Suggested training flags:")
    print(f"  --metadata {os.path.basename(train_meta_path)}")
    print(f"  --val_metadata {os.path.basename(val_meta_path)}")


if __name__ == "__main__":
    main()
