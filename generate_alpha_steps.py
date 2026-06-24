import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np
import torch

if os.environ.get("TRELLIS_REPO"):
    sys.path.append(os.environ["TRELLIS_REPO"])

from data.morph_dataset import MorphingDistillDataset
from eval_validation_latents import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    build_model,
    checkpoint_args,
    checkpoint_requires_source_images,
    detect_flow_target,
    detect_model_type,
    detect_slat_condition_source,
    ensure_batch_coords,
    load_checkpoint,
    load_decoders,
    preload_dino_if_needed,
    safe_slug,
    sample_slat_on_coords,
    sample_ss,
    save_slat_glb,
    save_voxel_glb,
    ss_coords_from_latent,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/alpha_steps"


def parse_alpha_list(value: str):
    raw = []
    for chunk in value.replace(",", " ").split():
        if chunk.strip():
            raw.append(float(chunk))
    if not raw:
        raise ValueError("--alphas must contain at least one value")
    for alpha in raw:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"alpha values must be in [0, 1], got {alpha}")
    return raw


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an alpha sweep between two dataset assets with a trained MorphFlow SS checkpoint."
    )
    parser.add_argument("--root_dir", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--metadata", type=str, default="metadata_test.json")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint_path", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--slat_checkpoint_path",
        type=str,
        default=None,
        help="Optional second-flow SLat checkpoint used to decode final meshes from generated SS coordinates.",
    )
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--src1_index", type=int, required=True)
    parser.add_argument("--src2_index", type=int, required=True)
    parser.add_argument(
        "--index_unit",
        type=str,
        choices=["asset", "metadata_src"],
        default="asset",
        help=(
            "asset: indices address the sorted unique asset list in the selected metadata. "
            "metadata_src: src1_index/src2_index address metadata rows and use their src_1 asset."
        ),
    )
    parser.add_argument("--src1_name", type=str, default=None)
    parser.add_argument("--src2_name", type=str, default=None)
    parser.add_argument("--alphas", type=str, required=True, help="Comma or space separated alpha values, e.g. '0,0.25,0.5,0.75,1'.")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--slat_steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--slat_cfg_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trellis_model", type=str, choices=["auto", "text_base", "image_large"], default="auto")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--allow_tf32", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save_latents", type=int, choices=[0, 1], default=1)
    parser.add_argument("--source_images_root", type=str, default=None, help="Root containing source images for DINO-conditioned SLat checkpoints.")
    parser.add_argument("--source_image_filename", type=str, default="", help="Optional fixed image filename inside each asset directory.")
    return parser.parse_args()


def load_tensor(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_asset(root: Path, name: str):
    asset_dir = root / "assets" / name
    if not asset_dir.is_dir():
        raise FileNotFoundError(f"Asset directory not found: {asset_dir}")

    feats = load_tensor(asset_dir / "slat_feats.pt").float()
    coords = load_tensor(asset_dir / "slat_coords.pt").int()
    ss_latent = load_tensor(asset_dir / "ss_latent.pt").float()

    coords = ensure_batch_coords(coords)
    coords = coords.clone()
    coords[:, 0] = 0

    if ss_latent.ndim == 4:
        ss_latent = ss_latent.unsqueeze(0)
    elif ss_latent.ndim == 6 and ss_latent.shape[0] == 1:
        ss_latent = ss_latent.squeeze(1)

    if ss_latent.ndim != 5:
        raise ValueError(f"Expected SS latent shape [C,D,H,W] or [1,C,D,H,W], got {tuple(ss_latent.shape)} for {name}")

    return {
        "name": name,
        "feats": feats,
        "coords": coords,
        "ss_latent": ss_latent,
    }


def resolve_asset_names(dataset, args):
    if args.src1_name and args.src2_name:
        return args.src1_name, args.src2_name, None

    if args.index_unit == "metadata_src":
        if args.src1_index < 0 or args.src1_index >= len(dataset.metadata):
            raise IndexError(f"--src1_index out of metadata range: {args.src1_index}")
        if args.src2_index < 0 or args.src2_index >= len(dataset.metadata):
            raise IndexError(f"--src2_index out of metadata range: {args.src2_index}")
        return (
            str(dataset.metadata[args.src1_index]["src_1"]),
            str(dataset.metadata[args.src2_index]["src_1"]),
            None,
        )

    assets = sorted({str(e["src_1"]) for e in dataset.metadata} | {str(e["src_2"]) for e in dataset.metadata})
    if args.src1_index < 0 or args.src1_index >= len(assets):
        raise IndexError(f"--src1_index out of asset range [0, {len(assets) - 1}]: {args.src1_index}")
    if args.src2_index < 0 or args.src2_index >= len(assets):
        raise IndexError(f"--src2_index out of asset range [0, {len(assets) - 1}]: {args.src2_index}")
    return assets[args.src1_index], assets[args.src2_index], assets


def batch_for_alpha(src1, src2, alpha):
    batch = {
        "src1_feats": src1["feats"],
        "src1_coords": src1["coords"],
        "src1_ss_latent": src1["ss_latent"],
        "src2_feats": src2["feats"],
        "src2_coords": src2["coords"],
        "src2_ss_latent": src2["ss_latent"],
        "alpha": torch.tensor([float(alpha)], dtype=torch.float32),
    }
    if "image" in src1 and "image" in src2:
        batch["src1_image"] = src1["image"].unsqueeze(0)
        batch["src2_image"] = src2["image"].unsqueeze(0)
    return batch


def attach_source_images(dataset, src1, src2):
    entry = {"src_1": src1["name"], "src_2": src2["name"]}
    src1["image"] = dataset._load_source_image(src1["name"], entry, "src1")
    src2["image"] = dataset._load_source_image(src2["name"], entry, "src2")


def alpha_color(alpha):
    alpha = float(alpha)
    src1 = np.array([80, 150, 255], dtype=np.float32)
    src2 = np.array([255, 150, 80], dtype=np.float32)
    # Dataset convention: alpha is fraction of src1.
    return tuple((alpha * src1 + (1.0 - alpha) * src2).round().astype(int).tolist())


def main():
    args = parse_args()
    alphas = parse_alpha_list(args.alphas)

    root = Path(args.root_dir).expanduser().resolve()
    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    slat_checkpoint_path = (
        Path(args.slat_checkpoint_path).expanduser().resolve()
        if args.slat_checkpoint_path
        else None
    )

    if args.allow_tf32 == 1 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if slat_checkpoint_path is not None and not slat_checkpoint_path.is_file():
        raise FileNotFoundError(f"SLat checkpoint not found: {slat_checkpoint_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt = load_checkpoint(str(checkpoint_path))
    flow_target = detect_flow_target(ckpt)
    if flow_target != "ss":
        raise ValueError(
            f"This alpha sweep script currently supports SS checkpoints only, got flow_target={flow_target!r}. "
            "SLat checkpoints need target sparse coordinates for each alpha."
        )
    slat_ckpt = load_checkpoint(str(slat_checkpoint_path)) if slat_checkpoint_path is not None else None
    if slat_ckpt is not None and detect_flow_target(slat_ckpt) != "slat":
        raise ValueError(
            f"--slat_checkpoint_path must point to a SLat checkpoint, got {detect_flow_target(slat_ckpt)!r}."
        )
    needs_source_images = slat_ckpt is not None and checkpoint_requires_source_images(slat_ckpt, "slat")
    source_images_root = args.source_images_root
    if needs_source_images and not source_images_root:
        source_images_root = checkpoint_args(slat_ckpt).get("source_images_root")
    if needs_source_images and not source_images_root:
        raise ValueError("A DINO-conditioned SLat checkpoint requires --source_images_root.")

    model_type = detect_model_type(ckpt, args.trellis_model)
    slat_model_type = detect_model_type(slat_ckpt, args.trellis_model) if slat_ckpt is not None else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MorphingDistillDataset(
        root=str(root),
        metadata_file=str(metadata_path),
        split=args.split,
        verbose=False,
        load_source_images=False,
        source_images_root=source_images_root,
        source_image_filename=args.source_image_filename,
    )
    src1_name, src2_name, asset_list = resolve_asset_names(dataset, args)
    src1 = load_asset(root, src1_name)
    src2 = load_asset(root, src2_name)
    if needs_source_images:
        attach_source_images(dataset, src1, src2)

    run_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{safe_slug(src1_name, 32)}_to_{safe_slug(src2_name, 32)}"
    )
    output_dir = Path(args.output_dir).expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("===== ALPHA SWEEP =====")
    print(f"dataset: {root}")
    print(f"metadata: {metadata_path}")
    print(f"split: {args.split}")
    print(f"checkpoint: {checkpoint_path}")
    if slat_checkpoint_path is not None:
        print(f"slat_checkpoint: {slat_checkpoint_path}")
        print("pipeline: ss checkpoint -> SS decoder coords -> slat checkpoint -> mesh decoder")
        print(f"slat_condition_source: {detect_slat_condition_source(slat_ckpt)}")
    if needs_source_images:
        print(f"source_images_root: {source_images_root}")
        print(f"source_image_filename: {args.source_image_filename or '<auto>'}")
    print(f"flow_target: {flow_target}")
    print(f"ss_flow_arch: {checkpoint_args(ckpt).get('ss_flow_arch', 'standard')}")
    print(f"model_type: {model_type}")
    if slat_model_type is not None:
        print(f"slat_model_type: {slat_model_type}")
    print(f"src1[{args.src1_index}]: {src1_name}")
    print(f"src2[{args.src2_index}]: {src2_name}")
    print(f"alphas: {alphas}")
    print(f"cfg_scale: {args.cfg_scale}")
    if slat_checkpoint_path is not None:
        print(f"slat_cfg_scale: {args.slat_cfg_scale if args.slat_cfg_scale is not None else args.cfg_scale}")
    print(f"steps: {args.steps}")
    if slat_checkpoint_path is not None:
        print(f"slat_steps: {args.slat_steps if args.slat_steps is not None else args.steps}")
    print(f"output: {output_dir}")
    print("alpha convention: alpha=1 is src1, alpha=0 is src2")
    print("=======================")

    model = build_model(ckpt, model_type, flow_target).to(device).eval()
    slat_model = (
        build_model(slat_ckpt, slat_model_type, "slat").to(device).eval()
        if slat_ckpt is not None
        else None
    )
    preload_dino_if_needed(model, device)
    if slat_model is not None:
        preload_dino_if_needed(slat_model, device)
    ss_decoder, mesh_decoder, sparse_tensor_cls = load_decoders(flow_target, device)

    save_slat_glb(
        mesh_decoder,
        sparse_tensor_cls,
        src1["feats"],
        src1["coords"],
        output_dir / "src1.glb",
        device,
        args.mixed_precision,
        fallback_color=(80, 150, 255),
    )
    save_slat_glb(
        mesh_decoder,
        sparse_tensor_cls,
        src2["feats"],
        src2["coords"],
        output_dir / "src2.glb",
        device,
        args.mixed_precision,
        fallback_color=(255, 150, 80),
    )

    rows = []
    template_ss = torch.zeros_like(src1["ss_latent"])

    for step_idx, alpha in enumerate(alphas):
        torch.manual_seed(args.seed + step_idx)
        batch = batch_for_alpha(src1, src2, alpha)
        pred_ss = sample_ss(
            model,
            batch,
            template_ss.to(device=device, dtype=torch.float32),
            args.steps,
            device,
            args.cfg_scale,
            args.mixed_precision,
        )

        step_name = f"alpha_{alpha:.4f}".replace(".", "p")
        step_dir = output_dir / step_name
        step_dir.mkdir(parents=True, exist_ok=True)
        voxel_info = save_voxel_glb(
            ss_decoder,
            pred_ss,
            step_dir / "pred_voxels.glb",
            device,
            args.mixed_precision,
            color=alpha_color(alpha),
        )

        row = {
            "step": step_idx,
            "alpha": float(alpha),
            "pred_voxels": str(step_dir / "pred_voxels.glb"),
            "voxel_info": voxel_info,
        }
        pred_coords = ss_coords_from_latent(ss_decoder, pred_ss, device, args.mixed_precision)
        row["pred_ss_points"] = int(pred_coords.shape[0])

        pred_final_slat = None
        if slat_model is not None:
            torch.manual_seed(args.seed + 10_000 + step_idx)
            slat_steps = args.slat_steps if args.slat_steps is not None else args.steps
            slat_cfg_scale = args.slat_cfg_scale if args.slat_cfg_scale is not None else args.cfg_scale
            pred_final_slat = sample_slat_on_coords(
                slat_model,
                batch,
                pred_coords,
                slat_steps,
                device,
                slat_cfg_scale,
                args.mixed_precision,
            )
            if pred_final_slat is None:
                row["pred_final_saved"] = False
                row["pred_final_reason"] = "empty_ss_coords"
            else:
                final_path = step_dir / "pred_final.glb"
                row["pred_final_saved"] = save_slat_glb(
                    mesh_decoder,
                    sparse_tensor_cls,
                    pred_final_slat.feats,
                    pred_final_slat.coords,
                    final_path,
                    device,
                    args.mixed_precision,
                    fallback_color=alpha_color(alpha),
                )
                row["pred_final"] = str(final_path)
                row["pred_final_points"] = int(pred_final_slat.feats.shape[0])

        if args.save_latents == 1:
            latent_path = step_dir / "pred_ss_latent.pt"
            payload = {
                "flow_target": flow_target,
                "ss_flow_arch": checkpoint_args(ckpt).get("ss_flow_arch", "standard"),
                "src1": src1_name,
                "src2": src2_name,
                "alpha": float(alpha),
                "pred_ss_latent": pred_ss.detach().cpu(),
                "pred_ss_coords": pred_coords.detach().cpu(),
            }
            if pred_final_slat is not None:
                payload.update(
                    {
                        "pred_final_slat_feats": pred_final_slat.feats.detach().cpu(),
                        "pred_final_slat_coords": pred_final_slat.coords.detach().cpu(),
                    }
                )
            torch.save(payload, latent_path)
            row["pred_ss_latent"] = str(latent_path)

        with (step_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)

        rows.append(row)
        final_msg = f", final={row.get('pred_final_saved')}" if slat_model is not None else ""
        print(f"[{step_idx + 1}/{len(alphas)}] alpha={alpha:.4f} points={row['pred_ss_points']}{final_msg} -> {step_dir}")

    summary = {
        "dataset": str(root),
        "metadata": str(metadata_path),
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "slat_checkpoint": str(slat_checkpoint_path) if slat_checkpoint_path is not None else None,
        "pipeline_mode": slat_checkpoint_path is not None,
        "flow_target": flow_target,
        "ss_flow_arch": checkpoint_args(ckpt).get("ss_flow_arch", "standard"),
        "slat_checkpoint_condition_source": detect_slat_condition_source(slat_ckpt) if slat_ckpt is not None else None,
        "source_images_root": str(source_images_root) if source_images_root else None,
        "model_type": model_type,
        "slat_model_type": slat_model_type,
        "src1_index": args.src1_index,
        "src2_index": args.src2_index,
        "index_unit": args.index_unit,
        "src1": src1_name,
        "src2": src2_name,
        "alphas": [float(a) for a in alphas],
        "cfg_scale": args.cfg_scale,
        "slat_cfg_scale": args.slat_cfg_scale if args.slat_cfg_scale is not None else args.cfg_scale,
        "steps": args.steps,
        "slat_steps": args.slat_steps if args.slat_steps is not None else args.steps,
        "seed": args.seed,
        "outputs": rows,
    }
    if asset_list is not None:
        summary["asset_count"] = len(asset_list)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("===== DONE =====")
    print(f"output: {output_dir}")
    print("================")


if __name__ == "__main__":
    main()
