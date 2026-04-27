import argparse
import os
from datetime import datetime
from glob import glob

os.environ["ATTN_BACKEND"] = "xformers"

import torch
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter

    TB_IMPORT_ERROR = None
except Exception as exc:
    SummaryWriter = None
    TB_IMPORT_ERROR = exc

from data.morph_dataset import MorphingDistillDataset, morphing_collate_fn
from models.morph_flow import MorphFlow
from models.sparse_structure_vae import SparseStructureDecoder

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MorphFlow and log only GT vs reconstructed object comparisons"
    )
    parser.add_argument("--root_dir", type=str, default="/home/filippo/datasets/3d/morphing_dataset_flux")
    parser.add_argument("--val_metadata", type=str, default="metadata_val_200_tail.json")
    parser.add_argument("--checkpoints_root", type=str, default="./outputs/morphflow")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--tb_out_dir", type=str, default="./outputs/morphflow_eval")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--max_items", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--decoder_checkpoint", type=str, default="")
    parser.add_argument("--decoder_out_channels", type=int, default=1)
    parser.add_argument("--decoder_latent_channels", type=int, default=8)
    parser.add_argument("--decoder_num_res_blocks", type=int, default=2)
    parser.add_argument("--decoder_num_res_blocks_middle", type=int, default=2)
    parser.add_argument("--decoder_channels", type=str, default="512,128,32")
    parser.add_argument("--decoder_norm_type", type=str, default="layer", choices=["group", "layer"])
    parser.add_argument("--decoder_use_fp16", type=int, default=0, choices=[0, 1])
    parser.add_argument("--decoder_threshold", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max_3d_points", type=int, default=12000)
    return parser.parse_args()


def find_latest_checkpoint(checkpoints_root):
    pattern = os.path.join(checkpoints_root, "**", "morphflow_epoch_*.pt")
    candidates = glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under: {checkpoints_root}")
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def load_checkpoint(path, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    return ckpt


def find_default_decoder_checkpoint():
    candidates = []

    patterns = [
        "/home/filippo/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/snapshots/*/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        "/home/filippo/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/snapshots/*/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    ]
    for pattern in patterns:
        candidates.extend(glob(pattern))

    local_candidates = [
        "/home/filippo/checkpoints/decoder/ss_dec_conv3d_16l8_fp16.safetensors",
        "/home/filippo/checkpoints/decoder/ss_decoder.pth",
    ]
    for path in local_candidates:
        if os.path.exists(path):
            candidates.append(path)

    if not candidates:
        return ""

    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "module"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise ValueError("Checkpoint does not contain a state dict.")

    cleaned = {}
    for key, value in obj.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        cleaned[new_key] = value
    return cleaned


def load_decoder_state_dict(path, device):
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError(
                "Decoder checkpoint is .safetensors but safetensors is not installed. "
                "Install it with: pip install safetensors"
            ) from exc
        return load_file(path, device=str(device))

    return unwrap_state_dict(load_checkpoint(path, device))


def first_weight_ndim(state_dict):
    for key, tensor in state_dict.items():
        if key.endswith("weight") and hasattr(tensor, "ndim"):
            return int(tensor.ndim)
    return -1


def parse_int_list(csv_values):
    values = [x.strip() for x in csv_values.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer in --decoder_channels")
    return [int(x) for x in values]


def select_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_target_latent(x):
    if x.ndim == 6 and x.shape[1] == 1:
        return x.squeeze(1)
    if x.ndim == 5:
        return x
    raise ValueError(f"Unexpected target_ss_latent shape: {tuple(x.shape)}")


def get_item_name(batch, key, idx):
    values = batch.get(key, None)
    if isinstance(values, list) and idx < len(values):
        return str(values[idx])
    return "unknown"


def squeeze_binary_grid(grid):
    out = grid.detach().cpu()
    while out.ndim > 3:
        out = out[0]
    if out.ndim != 3:
        raise ValueError(f"Expected a [D, H, W] binary grid, got shape {tuple(out.shape)}")
    return out.bool()


def build_projection_figure(gt_bin_grid, pred_bin_grid):
    if plt is None:
        return None

    gt = squeeze_binary_grid(gt_bin_grid)
    pred = squeeze_binary_grid(pred_bin_grid)

    planes = [
        (gt.float().amax(dim=0), pred.float().amax(dim=0), "XY"),
        (gt.float().amax(dim=1), pred.float().amax(dim=1), "XZ"),
        (gt.float().amax(dim=2), pred.float().amax(dim=2), "YZ"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for col, (gt_plane, pred_plane, name) in enumerate(planes):
        axes[0, col].imshow(gt_plane.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, col].set_title(f"GT {name}")
        axes[1, col].imshow(pred_plane.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, col].set_title(f"Pred {name}")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig


def sample_points(coords, max_points):
    if coords.shape[0] <= max_points:
        return coords
    idx = torch.randperm(coords.shape[0])[:max_points]
    return coords[idx]


def build_3d_scatter_figure(gt_bin_grid, pred_bin_grid, max_points=12000):
    if plt is None:
        return None

    gt = squeeze_binary_grid(gt_bin_grid)
    pred = squeeze_binary_grid(pred_bin_grid)

    gt_pts = torch.nonzero(gt, as_tuple=False)
    pred_pts = torch.nonzero(pred, as_tuple=False)
    if gt_pts.shape[0] == 0 and pred_pts.shape[0] == 0:
        return None

    gt_pts = sample_points(gt_pts, max_points)
    pred_pts = sample_points(pred_pts, max_points)

    fig = plt.figure(figsize=(12, 6))
    ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pr = fig.add_subplot(1, 2, 2, projection="3d")

    if gt_pts.shape[0] > 0:
        ax_gt.scatter(
            gt_pts[:, 2].numpy(),
            gt_pts[:, 1].numpy(),
            gt_pts[:, 0].numpy(),
            s=1,
            alpha=0.6,
        )

    if pred_pts.shape[0] > 0:
        ax_pr.scatter(
            pred_pts[:, 2].numpy(),
            pred_pts[:, 1].numpy(),
            pred_pts[:, 0].numpy(),
            s=1,
            alpha=0.6,
            color="tab:orange",
        )

    ax_gt.set_title(f"GT 3D ({gt_pts.shape[0]} pts)")
    ax_pr.set_title(f"Pred 3D ({pred_pts.shape[0]} pts)")

    d, h, w = gt.shape
    for ax in (ax_gt, ax_pr):
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_zlim(0, d)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    fig.tight_layout()
    return fig


def grid_iou_and_counts(gt_bin_grid, pred_bin_grid, eps=1e-8):
    gt = squeeze_binary_grid(gt_bin_grid).float()
    pred = squeeze_binary_grid(pred_bin_grid).float()
    intersection = (gt * pred).sum()
    union = ((gt + pred) > 0).float().sum()
    iou = float((intersection / (union + eps)).item())
    return iou, int(gt.sum().item()), int(pred.sum().item())


def save_figure(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")


def run_reverse_flow_sample(model, x0_shape, src1_feats, src2_feats, src1_coords, src2_coords, alpha, steps):
    x_t = torch.randn(x0_shape, device=alpha.device, dtype=torch.float32)
    t_seq = torch.linspace(1.0, 0.0, steps + 1, device=alpha.device, dtype=torch.float32)

    for i in range(steps):
        t_curr = t_seq[i]
        t_next = t_seq[i + 1]
        dt = t_curr - t_next
        t_batch = t_curr.unsqueeze(0)
        v_pred = model.forward_flow(
            x_t,
            t_batch,
            src1_feats,
            src2_feats,
            src1_coords,
            src2_coords,
            alpha,
        )
        x_t = x_t - dt * v_pred

    return x_t


def main():
    args = parse_args()
    if args.max_items <= 0:
        raise ValueError("--max_items must be > 0")

    device = select_device(args.device)
    torch.manual_seed(args.seed)

    val_metadata_path = os.path.join(args.root_dir, args.val_metadata)
    if not os.path.exists(val_metadata_path):
        raise FileNotFoundError(f"Validation metadata not found: {val_metadata_path}")

    ckpt_path = args.checkpoint or find_latest_checkpoint(args.checkpoints_root)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path, device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = MorphFlow().to(device)
    model.load_state_dict(unwrap_state_dict(state_dict), strict=True)
    model.eval()

    decoder_ckpt_path = args.decoder_checkpoint or find_default_decoder_checkpoint()
    if not decoder_ckpt_path:
        raise FileNotFoundError(
            "No sparse-structure decoder checkpoint found. "
            "Provide --decoder_checkpoint pointing to ss_dec_conv3d_16l8_fp16.safetensors."
        )
    if not os.path.exists(decoder_ckpt_path):
        raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_ckpt_path}")

    decoder = SparseStructureDecoder(
        out_channels=args.decoder_out_channels,
        latent_channels=args.decoder_latent_channels,
        num_res_blocks=args.decoder_num_res_blocks,
        channels=parse_int_list(args.decoder_channels),
        num_res_blocks_middle=args.decoder_num_res_blocks_middle,
        norm_type=args.decoder_norm_type,
        use_fp16=bool(args.decoder_use_fp16),
    ).to(device)

    decoder_state = load_decoder_state_dict(decoder_ckpt_path, device)
    decoder_weight_ndim = first_weight_ndim(decoder_state)
    if decoder_weight_ndim == 4:
        raise RuntimeError(
            f"Incompatible decoder checkpoint: {decoder_ckpt_path}. "
            "It appears to be a 2D image decoder (Conv2d weights), but this eval needs the "
            "3D sparse-structure decoder (ss_dec_conv3d_16l8_fp16.safetensors)."
        )
    decoder.load_state_dict(decoder_state, strict=True)
    decoder.eval()

    val_dataset = MorphingDistillDataset(
        root=args.root_dir,
        metadata_file=val_metadata_path,
        verbose=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=morphing_collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    run_name = args.run_name or f"object_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_dir = os.path.join(args.tb_out_dir, run_name)
    out_img_dir = os.path.join(tb_dir, "images")
    os.makedirs(out_img_dir, exist_ok=True)
    writer = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=tb_dir)
    else:
        print(
            "[WARN] TensorBoard not available in this Python environment. "
            "Only PNG images will be saved."
        )

    print(f"Using device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Decoder checkpoint: {decoder_ckpt_path}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"TensorBoard dir: {tb_dir}")
    print(f"Image output dir: {out_img_dir}")

    total_logged = 0
    iou_values = []

    with torch.no_grad():
        for batch in val_loader:
            src1_feats = batch["src1_feats"].to(device=device, dtype=torch.float32, non_blocking=True)
            src1_coords = batch["src1_coords"].to(device=device, dtype=torch.int32, non_blocking=True)

            src2_feats = batch["src2_feats"].to(device=device, dtype=torch.float32, non_blocking=True)
            src2_coords = batch["src2_coords"].to(device=device, dtype=torch.int32, non_blocking=True)

            target_ss_latent = batch["target_ss_latent"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            alpha = batch["alpha"].to(device=device, dtype=torch.float32, non_blocking=True)

            x0_gt = prepare_target_latent(target_ss_latent)
            bsz = x0_gt.shape[0]

            for b in range(bsz):
                if total_logged >= args.max_items:
                    break

                torch.manual_seed(args.seed + total_logged)

                x0_gt_i = x0_gt[b : b + 1]
                src1_feats_i = src1_feats[src1_coords[:, 0] == b]
                src1_coords_i = src1_coords[src1_coords[:, 0] == b].clone()
                src1_coords_i[:, 0] = 0

                src2_feats_i = src2_feats[src2_coords[:, 0] == b]
                src2_coords_i = src2_coords[src2_coords[:, 0] == b].clone()
                src2_coords_i[:, 0] = 0

                alpha_i = alpha[b : b + 1]

                x0_pred = run_reverse_flow_sample(
                    model=model,
                    x0_shape=x0_gt_i.shape,
                    src1_feats=src1_feats_i,
                    src2_feats=src2_feats_i,
                    src1_coords=src1_coords_i,
                    src2_coords=src2_coords_i,
                    alpha=alpha_i,
                    steps=args.steps,
                )

                gt_logits = decoder(x0_gt_i)
                pred_logits = decoder(x0_pred)
                gt_bin = torch.sigmoid(gt_logits) >= args.decoder_threshold
                pred_bin = torch.sigmoid(pred_logits) >= args.decoder_threshold

                iou, gt_count, pred_count = grid_iou_and_counts(gt_bin[0], pred_bin[0])
                iou_values.append(iou)

                src1_name = get_item_name(batch, "src1_name", b)
                src2_name = get_item_name(batch, "src2_name", b)
                target_name = get_item_name(batch, "target_name", b)

                sample_id = total_logged + 1
                sample_name = f"sample_{sample_id:03d}"
                tag = f"val_objects/{sample_name}"

                summary = (
                    f"src1={src1_name} | src2={src2_name} | target={target_name} | "
                    f"alpha={float(alpha_i.item()):.4f} | iou={iou:.4f} | "
                    f"gt_voxels={gt_count} | pred_voxels={pred_count}"
                )

                if writer is not None:
                    writer.add_text(f"{tag}/meta", summary, total_logged)
                    writer.add_scalar(f"{tag}/iou", iou, total_logged)

                fig_projection = build_projection_figure(gt_bin[0], pred_bin[0])
                if fig_projection is not None:
                    if writer is not None:
                        writer.add_figure(f"{tag}/projections", fig_projection, total_logged)
                    save_figure(fig_projection, os.path.join(out_img_dir, f"{sample_name}_projections.png"))
                    plt.close(fig_projection)

                fig_3d = build_3d_scatter_figure(
                    gt_bin[0],
                    pred_bin[0],
                    max_points=args.max_3d_points,
                )
                if fig_3d is not None:
                    if writer is not None:
                        writer.add_figure(f"{tag}/object_3d", fig_3d, total_logged)
                    save_figure(fig_3d, os.path.join(out_img_dir, f"{sample_name}_3d.png"))
                    plt.close(fig_3d)

                total_logged += 1

            if total_logged >= args.max_items:
                break

    if total_logged == 0:
        if writer is not None:
            writer.close()
        raise RuntimeError("No validation sample was logged.")

    if writer is not None:
        writer.add_scalar("val_objects/summary/iou_mean", sum(iou_values) / len(iou_values), 0)
        writer.flush()
        writer.close()

    if writer is not None:
        print(f"Logged {total_logged} validation samples to TensorBoard.")
    else:
        print(f"Processed {total_logged} validation samples.")
    print(f"Saved image comparisons in: {out_img_dir}")
    if writer is not None:
        print(f"TensorBoard command: tensorboard --logdir {args.tb_out_dir}")


if __name__ == "__main__":
    main()