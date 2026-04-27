import os
os.environ['ATTN_BACKEND'] = 'xformers'

import argparse
import json
from datetime import datetime

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data.morph_dataset import MorphingDistillDataset, morphing_collate_fn
from models.morph_flow import MorphFlow

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/home/filippo/datasets/3d/morphing_dataset_flux")
    parser.add_argument("--metadata", type=str, default="metadata_2.json")
    parser.add_argument("--val_metadata", type=str, default="metadata_val_200_tail.json")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--train_bs", type=int, default=1, help="Batch size per GPU/process")
    parser.add_argument("--val_bs", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_examples", type=int, default=2)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--val_max_items", type=int, default=200)
    parser.add_argument(
        "--exclude_val_assets_from_train",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, training excludes entries that use src assets found in val metadata.",
    )
    parser.add_argument("--use_ema", type=int, choices=[0, 1], default=0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    return parser


def init_ema_state_dict(model):
    ema_state = {}
    for name, tensor in model.state_dict().items():
        if torch.is_floating_point(tensor):
            ema_state[name] = tensor.detach().float().cpu().clone()
        else:
            ema_state[name] = tensor.detach().cpu().clone()
    return ema_state


@torch.no_grad()
def update_ema_state_dict(ema_state, model, decay):
    for name, tensor in model.state_dict().items():
        src = tensor.detach()
        if torch.is_floating_point(src):
            ema_state[name].mul_(decay).add_(src.float().cpu(), alpha=1.0 - decay)
        else:
            ema_state[name].copy_(src.cpu())


def load_assets_from_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    assets = set()
    for entry in entries:
        src_1 = entry.get("src_1")
        src_2 = entry.get("src_2")
        if src_1:
            assets.add(src_1)
        if src_2:
            assets.add(src_2)
    return assets


def _coords_to_xyz(coords):
    if coords.ndim != 2 or coords.shape[1] < 4:
        return torch.empty((0, 3), dtype=torch.float32)

    coords_cpu = coords.detach().cpu()
    mask = coords_cpu[:, 0] == 0
    xyz = coords_cpu[mask, 1:4].float()
    return xyz


def capture_validation_example(batch, loss_value):
    src1_name = batch.get("src1_name", ["unknown"])
    src2_name = batch.get("src2_name", ["unknown"])
    target_name = batch.get("target_name", ["unknown"])

    return {
        "src1_name": src1_name[0] if isinstance(src1_name, list) and src1_name else "unknown",
        "src2_name": src2_name[0] if isinstance(src2_name, list) and src2_name else "unknown",
        "target_name": target_name[0] if isinstance(target_name, list) and target_name else "unknown",
        "alpha": float(batch["alpha"][0].detach().cpu().item()),
        "src1_xyz": _coords_to_xyz(batch["src1_coords"]),
        "src2_xyz": _coords_to_xyz(batch["src2_coords"]),
        "target_xyz": _coords_to_xyz(batch["target_coords"]),
        "loss": float(loss_value),
    }


def build_validation_figure(example):
    if plt is None:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    views = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))

    for ax, (a, b, title) in zip(axes, views):
        src1 = example["src1_xyz"]
        src2 = example["src2_xyz"]
        target = example["target_xyz"]

        if src1.numel() > 0:
            ax.scatter(src1[:, a], src1[:, b], s=2, alpha=0.35, label="src1")
        if src2.numel() > 0:
            ax.scatter(src2[:, a], src2[:, b], s=2, alpha=0.35, label="src2")
        if target.numel() > 0:
            ax.scatter(target[:, a], target[:, b], s=2, alpha=0.5, label="target")

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)

    title = (
        f"src1={example['src1_name']} | src2={example['src2_name']} | "
        f"target={example['target_name']} | alpha={example['alpha']:.4f} | loss={example['loss']:.6f}"
    )
    fig.suptitle(title, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    return fig


def train(args):
    accelerator = Accelerator()
    device = accelerator.device

    metadata_path = os.path.join(args.root_dir, args.metadata)
    val_metadata_path = os.path.join(args.root_dir, args.val_metadata) if args.val_metadata else None

    excluded_assets = set()
    if (
        val_metadata_path
        and os.path.exists(val_metadata_path)
        and args.exclude_val_assets_from_train == 1
    ):
        excluded_assets = load_assets_from_metadata(val_metadata_path)

    dataset = MorphingDistillDataset(
        root=args.root_dir,
        metadata_file=metadata_path,
        verbose=accelerator.is_main_process,
        exclude_assets=excluded_assets,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.train_bs,
        shuffle=True,
        collate_fn=morphing_collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    val_dataset = None
    val_loader = None
    if val_metadata_path and os.path.exists(val_metadata_path):
        val_dataset = MorphingDistillDataset(
            root=args.root_dir,
            metadata_file=val_metadata_path,
            verbose=accelerator.is_main_process,
        )
        if args.val_max_items > 0 and len(val_dataset) > args.val_max_items:
            val_dataset = Subset(val_dataset, list(range(args.val_max_items)))

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_bs,
            shuffle=False,
            collate_fn=morphing_collate_fn,
            pin_memory=torch.cuda.is_available(),
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
    elif val_metadata_path:
        accelerator.print(f"Validation metadata not found, skipping validation: {val_metadata_path}")

    model = MorphFlow()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if val_loader is not None:
        model, optimizer, loader, val_loader = accelerator.prepare(model, optimizer, loader, val_loader)
    else:
        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    unwrapped_model = accelerator.unwrap_model(model)
    ema_state = None
    if args.use_ema == 1 and accelerator.is_main_process:
        ema_state = init_ema_state_dict(unwrapped_model)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, run_name)
    tb_dir = os.path.join(out_dir, "tb")

    writer = None
    if accelerator.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

    accelerator.wait_for_everyone()

    accelerator.print(f"Using device: {device}")
    accelerator.print(f"Num processes: {accelerator.num_processes}")
    accelerator.print(f"Dataset size: {len(dataset)}")
    if excluded_assets:
        accelerator.print(f"Excluded train assets from validation metadata: {len(excluded_assets)}")
    if val_dataset is not None:
        accelerator.print(f"Validation dataset size: {len(val_dataset)}")
    accelerator.print(f"Checkpoints and logs in: {out_dir}")
    accelerator.print(f"TensorBoard logs in: {tb_dir}")
    accelerator.print(f"EMA enabled: {args.use_ema == 1}")
    if args.use_ema == 1:
        accelerator.print(f"EMA decay: {args.ema_decay}")

    global_step = 0
    model.train()

    for epoch in range(1, args.train_epochs + 1):
        running_loss = 0.0

        progress_bar = tqdm(
            loader,
            desc=f"Epoch {epoch}/{args.train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            leave=True,
        )

        for batch_idx, batch in enumerate(progress_bar, start=1):
            src1_feats = batch["src1_feats"].to(device=device, dtype=torch.float32, non_blocking=True)
            src1_coords = batch["src1_coords"].to(device=device, dtype=torch.int32, non_blocking=True)

            src2_feats = batch["src2_feats"].to(device=device, dtype=torch.float32, non_blocking=True)
            src2_coords = batch["src2_coords"].to(device=device, dtype=torch.int32, non_blocking=True)

            target_ss_latent = batch["target_ss_latent"].to(device=device, dtype=torch.float32, non_blocking=True)
            alpha = batch["alpha"].to(device=device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            loss = model(
                target_ss_latent,
                src1_feats,
                src1_coords,
                src2_feats,
                src2_coords,
                alpha,
            )

            accelerator.backward(loss)
            optimizer.step()

            if args.use_ema == 1 and accelerator.is_main_process:
                update_ema_state_dict(ema_state, unwrapped_model, args.ema_decay)

            reduced_loss = accelerator.reduce(loss.detach(), reduction="mean")
            loss_value = float(reduced_loss.item())
            running_loss += loss_value
            global_step += 1
            avg_loss = running_loss / batch_idx

            if writer is not None:
                writer.add_scalar("train/loss_step", loss_value, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            if accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    loss=f"{loss_value:.6f}",
                    avg=f"{avg_loss:.6f}",
                    step=global_step,
                )

            if batch_idx % args.log_every == 0:
                accelerator.print(
                    f"[Epoch {epoch}/{args.train_epochs}] "
                    f"[Batch {batch_idx}/{len(loader)}] "
                    f"[Step {global_step}] "
                    f"loss={loss_value:.6f} avg_loss={avg_loss:.6f}"
                )

        epoch_avg = running_loss / max(1, len(loader))
        accelerator.print(f"Epoch {epoch} completed. avg_loss={epoch_avg:.6f}")

        val_avg = None
        val_examples = []
        if val_loader is not None and (epoch % max(1, args.val_every) == 0):
            model.eval()
            val_running_loss = 0.0

            val_bar = tqdm(
                val_loader,
                desc=f"Val {epoch}/{args.train_epochs}",
                disable=not accelerator.is_local_main_process,
                dynamic_ncols=True,
                leave=False,
            )

            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_bar, start=1):
                    src1_feats = val_batch["src1_feats"].to(device=device, dtype=torch.float32, non_blocking=True)
                    src1_coords = val_batch["src1_coords"].to(device=device, dtype=torch.int32, non_blocking=True)

                    src2_feats = val_batch["src2_feats"].to(device=device, dtype=torch.float32, non_blocking=True)
                    src2_coords = val_batch["src2_coords"].to(device=device, dtype=torch.int32, non_blocking=True)

                    target_ss_latent = val_batch["target_ss_latent"].to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                    alpha = val_batch["alpha"].to(device=device, dtype=torch.float32, non_blocking=True)

                    val_loss = model(
                        target_ss_latent,
                        src1_feats,
                        src1_coords,
                        src2_feats,
                        src2_coords,
                        alpha,
                    )

                    reduced_val_loss = accelerator.reduce(val_loss.detach(), reduction="mean")
                    val_loss_value = float(reduced_val_loss.item())
                    val_running_loss += val_loss_value

                    avg_val = val_running_loss / val_batch_idx
                    if accelerator.is_local_main_process:
                        val_bar.set_postfix(loss=f"{val_loss_value:.6f}", avg=f"{avg_val:.6f}")

                    if len(val_examples) < max(0, args.val_examples):
                        val_examples.append(capture_validation_example(val_batch, val_loss_value))

            val_avg = val_running_loss / max(1, len(val_loader))
            accelerator.print(f"Epoch {epoch} validation completed. val_loss={val_avg:.6f}")
            model.train()

        if writer is not None:
            writer.add_scalar("train/loss_epoch", epoch_avg, epoch)
            if val_avg is not None:
                writer.add_scalar("val/loss_epoch", val_avg, epoch)

            if val_examples:
                for idx, example in enumerate(val_examples, start=1):
                    writer.add_text(
                        f"val/examples/example_{idx}_meta",
                        (
                            f"src1={example['src1_name']} | src2={example['src2_name']} | "
                            f"target={example['target_name']} | alpha={example['alpha']:.4f} | "
                            f"loss={example['loss']:.6f}"
                        ),
                        epoch,
                    )

                    fig = build_validation_figure(example)
                    if fig is not None:
                        writer.add_figure(f"val/examples/example_{idx}", fig, epoch)
                        plt.close(fig)

            writer.flush()

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            ckpt_path = os.path.join(out_dir, f"morphflow_epoch_{epoch:04d}_step_{global_step:07d}.pt")
            ckpt = {
                "epoch": epoch,
                "step": global_step,
                "model": accelerator.get_state_dict(model),
                "optimizer": optimizer.state_dict(),
            }
            if args.use_ema == 1 and ema_state is not None:
                ckpt["model_ema"] = ema_state
                ckpt["ema_decay"] = args.ema_decay
            accelerator.save(
                ckpt,
                ckpt_path,
            )
            accelerator.print(f"Checkpoint saved: {ckpt_path}")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_ckpt = os.path.join(out_dir, f"morphflow_epoch_{args.train_epochs:04d}_step_{global_step:07d}.pt")
        ckpt = {
            "epoch": args.train_epochs,
            "step": global_step,
            "model": accelerator.get_state_dict(model),
            "optimizer": optimizer.state_dict(),
        }
        if args.use_ema == 1 and ema_state is not None:
            ckpt["model_ema"] = ema_state
            ckpt["ema_decay"] = args.ema_decay
        accelerator.save(
            ckpt,
            final_ckpt,
        )
        accelerator.print(f"Training completed. Final checkpoint: {final_ckpt}")

        if writer is not None:
            writer.close()
