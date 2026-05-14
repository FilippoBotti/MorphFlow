import argparse
import inspect
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("ATTN_BACKEND", "xformers")

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data.morph_dataset import MorphingDistillDataset, morphing_collate_fn
from models.morph_flow import MorphFlow


def build_parser():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--root_dir", type=str, default="/home/filippo/datasets/3d/morphing_dataset_flux")
    parser.add_argument("--metadata", type=str, default="metadata_2.json")
    parser.add_argument("--val_metadata", type=str, default="metadata_val_200_tail.json")
    parser.add_argument("--exclude_val_assets_from_train", type=int, choices=[0, 1], default=1)

    # Output
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--run_name", type=str, default=None)

    # Training
    parser.add_argument("--train_bs", type=int, default=1, help="Batch size per GPU/process")
    parser.add_argument("--val_bs", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--val_max_items", type=int, default=200)

    # Learning rates
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cond_lr", type=float, default=None)
    parser.add_argument(
        "--flow_lr",
        type=float,
        default=None,
        help=(
            "LR for TRELLIS sparse_structure_flow. "
            "If omitted, defaults to 1e-5 for image_large full fine-tuning, otherwise --lr."
        ),
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="0 disables gradient clipping")

    # Model
    parser.add_argument("--trellis_model", type=str, choices=["text_base", "image_large"], default="text_base")
    parser.add_argument("--separate_cond", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--separate_cond_gate",
        type=str,
        choices=["alpha_residual", "pair_channel", "token"],
        default="alpha_residual",
    )
    parser.add_argument("--cfg_drop_prob", type=float, default=0.0)

    # Optional future architecture args.
    # These are passed to MorphFlow only if its __init__ supports them.
    parser.add_argument("--cond_resample_tokens", type=int, default=0)
    parser.add_argument("--cond_resample_depth", type=int, default=1)
    parser.add_argument("--cond_resample_heads", type=int, default=8)

    # Optional future losses.
    # They are passed to MorphFlow.forward only if it supports them.
    parser.add_argument("--endpoint_loss_weight", type=float, default=0.0)
    parser.add_argument("--symmetry_loss_weight", type=float, default=0.0)

    # Precision / memory
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["auto", "no", "fp16", "bf16"],
        default="auto",
    )
    parser.add_argument("--allow_tf32", type=int, choices=[0, 1], default=1)
    parser.add_argument("--use_checkpoint", type=int, choices=[0, 1], default=0)

    # Checkpoint loading modes
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="True resume: same architecture, restores model + optimizer + scheduler + epoch/step.",
    )
    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Model-only initialization. Use when changing architecture, e.g. alpha_residual -> token.",
    )
    parser.add_argument("--resume_strict", type=int, choices=[0, 1], default=1)
    parser.add_argument("--init_strict", type=int, choices=[0, 1], default=0)
    parser.add_argument("--resume_optimizer", type=int, choices=[0, 1], default=1)

    # Accepted for backward compatibility with older SLURM scripts.
    # This simplified train.py does not implement LoRA or EMA.
    parser.add_argument("--use_lora", type=int, choices=[0, 1], default=0)
    parser.add_argument("--lora_lr", type=float, default=None)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="to_q,to_kv")
    parser.add_argument("--use_ema", type=int, choices=[0, 1], default=0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--val_examples", type=int, default=0)

    return parser


def resolve_mixed_precision(mode: str) -> str:
    if mode != "auto":
        return mode

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"

    return "no"


def resolve_existing_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return path

    candidates = [path]

    # Common mapping in your Singularity setup:
    # host:      /hpc/scratch/marco.barezzi/...
    # container: /scratch/...
    prefix = "/hpc/scratch/marco.barezzi/"
    if path.startswith(prefix):
        candidates.append("/scratch/" + path[len(prefix):])

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return path


def load_checkpoint_cpu(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_model_state(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    cleaned = {}
    for key, value in state.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value

    return cleaned


def load_model_state(
    model: torch.nn.Module,
    ckpt: Dict[str, Any],
    strict: bool,
    accelerator: Accelerator,
    label: str,
):
    state = extract_model_state(ckpt)
    result = model.load_state_dict(state, strict=strict)

    accelerator.print(f"{label}: loaded model weights with strict={strict}")

    if not strict:
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))

        accelerator.print(f"{label}: missing keys = {len(missing)}")
        for key in missing[:80]:
            accelerator.print(f"  missing: {key}")
        if len(missing) > 80:
            accelerator.print(f"  ... and {len(missing) - 80} more missing keys")

        accelerator.print(f"{label}: unexpected keys = {len(unexpected)}")
        for key in unexpected[:80]:
            accelerator.print(f"  unexpected: {key}")
        if len(unexpected) > 80:
            accelerator.print(f"  ... and {len(unexpected) - 80} more unexpected keys")

    return result


def load_assets_from_metadata(metadata_path: str) -> set:
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


def build_model(args, accelerator: Accelerator) -> MorphFlow:
    requested_kwargs = {
        "model_type": args.trellis_model,
        "separate_cond": args.separate_cond == 1,
        "use_checkpoint": args.use_checkpoint == 1,
        "separate_cond_gate": args.separate_cond_gate,
        "cond_resample_tokens": args.cond_resample_tokens,
        "cond_resample_depth": args.cond_resample_depth,
        "cond_resample_heads": args.cond_resample_heads,
    }

    signature = inspect.signature(MorphFlow.__init__)
    supported = set(signature.parameters.keys())

    model_kwargs = {
        key: value
        for key, value in requested_kwargs.items()
        if key in supported
    }

    ignored = sorted(set(requested_kwargs.keys()) - set(model_kwargs.keys()))
    if ignored:
        accelerator.print(f"MorphFlow does not support these constructor args; ignoring: {ignored}")

    accelerator.print("MorphFlow constructor kwargs:")
    for key, value in model_kwargs.items():
        accelerator.print(f"  {key}: {value}")

    model = MorphFlow(**model_kwargs)
    model.cfg_drop_prob = args.cfg_drop_prob
    return model


def model_forward_supports_extra_losses(model: torch.nn.Module) -> bool:
    signature = inspect.signature(model.forward)
    params = set(signature.parameters.keys())
    return "endpoint_loss_weight" in params and "symmetry_loss_weight" in params


def get_optional_tensor(batch: Dict[str, Any], key: str, device, dtype=torch.float32):
    value = batch.get(key, None)
    if value is None:
        return None
    return value.to(device=device, dtype=dtype, non_blocking=True)


def compute_loss(
    model,
    batch,
    device,
    supports_extra_losses: bool,
    endpoint_loss_weight: float,
    symmetry_loss_weight: float,
    use_extra_losses: bool,
):
    src1_feats = batch["src1_feats"].to(device=device, dtype=torch.float32, non_blocking=True)
    src1_coords = batch["src1_coords"].to(device=device, dtype=torch.int32, non_blocking=True)

    src2_feats = batch["src2_feats"].to(device=device, dtype=torch.float32, non_blocking=True)
    src2_coords = batch["src2_coords"].to(device=device, dtype=torch.int32, non_blocking=True)

    target_ss_latent = batch["target_ss_latent"].to(device=device, dtype=torch.float32, non_blocking=True)
    alpha = batch["alpha"].to(device=device, dtype=torch.float32, non_blocking=True)

    if supports_extra_losses and use_extra_losses:
        kwargs = {
            "endpoint_loss_weight": endpoint_loss_weight,
            "symmetry_loss_weight": symmetry_loss_weight,
        }

        if endpoint_loss_weight > 0.0:
            kwargs["src1_ss_latent"] = get_optional_tensor(batch, "src1_ss_latent", device)
            kwargs["src2_ss_latent"] = get_optional_tensor(batch, "src2_ss_latent", device)

        return model(
            target_ss_latent,
            src1_feats,
            src1_coords,
            src2_feats,
            src2_coords,
            alpha,
            **kwargs,
        )

    return model(
        target_ss_latent,
        src1_feats,
        src1_coords,
        src2_feats,
        src2_coords,
        alpha,
    )


def set_trainability(model: torch.nn.Module, args, accelerator: Accelerator):
    if args.use_lora == 1:
        raise ValueError(
            "This simplified train.py removed LoRA training. "
            "Set --use_lora 0 or restore the older LoRA code."
        )

    if args.use_ema == 1:
        accelerator.print("WARNING: --use_ema is accepted for compatibility but ignored in this simplified train.py.")

    # Full fine-tuning by default.
    for p in model.parameters():
        p.requires_grad = True

    # cond_fusion is unused when separate_cond=1.
    if args.separate_cond == 1 and hasattr(model, "cond_fusion"):
        for p in model.cond_fusion.parameters():
            p.requires_grad = False

    # null_cond is only useful when CFG dropout is enabled.
    if hasattr(model, "null_cond") and args.cfg_drop_prob <= 0.0:
        model.null_cond.requires_grad = False


def collect_param_groups(model: torch.nn.Module, args) -> Tuple[List[Dict[str, Any]], float, float]:
    cond_lr = args.lr if args.cond_lr is None else args.cond_lr

    if args.flow_lr is not None:
        flow_lr = args.flow_lr
    else:
        # Safer default for image_large full fine-tuning.
        flow_lr = 1e-5 if args.trellis_model == "image_large" else args.lr

    cond_modules = []
    for name in ["cond_encoder", "cond_fusion", "separate_cond_proj", "cond_resampler"]:
        module = getattr(model, name, None)
        if module is not None:
            cond_modules.append(module)

    cond_param_ids = set()
    cond_params = []
    for module in cond_modules:
        for p in module.parameters():
            if p.requires_grad and id(p) not in cond_param_ids:
                cond_params.append(p)
                cond_param_ids.add(id(p))

    if hasattr(model, "null_cond") and model.null_cond.requires_grad:
        cond_params.append(model.null_cond)
        cond_param_ids.add(id(model.null_cond))

    flow_params = []
    flow = getattr(model, "sparse_structure_flow", None)
    if flow is not None:
        for p in flow.parameters():
            if p.requires_grad and id(p) not in cond_param_ids:
                flow_params.append(p)

    param_groups = []
    if cond_params:
        param_groups.append({"params": cond_params, "lr": cond_lr, "name": "condition"})
    if flow_params:
        param_groups.append({"params": flow_params, "lr": flow_lr, "name": "flow"})

    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    return param_groups, cond_lr, flow_lr


def load_trellis_pretrained_if_needed(model, args, accelerator: Accelerator):
    if args.resume_from or args.init_from:
        mode = "--resume_from" if args.resume_from else "--init_from"
        accelerator.print(f"Skipping TRELLIS pretrained load because {mode} was provided.")
        return

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if args.trellis_model == "text_base":
        repo_id = "microsoft/TRELLIS-text-base"
        filename = "ckpts/ss_flow_txt_dit_B_16l8_fp16.safetensors"
    else:
        repo_id = "microsoft/TRELLIS-image-large"
        filename = "ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors"

    accelerator.print(f"Loading TRELLIS pretrained weights from {repo_id}...")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    trellis_state_dict = load_file(ckpt_path)

    missing, unexpected = model.sparse_structure_flow.load_state_dict(
        trellis_state_dict,
        strict=False,
    )

    accelerator.print(
        f"TRELLIS load_state_dict strict=False | missing={len(missing)} unexpected={len(unexpected)}"
    )

    if args.separate_cond == 1:
        copied_cross = 0
        copied_norm = 0

        for block in model.sparse_structure_flow.blocks:
            if hasattr(block, "cross_attn2"):
                block.cross_attn2.load_state_dict(block.cross_attn.state_dict())
                copied_cross += 1

            if hasattr(block, "norm4"):
                block.norm4.load_state_dict(block.norm2.state_dict())
                copied_norm += 1

        accelerator.print(
            f"Separate-cond init: copied cross_attn -> cross_attn2 for {copied_cross} blocks; "
            f"norm2 -> norm4 for {copied_norm} blocks."
        )

    accelerator.print(f"TRELLIS weights loaded successfully: {args.trellis_model}")


def save_checkpoint(
    accelerator: Accelerator,
    model,
    optimizer,
    scheduler,
    args,
    ckpt_dir: str,
    epoch: int,
    global_step: int,
    train_loss: Optional[float],
    val_loss: Optional[float],
    final: bool = False,
):
    if not accelerator.is_main_process:
        return

    if final:
        filename = f"morphflow_final_epoch_{epoch:04d}_step_{global_step:07d}.pt"
    else:
        filename = f"morphflow_epoch_{epoch:04d}_step_{global_step:07d}.pt"

    ckpt_path = os.path.join(ckpt_dir, filename)

    ckpt = {
        "epoch": epoch,
        "step": global_step,
        "model": accelerator.get_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "model_type": args.trellis_model,
        "sigma_min": accelerator.unwrap_model(model).sigma_min,
        "args": vars(args),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    accelerator.save(ckpt, ckpt_path)
    accelerator.print(f"Checkpoint saved: {ckpt_path}")


def train(args):
    args.resume_from = resolve_existing_path(args.resume_from)
    args.init_from = resolve_existing_path(args.init_from)

    mixed_precision = resolve_mixed_precision(args.mixed_precision)

    if args.allow_tf32 == 1 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )
    device = accelerator.device

    if args.resume_from and args.init_from:
        raise ValueError("Use either --resume_from or --init_from, not both.")

    if args.resume_from and not os.path.isfile(args.resume_from):
        raise FileNotFoundError(f"--resume_from checkpoint not found inside container: {args.resume_from}")

    if args.init_from and not os.path.isfile(args.init_from):
        raise FileNotFoundError(f"--init_from checkpoint not found inside container: {args.init_from}")

    if args.resume_from:
        accelerator.print(f"Resume mode: {args.resume_from}")
        accelerator.print("Use this only with the same architecture.")

    if args.init_from:
        accelerator.print(f"Init-from mode: {args.init_from}")
        accelerator.print("Model weights will be loaded; optimizer/scheduler start fresh.")

    metadata_path = os.path.join(args.root_dir, args.metadata)
    val_metadata_path = os.path.join(args.root_dir, args.val_metadata) if args.val_metadata else None

    excluded_assets = set()
    if val_metadata_path and os.path.exists(val_metadata_path) and args.exclude_val_assets_from_train == 1:
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

    model = build_model(args, accelerator)
    set_trainability(model, args, accelerator)

    load_trellis_pretrained_if_needed(model, args, accelerator)

    init_ckpt = None
    resume_ckpt = None

    if args.init_from:
        init_ckpt = load_checkpoint_cpu(args.init_from)
        load_model_state(
            model=model,
            ckpt=init_ckpt,
            strict=bool(args.init_strict),
            accelerator=accelerator,
            label="init_from",
        )

    if args.resume_from:
        resume_ckpt = load_checkpoint_cpu(args.resume_from)
        load_model_state(
            model=model,
            ckpt=resume_ckpt,
            strict=bool(args.resume_strict),
            accelerator=accelerator,
            label="resume_from",
        )

    supports_extra_losses = model_forward_supports_extra_losses(model)
    if not supports_extra_losses and (args.endpoint_loss_weight > 0.0 or args.symmetry_loss_weight > 0.0):
        accelerator.print(
            "WARNING: MorphFlow.forward does not support endpoint/symmetry loss kwargs. "
            "Those weights will be ignored."
        )

    param_groups, cond_lr, flow_lr = collect_param_groups(model, args)

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
    )

    from transformers import get_cosine_schedule_with_warmup

    total_training_steps = args.train_epochs * len(loader)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, max(1, total_training_steps // 10)),
        num_training_steps=total_training_steps,
    )

    trainable_tensors = [p for p in model.parameters() if p.requires_grad]
    trainable_params_count = sum(p.numel() for p in trainable_tensors)
    total_params_count = sum(p.numel() for p in model.parameters())

    accelerator.print(
        f"Before prepare | trainable tensors={len(trainable_tensors)} | "
        f"trainable params={trainable_params_count} | total params={total_params_count}"
    )

    if val_loader is not None:
        model, optimizer, loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, loader, val_loader, scheduler
        )
    else:
        model, optimizer, loader, scheduler = accelerator.prepare(
            model, optimizer, loader, scheduler
        )

    start_epoch = 1
    global_step = 0

    if resume_ckpt is not None:
        if args.resume_optimizer == 1:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
                accelerator.print("Optimizer state restored.")
            except Exception as exc:
                accelerator.print(f"Could not restore optimizer state: {exc}")

            if "scheduler" in resume_ckpt:
                try:
                    scheduler.load_state_dict(resume_ckpt["scheduler"])
                    accelerator.print("Scheduler state restored.")
                except Exception as exc:
                    accelerator.print(f"Could not restore scheduler state: {exc}")
        else:
            accelerator.print("Skipping optimizer/scheduler restore because --resume_optimizer=0.")

        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("step", 0))

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, run_name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    tb_dir = os.path.join(out_dir, "tb")
    logs_dir = os.path.join(out_dir, "logs")
    outputs_dir = os.path.join(out_dir, "outputs")

    writer = None
    if accelerator.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

    accelerator.wait_for_everyone()

    accelerator.print("===== TRAIN CONFIG =====")
    accelerator.print(f"Device: {device}")
    accelerator.print(f"Num processes: {accelerator.num_processes}")
    accelerator.print(f"Mixed precision: {mixed_precision}")
    accelerator.print(f"TF32 enabled: {args.allow_tf32 == 1 and torch.cuda.is_available()}")
    accelerator.print(f"Gradient checkpointing: {args.use_checkpoint == 1}")
    accelerator.print(f"TRELLIS model: {args.trellis_model}")
    accelerator.print(f"Separate cond: {args.separate_cond == 1}")
    accelerator.print(f"Separate cond gate: {args.separate_cond_gate}")
    accelerator.print(f"CFG drop probability: {args.cfg_drop_prob}")
    accelerator.print(f"Condition LR: {cond_lr}")
    accelerator.print(f"Flow LR: {flow_lr}")
    accelerator.print(f"Weight decay: {args.weight_decay}")
    accelerator.print(f"Grad clip: {args.grad_clip}")
    accelerator.print(f"Endpoint loss weight: {args.endpoint_loss_weight}")
    accelerator.print(f"Symmetry loss weight: {args.symmetry_loss_weight}")
    accelerator.print(f"Extra loss support in MorphFlow.forward: {supports_extra_losses}")
    accelerator.print(f"Dataset size: {len(dataset)}")
    if excluded_assets:
        accelerator.print(f"Excluded validation assets from train: {len(excluded_assets)}")
    if val_dataset is not None:
        accelerator.print(f"Validation dataset size: {len(val_dataset)}")
    accelerator.print(f"Run directory: {out_dir}")
    accelerator.print(f"Checkpoints in: {ckpt_dir}")
    accelerator.print(f"TensorBoard logs in: {tb_dir}")
    accelerator.print("========================")

    model.train()

    for epoch in range(start_epoch, args.train_epochs + 1):
        running_loss = 0.0

        progress_bar = tqdm(
            loader,
            desc=f"Epoch {epoch}/{args.train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            leave=True,
        )

        for batch_idx, batch in enumerate(progress_bar, start=1):
            optimizer.zero_grad(set_to_none=True)

            with accelerator.autocast():
                loss = compute_loss(
                    model=model,
                    batch=batch,
                    device=device,
                    supports_extra_losses=supports_extra_losses,
                    endpoint_loss_weight=args.endpoint_loss_weight,
                    symmetry_loss_weight=args.symmetry_loss_weight,
                    use_extra_losses=True,
                )

            accelerator.backward(loss)

            if args.grad_clip and args.grad_clip > 0.0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            reduced_loss = accelerator.reduce(loss.detach(), reduction="mean")
            loss_value = float(reduced_loss.item())

            running_loss += loss_value
            global_step += 1
            avg_loss = running_loss / batch_idx

            if writer is not None:
                writer.add_scalar("train/loss_step", loss_value, global_step)
                writer.add_scalar("train/loss_avg_epoch_running", avg_loss, global_step)
                writer.add_scalar("train/cond_lr", optimizer.param_groups[0]["lr"], global_step)

                if len(optimizer.param_groups) > 1:
                    writer.add_scalar("train/flow_lr", optimizer.param_groups[1]["lr"], global_step)

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
                    with accelerator.autocast():
                        val_loss = compute_loss(
                            model=model,
                            batch=val_batch,
                            device=device,
                            supports_extra_losses=supports_extra_losses,
                            endpoint_loss_weight=0.0,
                            symmetry_loss_weight=0.0,
                            use_extra_losses=False,
                        )

                    reduced_val_loss = accelerator.reduce(val_loss.detach(), reduction="mean")
                    val_loss_value = float(reduced_val_loss.item())

                    val_running_loss += val_loss_value
                    avg_val = val_running_loss / val_batch_idx

                    if accelerator.is_local_main_process:
                        val_bar.set_postfix(loss=f"{val_loss_value:.6f}", avg=f"{avg_val:.6f}")

            val_avg = val_running_loss / max(1, len(val_loader))
            accelerator.print(f"Epoch {epoch} validation completed. val_loss={val_avg:.6f}")
            model.train()

        if writer is not None:
            writer.add_scalar("train/loss_epoch", epoch_avg, epoch)
            if val_avg is not None:
                writer.add_scalar("val/loss_epoch", val_avg, epoch)
            writer.flush()

        accelerator.wait_for_everyone()

        save_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            ckpt_dir=ckpt_dir,
            epoch=epoch,
            global_step=global_step,
            train_loss=epoch_avg,
            val_loss=val_avg,
            final=False,
        )

    accelerator.wait_for_everyone()

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        ckpt_dir=ckpt_dir,
        epoch=args.train_epochs,
        global_step=global_step,
        train_loss=None,
        val_loss=None,
        final=True,
    )

    if writer is not None:
        writer.close()

    accelerator.print("Training completed.")


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)