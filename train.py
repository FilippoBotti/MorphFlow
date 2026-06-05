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
from models.lora import add_lora_to_attention
from models.morph_flow import MorphFlow
from models.morph_residual_flow import MorphResidualSSFlow
from models.morph_slat_flow import MorphSLatFlow


def build_parser():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument(
        "--root_dir",
        type=str,
        default=os.environ.get("ROOT_DIR", "/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v2"),
    )
    parser.add_argument("--metadata", type=str, default=os.environ.get("METADATA", "metadata_train.json"))
    parser.add_argument("--val_metadata", type=str, default=os.environ.get("VAL_METADATA", "metadata_val.json"))
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
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
        help="Save a regular epoch checkpoint every N epochs. Use 0 to disable.",
    )

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
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["cosine", "plateau"],
        default="cosine",
        help="cosine: step every batch. plateau: warm up, then ReduceLROnPlateau on validation loss.",
    )
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--plateau_patience", type=int, default=3)
    parser.add_argument("--plateau_threshold", type=float, default=1e-4)
    parser.add_argument("--plateau_min_lr", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="0 disables gradient clipping")

    # Model
    parser.add_argument("--trellis_model", type=str, choices=["text_base", "image_large"], default="text_base")
    parser.add_argument(
        "--flow_target",
        type=str,
        choices=["ss", "slat"],
        default="ss",
        help="Train the sparse-structure flow or the structured-latent flow.",
    )
    parser.add_argument(
        "--ss_flow_arch",
        type=str,
        choices=["standard", "residual_interp"],
        default="standard",
        help="SS flow architecture. residual_interp trains the flow in residual space over lerp(src1_ss, src2_ss, alpha).",
    )
    parser.add_argument(
        "--residual_interp_gate",
        type=str,
        choices=["none", "alpha"],
        default="alpha",
        help="For residual_interp: none uses target=base+residual; alpha uses target=base+alpha*(1-alpha)*residual.",
    )
    parser.add_argument(
        "--residual_interp_gate_min",
        type=float,
        default=1e-3,
        help="For residual_interp alpha gate: minimum divisor used while mapping target SS to residual space.",
    )
    parser.add_argument(
        "--residual_endpoint_prob",
        type=float,
        default=0.0,
        help="For residual_interp: probability of adding an auxiliary alpha=0/1 residual flow-matching batch.",
    )
    parser.add_argument(
        "--residual_endpoint_weight",
        type=float,
        default=1.0,
        help="For residual_interp: weight of the auxiliary alpha=0/1 residual flow-matching loss when active.",
    )
    parser.add_argument(
        "--residual_endpoint_max_items",
        type=int,
        default=1,
        help="For residual_interp: max batch items used by the auxiliary endpoint loss per GPU. Use 0 for full batch.",
    )
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
    parser.add_argument(
        "--endpoint_loss_prob",
        type=float,
        default=0.25,
        help="Probability of adding an alpha=0/1 endpoint flow-matching batch when endpoint loss is enabled.",
    )
    parser.add_argument("--symmetry_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--symmetry_loss_prob",
        type=float,
        default=1.0,
        help="Probability of adding src1/src2 alpha == src2/src1 1-alpha consistency when symmetry loss is enabled.",
    )

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

    # LoRA fine-tuning freezes the TRELLIS flow and trains LoRA adapters plus
    # MorphFlow conditioning modules. EMA remains accepted for script compatibility.
    parser.add_argument("--use_lora", type=int, choices=[0, 1], default=0)
    parser.add_argument("--lora_lr", type=float, default=None)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="to_q,to_kv")
    parser.add_argument(
        "--lora_attention_scope",
        type=str,
        choices=["all", "cross"],
        default="all",
        help="all preserves the legacy *_attn selection; cross only targets cross_attn/cross_attn2.",
    )
    parser.add_argument(
        "--trainable_scope",
        type=str,
        choices=["full", "cond_cross_attn"],
        default="full",
        help=(
            "Which non-LoRA parameters to train. full trains the whole model. "
            "cond_cross_attn trains MorphFlow conditioning modules plus TRELLIS "
            "cross-attention/alpha adapter parameters, freezing the rest of the flow."
        ),
    )
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


def load_metadata_entries(metadata_path: str) -> List[Dict[str, Any]]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if isinstance(entries, dict):
        entries = entries.get("samples", entries.get("metadata", []))
    if not isinstance(entries, list):
        raise ValueError(f"Metadata must be a list or contain a samples list: {metadata_path}")

    return entries


def load_assets_from_metadata(metadata_path: str, split: Optional[str] = None) -> set:
    entries = load_metadata_entries(metadata_path)

    assets = set()
    for entry in entries:
        if split is not None:
            entry_split = entry.get("split")
            if entry_split is not None and entry_split != split:
                continue
        src_1 = entry.get("src_1")
        src_2 = entry.get("src_2")
        if src_1:
            assets.add(str(src_1))
        if src_2:
            assets.add(str(src_2))

    return assets


def build_model(args, accelerator: Accelerator) -> torch.nn.Module:
    if args.flow_target == "slat":
        model_cls = MorphSLatFlow
    elif args.ss_flow_arch == "residual_interp":
        model_cls = MorphResidualSSFlow
    else:
        model_cls = MorphFlow

    requested_kwargs = {
        "model_type": args.trellis_model,
        "separate_cond": args.separate_cond == 1,
        "use_checkpoint": args.use_checkpoint == 1,
        "separate_cond_gate": args.separate_cond_gate,
        "cond_resample_tokens": args.cond_resample_tokens,
        "cond_resample_depth": args.cond_resample_depth,
        "cond_resample_heads": args.cond_resample_heads,
        "residual_interp_gate": args.residual_interp_gate,
        "residual_interp_gate_min": args.residual_interp_gate_min,
        "residual_endpoint_prob": args.residual_endpoint_prob,
        "residual_endpoint_weight": args.residual_endpoint_weight,
        "residual_endpoint_max_items": args.residual_endpoint_max_items,
    }

    signature = inspect.signature(model_cls.__init__)
    supported = set(signature.parameters.keys())

    model_kwargs = {
        key: value
        for key, value in requested_kwargs.items()
        if key in supported
    }

    ignored = sorted(set(requested_kwargs.keys()) - set(model_kwargs.keys()))
    if ignored:
        accelerator.print(f"{model_cls.__name__} does not support these constructor args; ignoring: {ignored}")

    accelerator.print(f"{model_cls.__name__} constructor kwargs:")
    for key, value in model_kwargs.items():
        accelerator.print(f"  {key}: {value}")

    model = model_cls(**model_kwargs)
    model.cfg_drop_prob = args.cfg_drop_prob
    return model


def model_forward_supports_extra_losses(model: torch.nn.Module) -> bool:
    signature = inspect.signature(model.forward)
    params = set(signature.parameters.keys())
    return "endpoint_loss_weight" in params and "symmetry_loss_weight" in params


def unwrap_model_for_attr(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def model_requires_source_ss_latents(model: torch.nn.Module) -> bool:
    return bool(getattr(unwrap_model_for_attr(model), "requires_source_ss_latents", False))


def model_forward_supports_source_ss_latents(model: torch.nn.Module) -> bool:
    signature = inspect.signature(unwrap_model_for_attr(model).forward)
    params = set(signature.parameters.keys())
    return "src1_ss_latent" in params and "src2_ss_latent" in params


def get_optional_tensor(batch: Dict[str, Any], key: str, device, dtype=torch.float32):
    value = batch.get(key, None)
    if value is None:
        return None
    return value.to(device=device, dtype=dtype, non_blocking=True)


def compute_loss(
    model,
    batch,
    device,
    flow_target: str,
    supports_extra_losses: bool,
    supports_source_ss_latents: bool,
    needs_source_ss_latents: bool,
    endpoint_loss_weight: float,
    symmetry_loss_weight: float,
    endpoint_loss_prob: float,
    symmetry_loss_prob: float,
    use_extra_losses: bool,
):
    src1_feats = batch["src1_feats"].to(device=device, dtype=torch.float32, non_blocking=True)
    src1_coords = batch["src1_coords"].to(device=device, dtype=torch.int32, non_blocking=True)

    src2_feats = batch["src2_feats"].to(device=device, dtype=torch.float32, non_blocking=True)
    src2_coords = batch["src2_coords"].to(device=device, dtype=torch.int32, non_blocking=True)

    alpha = batch["alpha"].to(device=device, dtype=torch.float32, non_blocking=True)

    if flow_target == "slat":
        target_feats = batch["target_feats"].to(device=device, dtype=torch.float32, non_blocking=True)
        target_coords = batch["target_coords"].to(device=device, dtype=torch.int32, non_blocking=True)
        return model(
            target_feats,
            target_coords,
            src1_feats,
            src1_coords,
            src2_feats,
            src2_coords,
            alpha,
        )

    target_ss_latent = batch["target_ss_latent"].to(device=device, dtype=torch.float32, non_blocking=True)

    source_kwargs = {}
    if supports_source_ss_latents and (
        needs_source_ss_latents
        or (supports_extra_losses and use_extra_losses and endpoint_loss_weight > 0.0)
    ):
        source_kwargs["src1_ss_latent"] = get_optional_tensor(batch, "src1_ss_latent", device)
        source_kwargs["src2_ss_latent"] = get_optional_tensor(batch, "src2_ss_latent", device)

    if supports_extra_losses and use_extra_losses:
        kwargs = {
            "endpoint_loss_weight": endpoint_loss_weight,
            "symmetry_loss_weight": symmetry_loss_weight,
            "endpoint_loss_prob": endpoint_loss_prob,
            "symmetry_loss_prob": symmetry_loss_prob,
            **source_kwargs,
        }

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
        **source_kwargs,
    )


def set_trainability(model: torch.nn.Module, args, accelerator: Accelerator):
    if args.use_lora == 1:
        if args.trainable_scope != "full":
            accelerator.print("WARNING: --trainable_scope is ignored when --use_lora=1.")
        for p in model.parameters():
            p.requires_grad = False

        flow = get_flow_module(model)
        if flow is None:
            raise RuntimeError("Cannot enable LoRA: model has no TRELLIS flow module.")

        lora_targets = tuple(
            item.strip()
            for item in args.lora_target_modules.split(",")
            if item.strip()
        )
        replaced = add_lora_to_attention(
            flow,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=lora_targets,
            attention_scope=args.lora_attention_scope,
        )
        model._lora_module_names = replaced
        accelerator.print(f"LoRA enabled on {len(replaced)} attention projections.")
        if not replaced:
            accelerator.print("WARNING: no LoRA modules were inserted. Check --lora_target_modules.")
    elif args.trainable_scope == "cond_cross_attn":
        for p in model.parameters():
            p.requires_grad = False
    else:
        # Full fine-tuning by default.
        for p in model.parameters():
            p.requires_grad = True

    if args.use_ema == 1:
        accelerator.print("WARNING: --use_ema is accepted for compatibility but ignored in this simplified train.py.")

    # cond_fusion is unused when separate_cond=1.
    if args.separate_cond == 1 and hasattr(model, "cond_fusion"):
        for p in model.cond_fusion.parameters():
            p.requires_grad = False
    elif hasattr(model, "cond_fusion"):
        for p in model.cond_fusion.parameters():
            p.requires_grad = True

    for name in ["cond_encoder", "separate_cond_proj", "cond_resampler"]:
        module = getattr(model, name, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True

    if args.use_lora == 1:
        flow = get_flow_module(model)
        if flow is not None:
            alpha_embedder = getattr(flow, "alpha_embedder", None)
            if alpha_embedder is not None:
                for p in alpha_embedder.parameters():
                    p.requires_grad = True
            for block in getattr(flow, "blocks", []):
                alpha_gate = getattr(block, "alpha_gate", None)
                if alpha_gate is not None:
                    for p in alpha_gate.parameters():
                        p.requires_grad = True
    elif args.trainable_scope == "cond_cross_attn":
        flow = get_flow_module(model)
        if flow is None:
            raise RuntimeError("Cannot use --trainable_scope cond_cross_attn: model has no TRELLIS flow module.")

        alpha_embedder = getattr(flow, "alpha_embedder", None)
        if alpha_embedder is not None:
            for p in alpha_embedder.parameters():
                p.requires_grad = True

        enabled_blocks = 0
        for block in getattr(flow, "blocks", []):
            block_enabled = False
            for module_name in ("cross_attn", "cross_attn2", "norm2", "norm4", "alpha_gate"):
                module = getattr(block, module_name, None)
                if module is None:
                    continue
                for p in module.parameters():
                    p.requires_grad = True
                block_enabled = True
            enabled_blocks += int(block_enabled)

        accelerator.print(f"Trainable scope cond_cross_attn: enabled cross-attention adapters in {enabled_blocks} flow blocks.")

    # null_cond is only useful when CFG dropout is enabled.
    if hasattr(model, "null_cond") and args.cfg_drop_prob <= 0.0:
        model.null_cond.requires_grad = False
    elif hasattr(model, "null_cond"):
        model.null_cond.requires_grad = True


def get_flow_module(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    flow = getattr(model, "sparse_structure_flow", None)
    if flow is not None:
        return flow
    return getattr(model, "slat_flow", None)


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

    lora_param_ids = set()
    lora_params = []
    if args.use_lora == 1:
        for name, p in model.named_parameters():
            if p.requires_grad and (".lora_A." in name or ".lora_B." in name):
                lora_params.append(p)
                lora_param_ids.add(id(p))

    flow_adapter_params = []
    if args.use_lora == 1:
        flow = get_flow_module(model)
        if flow is not None:
            for name, p in flow.named_parameters():
                if not p.requires_grad:
                    continue
                if id(p) in lora_param_ids:
                    continue
                flow_adapter_params.append(p)

    flow_params = []
    flow = get_flow_module(model)
    if flow is not None and args.use_lora != 1:
        for p in flow.parameters():
            if p.requires_grad and id(p) not in cond_param_ids:
                flow_params.append(p)

    param_groups = []
    if cond_params:
        param_groups.append({"params": cond_params, "lr": cond_lr, "name": "condition"})
    if lora_params:
        lora_lr = args.lr if args.lora_lr is None else args.lora_lr
        param_groups.append({"params": lora_params, "lr": lora_lr, "name": "lora"})
    if flow_adapter_params:
        param_groups.append({"params": flow_adapter_params, "lr": cond_lr, "name": "flow_adapter"})
    if flow_params:
        param_groups.append({"params": flow_params, "lr": flow_lr, "name": "flow"})

    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    return param_groups, cond_lr, flow_lr


def resolve_warmup_steps(requested_warmup_steps: int, total_training_steps: int) -> int:
    if requested_warmup_steps <= 0:
        return 0
    return min(requested_warmup_steps, max(1, total_training_steps // 10))


def remember_base_lrs(optimizer):
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group["lr"])


def restore_base_lrs(optimizer):
    for group in optimizer.param_groups:
        group["lr"] = group.get("base_lr", group["lr"])


def apply_plateau_warmup(optimizer, step: int, warmup_steps: int):
    if warmup_steps <= 0 or step > warmup_steps:
        return

    scale = float(step) / float(warmup_steps)
    for group in optimizer.param_groups:
        group["lr"] = group.get("base_lr", group["lr"]) * scale


def build_lr_scheduler(optimizer, args, total_training_steps: int, warmup_steps: int):
    remember_base_lrs(optimizer)

    if args.lr_scheduler == "cosine":
        from transformers import get_cosine_schedule_with_warmup

        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

    if args.lr_scheduler == "plateau":
        if warmup_steps > 0:
            for group in optimizer.param_groups:
                group["lr"] = 0.0

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            threshold=args.plateau_threshold,
            threshold_mode="rel",
            min_lr=args.plateau_min_lr,
        )

    raise ValueError(f"Unsupported lr scheduler: {args.lr_scheduler}")


def load_trellis_pretrained_if_needed(model, args, accelerator: Accelerator):
    if args.resume_from or args.init_from:
        mode = "--resume_from" if args.resume_from else "--init_from"
        accelerator.print(f"Skipping TRELLIS pretrained load because {mode} was provided.")
        return

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if args.trellis_model == "text_base":
        repo_id = "microsoft/TRELLIS-text-base"
        filename = (
            "ckpts/slat_flow_txt_dit_B_64l8p2_fp16.safetensors"
            if args.flow_target == "slat"
            else "ckpts/ss_flow_txt_dit_B_16l8_fp16.safetensors"
        )
    else:
        repo_id = "microsoft/TRELLIS-image-large"
        filename = (
            "ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors"
            if args.flow_target == "slat"
            else "ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors"
        )

    accelerator.print(f"Loading TRELLIS pretrained weights from {repo_id}...")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    trellis_state_dict = load_file(ckpt_path)

    flow = get_flow_module(model)
    if flow is None:
        raise RuntimeError("No flow module found for TRELLIS pretrained load.")

    missing, unexpected = flow.load_state_dict(
        trellis_state_dict,
        strict=False,
    )

    accelerator.print(
        f"TRELLIS load_state_dict strict=False | missing={len(missing)} unexpected={len(unexpected)}"
    )

    if args.separate_cond == 1:
        copied_cross = 0
        copied_norm = 0

        for block in flow.blocks:
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

    accelerator.print(f"TRELLIS weights loaded successfully: {args.trellis_model}/{args.flow_target}")


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
    best_val_loss: Optional[float] = None,
    best_epoch: Optional[int] = None,
    final: bool = False,
    best: bool = False,
):
    if not accelerator.is_main_process:
        return

    if best:
        filename = "morphflow_best.pt"
    elif final:
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
        "flow_target": args.flow_target,
        "sigma_min": accelerator.unwrap_model(model).sigma_min,
        "args": vars(args),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
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

    for name in ("endpoint_loss_prob", "symmetry_loss_prob"):
        value = getattr(args, name)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"--{name} must be in [0, 1], got {value}")
    if args.residual_endpoint_prob < 0.0 or args.residual_endpoint_prob > 1.0:
        raise ValueError(f"--residual_endpoint_prob must be in [0, 1], got {args.residual_endpoint_prob}")
    if args.residual_endpoint_weight < 0.0:
        raise ValueError(f"--residual_endpoint_weight must be >= 0, got {args.residual_endpoint_weight}")
    if args.residual_endpoint_max_items < 0:
        raise ValueError(f"--residual_endpoint_max_items must be >= 0, got {args.residual_endpoint_max_items}")
    if args.checkpoint_every < 0:
        raise ValueError(f"--checkpoint_every must be >= 0, got {args.checkpoint_every}")
    if args.residual_interp_gate_min <= 0.0:
        raise ValueError(f"--residual_interp_gate_min must be > 0, got {args.residual_interp_gate_min}")

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

    metadata_asset_overlap = set()
    train_metadata_asset_count = None
    val_metadata_asset_count = None
    if val_metadata_path and os.path.exists(val_metadata_path):
        train_metadata_assets = load_assets_from_metadata(metadata_path, split="train")
        val_metadata_assets = load_assets_from_metadata(val_metadata_path, split="val")
        metadata_asset_overlap = train_metadata_assets & val_metadata_assets
        train_metadata_asset_count = len(train_metadata_assets)
        val_metadata_asset_count = len(val_metadata_assets)

    dataset = MorphingDistillDataset(
        root=args.root_dir,
        metadata_file=metadata_path,
        split="train",
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
            split="val",
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
    load_trellis_pretrained_if_needed(model, args, accelerator)
    set_trainability(model, args, accelerator)

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
    supports_source_ss_latents = model_forward_supports_source_ss_latents(model)
    requires_source_ss_latents = model_requires_source_ss_latents(model)
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

    total_training_steps = args.train_epochs * len(loader)
    warmup_steps = resolve_warmup_steps(args.warmup_steps, total_training_steps)
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        args=args,
        total_training_steps=total_training_steps,
        warmup_steps=warmup_steps,
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
    best_val_loss = float("inf")
    best_epoch = 0
    optimizer_restored = False

    if resume_ckpt is not None:
        if args.resume_optimizer == 1:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
                remember_base_lrs(optimizer)
                optimizer_restored = True
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
        best_val_loss = float(resume_ckpt.get("best_val_loss", float("inf")))
        best_epoch = int(resume_ckpt.get("best_epoch", 0))

    if args.lr_scheduler == "plateau" and not optimizer_restored and global_step >= warmup_steps:
        restore_base_lrs(optimizer)

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
    accelerator.print(f"Flow target: {args.flow_target}")
    accelerator.print(f"SS flow architecture: {args.ss_flow_arch}")
    accelerator.print(f"Trainable scope: {args.trainable_scope}")
    if args.flow_target == "ss" and args.ss_flow_arch == "residual_interp":
        accelerator.print(f"Residual interpolation gate: {args.residual_interp_gate}")
        accelerator.print(f"Residual interpolation gate min: {args.residual_interp_gate_min}")
        accelerator.print(f"Residual endpoint probability: {args.residual_endpoint_prob}")
        accelerator.print(f"Residual endpoint weight: {args.residual_endpoint_weight}")
        accelerator.print(f"Residual endpoint max items: {args.residual_endpoint_max_items}")
    accelerator.print(f"Model requires source SS latents: {requires_source_ss_latents}")
    accelerator.print(f"Separate cond: {args.separate_cond == 1}")
    accelerator.print(f"Separate cond gate: {args.separate_cond_gate}")
    accelerator.print(f"CFG drop probability: {args.cfg_drop_prob}")
    if args.use_lora == 1:
        accelerator.print(f"LoRA attention scope: {args.lora_attention_scope}")
    accelerator.print(f"Condition LR: {cond_lr}")
    if args.use_lora == 1:
        lora_lr = args.lr if args.lora_lr is None else args.lora_lr
        accelerator.print(f"LoRA LR: {lora_lr}")
        accelerator.print("Flow LR: unused when --use_lora=1")
    else:
        accelerator.print(f"Flow LR: {flow_lr}")
    accelerator.print(f"LR scheduler: {args.lr_scheduler}")
    accelerator.print(f"Warmup steps: requested={args.warmup_steps} effective={warmup_steps}")
    if args.lr_scheduler == "plateau":
        accelerator.print(
            f"Plateau scheduler: factor={args.plateau_factor} "
            f"patience={args.plateau_patience} threshold={args.plateau_threshold} "
            f"min_lr={args.plateau_min_lr}"
        )
    for idx, group in enumerate(optimizer.param_groups):
        group_name = group.get("name", f"group_{idx}")
        accelerator.print(f"Optimizer LR [{idx}] {group_name}: {group['lr']}")
    accelerator.print(f"Weight decay: {args.weight_decay}")
    accelerator.print(f"Grad clip: {args.grad_clip}")
    accelerator.print(f"Checkpoint every: {args.checkpoint_every}")
    accelerator.print(f"Endpoint loss weight: {args.endpoint_loss_weight}")
    accelerator.print(f"Endpoint loss probability: {args.endpoint_loss_prob}")
    accelerator.print(f"Symmetry loss weight: {args.symmetry_loss_weight}")
    accelerator.print(f"Symmetry loss probability: {args.symmetry_loss_prob}")
    accelerator.print(f"Extra loss support in MorphFlow.forward: {supports_extra_losses}")
    accelerator.print(f"Dataset size: {len(dataset)}")
    if train_metadata_asset_count is not None and val_metadata_asset_count is not None:
        accelerator.print(f"Train metadata source assets: {train_metadata_asset_count}")
        accelerator.print(f"Validation metadata source assets: {val_metadata_asset_count}")
        accelerator.print(f"Train/validation source asset overlap: {len(metadata_asset_overlap)}")
        if metadata_asset_overlap:
            examples = sorted(metadata_asset_overlap)[:20]
            accelerator.print(f"Overlap examples: {examples}")
    if excluded_assets:
        accelerator.print(f"Excluded validation assets from train: {len(excluded_assets)}")
    if val_dataset is not None:
        accelerator.print(f"Validation dataset size: {len(val_dataset)}")
    else:
        accelerator.print("WARNING: validation is disabled; best-checkpoint saving will be skipped.")
        if args.lr_scheduler == "plateau":
            accelerator.print("WARNING: --lr_scheduler plateau requires validation to reduce LR.")
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
            if args.lr_scheduler == "plateau":
                apply_plateau_warmup(optimizer, global_step + 1, warmup_steps)

            optimizer.zero_grad(set_to_none=True)

            with accelerator.autocast():
                loss = compute_loss(
                    model=model,
                    batch=batch,
                    device=device,
                    flow_target=args.flow_target,
                    supports_extra_losses=supports_extra_losses,
                    supports_source_ss_latents=supports_source_ss_latents,
                    needs_source_ss_latents=requires_source_ss_latents,
                    endpoint_loss_weight=args.endpoint_loss_weight,
                    symmetry_loss_weight=args.symmetry_loss_weight,
                    endpoint_loss_prob=args.endpoint_loss_prob,
                    symmetry_loss_prob=args.symmetry_loss_prob,
                    use_extra_losses=True,
                )

            accelerator.backward(loss)

            if args.grad_clip and args.grad_clip > 0.0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            if args.lr_scheduler == "cosine":
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
                for group in optimizer.param_groups:
                    writer.add_scalar(f"train/{group.get('name', 'group')}_lr", group["lr"], global_step)

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
                            flow_target=args.flow_target,
                            supports_extra_losses=supports_extra_losses,
                            supports_source_ss_latents=supports_source_ss_latents,
                            needs_source_ss_latents=requires_source_ss_latents,
                            endpoint_loss_weight=0.0,
                            symmetry_loss_weight=0.0,
                            endpoint_loss_prob=0.0,
                            symmetry_loss_prob=0.0,
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

            if args.lr_scheduler == "plateau":
                lr_before = [group["lr"] for group in optimizer.param_groups]
                scheduler.step(val_avg)
                lr_after = [group["lr"] for group in optimizer.param_groups]
                if lr_after != lr_before:
                    accelerator.print(
                        "Plateau scheduler reduced LR: "
                        + ", ".join(
                            f"{before:.6e}->{after:.6e}"
                            for before, after in zip(lr_before, lr_after)
                        )
                    )

            if val_avg < best_val_loss:
                previous_best = best_val_loss
                best_val_loss = val_avg
                best_epoch = epoch
                accelerator.print(
                    f"Validation improved: {previous_best:.6f} -> {best_val_loss:.6f}. "
                    "Saving best checkpoint."
                )
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
                    best_val_loss=best_val_loss,
                    best_epoch=best_epoch,
                    best=True,
                )
            else:
                accelerator.print(
                    f"Validation did not improve. best_val_loss={best_val_loss:.6f} "
                    f"at epoch {best_epoch}; current_val_loss={val_avg:.6f}."
                )

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            accelerator.print(f"Saving periodic checkpoint at epoch {epoch}.")
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
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
            )

        if writer is not None:
            writer.add_scalar("train/loss_epoch", epoch_avg, epoch)
            if val_avg is not None:
                writer.add_scalar("val/loss_epoch", val_avg, epoch)
            writer.flush()

        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()

    if writer is not None:
        writer.close()

    accelerator.print("Training completed.")


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
