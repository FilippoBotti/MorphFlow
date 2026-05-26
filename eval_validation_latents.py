import argparse
import inspect
import json
import os
import re
import sys
from contextlib import nullcontext
from datetime import datetime
from glob import glob

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from trimesh.voxel.encoding import DenseEncoding

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

if os.environ.get("TRELLIS_REPO"):
    sys.path.append(os.environ["TRELLIS_REPO"])

from data.morph_dataset import MorphingDistillDataset, morphing_collate_fn
from models.morph_flow import MorphFlow
from models.morph_slat_flow import MorphSLatFlow
from models.lora import add_lora_to_attention, freeze_module


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        default=os.environ.get("ROOT_DIR", "/hpc/scratch/marco.barezzi/3d_dataset/morphing_dataset_v2"),
    )
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--metadata",
        type=str,
        default=os.environ.get("TEST_METADATA", os.environ.get("METADATA")),
        help="Metadata file to evaluate. Defaults to metadata_<split>.json.",
    )
    parser.add_argument(
        "--val_metadata",
        type=str,
        default=None,
        help="Deprecated alias kept for old scripts. Prefer --metadata with --split.",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Explicit checkpoint path. If omitted, the latest checkpoint under --checkpoints_root is used.",
    )
    parser.add_argument(
        "--checkpoints_root",
        type=str,
        default=os.environ.get("CHECKPOINTS_ROOT", "./outputs/morphflow"),
        help="Directory searched recursively for best/epoch/final checkpoints when --checkpoint_path is omitted.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "./outputs/eval_test_latents"),
    )
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument(
        "--selection_unit",
        type=str,
        choices=["pair", "entry"],
        default="pair",
        help="pair selects unique (src_1, src_2) pairs; entry samples raw metadata rows.",
    )
    parser.add_argument(
        "--selection_file",
        type=str,
        default=None,
        help="Optional JSON with indices from a previous run. If omitted, selection is seed-deterministic.",
    )
    parser.add_argument(
        "--compare_alpha",
        type=float,
        default=None,
        help="When selecting by pair, choose the available alpha closest to this value.",
    )
    parser.add_argument("--steps", type=int, default=50)

    parser.add_argument(
        "--trellis_model",
        type=str,
        choices=["auto", "text_base", "image_large"],
        default="auto",
    )
    parser.add_argument("--use_ema", type=int, choices=[0, 1], default=0)

    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument(
        "--strict_load",
        type=int,
        choices=[0, 1],
        default=1,
        help="Use 0 only when evaluating an older checkpoint after architecture changes.",
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default="bf16",
        help="Autocast precision used during sampling and decoding.",
    )
    parser.add_argument("--allow_tf32", type=int, choices=[0, 1], default=1)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--tensorboard",
        type=int,
        choices=[0, 1],
        default=1,
        help="Write scalar metrics and a text manifest under output_dir/tb.",
    )

    parser.add_argument("--export_ss_glb", type=int, choices=[0, 1], default=1)
    parser.add_argument("--export_slat_glb", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save_latents", type=int, choices=[0, 1], default=1)

    parser.add_argument(
        "--cfg_drop_prob",
        type=float,
        default=None,
        help="Ignored. Accepted only for compatibility with old eval.sh files.",
    )

    return parser.parse_args()


def safe_slug(value, max_len=80):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    return value[:max_len].strip("_") or "unknown"


def find_latest_checkpoint(checkpoints_root):
    candidates = []
    for filename in ("morphflow_best.pt", "morphflow_epoch_*.pt", "morphflow_final_*.pt"):
        pattern = os.path.join(checkpoints_root, "**", filename)
        candidates.extend(glob(pattern, recursive=True))

    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under: {checkpoints_root}")

    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def resolve_metadata_file(args):
    if args.metadata:
        return args.metadata
    if args.val_metadata:
        return args.val_metadata
    return f"metadata_{args.split}.json"


def detect_flow_target(ckpt):
    if isinstance(ckpt, dict):
        if ckpt.get("flow_target") in ("ss", "slat"):
            return ckpt["flow_target"]
        if isinstance(ckpt.get("args"), dict):
            flow_target = ckpt["args"].get("flow_target")
            if flow_target in ("ss", "slat"):
                return flow_target
    return "ss"


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "module"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    cleaned = {}
    for key, value in obj.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value

    return cleaned


def autocast_context(device, mixed_precision):
    if device.type != "cuda" or mixed_precision == "no":
        return nullcontext()

    if mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    if mixed_precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    return nullcontext()


def build_morphflow_from_checkpoint(ckpt, model_type, flow_target):
    ckpt_args = {}
    if isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict):
        ckpt_args = ckpt["args"]

    separate_cond = bool(int(ckpt_args.get("separate_cond", 0)))
    use_checkpoint = False
    model_cls = MorphSLatFlow if flow_target == "slat" else MorphFlow

    requested_kwargs = {
        "model_type": model_type,
        "separate_cond": separate_cond,
        "use_checkpoint": use_checkpoint,

        # These are used only if your local MorphFlow implementation supports them.
        "separate_cond_gate": ckpt_args.get("separate_cond_gate", "alpha_residual"),
        "cond_resample_tokens": int(ckpt_args.get("cond_resample_tokens", 0)),
        "cond_resample_depth": int(ckpt_args.get("cond_resample_depth", 1)),
        "cond_resample_heads": int(ckpt_args.get("cond_resample_heads", 8)),
        "normalize_flow_latents": bool(ckpt_args.get("normalize_flow_latents", True)),
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
        print(f"Ignoring constructor args not supported by current {model_cls.__name__}: {ignored}")

    print(f"{model_cls.__name__} constructor kwargs:")
    for key, value in model_kwargs.items():
        print(f"  {key}: {value}")

    model = model_cls(**model_kwargs)
    return model, ckpt_args


def get_flow_module(model):
    flow = getattr(model, "sparse_structure_flow", None)
    if flow is not None:
        return flow
    return getattr(model, "slat_flow", None)


def maybe_insert_lora(model, ckpt_args):
    if int(ckpt_args.get("use_lora", 0)) != 1:
        return []

    flow = get_flow_module(model)
    if flow is None:
        raise RuntimeError("Checkpoint has use_lora=1, but no TRELLIS flow module was found.")

    freeze_module(flow)

    lora_targets = ckpt_args.get("lora_target_modules", "to_q,to_kv")
    lora_targets = tuple(x.strip() for x in lora_targets.split(",") if x.strip())

    lora_modules = add_lora_to_attention(
        flow,
        rank=int(ckpt_args.get("lora_rank", 8)),
        alpha=int(ckpt_args.get("lora_alpha", 16)),
        dropout=float(ckpt_args.get("lora_dropout", 0.0)),
        target_modules=lora_targets,
    )

    print(f"LoRA enabled for eval on {len(lora_modules)} attention modules")
    if len(lora_modules) == 0:
        print("WARNING: checkpoint says use_lora=1, but no LoRA modules were inserted.")

    return lora_modules


@torch.no_grad()
def run_reverse_flow_sample(
    model,
    x0_shape,
    src1_feats,
    src2_feats,
    src1_coords,
    src2_coords,
    alpha,
    steps,
    device,
    cfg_scale=1.0,
    mixed_precision="bf16",
):
    B = x0_shape[0]

    x_t = torch.randn(x0_shape, device=device, dtype=torch.float32)
    t_seq = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)

    src1_feats = src1_feats.to(device=device, dtype=torch.float32, non_blocking=True)
    src2_feats = src2_feats.to(device=device, dtype=torch.float32, non_blocking=True)
    src1_coords = src1_coords.to(device=device, dtype=torch.int32, non_blocking=True)
    src2_coords = src2_coords.to(device=device, dtype=torch.int32, non_blocking=True)
    alpha = alpha.reshape(B).to(device=device, dtype=torch.float32, non_blocking=True)

    for i in range(steps):
        t_curr = t_seq[i]
        t_next = t_seq[i + 1]
        dt = t_curr - t_next

        t_batch = torch.full(
            (B,),
            float(t_curr.item()),
            device=device,
            dtype=torch.float32,
        )

        with autocast_context(device, mixed_precision):
            if cfg_scale == 1.0:
                # Same alpha convention as training: metadata alpha is passed
                # unchanged with src1 as the first endpoint and src2 as the second.
                v_pred = model.forward_flow(
                    x_t,
                    t_batch,
                    src1_feats,
                    src2_feats,
                    src1_coords,
                    src2_coords,
                    alpha,
                )
            else:
                v_pred = model.forward_flow_cfg(
                    x_t,
                    t_batch,
                    src1_feats,
                    src2_feats,
                    src1_coords,
                    src2_coords,
                    alpha,
                    guidance_scale=cfg_scale,
                )

        x_t = x_t - dt * v_pred.float()

    return x_t


@torch.no_grad()
def run_reverse_slat_flow_sample(
    model,
    target_feats,
    target_coords,
    src1_feats,
    src2_feats,
    src1_coords,
    src2_coords,
    alpha,
    steps,
    device,
    cfg_scale=1.0,
    mixed_precision="bf16",
):
    target_feats = target_feats.to(device=device, dtype=torch.float32, non_blocking=True)
    target_coords = target_coords.to(device=device, dtype=torch.int32, non_blocking=True)
    src1_feats = src1_feats.to(device=device, dtype=torch.float32, non_blocking=True)
    src2_feats = src2_feats.to(device=device, dtype=torch.float32, non_blocking=True)
    src1_coords = src1_coords.to(device=device, dtype=torch.int32, non_blocking=True)
    src2_coords = src2_coords.to(device=device, dtype=torch.int32, non_blocking=True)

    x_0 = model.make_slat(target_feats, target_coords)
    x_0 = model.normalize_slat(x_0)
    B = x_0.shape[0]

    alpha = alpha.reshape(B).to(device=device, dtype=torch.float32, non_blocking=True)
    x_t = x_0.replace(torch.randn_like(x_0.feats))
    t_seq = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)

    for i in range(steps):
        t_curr = t_seq[i]
        t_next = t_seq[i + 1]
        dt = t_curr - t_next

        t_batch = torch.full(
            (B,),
            float(t_curr.item()),
            device=device,
            dtype=torch.float32,
        )

        with autocast_context(device, mixed_precision):
            if cfg_scale == 1.0:
                # Same alpha convention as training: metadata alpha is passed
                # unchanged with src1 as the first endpoint and src2 as the second.
                v_pred = model.forward_flow(
                    x_t,
                    t_batch,
                    src1_feats,
                    src2_feats,
                    src1_coords,
                    src2_coords,
                    alpha,
                )
            else:
                v_pred = model.forward_flow_cfg(
                    x_t,
                    t_batch,
                    src1_feats,
                    src2_feats,
                    src1_coords,
                    src2_coords,
                    alpha,
                    guidance_scale=cfg_scale,
                )

        x_t = x_t - dt * v_pred.float()

    return model.denormalize_slat(x_t), x_t, x_0


def ensure_batch_coords(coords):
    if coords.shape[-1] == 3:
        b = torch.zeros(
            (coords.shape[0], 1),
            dtype=coords.dtype,
            device=coords.device,
        )
        return torch.cat([b, coords], dim=-1)

    return coords


@torch.no_grad()
def save_slat_glb(mesh_decoder, sparse_tensor_cls, feats, coords, out_path, device, mixed_precision):
    coords = ensure_batch_coords(coords).to(device=device, dtype=torch.int32)
    feats = feats.to(device=device, dtype=torch.float32)

    st = sparse_tensor_cls(feats=feats, coords=coords)

    with autocast_context(device, mixed_precision):
        mesh_out = mesh_decoder(st)[0]

    if not getattr(mesh_out, "success", False):
        print(f"Failed to decode SLAT mesh: {out_path}")
        return False

    mesh = trimesh.Trimesh(
        vertices=mesh_out.vertices.detach().float().cpu().numpy(),
        faces=mesh_out.faces.detach().cpu().numpy(),
        process=False,
    )
    mesh.export(out_path)
    return True


@torch.no_grad()
def decode_ss_logits(ss_decoder, latent, device, mixed_precision):
    latent = latent.to(device=device, dtype=torch.float32)

    if latent.ndim == 6:
        latent = latent.squeeze(1)

    with autocast_context(device, mixed_precision):
        logits = ss_decoder(latent)

    if isinstance(logits, (list, tuple)):
        logits = logits[0]

    logits = logits.float()

    # Common shapes:
    #   [B, C, D, H, W]
    #   [B, D, H, W]
    #   [D, H, W]
    if logits.ndim == 5:
        logits = logits[0]
    if logits.ndim == 4:
        logits = logits[0]

    return logits


@torch.no_grad()
def save_ss_glb(ss_decoder, latent, out_path, device, mixed_precision):
    logits = decode_ss_logits(ss_decoder, latent, device, mixed_precision)
    voxels = (logits > 0).detach().cpu().numpy().astype(bool)

    occupied = int(voxels.sum())
    total = int(voxels.size)

    if occupied == 0:
        print(f"Skipping empty voxel mesh: {out_path}")
        return {
            "saved": False,
            "occupied": occupied,
            "total": total,
            "occupancy_ratio": 0.0,
        }

    try:
        vg = trimesh.voxel.VoxelGrid(DenseEncoding(voxels))
        mesh = vg.marching_cubes
        mesh.export(out_path)
        saved = True
    except Exception as exc:
        print(f"Failed to export voxel mesh {out_path}: {exc}")
        saved = False

    return {
        "saved": saved,
        "occupied": occupied,
        "total": total,
        "occupancy_ratio": occupied / max(total, 1),
    }


def voxel_iou_from_logits(pred_logits, target_logits):
    pred = pred_logits > 0
    target = target_logits > 0

    intersection = torch.logical_and(pred, target).sum().item()
    union = torch.logical_or(pred, target).sum().item()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def _metadata_pair_key(entry):
    return str(entry.get("src_1")), str(entry.get("src_2"))


def select_eval_indices(dataset, args):
    if args.selection_file:
        with open(args.selection_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        indices = payload.get("indices", payload)
        indices = [int(idx) for idx in indices]
        return indices[:args.num_samples]

    rng = np.random.default_rng(args.seed)
    metadata = list(getattr(dataset, "metadata", []))
    num_samples = min(args.num_samples, len(metadata))

    if args.selection_unit == "entry":
        if num_samples >= len(metadata):
            return list(range(len(metadata)))
        return [int(idx) for idx in rng.choice(len(metadata), size=num_samples, replace=False).tolist()]

    pair_to_indices = {}
    for idx, entry in enumerate(metadata):
        pair_to_indices.setdefault(_metadata_pair_key(entry), []).append(idx)

    pairs = list(pair_to_indices.keys())
    if num_samples >= len(pairs):
        selected_pairs = pairs
    else:
        selected_pair_positions = rng.choice(len(pairs), size=num_samples, replace=False).tolist()
        selected_pairs = [pairs[int(pos)] for pos in selected_pair_positions]

    selected_indices = []
    for pair in selected_pairs:
        candidates = pair_to_indices[pair]
        if args.compare_alpha is not None:
            best_idx = min(
                candidates,
                key=lambda idx: abs(float(metadata[idx].get("alpha", 0.0)) - args.compare_alpha),
            )
        else:
            best_idx = int(rng.choice(candidates))
        selected_indices.append(best_idx)

    return selected_indices


def validate_alpha(alpha, sample_name):
    alpha_value = float(alpha.reshape(-1)[0].detach().cpu().item())
    if not np.isfinite(alpha_value):
        raise ValueError(f"Non-finite alpha for {sample_name}: {alpha_value}")
    if alpha_value < 0.0 or alpha_value > 1.0:
        print(f"WARNING: alpha outside [0, 1] for {sample_name}: {alpha_value}")
    return alpha_value


def maybe_load_trellis_decoders(args, device, flow_target):
    need_ss_decoder = args.export_ss_glb == 1 and flow_target == "ss"
    need_mesh_decoder = args.export_slat_glb == 1

    ss_decoder = None
    mesh_decoder = None
    sparse_tensor_cls = None

    if not need_ss_decoder and not need_mesh_decoder:
        return ss_decoder, mesh_decoder, sparse_tensor_cls

    from trellis.models import from_pretrained as trellis_from_pretrained

    if need_mesh_decoder:
        from trellis.modules.sparse.basic import SparseTensor
        sparse_tensor_cls = SparseTensor

    if need_ss_decoder:
        print("Loading TRELLIS sparse-structure decoder...")
        ss_decoder = trellis_from_pretrained(
            "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
        ).to(device)
        ss_decoder.eval()

    if need_mesh_decoder:
        print("Loading TRELLIS SLAT mesh decoder...")
        mesh_decoder = trellis_from_pretrained(
            "microsoft/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16"
        ).to(device)
        mesh_decoder.eval()

    return ss_decoder, mesh_decoder, sparse_tensor_cls


def main():
    args = parse_args()

    if args.allow_tf32 == 1 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = args.checkpoint_path or find_latest_checkpoint(args.checkpoints_root)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    flow_target = detect_flow_target(ckpt)
    metadata_file = resolve_metadata_file(args)
    metadata_path = metadata_file if os.path.isabs(metadata_file) else os.path.join(args.root_dir, metadata_file)

    model_type = args.trellis_model
    if model_type == "auto":
        model_type = "text_base"
        if isinstance(ckpt, dict):
            model_type = ckpt.get("model_type", model_type)
            if isinstance(ckpt.get("args"), dict):
                model_type = ckpt["args"].get("trellis_model", model_type)

    print("===== EVAL CONFIG =====")
    print(f"device: {device}")
    print(f"mixed_precision: {args.mixed_precision}")
    print(f"flow_target: {flow_target}")
    print(f"cfg_scale: {args.cfg_scale}")
    print(f"steps: {args.steps}")
    print(f"seed: {args.seed}")
    print(f"split: {args.split}")
    print(f"metadata: {metadata_path}")
    print(f"selection_unit: {args.selection_unit}")
    print(f"num_samples: {args.num_samples}")
    print(f"checkpoint: {ckpt_path}")
    print(f"output_dir: {output_dir}")
    print("Alpha convention: metadata alpha is passed unchanged to MorphFlow(src1, src2, alpha).")
    print("=======================")

    print(f"Using TRELLIS/MorphFlow model_type: {model_type}")

    model, ckpt_args = build_morphflow_from_checkpoint(ckpt, model_type, flow_target)
    maybe_insert_lora(model, ckpt_args)

    if args.use_ema == 1 and isinstance(ckpt, dict) and "model_ema" in ckpt:
        print("Loading EMA weights from checkpoint")
        state_to_load = ckpt["model_ema"]
    else:
        print("Loading regular model weights from checkpoint")
        state_to_load = ckpt

    state_dict = unwrap_state_dict(state_to_load)

    load_result = model.load_state_dict(
        state_dict,
        strict=bool(args.strict_load),
    )

    if not bool(args.strict_load):
        print("Missing keys:")
        for key in load_result.missing_keys:
            print(f"  {key}")

        print("Unexpected keys:")
        for key in load_result.unexpected_keys:
            print(f"  {key}")

    model = model.to(device)
    model.eval()

    ss_decoder, mesh_decoder, sparse_tensor_cls = maybe_load_trellis_decoders(args, device, flow_target)

    eval_dataset = MorphingDistillDataset(
        root=args.root_dir,
        metadata_file=metadata_path,
        split=args.split,
        verbose=False,
    )

    selected_indices = select_eval_indices(eval_dataset, args)
    selection_manifest = {
        "split": args.split,
        "metadata": metadata_path,
        "selection_unit": args.selection_unit,
        "seed": args.seed,
        "compare_alpha": args.compare_alpha,
        "indices": selected_indices,
        "samples": [],
    }
    for idx in selected_indices:
        entry = eval_dataset.metadata[idx]
        selection_manifest["samples"].append(
            {
                "dataset_index": int(idx),
                "src1_name": str(entry.get("src_1")),
                "src2_name": str(entry.get("src_2")),
                "target_name": str(entry.get("target")),
                "alpha": float(entry.get("alpha")),
            }
        )

    with open(os.path.join(output_dir, "selected_samples.json"), "w", encoding="utf-8") as f:
        json.dump(selection_manifest, f, indent=2)

    writer = None
    if args.tensorboard == 1:
        if SummaryWriter is None:
            print("WARNING: tensorboard is not installed; TensorBoard logging is disabled.")
        else:
            writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))
            writer.add_text("eval/config", json.dumps({
                "checkpoint": ckpt_path,
                "flow_target": flow_target,
                "model_type": model_type,
                "split": args.split,
                "metadata": metadata_path,
                "cfg_scale": args.cfg_scale,
                "steps": args.steps,
                "seed": args.seed,
                "selection_unit": args.selection_unit,
                "num_samples": args.num_samples,
                "compare_alpha": args.compare_alpha,
            }, indent=2))
            writer.add_text("eval/selected_samples", json.dumps(selection_manifest, indent=2))

    all_metrics = []

    with torch.no_grad():
        for processed, dataset_idx in enumerate(selected_indices):
            batch = morphing_collate_fn([eval_dataset[dataset_idx]])

            src1_name = batch.get("src1_name", [f"src1_{processed}"])[0]
            src2_name = batch.get("src2_name", [f"src2_{processed}"])[0]
            tgt_name = batch.get("target_name", [f"target_{processed}"])[0]

            sample_name = (
                f"sample_{processed:03d}_"
                f"{safe_slug(src1_name, 36)}_"
                f"{safe_slug(src2_name, 36)}_"
                f"a{float(batch['alpha'].reshape(-1)[0]):.4f}"
            )
            sample_dir = os.path.join(output_dir, sample_name)
            os.makedirs(sample_dir, exist_ok=True)

            src1_feats = batch["src1_feats"]
            src1_coords = batch["src1_coords"]
            src2_feats = batch["src2_feats"]
            src2_coords = batch["src2_coords"]

            alpha = batch["alpha"].to(device=device, dtype=torch.float32)
            alpha_value = validate_alpha(alpha, sample_name)

            metrics = {
                "sample_index": processed,
                "dataset_index": int(dataset_idx),
                "flow_target": flow_target,
                "src1_name": str(src1_name),
                "src2_name": str(src2_name),
                "target_name": str(tgt_name),
                "alpha": alpha_value,
                "sample_dir": sample_dir,
                "objects": {
                    "src1_slat_glb": os.path.join(sample_dir, "src1.glb"),
                    "src2_slat_glb": os.path.join(sample_dir, "src2.glb"),
                    "target_teacher_slat_glb": os.path.join(sample_dir, "target_teacher_slat.glb"),
                    "pred_morph_slat_glb": (
                        os.path.join(sample_dir, "pred_morph_slat.glb")
                        if flow_target == "slat"
                        else None
                    ),
                    "target_teacher_voxels_glb": (
                        os.path.join(sample_dir, "target_teacher_voxels.glb")
                        if flow_target == "ss"
                        else None
                    ),
                    "pred_student_voxels_glb": (
                        os.path.join(sample_dir, "pred_student_voxels.glb")
                        if flow_target == "ss"
                        else None
                    ),
                },
            }

            target_ss = None
            x0_pred = None
            pred_slat = None
            pred_slat_norm = None
            target_slat_norm = None

            if flow_target == "ss":
                target_ss = batch["target_ss_latent"].to(device=device, dtype=torch.float32)

                if target_ss.ndim == 6:
                    target_ss = target_ss.squeeze(1)

                x0_pred = run_reverse_flow_sample(
                    model=model,
                    x0_shape=target_ss.shape,
                    src1_feats=src1_feats,
                    src2_feats=src2_feats,
                    src1_coords=src1_coords,
                    src2_coords=src2_coords,
                    alpha=alpha,
                    steps=args.steps,
                    device=device,
                    cfg_scale=args.cfg_scale,
                    mixed_precision=args.mixed_precision,
                )

                metrics["latent_mse"] = float(F.mse_loss(x0_pred.float(), target_ss.float()).item())
                metrics["latent_l1"] = float(F.l1_loss(x0_pred.float(), target_ss.float()).item())

            elif flow_target == "slat":
                pred_slat, pred_slat_norm, target_slat_norm = run_reverse_slat_flow_sample(
                    model=model,
                    target_feats=batch["target_feats"],
                    target_coords=batch["target_coords"],
                    src1_feats=src1_feats,
                    src2_feats=src2_feats,
                    src1_coords=src1_coords,
                    src2_coords=src2_coords,
                    alpha=alpha,
                    steps=args.steps,
                    device=device,
                    cfg_scale=args.cfg_scale,
                    mixed_precision=args.mixed_precision,
                )

                target_feats = batch["target_feats"].to(device=device, dtype=torch.float32)
                target_coords = batch["target_coords"].to(device=device, dtype=torch.int32)
                metrics["slat_feat_mse"] = float(F.mse_loss(pred_slat.feats.float(), target_feats.float()).item())
                metrics["slat_feat_l1"] = float(F.l1_loss(pred_slat.feats.float(), target_feats.float()).item())
                metrics["slat_norm_feat_mse"] = float(
                    F.mse_loss(pred_slat_norm.feats.float(), target_slat_norm.feats.float()).item()
                )
                metrics["slat_norm_feat_l1"] = float(
                    F.l1_loss(pred_slat_norm.feats.float(), target_slat_norm.feats.float()).item()
                )
                metrics["target_slat_points"] = int(target_feats.shape[0])
                metrics["pred_target_coords_equal"] = bool(torch.equal(pred_slat.coords.cpu(), target_coords.cpu()))
            else:
                raise ValueError(f"Unsupported flow_target: {flow_target}")

            if args.save_latents == 1:
                payload = {
                    "flow_target": flow_target,
                    "alpha": alpha.detach().cpu(),
                    "src1_name": src1_name,
                    "src2_name": src2_name,
                    "target_name": tgt_name,
                    "dataset_index": int(dataset_idx),
                    "checkpoint": ckpt_path,
                    "cfg_scale": args.cfg_scale,
                    "steps": args.steps,
                }
                if flow_target == "ss":
                    payload["pred_ss_latent"] = x0_pred.detach().cpu()
                    payload["target_ss_latent"] = target_ss.detach().cpu()
                else:
                    payload["pred_slat_feats"] = pred_slat.feats.detach().cpu()
                    payload["pred_slat_coords"] = pred_slat.coords.detach().cpu()
                    payload["target_slat_feats"] = batch["target_feats"].detach().cpu()
                    payload["target_slat_coords"] = batch["target_coords"].detach().cpu()
                    payload["pred_slat_norm_feats"] = pred_slat_norm.feats.detach().cpu()
                    payload["target_slat_norm_feats"] = target_slat_norm.feats.detach().cpu()

                torch.save(payload, os.path.join(sample_dir, "latents.pt"))

            if args.export_slat_glb == 1 and mesh_decoder is not None:
                try:
                    save_slat_glb(
                        mesh_decoder,
                        sparse_tensor_cls,
                        src1_feats,
                        src1_coords,
                        os.path.join(sample_dir, "src1.glb"),
                        device,
                        args.mixed_precision,
                    )
                    save_slat_glb(
                        mesh_decoder,
                        sparse_tensor_cls,
                        src2_feats,
                        src2_coords,
                        os.path.join(sample_dir, "src2.glb"),
                        device,
                        args.mixed_precision,
                    )

                    if "target_feats" in batch and "target_coords" in batch:
                        save_slat_glb(
                            mesh_decoder,
                            sparse_tensor_cls,
                            batch["target_feats"],
                            batch["target_coords"],
                            os.path.join(sample_dir, "target_teacher_slat.glb"),
                            device,
                            args.mixed_precision,
                        )
                    else:
                        print("target_feats/target_coords not found; skipping target_teacher_slat.glb")

                    if flow_target == "slat" and pred_slat is not None:
                        saved = save_slat_glb(
                            mesh_decoder,
                            sparse_tensor_cls,
                            pred_slat.feats,
                            pred_slat.coords,
                            os.path.join(sample_dir, "pred_morph_slat.glb"),
                            device,
                            args.mixed_precision,
                        )
                        metrics["pred_slat_glb_saved"] = bool(saved)

                except Exception as exc:
                    print(f"SLAT export failed for {sample_name}: {exc}")

            if flow_target == "ss" and args.export_ss_glb == 1 and ss_decoder is not None:
                try:
                    target_logits = decode_ss_logits(
                        ss_decoder,
                        target_ss,
                        device,
                        args.mixed_precision,
                    )
                    pred_logits = decode_ss_logits(
                        ss_decoder,
                        x0_pred,
                        device,
                        args.mixed_precision,
                    )

                    metrics["voxel_iou"] = voxel_iou_from_logits(pred_logits, target_logits)

                    target_stats = save_ss_glb(
                        ss_decoder,
                        target_ss,
                        os.path.join(sample_dir, "target_teacher_voxels.glb"),
                        device,
                        args.mixed_precision,
                    )
                    pred_stats = save_ss_glb(
                        ss_decoder,
                        x0_pred,
                        os.path.join(sample_dir, "pred_student_voxels.glb"),
                        device,
                        args.mixed_precision,
                    )

                    metrics["target_occupancy_ratio"] = target_stats["occupancy_ratio"]
                    metrics["pred_occupancy_ratio"] = pred_stats["occupancy_ratio"]
                    metrics["target_occupied"] = target_stats["occupied"]
                    metrics["pred_occupied"] = pred_stats["occupied"]

                except Exception as exc:
                    print(f"SS export/metrics failed for {sample_name}: {exc}")

            with open(os.path.join(sample_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            all_metrics.append(metrics)

            if writer is not None:
                for key, value in metrics.items():
                    if isinstance(value, bool):
                        writer.add_scalar(f"sample/{key}", float(value), processed)
                    elif isinstance(value, (int, float)):
                        writer.add_scalar(f"sample/{key}", float(value), processed)
                writer.add_text(
                    f"samples/{processed:03d}",
                    json.dumps(
                        {
                            "dataset_index": int(dataset_idx),
                            "src1_name": str(src1_name),
                            "src2_name": str(src2_name),
                            "target_name": str(tgt_name),
                            "alpha": alpha_value,
                            "sample_dir": sample_dir,
                        },
                        indent=2,
                    ),
                    processed,
                )

            primary_metric = metrics.get("latent_mse", metrics.get("slat_feat_mse", float("nan")))
            secondary_metric = metrics.get("latent_l1", metrics.get("slat_feat_l1", float("nan")))
            print(
                f"[{processed + 1}/{args.num_samples}] "
                f"{sample_name} | "
                f"alpha={metrics['alpha']:.4f} | "
                f"mse={primary_metric:.6f} | "
                f"l1={secondary_metric:.6f} | "
                f"voxel_iou={metrics.get('voxel_iou', 'n/a')}"
            )

    summary = {
        "checkpoint": ckpt_path,
        "output_dir": output_dir,
        "flow_target": flow_target,
        "model_type": model_type,
        "split": args.split,
        "metadata": metadata_path,
        "selection_unit": args.selection_unit,
        "selection_seed": args.seed,
        "selection_file": os.path.join(output_dir, "selected_samples.json"),
        "num_samples": len(all_metrics),
        "cfg_scale": args.cfg_scale,
        "steps": args.steps,
        "mixed_precision": args.mixed_precision,
        "metrics": all_metrics,
    }

    metric_keys = [
        "latent_mse",
        "latent_l1",
        "voxel_iou",
        "target_occupancy_ratio",
        "pred_occupancy_ratio",
        "slat_feat_mse",
        "slat_feat_l1",
        "slat_norm_feat_mse",
        "slat_norm_feat_l1",
        "target_slat_points",
        "alpha",
    ]

    if all_metrics:
        for key in metric_keys:
            values = [
                float(m[key])
                for m in all_metrics
                if key in m and isinstance(m[key], (int, float))
            ]
            if values:
                summary[f"mean_{key}"] = float(np.mean(values))
                summary[f"std_{key}"] = float(np.std(values))

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if writer is not None:
        for key, value in summary.items():
            if key.startswith("mean_") or key.startswith("std_"):
                writer.add_scalar(f"summary/{key}", float(value), 0)
        for key in metric_keys:
            values = [
                float(m[key])
                for m in all_metrics
                if key in m and isinstance(m[key], (int, float))
            ]
            if values:
                writer.add_histogram(f"distribution/{key}", torch.tensor(values, dtype=torch.float32), 0)
        writer.add_text("eval/summary", json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2))
        writer.flush()
        writer.close()

    print("===== SUMMARY =====")
    for key, value in summary.items():
        if key != "metrics":
            print(f"{key}: {value}")
    print("===================")


if __name__ == "__main__":
    main()
