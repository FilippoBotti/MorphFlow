import argparse
import inspect
import json
import os
import re
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from trimesh.voxel.encoding import DenseEncoding

if os.environ.get("TRELLIS_REPO"):
    sys.path.append(os.environ["TRELLIS_REPO"])

from data.morph_dataset import MorphingDistillDataset, morphing_collate_fn
from models.lora import add_lora_to_attention
from models.morph_flow import MorphFlow
from models.morph_residual_flow import MorphResidualSSFlow
from models.morph_slat_flow import MorphSLatFlow


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = Path.home() / "datasets/3d/morphing_dataset_v2/morphing_dataset_v2"
DEFAULT_CHECKPOINT = PROJECT_DIR / "outputs/checkpoints/morphflow_best.pt"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/eval_test_latents"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MorphFlow on fixed test pairs.")
    parser.add_argument("--root_dir", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--metadata", type=str, default="metadata_test.json")
    parser.add_argument("--checkpoint_path", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--trellis_model", type=str, choices=["auto", "text_base", "image_large"], default="auto")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--allow_tf32", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save_latents", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def safe_slug(value, max_len=80):
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value))
    return value[:max_len].strip("_") or "unknown"


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "module"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    cleaned = {}
    for key, value in obj.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def checkpoint_args(ckpt):
    if isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict):
        return ckpt["args"]
    return {}


def detect_flow_target(ckpt):
    args = checkpoint_args(ckpt)
    if isinstance(ckpt, dict) and ckpt.get("flow_target") in ("ss", "slat"):
        return ckpt["flow_target"]
    return args.get("flow_target", "ss")


def detect_model_type(ckpt, requested):
    if requested != "auto":
        return requested
    args = checkpoint_args(ckpt)
    if args.get("trellis_model"):
        return args["trellis_model"]
    if isinstance(ckpt, dict):
        return ckpt.get("model_type", "image_large")
    return "image_large"


def build_model(ckpt, model_type, flow_target):
    args = checkpoint_args(ckpt)
    if flow_target == "slat":
        model_cls = MorphSLatFlow
    elif args.get("ss_flow_arch", "standard") == "residual_interp":
        model_cls = MorphResidualSSFlow
    else:
        model_cls = MorphFlow

    requested_kwargs = {
        "model_type": model_type,
        "separate_cond": bool(int(args.get("separate_cond", 0))),
        "use_checkpoint": False,
        "separate_cond_gate": args.get("separate_cond_gate", "alpha_residual"),
        "cond_resample_tokens": int(args.get("cond_resample_tokens", 0)),
        "cond_resample_depth": int(args.get("cond_resample_depth", 1)),
        "cond_resample_heads": int(args.get("cond_resample_heads", 8)),
        "residual_interp_gate": args.get("residual_interp_gate", "alpha"),
        "residual_interp_gate_min": float(args.get("residual_interp_gate_min", 1e-3)),
        "residual_endpoint_prob": float(args.get("residual_endpoint_prob", 0.0)),
        "residual_endpoint_weight": float(args.get("residual_endpoint_weight", 1.0)),
        "residual_endpoint_max_items": int(args.get("residual_endpoint_max_items", 1)),
    }

    supported = set(inspect.signature(model_cls.__init__).parameters)
    kwargs = {key: value for key, value in requested_kwargs.items() if key in supported}

    print(f"{model_cls.__name__} kwargs:")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")

    model = model_cls(**kwargs)
    maybe_insert_lora(model, args)
    model.load_state_dict(unwrap_state_dict(ckpt), strict=True)
    return model


def get_flow_module(model):
    return getattr(model, "sparse_structure_flow", None) or getattr(model, "slat_flow", None)


def maybe_insert_lora(model, args):
    if int(args.get("use_lora", 0)) != 1:
        return

    flow = get_flow_module(model)
    if flow is None:
        raise RuntimeError("Checkpoint uses LoRA, but no TRELLIS flow module was found.")

    targets = tuple(x.strip() for x in args.get("lora_target_modules", "to_q,to_kv").split(",") if x.strip())
    modules = add_lora_to_attention(
        flow,
        rank=int(args.get("lora_rank", 8)),
        alpha=int(args.get("lora_alpha", 16)),
        dropout=float(args.get("lora_dropout", 0.0)),
        target_modules=targets,
    )
    print(f"Inserted LoRA modules: {len(modules)}")


def autocast_context(device, mixed_precision):
    if device.type != "cuda" or mixed_precision == "no":
        return nullcontext()
    if mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def select_fixed_pair_indices(dataset, num_samples, seed):
    rng = np.random.default_rng(seed)
    pair_to_indices = {}
    for idx, entry in enumerate(dataset.metadata):
        pair = (str(entry["src_1"]), str(entry["src_2"]))
        pair_to_indices.setdefault(pair, []).append(idx)

    pairs = list(pair_to_indices)
    take = min(num_samples, len(pairs))
    pair_positions = rng.choice(len(pairs), size=take, replace=False)

    indices = []
    for pos in pair_positions:
        candidates = pair_to_indices[pairs[int(pos)]]
        indices.append(int(rng.choice(candidates)))
    return indices


@torch.no_grad()
def sample_ss(model, batch, target_ss, steps, device, cfg_scale, mixed_precision):
    x_t = torch.randn_like(target_ss, device=device, dtype=torch.float32)
    t_seq = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)

    src1_feats = batch["src1_feats"].to(device=device, dtype=torch.float32)
    src2_feats = batch["src2_feats"].to(device=device, dtype=torch.float32)
    src1_coords = batch["src1_coords"].to(device=device, dtype=torch.int32)
    src2_coords = batch["src2_coords"].to(device=device, dtype=torch.int32)
    alpha = batch["alpha"].reshape(target_ss.shape[0]).to(device=device, dtype=torch.float32)

    for i in range(steps):
        t = torch.full((target_ss.shape[0],), float(t_seq[i].item()), device=device)
        dt = t_seq[i] - t_seq[i + 1]
        with autocast_context(device, mixed_precision):
            if cfg_scale == 1.0:
                pred = model.forward_flow(x_t, t, src1_feats, src2_feats, src1_coords, src2_coords, alpha)
            else:
                pred = model.forward_flow_cfg(
                    x_t, t, src1_feats, src2_feats, src1_coords, src2_coords, alpha, guidance_scale=cfg_scale
                )
        x_t = x_t - dt * pred.float()

    if hasattr(model, "residual_to_ss"):
        src1_ss = batch["src1_ss_latent"].to(device=device, dtype=torch.float32)
        src2_ss = batch["src2_ss_latent"].to(device=device, dtype=torch.float32)
        x_t = model.residual_to_ss(x_t, src1_ss, src2_ss, alpha)
    return x_t


@torch.no_grad()
def sample_slat(model, batch, steps, device, cfg_scale, mixed_precision):
    target_feats = batch["target_feats"].to(device=device, dtype=torch.float32)
    target_coords = batch["target_coords"].to(device=device, dtype=torch.int32)
    x0 = model.normalize_slat(model.make_slat(target_feats, target_coords))
    x_t = x0.replace(torch.randn_like(x0.feats))

    src1_feats = batch["src1_feats"].to(device=device, dtype=torch.float32)
    src2_feats = batch["src2_feats"].to(device=device, dtype=torch.float32)
    src1_coords = batch["src1_coords"].to(device=device, dtype=torch.int32)
    src2_coords = batch["src2_coords"].to(device=device, dtype=torch.int32)
    alpha = batch["alpha"].reshape(x0.shape[0]).to(device=device, dtype=torch.float32)
    t_seq = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)

    for i in range(steps):
        t = torch.full((x0.shape[0],), float(t_seq[i].item()), device=device)
        dt = t_seq[i] - t_seq[i + 1]
        with autocast_context(device, mixed_precision):
            if cfg_scale == 1.0:
                pred = model.forward_flow(x_t, t, src1_feats, src2_feats, src1_coords, src2_coords, alpha)
            else:
                pred = model.forward_flow_cfg(
                    x_t, t, src1_feats, src2_feats, src1_coords, src2_coords, alpha, guidance_scale=cfg_scale
                )
        x_t = x_t - dt * pred.float()

    return model.denormalize_slat(x_t), x_t, x0


def ensure_batch_coords(coords):
    if coords.shape[-1] == 4:
        return coords
    batch = torch.zeros((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)
    return torch.cat([batch, coords], dim=-1)


def rgba_from_rgb(rgb, alpha=255):
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.max(initial=0) <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[0] == 4:
        return arr
    return np.concatenate([arr[:3], np.array([alpha], dtype=np.uint8)])


def decoded_vertex_colors(decoded, fallback_rgb=(210, 210, 210)):
    attrs = getattr(decoded, "vertex_attrs", None)
    if attrs is None:
        color = rgba_from_rgb(fallback_rgb)
        return np.repeat(color[None, :], int(decoded.vertices.shape[0]), axis=0)

    attrs = attrs.detach().float().cpu()
    if attrs.ndim != 2 or attrs.shape[0] != int(decoded.vertices.shape[0]) or attrs.shape[1] < 3:
        color = rgba_from_rgb(fallback_rgb)
        return np.repeat(color[None, :], int(decoded.vertices.shape[0]), axis=0)

    colors = attrs[:, :3].numpy()
    if colors.max(initial=0.0) <= 1.0 and colors.min(initial=0.0) >= 0.0:
        colors = colors * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
    return np.concatenate([colors, alpha], axis=1)


@torch.no_grad()
def save_slat_glb(
    mesh_decoder,
    sparse_tensor_cls,
    feats,
    coords,
    path,
    device,
    mixed_precision,
    fallback_color=(210, 210, 210),
):
    feats = feats.to(device=device, dtype=torch.float32)
    coords = ensure_batch_coords(coords).to(device=device, dtype=torch.int32)
    st = sparse_tensor_cls(feats=feats, coords=coords)

    with autocast_context(device, mixed_precision):
        decoded = mesh_decoder(st)[0]

    if not getattr(decoded, "success", False):
        return False

    mesh = trimesh.Trimesh(
        vertices=decoded.vertices.detach().float().cpu().numpy(),
        faces=decoded.faces.detach().cpu().numpy(),
        visual=trimesh.visual.ColorVisuals(
            vertex_colors=decoded_vertex_colors(decoded, fallback_color)
        ),
        process=False,
    )
    mesh.export(path)
    return True


@torch.no_grad()
def ss_logits(ss_decoder, latent, device, mixed_precision):
    latent = latent.to(device=device, dtype=torch.float32)
    if latent.ndim == 6:
        latent = latent.squeeze(1)
    with autocast_context(device, mixed_precision):
        logits = ss_decoder(latent)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    logits = logits.float()
    if logits.ndim == 5:
        logits = logits[0]
    if logits.ndim == 4:
        logits = logits[0]
    return logits


def voxel_iou(pred_logits, target_logits):
    pred = pred_logits > 0
    target = target_logits > 0
    inter = torch.logical_and(pred, target).sum().item()
    union = torch.logical_or(pred, target).sum().item()
    return float(inter / union) if union else 1.0


@torch.no_grad()
def save_voxel_glb(ss_decoder, latent, path, device, mixed_precision, color=(150, 170, 210)):
    logits = ss_logits(ss_decoder, latent, device, mixed_precision)
    voxels = (logits > 0).detach().cpu().numpy().astype(bool)
    if int(voxels.sum()) == 0:
        return {"saved": False, "occupancy_ratio": 0.0, "occupied": 0}

    mesh = trimesh.voxel.VoxelGrid(DenseEncoding(voxels)).marching_cubes
    mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.repeat(rgba_from_rgb(color)[None, :], len(mesh.vertices), axis=0)
    )
    mesh.export(path)
    return {
        "saved": True,
        "occupancy_ratio": float(voxels.sum() / max(voxels.size, 1)),
        "occupied": int(voxels.sum()),
    }


def load_decoders(flow_target, device):
    from trellis.models import from_pretrained as trellis_from_pretrained
    from trellis.modules.sparse.basic import SparseTensor

    print("Loading TRELLIS SLAT mesh decoder...")
    mesh_decoder = trellis_from_pretrained(
        "microsoft/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16"
    ).to(device).eval()

    ss_decoder = None
    if flow_target == "ss":
        print("Loading TRELLIS sparse-structure decoder...")
        ss_decoder = trellis_from_pretrained(
            "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
        ).to(device).eval()

    return ss_decoder, mesh_decoder, SparseTensor


def main():
    args = parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.allow_tf32 == 1 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(str(checkpoint_path))
    flow_target = detect_flow_target(ckpt)
    model_type = detect_model_type(ckpt, args.trellis_model)

    print("===== EVAL =====")
    print(f"dataset: {root}")
    print(f"metadata: {metadata_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"flow_target: {flow_target}")
    print(f"ss_flow_arch: {checkpoint_args(ckpt).get('ss_flow_arch', 'standard')}")
    print(f"model_type: {model_type}")
    print(f"cfg_scale: {args.cfg_scale}")
    print(f"steps: {args.steps}")
    print(f"output: {output_dir}")
    print("alpha: metadata alpha is passed unchanged as MorphFlow(src1, src2, alpha)")
    print("================")

    model = build_model(ckpt, model_type, flow_target).to(device).eval()
    ss_decoder, mesh_decoder, sparse_tensor_cls = load_decoders(flow_target, device)

    dataset = MorphingDistillDataset(
        root=str(root),
        metadata_file=str(metadata_path),
        split="test",
        verbose=False,
    )
    indices = select_fixed_pair_indices(dataset, args.num_samples, args.seed)

    selected = []
    metrics = []

    for sample_id, dataset_idx in enumerate(indices):
        batch = morphing_collate_fn([dataset[dataset_idx]])
        src1_name = batch["src1_name"][0]
        src2_name = batch["src2_name"][0]
        target_name = batch["target_name"][0]
        alpha = float(batch["alpha"].reshape(-1)[0].item())

        sample_name = (
            f"sample_{sample_id:03d}_"
            f"{safe_slug(src1_name, 36)}_"
            f"{safe_slug(src2_name, 36)}_"
            f"a{alpha:.4f}"
        )
        sample_dir = output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        selected.append(
            {
                "sample_id": sample_id,
                "dataset_idx": int(dataset_idx),
                "src1": src1_name,
                "src2": src2_name,
                "target": target_name,
                "alpha": alpha,
            }
        )

        row = {
            "sample_id": sample_id,
            "dataset_idx": int(dataset_idx),
            "src1": src1_name,
            "src2": src2_name,
            "target": target_name,
            "alpha": alpha,
            "sample_dir": str(sample_dir),
        }

        save_slat_glb(
            mesh_decoder,
            sparse_tensor_cls,
            batch["src1_feats"],
            batch["src1_coords"],
            sample_dir / "src1.glb",
            device,
            args.mixed_precision,
            fallback_color=(80, 150, 255),
        )
        save_slat_glb(
            mesh_decoder,
            sparse_tensor_cls,
            batch["src2_feats"],
            batch["src2_coords"],
            sample_dir / "src2.glb",
            device,
            args.mixed_precision,
            fallback_color=(255, 150, 80),
        )
        save_slat_glb(
            mesh_decoder,
            sparse_tensor_cls,
            batch["target_feats"],
            batch["target_coords"],
            sample_dir / "target.glb",
            device,
            args.mixed_precision,
            fallback_color=(120, 220, 140),
        )

        if flow_target == "ss":
            target_ss = batch["target_ss_latent"].to(device=device, dtype=torch.float32)
            if target_ss.ndim == 6:
                target_ss = target_ss.squeeze(1)
            pred_ss = sample_ss(model, batch, target_ss, args.steps, device, args.cfg_scale, args.mixed_precision)

            row["latent_mse"] = float(F.mse_loss(pred_ss.float(), target_ss.float()).item())
            row["latent_l1"] = float(F.l1_loss(pred_ss.float(), target_ss.float()).item())

            target_logits = ss_logits(ss_decoder, target_ss, device, args.mixed_precision)
            pred_logits = ss_logits(ss_decoder, pred_ss, device, args.mixed_precision)
            row["voxel_iou"] = voxel_iou(pred_logits, target_logits)
            row["target_voxel"] = save_voxel_glb(
                ss_decoder,
                target_ss,
                sample_dir / "target_voxels.glb",
                device,
                args.mixed_precision,
                color=(120, 220, 140),
            )
            row["pred_voxel"] = save_voxel_glb(
                ss_decoder,
                pred_ss,
                sample_dir / "pred_voxels.glb",
                device,
                args.mixed_precision,
                color=(220, 120, 220),
            )

            if args.save_latents == 1:
                torch.save(
                    {
                        "flow_target": flow_target,
                        "pred_ss_latent": pred_ss.detach().cpu(),
                        "target_ss_latent": target_ss.detach().cpu(),
                        "alpha": alpha,
                    },
                    sample_dir / "latents.pt",
                )

        else:
            pred_slat, pred_slat_norm, target_slat_norm = sample_slat(
                model, batch, args.steps, device, args.cfg_scale, args.mixed_precision
            )
            target_feats = batch["target_feats"].to(device=device, dtype=torch.float32)

            row["slat_feat_mse"] = float(F.mse_loss(pred_slat.feats.float(), target_feats.float()).item())
            row["slat_feat_l1"] = float(F.l1_loss(pred_slat.feats.float(), target_feats.float()).item())
            row["slat_norm_feat_mse"] = float(F.mse_loss(pred_slat_norm.feats.float(), target_slat_norm.feats.float()).item())
            row["slat_points"] = int(target_feats.shape[0])
            row["pred_slat_saved"] = save_slat_glb(
                mesh_decoder,
                sparse_tensor_cls,
                pred_slat.feats,
                pred_slat.coords,
                sample_dir / "pred_slat.glb",
                device,
                args.mixed_precision,
                fallback_color=(220, 120, 220),
            )

            if args.save_latents == 1:
                torch.save(
                    {
                        "flow_target": flow_target,
                        "pred_slat_feats": pred_slat.feats.detach().cpu(),
                        "pred_slat_coords": pred_slat.coords.detach().cpu(),
                        "target_slat_feats": batch["target_feats"].detach().cpu(),
                        "target_slat_coords": batch["target_coords"].detach().cpu(),
                        "alpha": alpha,
                    },
                    sample_dir / "latents.pt",
                )

        with (sample_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        metrics.append(row)

        primary = row.get("latent_mse", row.get("slat_feat_mse", float("nan")))
        print(f"[{sample_id + 1}/{len(indices)}] alpha={alpha:.4f} mse={primary:.6f} -> {sample_dir}")

    summary = {
        "dataset": str(root),
        "metadata": str(metadata_path),
        "checkpoint": str(checkpoint_path),
        "flow_target": flow_target,
        "model_type": model_type,
        "cfg_scale": args.cfg_scale,
        "steps": args.steps,
        "seed": args.seed,
        "selected": selected,
        "metrics": metrics,
    }

    for key in ("latent_mse", "latent_l1", "voxel_iou", "slat_feat_mse", "slat_feat_l1", "slat_norm_feat_mse"):
        values = [float(row[key]) for row in metrics if key in row]
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
            summary[f"std_{key}"] = float(np.std(values))

    with (output_dir / "selected_samples.json").open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("===== SUMMARY =====")
    for key, value in summary.items():
        if key not in ("selected", "metrics"):
            print(f"{key}: {value}")
    print("===================")


if __name__ == "__main__":
    main()
