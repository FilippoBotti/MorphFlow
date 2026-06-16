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
    parser.add_argument(
        "--slat_checkpoint_path",
        type=str,
        default=None,
        help="Optional second-flow SLat checkpoint. When set, --checkpoint_path must be an SS checkpoint.",
    )
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--slat_steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--slat_cfg_scale", type=float, default=None)
    parser.add_argument("--trellis_model", type=str, choices=["auto", "text_base", "image_large"], default="auto")
    parser.add_argument("--mixed_precision", type=str, choices=["auto", "no", "fp16", "bf16"], default="auto")
    parser.add_argument("--allow_tf32", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save_latents", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def resolve_mixed_precision(mode: str, device: torch.device) -> str:
    mode = mode.lower()
    if mode == "auto":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                mode = "bf16"
            else:
                mode = "fp16"
        else:
            mode = "no"

    if mode == "bf16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            print("WARNING: BF16 is not supported on this GPU; falling back to fp16.")
            mode = "fp16"
        else:
            prop = torch.cuda.get_device_properties(device)
            if prop.major < 8:
                print(
                    f"WARNING: BF16 with xformers is only supported on A100+ GPUs; "
                    f"device compute capability is {prop.major}.{prop.minor}. Falling back to fp16."
                )
                mode = "fp16"

    if mode == "fp16" and device.type != "cuda":
        mode = "no"

    return mode


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
        "t_schedule": args.get("t_schedule", args.get("slat_t_schedule", "logit_normal")),
        "t_logit_mean": float(args.get("t_logit_mean", args.get("slat_t_logit_mean", 0.0))),
        "t_logit_std": float(args.get("t_logit_std", args.get("slat_t_logit_std", 1.0))),
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
        attention_scope=args.get("lora_attention_scope", "all"),
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


@torch.no_grad()
def sample_slat_on_coords(model, batch, coords, steps, device, cfg_scale, mixed_precision):
    coords = ensure_batch_coords(coords).to(device=device, dtype=torch.int32)
    if coords.numel() == 0:
        return None
    coords = coords.clone()
    coords[:, 0] = 0

    in_channels = int(getattr(model.slat_flow, "in_channels", 8))
    noise_feats = torch.randn(coords.shape[0], in_channels, device=device, dtype=torch.float32)
    x_t = model.make_slat(noise_feats, coords)

    src1_feats = batch["src1_feats"].to(device=device, dtype=torch.float32)
    src2_feats = batch["src2_feats"].to(device=device, dtype=torch.float32)
    src1_coords = batch["src1_coords"].to(device=device, dtype=torch.int32)
    src2_coords = batch["src2_coords"].to(device=device, dtype=torch.int32)
    alpha = batch["alpha"].reshape(1).to(device=device, dtype=torch.float32)
    t_seq = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)

    for i in range(steps):
        t = torch.full((1,), float(t_seq[i].item()), device=device)
        dt = t_seq[i] - t_seq[i + 1]
        with autocast_context(device, mixed_precision):
            if cfg_scale == 1.0:
                pred = model.forward_flow(x_t, t, src1_feats, src2_feats, src1_coords, src2_coords, alpha)
            else:
                pred = model.forward_flow_cfg(
                    x_t,
                    t,
                    src1_feats,
                    src2_feats,
                    src1_coords,
                    src2_coords,
                    alpha,
                    guidance_scale=cfg_scale,
                )
        x_t = x_t - dt * pred.float()

    return model.denormalize_slat(x_t)


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

    # TRELLIS mesh extraction allocates several float32 work buffers internally
    # and expects attrs to match them. Keep this export path in fp32 even when
    # the flow sampling itself uses bf16/fp16 autocast.
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
def ss_logits_raw(ss_decoder, latent, device, mixed_precision):
    latent = latent.to(device=device, dtype=torch.float32)
    if latent.ndim == 6:
        latent = latent.squeeze(1)
    with autocast_context(device, mixed_precision):
        logits = ss_decoder(latent)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    return logits.float()


@torch.no_grad()
def ss_logits(ss_decoder, latent, device, mixed_precision):
    logits = ss_logits_raw(ss_decoder, latent, device, mixed_precision)
    if logits.ndim == 5:
        logits = logits[0]
    if logits.ndim == 4:
        logits = logits[0]
    return logits


@torch.no_grad()
def ss_coords_from_latent(ss_decoder, latent, device, mixed_precision, threshold=0.0):
    logits = ss_logits_raw(ss_decoder, latent, device, mixed_precision)
    coords = torch.argwhere(logits > threshold)
    if coords.numel() == 0:
        return torch.empty((0, 4), device=device, dtype=torch.int32)
    if coords.shape[1] == 5:
        coords = coords[:, [0, 2, 3, 4]]
    elif coords.shape[1] == 4:
        coords = coords[:, [0, 1, 2, 3]]
    elif coords.shape[1] == 3:
        batch = torch.zeros((coords.shape[0], 1), device=coords.device, dtype=coords.dtype)
        coords = torch.cat([batch, coords], dim=1)
    else:
        raise ValueError(f"Unexpected SS decoder output rank for coords: logits shape={tuple(logits.shape)}")
    return coords.int()


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


def patch_trellis_mesh_dtype():
    """Keep TRELLIS mesh extraction buffers in the same dtype as decoder attrs."""
    import trellis.representations.mesh.cube2mesh as cube2mesh
    import trellis.representations.mesh.utils_cube as utils_cube

    def cubes_to_verts(num_verts, cubes, value, reduce="mean"):
        channels = value.shape[2]
        reduced = torch.zeros(
            num_verts,
            channels,
            device=cubes.device,
            dtype=value.dtype,
        )
        return torch.scatter_reduce(
            reduced,
            0,
            cubes.unsqueeze(-1).expand(-1, -1, channels).flatten(0, 1),
            value.flatten(0, 1),
            reduce=reduce,
            include_self=False,
        )

    def get_dense_attrs(coords, feats, res, sdf_init=True):
        channels = feats.shape[-1]
        dense_attrs = torch.zeros(
            [res] * 3 + [channels],
            device=feats.device,
            dtype=feats.dtype,
        )
        if sdf_init:
            dense_attrs[..., 0] = 1
        dense_attrs[coords[:, 0], coords[:, 1], coords[:, 2], :] = feats
        return dense_attrs.reshape(-1, channels)

    utils_cube.cubes_to_verts = cubes_to_verts
    utils_cube.get_dense_attrs = get_dense_attrs
    cube2mesh.cubes_to_verts = cubes_to_verts
    cube2mesh.get_dense_attrs = get_dense_attrs


def load_decoders(flow_target, device):
    from trellis.models import from_pretrained as trellis_from_pretrained
    from trellis.modules.sparse.basic import SparseTensor

    patch_trellis_mesh_dtype()

    print("Loading TRELLIS SLAT mesh decoder...")
    mesh_decoder = trellis_from_pretrained(
        "microsoft/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16"
    ).to(device).eval()
    if hasattr(mesh_decoder, "convert_to_fp32"):
        mesh_decoder.convert_to_fp32()
    mesh_decoder.float()
    if hasattr(mesh_decoder, "dtype"):
        mesh_decoder.dtype = torch.float32

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
    slat_checkpoint_path = (
        Path(args.slat_checkpoint_path).expanduser().resolve()
        if args.slat_checkpoint_path
        else None
    )

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if slat_checkpoint_path is not None and not slat_checkpoint_path.is_file():
        raise FileNotFoundError(f"SLat checkpoint not found: {slat_checkpoint_path}")

    if args.allow_tf32 == 1 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.mixed_precision = resolve_mixed_precision(args.mixed_precision, device)
    print(f"Resolved mixed_precision: {args.mixed_precision}")

    output_dir = Path(args.output_dir).expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(str(checkpoint_path))
    flow_target = detect_flow_target(ckpt)
    model_type = detect_model_type(ckpt, args.trellis_model)
    slat_ckpt = load_checkpoint(str(slat_checkpoint_path)) if slat_checkpoint_path is not None else None
    slat_model_type = detect_model_type(slat_ckpt, args.trellis_model) if slat_ckpt is not None else None
    pipeline_mode = slat_ckpt is not None

    if pipeline_mode:
        if flow_target != "ss":
            raise ValueError(
                f"--slat_checkpoint_path requires --checkpoint_path to be an SS checkpoint, got {flow_target!r}."
            )
        slat_flow_target = detect_flow_target(slat_ckpt)
        if slat_flow_target != "slat":
            raise ValueError(f"--slat_checkpoint_path must point to a SLat checkpoint, got {slat_flow_target!r}.")
    else:
        slat_flow_target = None

    print("===== EVAL =====")
    print(f"dataset: {root}")
    print(f"metadata: {metadata_path}")
    print(f"checkpoint: {checkpoint_path}")
    if pipeline_mode:
        print(f"slat_checkpoint: {slat_checkpoint_path}")
    print(f"flow_target: {flow_target}")
    if pipeline_mode:
        print("pipeline: ss checkpoint -> SS decoder coords -> slat checkpoint -> mesh decoder")
    print(f"ss_flow_arch: {checkpoint_args(ckpt).get('ss_flow_arch', 'standard')}")
    print(f"model_type: {model_type}")
    if pipeline_mode:
        print(f"slat_model_type: {slat_model_type}")
    print(f"cfg_scale: {args.cfg_scale}")
    if pipeline_mode:
        print(f"slat_cfg_scale: {args.slat_cfg_scale if args.slat_cfg_scale is not None else args.cfg_scale}")
    print(f"steps: {args.steps}")
    if pipeline_mode:
        print(f"slat_steps: {args.slat_steps if args.slat_steps is not None else args.steps}")
    print(f"output: {output_dir}")
    print("alpha: metadata alpha is passed unchanged as MorphFlow(src1, src2, alpha)")
    print("================")

    model = build_model(ckpt, model_type, flow_target).to(device).eval()
    slat_model = (
        build_model(slat_ckpt, slat_model_type, "slat").to(device).eval()
        if pipeline_mode
        else None
    )
    ss_decoder, mesh_decoder, sparse_tensor_cls = load_decoders("ss" if pipeline_mode else flow_target, device)

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
            pred_coords = ss_coords_from_latent(ss_decoder, pred_ss, device, args.mixed_precision)
            row["pred_ss_points"] = int(pred_coords.shape[0])

            pred_final_slat = None
            if slat_model is not None:
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
                    row["pred_final_saved"] = save_slat_glb(
                        mesh_decoder,
                        sparse_tensor_cls,
                        pred_final_slat.feats,
                        pred_final_slat.coords,
                        sample_dir / "pred_final.glb",
                        device,
                        args.mixed_precision,
                        fallback_color=(220, 120, 220),
                    )
                    row["pred_final_points"] = int(pred_final_slat.feats.shape[0])

            if args.save_latents == 1:
                payload = {
                    "flow_target": flow_target,
                    "pred_ss_latent": pred_ss.detach().cpu(),
                    "target_ss_latent": target_ss.detach().cpu(),
                    "pred_ss_coords": pred_coords.detach().cpu(),
                    "alpha": alpha,
                }
                if pred_final_slat is not None:
                    payload.update(
                        {
                            "pred_final_slat_feats": pred_final_slat.feats.detach().cpu(),
                            "pred_final_slat_coords": pred_final_slat.coords.detach().cpu(),
                        }
                    )
                torch.save(payload, sample_dir / "latents.pt")

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
        "slat_checkpoint": str(slat_checkpoint_path) if slat_checkpoint_path is not None else None,
        "pipeline_mode": pipeline_mode,
        "flow_target": flow_target,
        "model_type": model_type,
        "slat_model_type": slat_model_type,
        "cfg_scale": args.cfg_scale,
        "slat_cfg_scale": args.slat_cfg_scale if args.slat_cfg_scale is not None else args.cfg_scale,
        "steps": args.steps,
        "slat_steps": args.slat_steps if args.slat_steps is not None else args.steps,
        "seed": args.seed,
        "selected": selected,
        "metrics": metrics,
    }

    for key in (
        "latent_mse",
        "latent_l1",
        "voxel_iou",
        "pred_ss_points",
        "pred_final_points",
        "slat_feat_mse",
        "slat_feat_l1",
        "slat_norm_feat_mse",
    ):
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
