
import torch
import inspect
import argparse
import os


os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")
from models.morph_flow import MorphFlow
from models.morph_slat_flow import MorphSLatFlow
import time
from tqdm import tqdm
from TRELLIS.trellis.models import from_pretrained as trellis_from_pretrained
from modules import sparse as sp
import trimesh
import numpy as np


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



def build_model(flow_target) -> torch.nn.Module:
    if flow_target == "slat":
        model_cls = MorphSLatFlow
        model_kwargs = {
            "sigma_min": 1e-5,
            "model_type": "image_large",
            "separate_cond": True,
            "use_checkpoint": False,
            "separate_cond_gate": "alpha_residual",
            "cond_resample_tokens": 0,
            "cond_resample_depth": 1,
            "cond_resample_heads": 8,
            "cond_encoder_type": "block",
            "normalize_cond_latents": False,
            "cond_token_norm": "layernorm",
            "cond_proj_norm": "none",
            "cond_style_tokens": 0,
            "t_schedule": "logit_normal",
            "t_logit_mean": 0.0,
            "t_logit_std": 1.0,
        }
    elif flow_target == "ss":
        model_cls = MorphFlow
        model_kwargs = {
            "sigma_min": 1e-5,
            "model_type": "image_large",
            "separate_cond": True,
            "use_checkpoint": False,
            "separate_cond_gate": "alpha_residual",
            "cond_resample_tokens": 0,
            "cond_resample_depth": 1,
            "cond_resample_heads": 8,
            "cond_encoder_type": "block",
            "normalize_cond_latents": False,
            "cond_style_tokens": 0,
            "t_schedule": "logit_normal",
            "t_logit_mean": 0.0,
            "t_logit_std": 1.0,
        }

    model = model_cls(**model_kwargs)
    return model


@torch.no_grad()
def sample_ss(model, src1_feats, src2_feats, src1_coords, src2_coords, alpha, steps, device):
    target_shape = [1,8,16,16,16]
    x_t = torch.randn_like(torch.empty(target_shape, device=device, dtype=torch.float32), device=device, dtype=torch.float32)
    t_seq = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)

    src1_feats = src1_feats.to(device=device, dtype=torch.float32)
    src2_feats = src2_feats.to(device=device, dtype=torch.float32)
    src1_coords = src1_coords.to(device=device, dtype=torch.int32)
    src2_coords = src2_coords.to(device=device, dtype=torch.int32)
    alpha = alpha.reshape(target_shape[0]).to(device=device, dtype=torch.float32)

    for i in tqdm(range(steps)):
        t = torch.full((target_shape[0],), float(t_seq[i].item()), device=device)
        dt = t_seq[i] - t_seq[i + 1]
        pred = model.forward_flow(x_t, t, src1_feats, src2_feats, src1_coords, src2_coords, alpha)

        x_t = x_t - dt * pred.float()

    return x_t

@torch.no_grad()
def sample_slat(model, coords, src1_feats, src2_feats, src1_coords, src2_coords, alpha, steps, device):
    src1_feats = src1_feats.to(device=device, dtype=torch.float32)
    src2_feats = src2_feats.to(device=device, dtype=torch.float32)
    src1_coords = src1_coords.to(device=device, dtype=torch.int32)
    src2_coords = src2_coords.to(device=device, dtype=torch.int32)
    alpha = alpha.reshape(1).to(device=device, dtype=torch.float32)
    t_seq = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    x_t = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], model.slat_flow.in_channels).to('cuda'),
            coords=coords.to('cuda'),
        )
    for i in tqdm(range(steps)):
        t = torch.full((x_t.shape[0],), float(t_seq[i].item()), device=device)
        dt = t_seq[i] - t_seq[i + 1]
        pred = model.forward_flow(
                    x_t,
                    t,
                    src1_feats,
                    src2_feats,
                    src1_coords,
                    src2_coords,
                    alpha,
                )
        x_t = x_t - dt * pred.float()

    return model.denormalize_slat(x_t)

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


def eval():
    slat_feats_1 = "/home/filippo/datasets/3d/morphing_dataset_v2/morphing_dataset_v2/assets/0050_scholarly-gargoyle-scout-guarding-stance/slat_feats.pt"
    slat_feats_2 = "/home/filippo/datasets/3d/morphing_dataset_v2/morphing_dataset_v2/assets/0970_jungle-anthropomorphic-lizard-battle-ready/slat_feats.pt"
    slat_target_feats = "/home/filippo/datasets/3d/morphing_dataset_v2/morphing_dataset_v2/targets/0050_scholarly-gargoyle-scout-guarding-stance+0970_jungle-anthropomorphic-lizard-battle-ready/alpha_0p506207/slat_feats.pt"
    
    slat_cords_1 = "/home/filippo/datasets/3d/morphing_dataset_v2/morphing_dataset_v2/assets/0050_scholarly-gargoyle-scout-guarding-stance/slat_coords.pt"
    slat_cords_2 = "/home/filippo/datasets/3d/morphing_dataset_v2/morphing_dataset_v2/assets/0970_jungle-anthropomorphic-lizard-battle-ready/slat_coords.pt"
    slat_cords_target = "/home/filippo/datasets/3d/morphing_dataset_v2/morphing_dataset_v2/targets/0050_scholarly-gargoyle-scout-guarding-stance+0970_jungle-anthropomorphic-lizard-battle-ready/alpha_0p506207/slat_coords.pt"

    src1_feats = torch.load(slat_feats_1, map_location="cuda")
    src2_feats = torch.load(slat_feats_2, map_location="cuda")
    target_feats = torch.load(slat_target_feats, map_location="cuda")
    src1_coords = torch.load(slat_cords_1, map_location="cuda")
    src2_coords = torch.load(slat_cords_2, map_location="cuda")
    target_coords = torch.load(slat_cords_target, map_location="cuda")

    ckpt_ss_flow = "/home/filippo/checkpoints/3d/morphflow_ss_best.pt"
    ckpt_slat_flow = "/home/filippo/checkpoints/3d/morphflow_slat_best.pt"

    ckpt_slat_flow = torch.load(ckpt_slat_flow, map_location="cpu")
    ckpt_ss_flow = torch.load(ckpt_ss_flow, map_location="cpu")
    alpha = 0.5
    alpha = torch.tensor([alpha], dtype=torch.float32)

    model_ss = build_model("ss")
    model_slat = build_model("slat")

    ss_decoder = trellis_from_pretrained(
            "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
        ).to('cuda').eval()
    slat_decoder = trellis_from_pretrained(
            "microsoft/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16"
        ).to('cuda').eval()

    model_ss.load_state_dict(ckpt_ss_flow["model"])
    model_slat.load_state_dict(ckpt_slat_flow["model"])

    model_ss.eval()
    model_slat.eval()
    
    model_slat.to("cuda")
    model_ss.to("cuda")
    with torch.no_grad():
        start_total_time = time.time()
        out_ss = sample_ss(
            model_ss,
            src1_feats,
            src2_feats,
            src1_coords,
            src2_coords,
            alpha,
            25,
            'cuda'
        )
        end_time = time.time()
        print(f"SS Morphing Time: {end_time - start_total_time:.4f} seconds")
        coords = torch.argwhere(ss_decoder(out_ss)>0)[:, [0, 2, 3, 4]].int()
        start_time = time.time()
        out_slat = sample_slat(
            model_slat,
            coords,
            src1_feats,
            src2_feats,
            src1_coords,
            src2_coords,
            alpha,
            25,
            'cuda'
        )   
        end_time = time.time()
        print(f"SLAT Morphing Time: {end_time - start_time:.4f} seconds")
        from TRELLIS.trellis.modules import sparse as sp
        save_slat_glb(
            slat_decoder,
            sp.SparseTensor,
            out_slat.feats,
            out_slat.coords,
            "./outputs/evaluation_time/morphing_result.glb",
            'cuda',
            mixed_precision=False,
        )
        end_time = time.time()
        print(f"Total Morphing Time: {end_time - start_total_time:.4f} seconds")

    save_slat_glb(
        slat_decoder,
        sp.SparseTensor,
        target_feats,
        target_coords,
        "./outputs/evaluation_time/target.glb",
        'cuda',
        mixed_precision=False,
    )
if __name__ == "__main__":
    eval()