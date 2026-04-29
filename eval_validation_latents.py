import argparse
import os
from glob import glob
from datetime import datetime

import torch
import numpy as np
import trimesh
from trimesh.voxel.encoding import DenseEncoding

os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

import sys
if os.environ.get("TRELLIS_REPO"):
    sys.path.append(os.environ.get("TRELLIS_REPO"))

from torch.utils.data import DataLoader
from data.morph_dataset import MorphingDistillDataset, morphing_collate_fn
from models.morph_flow import MorphFlow
from trellis.models import from_pretrained as trellis_from_pretrained
from trellis.modules.sparse.basic import SparseTensor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/home/filippo/datasets/3d/morphing_dataset_flux")
    parser.add_argument("--val_metadata", type=str, default="metadata_val_200_tail.json")
    parser.add_argument("--checkpoints_root", type=str, default="./outputs/morphflow")
    parser.add_argument("--output_dir", type=str, default="./outputs/eval_simplified")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    return parser.parse_args()

def find_latest_checkpoint(checkpoints_root):
    pattern = os.path.join(checkpoints_root, "**", "morphflow_epoch_*.pt")
    candidates = glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under: {checkpoints_root}")
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]

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

def run_reverse_flow_sample(model, x0_shape, src1_feats, src2_feats, src1_coords, src2_coords, alpha, steps, device):
    x_t = torch.randn(x0_shape, device=device, dtype=torch.float32)
    t_seq = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    for i in range(steps):
        t_curr = t_seq[i]
        t_next = t_seq[i + 1]
        dt = t_curr - t_next
        t_batch = t_curr.unsqueeze(0)
        v_pred = model.forward_flow(x_t, t_batch, src1_feats, src2_feats, src1_coords, src2_coords, alpha)
        x_t = x_t - dt * v_pred
    return x_t

def ensure_batch_coords(coords):
    if coords.shape[-1] == 3:
        b = torch.zeros((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)
        return torch.cat([b, coords], dim=-1)
    return coords

def save_slat_glb(decoder, feats, coords, out_path, device):
    coords = ensure_batch_coords(coords).to(device=device, dtype=torch.int32)
    feats = feats.to(device=device, dtype=torch.float32)
    st = SparseTensor(feats=feats, coords=coords)
    mesh_out = decoder(st)[0]
    if mesh_out.success:
        mesh = trimesh.Trimesh(vertices=mesh_out.vertices.cpu().numpy(), faces=mesh_out.faces.cpu().numpy(), process=False)
        mesh.export(out_path)
    else:
        print(f"Failed to decode slat to {out_path}")

def save_ss_glb(decoder, latent, out_path, device):
    latent = latent.to(device)
    if latent.ndim == 6: latent = latent.squeeze(1)
    logits = decoder(latent)
    if logits.ndim > 4: logits = logits[0]
    if logits.ndim == 4: logits = logits[0]
    voxels = (logits > 0).cpu().numpy().astype(bool)
    vg = trimesh.voxel.VoxelGrid(DenseEncoding(voxels))
    # We can export the voxel grid's marching cubes mesh
    mesh = vg.marching_cubes
    mesh.export(out_path)

def main():
    args = parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, timestamp)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    ckpt_path = find_latest_checkpoint(args.checkpoints_root)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MorphFlow().to(device)
    model.load_state_dict(unwrap_state_dict(ckpt), strict=True)
    model.eval()

    print("Loading TRELLIS decoders...")
    ss_decoder = trellis_from_pretrained("microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16").to(device)
    mesh_decoder = trellis_from_pretrained("microsoft/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16").to(device)
    ss_decoder.eval()
    mesh_decoder.eval()

    val_metadata_path = os.path.join(args.root_dir, args.val_metadata)
    val_dataset = MorphingDistillDataset(root=args.root_dir, metadata_file=val_metadata_path, verbose=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=morphing_collate_fn)

    processed = 0
    with torch.no_grad():
        for b, batch in enumerate(val_loader):
            if processed >= args.num_samples:
                break
            
            src1_name = batch.get("src1_name", [f"s1_{b}"])[0]
            src2_name = batch.get("src2_name", [f"s2_{b}"])[0]
            tgt_name = batch.get("target_name", [f"tg_{b}"])[0]
            sample_dir = os.path.join(args.output_dir, f"sample_{processed:03d}_{src1_name}_{src2_name}")
            os.makedirs(sample_dir, exist_ok=True)
            
            src1_feats, src1_coords = batch["src1_feats"], batch["src1_coords"]
            src2_feats, src2_coords = batch["src2_feats"], batch["src2_coords"]
            alpha = batch["alpha"].to(device)
            target_ss = batch["target_ss_latent"].to(device)
            if target_ss.ndim == 6:
                target_ss = target_ss.squeeze(1)
            
            # SLAT exports
            if "target_feats" in batch and "target_coords" in batch:
                save_slat_glb(mesh_decoder, batch["target_feats"], batch["target_coords"], f"{sample_dir}/target_mid.glb", device)
            else:
                print("target_feats not found in batch, skipping mid target GLB")
                
            save_slat_glb(mesh_decoder, src1_feats, src1_coords, f"{sample_dir}/src1.glb", device)
            save_slat_glb(mesh_decoder, src2_feats, src2_coords, f"{sample_dir}/src2.glb", device)

            # Predict SS
            x0_pred = run_reverse_flow_sample(
                model=model, x0_shape=target_ss.shape,
                src1_feats=src1_feats.to(device), src2_feats=src2_feats.to(device),
                src1_coords=src1_coords.to(device), src2_coords=src2_coords.to(device),
                alpha=alpha, steps=args.steps, device=device
            )
            
            # SS exports
            save_ss_glb(ss_decoder, target_ss, f"{sample_dir}/gt_voxels.glb", device)
            save_ss_glb(ss_decoder, x0_pred, f"{sample_dir}/pred_voxels.glb", device)
            
            print(f"Sample {processed+1}/{args.num_samples} exported to {sample_dir}")
            processed += 1

if __name__ == "__main__":
    main()
