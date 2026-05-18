import os
import glob
import torch
from tqdm import tqdm
import sys

os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

# Aggiusta il path se necessario per importare trellis
TRELLIS_REPO = os.environ.get("TRELLIS_REPO", "/home/filippo/projects/TRELLIS")
if TRELLIS_REPO not in sys.path:
    sys.path.append(TRELLIS_REPO)

from trellis.models import from_pretrained as trellis_from_pretrained

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading TRELLIS sparse_structure_decoder...")
    decoder = trellis_from_pretrained("microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16").to(device)
    decoder.eval()

    dataset_root = "/home/filippo/datasets/3d/morphing_dataset_flux"
    
    # 1. assets (opzionale, ma spesso "ogni gt voxel" include i target di partenza)
    print("Processing assets...")
    asset_zs_files = glob.glob(os.path.join(dataset_root, "assets", "*", "cache", "coords_zs_init.pt"))
    for zs_path in tqdm(asset_zs_files):
        out_path = os.path.join(os.path.dirname(zs_path), "..", "occupancy_grid.pt")
        if os.path.exists(out_path):
            continue
            
        z_s = torch.load(zs_path, map_location=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            occ_logits = decoder(z_s)
        torch.save(occ_logits.cpu(), out_path)

    # 2. pairs (contengono i veri e propri gt morph voxel)
    print("Processing pairs...")
    pair_zs_files = glob.glob(os.path.join(dataset_root, "pairs*", "*", "mid_sparse_structure_latent.pt"))
    for zs_path in tqdm(pair_zs_files):
        out_path = os.path.join(os.path.dirname(zs_path), "mid_occupancy_grid.pt")
        if os.path.exists(out_path):
            continue
            
        z_s = torch.load(zs_path, map_location=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            occ_logits = decoder(z_s)
        torch.save(occ_logits.cpu(), out_path)

    print("Done!")

if __name__ == "__main__":
    main()
