import torch
import torch.nn as nn
import time    


class PairConditionFusion(nn.Module):
    def __init__(self, cond_dim=128, out_dim=256):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, cond1, cond2, alpha):
        """
        cond1, cond2: [B, C]
        alpha: [B, 1]
        """
        mixed = (1.0 - alpha) * cond1 + alpha * cond2
     
        return self.fusion(mixed)
    

class BlockPoolConditionEncoder(nn.Module):
    def __init__(self, feat_dim=8, proj_dim=64, block_size=8, out_dim=128):
        super().__init__()
        self.block_size = block_size
        self.grid_size = 64
        self.n_blocks = self.grid_size // block_size
        self.num_blocks = self.n_blocks ** 3

        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
        )

        # mean + max pooling per block
        block_feat_dim = proj_dim * 2

        self.global_mlp = nn.Sequential(
            nn.Linear(proj_dim*2, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, feats, coords):
        """
        feats:  [B, N, 8]
        coords: [B, N, 4]  -> [batch_idx, x, y, z]
        """
        B, N, _ = feats.shape
        xyz = coords[..., 1:].long()   # [B, N, 3]

        feats = self.feat_proj(feats)  # [B, N, D]
        D = feats.shape[-1]

        bx = xyz[..., 0] // self.block_size
        by = xyz[..., 1] // self.block_size
        bz = xyz[..., 2] // self.block_size

        block_idx = bx * (self.n_blocks * self.n_blocks) + by * self.n_blocks + bz
        # [B, N], values in [0, num_blocks-1]

        pooled_blocks = []

        for b in range(B):
            f = feats[b]         # [N, D]
            idx = block_idx[b]   # [N]

            mean_pool = torch.zeros(self.num_blocks, D, device=f.device, dtype=f.dtype)
            max_pool = torch.full((self.num_blocks, D), -1e9, device=f.device, dtype=f.dtype)
            counts = torch.zeros(self.num_blocks, 1, device=f.device, dtype=f.dtype)

            mean_pool.index_add_(0, idx, f)
            counts.index_add_(0, idx, torch.ones_like(idx, dtype=f.dtype).unsqueeze(-1))
            mean_pool = mean_pool / counts.clamp(min=1.0)

            # max pooling per blocco
            for k in range(self.num_blocks):
                mask = (idx == k)
                if mask.any():
                    max_pool[k] = f[mask].max(dim=0).values
                else:
                    max_pool[k] = 0.0

            block_feat = torch.cat([mean_pool, max_pool], dim=-1)  # [num_blocks, 2D]
            pooled_blocks.append(block_feat)

        pooled_blocks = torch.stack(pooled_blocks, dim=0)  # [B, num_blocks, 2D]
        # print("Pooled blocks shape:", pooled_blocks.shape)
        # exit()
        # flat = pooled_blocks.reshape(B, -1)
        cond = self.global_mlp(pooled_blocks)
        return cond
    

class SparseShapeConditionEncoder(nn.Module):
    def __init__(self, feat_dim=8, hidden_dim=128, out_dim=128):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(feat_dim + 3, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )
        
        # mean + max pooling => 2 * hidden_dim
        # + geometric stats: center(3), std(3), bbox_min(3), bbox_max(3) = 12
        self.global_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 12, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, feats, coords):
        """
        feats:  [B, N, 8]
        coords: [B, N, 4]  # batch_idx, x, y, z
        """
        xyz = coords[..., 1:].float()  # [B, N, 3]

        # normalize coords from [0, 63] to [-1, 1]
        xyz = xyz / 63.0
        xyz = xyz * 2.0 - 1.0

        tokens = torch.cat([feats, xyz], dim=-1)  # [B, N, 11]
        tokens = self.token_mlp(tokens)           # [B, N, H]

        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.max(dim=1).values

        center = xyz.mean(dim=1)
        std = xyz.std(dim=1)
        bbox_min = xyz.min(dim=1).values
        bbox_max = xyz.max(dim=1).values
        geom = torch.cat([center, std, bbox_min, bbox_max], dim=-1)

        global_feat = torch.cat([mean_pool, max_pool, geom], dim=-1)
        cond = self.global_mlp(global_feat)
        return cond

if __name__ == "__main__":
    B, N = 1, 15803
    feats = torch.randn(B, N, 8)
    coords = torch.randint(0, 64, (B, N, 4))
    encoder = BlockPoolConditionEncoder()
    
    time_start = time.time()
    for i in range(1000):
        cond = encoder(feats, coords)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start:.4f} seconds")