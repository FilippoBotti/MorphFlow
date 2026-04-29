import torch
import torch.nn as nn
import time

class AlphaEmbedder(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, alpha):
        alpha = alpha.view(-1, 1)
        return self.mlp(alpha.float())
    
        
class PairConditionFusionV2(nn.Module):
    def __init__(self, cond_dim=128, alpha_dim=64, hidden_dim=512, out_dim=512):
        super().__init__()
        self.alpha_embed = AlphaEmbedder(alpha_dim)

        self.gate_mlp = nn.Sequential(
            nn.Linear(cond_dim * 3 + alpha_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cond_dim),
            nn.Sigmoid(),
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(cond_dim * 3 + alpha_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, cond1, cond2, alpha):
        alpha_emb = self.alpha_embed(alpha)                    # [B, 64]
        alpha_emb = alpha_emb.unsqueeze(1).expand(-1, cond1.shape[1], -1)  # [B, 512, 64]

        delta = cond2 - cond1
        fusion_input = torch.cat([cond1, cond2, delta, alpha_emb], dim=-1)

        gate = self.gate_mlp(fusion_input)                    # [B, 512, 128]
        mixed = gate * cond2 + (1.0 - gate) * cond1

        out_input = torch.cat([mixed, delta, cond1 * cond2, alpha_emb], dim=-1)
        return self.out_mlp(out_input)                        # [B, 512, 512]


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

        self.global_mlp = nn.Sequential(
            nn.Linear(proj_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

        self.pos_emb = nn.Parameter(
            torch.randn(1, self.num_blocks, out_dim) * 0.02
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

        self.pos_scale = nn.Parameter(torch.tensor(1.0))
        self.coord_scale = nn.Parameter(torch.tensor(1.0))

        block_coords = self._build_block_coords()
        self.register_buffer("block_coords", block_coords)

    def _build_block_coords(self):
        coords = []
        for bx in range(self.n_blocks):
            for by in range(self.n_blocks):
                for bz in range(self.n_blocks):
                    cx = (bx + 0.5) * self.block_size
                    cy = (by + 0.5) * self.block_size
                    cz = (bz + 0.5) * self.block_size

                    cx = cx / self.grid_size
                    cy = cy / self.grid_size
                    cz = cz / self.grid_size

                    cx = cx * 2.0 - 1.0
                    cy = cy * 2.0 - 1.0
                    cz = cz * 2.0 - 1.0

                    coords.append([cx, cy, cz])

        return torch.tensor(coords, dtype=torch.float32)

    def forward(self, feats, coords):
        """
        Formato supportato: sparse concatenato dal collate custom

        feats:  [sumN, C]
        coords: [sumN, 4] con coords[:, 0] = batch_idx
        """
        assert feats.ndim == 2, f"Expected feats [sumN, C], got shape {feats.shape}"
        assert coords.ndim == 2, f"Expected coords [sumN, 4], got shape {coords.shape}"

        batch_ids = coords[:, 0].long()
        xyz = coords[:, 1:].long()

        B = int(batch_ids.max().item()) + 1

        feats = self.feat_proj(feats)   # [sumN, D]
        D = feats.shape[-1]

        bx = xyz[:, 0] // self.block_size
        by = xyz[:, 1] // self.block_size
        bz = xyz[:, 2] // self.block_size

        block_idx = bx * (self.n_blocks * self.n_blocks) + by * self.n_blocks + bz

        pooled_blocks = []

        for b in range(B):
            mask_b = (batch_ids == b)
            f = feats[mask_b]
            idx = block_idx[mask_b]

            mean_pool = torch.zeros(self.num_blocks, D, device=f.device, dtype=f.dtype)
            max_pool = torch.zeros(self.num_blocks, D, device=f.device, dtype=f.dtype)
            counts = torch.zeros(self.num_blocks, 1, device=f.device, dtype=f.dtype)

            mean_pool.index_add_(0, idx, f)
            counts.index_add_(
                0,
                idx,
                torch.ones((f.shape[0], 1), device=f.device, dtype=f.dtype)
            )
            mean_pool = mean_pool / counts.clamp(min=1.0)

            for k in range(self.num_blocks):
                mask_k = (idx == k)
                if mask_k.any():
                    max_pool[k] = f[mask_k].max(dim=0).values

            block_feat = torch.cat([mean_pool, max_pool], dim=-1)
            pooled_blocks.append(block_feat)

        pooled_blocks = torch.stack(pooled_blocks, dim=0)   # [B, num_blocks, 2D]
        cond = self.global_mlp(pooled_blocks)               # [B, num_blocks, out_dim]

        coord_emb = self.coord_mlp(self.block_coords)       # [num_blocks, out_dim]
        coord_emb = coord_emb.unsqueeze(0)                  # [1, num_blocks, out_dim]

        cond = cond + self.pos_scale * self.pos_emb + self.coord_scale * coord_emb
        return cond