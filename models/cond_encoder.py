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
    
class ConditionComposer(nn.Module):
    """
    Costruisce token separati per:
    - source 1
    - source 2
    - relazione tra source 1 e source 2

    Input:
        cond1, cond2: [B, T, C]
        alpha:        [B]

    Output:
        cond_tokens:  [B, 3T, out_dim]
    """
    def __init__(self, cond_dim=128, alpha_dim=64, out_dim=256):
        super().__init__()
        self.alpha_embed = AlphaEmbedder(alpha_dim)

        self.src_proj = nn.Sequential(
            nn.Linear(cond_dim + alpha_dim, 256),
            nn.SiLU(),
            nn.Linear(256, out_dim),
        )

        self.rel_proj = nn.Sequential(
            nn.Linear(cond_dim * 3 + alpha_dim, 256),
            nn.SiLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, cond1, cond2, alpha):
        """
        cond1, cond2: [B, T, C]
        alpha: [B]
        """
        assert cond1.ndim == 3, f"Expected cond1 [B, T, C], got {cond1.shape}"
        assert cond2.ndim == 3, f"Expected cond2 [B, T, C], got {cond2.shape}"
        assert cond1.shape == cond2.shape, f"Shape mismatch: {cond1.shape} vs {cond2.shape}"

        alpha_emb = self.alpha_embed(alpha)                     # [B, A]
        alpha_emb = alpha_emb.unsqueeze(1).expand(-1, cond1.shape[1], -1)  # [B, T, A]

        # token source-specific
        src1_tokens = self.src_proj(torch.cat([cond1, alpha_emb], dim=-1))  # [B, T, out_dim]
        src2_tokens = self.src_proj(torch.cat([cond2, alpha_emb], dim=-1))  # [B, T, out_dim]

        # token relazionali
        delta = cond2 - cond1
        abs_delta = torch.abs(delta)
        rel_input = torch.cat([delta, abs_delta, cond1 * cond2, alpha_emb], dim=-1)
        rel_tokens = self.rel_proj(rel_input)  # [B, T, out_dim]

        # output finale: 3 gruppi di token
        cond_tokens = torch.cat([src1_tokens, src2_tokens, rel_tokens], dim=1)  # [B, 3T, out_dim]
        return cond_tokens
        
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
        cond1, cond2: [B, num_blocks, C] oppure [B, C]
        alpha: [B] oppure [B, 1]
        """
        alpha = alpha.to(cond1.dtype).view(alpha.shape[0], *([1] * (cond1.ndim - 1)))
        mixed = (1.0 - alpha) * cond1 + alpha * cond2
        return self.fusion(mixed)


class MultiScaleBlockPoolConditionEncoder(nn.Module):
    def __init__(
        self,
        feat_dim=8,
        proj_dim=64,
        out_dim=128,
        grid_size=64,
        block_sizes=(4, 8, 16),
    ):
        super().__init__()
        self.grid_size = grid_size
        self.block_sizes = block_sizes

        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.SiLU(),
        )

        self.scale_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(proj_dim * 2, proj_dim * 2),
                nn.SiLU(),
                nn.Linear(proj_dim * 2, out_dim),
            )
            for _ in block_sizes
        ])

    def _pool_one_scale(self, feats, coords, block_size):
        """
        feats:  [sumN, D]
        coords: [sumN, 4] con coords[:,0]=batch_idx
        output: [B, num_blocks_scale, 2D]
        """
        batch_ids = coords[:, 0].long()
        xyz = coords[:, 1:].long()

        B = int(batch_ids.max().item()) + 1
        n_blocks = self.grid_size // block_size
        num_blocks = n_blocks ** 3
        D = feats.shape[-1]

        bx = xyz[:, 0] // block_size
        by = xyz[:, 1] // block_size
        bz = xyz[:, 2] // block_size
        block_idx = bx * (n_blocks * n_blocks) + by * n_blocks + bz

        pooled_blocks = []
        for b in range(B):
            mask_b = batch_ids == b
            f = feats[mask_b]
            idx = block_idx[mask_b]

            mean_pool = torch.zeros(num_blocks, D, device=f.device, dtype=f.dtype)
            max_pool = torch.zeros(num_blocks, D, device=f.device, dtype=f.dtype)
            counts = torch.zeros(num_blocks, 1, device=f.device, dtype=f.dtype)

            mean_pool.index_add_(0, idx, f)
            counts.index_add_(0, idx, torch.ones((f.shape[0], 1), device=f.device, dtype=f.dtype))
            mean_pool = mean_pool / counts.clamp(min=1.0)

            for k in range(num_blocks):
                mask_k = idx == k
                if mask_k.any():
                    max_pool[k] = f[mask_k].max(dim=0).values

            pooled_blocks.append(torch.cat([mean_pool, max_pool], dim=-1))

        return torch.stack(pooled_blocks, dim=0)  # [B, T_scale, 2D]

    def forward(self, feats, coords):
        feats = self.feat_proj(feats)  # [sumN, D]

        all_scale_tokens = []
        for block_size, mlp in zip(self.block_sizes, self.scale_mlps):
            pooled = self._pool_one_scale(feats, coords, block_size)  # [B, T, 2D]
            tokens = mlp(pooled)  # [B, T, out_dim]
            all_scale_tokens.append(tokens)

        # concatena i token di tutte le scale
        cond_tokens = torch.cat(all_scale_tokens, dim=1)  # [B, T_total, out_dim]
        return cond_tokens

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
            f = feats[mask_b]       # [Nb, D]
            idx = block_idx[mask_b] # [Nb]

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

        pooled_blocks = torch.stack(pooled_blocks, dim=0)  # [B, num_blocks, 2D]
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

        self.global_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 12, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, feats, coords):
        """
        Formato supportato: sparse concatenato dal collate custom

        feats:  [sumN, C]
        coords: [sumN, 4] con coords[:, 0] = batch_idx
        """
        assert feats.ndim == 2, f"Expected feats [sumN, C], got shape {feats.shape}"
        assert coords.ndim == 2, f"Expected coords [sumN, 4], got shape {coords.shape}"

        batch_ids = coords[:, 0].long()
        xyz = coords[:, 1:].float()

        B = int(batch_ids.max().item()) + 1

        xyz = xyz / 63.0
        xyz = xyz * 2.0 - 1.0

        tokens = torch.cat([feats, xyz], dim=-1)
        tokens = self.token_mlp(tokens)  # [sumN, H]

        pooled = []

        for b in range(B):
            mask_b = (batch_ids == b)
            t = tokens[mask_b]  # [Nb, H]
            x = xyz[mask_b]     # [Nb, 3]

            mean_pool = t.mean(dim=0)
            max_pool = t.max(dim=0).values

            center = x.mean(dim=0)
            std = x.std(dim=0)
            bbox_min = x.min(dim=0).values
            bbox_max = x.max(dim=0).values
            geom = torch.cat([center, std, bbox_min, bbox_max], dim=-1)

            global_feat = torch.cat([mean_pool, max_pool, geom], dim=-1)
            pooled.append(global_feat)

        pooled = torch.stack(pooled, dim=0)  # [B, 2H+12]
        cond = self.global_mlp(pooled)
        return cond


if __name__ == "__main__":
    N = 15803
    feats = torch.randn(N, 8)
    coords = torch.randint(0, 64, (N, 4))
    coords[:, 0] = 0  # bs=1 nel formato sparse concatenato

    encoder = BlockPoolConditionEncoder()

    time_start = time.time()
    for _ in range(1000):
        cond = encoder(feats, coords)
    time_end = time.time()
    print(cond.shape)
    print(f"Time taken: {time_end - time_start:.4f} seconds")