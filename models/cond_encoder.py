import torch
import torch.nn as nn
import time
import math

from modules import sparse as sp

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
        alpha_emb = self.alpha_embed(alpha)                    
        alpha_emb = alpha_emb.unsqueeze(1).expand(-1, cond1.shape[1], -1)  

        delta = cond2 - cond1
        fusion_input = torch.cat([cond1, cond2, delta, alpha_emb], dim=-1)

        gate = self.gate_mlp(fusion_input)                    
        mixed = gate * cond2 + (1.0 - gate) * cond1

        out_input = torch.cat([mixed, delta, cond1 * cond2, alpha_emb], dim=-1)
        return self.out_mlp(out_input)                        


class BlockPoolConditionEncoder(nn.Module):
    def __init__(self, feat_dim=8, proj_dim=64, block_size=4, out_dim=128):
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
            nn.Linear(proj_dim * 2 + 2, 512),
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
        global_idx = batch_ids * self.num_blocks + block_idx   # [sumN]

        total_blocks = B * self.num_blocks

        mean_pool = torch.zeros(
            total_blocks, D,
            device=feats.device,
            dtype=feats.dtype,
        )

        counts = torch.zeros(
            total_blocks, 1,
            device=feats.device,
            dtype=feats.dtype,
        )

        mean_pool.index_add_(0, global_idx, feats)

        ones = torch.ones(
            feats.shape[0], 1,
            device=feats.device,
            dtype=feats.dtype,
        )
        counts.index_add_(0, global_idx, ones)

        mean_pool = mean_pool / counts.clamp(min=1.0)

        max_pool = torch.full(
            (total_blocks, D),
            -torch.inf,
            device=feats.device,
            dtype=feats.dtype,
        )

        expanded_idx = global_idx.view(-1, 1).expand(-1, D)

        max_pool.scatter_reduce_(
            0,
            expanded_idx,
            feats,
            reduce="amax",
            include_self=True,
        )

        max_pool = torch.where(
            torch.isinf(max_pool),
            torch.zeros_like(max_pool),
            max_pool,
        )

        occupancy = (counts > 0).to(dtype=feats.dtype)
        log_count = torch.log1p(counts) / math.log1p(float(self.block_size ** 3))

        block_feat = torch.cat(
            [mean_pool, max_pool, occupancy, log_count],
            dim=-1,
        )

        pooled_blocks = block_feat.view(B, self.num_blocks, -1)
        cond = self.global_mlp(pooled_blocks)

        coord_emb = self.coord_mlp(self.block_coords)
        coord_emb = coord_emb.unsqueeze(0)

        cond = cond + self.pos_scale * self.pos_emb + self.coord_scale * coord_emb
        return cond


class Conv3DConditionEncoder(nn.Module):
    """
    Dense 3D-conv condition encoder for sparse SLat source features.

    It keeps the same output contract as BlockPoolConditionEncoder:
        [B, 4096, 128]

    The sparse coordinates are voxelized on the 64^3 grid, then two stride-2
    3D convolutions reduce the grid to 16^3 tokens, matching TRELLIS SS token
    resolution and the existing condition-fusion/cross-attention path.
    """

    def __init__(
        self,
        feat_dim=8,
        hidden_dim=64,
        out_dim=128,
        grid_size=64,
        downsample_factor=4,
    ):
        super().__init__()
        if grid_size % downsample_factor != 0:
            raise ValueError(
                f"grid_size must be divisible by downsample_factor, got "
                f"{grid_size} and {downsample_factor}"
            )
        if downsample_factor != 4:
            raise ValueError("Conv3DConditionEncoder currently expects downsample_factor=4")
        if hidden_dim % 8 != 0:
            raise ValueError(f"hidden_dim must be divisible by 8 for GroupNorm, got {hidden_dim}")

        self.grid_size = int(grid_size)
        self.downsample_factor = int(downsample_factor)
        self.n_blocks = self.grid_size // self.downsample_factor
        self.num_blocks = self.n_blocks ** 3

        self.net = nn.Sequential(
            nn.Conv3d(feat_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, out_dim, kernel_size=1),
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
                    cx = (bx + 0.5) * self.downsample_factor
                    cy = (by + 0.5) * self.downsample_factor
                    cz = (bz + 0.5) * self.downsample_factor

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
        feats:  [sumN, C]
        coords: [sumN, 4] with coords[:, 0] = batch_idx
        """
        assert feats.ndim == 2, f"Expected feats [sumN, C], got shape {feats.shape}"
        assert coords.ndim == 2, f"Expected coords [sumN, 4], got shape {coords.shape}"

        batch_ids = coords[:, 0].long()
        xyz = coords[:, 1:].long().clamp(0, self.grid_size - 1)

        B = int(batch_ids.max().item()) + 1
        C = feats.shape[-1]

        dense = torch.zeros(
            B,
            C,
            self.grid_size,
            self.grid_size,
            self.grid_size,
            device=feats.device,
            dtype=feats.dtype,
        )
        dense[batch_ids, :, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = feats

        cond_grid = self.net(dense)
        expected_shape = (self.n_blocks, self.n_blocks, self.n_blocks)
        if cond_grid.shape[-3:] != expected_shape:
            raise RuntimeError(
                f"Unexpected Conv3DConditionEncoder output grid {cond_grid.shape[-3:]}; "
                f"expected {expected_shape}"
            )

        cond = cond_grid.flatten(2).transpose(1, 2)
        coord_emb = self.coord_mlp(self.block_coords).unsqueeze(0)

        cond = cond + self.pos_scale * self.pos_emb + self.coord_scale * coord_emb
        return cond


class SparseConv3DConditionEncoder(nn.Module):
    """
    Sparse 3D-conv condition encoder using the TRELLIS sparse/spconv wrapper.

    The output is intentionally dense and ordered like BlockPoolConditionEncoder:
        [B, 4096, 128]

    Sparse convolutions are used only to build spatially-aware asset tokens. The
    resulting active 16^3 features are scattered into the fixed token grid so the
    downstream pair fusion and cross-attention path can stay unchanged.
    """

    def __init__(
        self,
        feat_dim=8,
        hidden_dim=64,
        out_dim=128,
        grid_size=64,
        downsample_factor=4,
    ):
        super().__init__()
        if grid_size % downsample_factor != 0:
            raise ValueError(
                f"grid_size must be divisible by downsample_factor, got "
                f"{grid_size} and {downsample_factor}"
            )
        if downsample_factor != 4:
            raise ValueError("SparseConv3DConditionEncoder currently expects downsample_factor=4")
        if hidden_dim % 8 != 0:
            raise ValueError(f"hidden_dim must be divisible by 8 for GroupNorm, got {hidden_dim}")

        self.grid_size = int(grid_size)
        self.downsample_factor = int(downsample_factor)
        self.n_blocks = self.grid_size // self.downsample_factor
        self.num_blocks = self.n_blocks ** 3

        self.net = nn.Sequential(
            sp.SparseConv3d(feat_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            sp.SparseGroupNorm(8, hidden_dim),
            sp.SparseSiLU(),
            sp.SparseConv3d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            sp.SparseGroupNorm(8, hidden_dim),
            sp.SparseSiLU(),
            sp.SparseConv3d(hidden_dim, hidden_dim, kernel_size=3),
            sp.SparseGroupNorm(8, hidden_dim),
            sp.SparseSiLU(),
            sp.SparseConv3d(hidden_dim, out_dim, kernel_size=1),
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
                    cx = (bx + 0.5) * self.downsample_factor
                    cy = (by + 0.5) * self.downsample_factor
                    cz = (bz + 0.5) * self.downsample_factor

                    cx = cx / self.grid_size
                    cy = cy / self.grid_size
                    cz = cz / self.grid_size

                    cx = cx * 2.0 - 1.0
                    cy = cy * 2.0 - 1.0
                    cz = cz * 2.0 - 1.0

                    coords.append([cx, cy, cz])

        return torch.tensor(coords, dtype=torch.float32)

    def _make_sparse_tensor(self, feats, coords, batch_size):
        coords = coords.to(device=feats.device, dtype=torch.int32)
        if getattr(sp, "BACKEND", "spconv") == "spconv":
            import spconv.pytorch as spconv

            data = spconv.SparseConvTensor(
                feats.reshape(feats.shape[0], -1),
                coords,
                [self.grid_size, self.grid_size, self.grid_size],
                batch_size,
            )
            data._features = feats
            return sp.SparseTensor(data, shape=torch.Size([batch_size, feats.shape[-1]]))

        return sp.SparseTensor(
            feats=feats,
            coords=coords,
            shape=torch.Size([batch_size, feats.shape[-1]]),
        )

    def forward(self, feats, coords):
        """
        feats:  [sumN, C]
        coords: [sumN, 4] with coords[:, 0] = batch_idx
        """
        assert feats.ndim == 2, f"Expected feats [sumN, C], got shape {feats.shape}"
        assert coords.ndim == 2, f"Expected coords [sumN, 4], got shape {coords.shape}"

        batch_ids = coords[:, 0].long()
        xyz = coords[:, 1:].long().clamp(0, self.grid_size - 1)
        clipped_coords = torch.cat([batch_ids.view(-1, 1), xyz], dim=1)

        B = int(batch_ids.max().item()) + 1
        sparse = self._make_sparse_tensor(feats, clipped_coords, B)
        sparse = self.net(sparse)

        out_coords = sparse.coords.long()
        out_feats = sparse.feats

        out_batch = out_coords[:, 0]
        out_xyz = out_coords[:, 1:].clamp(0, self.n_blocks - 1)
        token_idx = (
            out_xyz[:, 0] * (self.n_blocks * self.n_blocks)
            + out_xyz[:, 1] * self.n_blocks
            + out_xyz[:, 2]
        )
        global_idx = out_batch * self.num_blocks + token_idx

        cond = torch.zeros(
            B * self.num_blocks,
            out_feats.shape[-1],
            device=out_feats.device,
            dtype=out_feats.dtype,
        )
        counts = torch.zeros(
            B * self.num_blocks,
            1,
            device=out_feats.device,
            dtype=out_feats.dtype,
        )

        cond.index_add_(0, global_idx, out_feats)
        ones = torch.ones(out_feats.shape[0], 1, device=out_feats.device, dtype=out_feats.dtype)
        counts.index_add_(0, global_idx, ones)
        cond = cond / counts.clamp(min=1.0)

        cond = cond.view(B, self.num_blocks, -1)
        coord_emb = self.coord_mlp(self.block_coords).unsqueeze(0)

        cond = cond + self.pos_scale * self.pos_emb + self.coord_scale * coord_emb
        return cond


def build_condition_encoder(encoder_type="block", **kwargs):
    if encoder_type == "block":
        return BlockPoolConditionEncoder(**kwargs)
    if encoder_type == "conv3d":
        return Conv3DConditionEncoder(**kwargs)
    if encoder_type in ("sparse_conv3d", "sparse_conv"):
        return SparseConv3DConditionEncoder(**kwargs)
    raise ValueError(f"Unknown condition encoder type: {encoder_type}")
