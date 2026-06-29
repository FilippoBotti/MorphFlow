import inspect
import math

import torch
import torch.nn as nn

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


class ConditionResamplerBlock(nn.Module):
    def __init__(self, dim=128, heads=8, mlp_ratio=4):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.ctx_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, queries, context):
        context = self.ctx_norm(context)
        attn_out, _ = self.attn(self.q_norm(queries), context, context, need_weights=False)
        queries = queries + attn_out
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class ConditionResampler(nn.Module):
    """Perceiver-style learned-query resampler: [B, N, D] -> [B, K, D]."""

    def __init__(self, dim=128, num_tokens=512, depth=1, heads=8):
        super().__init__()
        if int(num_tokens) <= 0:
            raise ValueError(f"num_tokens must be > 0, got {num_tokens}")
        self.num_tokens = int(num_tokens)
        self.dim = int(dim)
        self.queries = nn.Parameter(torch.randn(1, self.num_tokens, self.dim) * 0.02)
        self.layers = nn.ModuleList(
            [ConditionResamplerBlock(dim=self.dim, heads=heads) for _ in range(max(1, int(depth)))]
        )
        self.out_norm = nn.LayerNorm(self.dim)

    def forward(self, tokens):
        queries = self.queries.expand(tokens.shape[0], -1, -1).to(dtype=tokens.dtype)
        for layer in self.layers:
            queries = layer(queries, tokens)
        return self.out_norm(queries)


class SparseConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=8):
        super().__init__()
        self.conv = sp.SparseConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = sp.SparseGroupNorm(groups, out_channels)
        self.act = sp.SparseSiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SparseResidualBlock(nn.Module):
    """Submanifold residual block: keeps the same active sparse coordinates."""

    def __init__(self, channels, groups=8):
        super().__init__()
        self.conv1 = sp.SparseConv3d(channels, channels, kernel_size=3)
        self.norm1 = sp.SparseGroupNorm(groups, channels)
        self.act1 = sp.SparseSiLU()
        self.conv2 = sp.SparseConv3d(channels, channels, kernel_size=3)
        self.norm2 = sp.SparseGroupNorm(groups, channels)
        self.act_out = sp.SparseSiLU()

    def forward(self, x):
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act_out(x.replace(x.feats + h.feats))


class BlockPoolConditionEncoder(nn.Module):
    def __init__(self, feat_dim=8, proj_dim=64, block_size=4, out_dim=128):
        super().__init__()
        self.block_size = block_size
        self.grid_size = 64
        self.n_blocks = self.grid_size // block_size
        self.num_blocks = self.n_blocks ** 3
        self.num_output_tokens = self.num_blocks

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

        self.pos_emb = nn.Parameter(torch.randn(1, self.num_blocks, out_dim) * 0.02)

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
                    center = torch.tensor(
                        [
                            (bx + 0.5) * self.block_size,
                            (by + 0.5) * self.block_size,
                            (bz + 0.5) * self.block_size,
                        ],
                        dtype=torch.float32,
                    )
                    center = center / self.grid_size * 2.0 - 1.0
                    coords.append(center)
        return torch.stack(coords, dim=0)

    def forward(self, feats, coords):
        assert feats.ndim == 2, f"Expected feats [sumN, C], got shape {feats.shape}"
        assert coords.ndim == 2, f"Expected coords [sumN, 4], got shape {coords.shape}"

        batch_ids = coords[:, 0].long()
        xyz = coords[:, 1:].long()
        B = int(batch_ids.max().item()) + 1

        feats = self.feat_proj(feats)
        D = feats.shape[-1]

        bx = xyz[:, 0] // self.block_size
        by = xyz[:, 1] // self.block_size
        bz = xyz[:, 2] // self.block_size

        block_idx = bx * (self.n_blocks * self.n_blocks) + by * self.n_blocks + bz
        global_idx = batch_ids * self.num_blocks + block_idx
        total_blocks = B * self.num_blocks

        mean_pool = torch.zeros(total_blocks, D, device=feats.device, dtype=feats.dtype)
        counts = torch.zeros(total_blocks, 1, device=feats.device, dtype=feats.dtype)
        mean_pool.index_add_(0, global_idx, feats)
        counts.index_add_(0, global_idx, torch.ones(feats.shape[0], 1, device=feats.device, dtype=feats.dtype))
        mean_pool = mean_pool / counts.clamp(min=1.0)

        max_pool = torch.full((total_blocks, D), -torch.inf, device=feats.device, dtype=feats.dtype)
        expanded_idx = global_idx.view(-1, 1).expand(-1, D)
        max_pool.scatter_reduce_(0, expanded_idx, feats, reduce="amax", include_self=True)
        max_pool = torch.where(torch.isinf(max_pool), torch.zeros_like(max_pool), max_pool)

        occupancy = (counts > 0).to(dtype=feats.dtype)
        log_count = torch.log1p(counts) / math.log1p(float(self.block_size ** 3))

        block_feat = torch.cat([mean_pool, max_pool, occupancy, log_count], dim=-1)
        cond = self.global_mlp(block_feat.view(B, self.num_blocks, -1))
        coord_emb = self.coord_mlp(self.block_coords).unsqueeze(0)
        return cond + self.pos_scale * self.pos_emb + self.coord_scale * coord_emb


class Conv3DConditionEncoder(nn.Module):
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
            raise ValueError(f"grid_size must be divisible by downsample_factor, got {grid_size} and {downsample_factor}")
        if downsample_factor != 4:
            raise ValueError("Conv3DConditionEncoder currently expects downsample_factor=4")
        if hidden_dim % 8 != 0:
            raise ValueError(f"hidden_dim must be divisible by 8 for GroupNorm, got {hidden_dim}")

        self.grid_size = int(grid_size)
        self.downsample_factor = int(downsample_factor)
        self.n_blocks = self.grid_size // self.downsample_factor
        self.num_blocks = self.n_blocks ** 3
        self.num_output_tokens = self.num_blocks

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
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_blocks, out_dim) * 0.02)
        self.coord_mlp = nn.Sequential(nn.Linear(3, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))
        self.pos_scale = nn.Parameter(torch.tensor(1.0))
        self.coord_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("block_coords", self._build_block_coords())

    def _build_block_coords(self):
        coords = []
        for bx in range(self.n_blocks):
            for by in range(self.n_blocks):
                for bz in range(self.n_blocks):
                    center = torch.tensor(
                        [
                            (bx + 0.5) * self.downsample_factor,
                            (by + 0.5) * self.downsample_factor,
                            (bz + 0.5) * self.downsample_factor,
                        ],
                        dtype=torch.float32,
                    )
                    center = center / self.grid_size * 2.0 - 1.0
                    coords.append(center)
        return torch.stack(coords, dim=0)

    def forward(self, feats, coords):
        assert feats.ndim == 2, f"Expected feats [sumN, C], got shape {feats.shape}"
        assert coords.ndim == 2, f"Expected coords [sumN, 4], got shape {coords.shape}"
        batch_ids = coords[:, 0].long()
        xyz = coords[:, 1:].long().clamp(0, self.grid_size - 1)
        B = int(batch_ids.max().item()) + 1
        C = feats.shape[-1]
        dense = torch.zeros(B, C, self.grid_size, self.grid_size, self.grid_size, device=feats.device, dtype=feats.dtype)
        dense[batch_ids, :, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = feats
        cond_grid = self.net(dense)
        expected_shape = (self.n_blocks, self.n_blocks, self.n_blocks)
        if cond_grid.shape[-3:] != expected_shape:
            raise RuntimeError(f"Unexpected Conv3DConditionEncoder output grid {cond_grid.shape[-3:]}; expected {expected_shape}")
        cond = cond_grid.flatten(2).transpose(1, 2)
        coord_emb = self.coord_mlp(self.block_coords).unsqueeze(0)
        return cond + self.pos_scale * self.pos_emb + self.coord_scale * coord_emb


class SparseConv3DConditionEncoder(nn.Module):
    """
    Sparse 3D-conv condition encoder using the TRELLIS sparse/spconv wrapper.

    Optional upgrades:
      - global style tokens per source asset;
      - explicit occupied/empty token handling;
      - residual sparse blocks before downsampling at 64^3;
      - local pooling statistics fused next to sparse-conv features.
    """

    def __init__(
        self,
        feat_dim=8,
        hidden_dim=64,
        out_dim=128,
        grid_size=64,
        downsample_factor=4,
        style_tokens=0,
        use_occupancy=False,
        hybrid_pool_stats=False,
        residual_blocks_64=0,
        residual_blocks_32=0,
        residual_blocks_16=0,
    ):
        super().__init__()
        if grid_size % downsample_factor != 0:
            raise ValueError(f"grid_size must be divisible by downsample_factor, got {grid_size} and {downsample_factor}")
        if downsample_factor != 4:
            raise ValueError("SparseConv3DConditionEncoder currently expects downsample_factor=4")
        if hidden_dim % 8 != 0:
            raise ValueError(f"hidden_dim must be divisible by 8 for GroupNorm, got {hidden_dim}")

        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.grid_size = int(grid_size)
        self.downsample_factor = int(downsample_factor)
        self.n_blocks = self.grid_size // self.downsample_factor
        self.num_blocks = self.n_blocks ** 3
        self.style_tokens = int(style_tokens)
        self.use_occupancy = bool(use_occupancy)
        self.hybrid_pool_stats = bool(hybrid_pool_stats)
        self.residual_blocks_64 = int(residual_blocks_64)
        self.residual_blocks_32 = int(residual_blocks_32)
        self.residual_blocks_16 = int(residual_blocks_16)
        self.num_output_tokens = self.num_blocks + self.style_tokens

        self.input_proj = SparseConvNormAct(feat_dim, hidden_dim, kernel_size=1, groups=8)
        self.res64 = nn.Sequential(*[SparseResidualBlock(hidden_dim, groups=8) for _ in range(self.residual_blocks_64)])
        self.down1 = SparseConvNormAct(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, groups=8)
        self.res32 = nn.Sequential(*[SparseResidualBlock(hidden_dim, groups=8) for _ in range(self.residual_blocks_32)])
        self.down2 = SparseConvNormAct(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, groups=8)
        self.res16 = nn.Sequential(*[SparseResidualBlock(hidden_dim, groups=8) for _ in range(self.residual_blocks_16)])
        self.mid = SparseConvNormAct(hidden_dim, hidden_dim, kernel_size=3, groups=8)
        self.out_proj = sp.SparseConv3d(hidden_dim, out_dim, kernel_size=1)

        pool_dim = feat_dim * 3 + 2
        if self.hybrid_pool_stats:
            self.pool_stats_mlp = nn.Sequential(
                nn.Linear(out_dim + pool_dim, out_dim * 2),
                nn.SiLU(),
                nn.Linear(out_dim * 2, out_dim),
            )

        if self.use_occupancy:
            self.empty_token = nn.Parameter(torch.zeros(1, 1, out_dim))
            self.occupancy_mlp = nn.Sequential(
                nn.Linear(2, out_dim),
                nn.SiLU(),
                nn.Linear(out_dim, out_dim),
            )

        if self.style_tokens > 0:
            style_dim = feat_dim * 4 + 2
            self.style_mlp = nn.Sequential(
                nn.Linear(style_dim, out_dim * max(2, self.style_tokens)),
                nn.SiLU(),
                nn.Linear(out_dim * max(2, self.style_tokens), self.style_tokens * out_dim),
            )
            self.style_pos_emb = nn.Parameter(torch.randn(1, self.style_tokens, out_dim) * 0.02)

        self.pos_emb = nn.Parameter(torch.randn(1, self.num_blocks, out_dim) * 0.02)
        self.coord_mlp = nn.Sequential(nn.Linear(3, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))
        self.pos_scale = nn.Parameter(torch.tensor(1.0))
        self.coord_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("block_coords", self._build_block_coords())

    def _build_block_coords(self):
        coords = []
        for bx in range(self.n_blocks):
            for by in range(self.n_blocks):
                for bz in range(self.n_blocks):
                    center = torch.tensor(
                        [
                            (bx + 0.5) * self.downsample_factor,
                            (by + 0.5) * self.downsample_factor,
                            (bz + 0.5) * self.downsample_factor,
                        ],
                        dtype=torch.float32,
                    )
                    center = center / self.grid_size * 2.0 - 1.0
                    coords.append(center)
        return torch.stack(coords, dim=0)

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

        return sp.SparseTensor(feats=feats, coords=coords, shape=torch.Size([batch_size, feats.shape[-1]]))

    def _block_global_idx(self, batch_ids, xyz):
        bx = (xyz[:, 0] // self.downsample_factor).clamp(0, self.n_blocks - 1)
        by = (xyz[:, 1] // self.downsample_factor).clamp(0, self.n_blocks - 1)
        bz = (xyz[:, 2] // self.downsample_factor).clamp(0, self.n_blocks - 1)
        token_idx = bx * (self.n_blocks * self.n_blocks) + by * self.n_blocks + bz
        return batch_ids * self.num_blocks + token_idx

    def _local_pool_stats(self, feats, batch_ids, xyz, B):
        global_idx = self._block_global_idx(batch_ids, xyz)
        total = B * self.num_blocks
        D = feats.shape[-1]
        dtype = feats.dtype
        device = feats.device

        counts = torch.zeros(total, 1, device=device, dtype=dtype)
        ones = torch.ones(feats.shape[0], 1, device=device, dtype=dtype)
        counts.index_add_(0, global_idx, ones)

        sum_pool = torch.zeros(total, D, device=device, dtype=dtype)
        sum_pool.index_add_(0, global_idx, feats)
        mean_pool = sum_pool / counts.clamp(min=1.0)

        sum_sq = torch.zeros(total, D, device=device, dtype=dtype)
        sum_sq.index_add_(0, global_idx, feats * feats)
        mean_sq = sum_sq / counts.clamp(min=1.0)
        std_pool = (mean_sq - mean_pool * mean_pool).clamp(min=0).sqrt()

        max_pool = torch.full((total, D), -torch.inf, device=device, dtype=dtype)
        expanded = global_idx.view(-1, 1).expand(-1, D)
        max_pool.scatter_reduce_(0, expanded, feats, reduce="amax", include_self=True)
        max_pool = torch.where(torch.isinf(max_pool), torch.zeros_like(max_pool), max_pool)

        occupancy = (counts > 0).to(dtype=dtype)
        log_count = torch.log1p(counts) / math.log1p(float(self.downsample_factor ** 3))
        pool = torch.cat([mean_pool, max_pool, std_pool, occupancy, log_count], dim=-1)
        return (
            pool.view(B, self.num_blocks, -1),
            occupancy.view(B, self.num_blocks, 1),
            log_count.view(B, self.num_blocks, 1),
        )

    def _global_style_tokens(self, feats, batch_ids, B):
        if self.style_tokens <= 0:
            return None
        D = feats.shape[-1]
        dtype = feats.dtype
        device = feats.device
        counts = torch.zeros(B, 1, device=device, dtype=dtype)
        ones = torch.ones(feats.shape[0], 1, device=device, dtype=dtype)
        counts.index_add_(0, batch_ids, ones)

        sum_pool = torch.zeros(B, D, device=device, dtype=dtype)
        sum_pool.index_add_(0, batch_ids, feats)
        mean_pool = sum_pool / counts.clamp(min=1.0)

        sum_sq = torch.zeros(B, D, device=device, dtype=dtype)
        sum_sq.index_add_(0, batch_ids, feats * feats)
        mean_sq = sum_sq / counts.clamp(min=1.0)
        std_pool = (mean_sq - mean_pool * mean_pool).clamp(min=0).sqrt()

        expanded = batch_ids.view(-1, 1).expand(-1, D)
        min_pool = torch.full((B, D), torch.inf, device=device, dtype=dtype)
        max_pool = torch.full((B, D), -torch.inf, device=device, dtype=dtype)
        min_pool.scatter_reduce_(0, expanded, feats, reduce="amin", include_self=True)
        max_pool.scatter_reduce_(0, expanded, feats, reduce="amax", include_self=True)
        min_pool = torch.where(torch.isinf(min_pool), torch.zeros_like(min_pool), min_pool)
        max_pool = torch.where(torch.isinf(max_pool), torch.zeros_like(max_pool), max_pool)

        occupancy_ratio = counts / float(self.grid_size ** 3)
        log_count = torch.log1p(counts) / math.log1p(float(self.grid_size ** 3))
        style_stats = torch.cat([mean_pool, std_pool, min_pool, max_pool, occupancy_ratio, log_count], dim=-1)
        style = self.style_mlp(style_stats).view(B, self.style_tokens, self.out_dim)
        return style + self.style_pos_emb.to(device=device, dtype=style.dtype)

    def forward(self, feats, coords):
        assert feats.ndim == 2, f"Expected feats [sumN, C], got shape {feats.shape}"
        assert coords.ndim == 2, f"Expected coords [sumN, 4], got shape {coords.shape}"

        batch_ids = coords[:, 0].long()
        xyz = coords[:, 1:].long().clamp(0, self.grid_size - 1)
        clipped_coords = torch.cat([batch_ids.view(-1, 1), xyz], dim=1)
        B = int(batch_ids.max().item()) + 1

        local_stats = occupancy = log_count = None
        if self.hybrid_pool_stats or self.use_occupancy:
            local_stats, occupancy, log_count = self._local_pool_stats(feats, batch_ids, xyz, B)

        style_tokens = self._global_style_tokens(feats, batch_ids, B)

        sparse = self._make_sparse_tensor(feats, clipped_coords, B)
        sparse = self.input_proj(sparse)
        sparse = self.res64(sparse)
        sparse = self.down1(sparse)
        sparse = self.res32(sparse)
        sparse = self.down2(sparse)
        sparse = self.res16(sparse)
        sparse = self.mid(sparse)
        sparse = self.out_proj(sparse)

        out_coords = sparse.coords.long()
        out_feats = sparse.feats
        out_batch = out_coords[:, 0]
        out_xyz = out_coords[:, 1:].clamp(0, self.n_blocks - 1)
        token_idx = out_xyz[:, 0] * (self.n_blocks * self.n_blocks) + out_xyz[:, 1] * self.n_blocks + out_xyz[:, 2]
        global_idx = out_batch * self.num_blocks + token_idx

        cond = torch.zeros(B * self.num_blocks, out_feats.shape[-1], device=out_feats.device, dtype=out_feats.dtype)
        counts = torch.zeros(B * self.num_blocks, 1, device=out_feats.device, dtype=out_feats.dtype)
        cond.index_add_(0, global_idx, out_feats)
        counts.index_add_(0, global_idx, torch.ones(out_feats.shape[0], 1, device=out_feats.device, dtype=out_feats.dtype))
        cond = cond / counts.clamp(min=1.0)
        cond = cond.view(B, self.num_blocks, -1)

        if self.hybrid_pool_stats:
            cond = self.pool_stats_mlp(torch.cat([cond, local_stats.to(dtype=cond.dtype)], dim=-1))

        if self.use_occupancy:
            occupancy = occupancy.to(dtype=cond.dtype)
            log_count = log_count.to(dtype=cond.dtype)
            empty = self.empty_token.to(device=cond.device, dtype=cond.dtype).expand(B, self.num_blocks, -1)
            cond = torch.where(occupancy.bool(), cond, empty)
            cond = cond + self.occupancy_mlp(torch.cat([occupancy, log_count], dim=-1))

        coord_emb = self.coord_mlp(self.block_coords).unsqueeze(0)
        cond = cond + self.pos_scale * self.pos_emb + self.coord_scale * coord_emb

        if style_tokens is not None:
            cond = torch.cat([cond, style_tokens.to(dtype=cond.dtype)], dim=1)
        return cond


def build_condition_encoder(encoder_type="block", **kwargs):
    if encoder_type == "block":
        cls = BlockPoolConditionEncoder
    elif encoder_type == "conv3d":
        cls = Conv3DConditionEncoder
    elif encoder_type in ("sparse_conv3d", "sparse_conv"):
        cls = SparseConv3DConditionEncoder
    else:
        raise ValueError(f"Unknown condition encoder type: {encoder_type}")

    supported = set(inspect.signature(cls.__init__).parameters)
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    return cls(**filtered)
