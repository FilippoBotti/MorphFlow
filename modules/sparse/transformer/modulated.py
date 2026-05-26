from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor
from ..attention import SparseMultiHeadAttention, SerializeMode
from ...norm import LayerNorm32
from .blocks import SparseFeedForwardNet


class ModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        separate_cond: bool = False,
        separate_cond_gate: Literal["alpha_residual", "pair_channel", "token"] = "alpha_residual",

    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.separate_cond = separate_cond
        self.separate_cond_gate = separate_cond_gate
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        if self.separate_cond:
            self.cross_attn2 = SparseMultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
            self.norm4 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)

            if self.separate_cond_gate == "alpha_residual":
                self.alpha_gate = nn.Sequential(
                    nn.Linear(1, 64),
                    nn.SiLU(),
                    nn.Linear(64, channels),
                )
            elif self.separate_cond_gate == "pair_channel":
                gate_in_dim = 1 + 3 * ctx_channels
                self.alpha_gate = nn.Sequential(
                    nn.Linear(gate_in_dim, channels),
                    nn.SiLU(),
                    nn.Linear(channels, channels),
                )
            elif self.separate_cond_gate == "token":
                gate_in_dim = 3 * channels + 1
                hidden_dim = max(256, channels // 2)
                self.alpha_gate = nn.Sequential(
                    nn.LayerNorm(gate_in_dim),
                    nn.Linear(gate_in_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, channels),
                )
            else:
                raise ValueError(f"Unknown separate_cond_gate: {self.separate_cond_gate}")

            nn.init.zeros_(self.alpha_gate[-1].weight)
            nn.init.zeros_(self.alpha_gate[-1].bias)

        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    @staticmethod
    def _endpoint_preserving_gate(alpha_src1: torch.Tensor, gate_delta: torch.Tensor) -> torch.Tensor:
        alpha_src2 = 1.0 - alpha_src1
        return alpha_src2 + alpha_src1 * alpha_src2 * torch.tanh(gate_delta)

    @staticmethod
    def _sparse_alpha_rows(x: SparseTensor, alpha: torch.Tensor) -> torch.Tensor:
        alpha_rows = torch.empty(
            x.feats.shape[0],
            1,
            device=x.device,
            dtype=x.dtype,
        )
        alpha = alpha.reshape(x.shape[0], 1).to(device=x.device, dtype=x.dtype)
        for batch_idx in range(x.shape[0]):
            alpha_rows[x.layout[batch_idx]] = alpha[batch_idx]
        return alpha_rows

    def _mix_separate_conditions(
        self,
        x: SparseTensor,
        context1: torch.Tensor,
        context2: torch.Tensor,
        alpha: torch.Tensor,
    ) -> SparseTensor:
        h = x.replace(self.norm2(x.feats))
        h1 = self.cross_attn(h, context1)

        h = x.replace(self.norm4(x.feats))
        h2 = self.cross_attn2(h, context2)

        batch_size = x.shape[0]
        alpha_batch = alpha.reshape(batch_size, 1).to(device=x.device, dtype=h1.dtype)

        if self.separate_cond_gate == "alpha_residual":
            gate_delta = self.alpha_gate(alpha_batch)
            gate = self._endpoint_preserving_gate(alpha_batch, gate_delta)
            return h1 * (1.0 - gate) + h2 * gate

        if self.separate_cond_gate == "pair_channel":
            summary1 = context1.mean(dim=1)
            summary2 = context2.mean(dim=1)
            summary_delta = summary2 - summary1
            gate_input = torch.cat([alpha_batch, summary1, summary2, summary_delta], dim=-1)
            gate_delta = self.alpha_gate(gate_input)
            gate = self._endpoint_preserving_gate(alpha_batch, gate_delta)
            return h1 * (1.0 - gate) + h2 * gate

        if self.separate_cond_gate == "token":
            alpha_rows = self._sparse_alpha_rows(h1, alpha)
            gate_input = torch.cat(
                [h1.feats, h2.feats, h2.feats - h1.feats, alpha_rows],
                dim=-1,
            )
            gate_delta = self.alpha_gate(gate_input)
            gate = self._endpoint_preserving_gate(alpha_rows, gate_delta)
            feats = (1.0 - gate) * h1.feats + gate * h2.feats
            return h1.replace(feats)

        raise ValueError(f"Unknown separate_cond_gate: {self.separate_cond_gate}")

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h

        if self.separate_cond:
            context1, context2, alpha = context
            h = self._mix_separate_conditions(x, context1, context2, alpha)
        else:
            h = x.replace(self.norm2(x.feats))
            h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, context) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)
