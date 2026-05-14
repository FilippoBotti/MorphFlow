from typing import *
import torch
import torch.nn as nn
from ..attention import MultiHeadAttention
from ..norm import LayerNorm32
from .blocks import FeedForwardNet


class ModulatedTransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
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
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
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
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        if self.separate_cond:
            self.cross_attn2 = MultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )

            # Separate LayerNorm for the second cross-attention branch.
            # It is initialized/copy-loaded from norm2 in train.py.
            self.norm4 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
            if self.separate_cond_gate == "alpha_residual":
                self.alpha_gate = nn.Sequential(
                    nn.Linear(1, 64),
                    nn.SiLU(),
                    nn.Linear(64, channels),
                )

            elif self.separate_cond_gate == "pair_channel":
                # Input: alpha + mean(cond1) + mean(cond2) + mean(cond2-cond1)
                gate_in_dim = 1 + 3 * ctx_channels
                self.alpha_gate = nn.Sequential(
                    nn.Linear(gate_in_dim, channels),
                    nn.SiLU(),
                    nn.Linear(channels, channels),
                )

            elif self.separate_cond_gate == "token":
                # Input per token: h1 + h2 + (h2-h1) + alpha
                # Output: gate [B, N, C]
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

        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _endpoint_preserving_gate(self, alpha_base, gate_delta):
        """
        alpha_base:
        [B, 1, 1] oppure [B, N, 1]

        gate_delta:
        [B, 1, C] oppure [B, N, C]

        Output:
        gate = alpha + alpha * (1-alpha) * tanh(delta)

        Garantisce:
        alpha=0 -> gate=0
        alpha=1 -> gate=1
        """
        return alpha_base + alpha_base * (1.0 - alpha_base) * torch.tanh(gate_delta)

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        
        if self.separate_cond:
            context1, context2, alpha = context
            h = self.norm2(x)
            h1 = self.cross_attn(h, context1)
            
            h = self.norm4(x)
            h2 = self.cross_attn2(h, context2)
            
            B, N, C = h1.shape
            alpha_base = alpha.reshape(B, 1, 1).to(device=h1.device, dtype=h1.dtype)

            if self.separate_cond_gate == "alpha_residual":
                # Gate [B, 1, C], dipende solo da alpha.
                gate_delta = self.alpha_gate(alpha_base)
                gate = self._endpoint_preserving_gate(alpha_base, gate_delta)

            elif self.separate_cond_gate == "pair_channel":
                # Gate [B, 1, C], dipende da alpha e dal contenuto globale della coppia.
                summary1 = context1.mean(dim=1)
                summary2 = context2.mean(dim=1)
                summary_delta = summary2 - summary1

                gate_input = torch.cat(
                    [
                        alpha.reshape(B, 1).to(device=h1.device, dtype=h1.dtype),
                        summary1,
                        summary2,
                        summary_delta,
                    ],
                    dim=-1,
                )

                gate_delta = self.alpha_gate(gate_input).unsqueeze(1)
                gate = self._endpoint_preserving_gate(alpha_base, gate_delta)

            elif self.separate_cond_gate == "token":
                # Gate [B, N, C], dipende da ogni token.
                alpha_token = alpha_base.expand(B, N, 1)

                gate_input = torch.cat(
                    [
                        h1,
                        h2,
                        h2 - h1,
                        alpha_token,
                    ],
                    dim=-1,
                )

                gate_delta = self.alpha_gate(gate_input)
                gate = self._endpoint_preserving_gate(alpha_token, gate_delta)

            else:
                raise ValueError(f"Unknown separate_cond_gate: {self.separate_cond_gate}")

            h = (1.0 - gate) * h1 + gate * h2
        else:
            h = self.norm2(x)
            h = self.cross_attn(h, context)
            
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)
        