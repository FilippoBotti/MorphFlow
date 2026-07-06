#!/usr/bin/env python3
"""
Apply semantic token matching + optional cycle consistency to MorphFlow.
Run from the MorphFlow repository root:

    python3 apply_semantic_matching.py

The script is intentionally idempotent enough for one clean application and avoids
using git patch hunks, so it is safer than the malformed .patch file.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path.cwd()

SEMANTIC_TOKEN_MATCHING = r'''from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class SemanticTokenMatcher(nn.Module):
    """
    Differentiable bidirectional soft matching between two source-condition token
    sequences.

    The module preserves the original token axis of both condition streams:
      - cond1_out[i] is cond1[i] mixed with the src2 token distribution matched
        to cond1[i];
      - cond2_out[j] is cond2[j] mixed with the src1 token distribution matched
        to cond2[j].

    This is safe before either PairConditionFusionV2 (single-condition path) or
    separate-condition projection/gating. Optional tail tokens, for example
    global style tokens appended by the sparse-conv condition encoder, can be
    excluded from matching and copied unchanged.
    """

    def __init__(
        self,
        dim: int = 128,
        match_dim: int = 128,
        temperature: float = 0.1,
        max_align: float = 0.25,
        alpha_weight: bool = True,
        detach_scores: bool = False,
        style_tokens: int = 0,
        cycle_detach_targets: bool = True,
        cycle_alpha_weight: bool = True,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if match_dim <= 0:
            raise ValueError(f"match_dim must be > 0, got {match_dim}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if max_align < 0.0:
            raise ValueError(f"max_align must be >= 0, got {max_align}")
        if style_tokens < 0:
            raise ValueError(f"style_tokens must be >= 0, got {style_tokens}")

        self.dim = int(dim)
        self.match_dim = int(match_dim)
        self.temperature = float(temperature)
        self.max_align = float(max_align)
        self.alpha_weight = bool(alpha_weight)
        self.detach_scores = bool(detach_scores)
        self.style_tokens = int(style_tokens)
        self.cycle_detach_targets = bool(cycle_detach_targets)
        self.cycle_alpha_weight = bool(cycle_alpha_weight)

        self.q_proj = nn.Linear(self.dim, self.match_dim)
        self.k_proj = nn.Linear(self.dim, self.match_dim)
        self.out_norm = nn.Identity()

    def _split_tail_tokens(
        self,
        cond1: torch.Tensor,
        cond2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        tail = min(self.style_tokens, cond1.shape[1], cond2.shape[1])
        if tail <= 0:
            return cond1, None, cond2, None
        if tail == cond1.shape[1] or tail == cond2.shape[1]:
            # Degenerate but safe: there are no spatial tokens to match.
            return cond1[:, :0], cond1, cond2[:, :0], cond2
        return cond1[:, :-tail], cond1[:, -tail:], cond2[:, :-tail], cond2[:, -tail:]

    def _alpha_lambda(self, alpha: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        alpha = alpha.reshape(-1, 1, 1).to(device=cond.device, dtype=cond.dtype)
        if self.alpha_weight:
            # Endpoint-preserving profile: 0 at alpha=0/1, 1 at alpha=0.5.
            alpha_profile = 4.0 * alpha * (1.0 - alpha)
        else:
            alpha_profile = torch.ones_like(alpha)
        return (self.max_align * alpha_profile).clamp(min=0.0, max=self.max_align)

    def _affinity(self, cond1: torch.Tensor, cond2: torch.Tensor) -> torch.Tensor:
        q_input = cond1.detach() if self.detach_scores else cond1
        k_input = cond2.detach() if self.detach_scores else cond2
        q = self.q_proj(q_input)
        k = self.k_proj(k_input)
        q = F.normalize(q.float(), dim=-1)
        k = F.normalize(k.float(), dim=-1)
        return torch.matmul(q, k.transpose(-1, -2)) / self.temperature

    @staticmethod
    def _attention_entropy(attn: torch.Tensor) -> torch.Tensor:
        return -(attn * attn.clamp_min(1e-8).log()).sum(dim=-1).mean()

    @staticmethod
    def _attention_usage_stats(attn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Mean mass received by each target token. Multiplying by token count
        # makes the uniform value close to 1.0, which is easier to monitor.
        usage = attn.mean(dim=1) * float(attn.shape[-1])
        return usage.max(dim=-1).values.mean(), usage.min(dim=-1).values.mean()

    def _cycle_loss(
        self,
        cond1: torch.Tensor,
        cond2: torch.Tensor,
        a12: torch.Tensor,
        a21: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        cond1_f = cond1.float()
        cond2_f = cond2.float()

        cond1_to_2 = torch.matmul(a21, cond1_f)
        cond1_cycle = torch.matmul(a12, cond1_to_2)

        cond2_to_1 = torch.matmul(a12, cond2_f)
        cond2_cycle = torch.matmul(a21, cond2_to_1)

        target1 = cond1_f.detach() if self.cycle_detach_targets else cond1_f
        target2 = cond2_f.detach() if self.cycle_detach_targets else cond2_f

        per_item = (
            (cond1_cycle - target1).pow(2).mean(dim=(1, 2))
            + (cond2_cycle - target2).pow(2).mean(dim=(1, 2))
        )
        if self.cycle_alpha_weight:
            alpha_flat = alpha.reshape(-1).to(device=per_item.device, dtype=per_item.dtype)
            weights = 4.0 * alpha_flat * (1.0 - alpha_flat)
            per_item = per_item * weights.clamp(min=0.0)
        return per_item.mean()

    def forward(
        self,
        cond1: torch.Tensor,
        cond2: torch.Tensor,
        alpha: torch.Tensor,
        return_aux: bool = False,
        compute_cycle: bool = False,
    ):
        if cond1.ndim != 3 or cond2.ndim != 3:
            raise ValueError(
                "SemanticTokenMatcher expects cond1/cond2 with shape [B, tokens, C], "
                f"got {tuple(cond1.shape)} and {tuple(cond2.shape)}"
            )
        if cond1.shape[0] != cond2.shape[0] or cond1.shape[-1] != cond2.shape[-1]:
            raise ValueError(
                "cond1/cond2 must share batch and channel dimensions, "
                f"got {tuple(cond1.shape)} and {tuple(cond2.shape)}"
            )
        if cond1.shape[-1] != self.dim:
            raise ValueError(f"expected token dim={self.dim}, got {cond1.shape[-1]}")

        spatial1, tail1, spatial2, tail2 = self._split_tail_tokens(cond1, cond2)
        if spatial1.shape[1] == 0 or spatial2.shape[1] == 0 or self.max_align == 0.0:
            aux = {
                "cycle_loss": cond1.new_zeros(()),
                "metrics": {
                    "semantic_align_lambda": cond1.new_zeros(()),
                    "semantic_entropy_12": cond1.new_zeros(()),
                    "semantic_entropy_21": cond1.new_zeros(()),
                    "semantic_usage_12_max": cond1.new_zeros(()),
                    "semantic_usage_12_min": cond1.new_zeros(()),
                    "semantic_usage_21_max": cond1.new_zeros(()),
                    "semantic_usage_21_min": cond1.new_zeros(()),
                },
            }
            return (cond1, cond2, aux) if return_aux else (cond1, cond2)

        sim12 = self._affinity(spatial1, spatial2)
        a12 = torch.softmax(sim12, dim=-1)
        a21 = torch.softmax(sim12.transpose(-1, -2), dim=-1)

        spatial2_to_1 = torch.matmul(a12.to(dtype=spatial2.dtype), spatial2)
        spatial1_to_2 = torch.matmul(a21.to(dtype=spatial1.dtype), spatial1)

        lam1 = self._alpha_lambda(alpha, spatial1)
        lam2 = self._alpha_lambda(alpha, spatial2)
        spatial1_out = (1.0 - lam1) * spatial1 + lam1 * spatial2_to_1
        spatial2_out = (1.0 - lam2) * spatial2 + lam2 * spatial1_to_2

        cond1_out = spatial1_out if tail1 is None else torch.cat([spatial1_out, tail1], dim=1)
        cond2_out = spatial2_out if tail2 is None else torch.cat([spatial2_out, tail2], dim=1)
        cond1_out = self.out_norm(cond1_out)
        cond2_out = self.out_norm(cond2_out)

        if not return_aux:
            return cond1_out, cond2_out

        with torch.no_grad():
            usage12_max, usage12_min = self._attention_usage_stats(a12)
            usage21_max, usage21_min = self._attention_usage_stats(a21)
            metrics = {
                "semantic_align_lambda": torch.cat([lam1, lam2], dim=1).mean().detach(),
                "semantic_entropy_12": self._attention_entropy(a12).detach(),
                "semantic_entropy_21": self._attention_entropy(a21).detach(),
                "semantic_usage_12_max": usage12_max.detach(),
                "semantic_usage_12_min": usage12_min.detach(),
                "semantic_usage_21_max": usage21_max.detach(),
                "semantic_usage_21_min": usage21_min.detach(),
            }

        cycle_loss = self._cycle_loss(spatial1, spatial2, a12, a21, alpha) if compute_cycle else cond1.new_zeros(())
        aux = {"cycle_loss": cycle_loss, "metrics": metrics}
        return cond1_out, cond2_out, aux


class SemanticTokenMatchingMixin:
    """Shared wiring for MorphFlow and MorphSLatFlow."""

    def _init_semantic_token_matching(
        self,
        *,
        use_semantic_token_matching: bool = False,
        semantic_match_dim: int = 128,
        semantic_match_temperature: float = 0.1,
        semantic_match_max_align: float = 0.25,
        semantic_match_alpha_weight: bool = True,
        semantic_match_detach_scores: bool = False,
        semantic_match_style_tokens: int = 0,
        semantic_cycle_loss_weight: float = 0.0,
        semantic_cycle_loss_prob: float = 1.0,
        semantic_cycle_detach_targets: bool = True,
        semantic_cycle_alpha_weight: bool = True,
        semantic_match_log_stats: bool = True,
    ) -> None:
        if semantic_cycle_loss_weight < 0.0:
            raise ValueError(f"semantic_cycle_loss_weight must be >= 0, got {semantic_cycle_loss_weight}")
        if semantic_cycle_loss_prob < 0.0 or semantic_cycle_loss_prob > 1.0:
            raise ValueError(f"semantic_cycle_loss_prob must be in [0, 1], got {semantic_cycle_loss_prob}")
        if semantic_cycle_loss_weight > 0.0 and not use_semantic_token_matching:
            raise ValueError("semantic_cycle_loss_weight > 0 requires use_semantic_token_matching=True")

        self.use_semantic_token_matching = bool(use_semantic_token_matching)
        self.semantic_cycle_loss_weight = float(semantic_cycle_loss_weight)
        self.semantic_cycle_loss_prob = float(semantic_cycle_loss_prob)
        self.semantic_match_log_stats = bool(semantic_match_log_stats)
        self._semantic_match_record_aux = False
        self._semantic_match_cycle_terms = []
        self._semantic_match_last_metrics: Dict[str, torch.Tensor] = {}

        if self.use_semantic_token_matching:
            self.semantic_matcher = SemanticTokenMatcher(
                dim=128,
                match_dim=semantic_match_dim,
                temperature=semantic_match_temperature,
                max_align=semantic_match_max_align,
                alpha_weight=semantic_match_alpha_weight,
                detach_scores=semantic_match_detach_scores,
                style_tokens=semantic_match_style_tokens,
                cycle_detach_targets=semantic_cycle_detach_targets,
                cycle_alpha_weight=semantic_cycle_alpha_weight,
            )

    def _begin_semantic_match_record(self, device: torch.device) -> None:
        self._semantic_match_cycle_terms = []
        self._semantic_match_last_metrics = {}
        self._semantic_match_record_aux = False
        if (
            self.training
            and getattr(self, "use_semantic_token_matching", False)
            and self.semantic_cycle_loss_weight > 0.0
            and self.semantic_cycle_loss_prob > 0.0
        ):
            self._semantic_match_record_aux = bool(
                torch.rand((), device=device).item() < self.semantic_cycle_loss_prob
            )

    def _apply_semantic_token_matching(
        self,
        cond1: torch.Tensor,
        cond2: torch.Tensor,
        alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not getattr(self, "use_semantic_token_matching", False):
            return cond1, cond2

        return_aux = bool(self._semantic_match_record_aux or self.semantic_match_log_stats)
        if return_aux:
            cond1, cond2, aux = self.semantic_matcher(
                cond1,
                cond2,
                alpha,
                return_aux=True,
                compute_cycle=self._semantic_match_record_aux,
            )
            self._semantic_match_last_metrics = aux["metrics"]
            if self._semantic_match_record_aux:
                self._semantic_match_cycle_terms.append(aux["cycle_loss"])
            return cond1, cond2

        return self.semantic_matcher(cond1, cond2, alpha, return_aux=False)

    def _semantic_match_aux_loss(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if not self._semantic_match_cycle_terms:
            self._semantic_match_record_aux = False
            return torch.zeros((), device=device, dtype=dtype)
        raw = torch.stack([term.to(device=device, dtype=torch.float32) for term in self._semantic_match_cycle_terms]).mean()
        weighted = raw * self.semantic_cycle_loss_weight
        self._semantic_match_last_metrics = {
            **self._semantic_match_last_metrics,
            "semantic_cycle_loss": raw.detach(),
            "semantic_cycle_loss_weighted": weighted.detach(),
            "semantic_cycle_active": torch.ones((), device=device, dtype=torch.float32),
        }
        self._semantic_match_record_aux = False
        return weighted.to(dtype=dtype)

    def _semantic_match_metrics(self) -> Dict[str, torch.Tensor]:
        if not getattr(self, "use_semantic_token_matching", False):
            return {}
        metrics = dict(getattr(self, "_semantic_match_last_metrics", {}))
        if "semantic_cycle_active" not in metrics:
            device = next(self.parameters()).device
            metrics["semantic_cycle_active"] = torch.zeros((), device=device, dtype=torch.float32)
        return metrics
'''


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def read(rel: str) -> str:
    p = ROOT / rel
    if not p.is_file():
        die(f"missing file: {rel}. Run this script from the MorphFlow repo root.")
    return p.read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print(f"updated: {rel}")


def replace_once(text: str, old: str, new: str, rel: str, label: str) -> str:
    if new in text:
        print(f"skip already applied: {rel} :: {label}")
        return text
    count = text.count(old)
    if count != 1:
        die(f"expected one match for {rel} :: {label}, found {count}")
    return text.replace(old, new, 1)


def insert_after_once(text: str, marker: str, addition: str, rel: str, label: str) -> str:
    if addition.strip() in text:
        print(f"skip already applied: {rel} :: {label}")
        return text
    count = text.count(marker)
    if count != 1:
        die(f"expected one marker for {rel} :: {label}, found {count}")
    return text.replace(marker, marker + addition, 1)


def patch_morph_flow() -> None:
    rel = "models/morph_flow.py"
    text = read(rel)
    text = insert_after_once(
        text,
        "from models import cond_encoder\n",
        "from models.semantic_token_matching import SemanticTokenMatchingMixin\n",
        rel,
        "import mixin",
    )
    text = replace_once(text, "class MorphFlow(nn.Module):", "class MorphFlow(SemanticTokenMatchingMixin, nn.Module):", rel, "class mixin")

    old_args = '''        cond_residual_blocks_16=0,
        t_schedule="logit_normal",
        t_logit_mean=0.0,
        t_logit_std=1.0,
    ):'''
    new_args = '''        cond_residual_blocks_16=0,
        t_schedule="logit_normal",
        t_logit_mean=0.0,
        t_logit_std=1.0,
        use_semantic_token_matching=False,
        semantic_match_dim=128,
        semantic_match_temperature=0.1,
        semantic_match_max_align=0.25,
        semantic_match_alpha_weight=True,
        semantic_match_detach_scores=False,
        semantic_match_exclude_style_tokens=True,
        semantic_cycle_loss_weight=0.0,
        semantic_cycle_loss_prob=1.0,
        semantic_cycle_detach_targets=True,
        semantic_cycle_alpha_weight=True,
        semantic_match_log_stats=True,
    ):'''
    text = replace_once(text, old_args, new_args, rel, "constructor args")

    old_init = '''        if self.separate_cond:
            self.separate_cond_proj = nn.Linear(128, model_channels)
            if self.cond_proj_norm == "layernorm":
                self.cond_proj_layer_norm = nn.LayerNorm(model_channels)

        self.cfg_drop_prob = 0.0'''
    new_init = '''        if self.separate_cond:
            self.separate_cond_proj = nn.Linear(128, model_channels)
            if self.cond_proj_norm == "layernorm":
                self.cond_proj_layer_norm = nn.LayerNorm(model_channels)

        semantic_style_tokens = 0
        if semantic_match_exclude_style_tokens and not hasattr(self, "cond_resampler"):
            semantic_style_tokens = int(getattr(self.cond_encoder, "style_tokens", 0))
        self._init_semantic_token_matching(
            use_semantic_token_matching=use_semantic_token_matching,
            semantic_match_dim=semantic_match_dim,
            semantic_match_temperature=semantic_match_temperature,
            semantic_match_max_align=semantic_match_max_align,
            semantic_match_alpha_weight=semantic_match_alpha_weight,
            semantic_match_detach_scores=semantic_match_detach_scores,
            semantic_match_style_tokens=semantic_style_tokens,
            semantic_cycle_loss_weight=semantic_cycle_loss_weight,
            semantic_cycle_loss_prob=semantic_cycle_loss_prob,
            semantic_cycle_detach_targets=semantic_cycle_detach_targets,
            semantic_cycle_alpha_weight=semantic_cycle_alpha_weight,
            semantic_match_log_stats=semantic_match_log_stats,
        )

        self.cfg_drop_prob = 0.0'''
    text = replace_once(text, old_init, new_init, rel, "semantic init")

    build_condition = '''
    def _build_condition(
        self,
        src_1_feats,
        src_2_feats,
        src_1_coords,
        src_2_coords,
        alpha,
        apply_cfg_drop=True,
    ):
        src_1_feats = self.normalize_condition_feats(src_1_feats)
        src_2_feats = self.normalize_condition_feats(src_2_feats)
        cond1 = self.encode_condition_tokens(src_1_feats, src_1_coords)
        cond2 = self.encode_condition_tokens(src_2_feats, src_2_coords)
        cond1, cond2 = self.normalize_condition_tokens(cond1, cond2, alpha)
        cond1, cond2 = self._apply_semantic_token_matching(cond1, cond2, alpha)

        if not self.separate_cond:
            cond = self.cond_fusion(cond1, cond2, alpha)
        else:
            cond1 = self.separate_cond_proj(cond1)
            cond2 = self.separate_cond_proj(cond2)
            cond1, cond2 = self.normalize_projected_condition_tokens(cond1, cond2)
            cond = (cond1, cond2, alpha)

        if apply_cfg_drop and self.training and self.cfg_drop_prob > 0.0:
            B = cond1.shape[0] if self.separate_cond else cond.shape[0]
            drop_mask = torch.rand(B, device=cond1.device) < self.cfg_drop_prob
            null_cond = self.null_cond.expand(B, -1, -1).to(dtype=cond1.dtype)

            if not self.separate_cond:
                cond = torch.where(
                    drop_mask.view(B, 1, 1),
                    null_cond,
                    cond,
                )
            else:
                drop_mask = drop_mask.view(B, 1, 1)
                cond = (
                    torch.where(drop_mask, null_cond, cond1),
                    torch.where(drop_mask, null_cond, cond2),
                    alpha,
                )

        return cond
'''
    marker = '''    def get_v(self, x_0, noise):
'''
    text = replace_once(text, marker, build_condition + "\n" + marker, rel, "_build_condition")

    old_forward_flow_block = '''        # condition
        src_1_feats = self.normalize_condition_feats(src_1_feats)
        src_2_feats = self.normalize_condition_feats(src_2_feats)
        cond1 = self.encode_condition_tokens(src_1_feats, src_1_coords)
        cond2 = self.encode_condition_tokens(src_2_feats, src_2_coords)
        cond1, cond2 = self.normalize_condition_tokens(cond1, cond2, alpha)
        
        if not self.separate_cond:
            cond = self.cond_fusion(cond1, cond2, alpha)
        else:
            cond1 = self.separate_cond_proj(cond1)
            cond2 = self.separate_cond_proj(cond2)
            cond1, cond2 = self.normalize_projected_condition_tokens(cond1, cond2)
            cond = (cond1, cond2, alpha)

        if apply_cfg_drop and self.training and self.cfg_drop_prob > 0.0:
            B = cond1.shape[0] if self.separate_cond else cond.shape[0]
            drop_mask = torch.rand(B, device=cond1.device) < self.cfg_drop_prob
            null_cond = self.null_cond.expand(B, -1, -1).to(dtype=cond1.dtype)

            if not self.separate_cond:
                cond = torch.where(
                    drop_mask.view(B, 1, 1),
                    null_cond,
                    cond,
                )
            else:
                drop_mask = drop_mask.view(B, 1, 1)
                cond = (
                    torch.where(drop_mask, null_cond, cond1),
                    torch.where(drop_mask, null_cond, cond2),
                    alpha
                )

        # diffusion'''
    new_forward_flow_block = '''        cond = self._build_condition(
            src_1_feats,
            src_2_feats,
            src_1_coords,
            src_2_coords,
            alpha,
            apply_cfg_drop=apply_cfg_drop,
        )

        # diffusion'''
    text = replace_once(text, old_forward_flow_block, new_forward_flow_block, rel, "forward_flow condition block")

    old_cfg_block = '''        src_1_feats = self.normalize_condition_feats(src_1_feats)
        src_2_feats = self.normalize_condition_feats(src_2_feats)
        cond1 = self.encode_condition_tokens(src_1_feats, src_1_coords)
        cond2 = self.encode_condition_tokens(src_2_feats, src_2_coords)
        cond1, cond2 = self.normalize_condition_tokens(cond1, cond2, alpha)
        
        if not self.separate_cond:
            cond = self.cond_fusion(cond1, cond2, alpha)
            B = cond.shape[0]
            null_cond = self.null_cond.expand(B, -1, -1).to(dtype=cond.dtype)
        else:
            cond1 = self.separate_cond_proj(cond1)
            cond2 = self.separate_cond_proj(cond2)
            cond1, cond2 = self.normalize_projected_condition_tokens(cond1, cond2)
            cond = (cond1, cond2, alpha)
            B = cond1.shape[0]
            null_cond_tensor = self.null_cond.expand(B, -1, -1).to(dtype=cond1.dtype)
            null_cond = (null_cond_tensor, null_cond_tensor, alpha)

        t_flow = t.float() * 1000.0'''
    new_cfg_block = '''        cond = self._build_condition(
            src_1_feats,
            src_2_feats,
            src_1_coords,
            src_2_coords,
            alpha,
            apply_cfg_drop=False,
        )
        if not self.separate_cond:
            B = cond.shape[0]
            null_cond = self.null_cond.expand(B, -1, -1).to(dtype=cond.dtype)
        else:
            cond1, cond2, alpha_cond = cond
            B = cond1.shape[0]
            null_cond_tensor = self.null_cond.expand(B, -1, -1).to(dtype=cond1.dtype)
            null_cond = (null_cond_tensor, null_cond_tensor, alpha_cond)

        t_flow = t.float() * 1000.0'''
    text = replace_once(text, old_cfg_block, new_cfg_block, rel, "cfg condition block")

    old_loss_part = '''        velocity = self.get_v(x_0, noise)

        pred = self.forward_flow('''
    new_loss_part = '''        velocity = self.get_v(x_0, noise)
        self._begin_semantic_match_record(x_0.device)

        pred = self.forward_flow('''
    text = replace_once(text, old_loss_part, new_loss_part, rel, "begin semantic record")

    old_loss_finish = '''        )
        loss = F.mse_loss(pred, velocity)

        if return_terms:'''
    new_loss_finish = '''        )
        loss = F.mse_loss(pred, velocity)
        semantic_aux = self._semantic_match_aux_loss(x_0.device, loss.dtype)
        loss = loss + semantic_aux
        self.last_forward_metrics = self._semantic_match_metrics()

        if return_terms:'''
    text = replace_once(text, old_loss_finish, new_loss_finish, rel, "aux loss")
    write(rel, text)


def patch_morph_slat_flow() -> None:
    rel = "models/morph_slat_flow.py"
    text = read(rel)
    text = insert_after_once(text, "from models import cond_encoder\nfrom models import structured_latent_flow\n", "from models.semantic_token_matching import SemanticTokenMatchingMixin\n", rel, "import mixin")
    text = replace_once(text, "class MorphSLatFlow(nn.Module):", "class MorphSLatFlow(SemanticTokenMatchingMixin, nn.Module):", rel, "class mixin")

    old_args = '''        t_schedule: str = "logit_normal",
        t_logit_mean: float = 0.0,
        t_logit_std: float = 1.0,
    ):'''
    new_args = '''        t_schedule: str = "logit_normal",
        t_logit_mean: float = 0.0,
        t_logit_std: float = 1.0,
        use_semantic_token_matching: bool = False,
        semantic_match_dim: int = 128,
        semantic_match_temperature: float = 0.1,
        semantic_match_max_align: float = 0.25,
        semantic_match_alpha_weight: bool = True,
        semantic_match_detach_scores: bool = False,
        semantic_match_exclude_style_tokens: bool = True,
        semantic_cycle_loss_weight: float = 0.0,
        semantic_cycle_loss_prob: float = 1.0,
        semantic_cycle_detach_targets: bool = True,
        semantic_cycle_alpha_weight: bool = True,
        semantic_match_log_stats: bool = True,
    ):'''
    text = replace_once(text, old_args, new_args, rel, "constructor args")

    old_init = '''        if self.separate_cond:
            self.separate_cond_proj = nn.Linear(128, model_channels)
            if self.cond_proj_norm == "layernorm":
                self.cond_proj_layer_norm = nn.LayerNorm(model_channels)

        self.cfg_drop_prob = 0.0'''
    new_init = '''        if self.separate_cond:
            self.separate_cond_proj = nn.Linear(128, model_channels)
            if self.cond_proj_norm == "layernorm":
                self.cond_proj_layer_norm = nn.LayerNorm(model_channels)

        semantic_style_tokens = 0
        if semantic_match_exclude_style_tokens and not hasattr(self, "cond_resampler"):
            semantic_style_tokens = int(getattr(self.cond_encoder, "style_tokens", 0))
        self._init_semantic_token_matching(
            use_semantic_token_matching=use_semantic_token_matching,
            semantic_match_dim=semantic_match_dim,
            semantic_match_temperature=semantic_match_temperature,
            semantic_match_max_align=semantic_match_max_align,
            semantic_match_alpha_weight=semantic_match_alpha_weight,
            semantic_match_detach_scores=semantic_match_detach_scores,
            semantic_match_style_tokens=semantic_style_tokens,
            semantic_cycle_loss_weight=semantic_cycle_loss_weight,
            semantic_cycle_loss_prob=semantic_cycle_loss_prob,
            semantic_cycle_detach_targets=semantic_cycle_detach_targets,
            semantic_cycle_alpha_weight=semantic_cycle_alpha_weight,
            semantic_match_log_stats=semantic_match_log_stats,
        )

        self.cfg_drop_prob = 0.0'''
    text = replace_once(text, old_init, new_init, rel, "semantic init")

    old_build = '''        cond1 = self.encode_condition_tokens(src_1_feats, src_1_coords)
        cond2 = self.encode_condition_tokens(src_2_feats, src_2_coords)
        cond1, cond2 = self.normalize_condition_tokens(cond1, cond2, alpha)

        if not self.separate_cond:'''
    new_build = '''        cond1 = self.encode_condition_tokens(src_1_feats, src_1_coords)
        cond2 = self.encode_condition_tokens(src_2_feats, src_2_coords)
        cond1, cond2 = self.normalize_condition_tokens(cond1, cond2, alpha)
        cond1, cond2 = self._apply_semantic_token_matching(cond1, cond2, alpha)

        if not self.separate_cond:'''
    text = replace_once(text, old_build, new_build, rel, "apply semantic in _build_condition")

    old_cfg_block = '''        src_1_feats = self.normalize_condition_feats(src_1_feats)
        src_2_feats = self.normalize_condition_feats(src_2_feats)
        cond1 = self.encode_condition_tokens(src_1_feats, src_1_coords)
        cond2 = self.encode_condition_tokens(src_2_feats, src_2_coords)
        cond1, cond2 = self.normalize_condition_tokens(cond1, cond2, alpha)

        if not self.separate_cond:
            cond = self.cond_fusion(cond1, cond2, alpha)
            batch_size = cond.shape[0]
            null_cond = self.null_cond.expand(batch_size, -1, -1).to(dtype=cond.dtype)
        else:
            cond1 = self.separate_cond_proj(cond1)
            cond2 = self.separate_cond_proj(cond2)
            cond1, cond2 = self.normalize_projected_condition_tokens(cond1, cond2)
            cond = (cond1, cond2, alpha)
            batch_size = cond1.shape[0]
            null_tensor = self.null_cond.expand(batch_size, -1, -1).to(dtype=cond1.dtype)
            null_cond = (null_tensor, null_tensor, alpha)

        t_flow = t.float() * 1000.0'''
    new_cfg_block = '''        cond = self._build_condition(
            src_1_feats,
            src_1_coords,
            src_2_feats,
            src_2_coords,
            alpha,
        )
        if not self.separate_cond:
            batch_size = cond.shape[0]
            null_cond = self.null_cond.expand(batch_size, -1, -1).to(dtype=cond.dtype)
        else:
            cond1, cond2, alpha_cond = cond
            batch_size = cond1.shape[0]
            null_tensor = self.null_cond.expand(batch_size, -1, -1).to(dtype=cond1.dtype)
            null_cond = (null_tensor, null_tensor, alpha_cond)

        t_flow = t.float() * 1000.0'''
    text = replace_once(text, old_cfg_block, new_cfg_block, rel, "cfg condition block")

    old_forward_begin = '''        batch_size = x_0.shape[0]
        t = self.sample_t(batch_size, x_0.device)
        x_t, noise = self.diffuse(x_0, t)
        velocity = self.get_v(x_0, noise)

        pred = self.forward_flow('''
    new_forward_begin = '''        batch_size = x_0.shape[0]
        t = self.sample_t(batch_size, x_0.device)
        x_t, noise = self.diffuse(x_0, t)
        velocity = self.get_v(x_0, noise)

        self._begin_semantic_match_record(x_0.device)
        pred = self.forward_flow('''
    text = replace_once(text, old_forward_begin, new_forward_begin, rel, "begin record")

    old_forward_loss = '''        )

        loss = self.batch_mean_mse(pred, velocity)
        self._update_forward_metrics(pred, velocity, loss)
        return loss'''
    new_forward_loss = '''        )

        base_loss = self.batch_mean_mse(pred, velocity)
        semantic_aux = self._semantic_match_aux_loss(x_0.device, base_loss.dtype)
        loss = base_loss + semantic_aux
        self._update_forward_metrics(pred, velocity, loss)
        self.last_forward_metrics.update(self._semantic_match_metrics())
        return loss'''
    text = replace_once(text, old_forward_loss, new_forward_loss, rel, "aux loss")
    write(rel, text)


def patch_morph_residual_flow() -> None:
    rel = "models/morph_residual_flow.py"
    text = read(rel)
    old_args = '''        residual_endpoint_max_items=1,
        t_schedule="logit_normal",
        t_logit_mean=0.0,
        t_logit_std=1.0,
    ):'''
    new_args = '''        residual_endpoint_max_items=1,
        t_schedule="logit_normal",
        t_logit_mean=0.0,
        t_logit_std=1.0,
        use_semantic_token_matching=False,
        semantic_match_dim=128,
        semantic_match_temperature=0.1,
        semantic_match_max_align=0.25,
        semantic_match_alpha_weight=True,
        semantic_match_detach_scores=False,
        semantic_match_exclude_style_tokens=True,
        semantic_cycle_loss_weight=0.0,
        semantic_cycle_loss_prob=1.0,
        semantic_cycle_detach_targets=True,
        semantic_cycle_alpha_weight=True,
        semantic_match_log_stats=True,
    ):'''
    text = replace_once(text, old_args, new_args, rel, "constructor args")

    old_super = '''            t_schedule=t_schedule,
            t_logit_mean=t_logit_mean,
            t_logit_std=t_logit_std,
        )'''
    new_super = '''            t_schedule=t_schedule,
            t_logit_mean=t_logit_mean,
            t_logit_std=t_logit_std,
            use_semantic_token_matching=use_semantic_token_matching,
            semantic_match_dim=semantic_match_dim,
            semantic_match_temperature=semantic_match_temperature,
            semantic_match_max_align=semantic_match_max_align,
            semantic_match_alpha_weight=semantic_match_alpha_weight,
            semantic_match_detach_scores=semantic_match_detach_scores,
            semantic_match_exclude_style_tokens=semantic_match_exclude_style_tokens,
            semantic_cycle_loss_weight=semantic_cycle_loss_weight,
            semantic_cycle_loss_prob=semantic_cycle_loss_prob,
            semantic_cycle_detach_targets=semantic_cycle_detach_targets,
            semantic_cycle_alpha_weight=semantic_cycle_alpha_weight,
            semantic_match_log_stats=semantic_match_log_stats,
        )'''
    text = replace_once(text, old_super, new_super, rel, "super kwargs")

    old_begin = '''        velocity = self.get_v(x_0, noise)
        pred = self.forward_flow('''
    new_begin = '''        velocity = self.get_v(x_0, noise)
        self._begin_semantic_match_record(x_0.device)
        pred = self.forward_flow('''
    text = replace_once(text, old_begin, new_begin, rel, "begin record")

    old_loss = '''        )
        loss = F.mse_loss(pred, velocity)

        if return_terms:'''
    new_loss = '''        )
        loss = F.mse_loss(pred, velocity)
        semantic_aux = self._semantic_match_aux_loss(x_0.device, loss.dtype)
        loss = loss + semantic_aux
        self.last_forward_metrics = self._semantic_match_metrics()

        if return_terms:'''
    text = replace_once(text, old_loss, new_loss, rel, "aux loss")
    write(rel, text)


def patch_train() -> None:
    rel = "train.py"
    text = read(rel)
    parser_marker = '''    parser.add_argument("--dino_layer_norm", type=int, choices=[0, 1], default=1)

    # Optional future losses.'''
    parser_add = '''    parser.add_argument("--dino_layer_norm", type=int, choices=[0, 1], default=1)

    # Semantic token matching between the two source-SLat condition streams.
    parser.add_argument("--use_semantic_token_matching", type=int, choices=[0, 1], default=0)
    parser.add_argument("--semantic_match_dim", type=int, default=128)
    parser.add_argument("--semantic_match_temperature", type=float, default=0.1)
    parser.add_argument("--semantic_match_max_align", type=float, default=0.25)
    parser.add_argument("--semantic_match_alpha_weight", type=int, choices=[0, 1], default=1, help="Use 4*alpha*(1-alpha) so matching vanishes at endpoints.")
    parser.add_argument("--semantic_match_detach_scores", type=int, choices=[0, 1], default=0, help="Detach condition tokens before q/k projections used for matching scores.")
    parser.add_argument("--semantic_match_exclude_style_tokens", type=int, choices=[0, 1], default=1, help="Do not match sparse_conv3d global style tokens appended at the token tail.")
    parser.add_argument("--semantic_cycle_loss_weight", type=float, default=0.0)
    parser.add_argument("--semantic_cycle_loss_prob", type=float, default=1.0)
    parser.add_argument("--semantic_cycle_detach_targets", type=int, choices=[0, 1], default=1)
    parser.add_argument("--semantic_cycle_alpha_weight", type=int, choices=[0, 1], default=1)
    parser.add_argument("--semantic_match_log_stats", type=int, choices=[0, 1], default=1)

    # Optional future losses.'''
    text = replace_once(text, parser_marker, parser_add, rel, "parser args")

    build_marker = '''        "dino_model": args.dino_model,
        "dino_dim": args.dino_dim,
        "dino_layer_norm": args.dino_layer_norm == 1,
    }'''
    build_add = '''        "dino_model": args.dino_model,
        "dino_dim": args.dino_dim,
        "dino_layer_norm": args.dino_layer_norm == 1,
        "use_semantic_token_matching": args.use_semantic_token_matching == 1,
        "semantic_match_dim": args.semantic_match_dim,
        "semantic_match_temperature": args.semantic_match_temperature,
        "semantic_match_max_align": args.semantic_match_max_align,
        "semantic_match_alpha_weight": args.semantic_match_alpha_weight == 1,
        "semantic_match_detach_scores": args.semantic_match_detach_scores == 1,
        "semantic_match_exclude_style_tokens": args.semantic_match_exclude_style_tokens == 1,
        "semantic_cycle_loss_weight": args.semantic_cycle_loss_weight,
        "semantic_cycle_loss_prob": args.semantic_cycle_loss_prob,
        "semantic_cycle_detach_targets": args.semantic_cycle_detach_targets == 1,
        "semantic_cycle_alpha_weight": args.semantic_cycle_alpha_weight == 1,
        "semantic_match_log_stats": args.semantic_match_log_stats == 1,
    }'''
    text = replace_once(text, build_marker, build_add, rel, "build_model kwargs")

    text = replace_once(
        text,
        '''        ("relative_improvement", "slat_rel"),
        ("pred_target_cosine", "slat_cos"),
        ("pred_std", "pred_std"),''',
        '''        ("relative_improvement", "slat_rel"),
        ("pred_target_cosine", "slat_cos"),
        ("semantic_cycle_loss_weighted", "sem_cyc"),
        ("semantic_align_lambda", "sem_lam"),
        ("semantic_entropy_12", "sem_H12"),
        ("pred_std", "pred_std"),''',
        rel,
        "metric summary",
    )

    for old, new, label in [
        (
            '"cond_encoder*", "cond_fusion*", "separate_cond_proj*", "cond_proj_layer_norm*", "cond_resampler*", "cond_token_layer_norm*", "cond_alpha_mod*", "dino_norm*"',
            '"cond_encoder*", "cond_fusion*", "separate_cond_proj*", "cond_proj_layer_norm*", "cond_resampler*", "cond_token_layer_norm*", "cond_alpha_mod*", "semantic_matcher*", "dino_norm*"',
            "freeze alias condition lists",
        ),
    ]:
        text = text.replace(old, new)

    text = insert_after_once(text, '    "cond_token_norm": ["cond_token_layer_norm*", "cond_alpha_mod*"],\n', '    "semantic_matcher": ["semantic_matcher*"],\n    "semantic_token_matching": ["semantic_matcher*"],\n', rel, "freeze aliases semantic")

    text = text.replace(
        '"cond_resampler", "cond_token_layer_norm", "cond_alpha_mod"]',
        '"cond_resampler", "cond_token_layer_norm", "cond_alpha_mod", "semantic_matcher"]',
    )
    text = replace_once(
        text,
        '''        "cond_token_layer_norm",
        "cond_alpha_mod",
        "dino_norm",''',
        '''        "cond_token_layer_norm",
        "cond_alpha_mod",
        "semantic_matcher",
        "dino_norm",''',
        rel,
        "param groups semantic",
    )

    validation_marker = '''    if args.t_logit_std <= 0.0:
        raise ValueError(f"--t_logit_std must be > 0, got {args.t_logit_std}")
    if args.slat_condition_source == "dino" and args.flow_target != "slat":'''
    validation_add = '''    if args.t_logit_std <= 0.0:
        raise ValueError(f"--t_logit_std must be > 0, got {args.t_logit_std}")
    if args.semantic_match_temperature <= 0.0:
        raise ValueError(f"--semantic_match_temperature must be > 0, got {args.semantic_match_temperature}")
    if args.semantic_match_dim <= 0:
        raise ValueError(f"--semantic_match_dim must be > 0, got {args.semantic_match_dim}")
    if args.semantic_match_max_align < 0.0:
        raise ValueError(f"--semantic_match_max_align must be >= 0, got {args.semantic_match_max_align}")
    if args.semantic_cycle_loss_weight < 0.0:
        raise ValueError(f"--semantic_cycle_loss_weight must be >= 0, got {args.semantic_cycle_loss_weight}")
    if args.semantic_cycle_loss_prob < 0.0 or args.semantic_cycle_loss_prob > 1.0:
        raise ValueError(f"--semantic_cycle_loss_prob must be in [0, 1], got {args.semantic_cycle_loss_prob}")
    if args.semantic_cycle_loss_weight > 0.0 and args.use_semantic_token_matching == 0:
        raise ValueError("--semantic_cycle_loss_weight > 0 requires --use_semantic_token_matching=1")
    if args.slat_condition_source == "dino" and args.flow_target != "slat":'''
    text = replace_once(text, validation_marker, validation_add, rel, "arg validation")

    print_marker = '''    accelerator.print(f"Separate cond: {args.separate_cond == 1}")
    accelerator.print(f"Separate cond gate: {args.separate_cond_gate}")
    accelerator.print(f"CFG drop probability: {args.cfg_drop_prob}")'''
    print_add = '''    accelerator.print(f"Separate cond: {args.separate_cond == 1}")
    accelerator.print(f"Separate cond gate: {args.separate_cond_gate}")
    accelerator.print(f"Semantic token matching: {args.use_semantic_token_matching == 1}")
    if args.use_semantic_token_matching == 1:
        accelerator.print(f"Semantic match dim: {args.semantic_match_dim}")
        accelerator.print(f"Semantic match temperature: {args.semantic_match_temperature}")
        accelerator.print(f"Semantic match max align: {args.semantic_match_max_align}")
        accelerator.print(f"Semantic match alpha weighting: {args.semantic_match_alpha_weight == 1}")
        accelerator.print(f"Semantic match detach scores: {args.semantic_match_detach_scores == 1}")
        accelerator.print(f"Semantic match exclude style tokens: {args.semantic_match_exclude_style_tokens == 1}")
        accelerator.print(f"Semantic cycle loss weight: {args.semantic_cycle_loss_weight}")
        accelerator.print(f"Semantic cycle loss probability: {args.semantic_cycle_loss_prob}")
        accelerator.print(f"Semantic cycle detach targets: {args.semantic_cycle_detach_targets == 1}")
        accelerator.print(f"Semantic cycle alpha weighting: {args.semantic_cycle_alpha_weight == 1}")
    accelerator.print(f"CFG drop probability: {args.cfg_drop_prob}")'''
    text = replace_once(text, print_marker, print_add, rel, "config print")

    write(rel, text)


def patch_eval() -> None:
    rel = "eval_validation_latents.py"
    text = read(rel)
    marker = '''        "dino_model": args.get("dino_model", "dinov2_vitl14_reg"),
        "dino_dim": int(args.get("dino_dim", 1024)),
        "dino_layer_norm": bool(int(args.get("dino_layer_norm", 1))),
    }'''
    add = '''        "dino_model": args.get("dino_model", "dinov2_vitl14_reg"),
        "dino_dim": int(args.get("dino_dim", 1024)),
        "dino_layer_norm": bool(int(args.get("dino_layer_norm", 1))),
        "use_semantic_token_matching": bool(int(args.get("use_semantic_token_matching", 0))),
        "semantic_match_dim": int(args.get("semantic_match_dim", 128)),
        "semantic_match_temperature": float(args.get("semantic_match_temperature", 0.1)),
        "semantic_match_max_align": float(args.get("semantic_match_max_align", 0.25)),
        "semantic_match_alpha_weight": bool(int(args.get("semantic_match_alpha_weight", 1))),
        "semantic_match_detach_scores": bool(int(args.get("semantic_match_detach_scores", 0))),
        "semantic_match_exclude_style_tokens": bool(int(args.get("semantic_match_exclude_style_tokens", 1))),
        "semantic_cycle_loss_weight": float(args.get("semantic_cycle_loss_weight", 0.0)),
        "semantic_cycle_loss_prob": float(args.get("semantic_cycle_loss_prob", 1.0)),
        "semantic_cycle_detach_targets": bool(int(args.get("semantic_cycle_detach_targets", 1))),
        "semantic_cycle_alpha_weight": bool(int(args.get("semantic_cycle_alpha_weight", 1))),
        "semantic_match_log_stats": bool(int(args.get("semantic_match_log_stats", 1))),
    }'''
    text = replace_once(text, marker, add, rel, "checkpoint kwargs")
    write(rel, text)


def main() -> None:
    if not (ROOT / "models").is_dir() or not (ROOT / "train.py").is_file():
        die("run from MorphFlow repository root")
    write("models/semantic_token_matching.py", SEMANTIC_TOKEN_MATCHING)
    patch_morph_flow()
    patch_morph_slat_flow()
    patch_morph_residual_flow()
    patch_train()
    patch_eval()
    print("\nDone. Now run:")
    print("  python -m py_compile models/semantic_token_matching.py models/morph_flow.py models/morph_slat_flow.py models/morph_residual_flow.py train.py eval_validation_latents.py")
    print("  git diff --stat")


if __name__ == "__main__":
    main()
