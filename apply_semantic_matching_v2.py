#!/usr/bin/env python3
"""
Robust MorphFlow semantic token matching patcher.
Run from repo root:
    python3 apply_semantic_matching_v2.py

This v2 avoids fragile unified-patch hunks and rewrites whole target functions
when needed. It does not require a clean working tree, but review git diff.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path.cwd()

SEMANTIC_TOKEN_MATCHING = r'''from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class SemanticTokenMatcher(nn.Module):
    """
    Bidirectional soft semantic matching for MorphFlow condition tokens.

    Inputs are two source-condition token sequences [B, N, C]. The module keeps
    the token axis unchanged and mixes each stream with a soft correspondence
    from the opposite stream. It is therefore safe before both:
      - PairConditionFusionV2 (single-condition path), and
      - separate condition projection/gating (alpha_residual, pair_channel, token).

    Optional tail tokens, e.g. global style tokens appended by sparse_conv3d,
    can be excluded from matching and copied unchanged.
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
            return cond1[:, :0], cond1, cond2[:, :0], cond2
        return cond1[:, :-tail], cond1[:, -tail:], cond2[:, :-tail], cond2[:, -tail:]

    def _alpha_lambda(self, alpha: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        alpha = alpha.reshape(-1, 1, 1).to(device=cond.device, dtype=cond.dtype)
        if self.alpha_weight:
            profile = 4.0 * alpha * (1.0 - alpha)
        else:
            profile = torch.ones_like(alpha)
        return (self.max_align * profile).clamp(min=0.0, max=self.max_align)

    def _affinity(self, cond1: torch.Tensor, cond2: torch.Tensor) -> torch.Tensor:
        q_input = cond1.detach() if self.detach_scores else cond1
        k_input = cond2.detach() if self.detach_scores else cond2
        q = F.normalize(self.q_proj(q_input).float(), dim=-1)
        k = F.normalize(self.k_proj(k_input).float(), dim=-1)
        return torch.matmul(q, k.transpose(-1, -2)) / self.temperature

    @staticmethod
    def _attention_entropy(attn: torch.Tensor) -> torch.Tensor:
        return -(attn * attn.clamp_min(1e-8).log()).sum(dim=-1).mean()

    @staticmethod
    def _attention_usage_stats(attn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            a = alpha.reshape(-1).to(device=per_item.device, dtype=per_item.dtype)
            per_item = per_item * (4.0 * a * (1.0 - a)).clamp(min=0.0)
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
                "SemanticTokenMatcher expects cond1/cond2 [B, tokens, C], "
                f"got {tuple(cond1.shape)} and {tuple(cond2.shape)}"
            )
        if cond1.shape[0] != cond2.shape[0] or cond1.shape[-1] != cond2.shape[-1]:
            raise ValueError(
                "cond1/cond2 must share batch and channel dims, "
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
        return cond1_out, cond2_out, {"cycle_loss": cycle_loss, "metrics": metrics}


class SemanticTokenMatchingMixin:
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
        raw = torch.stack([t.to(device=device, dtype=torch.float32) for t in self._semantic_match_cycle_terms]).mean()
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


def path(rel: str) -> Path:
    return ROOT / rel


def read(rel: str) -> str:
    p = path(rel)
    if not p.is_file():
        die(f"missing {rel}. Run from repo root.")
    return p.read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    p = path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print(f"updated: {rel}")


def add_once(text: str, needle: str, repl: str, label: str) -> str:
    if repl in text:
        print(f"skip already applied: {label}")
        return text
    if needle not in text:
        die(f"missing marker for {label}")
    return text.replace(needle, repl, 1)


def regex_once(text: str, pattern: str, repl: str, label: str, flags: int = 0) -> str:
    if repl in text:
        print(f"skip already applied: {label}")
        return text
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        die(f"expected one regex match for {label}, found {n}")
    return new


def replace_function(text: str, name: str, new_func: str, label: str) -> str:
    pat = re.compile(rf"^    def {re.escape(name)}\([^\n]*\n(?:^        .*\n|^\s*$)*?", re.M)
    m = pat.search(text)
    if not m:
        # More robust manual scan for multiline signatures and body.
        start = text.find(f"    def {name}(")
        if start < 0:
            die(f"function not found for {label}: {name}")
        pos = start + 1
        next_def = None
        while True:
            idx = text.find("\n    def ", pos)
            if idx < 0:
                next_def = len(text)
                break
            next_def = idx + 1
            break
        return text[:start] + new_func.rstrip() + "\n\n" + text[next_def:]
    start, end = m.span()
    return text[:start] + new_func.rstrip() + "\n\n" + text[end:]


def append_constructor_args(text: str, class_name: str, anchor: str) -> str:
    if "use_semantic_token_matching" in text[text.find(f"class {class_name}"): text.find(f"class {class_name}") + 4000]:
        print(f"skip already applied: {class_name} constructor args")
        return text
    extra = '''        use_semantic_token_matching=False,
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
'''
    return add_once(text, anchor, anchor + extra, f"{class_name} constructor args")


def semantic_init_block(indent: str = "        ") -> str:
    return f'''
{indent}semantic_style_tokens = 0
{indent}if semantic_match_exclude_style_tokens and not hasattr(self, "cond_resampler"):
{indent}    semantic_style_tokens = int(getattr(self.cond_encoder, "style_tokens", 0))
{indent}self._init_semantic_token_matching(
{indent}    use_semantic_token_matching=use_semantic_token_matching,
{indent}    semantic_match_dim=semantic_match_dim,
{indent}    semantic_match_temperature=semantic_match_temperature,
{indent}    semantic_match_max_align=semantic_match_max_align,
{indent}    semantic_match_alpha_weight=semantic_match_alpha_weight,
{indent}    semantic_match_detach_scores=semantic_match_detach_scores,
{indent}    semantic_match_style_tokens=semantic_style_tokens,
{indent}    semantic_cycle_loss_weight=semantic_cycle_loss_weight,
{indent}    semantic_cycle_loss_prob=semantic_cycle_loss_prob,
{indent}    semantic_cycle_detach_targets=semantic_cycle_detach_targets,
{indent}    semantic_cycle_alpha_weight=semantic_cycle_alpha_weight,
{indent}    semantic_match_log_stats=semantic_match_log_stats,
{indent})
'''


def add_semantic_init_before_cfg(text: str, label: str) -> str:
    if "self._init_semantic_token_matching(" in text:
        print(f"skip already applied: {label} semantic init")
        return text
    marker = "        self.cfg_drop_prob = 0.0\n"
    return add_once(text, marker, semantic_init_block() + "\n" + marker, f"{label} semantic init")


MORPH_FLOW_BUILD_CONDITION = '''    def _build_condition(
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
                cond = torch.where(drop_mask.view(B, 1, 1), null_cond, cond)
            else:
                drop_mask = drop_mask.view(B, 1, 1)
                cond = (
                    torch.where(drop_mask, null_cond, cond1),
                    torch.where(drop_mask, null_cond, cond2),
                    alpha,
                )
        return cond
'''

MORPH_FLOW_FORWARD_FLOW = '''    def forward_flow(
        self,
        x_t,
        t,
        src_1_feats,
        src_2_feats,
        src_1_coords,
        src_2_coords,
        alpha,
        apply_cfg_drop=True,
    ):
        cond = self._build_condition(
            src_1_feats,
            src_2_feats,
            src_1_coords,
            src_2_coords,
            alpha,
            apply_cfg_drop=apply_cfg_drop,
        )
        t_flow = t.float() * 1000.0
        return self.sparse_structure_flow(x_t, t_flow, cond, alpha=alpha)
'''

MORPH_FLOW_FORWARD_FLOW_CFG = '''    def forward_flow_cfg(self, x_t, t, src_1_feats, src_2_feats, src_1_coords, src_2_coords, alpha, guidance_scale=1.0):
        if guidance_scale == 1.0:
            return self.forward_flow(
                x_t,
                t,
                src_1_feats,
                src_2_feats,
                src_1_coords,
                src_2_coords,
                alpha,
            )

        cond = self._build_condition(
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

        t_flow = t.float() * 1000.0
        v_cond = self.sparse_structure_flow(x_t, t_flow, cond, alpha=alpha)
        v_uncond = self.sparse_structure_flow(x_t, t_flow, null_cond, alpha=alpha)
        return v_uncond + guidance_scale * (v_cond - v_uncond)
'''

MORPH_FLOW_FLOW_MATCHING_LOSS = '''    def flow_matching_loss(
        self,
        x_0,
        src_1_feats,
        src_1_coords,
        src_2_feats,
        src_2_coords,
        alpha,
        return_terms=False,
        apply_cfg_drop=True,
    ):
        B = x_0.shape[0]
        x_0 = self._prepare_ss_latent(x_0)
        t = self.sample_t(B, x_0.device)
        x_t, noise = self.diffuse(x_0, t)
        velocity = self.get_v(x_0, noise)

        self._begin_semantic_match_record(x_0.device)
        pred = self.forward_flow(
            x_t,
            t,
            src_1_feats,
            src_2_feats,
            src_1_coords,
            src_2_coords,
            alpha,
            apply_cfg_drop=apply_cfg_drop,
        )
        loss = F.mse_loss(pred, velocity)
        semantic_aux = self._semantic_match_aux_loss(x_0.device, loss.dtype)
        loss = loss + semantic_aux
        self.last_forward_metrics = self._semantic_match_metrics()

        if return_terms:
            return loss, x_t, t, pred
        return loss
'''

MORPH_SLAT_FORWARD_FLOW_CFG = '''    def forward_flow_cfg(
        self,
        x_t: sp.SparseTensor,
        t: torch.Tensor,
        src_1_feats: torch.Tensor,
        src_2_feats: torch.Tensor,
        src_1_coords: torch.Tensor,
        src_2_coords: torch.Tensor,
        alpha: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> sp.SparseTensor:
        if guidance_scale == 1.0:
            return self.forward_flow(
                x_t,
                t,
                src_1_feats,
                src_2_feats,
                src_1_coords,
                src_2_coords,
                alpha,
            )

        cond = self._build_condition(
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

        t_flow = t.float() * 1000.0
        v_cond = self.slat_flow(x_t, t_flow, cond, alpha=alpha)
        v_uncond = self.slat_flow(x_t, t_flow, null_cond, alpha=alpha)
        return v_uncond + guidance_scale * (v_cond - v_uncond)
'''

MORPH_SLAT_FORWARD = '''    def forward(
        self,
        target_feats: torch.Tensor,
        target_coords: torch.Tensor,
        src_1_feats: torch.Tensor,
        src_1_coords: torch.Tensor,
        src_2_feats: torch.Tensor,
        src_2_coords: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        x_0 = self.make_slat(target_feats, target_coords)
        x_0 = self.normalize_slat(x_0)

        batch_size = x_0.shape[0]
        t = self.sample_t(batch_size, x_0.device)
        x_t, noise = self.diffuse(x_0, t)
        velocity = self.get_v(x_0, noise)

        self._begin_semantic_match_record(x_0.device)
        pred = self.forward_flow(
            x_t,
            t,
            src_1_feats,
            src_2_feats,
            src_1_coords,
            src_2_coords,
            alpha,
        )

        base_loss = self.batch_mean_mse(pred, velocity)
        semantic_aux = self._semantic_match_aux_loss(x_0.device, base_loss.dtype)
        loss = base_loss + semantic_aux
        self._update_forward_metrics(pred, velocity, loss)
        self.last_forward_metrics.update(self._semantic_match_metrics())
        return loss
'''

RESIDUAL_FLOW_MATCHING_LOSS = '''    def flow_matching_loss(
        self,
        x_0,
        src_1_feats,
        src_1_coords,
        src_2_feats,
        src_2_coords,
        alpha,
        return_terms=False,
        apply_cfg_drop=True,
        src1_ss_latent=None,
        src2_ss_latent=None,
    ):
        if src1_ss_latent is None or src2_ss_latent is None:
            raise ValueError("MorphResidualSSFlow requires src1_ss_latent and src2_ss_latent.")

        B = x_0.shape[0]
        x_0 = self.ss_to_residual(x_0, src1_ss_latent, src2_ss_latent, alpha)
        t = self.sample_t(B, x_0.device)
        x_t, noise = self.diffuse(x_0, t)
        velocity = self.get_v(x_0, noise)

        self._begin_semantic_match_record(x_0.device)
        pred = self.forward_flow(
            x_t,
            t,
            src_1_feats,
            src_2_feats,
            src_1_coords,
            src_2_coords,
            alpha,
            apply_cfg_drop=apply_cfg_drop,
        )
        loss = F.mse_loss(pred, velocity)
        semantic_aux = self._semantic_match_aux_loss(x_0.device, loss.dtype)
        loss = loss + semantic_aux
        self.last_forward_metrics = self._semantic_match_metrics()

        if return_terms:
            return loss, x_t, t, pred
        return loss
'''


def patch_morph_flow() -> None:
    rel = "models/morph_flow.py"
    text = read(rel)
    text = add_once(text, "from models import cond_encoder\n", "from models import cond_encoder\nfrom models.semantic_token_matching import SemanticTokenMatchingMixin\n", f"{rel} import")
    text = text.replace("class MorphFlow(nn.Module):", "class MorphFlow(SemanticTokenMatchingMixin, nn.Module):")
    text = append_constructor_args(text, "MorphFlow", "        t_logit_std=1.0,\n")
    text = add_semantic_init_before_cfg(text, rel)
    if "def _build_condition(" not in text:
        text = add_once(text, "    def get_v(self, x_0, noise):\n", MORPH_FLOW_BUILD_CONDITION + "\n    def get_v(self, x_0, noise):\n", f"{rel} build condition")
    text = replace_function(text, "forward_flow", MORPH_FLOW_FORWARD_FLOW, f"{rel} forward_flow")
    text = replace_function(text, "forward_flow_cfg", MORPH_FLOW_FORWARD_FLOW_CFG, f"{rel} forward_flow_cfg")
    text = replace_function(text, "flow_matching_loss", MORPH_FLOW_FLOW_MATCHING_LOSS, f"{rel} flow_matching_loss")
    write(rel, text)


def patch_morph_slat_flow() -> None:
    rel = "models/morph_slat_flow.py"
    text = read(rel)
    text = add_once(text, "from models import structured_latent_flow\n", "from models import structured_latent_flow\nfrom models.semantic_token_matching import SemanticTokenMatchingMixin\n", f"{rel} import")
    text = text.replace("class MorphSLatFlow(nn.Module):", "class MorphSLatFlow(SemanticTokenMatchingMixin, nn.Module):")
    text = append_constructor_args(text, "MorphSLatFlow", "        t_logit_std: float = 1.0,\n")
    text = add_semantic_init_before_cfg(text, rel)
    if "self._apply_semantic_token_matching(cond1, cond2, alpha)" not in text:
        text = add_once(
            text,
            "        cond1, cond2 = self.normalize_condition_tokens(cond1, cond2, alpha)\n\n        if not self.separate_cond:",
            "        cond1, cond2 = self.normalize_condition_tokens(cond1, cond2, alpha)\n        cond1, cond2 = self._apply_semantic_token_matching(cond1, cond2, alpha)\n\n        if not self.separate_cond:",
            f"{rel} apply matching",
        )
    text = replace_function(text, "forward_flow_cfg", MORPH_SLAT_FORWARD_FLOW_CFG, f"{rel} forward_flow_cfg")
    text = replace_function(text, "forward", MORPH_SLAT_FORWARD, f"{rel} forward")
    write(rel, text)


def patch_residual_flow() -> None:
    rel = "models/morph_residual_flow.py"
    text = read(rel)
    if "use_semantic_token_matching" not in text[text.find("def __init__"): text.find("def __init__") + 2500]:
        text = add_once(text, "        t_logit_std=1.0,\n", "        t_logit_std=1.0,\n        use_semantic_token_matching=False,\n        semantic_match_dim=128,\n        semantic_match_temperature=0.1,\n        semantic_match_max_align=0.25,\n        semantic_match_alpha_weight=True,\n        semantic_match_detach_scores=False,\n        semantic_match_exclude_style_tokens=True,\n        semantic_cycle_loss_weight=0.0,\n        semantic_cycle_loss_prob=1.0,\n        semantic_cycle_detach_targets=True,\n        semantic_cycle_alpha_weight=True,\n        semantic_match_log_stats=True,\n", f"{rel} constructor args")
        text = add_once(text, "            t_logit_std=t_logit_std,\n", "            t_logit_std=t_logit_std,\n            use_semantic_token_matching=use_semantic_token_matching,\n            semantic_match_dim=semantic_match_dim,\n            semantic_match_temperature=semantic_match_temperature,\n            semantic_match_max_align=semantic_match_max_align,\n            semantic_match_alpha_weight=semantic_match_alpha_weight,\n            semantic_match_detach_scores=semantic_match_detach_scores,\n            semantic_match_exclude_style_tokens=semantic_match_exclude_style_tokens,\n            semantic_cycle_loss_weight=semantic_cycle_loss_weight,\n            semantic_cycle_loss_prob=semantic_cycle_loss_prob,\n            semantic_cycle_detach_targets=semantic_cycle_detach_targets,\n            semantic_cycle_alpha_weight=semantic_cycle_alpha_weight,\n            semantic_match_log_stats=semantic_match_log_stats,\n", f"{rel} super kwargs")
    else:
        print(f"skip already applied: {rel} constructor args")
    text = replace_function(text, "flow_matching_loss", RESIDUAL_FLOW_MATCHING_LOSS, f"{rel} flow_matching_loss")
    write(rel, text)


def patch_train() -> None:
    rel = "train.py"
    text = read(rel)
    parser_marker = "    # Optional future losses.\n"
    parser_add = '''    # Semantic token matching / cycle consistency on source condition tokens.
    parser.add_argument("--use_semantic_token_matching", type=int, choices=[0, 1], default=0)
    parser.add_argument("--semantic_match_dim", type=int, default=128)
    parser.add_argument("--semantic_match_temperature", type=float, default=0.1)
    parser.add_argument("--semantic_match_max_align", type=float, default=0.25)
    parser.add_argument("--semantic_match_alpha_weight", type=int, choices=[0, 1], default=1)
    parser.add_argument("--semantic_match_detach_scores", type=int, choices=[0, 1], default=0)
    parser.add_argument("--semantic_match_exclude_style_tokens", type=int, choices=[0, 1], default=1)
    parser.add_argument("--semantic_cycle_loss_weight", type=float, default=0.0)
    parser.add_argument("--semantic_cycle_loss_prob", type=float, default=1.0)
    parser.add_argument("--semantic_cycle_detach_targets", type=int, choices=[0, 1], default=1)
    parser.add_argument("--semantic_cycle_alpha_weight", type=int, choices=[0, 1], default=1)
    parser.add_argument("--semantic_match_log_stats", type=int, choices=[0, 1], default=1)

    # Optional future losses.
'''
    if "--use_semantic_token_matching" not in text:
        text = add_once(text, parser_marker, parser_add, f"{rel} parser args")

    if '"use_semantic_token_matching": args.use_semantic_token_matching == 1,' not in text:
        text = add_once(
            text,
            '        "dino_layer_norm": args.dino_layer_norm == 1,\n',
            '        "dino_layer_norm": args.dino_layer_norm == 1,\n        "use_semantic_token_matching": args.use_semantic_token_matching == 1,\n        "semantic_match_dim": args.semantic_match_dim,\n        "semantic_match_temperature": args.semantic_match_temperature,\n        "semantic_match_max_align": args.semantic_match_max_align,\n        "semantic_match_alpha_weight": args.semantic_match_alpha_weight == 1,\n        "semantic_match_detach_scores": args.semantic_match_detach_scores == 1,\n        "semantic_match_exclude_style_tokens": args.semantic_match_exclude_style_tokens == 1,\n        "semantic_cycle_loss_weight": args.semantic_cycle_loss_weight,\n        "semantic_cycle_loss_prob": args.semantic_cycle_loss_prob,\n        "semantic_cycle_detach_targets": args.semantic_cycle_detach_targets == 1,\n        "semantic_cycle_alpha_weight": args.semantic_cycle_alpha_weight == 1,\n        "semantic_match_log_stats": args.semantic_match_log_stats == 1,\n',
            f"{rel} model kwargs",
        )

    text = text.replace('"cond_alpha_mod*", "dino_norm*"', '"cond_alpha_mod*", "semantic_matcher*", "dino_norm*"')
    text = text.replace('"cond_alpha_mod*", "null_cond"', '"cond_alpha_mod*", "semantic_matcher*", "null_cond"')
    if '"semantic_matcher": ["semantic_matcher*"],' not in text:
        text = add_once(text, '    "cond_token_norm": ["cond_token_layer_norm*", "cond_alpha_mod*"],\n', '    "cond_token_norm": ["cond_token_layer_norm*", "cond_alpha_mod*"],\n    "semantic_matcher": ["semantic_matcher*"],\n    "semantic_token_matching": ["semantic_matcher*"],\n', f"{rel} freeze aliases")
    text = text.replace('"cond_resampler", "cond_token_layer_norm", "cond_alpha_mod"]', '"cond_resampler", "cond_token_layer_norm", "cond_alpha_mod", "semantic_matcher"]')
    if '        "semantic_matcher",\n        "dino_norm",' not in text:
        text = add_once(text, '        "cond_alpha_mod",\n        "dino_norm",\n', '        "cond_alpha_mod",\n        "semantic_matcher",\n        "dino_norm",\n', f"{rel} param groups")

    if "--semantic_match_temperature must be > 0" not in text:
        text = add_once(
            text,
            '    if args.t_logit_std <= 0.0:\n        raise ValueError(f"--t_logit_std must be > 0, got {args.t_logit_std}")\n',
            '    if args.t_logit_std <= 0.0:\n        raise ValueError(f"--t_logit_std must be > 0, got {args.t_logit_std}")\n    if args.semantic_match_temperature <= 0.0:\n        raise ValueError(f"--semantic_match_temperature must be > 0, got {args.semantic_match_temperature}")\n    if args.semantic_match_dim <= 0:\n        raise ValueError(f"--semantic_match_dim must be > 0, got {args.semantic_match_dim}")\n    if args.semantic_match_max_align < 0.0:\n        raise ValueError(f"--semantic_match_max_align must be >= 0, got {args.semantic_match_max_align}")\n    if args.semantic_cycle_loss_weight < 0.0:\n        raise ValueError(f"--semantic_cycle_loss_weight must be >= 0, got {args.semantic_cycle_loss_weight}")\n    if args.semantic_cycle_loss_prob < 0.0 or args.semantic_cycle_loss_prob > 1.0:\n        raise ValueError(f"--semantic_cycle_loss_prob must be in [0, 1], got {args.semantic_cycle_loss_prob}")\n    if args.semantic_cycle_loss_weight > 0.0 and args.use_semantic_token_matching == 0:\n        raise ValueError("--semantic_cycle_loss_weight > 0 requires --use_semantic_token_matching=1")\n',
            f"{rel} validation",
        )

    if '("semantic_cycle_loss_weighted", "sem_cyc"),' not in text:
        text = text.replace(
            '        ("pred_target_cosine", "slat_cos"),\n        ("pred_std", "pred_std"),',
            '        ("pred_target_cosine", "slat_cos"),\n        ("semantic_cycle_loss_weighted", "sem_cyc"),\n        ("semantic_align_lambda", "sem_lam"),\n        ("semantic_entropy_12", "sem_H12"),\n        ("pred_std", "pred_std"),',
        )

    if "Semantic token matching:" not in text:
        text = add_once(
            text,
            '    accelerator.print(f"Separate cond gate: {args.separate_cond_gate}")\n',
            '    accelerator.print(f"Separate cond gate: {args.separate_cond_gate}")\n    accelerator.print(f"Semantic token matching: {args.use_semantic_token_matching == 1}")\n    if args.use_semantic_token_matching == 1:\n        accelerator.print(f"Semantic match dim: {args.semantic_match_dim}")\n        accelerator.print(f"Semantic match temperature: {args.semantic_match_temperature}")\n        accelerator.print(f"Semantic match max align: {args.semantic_match_max_align}")\n        accelerator.print(f"Semantic match alpha weighting: {args.semantic_match_alpha_weight == 1}")\n        accelerator.print(f"Semantic match detach scores: {args.semantic_match_detach_scores == 1}")\n        accelerator.print(f"Semantic match exclude style tokens: {args.semantic_match_exclude_style_tokens == 1}")\n        accelerator.print(f"Semantic cycle loss weight: {args.semantic_cycle_loss_weight}")\n        accelerator.print(f"Semantic cycle loss probability: {args.semantic_cycle_loss_prob}")\n        accelerator.print(f"Semantic cycle detach targets: {args.semantic_cycle_detach_targets == 1}")\n        accelerator.print(f"Semantic cycle alpha weighting: {args.semantic_cycle_alpha_weight == 1}")\n',
            f"{rel} config print",
        )

    write(rel, text)


def patch_eval() -> None:
    rel = "eval_validation_latents.py"
    text = read(rel)
    if '"use_semantic_token_matching": bool(int(args.get("use_semantic_token_matching", 0))),' not in text:
        text = add_once(
            text,
            '        "dino_layer_norm": bool(int(args.get("dino_layer_norm", 1))),\n',
            '        "dino_layer_norm": bool(int(args.get("dino_layer_norm", 1))),\n        "use_semantic_token_matching": bool(int(args.get("use_semantic_token_matching", 0))),\n        "semantic_match_dim": int(args.get("semantic_match_dim", 128)),\n        "semantic_match_temperature": float(args.get("semantic_match_temperature", 0.1)),\n        "semantic_match_max_align": float(args.get("semantic_match_max_align", 0.25)),\n        "semantic_match_alpha_weight": bool(int(args.get("semantic_match_alpha_weight", 1))),\n        "semantic_match_detach_scores": bool(int(args.get("semantic_match_detach_scores", 0))),\n        "semantic_match_exclude_style_tokens": bool(int(args.get("semantic_match_exclude_style_tokens", 1))),\n        "semantic_cycle_loss_weight": float(args.get("semantic_cycle_loss_weight", 0.0)),\n        "semantic_cycle_loss_prob": float(args.get("semantic_cycle_loss_prob", 1.0)),\n        "semantic_cycle_detach_targets": bool(int(args.get("semantic_cycle_detach_targets", 1))),\n        "semantic_cycle_alpha_weight": bool(int(args.get("semantic_cycle_alpha_weight", 1))),\n        "semantic_match_log_stats": bool(int(args.get("semantic_match_log_stats", 1))),\n',
            f"{rel} kwargs",
        )
    write(rel, text)


def main() -> None:
    if not path("train.py").is_file() or not path("models").is_dir():
        die("run this script from MorphFlow repo root")
    write("models/semantic_token_matching.py", SEMANTIC_TOKEN_MATCHING)
    patch_morph_flow()
    patch_morph_slat_flow()
    patch_residual_flow()
    patch_train()
    patch_eval()
    print("\nDone. Now run:")
    print("  python -m py_compile models/semantic_token_matching.py models/morph_flow.py models/morph_slat_flow.py models/morph_residual_flow.py train.py eval_validation_latents.py")
    print("  git diff --stat")


if __name__ == "__main__":
    main()
