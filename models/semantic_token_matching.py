
import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


# Every enabled matcher must report this same schema on every rank, even when
# cycle loss is sampled off locally or attention-stat logging is disabled.
SEMANTIC_METRIC_NAMES = (
    "semantic_align_lambda",
    "semantic_entropy_12",
    "semantic_entropy_21",
    "semantic_usage_12_max",
    "semantic_usage_12_min",
    "semantic_usage_21_max",
    "semantic_usage_21_min",
    "semantic_cycle_loss",
    "semantic_cycle_loss_weighted",
    "semantic_cycle_active",
    "semantic_usage_loss",
    "semantic_usage_loss_weighted",
    "semantic_usage_active",
)


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
        usage_cap: float = 4.0,
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
        if not math.isfinite(usage_cap) or usage_cap < 1.0:
            raise ValueError(f"usage_cap must be finite and >= 1, got {usage_cap}")

        self.dim = int(dim)
        self.match_dim = int(match_dim)
        self.temperature = float(temperature)
        self.max_align = float(max_align)
        self.alpha_weight = bool(alpha_weight)
        self.detach_scores = bool(detach_scores)
        self.style_tokens = int(style_tokens)
        self.cycle_detach_targets = bool(cycle_detach_targets)
        self.cycle_alpha_weight = bool(cycle_alpha_weight)
        self.usage_cap = float(usage_cap)

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

    def _usage_loss(self, a12: torch.Tensor, a21: torch.Tensor) -> torch.Tensor:
        """Penalize only excessive destination-token usage in both directions.

        For [B, queries, destinations], destinations * mean_queries(A) has mean
        one even when the two sources have different token counts. This loss
        leaves all usage <= cap unpenalized; it does not enforce uniform usage.
        Keep the reductions in fp32 and attached to Q/K's computation graph.
        """
        u12 = a12.float().mean(dim=1) * float(a12.shape[-1])
        u21 = a21.float().mean(dim=1) * float(a21.shape[-1])
        return 0.5 * (
            torch.relu(u12 - self.usage_cap).square().mean()
            + torch.relu(u21 - self.usage_cap).square().mean()
        )

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
        compute_usage: bool = False,
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
                "usage_loss": cond1.new_zeros(()),
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
        usage_loss = self._usage_loss(a12, a21) if compute_usage else cond1.new_zeros(())
        return cond1_out, cond2_out, {"cycle_loss": cycle_loss, "usage_loss": usage_loss, "metrics": metrics}


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
        semantic_usage_loss_weight: float = 0.0,
        semantic_usage_cap: float = 4.0,
    ) -> None:
        if semantic_cycle_loss_weight < 0.0:
            raise ValueError(f"semantic_cycle_loss_weight must be >= 0, got {semantic_cycle_loss_weight}")
        if semantic_cycle_loss_prob < 0.0 or semantic_cycle_loss_prob > 1.0:
            raise ValueError(f"semantic_cycle_loss_prob must be in [0, 1], got {semantic_cycle_loss_prob}")
        if semantic_cycle_loss_weight > 0.0 and not use_semantic_token_matching:
            raise ValueError("semantic_cycle_loss_weight > 0 requires use_semantic_token_matching=True")
        if not math.isfinite(semantic_usage_loss_weight) or semantic_usage_loss_weight < 0.0:
            raise ValueError("semantic_usage_loss_weight must be finite and >= 0")
        if not math.isfinite(semantic_usage_cap) or semantic_usage_cap < 1.0:
            raise ValueError("semantic_usage_cap must be finite and >= 1")
        if semantic_usage_loss_weight > 0.0 and not use_semantic_token_matching:
            raise ValueError("semantic_usage_loss_weight > 0 requires use_semantic_token_matching=True")

        self.use_semantic_token_matching = bool(use_semantic_token_matching)
        self.semantic_cycle_loss_weight = float(semantic_cycle_loss_weight)
        self.semantic_cycle_loss_prob = float(semantic_cycle_loss_prob)
        self.semantic_match_log_stats = bool(semantic_match_log_stats)
        self.semantic_usage_loss_weight = float(semantic_usage_loss_weight)
        self.semantic_usage_cap = float(semantic_usage_cap)
        self._semantic_match_record_aux = False
        self._semantic_match_cycle_terms = []
        self._semantic_match_record_usage = False
        self._semantic_match_usage_terms = []
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
                usage_cap=semantic_usage_cap,
            )

    def _begin_semantic_match_record(self, device: torch.device) -> None:
        self._semantic_match_cycle_terms = []
        self._semantic_match_usage_terms = []
        self._semantic_match_last_metrics = {}
        self._semantic_match_record_aux = False
        # Usage regularization is cheap and deterministic: apply on every loss
        # forward, including validation, independently of cycle activation/alpha.
        self._semantic_match_record_usage = bool(
            self.use_semantic_token_matching and self.semantic_usage_loss_weight > 0.0
        )
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
        return_aux = bool(self._semantic_match_record_aux or self._semantic_match_record_usage or self.semantic_match_log_stats)
        if return_aux:
            cond1, cond2, aux = self.semantic_matcher(
                cond1,
                cond2,
                alpha,
                return_aux=True,
                compute_cycle=self._semantic_match_record_aux,
                compute_usage=self._semantic_match_record_usage,
            )
            self._semantic_match_last_metrics = aux["metrics"]
            if self._semantic_match_record_aux:
                self._semantic_match_cycle_terms.append(aux["cycle_loss"])
            if self._semantic_match_record_usage:
                self._semantic_match_usage_terms.append(aux["usage_loss"])
            return cond1, cond2
        return self.semantic_matcher(cond1, cond2, alpha, return_aux=False)

    def _semantic_match_aux_loss(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        total = torch.zeros((), device=device, dtype=torch.float32)
        for prefix, terms, weight in (
            ("semantic_cycle", self._semantic_match_cycle_terms, self.semantic_cycle_loss_weight),
            ("semantic_usage", self._semantic_match_usage_terms, self.semantic_usage_loss_weight),
        ):
            if not terms:
                continue
            raw = torch.stack([t.to(device=device, dtype=torch.float32) for t in terms]).mean()
            weighted = raw * weight
            total = total + weighted
            self._semantic_match_last_metrics.update({
                f"{prefix}_loss": raw.detach(),
                f"{prefix}_loss_weighted": weighted.detach(),
                f"{prefix}_active": torch.ones((), device=device, dtype=torch.float32),
            })
        self._semantic_match_record_aux = False
        self._semantic_match_record_usage = False
        return total.to(dtype=dtype)

    def _semantic_match_metrics(self) -> Dict[str, torch.Tensor]:
        if not getattr(self, "use_semantic_token_matching", False):
            return {}
        metrics = getattr(self, "_semantic_match_last_metrics", {})
        device = next(self.parameters()).device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        # Missing cycle metrics represent a zero contribution from this rank.
        # Keep both keys and order independent of its random loss activation.
        return {name: metrics.get(name, zero) for name in SEMANTIC_METRIC_NAMES}
