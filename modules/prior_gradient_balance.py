"""Measured projection/FM gradient balance for MorphFlow's single-node DDP run.

Active updates: forward AND all backwards run inside DDP.no_sync(). Gradients
are accumulated with backward(), packed as FP32, averaged across ranks, and
then combined exactly once. No autograd.grad() is used on the DDP model.
Supported: ordinary AdamW, FP32 parameters, BF16/no AMP, no accumulation,
non-reentrant activation checkpointing. FSDP/DeepSpeed/FP16 scaler are rejected.
"""
from dataclasses import asdict, dataclass
import json
import math
import time

import torch
import torch.distributed as dist

PATCH_VERSION = "mf-prior-balance-v3"


def add_balance_args(parser):
    g = parser.add_argument_group("Measured TRELLIS projection gradient balance")
    g.add_argument("--trellis_prior_grad_balance", type=int, choices=[0, 1], default=0)
    g.add_argument("--trellis_prior_ratio_target", type=float, default=0.5)
    g.add_argument("--trellis_prior_ratio_max", type=float, default=1.0)
    g.add_argument("--trellis_prior_group_ratio_max", type=float, default=1.0)
    g.add_argument("--trellis_prior_scale_ratio_max", type=float, default=0.20)
    g.add_argument("--trellis_prior_guard_ratio_max", type=float, default=0.25)
    g.add_argument("--trellis_prior_balance_ema", type=float, default=0.95)
    g.add_argument("--trellis_prior_lambda_min", type=float, default=0.01)
    g.add_argument("--trellis_prior_lambda_max", type=float, default=100.0)
    g.add_argument("--trellis_prior_phase_start_weight", type=float, default=0.1)
    g.add_argument("--trellis_prior_phase_warmup_steps", type=int, default=500)
    g.add_argument("--trellis_prior_guard_scale", type=float, default=0.1)
    g.add_argument("--trellis_prior_reset_phase", type=int, choices=[0, 1], default=0)
    g.add_argument("--trellis_prior_weak_patience", type=int, default=25,
                   help="Abort after this many consecutive weak active updates after phase warmup.")
    g.add_argument("--trellis_prior_log_every", type=int, default=20,
                   help="Console interval in ACTIVE prior applications, not FM updates.")
    g.add_argument("--trellis_prior_phase_eval_steps", default="500,1500,3000",
                   help="Additional evaluation-only checkpoints at new-phase offsets.")


def validate_balance_args(args):
    if not getattr(args, "trellis_prior_grad_balance", 0):
        return
    finite_positive = ("trellis_prior_ratio_target", "trellis_prior_ratio_max",
                       "trellis_prior_group_ratio_max",
                       "trellis_prior_scale_ratio_max",
                       "trellis_prior_guard_ratio_max",
                       "trellis_prior_lambda_min", "trellis_prior_lambda_max")
    for name in finite_positive:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError("--%s must be finite and positive" % name)
    for name in ("trellis_prior_phase_start_weight", "trellis_prior_guard_scale"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError("--%s must be finite and nonnegative" % name)
    if not 0 <= args.trellis_prior_balance_ema < 1:
        raise ValueError("--trellis_prior_balance_ema must be in [0,1)")
    if args.trellis_prior_ratio_target > args.trellis_prior_ratio_max:
        raise ValueError("ratio_target cannot exceed ratio_max")
    if args.trellis_prior_lambda_min > args.trellis_prior_lambda_max:
        raise ValueError("lambda_min cannot exceed lambda_max")
    if args.trellis_prior_weak_patience < 1:
        raise ValueError("weak_patience must be >=1")
    if args.trellis_prior_phase_warmup_steps < 0 or args.trellis_prior_log_every < 1:
        raise ValueError("Phase warmup must be >=0 and prior log interval >=1")
    parse_phase_eval_steps(args.trellis_prior_phase_eval_steps)
    if args.trellis_prior_weight <= 0: raise ValueError("Gradient balancing requires prior_weight > 0")
    if args.flow_target == "ss" and args.ss_flow_arch != "standard": raise ValueError("SS balancing requires standard SS")
    if args.flow_target not in ("ss","slat"): raise ValueError("balancing supports ss or slat")
    if args.endpoint_loss_weight != 0 or args.symmetry_loss_weight != 0:
        raise ValueError("This experiment requires endpoint/symmetry losses disabled")
    if args.semantic_cycle_loss_weight != 0:
        raise ValueError("Keep semantic cycle loss disabled in this experiment")


def parse_phase_eval_steps(value):
    if not str(value).strip():
        return set()
    steps = {int(x.strip()) for x in str(value).split(",") if x.strip()}
    if any(x <= 0 for x in steps):
        raise ValueError("Phase evaluation checkpoint offsets must be positive")
    return steps


def validate_balance_runtime(accelerator, model):
    if accelerator.mixed_precision not in ("bf16", "no"):
        raise ValueError("Measured gradient balance supports mixed_precision=bf16 or no only")
    if getattr(accelerator, "scaler", None) is not None:
        raise ValueError("AMP GradScaler is not supported by measured gradient balance")
    if getattr(accelerator, "gradient_accumulation_steps", 1) != 1:
        raise ValueError("Measured balance currently requires accumulation_steps=1")
    kind = str(getattr(accelerator, "distributed_type", "NO"))
    if not any(kind.endswith(x) for x in (".NO", ".MULTI_GPU", ".MULTI_CPU")) and kind not in ("NO", "MULTI_GPU", "MULTI_CPU"):
        raise ValueError("Only ordinary DDP is supported, not %s" % kind)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            raise TypeError("Expected a standard DDP model after accelerator.prepare()")


@dataclass(frozen=True)
class BalanceConfig:
    target: float = 0.5
    max_ratio: float = 1.0
    group_max_ratio: float = 1.0
    scale_ratio_max: float = 0.20
    guard_ratio_max: float = 0.25
    ema: float = 0.95
    min_weight: float = 0.01
    max_weight: float = 100.0
    start_weight: float = 0.1
    warmup_steps: int = 500
    guard_scale: float = 0.1
    min_norm: float = 1e-12
    weak_patience: int = 25

    @classmethod
    def from_args(cls, args):
        return cls(target=args.trellis_prior_ratio_target,
                   max_ratio=args.trellis_prior_ratio_max,
                   group_max_ratio=args.trellis_prior_group_ratio_max,
                   scale_ratio_max=args.trellis_prior_scale_ratio_max,
                   guard_ratio_max=args.trellis_prior_guard_ratio_max,
                   ema=args.trellis_prior_balance_ema,
                   min_weight=args.trellis_prior_lambda_min,
                   max_weight=args.trellis_prior_lambda_max,
                   start_weight=args.trellis_prior_phase_start_weight,
                   warmup_steps=args.trellis_prior_phase_warmup_steps,
                   guard_scale=args.trellis_prior_guard_scale,
                   weak_patience=args.trellis_prior_weak_patience)


class PriorGradientBalancer:
    def __init__(self, optimizer, config, phase_start, process_group=None):
        self.config = config
        self.phase_start = int(phase_start)
        self.process_group = process_group
        self.active_count = 0
        self.weak_streak = 0
        self.ema_log_weight = None
        self.params = []
        self.slices = []
        self.groups = []
        seen = set()
        offset = 0
        for i, group in enumerate(optimizer.param_groups):
            name = str(group.get("name", "group_%d" % i))
            if any(g[0] == name for g in self.groups):
                raise ValueError("Optimizer group names must be unique")
            start = offset
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if id(p) in seen:
                    raise ValueError("Duplicate trainable optimizer parameter")
                if p.dtype != torch.float32:
                    raise ValueError("Keep trainable parameters FP32; use BF16 autocast")
                seen.add(id(p))
                self.params.append(p)
                self.slices.append((offset, offset + p.numel()))
                offset += p.numel()
            self.groups.append((name, start, offset))
        if not self.params:
            raise ValueError("No trainable parameters to balance")
        self.device = self.params[0].device
        if any(p.device != self.device for p in self.params):
            raise ValueError("One device per process is required")
        self.numel = offset
        self.world = (dist.get_world_size(process_group)
                      if dist.is_available() and dist.is_initialized() else 1)
        self.chunk_elems = 2 * 1024 * 1024  # 8 MiB FP32 communication chunks
        self._assert_layout_matches_ranks()

    def _assert_layout_matches_ranks(self):
        if self.world == 1:
            return
        # Constructor-only metadata check prevents mismatched collectives later.
        layouts = [None] * self.world
        layout = (self.numel, [(tuple(p.shape), str(p.dtype)) for p in self.params],
                  self.groups)
        dist.all_gather_object(layouts, layout, group=self.process_group)
        if any(other != layout for other in layouts):
            raise RuntimeError("Trainable parameter layout differs across DDP ranks")

    def state_dict(self):
        return dict(version=PATCH_VERSION, config=asdict(self.config),
                    phase_start=self.phase_start, active_count=self.active_count,
                    weak_streak=self.weak_streak, ema_log_weight=self.ema_log_weight)

    def load_state_dict(self, state):
        if state.get("version") != PATCH_VERSION:
            raise ValueError("Unsupported prior balance checkpoint version")
        if state.get("config") != asdict(self.config):
            raise ValueError("Balance settings changed: use reset_phase=1 for a NEW experiment")
        self.phase_start = int(state["phase_start"])
        self.active_count = int(state["active_count"])
        self.weak_streak = int(state["weak_streak"])
        value = state["ema_log_weight"]
        self.ema_log_weight = None if value is None else float(value)

    def active(self, global_step, every):
        return (int(global_step) - self.phase_start) % int(every) == 0

    def _all_reduce(self, tensor, op=dist.ReduceOp.SUM):
        if self.world > 1:
            dist.all_reduce(tensor, op=op, group=self.process_group)
        return tensor

    def _finite_everywhere(self, tensor, label):
        bad = (~torch.isfinite(tensor).all()).to(dtype=torch.int32).reshape(1)
        self._all_reduce(bad, dist.ReduceOp.MAX)
        if bad.item():
            raise FloatingPointError("Non-finite %s on at least one rank; optimizer not stepped" % label)

    def _clear(self):
        for p in self.params:
            p.grad = None

    def _capture_global_gradient(self, loss, accelerator, retain_graph):
        self._clear()
        accelerator.backward(loss, retain_graph=retain_graph)
        flat = torch.zeros(self.numel, dtype=torch.float32, device=self.device)
        present = torch.zeros(len(self.params), dtype=torch.int32, device=self.device)
        for i, (p, (start, end)) in enumerate(zip(self.params, self.slices)):
            if p.grad is not None:
                grad = p.grad.detach()
                if grad.is_sparse:
                    grad = grad.to_dense()
                flat[start:end].copy_(grad.reshape(-1))
                present[i] = 1
        self._finite_everywhere(flat, "component gradient")
        for start in range(0, self.numel, self.chunk_elems):
            part = flat[start:start + self.chunk_elems]
            self._all_reduce(part)
        flat.div_(self.world)
        self._all_reduce(present, dist.ReduceOp.MAX)
        self._finite_everywhere(flat, "averaged component gradient")
        self._clear()
        return flat, present

    @staticmethod
    def _norm(x):
        return float(torch.linalg.vector_norm(x).item())

    def _metrics_for_vectors(self, gf, gp, gg, weight, guard_weight, suffix=""):
        nf, np_ = self._norm(gf), self._norm(gp)
        ng = 0.0 if gg is None else self._norm(gg)
        dot = float(torch.dot(gf, gp).item())
        eps = self.config.min_norm
        cosine = max(-1.0, min(1.0, dot / max(nf * np_, eps)))
        return {
            "balance/g_fm" + suffix: nf,
            "balance/g_projection_raw" + suffix: np_,
            "balance/g_projection_weighted" + suffix: weight * np_,
            "balance/ratio_projection_fm" + suffix: weight * np_ / max(nf, eps),
            "balance/cos_projection_fm" + suffix: cosine,
            "balance/cos_valid" + suffix: float(nf > eps and np_ > eps),
            "balance/g_guard_weighted" + suffix: guard_weight * ng,
            "balance/ratio_guard_fm" + suffix: guard_weight * ng / max(nf, eps),
        }

    def backward_parts(self, parts, accelerator, model, global_step):
        """Balance FM, TRELLIS shape, proactive scale, and emergency guard."""
        parts = dict(parts)
        if "scale" not in parts:
            parts["scale"] = parts["projection"] * 0.0
        expected = {"fm", "projection", "scale", "guard"}
        if set(parts) != expected:
            raise ValueError("Expected live scalar losses: fm, projection, scale, guard")
        if hasattr(model, "require_backward_grad_sync") and model.require_backward_grad_sync:
            raise RuntimeError("Forward and split backwards MUST be inside no_sync")
        if getattr(accelerator, "scaler", None) is not None:
            raise ValueError("Do not use a GradScaler in split gradient mode")
        for name, value in parts.items():
            if value.ndim != 0 or not value.requires_grad:
                raise ValueError("%s must be a scalar attached to the student graph" % name)

        order = ("fm", "projection", "scale", "guard")
        local_values = torch.stack([parts[n].detach().float() for n in order])
        self._finite_everywhere(local_values, "loss")

        scale_flag = (local_values[2].abs() > 0).to(torch.int32).reshape(1)
        guard_flag = (local_values[3].abs() > 0).to(torch.int32).reshape(1)
        self._all_reduce(scale_flag, dist.ReduceOp.MAX)
        self._all_reduce(guard_flag, dist.ReduceOp.MAX)
        scale_on = bool(scale_flag.item())
        guard_on = bool(guard_flag.item()) and self.config.guard_scale > 0

        gf, used = self._capture_global_gradient(parts["fm"], accelerator, True)
        gp, used_p = self._capture_global_gradient(
            parts["projection"], accelerator, scale_on or guard_on
        )
        used.copy_(torch.maximum(used, used_p))

        gs = None
        if scale_on:
            gs, used_s = self._capture_global_gradient(
                parts["scale"], accelerator, guard_on
            )
            used.copy_(torch.maximum(used, used_s))

        gg = None
        if guard_on:
            gg, used_g = self._capture_global_gradient(
                parts["guard"], accelerator, False
            )
            used.copy_(torch.maximum(used, used_g))

        nf, np_ = self._norm(gf), self._norm(gp)
        if nf <= self.config.min_norm or np_ <= self.config.min_norm:
            raise RuntimeError(
                "Cannot calibrate: gFM=%.3e gProjection=%.3e. Check gradient connectivity."
                % (nf, np_)
            )

        ideal = self.config.target * nf / np_
        log_ideal = math.log(ideal)
        if self.ema_log_weight is None:
            self.ema_log_weight = log_ideal
        else:
            self.ema_log_weight = (
                self.config.ema * self.ema_log_weight
                + (1 - self.config.ema) * log_ideal
            )
        estimate = math.exp(self.ema_log_weight)
        estimate_bounded = min(
            self.config.max_weight, max(self.config.min_weight, estimate)
        )
        phase_step = int(global_step) + 1 - self.phase_start
        if phase_step <= 0:
            raise ValueError("global_step precedes new phase start")
        ramp = min(1.0, phase_step / max(1, self.config.warmup_steps))
        nominal = (
            (1 - ramp) * self.config.start_weight
            + ramp * estimate_bounded
        )

        # Shape prior: global cap plus cap for every optimizer parameter group.
        cap = self.config.max_ratio * nf / np_
        group_cap = float("inf")
        for _name, start, end in self.groups:
            nf_g = self._norm(gf[start:end])
            np_g = self._norm(gp[start:end])
            if np_g > self.config.min_norm:
                if nf_g <= self.config.min_norm:
                    group_cap = 0.0
                else:
                    group_cap = min(
                        group_cap,
                        self.config.group_max_ratio * nf_g / np_g,
                    )
        weight = min(nominal, cap, group_cap)

        # Scale anchor: preserve the natural vanishing gradient near the dead-zone.
        # The configured anchor weight lives inside parts["scale"]; this multiplier
        # only reduces it when its parameter-gradient norm would exceed the cap.
        ns = 0.0 if gs is None else self._norm(gs)
        scale_multiplier = 0.0 if gs is None else 1.0
        scale_cap = float("inf")
        if gs is not None and ns > self.config.min_norm:
            scale_cap = self.config.scale_ratio_max * nf / ns
            scale_multiplier = min(1.0, scale_cap)

        # Emergency guard: also hard-cap its parameter-gradient contribution.
        ng = 0.0 if gg is None else self._norm(gg)
        guard_weight = 0.0 if gg is None else self.config.guard_scale
        guard_cap = float("inf")
        if gg is not None and ng > self.config.min_norm:
            guard_cap = self.config.guard_ratio_max * nf / ng
            guard_weight = min(self.config.guard_scale, guard_cap)

        report = self._metrics_for_vectors(
            gf, gp, gg, weight, guard_weight
        )
        eps = self.config.min_norm
        report.update({
            "balance/g_scale_raw": ns,
            "balance/g_scale_weighted": scale_multiplier * ns,
            "balance/ratio_scale_fm": scale_multiplier * ns / max(nf, eps),
            "balance/scale_multiplier": scale_multiplier,
            "balance/scale_ratio_cap_hit": float(
                gs is not None and scale_multiplier < 0.999999
            ),
            "balance/guard_weight": guard_weight,
            "balance/guard_ratio_cap_hit": float(
                gg is not None and guard_weight + 1e-12 < self.config.guard_scale
            ),
        })

        for name, start, end in self.groups:
            report.update(self._metrics_for_vectors(
                gf[start:end],
                gp[start:end],
                None if gg is None else gg[start:end],
                weight,
                guard_weight,
                "/" + name,
            ))
            if gs is not None:
                nf_g = self._norm(gf[start:end])
                ns_g = self._norm(gs[start:end])
                report["balance/ratio_scale_fm/" + name] = (
                    scale_multiplier * ns_g / max(nf_g, eps)
                )

        self.active_count += 1
        ratio = report["balance/ratio_projection_fm"]
        weak_threshold = 0.5 * self.config.target
        weak = (
            phase_step >= self.config.warmup_steps
            and ratio < weak_threshold
        )
        self.weak_streak = self.weak_streak + 1 if weak else 0
        if self.weak_streak >= self.config.weak_patience:
            raise RuntimeError(
                "Projection remains too weak after phase warmup: "
                "R_P_F=%.4g threshold=%.4g lambda=%.4g lambda_ideal=%.4g, "
                "lambda_max=%.4g for %d consecutive active updates. "
                "No optimizer step was taken for this batch."
                % (ratio, weak_threshold, weight, ideal, self.config.max_weight,
                   self.weak_streak)
            )

        report.update({
            "balance/phase_step": float(phase_step),
            "balance/lambda": weight,
            "balance/lambda_ema": estimate,
            "balance/lambda_ideal": ideal,
            "balance/ratio_target": self.config.target,
            "balance/ramp": ramp,
            "balance/lambda_max_hit": float(estimate >= self.config.max_weight),
            "balance/ratio_cap_hit": float(nominal > cap),
            "balance/group_ratio_cap_hit": float(nominal > group_cap),
            "balance/weak_threshold": weak_threshold,
            "balance/weak_streak": float(self.weak_streak),
        })

        gf.add_(gp, alpha=weight)
        if gs is not None:
            gf.add_(gs, alpha=scale_multiplier)
        if gg is not None:
            gf.add_(gg, alpha=guard_weight)
        self._finite_everywhere(gf, "combined gradient")
        report["balance/joint_to_fm"] = self._norm(gf) / nf

        flags = used.cpu().tolist()
        for p, (start, end), flag in zip(self.params, self.slices, flags):
            p.grad = gf[start:end].view_as(p).clone() if flag else None

        mean_values = local_values.clone()
        self._all_reduce(mean_values)
        mean_values.div_(self.world)
        report.update({
            "trellis_prior_weight": weight,
            "trellis_prior_projection_loss_weighted": weight * float(mean_values[1]),
            "trellis_prior_scale_loss_weighted": scale_multiplier * float(mean_values[2]),
            "trellis_prior_guard_loss_weighted": guard_weight * float(mean_values[3]),
            "trellis_prior_loss_weighted": (
                weight * float(mean_values[1])
                + scale_multiplier * float(mean_values[2])
                + guard_weight * float(mean_values[3])
            ),
        })
        objective = (
            local_values[0]
            + weight * local_values[1]
            + scale_multiplier * local_values[2]
            + guard_weight * local_values[3]
        )
        return objective, report


def balanced_prior_kwargs(args, prior, global_step, balancer, ordinary_fn):
    if balancer is None:
        return ordinary_fn(args, prior, global_step)
    if prior is None:
        raise ValueError("Balance enabled without a TRELLIS prior")
    if not balancer.active(global_step, args.trellis_prior_every):
        return {}
    return dict(trellis_prior=prior, trellis_prior_weight=args.trellis_prior_weight,
                trellis_prior_rollout_steps=args.trellis_prior_rollout_steps,
                trellis_prior_grad_steps=args.trellis_prior_grad_steps,
                trellis_prior_max_items=args.trellis_prior_max_items,
                trellis_prior_checkpoint=bool(args.trellis_prior_checkpoint),
                trellis_prior_return_loss_terms=True)


def concise_metric_summary(metrics):
    pairs = (("base_mse", "base_mse"), ("teacher_loss_weighted", "teacher"),
             ("teacher_weight_mean", "teacher_w"), ("semantic_usage_loss_weighted", "hub"),
             ("semantic_entropy_12", "H12"), ("semantic_entropy_21", "H21"),
             ("semantic_usage_12_max", "usage12"), ("semantic_usage_21_max", "usage21"),
             ("grad_norm_preclip", "grad_pre"), ("grad_norm_postclip", "grad_post"),
             ("relative_improvement", "slat_rel"), ("pred_target_cosine", "slat_cos"))
    return "".join(" %s=%.6g" % (label, metrics[key])
                   for key, label in pairs if key in metrics)


def prior_console(metrics, step):
    keys = (("balance/phase_step", "phase"), ("trellis_prior_weight", "w"),
            ("balance/lambda_ema", "w_ema"),
            ("balance/ratio_projection_fm", "R_P_F"),
            ("balance/ratio_target", "R_target"),
            ("balance/cos_projection_fm", "cos_P_F"),
            ("balance/ratio_guard_fm", "R_G_F"),
            ("balance/ratio_scale_fm", "R_S_F"),
            ("balance/joint_to_fm", "R_joint_F"),
            ("trellis_prior_projection_loss_weighted", "P_w"),
            ("trellis_prior_scale_loss_weighted", "S_w"),
            ("trellis_prior_guard_loss_weighted", "G_w"),
            ("trellis_prior_sample_endpoint_rms_ratio", "z_ep"),
            ("trellis_prior_scale_ratio", "z_ref"),
            ("trellis_prior_scale_active_fraction", "scale_frac"),
            ("trellis_prior_sample_rms", "z_rms"),
            ("trellis_prior_guard_fraction", "guard_frac"),
            ("trellis_prior_projection_delta_rms", "delta_raw"),
            ("trellis_prior_delta_endpoint_rms_ratio", "delta_ep"),
            ("trellis_prior_projection_clip_fraction", "clip_frac"),
            ("trellis_prior_t_mean", "tau"),
            ("grad_norm_preclip", "grad_pre"), ("grad_norm_postclip", "grad_post"),
            ("balance/lambda_max_hit", "lambda_capped"),
            ("balance/ratio_cap_hit", "ratio_capped"),
            ("balance/group_ratio_cap_hit", "group_capped"),
            ("balance/scale_ratio_cap_hit", "scale_capped"),
            ("balance/guard_ratio_cap_hit", "guard_capped"),
            ("balance/scale_multiplier", "scale_mul"),
            ("balance/guard_weight", "guard_mul"),
            ("trellis_prior_projection_tangent_delta_rms", "delta_tan"),
            ("trellis_prior_projection_radial_fraction", "radial_frac"),
            ("trellis_prior_projection_raw_cosine_z", "cos_dz"),
            ("perf/peak_allocated_gib_rank0", "peak_GiB_r0"),
            ("perf/step_seconds_rank0", "step_s_r0"))
    line = "[PRIOR] step=%d" % step
    for key, label in keys:
        if key in metrics:
            line += " %s=%.5g" % (label, metrics[key])
    for group in ("condition", "lora", "flow_adapter"):
        rk = "balance/ratio_projection_fm/" + group
        ck = "balance/cos_projection_fm/" + group
        if rk in metrics:
            sk = "balance/ratio_scale_fm/" + group
            if sk in metrics:
                line += " %s[R=%.4g,S=%.4g,cos=%.4g]" % (
                    group, metrics[rk], metrics[sk], metrics[ck]
                )
            else:
                line += " %s[R=%.4g,cos=%.4g]" % (group, metrics[rk], metrics[ck])
    if metrics.get("balance/weak_streak", 0) > 0:
        line += " WARNING=PROJECTION_STILL_WEAK"
    return line


def write_step_metrics(writer, metrics, step, active):
    if writer is None:
        return
    for key, value in metrics.items():
        if key.startswith("trellis_prior_"):
            if active:
                writer.add_scalar("train/prior/" + key[len("trellis_prior_"):], value, step)
        elif key.startswith("balance/") or key.startswith("perf/"):
            writer.add_scalar("train/" + key, value, step)
        elif key not in ("grad_norm_preclip", "grad_norm_postclip"):
            writer.add_scalar("train/fm/" + key, value, step)


def append_prior_json(path, metrics, step, epoch):
    record = dict(step=int(step), epoch=int(epoch), gradient_scope="mean_across_DDP_ranks")
    record.update({k: float(v) for k, v in metrics.items()
                   if k.startswith(("balance/", "trellis_prior_", "perf/", "grad_norm_"))})
    with open(path, "a", encoding="utf-8") as out:
        out.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")


def optimizer_grad_norms_batched(optimizer):
    """One host synchronization per optimizer group, not per parameter."""
    group_norms = {}
    total_sq = 0.0
    for index, group in enumerate(optimizer.param_groups):
        norms = []
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            if gradient.is_sparse:
                gradient = gradient.coalesce().values()
            norms.append(torch.linalg.vector_norm(gradient.float()))
        value = float(torch.linalg.vector_norm(torch.stack(norms)).item()) if norms else 0.0
        name = str(group.get("name", "group_%d" % index))
        group_norms[name] = value
        total_sq += value * value
    return math.sqrt(total_sq), group_norms
