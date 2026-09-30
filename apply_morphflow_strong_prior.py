#!/usr/bin/env python3
"""Transactional MorphFlow strong-prior installer. Python >=3.9, standard library only.

Use the TRELLIS container's Python 3.10, NOT the login node's old Python.
  python3 apply_morphflow_strong_prior.py --check
  python3 apply_morphflow_strong_prior.py --apply
Backups live under the repository's git directory, never as tracked .bak files.
"""
import argparse
import ast
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

VERSION = 'mf-prior-balance-v1'
BASE_COMMIT = '52c05c56d9d9a074610d51ab803ab1788ee70484'
PAYLOADS = {'modules/prior_gradient_balance.py': '"""Measured projection/FM gradient balance for MorphFlow\'s single-node DDP run.\n\nActive updates: forward AND all backwards run inside DDP.no_sync(). Gradients\nare accumulated with backward(), packed as FP32, averaged across ranks, and\nthen combined exactly once. No autograd.grad() is used on the DDP model.\nSupported: ordinary AdamW, FP32 parameters, BF16/no AMP, no accumulation,\nnon-reentrant activation checkpointing. FSDP/DeepSpeed/FP16 scaler are rejected.\n"""\nfrom dataclasses import asdict, dataclass\nimport json\nimport math\nimport time\n\nimport torch\nimport torch.distributed as dist\n\nPATCH_VERSION = "mf-prior-balance-v1"\n\n\ndef add_balance_args(parser):\n    g = parser.add_argument_group("Measured TRELLIS projection gradient balance")\n    g.add_argument("--trellis_prior_grad_balance", type=int, choices=[0, 1], default=0)\n    g.add_argument("--trellis_prior_ratio_target", type=float, default=0.5)\n    g.add_argument("--trellis_prior_ratio_max", type=float, default=1.0)\n    g.add_argument("--trellis_prior_balance_ema", type=float, default=0.95)\n    g.add_argument("--trellis_prior_lambda_min", type=float, default=0.01)\n    g.add_argument("--trellis_prior_lambda_max", type=float, default=100.0)\n    g.add_argument("--trellis_prior_phase_start_weight", type=float, default=0.1)\n    g.add_argument("--trellis_prior_phase_warmup_steps", type=int, default=500)\n    g.add_argument("--trellis_prior_guard_scale", type=float, default=0.1)\n    g.add_argument("--trellis_prior_reset_phase", type=int, choices=[0, 1], default=0)\n    g.add_argument("--trellis_prior_weak_patience", type=int, default=25,\n                   help="Abort after this many consecutive weak active updates after phase warmup.")\n    g.add_argument("--trellis_prior_log_every", type=int, default=20,\n                   help="Console interval in ACTIVE prior applications, not FM updates.")\n    g.add_argument("--trellis_prior_phase_eval_steps", default="500,1500,3000",\n                   help="Additional evaluation-only checkpoints at new-phase offsets.")\n\n\ndef validate_balance_args(args):\n    if not getattr(args, "trellis_prior_grad_balance", 0):\n        return\n    finite_positive = ("trellis_prior_ratio_target", "trellis_prior_ratio_max",\n                       "trellis_prior_lambda_min", "trellis_prior_lambda_max")\n    for name in finite_positive:\n        value = float(getattr(args, name))\n        if not math.isfinite(value) or value <= 0:\n            raise ValueError("--%s must be finite and positive" % name)\n    for name in ("trellis_prior_phase_start_weight", "trellis_prior_guard_scale"):\n        value = float(getattr(args, name))\n        if not math.isfinite(value) or value < 0:\n            raise ValueError("--%s must be finite and nonnegative" % name)\n    if not 0 <= args.trellis_prior_balance_ema < 1:\n        raise ValueError("--trellis_prior_balance_ema must be in [0,1)")\n    if args.trellis_prior_ratio_target > args.trellis_prior_ratio_max:\n        raise ValueError("ratio_target cannot exceed ratio_max")\n    if args.trellis_prior_lambda_min > args.trellis_prior_lambda_max:\n        raise ValueError("lambda_min cannot exceed lambda_max")\n    if args.trellis_prior_weak_patience < 1:\n        raise ValueError("weak_patience must be >=1")\n    if args.trellis_prior_phase_warmup_steps < 0 or args.trellis_prior_log_every < 1:\n        raise ValueError("Phase warmup must be >=0 and prior log interval >=1")\n    parse_phase_eval_steps(args.trellis_prior_phase_eval_steps)\n    if (args.flow_target != "ss" or args.ss_flow_arch != "standard"\n            or args.trellis_prior_weight <= 0):\n        raise ValueError("Gradient balancing requires standard SS and prior_weight > 0")\n    if args.endpoint_loss_weight != 0 or args.symmetry_loss_weight != 0:\n        raise ValueError("This experiment requires endpoint/symmetry losses disabled")\n    if args.semantic_cycle_loss_weight != 0:\n        raise ValueError("Keep semantic cycle loss disabled in this experiment")\n\n\ndef parse_phase_eval_steps(value):\n    if not str(value).strip():\n        return set()\n    steps = {int(x.strip()) for x in str(value).split(",") if x.strip()}\n    if any(x <= 0 for x in steps):\n        raise ValueError("Phase evaluation checkpoint offsets must be positive")\n    return steps\n\n\ndef validate_balance_runtime(accelerator, model):\n    if accelerator.mixed_precision not in ("bf16", "no"):\n        raise ValueError("Measured gradient balance supports mixed_precision=bf16 or no only")\n    if getattr(accelerator, "scaler", None) is not None:\n        raise ValueError("AMP GradScaler is not supported by measured gradient balance")\n    if getattr(accelerator, "gradient_accumulation_steps", 1) != 1:\n        raise ValueError("Measured balance currently requires accumulation_steps=1")\n    kind = str(getattr(accelerator, "distributed_type", "NO"))\n    if not any(kind.endswith(x) for x in (".NO", ".MULTI_GPU", ".MULTI_CPU")) and kind not in ("NO", "MULTI_GPU", "MULTI_CPU"):\n        raise ValueError("Only ordinary DDP is supported, not %s" % kind)\n    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:\n        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):\n            raise TypeError("Expected a standard DDP model after accelerator.prepare()")\n\n\n@dataclass(frozen=True)\nclass BalanceConfig:\n    target: float = 0.5\n    max_ratio: float = 1.0\n    ema: float = 0.95\n    min_weight: float = 0.01\n    max_weight: float = 100.0\n    start_weight: float = 0.1\n    warmup_steps: int = 500\n    guard_scale: float = 0.1\n    min_norm: float = 1e-12\n    weak_patience: int = 25\n\n    @classmethod\n    def from_args(cls, args):\n        return cls(target=args.trellis_prior_ratio_target,\n                   max_ratio=args.trellis_prior_ratio_max,\n                   ema=args.trellis_prior_balance_ema,\n                   min_weight=args.trellis_prior_lambda_min,\n                   max_weight=args.trellis_prior_lambda_max,\n                   start_weight=args.trellis_prior_phase_start_weight,\n                   warmup_steps=args.trellis_prior_phase_warmup_steps,\n                   guard_scale=args.trellis_prior_guard_scale,\n                   weak_patience=args.trellis_prior_weak_patience)\n\n\nclass PriorGradientBalancer:\n    def __init__(self, optimizer, config, phase_start, process_group=None):\n        self.config = config\n        self.phase_start = int(phase_start)\n        self.process_group = process_group\n        self.active_count = 0\n        self.weak_streak = 0\n        self.ema_log_weight = None\n        self.params = []\n        self.slices = []\n        self.groups = []\n        seen = set()\n        offset = 0\n        for i, group in enumerate(optimizer.param_groups):\n            name = str(group.get("name", "group_%d" % i))\n            if any(g[0] == name for g in self.groups):\n                raise ValueError("Optimizer group names must be unique")\n            start = offset\n            for p in group["params"]:\n                if not p.requires_grad:\n                    continue\n                if id(p) in seen:\n                    raise ValueError("Duplicate trainable optimizer parameter")\n                if p.dtype != torch.float32:\n                    raise ValueError("Keep trainable parameters FP32; use BF16 autocast")\n                seen.add(id(p))\n                self.params.append(p)\n                self.slices.append((offset, offset + p.numel()))\n                offset += p.numel()\n            self.groups.append((name, start, offset))\n        if not self.params:\n            raise ValueError("No trainable parameters to balance")\n        self.device = self.params[0].device\n        if any(p.device != self.device for p in self.params):\n            raise ValueError("One device per process is required")\n        self.numel = offset\n        self.world = (dist.get_world_size(process_group)\n                      if dist.is_available() and dist.is_initialized() else 1)\n        self.chunk_elems = 2 * 1024 * 1024  # 8 MiB FP32 communication chunks\n        self._assert_layout_matches_ranks()\n\n    def _assert_layout_matches_ranks(self):\n        if self.world == 1:\n            return\n        # Constructor-only metadata check prevents mismatched collectives later.\n        layouts = [None] * self.world\n        layout = (self.numel, [(tuple(p.shape), str(p.dtype)) for p in self.params],\n                  self.groups)\n        dist.all_gather_object(layouts, layout, group=self.process_group)\n        if any(other != layout for other in layouts):\n            raise RuntimeError("Trainable parameter layout differs across DDP ranks")\n\n    def state_dict(self):\n        return dict(version=PATCH_VERSION, config=asdict(self.config),\n                    phase_start=self.phase_start, active_count=self.active_count,\n                    weak_streak=self.weak_streak, ema_log_weight=self.ema_log_weight)\n\n    def load_state_dict(self, state):\n        if state.get("version") != PATCH_VERSION:\n            raise ValueError("Unsupported prior balance checkpoint version")\n        if state.get("config") != asdict(self.config):\n            raise ValueError("Balance settings changed: use reset_phase=1 for a NEW experiment")\n        self.phase_start = int(state["phase_start"])\n        self.active_count = int(state["active_count"])\n        self.weak_streak = int(state["weak_streak"])\n        value = state["ema_log_weight"]\n        self.ema_log_weight = None if value is None else float(value)\n\n    def active(self, global_step, every):\n        return (int(global_step) - self.phase_start) % int(every) == 0\n\n    def _all_reduce(self, tensor, op=dist.ReduceOp.SUM):\n        if self.world > 1:\n            dist.all_reduce(tensor, op=op, group=self.process_group)\n        return tensor\n\n    def _finite_everywhere(self, tensor, label):\n        bad = (~torch.isfinite(tensor).all()).to(dtype=torch.int32).reshape(1)\n        self._all_reduce(bad, dist.ReduceOp.MAX)\n        if bad.item():\n            raise FloatingPointError("Non-finite %s on at least one rank; optimizer not stepped" % label)\n\n    def _clear(self):\n        for p in self.params:\n            p.grad = None\n\n    def _capture_global_gradient(self, loss, accelerator, retain_graph):\n        self._clear()\n        accelerator.backward(loss, retain_graph=retain_graph)\n        flat = torch.zeros(self.numel, dtype=torch.float32, device=self.device)\n        present = torch.zeros(len(self.params), dtype=torch.int32, device=self.device)\n        for i, (p, (start, end)) in enumerate(zip(self.params, self.slices)):\n            if p.grad is not None:\n                grad = p.grad.detach()\n                if grad.is_sparse:\n                    grad = grad.to_dense()\n                flat[start:end].copy_(grad.reshape(-1))\n                present[i] = 1\n        self._finite_everywhere(flat, "component gradient")\n        for start in range(0, self.numel, self.chunk_elems):\n            part = flat[start:start + self.chunk_elems]\n            self._all_reduce(part)\n        flat.div_(self.world)\n        self._all_reduce(present, dist.ReduceOp.MAX)\n        self._finite_everywhere(flat, "averaged component gradient")\n        self._clear()\n        return flat, present\n\n    @staticmethod\n    def _norm(x):\n        return float(torch.linalg.vector_norm(x).item())\n\n    def _metrics_for_vectors(self, gf, gp, gg, weight, suffix=""):\n        nf, np_ = self._norm(gf), self._norm(gp)\n        ng = 0.0 if gg is None else self._norm(gg)\n        dot = float(torch.dot(gf, gp).item())\n        eps = self.config.min_norm\n        cosine = max(-1.0, min(1.0, dot / max(nf * np_, eps)))\n        return {\n            "balance/g_fm" + suffix: nf,\n            "balance/g_projection_raw" + suffix: np_,\n            "balance/g_projection_weighted" + suffix: weight * np_,\n            "balance/ratio_projection_fm" + suffix: weight * np_ / max(nf, eps),\n            "balance/cos_projection_fm" + suffix: cosine,\n            "balance/cos_valid" + suffix: float(nf > eps and np_ > eps),\n            "balance/g_guard_weighted" + suffix: self.config.guard_scale * ng,\n            "balance/ratio_guard_fm" + suffix: self.config.guard_scale * ng / max(nf, eps),\n        }\n\n    def backward_parts(self, parts, accelerator, model, global_step):\n        """Return detached local objective and GLOBAL gradient diagnostics.\n\n        Call inside accelerator.no_sync(model), covering its forward too.\n        FM and rollout have separate forwards; retain_graph on FM also supports\n        shared-subgraph unit tests. Projection and guard share the rollout.\n        """\n        if set(parts) != {"fm", "projection", "guard"}:\n            raise ValueError("Expected live scalar losses: fm, projection, guard")\n        if hasattr(model, "require_backward_grad_sync") and model.require_backward_grad_sync:\n            raise RuntimeError("Forward and split backwards MUST be inside no_sync")\n        if getattr(accelerator, "scaler", None) is not None:\n            raise ValueError("Do not use a GradScaler in split gradient mode")\n        for name, value in parts.items():\n            if value.ndim != 0 or not value.requires_grad:\n                raise ValueError("%s must be a scalar attached to the student graph" % name)\n        local_values = torch.stack([parts[n].detach().float() for n in ("fm", "projection", "guard")])\n        self._finite_everywhere(local_values, "loss")\n        guard_flag = (local_values[2].abs() > 0).to(torch.int32).reshape(1)\n        self._all_reduce(guard_flag, dist.ReduceOp.MAX)\n        guard_on = bool(guard_flag.item()) and self.config.guard_scale > 0\n\n        gf, used = self._capture_global_gradient(parts["fm"], accelerator, True)\n        gp, used_p = self._capture_global_gradient(parts["projection"], accelerator, guard_on)\n        used.copy_(torch.maximum(used, used_p))\n        gg = None\n        if guard_on:\n            gg, used_g = self._capture_global_gradient(parts["guard"], accelerator, False)\n            used.copy_(torch.maximum(used, used_g))\n        nf, np_ = self._norm(gf), self._norm(gp)\n        if nf <= self.config.min_norm or np_ <= self.config.min_norm:\n            raise RuntimeError("Cannot calibrate: gFM=%.3e gProjection=%.3e. Check gradient connectivity." % (nf, np_))\n\n        ideal = self.config.target * nf / np_\n        log_ideal = math.log(ideal)\n        if self.ema_log_weight is None:\n            self.ema_log_weight = log_ideal\n        else:\n            self.ema_log_weight = (self.config.ema * self.ema_log_weight\n                                   + (1 - self.config.ema) * log_ideal)\n        estimate = math.exp(self.ema_log_weight)\n        estimate_bounded = min(self.config.max_weight, max(self.config.min_weight, estimate))\n        phase_step = int(global_step) + 1 - self.phase_start\n        if phase_step <= 0:\n            raise ValueError("global_step precedes new phase start")\n        ramp = min(1.0, phase_step / max(1, self.config.warmup_steps))\n        nominal = (1 - ramp) * self.config.start_weight + ramp * estimate_bounded\n        # Cap instantaneous projection norm, not just an EMA estimate.\n        cap = self.config.max_ratio * nf / np_\n        weight = min(nominal, cap)\n        report = self._metrics_for_vectors(gf, gp, gg, weight)\n        for name, start, end in self.groups:\n            report.update(self._metrics_for_vectors(\n                gf[start:end], gp[start:end], None if gg is None else gg[start:end],\n                weight, "/" + name))\n        self.active_count += 1\n        ratio = report["balance/ratio_projection_fm"]\n        weak = phase_step >= self.config.warmup_steps and ratio < 0.3\n        self.weak_streak = self.weak_streak + 1 if weak else 0\n        if self.weak_streak >= self.config.weak_patience:\n            raise RuntimeError(\n                "Projection remains too weak after phase warmup: "\n                "R_P_F=%.4g, lambda=%.4g, lambda_ideal=%.4g, "\n                "lambda_max=%.4g for %d consecutive active updates. "\n                "No optimizer step was taken for this batch. Inspect diagnostics "\n                "before changing the maximum coefficient." % (\n                    ratio, weight, ideal, self.config.max_weight, self.weak_streak)\n            )\n        report.update({\n            "balance/phase_step": float(phase_step),\n            "balance/lambda": weight,\n            "balance/lambda_ema": estimate,\n            "balance/lambda_ideal": ideal,\n            "balance/ratio_target": self.config.target,\n            "balance/ramp": ramp,\n            "balance/lambda_max_hit": float(estimate >= self.config.max_weight),\n            "balance/ratio_cap_hit": float(nominal > cap),\n            "balance/weak_streak": float(self.weak_streak),\n        })\n        gf.add_(gp, alpha=weight)\n        if gg is not None:\n            gf.add_(gg, alpha=self.config.guard_scale)\n        self._finite_everywhere(gf, "combined gradient")\n        report["balance/joint_to_fm"] = self._norm(gf) / nf\n        flags = used.cpu().tolist()\n        for p, (start, end), flag in zip(self.params, self.slices, flags):\n            # Preserve grad=None for globally unused parameters (AdamW semantics).\n            p.grad = gf[start:end].view_as(p).clone() if flag else None\n\n        mean_values = local_values.clone()\n        self._all_reduce(mean_values)\n        mean_values.div_(self.world)\n        report.update({\n            "trellis_prior_weight": weight,\n            "trellis_prior_projection_loss_weighted": weight * float(mean_values[1]),\n            "trellis_prior_guard_loss_weighted": self.config.guard_scale * float(mean_values[2]),\n            "trellis_prior_loss_weighted": (weight * float(mean_values[1])\n                                            + self.config.guard_scale * float(mean_values[2])),\n        })\n        # Only for logging; gradients have already been installed above.\n        objective = local_values[0] + weight * local_values[1] + self.config.guard_scale * local_values[2]\n        return objective, report\n\n\ndef balanced_prior_kwargs(args, prior, global_step, balancer, ordinary_fn):\n    if balancer is None:\n        return ordinary_fn(args, prior, global_step)\n    if prior is None:\n        raise ValueError("Balance enabled without a TRELLIS prior")\n    if not balancer.active(global_step, args.trellis_prior_every):\n        return {}\n    return dict(trellis_prior=prior, trellis_prior_weight=args.trellis_prior_weight,\n                trellis_prior_rollout_steps=args.trellis_prior_rollout_steps,\n                trellis_prior_grad_steps=args.trellis_prior_grad_steps,\n                trellis_prior_max_items=args.trellis_prior_max_items,\n                trellis_prior_checkpoint=bool(args.trellis_prior_checkpoint),\n                trellis_prior_return_loss_terms=True)\n\n\ndef concise_metric_summary(metrics):\n    pairs = (("base_mse", "base_mse"), ("teacher_loss_weighted", "teacher"),\n             ("teacher_weight_mean", "teacher_w"), ("semantic_usage_loss_weighted", "hub"),\n             ("semantic_entropy_12", "H12"), ("semantic_entropy_21", "H21"),\n             ("semantic_usage_12_max", "usage12"), ("semantic_usage_21_max", "usage21"),\n             ("grad_norm_preclip", "grad_pre"), ("grad_norm_postclip", "grad_post"),\n             ("relative_improvement", "slat_rel"), ("pred_target_cosine", "slat_cos"))\n    return "".join(" %s=%.6g" % (label, metrics[key])\n                   for key, label in pairs if key in metrics)\n\n\ndef prior_console(metrics, step):\n    keys = (("balance/phase_step", "phase"), ("trellis_prior_weight", "w"),\n            ("balance/lambda_ema", "w_ema"),\n            ("balance/ratio_projection_fm", "R_P_F"),\n            ("balance/ratio_target", "R_target"),\n            ("balance/cos_projection_fm", "cos_P_F"),\n            ("balance/ratio_guard_fm", "R_G_F"),\n            ("balance/joint_to_fm", "R_joint_F"),\n            ("trellis_prior_projection_loss_weighted", "P_w"),\n            ("trellis_prior_guard_loss_weighted", "G_w"),\n            ("trellis_prior_sample_endpoint_rms_ratio", "z_ep"),\n            ("trellis_prior_sample_rms", "z_rms"),\n            ("trellis_prior_guard_fraction", "guard_frac"),\n            ("trellis_prior_projection_delta_rms", "delta_raw"),\n            ("trellis_prior_delta_endpoint_rms_ratio", "delta_ep"),\n            ("trellis_prior_projection_clip_fraction", "clip_frac"),\n            ("trellis_prior_t_mean", "tau"),\n            ("grad_norm_preclip", "grad_pre"), ("grad_norm_postclip", "grad_post"),\n            ("balance/lambda_max_hit", "lambda_capped"),\n            ("balance/ratio_cap_hit", "ratio_capped"),\n            ("perf/peak_allocated_gib_rank0", "peak_GiB_r0"),\n            ("perf/step_seconds_rank0", "step_s_r0"))\n    line = "[PRIOR] step=%d" % step\n    for key, label in keys:\n        if key in metrics:\n            line += " %s=%.5g" % (label, metrics[key])\n    for group in ("condition", "lora", "flow_adapter"):\n        rk = "balance/ratio_projection_fm/" + group\n        ck = "balance/cos_projection_fm/" + group\n        if rk in metrics:\n            line += " %s[R=%.4g,cos=%.4g]" % (group, metrics[rk], metrics[ck])\n    if metrics.get("balance/weak_streak", 0) > 0:\n        line += " WARNING=PROJECTION_STILL_WEAK"\n    return line\n\n\ndef write_step_metrics(writer, metrics, step, active):\n    if writer is None:\n        return\n    for key, value in metrics.items():\n        if key.startswith("trellis_prior_"):\n            if active:\n                writer.add_scalar("train/prior/" + key[len("trellis_prior_"):], value, step)\n        elif key.startswith("balance/") or key.startswith("perf/"):\n            writer.add_scalar("train/" + key, value, step)\n        elif key not in ("grad_norm_preclip", "grad_norm_postclip"):\n            writer.add_scalar("train/fm/" + key, value, step)\n\n\ndef append_prior_json(path, metrics, step, epoch):\n    record = dict(step=int(step), epoch=int(epoch), gradient_scope="mean_across_DDP_ranks")\n    record.update({k: float(v) for k, v in metrics.items()\n                   if k.startswith(("balance/", "trellis_prior_", "perf/", "grad_norm_"))})\n    with open(path, "a", encoding="utf-8") as out:\n        out.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\\n")\n\n\ndef optimizer_grad_norms_batched(optimizer):\n    """One host synchronization per optimizer group, not per parameter."""\n    group_norms = {}\n    total_sq = 0.0\n    for index, group in enumerate(optimizer.param_groups):\n        norms = []\n        for parameter in group["params"]:\n            if parameter.grad is None:\n                continue\n            gradient = parameter.grad.detach()\n            if gradient.is_sparse:\n                gradient = gradient.coalesce().values()\n            norms.append(torch.linalg.vector_norm(gradient.float()))\n        value = float(torch.linalg.vector_norm(torch.stack(norms)).item()) if norms else 0.0\n        name = str(group.get("name", "group_%d" % index))\n        group_norms[name] = value\n        total_sq += value * value\n    return math.sqrt(total_sq), group_norms\n', 'tests/test_prior_gradient_balance.py': '"""Small CPU tests; no TRELLIS weights, DINO, spconv or GPU are loaded."""\nimport copy\nimport unittest\nfrom types import SimpleNamespace\n\nimport torch\nfrom torch import nn\nfrom torch.utils.checkpoint import checkpoint\n\nfrom modules.prior_gradient_balance import BalanceConfig, PriorGradientBalancer\n\n\nclass FakeAccelerator:\n    scaler = None\n\n    def backward(self, loss, **kwargs):\n        loss.backward(**kwargs)\n\n\nclass ToyModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.a = nn.Linear(3, 4)\n        self.b = nn.Linear(4, 2)\n        self.unused = nn.Parameter(torch.tensor(3.0))\n\n    def forward(self, x, target, checkpointed=False, active_guard=False):\n        # Separate FM and rollout forwards sharing the same parameter leaves.\n        def evaluate(v):\n            return self.b(torch.tanh(self.a(v)))\n        fm = (evaluate(x) - target).square().mean()\n        z = checkpoint(evaluate, x + 0.3, use_reentrant=False) if checkpointed else evaluate(x + 0.3)\n        # Detached pseudo-teacher target; not an endpoint RMS target.\n        projection = 0.5 * (z - (z.detach() + 0.12)).square().mean()\n        guard = torch.relu((10.0 if active_guard else 0.0) - z.square().mean().clamp_min(1e-12).sqrt()).square()\n        return dict(fm=fm, projection=projection, guard=guard)\n\n\ndef setup(config=None):\n    torch.manual_seed(12)\n    net = ToyModel()\n    opt = torch.optim.AdamW([\n        dict(params=list(net.a.parameters()), name=\'condition\'),\n        dict(params=list(net.b.parameters()), name=\'lora\'),\n        dict(params=[net.unused], name=\'flow_adapter\')], lr=1e-3)\n    bal = PriorGradientBalancer(opt, config or BalanceConfig(warmup_steps=0, ema=0), 70000)\n    return net, opt, bal\n\n\nclass BalanceTests(unittest.TestCase):\n    @classmethod\n    def setUpClass(cls):\n        torch.set_num_threads(1)\n\n    def test_ratio_and_weighted_gradient_match_ordinary_backward(self):\n        net, opt, bal = setup()\n        reference = copy.deepcopy(net)\n        x, target = torch.randn(2, 3), torch.randn(2, 2)\n        parts = net(x, target)\n        loss, metrics = bal.backward_parts(parts, FakeAccelerator(), net, 70000)\n        weight = metrics[\'trellis_prior_weight\']\n        ref_parts = reference(x, target)\n        (ref_parts[\'fm\'] + weight * ref_parts[\'projection\']).backward()\n        self.assertAlmostEqual(metrics[\'balance/ratio_projection_fm\'], 0.5, places=6)\n        self.assertFalse(loss.requires_grad)\n        for p, r in zip(net.parameters(), reference.parameters()):\n            if r.grad is None:\n                self.assertIsNone(p.grad)\n            else:\n                torch.testing.assert_close(p.grad, r.grad, atol=2e-6, rtol=2e-5)\n\n    def test_guard_has_independent_coefficient(self):\n        net, opt, bal = setup()\n        reference = copy.deepcopy(net)\n        x, target = torch.randn(2, 3), torch.randn(2, 2)\n        parts = net(x, target, active_guard=True)\n        loss, metrics = bal.backward_parts(parts, FakeAccelerator(), net, 70000)\n        ref_parts = reference(x, target, active_guard=True)\n        (ref_parts[\'fm\'] + metrics[\'trellis_prior_weight\'] * ref_parts[\'projection\']\n         + 0.1 * ref_parts[\'guard\']).backward()\n        self.assertGreater(metrics[\'balance/ratio_guard_fm\'], 0)\n        self.assertAlmostEqual(metrics[\'balance/ratio_projection_fm\'], 0.5, places=6)\n        for p, r in zip(net.parameters(), reference.parameters()):\n            if r.grad is not None:\n                torch.testing.assert_close(p.grad, r.grad, atol=2e-6, rtol=2e-5)\n\n    def test_phase_warmup_uses_new_phase_not_global_step(self):\n        net, opt, bal = setup(BalanceConfig(warmup_steps=500, ema=0))\n        parts = net(torch.randn(2, 3), torch.randn(2, 2))\n        _, metrics = bal.backward_parts(parts, FakeAccelerator(), net, 70000)\n        self.assertAlmostEqual(metrics[\'balance/ramp\'], 1/500)\n        self.assertEqual(metrics[\'balance/phase_step\'], 1)\n        self.assertTrue(bal.active(70000, 2))\n        self.assertFalse(bal.active(70001, 2))\n        self.assertTrue(bal.active(70002, 2))\n\n    def test_checkpointed_bfloat_graphs_with_guard(self):\n        net, opt, bal = setup()\n        with torch.autocast(\'cpu\', dtype=torch.bfloat16, cache_enabled=False):\n            parts = net(torch.randn(2, 3), torch.randn(2, 2), checkpointed=True, active_guard=True)\n        _, metrics = bal.backward_parts(parts, FakeAccelerator(), net, 70010)\n        self.assertAlmostEqual(metrics[\'balance/ratio_projection_fm\'], 0.5, places=6)\n        for p in bal.params:\n            if p.grad is not None:\n                self.assertTrue(torch.isfinite(p.grad).all())\n\n    def test_ratio_cap_and_weak_weight_limit_are_visible(self):\n        net, opt, bal = setup(BalanceConfig(start_weight=1e5, warmup_steps=500, max_ratio=1.0))\n        _, metrics = bal.backward_parts(net(torch.randn(2, 3), torch.randn(2, 2)), FakeAccelerator(), net, 70000)\n        self.assertLessEqual(metrics[\'balance/ratio_projection_fm\'], 1.000001)\n        self.assertEqual(metrics[\'balance/ratio_cap_hit\'], 1.0)\n        net, opt, bal = setup(BalanceConfig(min_weight=1e-6, max_weight=1e-6, warmup_steps=0))\n        _, metrics = bal.backward_parts(net(torch.randn(2, 3), torch.randn(2, 2)), FakeAccelerator(), net, 70000)\n        self.assertEqual(metrics[\'balance/lambda_max_hit\'], 1.0)\n        self.assertEqual(metrics[\'balance/weak_streak\'], 1.0)\n\n    def test_persistent_weak_signal_aborts(self):\n        net, opt, bal = setup(BalanceConfig(min_weight=1e-6, max_weight=1e-6,\n                                          warmup_steps=0, weak_patience=1))\n        with self.assertRaisesRegex(RuntimeError, \'remains too weak\'):\n            bal.backward_parts(net(torch.randn(2, 3), torch.randn(2, 2)),\n                               FakeAccelerator(), net, 70000)\n\n    def test_controller_state_resume(self):\n        net, opt, bal = setup()\n        bal.backward_parts(net(torch.randn(2, 3), torch.randn(2, 2)), FakeAccelerator(), net, 70000)\n        net2, opt2, bal2 = setup()\n        bal2.load_state_dict(bal.state_dict())\n        self.assertEqual(bal.state_dict(), bal2.state_dict())\n\n    def test_zero_gradient_fails_before_optimizer_step(self):\n        net, opt, bal = setup()\n        parts = net(torch.randn(2, 3), torch.randn(2, 2))\n        parts[\'projection\'] = parts[\'projection\'] * 0\n        with self.assertRaisesRegex(RuntimeError, \'connectivity\'):\n            bal.backward_parts(parts, FakeAccelerator(), net, 70000)\n\n    def test_shape_grid_and_clipped_target_mean_reduction(self):\n        # Real SS grid size, distinct endpoint scale per item. Targets detached.\n        z = nn.Parameter(torch.randn(2, 8, 16, 16, 16))\n        opt = torch.optim.SGD([dict(params=[z], name=\'condition\')], lr=0.1)\n        bal = PriorGradientBalancer(opt, BalanceConfig(warmup_steps=0, ema=0), 70000)\n        raw = torch.randn_like(z)\n        raw_rms = raw.flatten(1).square().mean(1).sqrt()\n        max_rms = torch.tensor([0.6, 0.8]) * 0.15\n        delta = raw * (max_rms / raw_rms).clamp(max=1).view(2,1,1,1,1)\n        parts = dict(fm=z.square().mean(),\n                     projection=0.5*(z-(z.detach()+delta)).square().mean(),\n                     guard=z.square().mean()*0)\n        _, metrics = bal.backward_parts(parts, FakeAccelerator(), nn.Identity(), 70000)\n        self.assertAlmostEqual(metrics[\'balance/ratio_projection_fm\'], 0.5, places=6)\n        self.assertEqual(z.grad.shape, z.shape)\n\n\nif __name__ == \'__main__\':\n    unittest.main()\n', 'tools/smoke_prior_balance_ddp.py': '"""Run: python3 -m torch.distributed.run --standalone --nproc_per_node=2 tools/smoke_prior_balance_ddp.py\n\nCPU/Gloo, tiny model, five alternating split/ordinary updates. Compares global\ncomponent gradients, weighted updates and AdamW states with a serial reference.\nNo real TRELLIS assets or GPU allocation required. Add --bf16 for CPU autocast.\n"""\nimport argparse\nimport copy\nfrom contextlib import nullcontext\nfrom pathlib import Path\nimport os\nimport sys\n\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\nimport torch\nfrom torch import nn\nimport torch.distributed as dist\nfrom torch.utils.checkpoint import checkpoint\nfrom modules.prior_gradient_balance import BalanceConfig, PriorGradientBalancer\n\n\nclass AcceleratorStub:\n    scaler = None\n    def backward(self, loss, **kwargs):\n        loss.backward(**kwargs)\n\n\nclass SmallModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.condition = nn.Linear(3, 4)\n        self.lora = nn.Linear(4, 2)\n\n    def forward(self, x, target, threshold, split):\n        def apply(v):\n            return self.lora(torch.tanh(self.condition(v)))\n        fm = (apply(x) - target).float().square().mean()\n        if not split:\n            return fm\n        z = checkpoint(apply, x + 0.2, use_reentrant=False).float()\n        projection = 0.5 * (z - (z.detach() + 0.13)).square().mean()\n        guard = torch.relu(threshold - z.square().mean(1).clamp_min(1e-12).sqrt()).square().mean()\n        return dict(fm=fm, projection=projection, guard=guard)\n\n\ndef main():\n    ap = argparse.ArgumentParser()\n    ap.add_argument(\'--bf16\', action=\'store_true\')\n    args = ap.parse_args()\n    torch.set_num_threads(1)\n    dist.init_process_group(\'gloo\')\n    rank, world = dist.get_rank(), dist.get_world_size()\n    torch.manual_seed(340)\n    net = SmallModel()\n    reference = copy.deepcopy(net)\n    ddp = torch.nn.parallel.DistributedDataParallel(net, find_unused_parameters=False)\n    groups = lambda m: [dict(params=list(m.condition.parameters()), name=\'condition\'),\n                        dict(params=list(m.lora.parameters()), name=\'lora\')]\n    opt = torch.optim.AdamW(groups(net), lr=1e-3)\n    refopt = torch.optim.AdamW(groups(reference), lr=1e-3)\n    controller = PriorGradientBalancer(opt, BalanceConfig(warmup_steps=0, ema=0), 70000)\n    tol = dict(atol=1e-2, rtol=2e-2) if args.bf16 else dict(atol=3e-6, rtol=3e-5)\n    for i in range(6):\n        all_x = torch.arange(world * 6, dtype=torch.float32).reshape(world * 2, 3) / 13 + i * 0.1\n        all_y = torch.sin(all_x[:, :2])\n        thresholds = torch.zeros(world * 2)\n        if i == 2:  # Guard active only on one rank: collective branch must agree.\n            thresholds[-2:] = 2.0\n        x = all_x[rank*2:(rank+1)*2]\n        y = all_y[rank*2:(rank+1)*2]\n        threshold = thresholds[rank*2:(rank+1)*2]\n        active = i % 2 == 0\n        opt.zero_grad(set_to_none=True)\n        refopt.zero_grad(set_to_none=True)\n        context = ddp.no_sync() if active else nullcontext()\n        with context:\n            with torch.autocast(\'cpu\', enabled=args.bf16, dtype=torch.bfloat16, cache_enabled=False):\n                output = ddp(x, y, threshold, active)\n            if active:\n                _, metrics = controller.backward_parts(output, AcceleratorStub(), ddp, 70000+i)\n                weight = metrics[\'trellis_prior_weight\']\n                assert abs(metrics[\'balance/ratio_projection_fm\'] - 0.5) < 1e-5\n            else:\n                output.backward()\n                weight = 0.0\n        with torch.autocast(\'cpu\', enabled=args.bf16, dtype=torch.bfloat16, cache_enabled=False):\n            ref_out = reference(all_x, all_y, thresholds, active)\n        if active:\n            (ref_out[\'fm\'] + weight*ref_out[\'projection\'] + 0.1*ref_out[\'guard\']).backward()\n        else:\n            ref_out.backward()\n        for p, q in zip(net.parameters(), reference.parameters()):\n            torch.testing.assert_close(p.grad, q.grad, **tol)\n        opt.step()\n        refopt.step()\n        for p, q in zip(net.parameters(), reference.parameters()):\n            torch.testing.assert_close(p, q, **tol)\n        for actual, expected in zip(opt.state.values(), refopt.state.values()):\n            for key in (\'step\', \'exp_avg\', \'exp_avg_sq\'):\n                torch.testing.assert_close(actual[key], expected[key], **tol)\n        if i == 3:\n            state = controller.state_dict()\n            controller = PriorGradientBalancer(opt, BalanceConfig(warmup_steps=0, ema=0), 70000)\n            controller.load_state_dict(state)\n        if rank == 0:\n            print(\'PASS update=%d split=%s guard=%s\' % (i+1, active, i == 2), flush=True)\n    dist.barrier()\n    if rank == 0:\n        print(\'OK: global gradients, alternating DDP updates, guard branch and controller resume\', flush=True)\n    dist.destroy_process_group()\n\n\nif __name__ == \'__main__\':\n    main()\n', 'tools/check_strong_checkpoint.py': '"""CPU-only validation of the immutable epoch-14 restart checkpoint."""\nimport argparse\nfrom pathlib import Path\nimport torch\n\n\ndef main():\n    ap = argparse.ArgumentParser()\n    ap.add_argument(\'checkpoint\')\n    args = ap.parse_args()\n    path = Path(args.checkpoint)\n    if not path.is_file():\n        raise SystemExit(\'Checkpoint missing: \' + str(path))\n    ckpt = torch.load(path, map_location=\'cpu\', weights_only=False)\n    if not isinstance(ckpt, dict):\n        raise SystemExit(\'Expected a full MorphFlow checkpoint dictionary\')\n    epoch, step = int(ckpt.get(\'epoch\', -1)), int(ckpt.get(\'step\', -1))\n    if (epoch, step) != (14, 70000):\n        raise SystemExit(\'Wrong checkpoint: epoch=%d step=%d; require epoch=14 step=70000\' % (epoch, step))\n    if ckpt.get(\'evaluation_only\', False):\n        raise SystemExit(\'An evaluation-only snapshot cannot be resumed\')\n    for field in (\'model\', \'optimizer\', \'scheduler\'):\n        if ckpt.get(field) is None:\n            raise SystemExit(\'Missing \' + field + \' state\')\n    config = ckpt.get(\'args\', {})\n    if not isinstance(config, dict):\n        config = vars(config)\n    flow = ckpt.get(\'flow_target\', config.get(\'flow_target\'))\n    if flow != \'ss\' or config.get(\'ss_flow_arch\', \'standard\') != \'standard\':\n        raise SystemExit(\'Require a standard SS checkpoint\')\n    if int(config.get(\'use_lora\', 0)) != 1:\n        raise SystemExit(\'This experiment expects the existing LoRA run\')\n    print(\'CHECKPOINT OK:\', path)\n    print(\'epoch:\', epoch, \'| global_step:\', step, \'| val_loss:\', ckpt.get(\'val_loss\'))\n    print(\'model tensors:\', len(ckpt[\'model\']))\n    for index, group in enumerate(ckpt[\'optimizer\'].get(\'param_groups\', [])):\n        print(\'restored group\', group.get(\'name\', index), \'lr=\', group[\'lr\'])\n    print(\'Resume will start at epoch 15; --train_epochs 25 means 11 remaining epochs.\')\n\n\nif __name__ == \'__main__\':\n    main()\n', 'tools/launch_prior_strong.py': '#!/usr/bin/env python3\n"""Submit the new phase from the HOST login shell (no torch needed).\n\npython3 tools/launch_prior_strong.py --dry-run\npython3 tools/launch_prior_strong.py --submit\n"""\nimport argparse\nimport os\nfrom pathlib import Path\nimport shlex\nimport shutil\nimport subprocess\n\n\nDEFAULT_RUN = \'v3_ss_projectionStrong_ep14_r05_c015_g4\'\nARCHIVE = \'/hpc/archive/G_VBD/marco.barezzi/morphflow_runs\'\nDEFAULT_CKPT = (ARCHIVE + \'/v3_ss_trellisProjection_t005_020_test2/checkpoints/\'\n                \'morphflow_epoch_0014_step_0070000.pt\')\n\n\ndef main():\n    parser = argparse.ArgumentParser()\n    parser.add_argument(\'--checkpoint\', default=DEFAULT_CKPT)\n    parser.add_argument(\'--run-name\', default=DEFAULT_RUN)\n    action = parser.add_mutually_exclusive_group(required=True)\n    action.add_argument(\'--dry-run\', action=\'store_true\')\n    action.add_argument(\'--submit\', action=\'store_true\')\n    args = parser.parse_args()\n    root = Path(__file__).resolve().parents[1]\n    checkpoint = Path(args.checkpoint).resolve()\n    if not checkpoint.is_file():\n        raise SystemExit(\'Checkpoint missing: \' + str(checkpoint))\n    if \'/\' in args.run_name or args.run_name in (\'\', \'.\', \'..\'):\n        raise SystemExit(\'run-name must be a directory name, not a path\')\n    output = Path(ARCHIVE) / args.run_name\n    if (output / \'checkpoints\').exists() and any((output / \'checkpoints\').glob(\'*.pt\')):\n        raise SystemExit(\'Output contains checkpoints. Use a new run-name; do not overwrite the experiment.\')\n    launcher = root / \'slurm/train_morphflow_v3_prior_strong.slurm\'\n    if not launcher.is_file():\n        raise SystemExit(\'Apply the patch before submitting\')\n    settings = dict(\n        RUN_NAME=args.run_name, AUTO_RESUME=\'0\', RESUME_FROM=str(checkpoint),\n        TRELLIS_PRIOR_WEIGHT=\'1.0\', TRELLIS_PRIOR_EVERY=\'2\',\n        TRELLIS_PRIOR_ROLLOUT_STEPS=\'8\', TRELLIS_PRIOR_GRAD_STEPS=\'4\',\n        TRELLIS_PRIOR_MAX_ITEMS=\'1\', TRELLIS_PRIOR_CHECKPOINT=\'1\',\n        TRELLIS_PRIOR_T_MIN=\'0.05\', TRELLIS_PRIOR_T_MAX=\'0.20\',\n        TRELLIS_PRIOR_PROJECTION_CLIP_RATIO=\'0.15\',\n        TRELLIS_PRIOR_RMS_GUARD_WEIGHT=\'1.0\',\n        TRELLIS_PRIOR_RMS_GUARD_LOW_RATIO=\'0.25\',\n        TRELLIS_PRIOR_RMS_GUARD_HIGH_RATIO=\'2.0\',\n        TRELLIS_PRIOR_GRAD_BALANCE=\'1\', TRELLIS_PRIOR_RATIO_TARGET=\'0.5\',\n        TRELLIS_PRIOR_RATIO_MAX=\'1.0\', TRELLIS_PRIOR_BALANCE_EMA=\'0.95\',\n        TRELLIS_PRIOR_LAMBDA_MIN=\'0.01\', TRELLIS_PRIOR_LAMBDA_MAX=\'100.0\',\n        TRELLIS_PRIOR_PHASE_START_WEIGHT=\'0.1\',\n        TRELLIS_PRIOR_PHASE_WARMUP_STEPS=\'500\', TRELLIS_PRIOR_GUARD_SCALE=\'0.1\',\n        TRELLIS_PRIOR_RESET_PHASE=\'1\', TRELLIS_PRIOR_LOG_EVERY=\'20\',\n        TRELLIS_PRIOR_WEAK_PATIENCE=\'25\',\n        TRELLIS_PRIOR_PHASE_EVAL_STEPS=\'500,1500,3000\',\n    )\n    env = os.environ.copy()\n    env.update(settings)\n    command = [\'sbatch\', \'--parsable\', \'--export=ALL\', \'--nodelist=wn49\',\n               \'--gres=gpu:l40s_vbd:8\', \'--job-name=mf_ss_prior_strong\', str(launcher)]\n    print(\'Checkpoint:\', checkpoint)\n    print(\'Output:\', output)\n    print(\'Resources: wn49, 8 GPUs. Epochs: 15 through 25.\')\n    print(\' \'.join(shlex.quote(s) for s in command))\n    for key in sorted(settings):\n        print(\'%s=%s\' % (key, settings[key]))\n    if args.submit:\n        if shutil.which(\'sbatch\') is None:\n            raise SystemExit(\'Run this launcher on the HOST, not inside Singularity.\')\n        result = subprocess.run(command, env=env, cwd=str(root), check=True,\n                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,\n                                universal_newlines=True)\n        print(\'SUBMITTED JOB:\', result.stdout.strip())\n        if result.stderr:\n            print(result.stderr.strip())\n\n\nif __name__ == \'__main__\':\n    main()\n', 'STRONG_PRIOR_NOTES.md': "# MorphFlow: measured strong projection prior\n\nBase reviewed: `52c05c56d9d9a074610d51ab803ab1788ee70484`.\n\n## Experimental definition\n\nResume the immutable epoch-14 / step-70000 checkpoint with its optimizer and\nscheduler. The original student architecture and ordinary launcher are unchanged.\nThe dedicated launcher finishes epochs 15 through 25, not 25 additional epochs.\n\nProjection: tau [0.05, 0.20], no CFG, clip 0.15, every 2 updates, one sample/GPU,\n8 Euler steps with 4 differentiated suffix steps, activation checkpointing on.\nThe old teacher/FM weighting, LoRA configuration and learning rates are retained.\n\nOn every active update, measure GLOBAL mean gradients separately for:\n- weighted teacher FM + existing semantic usage regularization;\n- local projection loss (WITHOUT the RMS guard);\n- RMS guard, only when active on any rank.\n\nNo `autograd.grad` calls on DDP. The active forward and backwards are enclosed\nin `no_sync`, then FP32 gradients are explicitly all-reduced and divided by the\nnumber of ranks. Off-prior updates retain the ordinary DDP backward.\n\nUse an EMA in log space of lambda_ideal = 0.5 ||g_FM|| / ||g_projection||.\nRamp from lambda=0.1 to the calibrated estimate over 500 NEW optimizer steps.\nNominal lambda is bounded to [0.01,100]; instantaneous projection/FM norm ratio\nis capped at 1.0. Low ratios below 0.3 after warmup are explicitly warned. After 25 consecutive\nweak active updates training aborts before the next optimizer step, rather than\nsilently running a weak-prior experiment.\nThis is a heuristic balance controller, not the full GradNorm algorithm.\n\nThe guard has a separate fixed external coefficient 0.1. The existing internal\nguard weight remains 1.0. It is never multiplied by the adaptive projection\ncoefficient. This preserves the previous guard scale per application, although\napplications are twice as frequent.\n\nLarge scalar lambda alone is NOT evidence of strong influence. R_P_F, computed\nfrom mean gradients, is the diagnostic. It is not the ratio of AdamW parameter\nupdates. Increasing the ratio can strengthen endpoint bias or worsen geometry.\n\n## Logs\n\nConsole [PRIOR]: phase counter, lambda/EMA, projection/FM ratio, cosine,\nguard/FM ratio, combined/FM ratio, per-group ratios and cosines, guard fraction,\nraw correction, relative clipped correction, clipping fraction, latent scale,\ngradient norm before/after clipping, and rank-0 compute time / CUDA memory peak.\n\n`logs/prior_diagnostics.jsonl` contains every active update. TensorBoard stores\nprior values only on active updates; the fake zero prior series in validation\nand duplicate slat/prior tags are removed. Raw latent statistics are retained\nin the diagnostic data but no longer clutter every console line.\n\nAdditional snapshots at +500, +1500 and +3000 updates are EVALUATION ONLY.\nThe trainer rejects resuming them because dataloader position is not stored.\nResume later phases from best/last/epoch-end checkpoints instead.\n\nThe balancer's state (phase start, EMA, counters and settings) is saved in each\ncheckpoint. Set reset_phase=0 for resuming this same phase. A new phase resets\nthe best-FM comparator in the new output directory, not model/optimizer/scheduler.\nThe old checkpoint stays untouched. Its RNG and dataloader positions were not\nsaved by the original trainer, so this is not a bitwise-exact continuation.\n\n## Supported environment\n\nOrdinary single-device-per-process DDP and AdamW; FP32 trainable parameters;\nBF16 or no autocast; gradient accumulation=1; non-reentrant checkpointing.\nFP16/GradScaler, DeepSpeed and FSDP are explicitly rejected in balanced mode.\n\nExtra FP32 gradient storage is proportional to trainable parameters, not frozen\nTRELLIS weights: normally two packed component vectors, three if guard is active,\nin addition to parameter gradients and DDP buckets. Runtime and real GPU peak\nmust be measured. More backward/communication work is expected on active steps.\n\n## Tests\n\nThe package includes 9 CPU gradient/controller tests and a two-process Gloo\nsmoke test. The smoke test alternates split and ordinary DDP updates and checks\nagainst a serial global-batch reference, including a guard active on one rank,\ncheckpointed forwards, AdamW updates and balancer-state resume.\nRun both tests in the user's PyTorch 2.4 TRELLIS container before training.\nThe development run used PyTorch 2.10 CPU; no full TRELLIS GPU run was performed.\n\nThe installer validates every transformation and syntax before writing; backups\nand a manifest are saved under `.git/morphflow_patches/`, not as tracked bak files.\nThe transformer was exercised on integration fixtures matching the inspected\nsource structure. `--check` additionally validates the actual local files.\n\nDo not interpret clipped projection loss or clip_fraction as a distance-to-\nmanifold metric. Stochastic denoising and endpoint conditioning can produce\nnonzero corrections even for valid samples. Decode fixed held-out SS examples\nand examine central alpha values, endpoint bias and full-pipeline geometry.\n"}


def die(message):
    raise RuntimeError(message)


def clean(text):
    return '\n'.join(line.rstrip() for line in text.splitlines()) + '\n'


def exact(text, old, new, label):
    n = text.count(old)
    if n != 1:
        die('%s: expected one anchor, found %d. No files have been written.' % (label, n))
    return text.replace(old, new, 1)


def find_fn(text, name, cls=None):
    tree = ast.parse(text)
    scope = tree
    if cls:
        candidates = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
        if len(candidates) != 1:
            die('Cannot uniquely locate class ' + cls)
        scope = candidates[0]
    matches = [n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        die('Cannot uniquely locate function ' + str(cls) + '.' + name)
    return matches[0]


def segment(text, node):
    lines = text.splitlines(keepends=True)
    return ''.join(lines[node.lineno-1:node.end_lineno])


def replace_node(text, node, replacement):
    lines = text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1]) + replacement.rstrip() + '\n' + ''.join(lines[node.end_lineno:])


def add_multiline_arg(text, fn_name, argument, cls=None):
    fn = find_fn(text, fn_name, cls)
    lines = text.splitlines(keepends=True)
    end = fn.body[0].lineno - 1
    header = ''.join(lines[fn.lineno-1:end])
    closing = re.search(r'^( *)\)\s*(?:->[^\n]+)?\s*:\s*$', header, re.M)
    if not closing:
        die('Expected multiline signature for ' + fn_name)
    indent = ' ' * (fn.col_offset + 4)
    header = header[:closing.start()] + indent + argument + ',\n' + header[closing.start():]
    return ''.join(lines[:fn.lineno-1]) + header + ''.join(lines[end:])


def rewrite_prior(text):
    text = add_multiline_arg(text, 'forward', 'return_loss_terms=False', 'TrellisSSPrior')
    fn = find_fn(text, 'forward', 'TrellisSSPrior')
    last = fn.body[-1]
    if not isinstance(last, ast.Return) or ast.unparse(last.value) != '(loss, metrics)':
        die('Unexpected prior return statement')
    return replace_node(text, last, '''        if return_loss_terms:
            return loss, metrics, {
                "projection": projection_loss,
                "guard": self.rms_guard_weight * guard_loss,
            }
        return loss, metrics''')


def rewrite_morph(text):
    text = add_multiline_arg(text, 'forward', 'trellis_prior_return_loss_terms=False', 'MorphFlow')
    fn = find_fn(text, 'forward', 'MorphFlow')
    assign = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
              and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
              and n.value.func.id == 'trellis_prior']
    if len(assign) != 1:
        die('Cannot uniquely locate the student call to the prior')
    call = assign[0]
    new_call = copy.deepcopy(call)
    new_call.targets = [ast.Name(id='prior_result', ctx=ast.Store())]
    new_call.value.keywords.append(ast.keyword(arg='return_loss_terms',
                                     value=ast.Name(id='trellis_prior_return_loss_terms', ctx=ast.Load())))
    code = ast.unparse(ast.fix_missing_locations(new_call))
    code = '            ' + code + '''
            if trellis_prior_return_loss_terms:
                prior_term, measured_prior_metrics, prior_live_terms = prior_result
            else:
                prior_term, measured_prior_metrics = prior_result
'''
    text = replace_node(text, call, code)
    text = exact(text, '        prior_term = None\n',
                 '        prior_base_loss = loss\n        prior_live_terms = None\n        prior_term = None\n',
                 'student base loss capture')
    fn = find_fn(text, 'forward', 'MorphFlow')
    last = fn.body[-1]
    if not isinstance(last, ast.Return) or ast.unparse(last.value) != 'loss':
        die('Unexpected MorphFlow.forward return statement')
    return replace_node(text, last, '''        if trellis_prior_return_loss_terms:
            if prior_live_terms is None:
                raise RuntimeError("Split loss requested without an active prior")
            return {
                "fm": prior_base_loss,
                "projection": prior_live_terms["projection"],
                "guard": prior_live_terms["guard"],
            }
        return loss''')


def rewrite_train(text):
    imports = '''from contextlib import nullcontext
import time
from modules.prior_gradient_balance import (
    BalanceConfig, PriorGradientBalancer, add_balance_args,
    validate_balance_args, validate_balance_runtime, balanced_prior_kwargs,
    concise_metric_summary, prior_console, write_step_metrics,
    append_prior_json, parse_phase_eval_steps, optimizer_grad_norms_batched,
)
'''
    text = exact(text, 'import argparse\n', 'import argparse\n' + imports, 'training imports')
    text = exact(text, '    add_trellis_prior_args(parser)\n',
                 '    add_trellis_prior_args(parser)\n    add_balance_args(parser)\n', 'parser')
    text = exact(text, '    validate_trellis_prior_args(args)\n',
                 '    validate_trellis_prior_args(args)\n    validate_balance_args(args)\n', 'validation')
    fn = find_fn(text, 'format_slat_metric_summary')
    text = replace_node(text, fn, '''def format_slat_metric_summary(metrics):
    # No zero-filled prior fields, duplicate total/FM values or inactive losses.
    return concise_metric_summary(metrics)''')

    fn = find_fn(text, 'optimizer_grad_norms')
    text = replace_node(text, fn, '''def optimizer_grad_norms(optimizer):
    return optimizer_grad_norms_batched(optimizer)''')

    # Save controller state while preserving all existing checkpoint fields.
    text = add_multiline_arg(text, 'save_checkpoint', 'prior_balancer=None')
    text = add_multiline_arg(text, 'save_checkpoint', 'evaluation_only=False')
    text = exact(text, '        "args": vars(args),\n', '''        "args": vars(args),
        "trellis_prior_balance_state": (
            prior_balancer.state_dict() if prior_balancer is not None else None
        ),
        "evaluation_only": bool(evaluation_only),
''', 'checkpoint controller state')
    text = exact(text, '        resume_ckpt = load_checkpoint_cpu(args.resume_from)\n',
                 '''        resume_ckpt = load_checkpoint_cpu(args.resume_from)
        if resume_ckpt.get("evaluation_only", False):
            raise ValueError(
                "This is an evaluation-only mid-epoch snapshot. Resume from an "
                "epoch-end best/last/periodic checkpoint instead."
            )
''', 'reject mid-epoch resume')

    # Initialize AFTER the original optimizer/scheduler/global_step restoration.
    anchor = '    if args.lr_scheduler == "plateau" and not optimizer_restored and global_step >= warmup_steps:\n'
    setup = '''    prior_balancer = None
    phase_eval_steps = set()
    if args.trellis_prior_grad_balance:
        if resume_ckpt is None or not optimizer_restored:
            raise RuntimeError(
                "The strong-prior phase must resume an existing student and "
                "restore its optimizer. Use --resume_from and --resume_optimizer 1."
            )
        validate_balance_runtime(accelerator, model)
        prior_balancer = PriorGradientBalancer(
            optimizer, BalanceConfig.from_args(args), global_step,
            process_group=getattr(model, "process_group", None),
        )
        saved_balance = resume_ckpt.get("trellis_prior_balance_state")
        if saved_balance and not args.trellis_prior_reset_phase:
            prior_balancer.load_state_dict(saved_balance)
            accelerator.print("Restored prior balance EMA and phase counter.")
        else:
            # Model/optimizer/scheduler are restored; ONLY prior phase and the
            # best-FM selection for the new output directory start afresh.
            best_val_loss, best_epoch = float("inf"), 0
            accelerator.print(
                f"New measured-prior phase: start_step={global_step}, "
                f"ratio_target={args.trellis_prior_ratio_target}, "
                f"relative_warmup={args.trellis_prior_phase_warmup_steps}, "
                f"guard_scale={args.trellis_prior_guard_scale}"
            )
        phase_eval_steps = parse_phase_eval_steps(args.trellis_prior_phase_eval_steps)
        accelerator.print(
            "Gradient balance: global FP32 mean gradients; "
            "FM includes weighted teacher FM plus configured semantic usage. "
            "Guard is NOT included in projection calibration."
        )
        accelerator.print(
            f"One FP32 gradient buffer: {4*prior_balancer.numel/(1024**3):.3f} GiB; "
            "2 component buffers normally, 3 when the guard is active, "
            "plus parameter gradients and existing DDP buckets."
        )
        accelerator.print(
            "Restored actual learning rates: " + ", ".join(
                f"{g.get('name', i)}={g['lr']:.6g}" for i, g in enumerate(optimizer.param_groups)
            )
        )

'''
    text = exact(text, anchor, setup + anchor, 'relative phase setup')

    # Avoid stale description of the nominal CLI weight in balanced mode.
    text = exact(text, '    if trellis_prior is not None:\n',
                 '    if trellis_prior is not None and prior_balancer is None:\n',
                 'ordinary prior configuration block')

    # Add balancer kwarg to each pre-existing save_checkpoint call (not definition).
    tree = ast.parse(text)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'save_checkpoint']
    if not calls:
        die('No checkpoint save calls found')
    lines = text.splitlines(keepends=True)
    for call in sorted(calls, key=lambda n: n.end_lineno, reverse=True):
        closing_index = call.end_lineno - 1
        indent = re.match(r' *', lines[closing_index]).group(0) + '    '
        lines.insert(closing_index, indent + 'prior_balancer=prior_balancer,\n')
    text = ''.join(lines)

    # Rewrite only the training forward/backward block, using its existing call
    # arguments so architecture/dataset/teacher settings are not reconstructed.
    fn = find_fn(text, 'train')
    withs = [n for n in ast.walk(fn) if isinstance(n, ast.With)
             and any(isinstance(x, ast.Assign) and isinstance(x.value, ast.Call)
                     and isinstance(x.value.func, ast.Name) and x.value.func.id == 'compute_loss'
                     and any(isinstance(t, ast.Name) and t.id == 'loss' for t in x.targets)
                     for x in n.body)]
    if len(withs) != 1:
        die('Cannot uniquely locate the training autocast/compute_loss block')
    block = withs[0]
    forward = segment(text, block)
    forward = exact(forward,
        'trellis_prior_kwargs=trellis_prior_forward_kwargs(args, trellis_prior, global_step),',
        'trellis_prior_kwargs=prior_kwargs,', 'training prior kwargs')
    forward = ''.join('    ' + line if line.strip() else line for line in forward.splitlines(keepends=True))
    lines = text.splitlines(keepends=True)
    stop = block.end_lineno
    # The next nonempty statement must be exactly accelerator.backward(loss).
    j = stop
    while j < len(lines) and not lines[j].strip():
        j += 1
    if lines[j].strip() != 'accelerator.backward(loss)':
        die('Unexpected statement after the training forward')
    newblock = '''            step_started = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            prior_kwargs = balanced_prior_kwargs(
                args, trellis_prior, global_step, prior_balancer,
                trellis_prior_forward_kwargs,
            )
            split_active = bool(prior_kwargs.get("trellis_prior_return_loss_terms", False))
            balance_metrics = {}
            sync_context = accelerator.no_sync(model) if split_active else nullcontext()
            # no_sync must enclose BOTH forward and every split backward.
            with sync_context:
''' + forward + '''                if split_active:
                    loss, balance_metrics = prior_balancer.backward_parts(
                        loss, accelerator, model, global_step,
                    )
                else:
                    accelerator.backward(loss)
'''
    text = ''.join(lines[:block.lineno-1]) + newblock + ''.join(lines[j+1:])
    anchor = '            forward_metrics["grad_norm_preclip"] = grad_norm_preclip\n'
    text = exact(text, anchor, '''            forward_metrics.update(balance_metrics)
            # Values here are actual post-calibration objective values, not the
            # nominal temporary weight used inside MorphFlow.forward.
            forward_metrics["total_loss"] = loss_value
            forward_metrics["perf/step_seconds_rank0"] = time.perf_counter() - step_started
            if torch.cuda.is_available():
                forward_metrics["perf/peak_allocated_gib_rank0"] = (
                    torch.cuda.max_memory_allocated(device) / (1024**3)
                )
''' + anchor, 'merge gradient diagnostics')

    # Replace verbose PRIOR print and add active-only JSONL diagnostics.
    start = text.find('            if prior_active:\n')
    stop = text.find('            if writer is not None:\n', start)
    if start < 0 or stop < 0:
        die('Cannot locate PRIOR logging section')
    text = text[:start] + '''            if prior_active:
                prior_debug_counter += 1
                if accelerator.is_main_process:
                    append_prior_json(
                        os.path.join(logs_dir, "prior_diagnostics.jsonl"),
                        forward_metrics, global_step, epoch,
                    )
                if prior_debug_counter == 1 or prior_debug_counter % args.trellis_prior_log_every == 0:
                    accelerator.print(prior_console(forward_metrics, global_step))

''' + text[stop:]
    old = '''                for metric_name, metric_value in forward_metrics.items():
                    writer.add_scalar(f"train/slat_{metric_name}", metric_value, global_step)
                    if metric_name.startswith("trellis_prior_"):
                        writer.add_scalar(f"train/{metric_name}", metric_value, global_step)
'''
    new = '''                write_step_metrics(writer, forward_metrics, global_step, prior_active)
'''
    text = exact(text, old, new, 'remove duplicate/zero-filled TensorBoard series')
    text = exact(text, '                if forward_metrics:\n',
                 '                if args.flow_target == "slat" and forward_metrics:\n', 'remove fake SS slat postfix')
    old = '''                for metric_name, metric_value in val_forward_metric_avgs.items():
                    writer.add_scalar(f"val/slat_{metric_name}", metric_value, epoch)
'''
    new = '''                for metric_name, metric_value in val_forward_metric_avgs.items():
                    if not metric_name.startswith("trellis_prior_"):
                        writer.add_scalar(f"val/fm/{metric_name}", metric_value, epoch)
'''
    text = exact(text, old, new, 'remove zero prior validation curves')

    anchor = '        epoch_avg = running_loss / max(1, len(loader))\n'
    mid_save = '''            if prior_balancer is not None and global_step - prior_balancer.phase_start in phase_eval_steps:
                accelerator.wait_for_everyone()
                save_checkpoint(
                    accelerator=accelerator, model=model, optimizer=optimizer,
                    scheduler=scheduler, args=args, ckpt_dir=ckpt_dir,
                    epoch=epoch, global_step=global_step, train_loss=avg_loss,
                    val_loss=None, best_val_loss=best_val_loss, best_epoch=best_epoch,
                    prior_balancer=prior_balancer, evaluation_only=True,
                )
                accelerator.wait_for_everyone()

'''
    text = exact(text, anchor, mid_save + anchor, 'phase evaluation snapshots')
    return text


def make_launcher(text):
    extra = {
        'TRELLIS_PRIOR_GRAD_BALANCE': '1',
        'TRELLIS_PRIOR_RATIO_TARGET': '0.5',
        'TRELLIS_PRIOR_RATIO_MAX': '1.0',
        'TRELLIS_PRIOR_BALANCE_EMA': '0.95',
        'TRELLIS_PRIOR_LAMBDA_MIN': '0.01',
        'TRELLIS_PRIOR_LAMBDA_MAX': '100.0',
        'TRELLIS_PRIOR_PHASE_START_WEIGHT': '0.1',
        'TRELLIS_PRIOR_PHASE_WARMUP_STEPS': '500',
        'TRELLIS_PRIOR_GUARD_SCALE': '0.1',
        'TRELLIS_PRIOR_RESET_PHASE': '0',
        'TRELLIS_PRIOR_LOG_EVERY': '20',
        'TRELLIS_PRIOR_WEAK_PATIENCE': '25',
        'TRELLIS_PRIOR_PHASE_EVAL_STEPS': '500,1500,3000',
    }
    overrides = {
        'TRELLIS_PRIOR_WEIGHT': '1.0', 'TRELLIS_PRIOR_EVERY': '2',
        'TRELLIS_PRIOR_ROLLOUT_STEPS': '8', 'TRELLIS_PRIOR_GRAD_STEPS': '4',
        'TRELLIS_PRIOR_MAX_ITEMS': '1', 'TRELLIS_PRIOR_CHECKPOINT': '1',
        'TRELLIS_PRIOR_T_MIN': '0.05', 'TRELLIS_PRIOR_T_MAX': '0.20',
        'TRELLIS_PRIOR_PROJECTION_CLIP_RATIO': '0.15',
    }
    text, n = re.subn(r'^#SBATCH --job-name=.*$', '#SBATCH --job-name=mf_ss_prior_strong', text, count=1, flags=re.M)
    if n != 1:
        die('Cannot locate SLURM job name')
    additions = '\n# Strong-prior defaults; the ordinary launcher is left untouched.\n'
    for name, value in {**overrides, **extra}.items():
        additions += ': "${%s:=%s}"\n' % (name, value)
    # First occurrence only: this file also contains set -euo in a heredoc.
    text = text.replace('set -euo pipefail\n', 'set -euo pipefail\n' + additions, 1)
    exports = '\n'.join('export SINGULARITYENV_%s="$%s"' % (key, key) for key in extra) + '\n\n'
    text = exact(text, 'singularity exec --cleanenv --nv \\\n', exports + 'singularity exec --cleanenv --nv \\\n', 'export new balance environment')
    cli = ''.join('  --%s "$%s" \\\n' % (key.lower(), key) for key in extra)
    text = exact(text, '  $RESUME_ARG\n', cli + '  $RESUME_ARG\n', 'strong-prior CLI arguments')
    if '--train_epochs 25' not in text:
        die('Expected the current 25-epoch launcher; inspect your local launcher')
    return text


def build_changes(root):
    paths = ('train.py', 'models/morph_flow.py', 'models/trellis_ss_prior.py',
             'slurm/train_morphflow_v3.slurm')
    originals = {path: (root / path).read_text(encoding='utf-8') for path in paths}
    if 'from modules.prior_gradient_balance import (' in originals['train.py']:
        die('Strong-prior patch is already present. Do not apply it twice; use --verify-installed.')
    changes = {
        'train.py': rewrite_train(originals['train.py']),
        'models/morph_flow.py': rewrite_morph(originals['models/morph_flow.py']),
        'models/trellis_ss_prior.py': rewrite_prior(originals['models/trellis_ss_prior.py']),
        'slurm/train_morphflow_v3_prior_strong.slurm': make_launcher(originals['slurm/train_morphflow_v3.slurm']),
        **PAYLOADS,
    }
    for name in (set(PAYLOADS) | {"slurm/train_morphflow_v3_prior_strong.slurm"}):
        if (root / name).exists() and (root / name).read_text(encoding='utf-8') != clean(changes[name]):
            die('Refusing to overwrite pre-existing new file: ' + name)
    return {k: clean(v) for k, v in changes.items()}


def check_syntax(changes):
    for name, data in changes.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
        elif name.endswith('.slurm'):
            out = subprocess.run(['bash', '-n'], input=data, text=True, capture_output=True)
            if out.returncode:
                die('Bash syntax error in %s:\n%s' % (name, out.stderr))


def write_atomic(path, content, mode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply(root, changes):
    git_entry = root / '.git'
    if git_entry.is_dir():
        git_dir = git_entry
    elif git_entry.is_file() and git_entry.read_text().startswith('gitdir: '):
        git_dir = Path(git_entry.read_text().strip()[8:])
        if not git_dir.is_absolute():
            git_dir = root / git_dir
    else:
        die('Run inside the MorphFlow git checkout: .git was not found')
    base = git_dir.resolve() / 'morphflow_patches'
    backup = base / (VERSION + '-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    backup.mkdir(parents=True, exist_ok=False)
    before = {}
    manifest = {'version': VERSION, 'root': str(root), 'files': {}}
    for name, text in changes.items():
        p = root / name
        before[name] = p.read_bytes() if p.exists() else None
        if p.exists():
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        manifest['files'][name] = dict(existed=p.exists(), sha256=hashlib.sha256(text.encode()).hexdigest())
    (backup / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    completed = []
    try:
        for name, text in changes.items():
            p = root / name
            mode = p.stat().st_mode & 0o777 if p.exists() else 0o644
            write_atomic(p, text.encode('utf-8'), mode)
            completed.append(name)
    except BaseException:
        for name in reversed(completed):
            p = root / name
            if before[name] is None:
                p.unlink(missing_ok=True)
            else:
                shutil.copy2(backup / name, p)
        raise
    print('APPLIED. Backup (outside tracked files):', backup)
    print('Resume from the immutable epoch-14 checkpoint; do not use a moving best.pt.')


def main():
    if sys.version_info < (3, 9):
        raise SystemExit('Use Python 3.10 inside the TRELLIS container, not the login-node Python.')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.')
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--apply', action='store_true')
    action.add_argument('--verify-installed', action='store_true')
    action.add_argument('--check-checkpoint', metavar='PATH')
    p.add_argument('--diff', action='store_true')
    args = p.parse_args()
    root = Path(args.root).resolve()
    if args.check_checkpoint:
        sys.argv = ['check_strong_checkpoint.py', args.check_checkpoint]
        namespace = {'__name__': '__main__'}
        exec(compile(PAYLOADS['tools/check_strong_checkpoint.py'],
                     'tools/check_strong_checkpoint.py', 'exec'), namespace)
        return
    if args.verify_installed:
        names = ['train.py', 'models/morph_flow.py', 'models/trellis_ss_prior.py',
                 'slurm/train_morphflow_v3_prior_strong.slurm', *PAYLOADS]
        content = {name: (root/name).read_text(encoding='utf-8') for name in names}
        if 'from modules.prior_gradient_balance import (' not in content['train.py']:
            die('Patch is not installed')
        check_syntax(content)
        print('Installed files parse successfully. Now run the CPU and DDP tests.')
        return
    changes = build_changes(root)
    check_syntax(changes)
    for name, data in changes.items():
        old = (root / name).read_text(encoding='utf-8') if (root / name).exists() else ''
        print('[checked]', name, '(new)' if not old else '(modified)')
        if args.diff:
            print(''.join(difflib.unified_diff(old.splitlines(True), data.splitlines(True),
                                             fromfile='a/'+name, tofile='b/'+name)))
    if args.apply:
        apply(root, changes)
    else:
        print('CHECK PASSED. No repository files written. Re-run with --apply.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        raise SystemExit('PATCH ABORTED: %s' % exc)
