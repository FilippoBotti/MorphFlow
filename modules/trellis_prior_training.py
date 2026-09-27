"""CLI and deterministic scheduling for the optional frozen SS prior."""

import math


def add_trellis_prior_args(parser):
    group = parser.add_argument_group("Frozen TRELLIS SS prior (RFDS)")
    group.add_argument("--trellis_prior_weight", type=float, default=0.0,
                       help="RFDS weight on generated student SS latents. 0 avoids loading the prior.")
    group.add_argument("--trellis_prior_every", type=int, default=4,
                       help="Apply RFDS every N training updates, starting at update 0; no frequency compensation.")
    group.add_argument("--trellis_prior_warmup_steps", type=int, default=1000,
                       help="Linearly ramp the prior weight over this many updates; 0 starts at full weight.")
    group.add_argument("--trellis_prior_rollout_steps", type=int, default=8,
                       help="Euler steps from independent Gaussian noise to a generated SS latent.")
    group.add_argument("--trellis_prior_grad_steps", type=int, default=2,
                       help="Differentiate the last K rollout steps; 0 differentiates the entire rollout.")
    group.add_argument("--trellis_prior_max_items", type=int, default=1,
                       help="Maximum rollout items per GPU per active update; 0 uses the full batch.")
    group.add_argument("--trellis_prior_checkpoint", type=int, choices=[0, 1], default=1,
                       help="Activation checkpointing for differentiated rollout steps.")
    group.add_argument("--trellis_prior_t_min", type=float, default=0.05,
                       help="Lower bound of independent uniform prior noise time.")
    group.add_argument("--trellis_prior_t_max", type=float, default=0.95,
                       help="Upper bound of independent uniform prior noise time.")
    group.add_argument("--trellis_prior_grad_clip", type=float, default=0.0,
                       help="Clip each RFDS residual's RMS before applying its weight; 0 disables clipping.")


def validate_trellis_prior_args(args):
    for name in ("trellis_prior_weight", "trellis_prior_grad_clip"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name} must be finite and >= 0")
    for name in ("trellis_prior_every", "trellis_prior_rollout_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be >= 1")
    for name in ("trellis_prior_warmup_steps", "trellis_prior_grad_steps", "trellis_prior_max_items"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be >= 0")
    if args.trellis_prior_grad_steps > args.trellis_prior_rollout_steps:
        raise ValueError("--trellis_prior_grad_steps must not exceed --trellis_prior_rollout_steps")
    if not 0 < args.trellis_prior_t_min < args.trellis_prior_t_max < 1:
        raise ValueError("Prior noise times must satisfy 0 < --trellis_prior_t_min < --trellis_prior_t_max < 1")
    if args.trellis_prior_weight > 0 and (
        args.flow_target != "ss" or args.ss_flow_arch != "standard" or args.trellis_model != "image_large"
    ):
        raise ValueError(
            "--trellis_prior_weight > 0 requires --flow_target ss --ss_flow_arch standard "
            "--trellis_model image_large (native image unconditional prior)"
        )


def trellis_prior_forward_kwargs(args, prior, global_step):
    """Keep cadence identical across DDP ranks and continuous after resume.

    global_step is the number of already completed training updates. Validation
    deliberately does not call this function: checkpoint selection keeps the
    existing validation objective, not a stochastic RFDS surrogate.
    """
    if prior is None or args.trellis_prior_weight == 0 or global_step % args.trellis_prior_every:
        return {}
    ramp = min(1.0, (global_step + 1) / max(1, args.trellis_prior_warmup_steps))
    return {
        "trellis_prior": prior,
        "trellis_prior_weight": args.trellis_prior_weight * ramp,
        "trellis_prior_rollout_steps": args.trellis_prior_rollout_steps,
        "trellis_prior_grad_steps": args.trellis_prior_grad_steps,
        "trellis_prior_max_items": args.trellis_prior_max_items,
        "trellis_prior_checkpoint": bool(args.trellis_prior_checkpoint),
    }
