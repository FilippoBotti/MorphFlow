"""CLI and scheduling for the endpoint-conditioned TRELLIS SS projection prior."""

import math


def add_trellis_prior_args(parser):
    group = parser.add_argument_group(
        "Frozen endpoint-conditioned TRELLIS SS projection prior"
    )

    group.add_argument(
        "--trellis_prior_weight",
        type=float,
        default=0.0,
        help="Weight of TRELLIS projection supervision. 0 disables it.",
    )
    group.add_argument(
        "--trellis_prior_every",
        type=int,
        default=4,
        help="Apply projection supervision every N optimizer updates.",
    )
    group.add_argument(
        "--trellis_prior_warmup_steps",
        type=int,
        default=10000,
        help="Linearly ramp projection weight over this many updates.",
    )
    group.add_argument(
        "--trellis_prior_rollout_steps",
        type=int,
        default=8,
        help="Student Euler rollout steps from Gaussian noise.",
    )
    group.add_argument(
        "--trellis_prior_grad_steps",
        type=int,
        default=2,
        help="Differentiate final K student rollout steps; 0 means all.",
    )
    group.add_argument(
        "--trellis_prior_max_items",
        type=int,
        default=1,
        help="Maximum prior samples per GPU; 0 uses the full batch.",
    )
    group.add_argument(
        "--trellis_prior_checkpoint",
        type=int,
        choices=[0, 1],
        default=1,
        help="Activation checkpoint differentiated rollout steps.",
    )
    group.add_argument(
        "--trellis_prior_t_min",
        type=float,
        default=0.05,
        help="Minimum local TRELLIS projection timestep.",
    )
    group.add_argument(
        "--trellis_prior_t_max",
        type=float,
        default=0.20,
        help="Maximum local TRELLIS projection timestep.",
    )
    group.add_argument(
        "--trellis_prior_projection_clip_ratio",
        type=float,
        default=0.10,
        help=(
            "Maximum projection-delta RMS relative to selected "
            "endpoint SS RMS. 0 disables the trust-region clip."
        ),
    )
    group.add_argument(
        "--trellis_prior_tangent_projection",
        type=int,
        choices=[0, 1],
        default=0,
        help="Remove projection component parallel to current student latent.",
    )
    group.add_argument(
        "--trellis_prior_rms_guard_weight",
        type=float,
        default=1.0,
        help="Weight of the loose anti-collapse RMS guard.",
    )
    group.add_argument(
        "--trellis_prior_rms_guard_low_ratio",
        type=float,
        default=0.25,
        help="Lower RMS bound relative to smaller endpoint RMS.",
    )
    group.add_argument(
        "--trellis_prior_rms_guard_high_ratio",
        type=float,
        default=2.0,
        help="Upper RMS bound relative to larger endpoint RMS.",
    )


def validate_trellis_prior_args(args):
    for name in (
        "trellis_prior_weight",
        "trellis_prior_projection_clip_ratio",
        "trellis_prior_rms_guard_weight",
        "trellis_prior_rms_guard_low_ratio",
        "trellis_prior_rms_guard_high_ratio",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name} must be finite and >= 0")

    for name in (
        "trellis_prior_every",
        "trellis_prior_rollout_steps",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be >= 1")

    for name in (
        "trellis_prior_warmup_steps",
        "trellis_prior_grad_steps",
        "trellis_prior_max_items",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be >= 0")

    if args.trellis_prior_grad_steps > args.trellis_prior_rollout_steps:
        raise ValueError(
            "--trellis_prior_grad_steps must not exceed "
            "--trellis_prior_rollout_steps"
        )

    if not (
        0
        < args.trellis_prior_t_min
        < args.trellis_prior_t_max
        < 1
    ):
        raise ValueError(
            "Prior timesteps must satisfy "
            "0 < --trellis_prior_t_min < --trellis_prior_t_max < 1"
        )

    if (
        args.trellis_prior_rms_guard_low_ratio
        >= args.trellis_prior_rms_guard_high_ratio
    ):
        raise ValueError(
            "RMS guard low ratio must be smaller than high ratio"
        )

    if args.trellis_prior_weight > 0 and (
        args.flow_target != "ss"
        or args.ss_flow_arch != "standard"
        or args.trellis_model != "image_large"
    ):
        raise ValueError(
            "--trellis_prior_weight > 0 requires "
            "--flow_target ss --ss_flow_arch standard "
            "--trellis_model image_large"
        )


def trellis_prior_forward_kwargs(args, prior, global_step):
    if (
        prior is None
        or args.trellis_prior_weight == 0
        or global_step % args.trellis_prior_every
    ):
        return {}

    ramp = min(
        1.0,
        (global_step + 1)
        / max(1, args.trellis_prior_warmup_steps),
    )

    return {
        "trellis_prior": prior,
        "trellis_prior_weight": args.trellis_prior_weight * ramp,
        "trellis_prior_rollout_steps": args.trellis_prior_rollout_steps,
        "trellis_prior_grad_steps": args.trellis_prior_grad_steps,
        "trellis_prior_max_items": args.trellis_prior_max_items,
        "trellis_prior_checkpoint": bool(args.trellis_prior_checkpoint),
    }
