"""Training-time SS sampling with a differentiable Euler suffix."""

from contextlib import contextmanager

import torch
from torch.utils.checkpoint import checkpoint


@contextmanager
def rollout_evaluation_mode(module, *, disable_checkpointing=False):
    """Use sampling behavior, restoring all individual module flags afterwards.

    This context is entered inside each checkpointed function as well, so
    recomputation sees the same dropout behavior as its original forward.
    Whole-flow checkpointing replaces internal block checkpointing during a
    rollout; otherwise nested recomputation could run after eval mode is reset.
    """
    modes = [(child, child.training) for child in module.modules()]
    checkpoints = [
        (child, child.use_checkpoint)
        for child, _ in modes
        if disable_checkpointing and hasattr(child, "use_checkpoint")
    ]
    try:
        for child, _ in modes:
            child.training = False
        for child, _ in checkpoints:
            child.use_checkpoint = False
        yield
    finally:
        for child, training in modes:
            child.training = training
        for child, enabled in checkpoints:
            child.use_checkpoint = enabled


def differentiable_ss_rollout(
    flow,
    noise,
    condition,
    alpha,
    *,
    steps=8,
    grad_steps=2,
    use_checkpoint=True,
):
    """Generate an SS sample, retaining gradients through the final K steps.

    Uses the evaluation sampler's linear t=1 -> 0 schedule (time rescale=1),
    native 0..1000 model timesteps, and an FP32 Euler state. ``condition`` must
    have been encoded once *with gradients*, before this function. Its graph
    remains connected to the differentiated suffix even when the prefix is
    sampled without gradients. ``grad_steps=0`` differentiates every step.
    The initial noise is independent of FM targets and is always detached.
    """
    if int(steps) != steps or steps < 1:
        raise ValueError("rollout steps must be an integer >= 1")
    if int(grad_steps) != grad_steps or not 0 <= grad_steps <= steps:
        raise ValueError("rollout grad_steps must be in [0, steps] (0 means all steps)")
    steps, grad_steps = int(steps), int(grad_steps)
    tuple_condition = isinstance(condition, tuple)
    tensors = condition if tuple_condition else (condition,)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("rollout condition must be a tensor or tuple of tensors")

    def velocity(x_t, timestep, alpha_value, *condition_tensors):
        cond = tuple(condition_tensors) if tuple_condition else condition_tensors[0]
        # This also executes during backward checkpoint recomputation.
        # A surrounding training autocast context spans the no-grad prefix and
        # differentiated suffix. Cached low-precision parameter copies from
        # no-grad steps can otherwise silently cut gradients or disagree with
        # checkpoint recomputation. Preserve autocast settings, disable cache.
        with rollout_evaluation_mode(flow, disable_checkpointing=True), torch.autocast(
            device_type=x_t.device.type,
            enabled=torch.is_autocast_enabled(x_t.device.type),
            dtype=torch.get_autocast_dtype(x_t.device.type),
            cache_enabled=False,
        ):
            return flow(x_t, timestep * 1000.0, cond, alpha=alpha_value).float()

    x_t = noise.detach().float()
    times = torch.linspace(1.0, 0.0, steps + 1, device=x_t.device, dtype=torch.float32)
    first_grad_step = 0 if grad_steps == 0 else steps - grad_steps
    for step in range(steps):
        timestep = times[step].expand(x_t.shape[0])
        dt = times[step] - times[step + 1]
        inputs = (x_t, timestep, alpha, *tensors)
        if step < first_grad_step:
            with torch.no_grad():
                x_t = x_t - dt * velocity(*inputs)
        elif use_checkpoint and torch.is_grad_enabled():
            # Explicit condition inputs preserve encoder/gate gradients even
            # when x_t came from a no-grad prefix and has requires_grad=False.
            x_t = x_t - dt * checkpoint(velocity, *inputs, use_reentrant=False)
        else:
            x_t = x_t - dt * velocity(*inputs)
    return x_t


def differentiable_slat_rollout(flow, noise, condition, alpha, *, steps=8, grad_steps=2, use_checkpoint=True):
    """SLat rollout on fixed sparse coords with a differentiable suffix."""
    if int(steps) != steps or steps < 1:
        raise ValueError("rollout steps must be an integer >= 1")
    if int(grad_steps) != grad_steps or not 0 <= grad_steps <= steps:
        raise ValueError("rollout grad_steps must be in [0, steps]")
    steps, grad_steps = int(steps), int(grad_steps)
    tuple_condition = isinstance(condition, tuple)
    tensors = condition if tuple_condition else (condition,)
    template = noise
    batch_size = int(noise.shape[0])
    def velocity(feats, timestep, alpha_value, *condition_tensors):
        cond = tuple(condition_tensors) if tuple_condition else condition_tensors[0]
        x_sparse = template.replace(feats)
        with rollout_evaluation_mode(flow, disable_checkpointing=True), torch.autocast(
            device_type=feats.device.type,
            enabled=torch.is_autocast_enabled(feats.device.type),
            dtype=torch.get_autocast_dtype(feats.device.type),
            cache_enabled=False,
        ):
            return flow(x_sparse, timestep * 1000.0, cond, alpha=alpha_value).feats.float()
    feats = noise.feats.detach().float()
    times = torch.linspace(1.0, 0.0, steps + 1, device=feats.device, dtype=torch.float32)
    first_grad_step = 0 if grad_steps == 0 else steps - grad_steps
    for step in range(steps):
        timestep = times[step].expand(batch_size)
        dt = times[step] - times[step + 1]
        inputs = (feats, timestep, alpha, *tensors)
        if step < first_grad_step:
            with torch.no_grad(): feats = feats - dt * velocity(*inputs)
        elif use_checkpoint and torch.is_grad_enabled():
            feats = feats - dt * checkpoint(velocity, *inputs, use_reentrant=False)
        else:
            feats = feats - dt * velocity(*inputs)
    return template.replace(feats)
