"""Reduction of scalar training metrics without importing the GPU model stack."""

from typing import Dict

import torch


def collect_reduced_forward_metrics(accelerator, model: torch.nn.Module) -> Dict[str, float]:
    """Reduce one ordered vector, using the model's rank-independent metric schema.

    Models must emit the same keys on all ranks, filling inactive auxiliary
    terms with zero. Sorting also removes differences in dictionary insertion
    order. Values are detached so logging cannot change the loss or gradients.
    """
    metrics = getattr(accelerator.unwrap_model(model), "last_forward_metrics", None)
    if not metrics:
        return {}

    names = sorted(metrics)
    values = []
    for name in names:
        value = metrics[name]
        if not torch.is_tensor(value):
            value = torch.tensor(float(value), device=accelerator.device)
        value = value.detach().to(device=accelerator.device, dtype=torch.float32)
        if value.numel() != 1:
            raise ValueError(f"Forward metric {name!r} must be scalar, got {tuple(value.shape)}")
        values.append(value.reshape(()))

    reduced = accelerator.reduce(torch.stack(values), reduction="mean")
    return dict(zip(names, reduced.cpu().tolist()))
