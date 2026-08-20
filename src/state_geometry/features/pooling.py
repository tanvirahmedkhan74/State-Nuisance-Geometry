from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masks_to_tubelet_occupancy(
    masks: Tensor,
    tubelet_size: int = 2,
    patch_size: int = 16,
) -> Tensor:
    """Area-average binary masks [B,T,H,W] onto [B,T/tubelet,H/patch,W/patch]."""
    if masks.ndim != 4:
        raise ValueError("masks must have shape [B,T,H,W]")
    batch, frames, height, width = masks.shape
    if frames % tubelet_size or height % patch_size or width % patch_size:
        raise ValueError("mask dimensions must be divisible by tubelet and patch sizes")
    values = masks.to(dtype=torch.float32).unsqueeze(1)
    occupancy = F.avg_pool3d(
        values,
        kernel_size=(tubelet_size, patch_size, patch_size),
        stride=(tubelet_size, patch_size, patch_size),
    )
    expected = (batch, frames // tubelet_size, height // patch_size, width // patch_size)
    result = occupancy.squeeze(1)
    if result.shape != expected:
        raise AssertionError(f"unexpected occupancy shape {result.shape}, expected {expected}")
    return result


def weighted_region_pool(
    tokens: Tensor,
    occupancy: Tensor,
    minimum_mass: float = 1.0,
    epsilon: float = 1e-8,
) -> Tensor:
    """FP32 weighted pooling for contextual tubelet tokens."""
    if tokens.ndim != 5:
        raise ValueError("tokens must have shape [B,T',H',W',D]")
    if occupancy.shape != tokens.shape[:-1]:
        raise ValueError("occupancy grid must match token grid")
    token_values = tokens.float()
    # Region geometry is intentionally constructed on CPU; move only the small
    # occupancy grid to the encoder-token device at the pooling boundary.
    weights = occupancy.to(device=token_values.device, dtype=torch.float32).clamp(0.0, 1.0)
    mass = weights.sum(dim=(1, 2, 3))
    if not torch.isfinite(token_values).all() or not torch.isfinite(weights).all():
        raise ValueError("nonfinite token/occupancy value")
    if (mass < minimum_mass).any():
        bad = torch.nonzero(mass < minimum_mass, as_tuple=False).flatten().tolist()
        raise ValueError(f"region mass below minimum for batch indices {bad}")
    pooled = (token_values * weights.unsqueeze(-1)).sum(dim=(1, 2, 3))
    return pooled / (mass.unsqueeze(-1) + epsilon)


def full_token_pool(tokens: Tensor) -> Tensor:
    if tokens.ndim != 5:
        raise ValueError("tokens must have shape [B,T',H',W',D]")
    return tokens.float().mean(dim=(1, 2, 3))
