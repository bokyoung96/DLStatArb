from __future__ import annotations

from enum import Enum

import torch
from torch import nn

__all__ = [
    "PoolingKind",
    "MultiScalePool",
    "LastStepPool",
    "MeanPool",
    "build_pool",
]


class PoolingKind(Enum):
    MULTISCALE = "multiscale"
    MEAN = "mean"
    LAST = "last"


class MultiScalePool(nn.Module):
    """
    Input:  (B, C, L)
    Output: (B, 3*C)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("MultiScalePool expects (B, C, L).")
        mean = x.mean(dim=-1)
        max_val = x.max(dim=-1).values
        var = x.var(dim=-1, unbiased=False)
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        return torch.cat([mean, max_val, std], dim=1)


class LastStepPool(nn.Module):
    """
    Input:  (B, C, L)
    Output: (B, C)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("LastStepPool expects (B, C, L).")
        return x[:, :, -1]


class MeanPool(nn.Module):
    """
    Input:  (B, C, L)
    Output: (B, C)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("MeanPool expects (B, C, L).")
        return x.mean(dim=-1)


def build_pool(kind: PoolingKind | str) -> nn.Module:
    if isinstance(kind, str):
        try:
            kind = PoolingKind(kind.lower())
        except ValueError as err:
            valid = ", ".join(k.value for k in PoolingKind)
            raise ValueError(f"Unsupported pooling kind '{kind}'. Available: {valid}") from err

    if kind is PoolingKind.MULTISCALE:
        return MultiScalePool()
    if kind is PoolingKind.MEAN:
        return MeanPool()
    if kind is PoolingKind.LAST:
        return LastStepPool()
