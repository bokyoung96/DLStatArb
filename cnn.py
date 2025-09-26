from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

__all__ = [
    "CNNConfig",
    "ResidualConvBlock",
    "ConvBackbone",
]


_ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=True),
    "gelu": lambda: nn.GELU(),
    "silu": lambda: nn.SiLU(inplace=True),
}

_NORMALIZATIONS = {
    "instance": lambda ch: nn.InstanceNorm1d(ch, affine=True),
    "batch": lambda ch: nn.BatchNorm1d(ch),
    "layer": lambda ch: nn.GroupNorm(1, ch),
}


def _build_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name.lower()]()
    except KeyError as err:
        raise ValueError(
            f"Unsupported activation '{name}'. Available: {list(_ACTIVATIONS)}"
        ) from err


def _build_norm(norm: str, channels: int) -> nn.Module:
    try:
        return _NORMALIZATIONS[norm.lower()](channels)
    except KeyError as err:
        raise ValueError(
            f"Unsupported normalization '{norm}'. Available: {list(_NORMALIZATIONS)}"
        ) from err


@dataclass
class CNNConfig:
    in_channels: int = 1
    channels: Sequence[int] = (8,)
    kernel_sizes: Sequence[int] | None = None
    dilations: Sequence[int] | int = 1
    dropout: float = 0.0
    activation: str = "relu"
    normalization: str = "instance"
    residual_scaling: float = 1.0
    causal_padding: bool = True

    def __post_init__(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if not self.channels or any(ch <= 0 for ch in self.channels):
            raise ValueError("channels must be non-empty and positive.")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in [0,1).")
        if self.residual_scaling <= 0:
            raise ValueError("residual_scaling must be positive.")

        if self.kernel_sizes is None:
            self.kernel_sizes = tuple(2 for _ in self.channels)
        else:
            ks = tuple(int(k) for k in self.kernel_sizes)
            if len(ks) != len(self.channels):
                raise ValueError("kernel_sizes length must match channels length.")
            if any(k <= 0 for k in ks):
                raise ValueError("kernel_sizes must be positive integers.")
            self.kernel_sizes = ks

        if isinstance(self.dilations, Iterable) and not isinstance(self.dilations, (str, bytes)):
            ds = tuple(int(d) for d in self.dilations)
            if len(ds) != len(self.channels):
                raise ValueError("dilations length must match channels length.")
            if any(d <= 0 for d in ds):
                raise ValueError("All dilations must be positive.")
            self.dilations = ds
        else:
            d = int(self.dilations)
            if d <= 0:
                raise ValueError("dilations must be positive.")
            self.dilations = tuple(d for _ in self.channels)

    def effective_out_dim(self) -> int:
        return self.channels[-1]


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        normalization: str = "instance",
        residual_scaling: float = 1.0,
        causal_padding: bool = True,
    ) -> None:
        super().__init__()

        pad_len = (kernel_size - 1) * dilation if causal_padding else 0
        conv_padding = 0 if causal_padding else ((kernel_size - 1) // 2) * dilation

        def conv_layer(in_ch: int, out_ch: int) -> nn.Module:
            layers: list[nn.Module] = []
            if causal_padding and pad_len > 0:
                layers.append(nn.ConstantPad1d((pad_len, 0), 0.0))
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size,
                                    padding=conv_padding, dilation=dilation))
            return nn.Sequential(*layers)

        self.block = nn.Sequential(
            _build_norm(normalization, in_channels),
            _build_activation(activation),
            conv_layer(in_channels, out_channels),
            _build_norm(normalization, out_channels),
            _build_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            conv_layer(out_channels, out_channels),
        )

        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
        self.residual_scaling = residual_scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.block(x)
        return residual + self.residual_scaling * out


class ConvBackbone(nn.Module):
    def __init__(self, cfg: CNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        blocks: list[nn.Module] = []
        in_ch = cfg.in_channels
        for out_ch, kernel, dil in zip(cfg.channels, cfg.kernel_sizes, cfg.dilations):
            blocks.append(
                ResidualConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel,
                    dilation=dil,
                    dropout=cfg.dropout,
                    activation=cfg.activation,
                    normalization=cfg.normalization,
                    residual_scaling=cfg.residual_scaling,
                    causal_padding=cfg.causal_padding,
                )
            )
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self._embed_dim = cfg.channels[-1]

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Input tensor must have shape (B, C, L).")
        return self.blocks(x)


if __name__ == "__main__":
    for causal in [False, True]:
        cfg = CNNConfig(channels=(32, 64), dropout=0.1, causal_padding=causal)
        model = ConvBackbone(cfg)
