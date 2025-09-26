from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from cnn import CNNConfig, ConvBackbone, _build_activation

__all__ = [
    "TransformerConfig",
    "ModelConfig",
    "StatArbModel",
]


@dataclass
class TransformerConfig:
    d_model: int
    nhead: int = 4
    num_layers: int = 1
    dim_feedforward: int = 0
    dropout: float = 0.25
    batch_first: bool = False  # NOTE: (T, B*N, C)
    activation: str = "relu"

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.nhead <= 0:
            raise ValueError("nhead must be positive.")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in [0, 1).")


@dataclass
class ModelConfig:
    cnn: CNNConfig
    transformer: TransformerConfig


class StatArbModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # NOTE: (B*N, C_last, L)
        self.cnn = ConvBackbone(cfg.cnn)
        c_last = self.cnn.embedding_dim

        # NOTE: Optional projection to match transformer d_model
        if c_last != cfg.transformer.d_model:
            self.pre_transform = nn.Linear(c_last, cfg.transformer.d_model)
            d_model = cfg.transformer.d_model
        else:
            self.pre_transform = nn.Identity()
            d_model = c_last

        ff_dim = cfg.transformer.dim_feedforward or (2 * d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.transformer.nhead,
            dim_feedforward=ff_dim,
            dropout=cfg.transformer.dropout,
            batch_first=cfg.transformer.batch_first,
            activation=_build_activation(cfg.transformer.activation),
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.transformer.num_layers)

        # NOTE: Linear head to map last time step features to scores per asset
        self.linear = nn.Linear(d_model, 1)

    def forward(
        self,
        panel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if panel.ndim != 3:
            raise ValueError("panel must have shape (batch, assets, lookback).")
        B, N, L = panel.shape
        if L <= 0:
            raise ValueError("lookback dimension must be positive.")

        device = panel.device
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)
        if mask.ndim != 2:
            raise ValueError("mask must have shape (batch, assets).")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Each sample must have at least one valid asset.")

        # NOTE: (B*N, C_last, L)
        x = panel.view(B * N, 1, L)
        feats = self.cnn(x)

        # NOTE: Time-axis Transformer input
        if self.cfg.transformer.batch_first:
            # NOTE: (B*N, L, C_last) -> (B*N, L, d_model)
            feats = feats.permute(0, 2, 1)
            feats = self.pre_transform(feats)
            encoded = self.encoder(feats)
            last = encoded[:, -1, :]
        else:
            # NOTE: (L, B*N, C_last) -> (L, B*N, d_model)
            feats = feats.permute(2, 0, 1)
            feats = self.pre_transform(feats)
            encoded = self.encoder(feats)
            last = encoded[-1, :, :]
        
        # NOTE: (B*N,)
        scores_flat = self.linear(last).squeeze(-1)
        scores = scores_flat.view(B, N)
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        return weights, scores

    @torch.no_grad()
    def predict(
        self,
        panel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weights, _ = self.forward(panel, mask)
        return weights


if __name__ == "__main__":
    cfg = ModelConfig(
        cnn=CNNConfig(channels=(8,), activation="relu", normalization="instance", causal_padding=True),
        transformer=TransformerConfig(d_model=8, nhead=4, num_layers=1, dim_feedforward=16, dropout=0.25),
    )
    model = StatArbModel(cfg)
    panel = torch.randn(2, 10, 30)
    validity = torch.randint(0, 2, (2, 10), dtype=torch.bool)
    validity[:, 0] = True
    w, s = model(panel, validity)
    print(w.shape, s.shape)
