from __future__ import annotations

"""CNN + Transformer portfolio model for DLStatArb."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from cnn import CNNConfig, ConvBackbone

__all__ = [
    "TransformerConfig",
    "ModelConfig",
    "SinusoidalPosition",
    "AssetTransformerLayer",
    "CrossSectionalTransformer",
    "PortfolioHead",
    "StatArbModel",
]


_ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=True),
    "gelu": lambda: nn.GELU(),
    "silu": lambda: nn.SiLU(inplace=True),
    "tanh": lambda: nn.Tanh(),
}


def _build_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name.lower()]()
    except KeyError as err:
        raise ValueError(f"Unsupported activation '{name}'.") from err


@dataclass
class TransformerConfig:
    """Hyperparameters for the cross-sectional transformer encoder."""

    d_model: int
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    prenorm: bool = True

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
    """Compound configuration for the full CNN+Transformer model."""

    cnn: CNNConfig
    transformer: TransformerConfig
    dropout: float = 0.1
    score_activation: str = "tanh"
    use_context_vector: bool = True


class SinusoidalPosition(nn.Module):
    """Deterministic sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 8192) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence length exceeds positional encoding capacity. Increase max_len.")
        return x + self.pe[:, : x.size(1)]


class AssetTransformerLayer(nn.Module):
    """Single encoder layer operating across the asset dimension."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.nhead,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.dropout_attn = nn.Dropout(cfg.dropout)
        self.dropout_ffn = nn.Dropout(cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_feedforward),
            _build_activation(cfg.activation),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_feedforward, cfg.d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.cfg.prenorm:
            y = self.norm1(x)
            attn_out, _ = self.attn(
                y,
                y,
                y,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
            )
            x = x + self.dropout_attn(attn_out)
            y = self.norm2(x)
            x = x + self.dropout_ffn(self.ffn(y))
            return x

        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout_attn(attn_out))
        x = self.norm2(x + self.dropout_ffn(self.ffn(x)))
        return x


class CrossSectionalTransformer(nn.Module):
    """Stack of transformer layers applied across assets."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(AssetTransformerLayer(cfg) for _ in range(cfg.num_layers))

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = tokens
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=padding_mask)
        return output


class PortfolioHead(nn.Module):
    """Maps encoded asset tokens to cross-sectional scores."""

    def __init__(self, d_model: int, *, dropout: float, activation: str, use_context: bool) -> None:
        super().__init__()
        self.use_context = use_context
        nonlinearity = _build_activation(activation)
        if use_context:
            self.context_norm = nn.LayerNorm(d_model)
            self.context_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nonlinearity,
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
        hidden = max(d_model // 2, 16)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden),
            nonlinearity,
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean tensor.")
        mask_float = mask.float()
        if self.use_context:
            denom = mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)
            context = (encoded * mask_float.unsqueeze(-1)).sum(dim=1) / denom
            gate = self.context_gate(self.context_norm(context))
            encoded = encoded * gate.unsqueeze(1)
        scores = self.scorer(encoded).squeeze(-1)
        return scores


class StatArbModel(nn.Module):
    """End-to-end CNN+Transformer model that outputs portfolio weights."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.cnn = ConvBackbone(cfg.cnn)
        embed_dim = self.cnn.embedding_dim
        if embed_dim != cfg.transformer.d_model:
            self.project = nn.Sequential(
                nn.Linear(embed_dim, cfg.transformer.d_model),
                _build_activation(cfg.transformer.activation),
            )
        else:
            self.project = nn.Identity()

        self.position = SinusoidalPosition(cfg.transformer.d_model)
        self.transformer = CrossSectionalTransformer(cfg.transformer)
        self.head = PortfolioHead(
            cfg.transformer.d_model,
            dropout=cfg.dropout,
            activation=cfg.score_activation,
            use_context=cfg.use_context_vector,
        )

    def forward(
        self,
        panel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if panel.ndim != 3:
            raise ValueError("panel must have shape (batch, assets, lookback).")
        batch, assets, lookback = panel.shape
        if lookback <= 0:
            raise ValueError("lookback dimension must be positive.")

        device = panel.device
        if mask is None:
            mask = torch.ones(batch, assets, dtype=torch.bool, device=device)
        if mask.ndim != 2:
            raise ValueError("mask must have shape (batch, assets).")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Each sample must have at least one valid asset.")

        flat = panel.view(batch * assets, 1, lookback)
        features = self.cnn(flat)
        features = features.view(batch, assets, -1)
        tokens = self.project(features)
        tokens = self.position(tokens)

        padding_mask = ~mask
        encoded = self.transformer(tokens, padding_mask=padding_mask)
        scores = self.head(encoded, mask)
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
        cnn=CNNConfig(channels=(32, 64, 128, 256), dropout=0.2, multiscale_pool=True),
        transformer=TransformerConfig(d_model=256, nhead=8, num_layers=4, dim_feedforward=512),
        dropout=0.1,
    )
    model = StatArbModel(cfg)
    panel = torch.randn(2, 40, 60)
    validity = torch.randint(0, 2, (2, 40), dtype=torch.bool)
    validity[:, 0] = True
    weights, scores = model(panel, validity)
    print(weights.shape, scores.shape)
