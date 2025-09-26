from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformer import StatArbModel


__all__ = [
    "TrainerConfig",
    "SharpeLoss",
    "TrainingLoop",
    "train_one_epoch",
    "evaluate",
]


@dataclass
class TrainerConfig:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    eval_interval: int = 1
    dtype: torch.dtype = torch.float32


class SharpeLoss(nn.Module):
    """Negative Sharpe ratio computed over a mini-batch of portfolio returns."""

    def __init__(self, *, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        weights: torch.Tensor,
        residual_next: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if residual_next.ndim != 2:
            raise ValueError("residual_next must have shape (batch, assets).")
        masked_weights = torch.where(mask, weights, torch.zeros_like(weights))
        pnl = (masked_weights * residual_next).sum(dim=-1)
        mean = pnl.mean()
        std = pnl.std(unbiased=False)
        sharpe = mean / (std + self.epsilon)
        return -sharpe


def train_one_epoch(
    model: StatArbModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    grad_clip: Optional[float] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        if len(batch) != 3:
            raise ValueError("Data loader must yield (panel, mask, next_residual).")
        panel, mask, next_residual = batch
        panel = panel.to(device=device)
        mask = mask.to(device=device)
        next_residual = next_residual.to(device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            weights, _ = model(panel, mask)
            loss = loss_fn(weights, next_residual, mask)

        if scaler is None:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.detach())
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: StatArbModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    returns: list[torch.Tensor] = []

    for batch in loader:
        if len(batch) != 3:
            raise ValueError("Data loader must yield (panel, mask, next_residual).")
        panel, mask, next_residual = batch
        panel = panel.to(device=device)
        mask = mask.to(device=device)
        next_residual = next_residual.to(device=device)

        weights, _ = model(panel, mask)
        loss = loss_fn(weights, next_residual, mask)
        losses.append(float(loss))

        pnl = (weights * next_residual).sum(dim=-1)
        returns.append(pnl.cpu())

    if not returns:
        return {"loss": float("nan"), "sharpe": float("nan"), "mean_return": float("nan"), "vol": float("nan")}

    pnl_vec = torch.cat(returns)
    mean_return = float(pnl_vec.mean())
    vol = float(pnl_vec.std(unbiased=False))
    sharpe = mean_return / (vol + 1e-6)
    return {"loss": float(sum(losses) / len(losses)), "sharpe": sharpe, "mean_return": mean_return, "vol": vol}


def TrainingLoop(
    model: StatArbModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    *,
    config: TrainerConfig,
    optimizer_factory: Optional[Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer]] = None,
    loss_fn: Optional[nn.Module] = None,
) -> list[dict[str, float]]:
    """Run the optimisation loop and return epoch-wise metrics."""

    device = config.device
    model = model.to(device=device, dtype=config.dtype)
    loss_fn = loss_fn or SharpeLoss()

    if optimizer_factory is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = optimizer_factory(model.parameters())

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device=device,
            grad_clip=config.grad_clip,
            scaler=scaler,
        )

        metrics = {"epoch": epoch, "train_loss": train_loss}
        if val_loader is not None and epoch % config.eval_interval == 0:
            val_metrics = evaluate(model, val_loader, loss_fn, device=device)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

        history.append(metrics)

    return history
