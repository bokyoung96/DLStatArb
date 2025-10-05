from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import torch
from torch import nn
from torch.amp import GradScaler as AmpGradScaler
from torch.amp import autocast as amp_autocast
from torch.utils.data import DataLoader
from transformer import StatArbModel

__all__ = [
    "TrainerConfig",
    "SharpeLoss",
    "MeanVarianceLoss",
    "SqrtMeanSharpeLoss",
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
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))
    eval_interval: int = 1
    dtype: torch.dtype = torch.float32
    # NOTE: 'sharpe' | 'meanvar' | 'sqrtMeanSharpe'
    objective: str = "sharpe"
    # NOTE: 'softmax' (use model) | 'l1' (scores L1-normalized)
    weight_scheme: str = "softmax"
    trans_cost: float = 0.0
    hold_cost: float = 0.0
    early_stopping: bool = False
    early_stopping_patience: int = 50
    early_stopping_max_trials: int = 5
    lr_decay: float = 0.5
    checkpoint_path: Optional[str] = None
    force_retrain: bool = True


class SharpeLoss(nn.Module):
    def __init__(self, *, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        if returns.ndim != 1:
            raise ValueError(
                "returns must be a 1D tensor of per-sample pnl values.")
        mean = returns.mean()
        std = returns.std(unbiased=False)
        sharpe = mean / (std + self.epsilon)
        return -sharpe


class MeanVarianceLoss(nn.Module):
    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        if returns.ndim != 1:
            raise ValueError(
                "returns must be a 1D tensor of per-sample pnl values.")
        mean = returns.mean()
        std = returns.std(unbiased=False)
        return -(mean * 252.0) + std * 15.9


class SqrtMeanSharpeLoss(nn.Module):
    def __init__(self, *, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        if returns.ndim != 1:
            raise ValueError(
                "returns must be a 1D tensor of per-sample pnl values.")
        mean = returns.mean()
        std = returns.std(unbiased=False)
        return -torch.sign(mean) * torch.sqrt(torch.abs(mean)) / (std + self.epsilon)


def _resolve_loss(config: TrainerConfig, loss_fn: Optional[nn.Module]) -> nn.Module:
    if loss_fn is not None:
        return loss_fn
    objective = (config.objective or "sharpe").lower()
    if objective == "sharpe":
        return SharpeLoss()
    if objective == "meanvar":
        return MeanVarianceLoss()
    if objective == "sqrtmeansharpe":
        return SqrtMeanSharpeLoss()
    raise ValueError(
        "Unsupported objective. Use 'sharpe', 'meanvar', or 'sqrtMeanSharpe', "
        "or provide a custom loss_fn."
    )


def _should_use_amp(device: torch.device, dtype: torch.dtype) -> bool:
    return device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)


def _make_grad_scaler(device: torch.device, use_amp: bool) -> Optional["AmpGradScaler"]:
    if not use_amp:
        return None
    return AmpGradScaler(device_type=device.type)


def _autocast_context(device: torch.device, use_amp: bool, dtype: torch.dtype):
    if not use_amp:
        return nullcontext()
    kwargs: dict[str, object] = {"device_type": device.type}
    if dtype in (torch.float16, torch.bfloat16):
        kwargs["dtype"] = dtype
    return amp_autocast(**kwargs)


def _renorm_weights(
    model_weights: torch.Tensor,
    scores: torch.Tensor,
    mask: torch.Tensor,
    *,
    scheme: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    if scheme.lower() == "softmax":
        return torch.where(mask, model_weights, torch.zeros_like(model_weights))

    if scheme.lower() == "l1":
        s = torch.where(mask, scores, torch.zeros_like(scores))
        denom = s.abs().sum(dim=1, keepdim=True)
        denom = denom + (denom == 0.0).float() * eps
        w = s / denom
        return torch.where(mask, w, torch.zeros_like(w))

    raise ValueError("weight_scheme must be either 'softmax' or 'l1'.")


def _apply_costs(
    returns: torch.Tensor,
    weights: torch.Tensor,
    *,
    trans_cost: float,
    hold_cost: float,
) -> torch.Tensor:
    if trans_cost == 0.0 and hold_cost == 0.0:
        return returns
    b, n = weights.shape
    if b <= 1:
        turn = torch.zeros(b, device=weights.device, dtype=weights.dtype)
    else:
        turn = torch.sum((weights[1:] - weights[:-1]).abs(), dim=1)
        turn = torch.cat(
            [torch.zeros(1, device=weights.device, dtype=weights.dtype), turn], dim=0)
    short_prop = torch.sum(torch.abs(torch.minimum(weights, torch.zeros(
        1, device=weights.device, dtype=weights.dtype))), dim=1)
    return returns - trans_cost * turn - hold_cost * short_prop


def train_one_epoch(
    model: StatArbModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    grad_clip: Optional[float] = None,
    scaler: Optional["AmpGradScaler"] = None,
    use_amp: bool = False,
    config: Optional[TrainerConfig] = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    amp_dtype = config.dtype if config else torch.float32

    for batch in loader:
        if len(batch) != 3:
            raise ValueError(
                "Data loader must yield (panel, mask, next_residual).")
        panel, mask, next_residual = batch
        panel = panel.to(device=device)
        mask = mask.to(device=device)
        next_residual = next_residual.to(device=device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp, amp_dtype):
            weights_soft, scores = model(panel, mask)
            eff_weights = _renorm_weights(
                weights_soft, scores, mask,
                scheme=(config.weight_scheme if config else "softmax"),
            )
            pnl = (eff_weights * next_residual).sum(dim=-1)
            pnl = _apply_costs(
                pnl, eff_weights,
                trans_cost=(config.trans_cost if config else 0.0),
                hold_cost=(config.hold_cost if config else 0.0),
            )
            loss = loss_fn(pnl)

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
    config: Optional[TrainerConfig] = None,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    returns: list[torch.Tensor] = []

    for batch in loader:
        if len(batch) != 3:
            raise ValueError(
                "Data loader must yield (panel, mask, next_residual).")
        panel, mask, next_residual = batch
        panel = panel.to(device=device)
        mask = mask.to(device=device)
        next_residual = next_residual.to(device=device)

        weights_soft, scores = model(panel, mask)
        eff_weights = _renorm_weights(
            weights_soft, scores, mask,
            scheme=(config.weight_scheme if config else "softmax"),
        )
        pnl = (eff_weights * next_residual).sum(dim=-1)
        pnl = _apply_costs(
            pnl, eff_weights,
            trans_cost=(config.trans_cost if config else 0.0),
            hold_cost=(config.hold_cost if config else 0.0),
        )
        loss = loss_fn(pnl)
        losses.append(float(loss))

        returns.append(pnl.detach().cpu())

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
    optimizer_factory: Optional[Callable[[
        Iterable[torch.nn.Parameter]], torch.optim.Optimizer]] = None,
    loss_fn: Optional[nn.Module] = None,
) -> list[dict[str, float]]:
    device = config.device
    model = model.to(device=device, dtype=config.dtype)
    loss_fn = _resolve_loss(config, loss_fn)

    use_amp = _should_use_amp(device, config.dtype)
    if optimizer_factory is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = optimizer_factory(model.parameters())

    scaler = _make_grad_scaler(device, use_amp)

    history: list[dict[str, float]] = []
    if config.checkpoint_path and os.path.isfile(config.checkpoint_path) and not config.force_retrain:
        try:
            ckpt = torch.load(config.checkpoint_path, map_location=device)
            model.load_state_dict(ckpt.get("model_state_dict", {}))
            optimizer.load_state_dict(ckpt.get("optimizer_state_dict", {}))
        except Exception:
            pass

    best_val = float("inf")
    patience = 0
    reductions = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device=device,
            grad_clip=config.grad_clip,
            scaler=scaler,
            use_amp=use_amp,
            config=config,
        )

        metrics = {"epoch": epoch, "train_loss": train_loss}
        if val_loader is not None and epoch % config.eval_interval == 0:
            val_metrics = evaluate(
                model, val_loader, loss_fn, device=device, config=config)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            if config.early_stopping:
                val_loss = val_metrics.get("loss", float("inf"))
                if val_loss < best_val:
                    best_val = val_loss
                    patience = 0
                    if config.checkpoint_path:
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "val_loss": best_val,
                        }, config.checkpoint_path)
                else:
                    patience += 1
                    if patience >= config.early_stopping_patience:
                        for pg in optimizer.param_groups:
                            pg["lr"] *= config.lr_decay
                        if config.checkpoint_path and os.path.isfile(config.checkpoint_path):
                            try:
                                ckpt = torch.load(
                                    config.checkpoint_path, map_location=device)
                                model.load_state_dict(
                                    ckpt.get("model_state_dict", {}))
                            except Exception:
                                pass
                        patience = 0
                        reductions += 1
                        if reductions >= config.early_stopping_max_trials:
                            metrics["stopped"] = 1.0
                            history.append(metrics)
                            break

        history.append(metrics)
    return history
