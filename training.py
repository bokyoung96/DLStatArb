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
from tqdm.auto import tqdm
from transformer import StatArbModel

__all__ = [
    "TrainerConfig",
    "select_device",
    "SharpeLoss",
    "MeanVarianceLoss",
    "SqrtMeanSharpeLoss",
    "TrainingLoop",
    "train_one_epoch",
    "evaluate",
]


def select_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainerConfig:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    device: torch.device = field(default_factory=select_device)
    eval_interval: int = 1
    dtype: torch.dtype = torch.float32
    objective: str = "sharpe"
    weight_scheme: str = "softmax"
    hold_cost: float = 0.0
    buy_cost: float = 0.0002
    sell_cost: float = 0.0002
    sell_tax: float = 0.0015
    slippage: float = 0.0001
    early_stopping: bool = False
    early_stopping_patience: int = 50
    early_stopping_max_trials: int = 5
    lr_decay: float = 0.5
    checkpoint_path: Optional[str] = None
    force_retrain: bool = True
    show_progress: bool = True
    micro_batch_size: Optional[int] = None
    accumulate_steps: int = 1


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
    objective = config.objective.lower()
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


def _progress_iter(loader: Iterable, enable: bool, desc: str):
    if not enable:
        for item in loader:
            yield item
        return
    for item in tqdm(loader, desc=desc, leave=False):
        yield item


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
    hold_cost: float,
    buy_cost: float,
    sell_cost: float,
    sell_tax: float,
    slippage: float,
) -> torch.Tensor:
    b, _ = weights.shape
    if b <= 1:
        buys = torch.zeros(b, device=weights.device, dtype=weights.dtype)
        sells = torch.zeros(b, device=weights.device, dtype=weights.dtype)
    else:
        dW = weights[1:] - weights[:-1]
        buy_amt = torch.relu(dW).sum(dim=1)
        sell_amt = torch.relu(-dW).sum(dim=1)
        buys = torch.cat([
            torch.zeros(1, device=weights.device, dtype=weights.dtype),
            buy_amt,
        ], dim=0)
        sells = torch.cat([
            torch.zeros(1, device=weights.device, dtype=weights.dtype),
            sell_amt,
        ], dim=0)
    buy_fee = (buy_cost + slippage) * buys
    sell_fee = (sell_cost + sell_tax + slippage) * sells
    short_prop = torch.sum(
        torch.abs(torch.minimum(weights, torch.zeros(
            1, device=weights.device, dtype=weights.dtype))),
        dim=1,
    )
    hold_fee = hold_cost * short_prop
    return returns - (buy_fee + sell_fee + hold_fee)


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
    config: TrainerConfig,
    progress_desc: str = "Train",
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    amp_dtype = config.dtype

    iterator = _progress_iter(
        loader,
        enable=config.show_progress,
        desc=progress_desc,
    )

    for batch in iterator:
        if len(batch) != 3:
            raise ValueError(
                "Data loader must yield (panel, mask, next_residual).")
        panel, mask, next_residual = batch
        panel = panel.to(device=device)
        mask = mask.to(device=device)
        next_residual = next_residual.to(device=device)

        batch_size = panel.shape[0]
        micro_bs = (config.micro_batch_size if (config.micro_batch_size and config.micro_batch_size > 0)
                    else batch_size)
        accum_steps = max(int(config.accumulate_steps), 1)
        optimizer.zero_grad(set_to_none=True)

        micro_count = 0
        running_loss = 0.0
        for start in range(0, batch_size, micro_bs):
            end = min(start + micro_bs, batch_size)
            p = panel[start:end]
            m = mask[start:end]
            r = next_residual[start:end]

            with _autocast_context(device, use_amp, amp_dtype):
                weights_soft, scores = model(p, m)
                eff_weights = _renorm_weights(
                    weights_soft, scores, m,
                    scheme=config.weight_scheme,
                )
                pnl = (eff_weights * r).sum(dim=-1)
                pnl = _apply_costs(
                    pnl, eff_weights,
                    hold_cost=config.hold_cost,
                    buy_cost=config.buy_cost,
                    sell_cost=config.sell_cost,
                    sell_tax=config.sell_tax,
                    slippage=config.slippage,
                )
                loss = loss_fn(pnl)

            scaled_loss = loss / float(accum_steps)
            if scaler is None:
                scaled_loss.backward()
            else:
                scaler.scale(scaled_loss).backward()

            micro_count += 1
            running_loss += float(loss.detach()) * ((end - start) / batch_size)

            if (micro_count % accum_steps == 0) or (end == batch_size):
                if grad_clip is not None:
                    if scaler is None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), grad_clip)
                    else:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), grad_clip)
                if scaler is None:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)

        total_loss += running_loss
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: StatArbModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    *,
    device: torch.device,
    config: TrainerConfig,
    progress_desc: str = "Eval",
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    returns: list[torch.Tensor] = []

    iterator = _progress_iter(
        loader,
        enable=config.show_progress,
        desc=progress_desc,
    )

    for batch in iterator:
        if len(batch) != 3:
            raise ValueError(
                "Data loader must yield (panel, mask, next_residual).")
        panel, mask, next_residual = batch
        panel = panel.to(device=device)
        mask = mask.to(device=device)
        next_residual = next_residual.to(device=device)

        weights_soft, scores = model(panel, mask)
        eff_weights = _renorm_weights(
            weights_soft, scores, mask, scheme=config.weight_scheme)
        pnl = (eff_weights * next_residual).sum(dim=-1)
        pnl = _apply_costs(
            pnl, eff_weights,
            hold_cost=config.hold_cost,
            buy_cost=config.buy_cost,
            sell_cost=config.sell_cost,
            sell_tax=config.sell_tax,
            slippage=config.slippage,
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
    saved_any_ckpt = False

    if config.show_progress:
        epoch_iter: Iterable[int] = tqdm(
            range(1, config.epochs + 1), desc="Epoch", leave=True)
    else:
        epoch_iter = range(1, config.epochs + 1)

    for epoch in epoch_iter:
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
            progress_desc=f"Train (epoch {epoch})",
        )

        metrics = {"epoch": epoch, "train_loss": train_loss}
        if val_loader is not None and epoch % config.eval_interval == 0:
            val_metrics = evaluate(
                model,
                val_loader,
                loss_fn,
                device=device,
                config=config,
                progress_desc=f"Eval (epoch {epoch})",
            )
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
                        saved_any_ckpt = True
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

    # Always save a final checkpoint if a path is provided and no checkpoint
    # was saved during training (e.g., early_stopping disabled or no improvement).
    if config.checkpoint_path and not saved_any_ckpt:
        try:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": history[-1]["epoch"] if history else 0,
                "val_loss": best_val,
            }, config.checkpoint_path)
        except Exception:
            pass

    return history
