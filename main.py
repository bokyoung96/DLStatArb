from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from classify import WindowClassifier, WindowsSplit
from cnn import CNNConfig
from preprocess import ResidualsDataset, WindowBuilder, WindowConfig, Windows
from torch.utils.data import DataLoader
from training import (MeanVarianceLoss, SharpeLoss, SqrtMeanSharpeLoss,
                      TrainerConfig, TrainingLoop, evaluate)
from transformer import ModelConfig, StatArbModel, TransformerConfig

CONFIG_PATH = Path("train_config.json")


logger = logging.getLogger(__name__)


@dataclass
class ScriptConfig:
    windows_path: Path = Path("PROCESSED/residual_windows.pt")
    residuals_path: Path = Path("DATA/residuals.parquet")
    lookback: int = 30
    stride: int = 1
    min_assets: int = 25
    min_valid_ratio: float = 1.0
    horizon: int = 0
    zero_as_invalid: bool = False
    train_end: Optional[str] = None
    val_end: Optional[str] = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    batch_size: int = 200
    num_workers: int = 0
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_interval: int = 1
    objective: str = "sharpe"
    weight_scheme: str = "l1"
    trans_cost: float = 0.0005
    hold_cost: float = 0.0001
    early_stopping: bool = False
    early_stopping_patience: int = 50
    early_stopping_max_trials: int = 5
    lr_decay: float = 0.5
    checkpoint: Optional[Path] = None
    force_retrain: bool = False
    dtype: str = "float32"
    device: Optional[str] = None
    history_out: Optional[Path] = None
    log_test: bool = False
    cnn_in_channels: int = 1
    cnn_channels: list[int] = None  # type: ignore[assignment]
    cnn_kernel_sizes: list[int] = None  # type: ignore[assignment]
    cnn_dilations: int | list[int] = 1
    cnn_dropout: float = 0.1
    cnn_activation: str = "relu"
    cnn_normalization: str = "instance"
    cnn_residual_scaling: float = 1.0
    cnn_causal_padding: bool = True
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: Optional[int] = None
    transformer_dropout: float = 0.1
    transformer_batch_first: bool = False
    transformer_activation: str = "relu"

    def __post_init__(self) -> None:
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 64]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 3, 3]


def load_config(path: Path = CONFIG_PATH) -> tuple[ScriptConfig, bool]:
    cfg = ScriptConfig()
    if not path.is_file():
        return cfg, False
    with path.open("r", encoding="utf-8") as fp:
        overrides = json.load(fp)
    apply_overrides(cfg, overrides)
    return cfg, True


def apply_overrides(cfg: ScriptConfig, overrides: dict) -> None:
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            continue
        if key in {"windows_path", "residuals_path", "checkpoint", "history_out"} and value is not None:
            setattr(cfg, key, Path(value))
        elif key in {"cnn_channels", "cnn_kernel_sizes"} and value is not None:
            setattr(cfg, key, [int(v) for v in value])
        else:
            setattr(cfg, key, value)


def load_or_build_windows(cfg: ScriptConfig) -> Windows:
    if cfg.windows_path.is_file():
        return WindowBuilder.load_windows(cfg.windows_path)

    builder = WindowBuilder(
        WindowConfig(
            residuals_path=cfg.residuals_path,
            output_path=cfg.windows_path,
            lookback=cfg.lookback,
            stride=cfg.stride,
            min_assets=cfg.min_assets,
            min_valid_ratio=cfg.min_valid_ratio,
            zero_as_invalid=cfg.zero_as_invalid,
            horizon=cfg.horizon,
        )
    )
    return builder.build()


def split_windows(windows: Windows, cfg: ScriptConfig) -> WindowsSplit:
    if cfg.train_end and windows.dates is not None and windows.dates.size > 0:
        classifier = WindowClassifier(windows)
        return classifier.by_date(train_end=cfg.train_end, val_end=cfg.val_end)

    total = windows.data.shape[0]
    if total < 3:
        raise ValueError(
            "Not enough windows to split into train/val/test segments.")

    train_end = max(1, int(total * cfg.train_ratio))
    val_candidate = int(total * (cfg.train_ratio + cfg.val_ratio))
    val_end = max(train_end + 1, min(val_candidate, total - 1))

    idx_all = np.arange(total)
    train_idx = idx_all[:train_end]
    val_idx = idx_all[train_end:val_end]
    test_idx = idx_all[val_end:]

    return WindowsSplit(
        train=WindowClassifier._slice(windows, train_idx),
        val=WindowClassifier._slice(windows, val_idx),
        test=WindowClassifier._slice(windows, test_idx),
    )


def make_loader(windows: Windows, batch_size: int, num_workers: int, device: torch.device) -> Optional[DataLoader]:
    dataset = ResidualsDataset(windows)
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def build_model(cfg: ScriptConfig) -> StatArbModel:
    dilations = tuple(cfg.cnn_dilations) if isinstance(
        cfg.cnn_dilations, (list, tuple)) else cfg.cnn_dilations
    cnn_cfg = CNNConfig(
        in_channels=cfg.cnn_in_channels,
        channels=tuple(cfg.cnn_channels),
        kernel_sizes=tuple(cfg.cnn_kernel_sizes),
        dilations=dilations,
        dropout=cfg.cnn_dropout,
        activation=cfg.cnn_activation,
        normalization=cfg.cnn_normalization,
        residual_scaling=cfg.cnn_residual_scaling,
        causal_padding=cfg.cnn_causal_padding,
    )
    transformer_cfg = TransformerConfig(
        d_model=cfg.transformer_d_model,
        nhead=cfg.transformer_nhead,
        num_layers=cfg.transformer_num_layers,
        dim_feedforward=cfg.transformer_dim_feedforward or 0,
        dropout=cfg.transformer_dropout,
        batch_first=cfg.transformer_batch_first,
        activation=cfg.transformer_activation,
    )
    return StatArbModel(ModelConfig(cnn=cnn_cfg, transformer=transformer_cfg))


def build_loss(cfg: ScriptConfig) -> torch.nn.Module:
    objective = cfg.objective.lower()
    if objective == "sharpe":
        return SharpeLoss()
    if objective == "meanvar":
        return MeanVarianceLoss()
    if objective == "sqrtmeansharpe":
        return SqrtMeanSharpeLoss()
    raise ValueError(f"Unsupported objective: {cfg.objective}")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")
    cfg, loaded_from_file = load_config()
    if loaded_from_file:
        logger.info("Loaded config overrides from %s", CONFIG_PATH)
    else:
        logger.info("Using default configuration (no %s found)", CONFIG_PATH)

    windows = load_or_build_windows(cfg)
    split = split_windows(windows, cfg)

    device = torch.device(cfg.device) if cfg.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(
        split.train, cfg.batch_size, cfg.num_workers, device)
    val_loader = make_loader(split.val, cfg.batch_size,
                             cfg.num_workers, device) if split.val.data.size else None
    test_loader = make_loader(split.test, cfg.batch_size,
                              cfg.num_workers, device) if split.test.data.size else None

    if train_loader is None:
        raise RuntimeError(
            "No training windows available. Check preprocessing configuration.")

    trainer_cfg = TrainerConfig(
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        device=device,
        eval_interval=cfg.eval_interval,
        dtype=parse_dtype(cfg.dtype),
        objective=cfg.objective,
        weight_scheme=cfg.weight_scheme,
        trans_cost=cfg.trans_cost,
        hold_cost=cfg.hold_cost,
        early_stopping=cfg.early_stopping,
        early_stopping_patience=cfg.early_stopping_patience,
        early_stopping_max_trials=cfg.early_stopping_max_trials,
        lr_decay=cfg.lr_decay,
        checkpoint_path=str(cfg.checkpoint) if cfg.checkpoint else None,
        force_retrain=cfg.force_retrain,
    )

    model = build_model(cfg)
    loss_module = build_loss(cfg)

    history = TrainingLoop(
        model,
        train_loader,
        val_loader,
        config=trainer_cfg,
        loss_fn=loss_module,
    )

    if history:
        logger.info("Training completed. Last metrics: %s", history[-1])
    else:
        logger.info("Training completed with empty history.")

    if cfg.history_out is not None:
        cfg.history_out.parent.mkdir(parents=True, exist_ok=True)
        with cfg.history_out.open("w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)
        logger.info("Saved history to %s", cfg.history_out)

    if cfg.log_test and test_loader is not None:
        test_metrics = evaluate(model, test_loader, build_loss(
            cfg), device=trainer_cfg.device, config=trainer_cfg)
        logger.info("Test metrics: %s", test_metrics)


if __name__ == "__main__":
    main()
