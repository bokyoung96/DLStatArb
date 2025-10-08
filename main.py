from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from cnn import CNNConfig
from datas import DataConfig, DataPipeline
from training import (MeanVarianceLoss, SharpeLoss, SqrtMeanSharpeLoss,
                      TrainerConfig, TrainingLoop, evaluate, select_device)
from transformer import ModelConfig, StatArbModel, TransformerConfig

CONFIG_PATH = Path("train_config.json")
logger = logging.getLogger(__name__)


def load_cfg(path: Path = CONFIG_PATH) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def build_model(cfg: dict) -> StatArbModel:
    dilations = cfg.get("cnn_dilations", 1)
    dilations = tuple(dilations) if isinstance(
        dilations, (list, tuple)) else int(dilations)
    cnn_cfg = CNNConfig(
        in_channels=int(cfg.get("cnn_in_channels", 1)),
        channels=tuple(cfg.get("cnn_channels", [32, 64, 64])),
        kernel_sizes=tuple(cfg.get("cnn_kernel_sizes", [3, 3, 3])),
        dilations=dilations,
        dropout=float(cfg.get("cnn_dropout", 0.1)),
        activation=str(cfg.get("cnn_activation", "relu")),
        normalization=str(cfg.get("cnn_normalization", "instance")),
        residual_scaling=float(cfg.get("cnn_residual_scaling", 1.0)),
        causal_padding=bool(cfg.get("cnn_causal_padding", True)),
    )
    d_model = int(cfg.get("transformer_d_model", 64))
    transformer_cfg = TransformerConfig(
        d_model=d_model,
        nhead=int(cfg.get("transformer_nhead", 4)),
        num_layers=int(cfg.get("transformer_num_layers", 2)),
        dim_feedforward=int(cfg.get("transformer_dim_feedforward", d_model)),
        dropout=float(cfg.get("transformer_dropout", 0.1)),
        batch_first=bool(cfg.get("transformer_batch_first", True)),
        activation=str(cfg.get("transformer_activation", "relu")),
    )
    return StatArbModel(ModelConfig(cnn=cnn_cfg, transformer=transformer_cfg))


def build_loss(cfg: dict) -> torch.nn.Module:
    objective = str(cfg.get("objective", "sharpe")).lower()
    if objective == "sharpe":
        return SharpeLoss()
    if objective == "meanvar":
        return MeanVarianceLoss()
    if objective == "sqrtmeansharpe":
        return SqrtMeanSharpeLoss()
    raise ValueError(f"Unsupported objective: {cfg.get('objective')}")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")
    cfg = load_cfg(CONFIG_PATH)
    logger.info("Loaded config: %s", CONFIG_PATH)

    data_cfg = DataConfig(
        windows_path=Path(
            cfg.get("windows_path", "PROCESSED/residual_windows.pt")),
        residuals_path=Path(
            cfg.get("residuals_path", "DATA/residuals.parquet")),
        lookback=int(cfg.get("lookback", 30)),
        stride=int(cfg.get("stride", 1)),
        min_assets=int(cfg.get("min_assets", 25)),
        min_valid_ratio=float(cfg.get("min_valid_ratio", 1.0)),
        zero_as_invalid=bool(cfg.get("zero_as_invalid", False)),
        factor_n_components=int(cfg.get("factor_n_components", 5)),
        factor_win_pca=int(cfg.get("factor_win_pca", 252)),
        factor_win_beta=int(cfg.get("factor_win_beta", 60)),
        data_dir=(Path(cfg["data_dir"]) if cfg.get("data_dir") else None),
        horizon=int(cfg.get("horizon", 0)),
        train_end=cfg.get("train_end"),
        val_end=cfg.get("val_end"),
        train_ratio=float(cfg.get("train_ratio", 0.8)),
        val_ratio=float(cfg.get("val_ratio", 0.1)),
    )

    pipeline = DataPipeline(data_cfg)
    device = select_device(cfg.get("device"))
    artifacts = pipeline.run(batch_size=int(cfg.get("batch_size", 200)),
                             num_workers=int(cfg.get("num_workers", 2)), device=device)
    train_loader = artifacts.train_loader
    val_loader = artifacts.val_loader
    test_loader = artifacts.test_loader

    logger.info("Device=%s, dtype=%s, loaders: train=%s, val=%s, test=%s",
                device, parse_dtype(cfg.get("dtype", "float32")),
                0 if train_loader is None else len(train_loader),
                0 if val_loader is None else len(val_loader),
                0 if test_loader is None else len(test_loader))

    if train_loader is None:
        raise RuntimeError(
            "No training windows available. Check preprocessing configuration.")

    trainer_cfg = TrainerConfig(
        epochs=int(cfg.get("epochs", 100)),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        grad_clip=float(cfg.get("grad_clip", 1.0)),
        device=device,
        eval_interval=int(cfg.get("eval_interval", 1)),
        dtype=parse_dtype(cfg.get("dtype", "float32")),
        objective=str(cfg.get("objective", "sharpe")),
        weight_scheme=str(cfg.get("weight_scheme", "l1")),
        hold_cost=float(cfg.get("hold_cost", 0.0001)),
        early_stopping=bool(cfg.get("early_stopping", False)),
        early_stopping_patience=int(cfg.get("early_stopping_patience", 50)),
        early_stopping_max_trials=int(cfg.get("early_stopping_max_trials", 5)),
        lr_decay=float(cfg.get("lr_decay", 0.5)),
        checkpoint_path=(str(cfg.get("checkpoint"))
                         if cfg.get("checkpoint") else None),
        force_retrain=bool(cfg.get("force_retrain", False)),
        micro_batch_size=(int(cfg.get("micro_batch_size")) if cfg.get(
            "micro_batch_size") is not None else None),
        accumulate_steps=int(cfg.get("accumulate_steps", 1)),
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

    if cfg.get("history_out"):
        hpath = Path(cfg["history_out"])
        hpath.parent.mkdir(parents=True, exist_ok=True)
        with hpath.open("w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)
        logger.info("Saved history to %s", hpath)

    if cfg.get("model_out"):
        mpath = Path(cfg["model_out"])
        mpath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), mpath)
        logger.info("Saved final model state_dict to %s", mpath)

    if cfg.get("config_out"):
        cpath = Path(cfg["config_out"])
        cpath.parent.mkdir(parents=True, exist_ok=True)
        with cpath.open("w", encoding="utf-8") as fp:
            json.dump(cfg, fp, indent=2)
        logger.info("Saved resolved config to %s", cpath)

    if cfg.get("log_test", False) and test_loader is not None:
        test_metrics = evaluate(model, test_loader, loss_module,
                                device=trainer_cfg.device, config=trainer_cfg)
        logger.info("Test metrics: %s", test_metrics)
        if cfg.get("test_out"):
            tpath = Path(cfg["test_out"])
            tpath.parent.mkdir(parents=True, exist_ok=True)
            with tpath.open("w", encoding="utf-8") as fp:
                json.dump(test_metrics, fp, indent=2)
            logger.info("Saved test metrics to %s", tpath)


if __name__ == "__main__":
    main()
