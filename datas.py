from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from classify import WindowClassifier, WindowsSplit
from factors import FactorModel
from preprocess import ResidualsDataset, WindowBuilder, WindowConfig, Windows
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class DataConfig:
    windows_path: Path
    residuals_path: Path
    lookback: int
    stride: int
    min_assets: int
    min_valid_ratio: float
    zero_as_invalid: bool
    horizon: int
    factor_n_components: int = 5
    factor_win_pca: int = 252
    factor_win_beta: int = 60
    data_dir: Optional[Path] = None
    train_end: Optional[str] = None
    val_end: Optional[str] = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1


@dataclass(frozen=True)
class DataArtifacts:
    windows: Windows
    split: WindowsSplit
    train_loader: Optional[DataLoader]
    val_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader]


class DataPipeline:
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg

    def ensure_residuals(self) -> Path:
        if self.cfg.residuals_path.is_file():
            return self.cfg.residuals_path
        out_dir = self.cfg.residuals_path.parent
        model = FactorModel(
            n_components=self.cfg.factor_n_components,
            win_pca=self.cfg.factor_win_pca,
            win_beta=self.cfg.factor_win_beta,
            data_dir=str(
                self.cfg.data_dir) if self.cfg.data_dir is not None else None,
        )
        model.save(out_dir)
        return self.cfg.residuals_path

    def build_windows(self) -> Windows:
        if self.cfg.windows_path.is_file():
            return WindowBuilder.load_windows(self.cfg.windows_path)
        self.ensure_residuals()
        builder = WindowBuilder(
            WindowConfig(
                residuals_path=self.cfg.residuals_path,
                output_path=self.cfg.windows_path,
                lookback=self.cfg.lookback,
                stride=self.cfg.stride,
                min_assets=self.cfg.min_assets,
                min_valid_ratio=self.cfg.min_valid_ratio,
                zero_as_invalid=self.cfg.zero_as_invalid,
                horizon=self.cfg.horizon,
            )
        )
        return builder.build()

    def split_windows(self, windows: Windows) -> WindowsSplit:
        classifier = WindowClassifier(windows)
        if self.cfg.train_end and windows.dates is not None:
            return classifier.by_date(train_end=self.cfg.train_end, val_end=self.cfg.val_end)

        total = int(windows.data.shape[0])
        train_end = max(1, int(total * self.cfg.train_ratio))
        val_end = min(max(train_end + 1, int(total *
                      (self.cfg.train_ratio + self.cfg.val_ratio))), total)

        idx = np.arange(total)
        return WindowsSplit(
            train=WindowClassifier._slice(windows, idx[:train_end]),
            val=WindowClassifier._slice(windows, idx[train_end:val_end]),
            test=WindowClassifier._slice(windows, idx[val_end:]),
        )

    def make_loader(self, windows: Windows, batch_size: int, num_workers: int, device: torch.device) -> Optional[DataLoader]:
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

    def run(self, *, batch_size: int, num_workers: int, device: torch.device) -> "DataArtifacts":
        windows = self.build_windows()
        split = self.split_windows(windows)
        train_loader = self.make_loader(
            split.train, batch_size, num_workers, device)
        val_loader = self.make_loader(
            split.val, batch_size, num_workers, device) if split.val.data.size else None
        test_loader = self.make_loader(
            split.test, batch_size, num_workers, device) if split.test.data.size else None
        return DataArtifacts(
            windows=windows,
            split=split,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    cfg = DataConfig(
        windows_path=base / "PROCESSED" / "residual_windows.pt",
        residuals_path=base / "DATA" / "residuals.parquet",
        lookback=30,
        stride=1,
        min_assets=25,
        min_valid_ratio=1.0,
        zero_as_invalid=False,
        horizon=0,
        factor_n_components=5,
        factor_win_pca=252,
        factor_win_beta=60,
        data_dir=None,
        train_end=None,
        val_end=None,
        train_ratio=0.8,
        val_ratio=0.1,
    )

    pipe = DataPipeline(cfg)
    device = torch.device("cpu")
    artifacts = pipe.run(batch_size=32, num_workers=2, device=device)

    def size(x):
        return 0 if x is None else len(x)

    summary = {
        "train_batches": size(artifacts.train_loader),
        "val_batches": size(artifacts.val_loader),
        "test_batches": size(artifacts.test_loader),
        "window_shape": artifacts.windows.data.shape,
    }
    if artifacts.train_loader is not None:
        batch = next(iter(artifacts.train_loader))
        if len(batch) == 3:
            panel, mask, target = batch
            summary.update({
                "sample_panel_shape": tuple(panel.shape),
                "sample_mask_shape": tuple(mask.shape),
                "sample_target_shape": tuple(target.shape),
            })
    print(summary)
