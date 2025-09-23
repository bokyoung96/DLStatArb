from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


__all__ = [
    "WindowConfig",
    "Windows",
    "WindowBuilder",
    "ResidualsDataset",
]


@dataclass(frozen=True)
class WindowConfig:
    residuals_path: str | Path
    output_path: Optional[str | Path] = None
    lookback: int = 21
    stride: int = 5
    min_assets: int = 25
    min_valid_ratio: float = 1.0
    zero_as_invalid: bool = False
    horizon: int = 0

    def __post_init__(self) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        if self.min_assets <= 0:
            raise ValueError("min_assets must be positive.")
        ratio = float(self.min_valid_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError("min_valid_ratio must be in (0, 1].")
        object.__setattr__(self, "min_valid_ratio", ratio)
        if self.horizon < 0:
            raise ValueError("horizon must be non-negative.")
        if not str(self.residuals_path):
            raise ValueError("residuals_path must be a non-empty path.")


@dataclass(frozen=True)
class Windows:
    data: np.ndarray
    mask: torch.BoolTensor
    dates: Optional[np.ndarray] = None
    assets: Optional[Tuple[str, ...]] = None
    targets: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.data.shape)

    @property
    def data_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.data)

    @property
    def targets_tensor(self) -> Optional[torch.Tensor]:
        if self.targets is None:
            return None
        return torch.from_numpy(self.targets)


class ResidualsDataset(Dataset):
    def __init__(self, windows: Windows, *, include_targets: bool = True) -> None:
        self._panel = torch.from_numpy(np.ascontiguousarray(windows.data))
        self._mask = windows.mask.bool()
        if include_targets and windows.targets is not None:
            self._targets = torch.from_numpy(np.ascontiguousarray(windows.targets))
        else:
            self._targets = None

    def __len__(self) -> int:
        return int(self._panel.shape[0])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.BoolTensor] | tuple[torch.Tensor, torch.BoolTensor, torch.Tensor]:
        panel = self._panel[index]
        mask = self._mask[index]
        if self._targets is None:
            return panel, mask
        return panel, mask, self._targets[index]


class WindowBuilder:
    def __init__(self, cfg: WindowConfig) -> None:
        self.cfg = cfg

    def load_residuals(self) -> pd.DataFrame:
        p = Path(self.cfg.residuals_path)
        df = pd.read_parquet(p)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def build(
        self,
        residuals: pd.DataFrame | np.ndarray | None = None,
        *,
        dates: Optional[np.ndarray] = None,
        assets: Optional[Iterable[str]] = None,
    ) -> Windows:
        data, dates, assets_tuple = self._get_inputs(
            residuals=residuals,
            dates=dates,
            assets=assets,
        )

        cumsum_windows, base_mask = self._get_windows(data)

        idx, asset_mask = self._select_indices(base_mask)
        cumsum_windows = cumsum_windows[idx]

        features = self._build_features(cumsum_windows, asset_mask)
        targets = self._build_targets(data, asset_mask, idx)

        window_dates: Optional[np.ndarray] = None
        lookback = self.cfg.lookback
        horizon = self.cfg.horizon
        if dates is not None:
            effective_dates = dates[lookback + horizon :]
            window_dates = effective_dates[idx]

        windows = Windows(
            data=features.astype(np.float32, copy=False),
            mask=torch.as_tensor(asset_mask, dtype=torch.bool),
            dates=window_dates,
            assets=assets_tuple,
            targets=targets,
        )

        if self.cfg.output_path is not None:
            self.save_windows(windows, out_path=Path(self.cfg.output_path))
        return windows

    def save_windows(self, windows: Windows, out_path: Optional[Path] = None) -> Path:
        if out_path is None:
            if self.cfg.output_path is None:
                raise ValueError("No output_path specified.")
            out_path = Path(self.cfg.output_path)

        out_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(
            {
            "data": windows.data,
            "mask": windows.mask,
            "targets": windows.targets,
            "dates": windows.dates,
            "assets": windows.assets,
            },
            out_path,
        )
        print(f"[INFO] Saved processed windows to {out_path}")
        return out_path

    @staticmethod
    def load_windows(path: Path) -> Windows:
        obj = torch.load(path, weights_only=False)
        return Windows(
            data=obj["data"],
            mask=obj["mask"],
            targets=obj.get("targets"),
            dates=obj.get("dates"),
            assets=obj.get("assets"),
        )

    def _get_windows(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lookback = self.cfg.lookback
        horizon = self.cfg.horizon
        num_steps, _ = data.shape

        if num_steps <= lookback + horizon:
            raise ValueError("Time series must be longer than lookback + horizon.")

        valid = np.isfinite(data)
        if self.cfg.zero_as_invalid:
            valid &= data != 0.0

        win_valid = np.lib.stride_tricks.sliding_window_view(valid, lookback, axis=0)

        ratio = float(self.cfg.min_valid_ratio)
        if ratio >= 1.0:
            mask_all = win_valid.all(axis=2)
        else:
            required = int(np.ceil(lookback * ratio))
            valid_counts = win_valid.sum(axis=2)
            mask_all = valid_counts >= required

        clean = np.nan_to_num(data, nan=0.0, copy=False).astype(np.float32, copy=False)
        win_view = np.lib.stride_tricks.sliding_window_view(clean, lookback, axis=0)
        csum = np.cumsum(win_view, axis=2)

        num_windows = num_steps - lookback - horizon
        if num_windows <= 0:
            raise ValueError("No valid windows after applying horizon constraint.")
        csum = csum[: num_windows]
        mask = mask_all[: num_windows]
        return csum, mask

    def _build_features(self, cumsum_windows: np.ndarray, asset_mask: np.ndarray) -> np.ndarray:
        features = cumsum_windows
        features = np.where(np.isfinite(features), features, 0.0)
        features = np.where(asset_mask[..., None], features, 0.0)
        return features

    def _build_targets(
        self,
        residuals: np.ndarray,
        asset_mask: np.ndarray,
        indices: np.ndarray,
    ) -> Optional[np.ndarray]:
        future = residuals[self.cfg.lookback + self.cfg.horizon :]
        if future.shape[0] == 0:
            return None
        target = future[indices]
        finite = np.isfinite(target)
        combined_mask = asset_mask & finite
        target = np.where(combined_mask, target, 0.0).astype(np.float32, copy=False)
        return target

    def _get_inputs(
        self,
        residuals: pd.DataFrame | np.ndarray | None,
        *,
        dates: Optional[np.ndarray],
        assets: Optional[Iterable[str]],
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple[str, ...]]]:
        if residuals is None:
            df = self.load_residuals()
        elif isinstance(residuals, pd.DataFrame):
            df = residuals
        else:
            df = None

        if df is not None:
            assets = tuple(df.columns.astype(str))
            dates = df.index.to_numpy()
            data = df.to_numpy(dtype=np.float32, copy=True)
        else:
            data = np.asarray(residuals, dtype=np.float32)
            assets = tuple(str(a) for a in assets)

        if data.ndim != 2:
            raise ValueError("residuals must be a 2D array shaped as (time, assets).")

        finite = np.isfinite(data)
        data = np.where(finite, data, np.nan)

        num_steps, _ = data.shape
        if num_steps <= self.cfg.lookback + self.cfg.horizon:
            raise ValueError("Time series must be longer than lookback + horizon.")
        return data, dates, assets

    def _select_windows(self, asset_mask: np.ndarray) -> np.ndarray:
        return asset_mask.sum(axis=1) >= self.cfg.min_assets

    def _select_indices(self, asset_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep_mask = self._select_windows(asset_mask)
        valid_indices = np.nonzero(keep_mask)[0]
        if valid_indices.size == 0:
            raise ValueError(
                "No window satisfies the constraints. Check min_assets and stride settings."
            )
        idx = valid_indices[:: self.cfg.stride]
        return idx, asset_mask[idx]


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent / "DATA" / "residuals.parquet"
    out_path = Path(__file__).resolve().parent / "PROCESSED" / "residual_windows.pt"

    cfg = WindowConfig(
        residuals_path=data_path,
        output_path=out_path,
        lookback=21,
        stride=5,
        min_assets=25,
        min_valid_ratio=1.0,
        zero_as_invalid=True,
        horizon=0,
    )

    builder = WindowBuilder(cfg)
    windows = builder.build()

    pts = WindowBuilder.load_windows(out_path)