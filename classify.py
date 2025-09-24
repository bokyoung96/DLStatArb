from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np

from preprocess import Windows, ResidualsDataset, WindowBuilder

__all__ = ["WindowsSplit", "WindowClassifier"]


@dataclass(frozen=True)
class WindowsSplit:
    train: Windows
    val: Windows
    test: Windows

    def as_datasets(self, *, include_targets: bool = True) -> Dict[str, ResidualsDataset]:
        return {
            "train": ResidualsDataset(self.train, include_targets=include_targets),
            "val": ResidualsDataset(self.val, include_targets=include_targets),
            "test": ResidualsDataset(self.test, include_targets=include_targets),
        }


class WindowClassifier:
    def __init__(self, windows: Windows) -> None:
        self._w = windows
        
    @staticmethod
    def _slice(w: Windows, idx: np.ndarray) -> Windows:
        data = w.data[idx]
        mask = w.mask[idx]
        dates = w.dates[idx] if w.dates is not None else None
        targets = w.targets[idx] if w.targets is not None else None
        return Windows(
            data=data.astype(np.float32, copy=False),
            mask=mask.bool(),
            dates=dates,
            assets=w.assets,
            targets=targets,
        )

    def by_date(
        self,
        *,
        train_end: str | np.datetime64,
        val_end: Optional[str | np.datetime64] = None,
    ) -> WindowsSplit:
        if self._w.dates is None:
            raise ValueError("Windows.dates is None; date-based split requires dates.")
        def to_dt64(x: str | np.datetime64) -> np.datetime64:
            return np.datetime64(x)
        t_end = to_dt64(train_end)
        v_end = to_dt64(val_end) if val_end is not None else None
        dates = self._w.dates.astype("datetime64[ns]")
        train_idx = np.where(dates <= np.datetime64(t_end, "ns"))[0]
        if v_end is not None:
            val_idx = np.where((dates > np.datetime64(t_end, "ns")) & (dates <= np.datetime64(v_end, "ns")))[0]
            test_idx = np.where(dates > np.datetime64(v_end, "ns"))[0]
        else:
            val_idx = np.empty(0, dtype=int)
            test_idx = np.where(dates > np.datetime64(t_end, "ns"))[0]
        return WindowsSplit(
            train=self._slice(self._w, train_idx),
            val=self._slice(self._w, val_idx),
            test=self._slice(self._w, test_idx),
        )


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    wins = WindowBuilder.load_windows(base / "PROCESSED" / "residual_windows.pt")

    W = wins.data.shape[0]
    train_end = wins.dates[int(W * 0.6)]
    val_end = wins.dates[int(W * 0.8)]

    split = WindowClassifier(wins).by_date(train_end=train_end, val_end=val_end)