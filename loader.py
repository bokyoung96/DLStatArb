import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self, data_dir: str | Path | None = None) -> None:
        base = Path(__file__).resolve().parent
        self._dir = Path(data_dir) if data_dir is not None else base / "DATA"
        self._valid = {"open", "high", "low", "close", "volume"}
        self._all_na: set[str] = set()

    def _path_for(self, name: str) -> Path:
        return self._dir / f"{name}.parquet"

    def __call__(self, data_name: str) -> pd.DataFrame:
        df = pd.read_parquet(self._path_for(data_name))
        df = df.apply(pd.to_numeric, errors="coerce")
        self._all_na |= set(df.columns[df.isna().all()])
        cols = [c for c in df.columns if c not in self._all_na]
        return df[cols].dropna(axis=1, how="all")

    def available(self) -> list[str]:
        present = {p.stem for p in self._dir.glob("*.parquet")}
        return sorted(self._valid & present)


if __name__ == "__main__":
    loader = DataLoader()
