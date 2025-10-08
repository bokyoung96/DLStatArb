import logging
from pathlib import Path

import numpy as np
import pandas as pd
from loader import DataLoader
from sklearn.decomposition import PCA
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


class FactorModel:
    def __init__(self, n_components: int = 5, win_pca: int = 252, win_beta: int = 60, data_dir: str | Path | None = None):
        self.n_components = n_components
        self.win_pca = win_pca
        self.win_beta = win_beta
        self.data_dir = data_dir

        self.vectors = None
        self.betas = None
        self.stocks = None

        self._residuals = None
        self._explained_variances = None

    def _prepare_window(self, returns: pd.DataFrame, length: int) -> pd.DataFrame:
        return returns.tail(length).dropna(axis=1, how="any")

    def fit_pca(self, window: pd.DataFrame) -> None:
        pca = PCA(n_components=self.n_components)
        pca.fit(window)

        self.vectors = pd.DataFrame(
            pca.components_.T,
            index=window.columns,
            columns=[f"PC{i+1}" for i in range(self.n_components)]
        )
        self.explained_ratio_ = pca.explained_variance_ratio_

    def fit_betas(self, window: pd.DataFrame, common: pd.Index) -> None:
        R = window[common].to_numpy()
        V = self.vectors.loc[common].to_numpy()
        F = R @ V

        # NOTE: OLS: Beta^T = (F^T F)^{-1} F^T R
        G = F.T @ F
        RHS = F.T @ R
        Beta_T = np.linalg.solve(G, RHS)
        Beta = Beta_T.T

        self.betas = pd.DataFrame(
            Beta, index=common, columns=self.vectors.columns)

    def residuals_at(self, today: pd.DataFrame, common: pd.Index) -> pd.Series:
        r_t = today[common].to_numpy().ravel()
        f_t = (today[common].to_numpy() @
               self.vectors.loc[common].to_numpy()).ravel()
        fitted = (self.betas.loc[common].to_numpy() @ f_t)
        resid = r_t - fitted
        return pd.Series(resid, index=common, name=today.index[0])

    def fit_all(self) -> None:
        returns_df = self.returns.dropna(how='all')

        residuals_all = []
        explained = []

        start = max(self.win_pca, self.win_beta)
        for t in tqdm(range(start, len(returns_df)), desc="Fitting windows"):
            win_pca = returns_df.iloc[t -
                                      self.win_pca:t].dropna(axis=1, how="any")
            win_beta = returns_df.iloc[t -
                                       self.win_beta:t].dropna(axis=1, how="any")
            today = returns_df.iloc[[t]].dropna(axis=1, how="any")

            self.fit_pca(win_pca)

            common = win_pca.columns.intersection(
                win_beta.columns).intersection(today.columns)

            if len(common) <= self.n_components:
                continue

            self.fit_betas(win_beta, common)
            resid = self.residuals_at(today, common)
            residuals_all.append(resid)

            cum_var = float(np.sum(self.explained_ratio_))
            explained.append((today.index[0], cum_var))

        self._residuals = pd.DataFrame(residuals_all)
        self._explained_variances = (
            pd.DataFrame(explained, columns=[
                         "date", "cumulative_explained_variance"])
            .set_index("date")
        )

    def save(self, path: str | Path = 'DATA') -> None:
        if self._residuals is None or self._explained_variances is None:
            self.fit_all()

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._residuals.to_parquet(path / "residuals.parquet")
        self._explained_variances.to_parquet(
            path / "explained_variances.parquet")

    @property
    def returns(self) -> pd.DataFrame:
        loader = DataLoader(self.data_dir)
        close = loader("close")
        close = close.sort_index()
        res = close.pct_change(fill_method=None)
        res = res.replace([np.inf, -np.inf], np.nan)
        return res

    @property
    def residuals(self) -> pd.DataFrame:
        return self._residuals

    @property
    def explained_variances(self) -> pd.DataFrame:
        return self._explained_variances
