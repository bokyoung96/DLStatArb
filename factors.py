import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from loader import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class FactorModel:
    def __init__(self, n_components: int = 5, win_pca: int = 252, win_beta: int = 60):
        self.n_components = n_components
        self.win_pca = win_pca
        self.win_beta = win_beta

        self.vectors = None
        self.betas = None
        self.stocks = None

        self._residuals = None
        self._explained_variances = None

    def _prepare_window(self, returns: pd.DataFrame, length: int) -> pd.DataFrame:
        return returns.tail(length).dropna(axis=1, how="any")

    def fit_pca(self, returns: pd.DataFrame) -> None:
        window = self._prepare_window(returns, self.win_pca)
        pca = PCA(n_components=self.n_components)
        pca.fit(window)

        self.vectors = pd.DataFrame(
            pca.components_.T,
            index=window.columns,
            columns=[f"PC{i+1}" for i in range(self.n_components)]
        )
        self.explained_ratio_ = pca.explained_variance_ratio_

    def fit_betas(self, returns: pd.DataFrame) -> None:
        window = self._prepare_window(returns, self.win_beta)
        self.stocks = window.columns.intersection(self.vectors.index)

        F = window[self.stocks].to_numpy() @ self.vectors.loc[self.stocks].to_numpy()
        betas = {}
        for stock in self.stocks:
            y = window[stock].to_numpy()
            X = F
            model = LinearRegression().fit(X, y)
            betas[stock] = model.coef_

        self.betas = pd.DataFrame(betas, index=self.vectors.columns).T.loc[self.stocks]

    def residuals_at(self, today: pd.DataFrame) -> pd.Series:
        common = today.columns.intersection(self.stocks)
        Ft = today[common].to_numpy() @ self.vectors.loc[common].to_numpy()
        fitted = (self.betas.loc[common].to_numpy() @ Ft.T).ravel()
        resid = today[common].to_numpy().ravel() - fitted
        return pd.Series(resid, index=common, name=today.index[0])

    def fit_all(self, returns: pd.DataFrame) -> None:
        residuals_all = []
        explained = []

        start = self.win_pca + self.win_beta
        for t in tqdm(range(start, len(returns)), desc="Fitting windows"):
            window = returns.iloc[:t]
            today = returns.iloc[[t]]

            self.fit_pca(window)
            self.fit_betas(window)
            resid = self.residuals_at(today)

            residuals_all.append(resid)
            explained.append((today.index[0], float(np.sum(self.explained_ratio_))))

        self._residuals = pd.DataFrame(residuals_all)
        self._explained_variances = pd.DataFrame(
            explained, columns=["date", "cumulative_explained_variance"]
        ).set_index("date")

    @property
    def residuals(self) -> pd.DataFrame:
        return self._residuals

    @property
    def explained_variances(self) -> pd.DataFrame:
        return self._explained_variances


if __name__ == "__main__":
    loader = DataLoader()
    close = loader("close")
    rets = close.pct_change(fill_method=None)

    model = FactorModel(n_components=5)
    model.fit_all(rets)