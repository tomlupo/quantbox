"""Multi-asset Monte Carlo simulation engine.

Orchestrates correlated simulations across multiple stochastic models.

Ported from quantlabnew/src/market-simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import GBM, BaseModel


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""

    n_paths: int = 10000
    n_steps: int = 252
    dt: float = 1 / 252
    random_state: int | None = None


@dataclass
class SimulationResult:
    """Results container for market simulation.

    Attributes:
        prices: shape ``(n_assets, n_paths, n_steps + 1)``
        returns: shape ``(n_assets, n_paths, n_steps)``
    """

    prices: np.ndarray
    returns: np.ndarray
    asset_names: list[str]
    config: SimulationConfig
    correlation_matrix: np.ndarray | None = None

    def get_terminal_prices(self) -> pd.DataFrame:
        return pd.DataFrame({name: self.prices[i, :, -1] for i, name in enumerate(self.asset_names)})

    def get_terminal_returns(self) -> pd.DataFrame:
        total_returns = self.prices[:, :, -1] / self.prices[:, :, 0] - 1
        return pd.DataFrame({name: total_returns[i] for i, name in enumerate(self.asset_names)})

    def get_path_statistics(self) -> pd.DataFrame:
        stats = []
        for i, name in enumerate(self.asset_names):
            terminal = self.prices[i, :, -1] / self.prices[i, :, 0] - 1
            stats.append(
                {
                    "asset": name,
                    "mean_return": np.mean(terminal),
                    "median_return": np.median(terminal),
                    "std_return": np.std(terminal),
                    "skewness": self._skewness(terminal),
                    "kurtosis": self._kurtosis(terminal),
                    "var_95": np.percentile(terminal, 5),
                    "var_99": np.percentile(terminal, 1),
                    "cvar_95": np.mean(terminal[terminal <= np.percentile(terminal, 5)]),
                    "max_drawdown_mean": np.mean(self._max_drawdowns(self.prices[i])),
                    "sharpe_ratio": np.mean(terminal) / np.std(terminal) if np.std(terminal) > 0 else 0,
                }
            )
        return pd.DataFrame(stats).set_index("asset")

    def get_percentile_paths(self, asset: str, percentiles: list[float] | None = None) -> pd.DataFrame:
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        idx = self.asset_names.index(asset)
        prices = self.prices[idx]
        return pd.DataFrame({f"p{p}": np.percentile(prices, p, axis=0) for p in percentiles})

    def _skewness(self, x):
        n = len(x)
        m, s = np.mean(x), np.std(x)
        return np.sum(((x - m) / s) ** 3) / n if s > 0 else 0

    def _kurtosis(self, x):
        n = len(x)
        m, s = np.mean(x), np.std(x)
        return np.sum(((x - m) / s) ** 4) / n - 3 if s > 0 else 0

    def _max_drawdowns(self, prices):
        running_max = np.maximum.accumulate(prices, axis=1)
        drawdowns = (prices - running_max) / running_max
        return np.min(drawdowns, axis=1)


class MarketSimulator:
    """Multi-asset market simulator with correlated returns."""

    def __init__(
        self,
        models: dict[str, BaseModel] | None = None,
        initial_prices: dict[str, float] | None = None,
        correlation_matrix: np.ndarray | None = None,
    ):
        self.models = models or {}
        self.initial_prices = initial_prices or {}
        self.correlation_matrix = correlation_matrix
        self._cholesky = None
        if correlation_matrix is not None:
            self._cholesky = np.linalg.cholesky(correlation_matrix)

    def add_asset(self, name: str, model: BaseModel, initial_price: float = 100.0) -> MarketSimulator:
        self.models[name] = model
        self.initial_prices[name] = initial_price
        return self

    def set_correlation_matrix(self, corr_matrix: np.ndarray) -> MarketSimulator:
        n_assets = len(self.models)
        if corr_matrix.shape != (n_assets, n_assets):
            raise ValueError(f"Correlation matrix must be {n_assets}x{n_assets}")
        self.correlation_matrix = corr_matrix
        self._cholesky = np.linalg.cholesky(corr_matrix)
        return self

    def simulate(self, config: SimulationConfig | None = None) -> SimulationResult:
        config = config or SimulationConfig()
        rng = np.random.default_rng(config.random_state)

        asset_names = list(self.models.keys())
        n_assets = len(asset_names)

        Z = rng.standard_normal((n_assets, config.n_paths, config.n_steps))
        if self._cholesky is not None:
            for t in range(config.n_steps):
                Z[:, :, t] = self._cholesky @ Z[:, :, t]

        all_prices = np.zeros((n_assets, config.n_paths, config.n_steps + 1))
        all_returns = np.zeros((n_assets, config.n_paths, config.n_steps))

        for i, name in enumerate(asset_names):
            model = self.models[name]
            S0 = self.initial_prices.get(name, 100.0)
            params = model.params
            dt = config.dt

            if hasattr(params, "mu") and hasattr(params, "sigma"):
                drift = (params.mu - 0.5 * params.sigma**2) * dt
                diffusion = params.sigma * np.sqrt(dt)
                log_returns = drift + diffusion * Z[i]

                if hasattr(params, "jump_intensity"):
                    lam = params.jump_intensity
                    N = rng.poisson(lam * dt, (config.n_paths, config.n_steps))
                    J = rng.normal(params.jump_mean, params.jump_std, (config.n_paths, config.n_steps))
                    log_returns += N * J

                log_prices = np.zeros((config.n_paths, config.n_steps + 1))
                log_prices[:, 0] = np.log(S0)
                log_prices[:, 1:] = np.log(S0) + np.cumsum(log_returns, axis=1)

                all_prices[i] = np.exp(log_prices)
                all_returns[i] = np.exp(log_returns) - 1
            else:
                all_prices[i] = model.simulate(S0, config.n_steps, config.n_paths, config.random_state)
                all_returns[i] = np.diff(all_prices[i], axis=1) / all_prices[i, :, :-1]

        return SimulationResult(
            prices=all_prices,
            returns=all_returns,
            asset_names=asset_names,
            config=config,
            correlation_matrix=self.correlation_matrix,
        )

    @classmethod
    def from_historical_data(
        cls,
        prices: pd.DataFrame,
        model_type: str = "gbm",
        lookback_days: int = 252,
    ) -> MarketSimulator:
        """Create a simulator calibrated to historical price data."""
        returns = prices.pct_change().dropna().tail(lookback_days)
        models: dict[str, BaseModel] = {}
        initial_prices: dict[str, float] = {}

        for col in prices.columns:
            asset_returns = returns[col].values
            if model_type == "gbm":
                models[col] = GBM.fit(asset_returns)
            elif model_type == "garch":
                from .models import GARCH

                models[col] = GARCH.fit(asset_returns)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            initial_prices[col] = float(prices[col].iloc[-1])

        corr_matrix = returns.corr().values
        return cls(models=models, initial_prices=initial_prices, correlation_matrix=corr_matrix)


def generate_correlated_returns(
    n_assets: int,
    n_steps: int,
    n_paths: int,
    means: np.ndarray,
    stds: np.ndarray,
    correlation_matrix: np.ndarray,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate correlated multivariate returns.

    Returns array of shape ``(n_assets, n_paths, n_steps)``.
    """
    rng = np.random.default_rng(random_state)
    L = np.linalg.cholesky(correlation_matrix)
    Z = rng.standard_normal((n_assets, n_paths, n_steps))

    for t in range(n_steps):
        Z[:, :, t] = L @ Z[:, :, t]

    returns = np.zeros_like(Z)
    for i in range(n_assets):
        returns[i] = means[i] + stds[i] * Z[i]
    return returns


__all__ = [
    "SimulationConfig",
    "SimulationResult",
    "MarketSimulator",
    "generate_correlated_returns",
]
