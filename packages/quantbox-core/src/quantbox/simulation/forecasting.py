"""Multi-term expected returns forecasting.

Methods: historical, bootstrap, parametric (normal), GARCH,
mean reversion (OU), Bayesian shrinkage, correlation-adjusted portfolio.

Ported from quantlabnew/src/market-simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from arch import arch_model as _arch_model

    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


class Horizon(Enum):
    """Standard forecast horizons (trading days)."""

    DAILY = 1
    WEEKLY = 5
    MONTHLY = 21
    QUARTERLY = 63
    SEMI_ANNUAL = 126
    ANNUAL = 252
    TWO_YEAR = 504
    FIVE_YEAR = 1260
    TEN_YEAR = 2520


@dataclass
class ForecastResult:
    """Container for a single-horizon return forecast."""

    asset: str
    horizon: int
    horizon_name: str
    expected_return: float
    volatility: float
    confidence_intervals: dict[float, tuple[float, float]]
    percentiles: dict[int, float]
    distribution_params: dict
    method: str

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "horizon": self.horizon,
            "horizon_name": self.horizon_name,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "ci_95_lower": self.confidence_intervals.get(0.95, (None, None))[0],
            "ci_95_upper": self.confidence_intervals.get(0.95, (None, None))[1],
            "ci_99_lower": self.confidence_intervals.get(0.99, (None, None))[0],
            "ci_99_upper": self.confidence_intervals.get(0.99, (None, None))[1],
            "method": self.method,
        }


@dataclass
class MultiHorizonForecast:
    """Container for multi-horizon forecasts."""

    asset: str
    forecasts: dict[int, ForecastResult]
    term_structure: pd.DataFrame

    def get_horizon(self, horizon: int | Horizon) -> ForecastResult:
        if isinstance(horizon, Horizon):
            horizon = horizon.value
        return self.forecasts[horizon]

    def plot_term_structure(self):
        return self.term_structure


class ReturnForecaster:
    """Multi-term expected returns forecaster."""

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.04):
        self.returns = returns
        self.asset_names = list(returns.columns)
        self.risk_free_rate = risk_free_rate
        self._daily_rf = risk_free_rate / 252

    def forecast_single_horizon(
        self,
        asset: str,
        horizon: int | Horizon,
        method: str = "historical",
        confidence_levels: list[float] | None = None,
        n_simulations: int = 10000,
    ) -> ForecastResult:
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]

        if isinstance(horizon, Horizon):
            horizon_name = horizon.name
            horizon = horizon.value
        else:
            horizon_name = self._horizon_to_name(horizon)

        daily_returns = self.returns[asset].values

        if method == "historical":
            result = self._historical_forecast(daily_returns, horizon, confidence_levels)
        elif method == "bootstrap":
            result = self._bootstrap_forecast(daily_returns, horizon, confidence_levels, n_simulations)
        elif method == "parametric":
            result = self._parametric_forecast(daily_returns, horizon, confidence_levels)
        elif method == "garch":
            result = self._garch_forecast(daily_returns, horizon, confidence_levels, n_simulations)
        else:
            raise ValueError(f"Unknown method: {method}")

        return ForecastResult(
            asset=asset,
            horizon=horizon,
            horizon_name=horizon_name,
            expected_return=result["expected_return"],
            volatility=result["volatility"],
            confidence_intervals=result["confidence_intervals"],
            percentiles=result["percentiles"],
            distribution_params=result["distribution_params"],
            method=method,
        )

    def forecast_multi_horizon(
        self,
        asset: str,
        horizons: list[int | Horizon] | None = None,
        method: str = "historical",
        **kwargs,
    ) -> MultiHorizonForecast:
        if horizons is None:
            horizons = list(Horizon)

        forecasts = {}
        for h in horizons:
            horizon_days = h.value if isinstance(h, Horizon) else h
            forecast = self.forecast_single_horizon(asset, h, method, **kwargs)
            forecasts[horizon_days] = forecast

        term_data = []
        for h, f in sorted(forecasts.items()):
            term_data.append(
                {
                    "horizon_days": h,
                    "horizon_name": f.horizon_name,
                    "expected_return": f.expected_return,
                    "annualized_return": f.expected_return * (252 / h),
                    "volatility": f.volatility,
                    "annualized_volatility": f.volatility * np.sqrt(252 / h),
                    "sharpe_ratio": (f.expected_return - self._daily_rf * h) / f.volatility if f.volatility > 0 else 0,
                    "ci_95_lower": f.confidence_intervals.get(0.95, (0, 0))[0],
                    "ci_95_upper": f.confidence_intervals.get(0.95, (0, 0))[1],
                }
            )

        return MultiHorizonForecast(
            asset=asset,
            forecasts=forecasts,
            term_structure=pd.DataFrame(term_data),
        )

    def forecast_all_assets(
        self,
        horizons: list[int | Horizon] | None = None,
        method: str = "historical",
        **kwargs,
    ) -> dict[str, MultiHorizonForecast]:
        return {asset: self.forecast_multi_horizon(asset, horizons, method, **kwargs) for asset in self.asset_names}

    def expected_return_with_mean_reversion(
        self,
        asset: str,
        horizon: int,
        long_term_return: float,
        mean_reversion_speed: float = 0.2,
    ) -> ForecastResult:
        """Forecast with mean reversion (OU process)."""
        if not HAS_SCIPY:
            raise ImportError("scipy is required for mean reversion forecast.")

        daily_returns = self.returns[asset].values
        current_return = np.mean(daily_returns[-21:]) * 252
        theta = mean_reversion_speed
        t = horizon / 252

        expected_annual = current_return + (long_term_return - current_return) * (1 - np.exp(-theta * t))
        expected_horizon = expected_annual * (horizon / 252)

        daily_vol = np.std(daily_returns)
        long_term_var = (daily_vol**2) / (2 * theta / 252)
        horizon_vol = np.sqrt(long_term_var * (1 - np.exp(-2 * theta * t)))

        confidence_intervals = {}
        for level in [0.90, 0.95, 0.99]:
            z = _scipy_stats.norm.ppf((1 + level) / 2)
            confidence_intervals[level] = (expected_horizon - z * horizon_vol, expected_horizon + z * horizon_vol)

        return ForecastResult(
            asset=asset,
            horizon=horizon,
            horizon_name=self._horizon_to_name(horizon),
            expected_return=expected_horizon,
            volatility=horizon_vol,
            confidence_intervals=confidence_intervals,
            percentiles={
                5: expected_horizon - 1.645 * horizon_vol,
                50: expected_horizon,
                95: expected_horizon + 1.645 * horizon_vol,
            },
            distribution_params={
                "current_return": current_return,
                "long_term_return": long_term_return,
                "mean_reversion_speed": mean_reversion_speed,
            },
            method="mean_reversion",
        )

    def bayesian_shrinkage_forecast(
        self,
        asset: str,
        horizon: int,
        prior_return: float = 0.06,
        prior_weight: float = 0.5,
    ) -> ForecastResult:
        """Bayesian shrinkage combining prior and historical."""
        if not HAS_SCIPY:
            raise ImportError("scipy is required for Bayesian shrinkage forecast.")

        daily_returns = self.returns[asset].values
        hist_mean = np.mean(daily_returns) * 252
        shrunk_return = prior_weight * prior_return + (1 - prior_weight) * hist_mean
        expected_horizon = shrunk_return * (horizon / 252)

        daily_vol = np.std(daily_returns)
        horizon_vol = daily_vol * np.sqrt(horizon)

        confidence_intervals = {}
        for level in [0.90, 0.95, 0.99]:
            z = _scipy_stats.norm.ppf((1 + level) / 2)
            confidence_intervals[level] = (expected_horizon - z * horizon_vol, expected_horizon + z * horizon_vol)

        return ForecastResult(
            asset=asset,
            horizon=horizon,
            horizon_name=self._horizon_to_name(horizon),
            expected_return=expected_horizon,
            volatility=horizon_vol,
            confidence_intervals=confidence_intervals,
            percentiles={
                5: expected_horizon - 1.645 * horizon_vol,
                50: expected_horizon,
                95: expected_horizon + 1.645 * horizon_vol,
            },
            distribution_params={
                "historical_return": hist_mean,
                "prior_return": prior_return,
                "prior_weight": prior_weight,
                "shrunk_return": shrunk_return,
            },
            method="bayesian_shrinkage",
        )

    def generate_fan_chart_data(
        self,
        asset: str,
        max_horizon: int = 252,
        step: int = 5,
        percentiles: list[int] | None = None,
        n_simulations: int = 10000,
    ) -> pd.DataFrame:
        """Generate data for fan chart visualization."""
        if percentiles is None:
            percentiles = [5, 10, 25, 50, 75, 90, 95]

        daily_returns = self.returns[asset].values
        mu = np.mean(daily_returns)
        sigma = np.std(daily_returns)
        rng = np.random.default_rng(42)

        horizons = list(range(1, max_horizon + 1, step))
        simulated = np.zeros((n_simulations, len(horizons)))
        for i, h in enumerate(horizons):
            random_returns = rng.normal(mu, sigma, (n_simulations, h))
            simulated[:, i] = np.sum(random_returns, axis=1)

        data: dict = {"horizon": horizons}
        for p in percentiles:
            data[f"p{p}"] = np.percentile(simulated, p, axis=0)
        return pd.DataFrame(data)

    def correlation_adjusted_forecast(
        self,
        weights: dict[str, float],
        horizon: int,
        method: str = "historical",
    ) -> ForecastResult:
        """Forecast portfolio returns accounting for correlations."""
        if not HAS_SCIPY:
            raise ImportError("scipy is required for correlation-adjusted forecast.")

        forecasts = {}
        for asset in weights:
            forecasts[asset] = self.forecast_single_horizon(asset, horizon, method)

        weight_arr = np.array([weights[a] for a in weights])
        expected_arr = np.array([forecasts[a].expected_return for a in weights])
        portfolio_expected = float(np.sum(weight_arr * expected_arr))

        horizon_returns = self.returns[list(weights.keys())].rolling(horizon).sum().dropna()
        cov_matrix = horizon_returns.cov().values
        portfolio_var = weight_arr @ cov_matrix @ weight_arr
        portfolio_vol = float(np.sqrt(portfolio_var))

        confidence_intervals = {}
        for level in [0.90, 0.95, 0.99]:
            z = _scipy_stats.norm.ppf((1 + level) / 2)
            confidence_intervals[level] = (
                portfolio_expected - z * portfolio_vol,
                portfolio_expected + z * portfolio_vol,
            )

        return ForecastResult(
            asset="Portfolio",
            horizon=horizon,
            horizon_name=self._horizon_to_name(horizon),
            expected_return=portfolio_expected,
            volatility=portfolio_vol,
            confidence_intervals=confidence_intervals,
            percentiles={
                5: portfolio_expected - 1.645 * portfolio_vol,
                50: portfolio_expected,
                95: portfolio_expected + 1.645 * portfolio_vol,
            },
            distribution_params={"weights": weights},
            method=f"portfolio_{method}",
        )

    # -- private methods --

    def _historical_forecast(self, returns, horizon, confidence_levels):
        if horizon > 1:
            n = len(returns)
            horizon_returns = np.array([np.sum(returns[i : i + horizon]) for i in range(n - horizon + 1)])
        else:
            horizon_returns = returns

        expected = np.mean(horizon_returns)
        volatility = np.std(horizon_returns)
        confidence_intervals = {}
        for level in confidence_levels:
            alpha = (1 - level) / 2
            confidence_intervals[level] = (
                float(np.percentile(horizon_returns, alpha * 100)),
                float(np.percentile(horizon_returns, (1 - alpha) * 100)),
            )

        percentiles = {p: float(np.percentile(horizon_returns, p)) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]}

        dist_params: dict = {"n_observations": len(horizon_returns)}
        if HAS_SCIPY:
            dist_params["skewness"] = float(_scipy_stats.skew(horizon_returns))
            dist_params["kurtosis"] = float(_scipy_stats.kurtosis(horizon_returns))

        return {
            "expected_return": float(expected),
            "volatility": float(volatility),
            "confidence_intervals": confidence_intervals,
            "percentiles": percentiles,
            "distribution_params": dist_params,
        }

    def _bootstrap_forecast(self, returns, horizon, confidence_levels, n_simulations):
        rng = np.random.default_rng(42)
        block_size = min(21, horizon)
        n_blocks = (horizon + block_size - 1) // block_size
        simulated_returns = np.zeros(n_simulations)

        for i in range(n_simulations):
            path = []
            for _ in range(n_blocks):
                start = rng.integers(0, len(returns) - block_size)
                path.extend(returns[start : start + block_size])
            simulated_returns[i] = np.sum(path[:horizon])

        expected = np.mean(simulated_returns)
        volatility = np.std(simulated_returns)
        confidence_intervals = {}
        for level in confidence_levels:
            alpha = (1 - level) / 2
            confidence_intervals[level] = (
                float(np.percentile(simulated_returns, alpha * 100)),
                float(np.percentile(simulated_returns, (1 - alpha) * 100)),
            )

        percentiles = {p: float(np.percentile(simulated_returns, p)) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]}
        return {
            "expected_return": float(expected),
            "volatility": float(volatility),
            "confidence_intervals": confidence_intervals,
            "percentiles": percentiles,
            "distribution_params": {"n_simulations": n_simulations, "block_size": block_size},
        }

    def _parametric_forecast(self, returns, horizon, confidence_levels):
        if not HAS_SCIPY:
            raise ImportError("scipy is required for parametric forecast.")

        daily_mu = np.mean(returns)
        daily_sigma = np.std(returns)
        expected = daily_mu * horizon
        volatility = daily_sigma * np.sqrt(horizon)

        confidence_intervals = {}
        for level in confidence_levels:
            z = _scipy_stats.norm.ppf((1 + level) / 2)
            confidence_intervals[level] = (float(expected - z * volatility), float(expected + z * volatility))

        percentiles = {
            p: float(expected + _scipy_stats.norm.ppf(p / 100) * volatility) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }
        return {
            "expected_return": float(expected),
            "volatility": float(volatility),
            "confidence_intervals": confidence_intervals,
            "percentiles": percentiles,
            "distribution_params": {
                "daily_mu": float(daily_mu),
                "daily_sigma": float(daily_sigma),
                "distribution": "normal",
            },
        }

    def _garch_forecast(self, returns, horizon, confidence_levels, n_simulations):
        if not HAS_ARCH:
            raise ImportError("The 'arch' package is required for GARCH forecasting.")
        if not HAS_SCIPY:
            raise ImportError("scipy is required for GARCH forecast confidence intervals.")

        model = _arch_model(returns * 100, vol="Garch", p=1, q=1)
        result = model.fit(disp="off")
        forecasts = result.forecast(horizon=horizon, method="simulation", simulations=n_simulations)

        mean_returns = result.params["mu"] / 100
        simulated_variance = forecasts.variance.values[-1, :]
        expected_vol = np.sqrt(np.mean(simulated_variance)) / 100

        expected = mean_returns * horizon
        volatility = expected_vol * np.sqrt(horizon)

        confidence_intervals = {}
        for level in confidence_levels:
            z = _scipy_stats.norm.ppf((1 + level) / 2)
            confidence_intervals[level] = (float(expected - z * volatility), float(expected + z * volatility))

        percentiles = {
            p: float(expected + _scipy_stats.norm.ppf(p / 100) * volatility) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }
        return {
            "expected_return": float(expected),
            "volatility": float(volatility),
            "confidence_intervals": confidence_intervals,
            "percentiles": percentiles,
            "distribution_params": {
                "omega": float(result.params.get("omega", 0)),
                "alpha": float(result.params.get("alpha[1]", 0)),
                "beta": float(result.params.get("beta[1]", 0)),
            },
        }

    def _horizon_to_name(self, horizon: int) -> str:
        for h in Horizon:
            if h.value == horizon:
                return h.name
        if horizon <= 5:
            return f"{horizon}D"
        elif horizon <= 21:
            return f"{horizon // 5}W"
        elif horizon <= 63:
            return f"{horizon // 21}M"
        elif horizon <= 252:
            return f"{horizon // 63}Q"
        return f"{horizon // 252}Y"


__all__ = [
    "Horizon",
    "ForecastResult",
    "MultiHorizonForecast",
    "ReturnForecaster",
]
