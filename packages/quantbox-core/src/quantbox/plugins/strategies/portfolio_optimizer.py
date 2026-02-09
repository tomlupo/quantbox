"""Mean-variance portfolio optimizer strategy plugin."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta

try:
    from scipy.optimize import minimize as scipy_minimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

_SCIPY_MSG = "scipy is required for max_sharpe and min_variance optimization"


# ---------------------------------------------------------------------------
# Private analyzer (not exported)
# ---------------------------------------------------------------------------

class _PortfolioAnalyzer:
    """Compute portfolio metrics and run optimizations on a returns matrix."""

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02,
        trading_days: int = 252,
    ) -> None:
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.n_assets = len(returns.columns)

    # -- helpers -------------------------------------------------------------

    def _cov_annual(self) -> np.ndarray:
        return self.returns.cov().values * self.trading_days

    def _mean_annual(self) -> np.ndarray:
        return self.returns.mean().values * self.trading_days

    # -- portfolio metrics ---------------------------------------------------

    def portfolio_performance(
        self, weights: np.ndarray
    ) -> tuple[float, float, float]:
        """Return (annualised return, annualised vol, Sharpe ratio)."""
        w = np.asarray(weights)
        ret = float(np.dot(self._mean_annual(), w))
        vol = float(np.sqrt(np.dot(w, np.dot(self._cov_annual(), w))))
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    def statistics(self) -> pd.DataFrame:
        """Per-asset annualised statistics."""
        mean_r = self.returns.mean() * self.trading_days
        vol = self.returns.std() * np.sqrt(self.trading_days)
        sharpe = (mean_r - self.risk_free_rate) / vol
        return pd.DataFrame(
            {"ann_return": mean_r, "ann_vol": vol, "sharpe": sharpe},
            index=self.returns.columns,
        )

    # -- optimizers ----------------------------------------------------------

    def optimize_sharpe(
        self,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> np.ndarray:
        """Maximum Sharpe ratio portfolio. Requires scipy."""
        if not HAS_SCIPY:
            raise ImportError(_SCIPY_MSG)

        n = self.n_assets

        def neg_sharpe(w: np.ndarray) -> float:
            return -self.portfolio_performance(w)[2]

        result = scipy_minimize(
            neg_sharpe,
            np.full(n, 1.0 / n),
            method="SLSQP",
            bounds=[(min_weight, max_weight)] * n,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        )
        return result.x

    def optimize_min_variance(
        self,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> np.ndarray:
        """Minimum variance portfolio. Requires scipy."""
        if not HAS_SCIPY:
            raise ImportError(_SCIPY_MSG)

        n = self.n_assets
        cov = self._cov_annual()

        def variance(w: np.ndarray) -> float:
            return float(np.dot(w, np.dot(cov, w)))

        result = scipy_minimize(
            variance,
            np.full(n, 1.0 / n),
            method="SLSQP",
            bounds=[(min_weight, max_weight)] * n,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        )
        return result.x

    def equal_weight(self) -> np.ndarray:
        """Equal 1/N allocation."""
        return np.full(self.n_assets, 1.0 / self.n_assets)

    def risk_parity(self) -> np.ndarray:
        """Inverse-volatility weights, normalised to sum to 1."""
        vol = self.returns.std().values
        inv_vol = np.where(vol > 0, 1.0 / vol, 0.0)
        total = inv_vol.sum()
        if total == 0:
            return self.equal_weight()
        return inv_vol / total

    def inverse_vol(self) -> np.ndarray:
        """Alias for risk_parity."""
        return self.risk_parity()

    # -- risk metrics --------------------------------------------------------

    def var(self, weights: np.ndarray, confidence: float = 0.95) -> float:
        """Historical Value at Risk (positive = loss)."""
        port_returns = (self.returns * weights).sum(axis=1)
        return float(-np.percentile(port_returns, (1 - confidence) * 100))

    def cvar(self, weights: np.ndarray, confidence: float = 0.95) -> float:
        """Conditional VaR / Expected Shortfall (positive = loss)."""
        port_returns = (self.returns * weights).sum(axis=1)
        threshold = -self.var(weights, confidence)
        tail = port_returns[port_returns <= threshold]
        if tail.empty:
            return self.var(weights, confidence)
        return float(-tail.mean())


# ---------------------------------------------------------------------------
# Strategy plugin
# ---------------------------------------------------------------------------

_METHODS = ("max_sharpe", "min_variance", "equal_weight", "risk_parity", "inverse_vol")


@dataclass
class PortfolioOptimizerStrategy:
    """Mean-variance portfolio optimizer with multiple allocation methods."""

    meta = PluginMeta(
        name="strategy.portfolio_optimizer.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Mean-variance portfolio optimizer (max Sharpe, min variance, risk parity, equal weight)",
        tags=("optimization", "mean-variance", "multi-asset"),
    )

    method: str = "max_sharpe"
    lookback: int = 252
    risk_free_rate: float = 0.02
    trading_days: int = 252
    min_weight: float = 0.0
    max_weight: float = 1.0
    rolling: bool = False
    rebalance_every: int = 21
    output_periods: int = 30

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute portfolio weights using the configured optimization method."""
        if params:
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        if self.method not in _METHODS:
            raise ValueError(
                f"Unknown method '{self.method}'. Choose from: {_METHODS}"
            )

        prices: pd.DataFrame = data["prices"]

        if self.rolling:
            weights_df = self._rolling_weights(prices)
        else:
            weights_df = self._single_weights(prices)

        # Build simple_weights from the last row
        latest = weights_df.iloc[-1]
        simple = latest[latest.abs() > 1e-6].to_dict()

        # Statistics and risk from the latest window
        tail = prices.iloc[-self.lookback :]
        returns = tail.pct_change().dropna()
        analyzer = _PortfolioAnalyzer(
            returns, self.risk_free_rate, self.trading_days
        )
        w_arr = latest.values
        var_val = analyzer.var(w_arr)
        cvar_val = analyzer.cvar(w_arr)

        return {
            "weights": weights_df,
            "simple_weights": simple,
            "details": {
                "statistics": analyzer.statistics(),
                "var": var_val,
                "cvar": cvar_val,
            },
        }

    # -- private helpers -----------------------------------------------------

    def _optimize(self, analyzer: _PortfolioAnalyzer) -> np.ndarray:
        """Dispatch to the selected optimization method."""
        if self.method == "max_sharpe":
            return analyzer.optimize_sharpe(self.min_weight, self.max_weight)
        if self.method == "min_variance":
            return analyzer.optimize_min_variance(self.min_weight, self.max_weight)
        if self.method == "equal_weight":
            return analyzer.equal_weight()
        if self.method == "risk_parity":
            return analyzer.risk_parity()
        # inverse_vol
        return analyzer.inverse_vol()

    def _single_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute weights once for the latest lookback window."""
        tail = prices.iloc[-self.lookback :]
        returns = tail.pct_change().dropna()
        analyzer = _PortfolioAnalyzer(
            returns, self.risk_free_rate, self.trading_days
        )
        w = self._optimize(analyzer)
        return pd.DataFrame(
            [w], columns=prices.columns, index=[prices.index[-1]]
        )

    def _rolling_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute a full weight history, rebalancing every N days."""
        dates = prices.index
        start = self.lookback
        if start >= len(dates):
            start = len(dates) - 1

        rows = []
        last_weights: Optional[np.ndarray] = None
        steps_since_rebalance = self.rebalance_every  # force first rebalance

        for i in range(start, len(dates)):
            steps_since_rebalance += 1
            if steps_since_rebalance >= self.rebalance_every or last_weights is None:
                window = prices.iloc[i - self.lookback : i]
                returns = window.pct_change().dropna()
                if len(returns) < 2:
                    last_weights = np.full(len(prices.columns), 1.0 / len(prices.columns))
                else:
                    analyzer = _PortfolioAnalyzer(
                        returns, self.risk_free_rate, self.trading_days
                    )
                    last_weights = self._optimize(analyzer)
                steps_since_rebalance = 0
            rows.append(last_weights)

        idx = dates[start : start + len(rows)]
        weights_df = pd.DataFrame(rows, columns=prices.columns, index=idx)
        return weights_df.tail(self.output_periods)
