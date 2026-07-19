"""Post-backtest analysis utilities.

Lightweight helpers that operate on strategy outputs (weights, portfolios)
without being plugins themselves. Reusable across research projects.
"""

from .dsr import (
    DSRResult,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
    expected_max_sr,
    sr_estimator_std,
)
from .hac import factor_regression, newey_west_auto_lags, newey_west_tstat, require_finite
from .parameter_grid import DEFAULT_METRICS, load_parquet_market_data, plot_heatmaps, run_grid, sweep

__all__ = [
    "DEFAULT_METRICS",
    "DSRResult",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_returns",
    "expected_max_sr",
    "factor_regression",
    "load_parquet_market_data",
    "newey_west_auto_lags",
    "newey_west_tstat",
    "plot_heatmaps",
    "require_finite",
    "run_grid",
    "sr_estimator_std",
    "sweep",
]
