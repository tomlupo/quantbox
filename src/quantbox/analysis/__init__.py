"""Post-backtest analysis utilities.

Lightweight helpers that operate on strategy outputs (weights, portfolios)
without being plugins themselves. Reusable across research projects.
"""

from .parameter_grid import DEFAULT_METRICS, load_parquet_market_data, plot_heatmaps, run_grid, sweep

__all__ = ["DEFAULT_METRICS", "load_parquet_market_data", "plot_heatmaps", "run_grid", "sweep"]
