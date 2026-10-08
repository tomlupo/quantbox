"""Post-backtest analysis utilities.

The statistics that lived here moved to :mod:`quantbox.inference` (TOM-1618);
their old names here still resolve, with a ``DeprecationWarning``.
"""

from quantbox._deprecation import moved

from .parameter_grid import DEFAULT_METRICS, load_parquet_market_data, plot_heatmaps, run_grid, sweep

__getattr__ = moved(__name__, "quantbox.inference")

__all__ = [
    "DEFAULT_METRICS",
    "load_parquet_market_data",
    "plot_heatmaps",
    "run_grid",
    "sweep",
]
