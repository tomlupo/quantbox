"""Deprecated path: the parameter-grid sweep lives in :mod:`quantbox.sweep` (TOM-1618).

Every name (``sweep``, ``run_grid``, ``plot_heatmaps``, ``align_market_data``,
``load_parquet_market_data``, ``DEFAULT_METRICS``, ...) resolves to the same
object in :mod:`quantbox.sweep`, with a ``DeprecationWarning``.
"""

from quantbox._deprecation import moved

__getattr__ = moved(__name__, "quantbox.sweep")
