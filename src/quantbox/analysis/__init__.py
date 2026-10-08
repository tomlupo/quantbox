"""Deprecated package: its contents moved in TOM-1618.

- The statistics (DSR, Newey-West, factor regression, ``require_finite``) are in
  :mod:`quantbox.inference`; the gates in :mod:`quantbox.gates`.
- The parameter-grid sweep (``sweep``, ``run_grid``, ``plot_heatmaps``,
  ``load_parquet_market_data``, ``DEFAULT_METRICS``) is in :mod:`quantbox.sweep`.

Every old name here and in the old submodules (``analysis.dsr``,
``analysis.hac``, ``analysis.gates``, ``analysis.parameter_grid``) still
resolves to the same object, with a ``DeprecationWarning``.
"""

from quantbox._deprecation import moved

__getattr__ = moved(__name__, "quantbox.sweep", "quantbox.inference")
