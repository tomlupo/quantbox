"""Deprecated: the Monte Carlo simulator moved to :mod:`quantbox.montecarlo.engine` (TOM-1451).

Every name resolves to the SAME object and emits a ``DeprecationWarning`` that
names the new path. ``from quantbox.simulation import MarketSimulator`` still
works with no warning. This shim stays for one minor version.
"""

from quantbox._deprecation import moved

__getattr__ = moved("quantbox.simulation.engine", "quantbox.montecarlo.engine", card="TOM-1451")
