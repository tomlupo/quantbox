"""Deprecated: the stochastic price models moved to :mod:`quantbox.montecarlo.models` (TOM-1451).

Every name resolves to the SAME object and emits a ``DeprecationWarning`` that
names the new path. ``from quantbox.simulation import GBM`` still works with no
warning. This shim stays for one minor version.
"""

from quantbox._deprecation import moved

__getattr__ = moved("quantbox.simulation.models", "quantbox.montecarlo.models", card="TOM-1451")
