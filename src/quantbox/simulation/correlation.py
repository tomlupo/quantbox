"""Deprecated: correlation structures moved to :mod:`quantbox.montecarlo.correlation` (TOM-1451).

Every name resolves to the SAME object and emits a ``DeprecationWarning`` that
names the new path. ``from quantbox.simulation import CorrelationEngine`` still
works with no warning. This shim stays for one minor version.
"""

from quantbox._deprecation import moved

__getattr__ = moved("quantbox.simulation.correlation", "quantbox.montecarlo.correlation", card="TOM-1451")
