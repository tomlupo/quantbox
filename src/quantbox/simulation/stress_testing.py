"""Deprecated: stress testing moved to :mod:`quantbox.montecarlo.stress_testing` (TOM-1451).

Every name resolves to the SAME object and emits a ``DeprecationWarning`` that
names the new path. ``from quantbox.simulation import StressTestEngine`` still
works with no warning. This shim stays for one minor version.
"""

from quantbox._deprecation import moved

__getattr__ = moved("quantbox.simulation.stress_testing", "quantbox.montecarlo.stress_testing", card="TOM-1451")
