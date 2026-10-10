"""Removed in quantbox 0.13.0 (TOM-1457). Importing this raises ImportError naming the new homes.

The statistics are in :mod:`quantbox.inference`, the gates in :mod:`quantbox.gates`
and the parameter-grid sweep in :mod:`quantbox.sweep` (TOM-1618, ADR-0009).
"""

from quantbox._removed import removed

removed(__name__)
