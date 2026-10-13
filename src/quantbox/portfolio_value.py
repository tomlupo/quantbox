"""Removed in quantbox 0.16.0 (TOM-1455). Importing this raises ImportError naming the new home.

The valuation rule is part of the marked book: :mod:`quantbox.trade.book` (docs/adr/0011).
"""

from quantbox._removed import removed

removed(__name__)
