"""Removed in quantbox 0.16.0 (TOM-1455). Importing this raises ImportError naming the new home.

The package moved to :mod:`quantbox.trade.reconciliation` (docs/adr/0011). A submodule
import (``quantbox.reconciliation.ledger``) raises here too, with the same message.
"""

from quantbox._removed import removed

removed(__name__)
