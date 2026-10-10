"""Removed in quantbox 0.13.0 (TOM-1711). Importing this raises ImportError naming the new home.

The scheduled book is part of the engine seam: :mod:`quantbox.engine.schedule` (docs/adr/0008).
"""

from quantbox._removed import removed

removed(__name__)
