"""Deprecated: universe selection moved to :mod:`quantbox.universe` (TOM-1449).

Every name resolves to the SAME object in :mod:`quantbox.universe` and emits a
``DeprecationWarning`` that names the new path. This shim stays for one minor
version.
"""

from quantbox._deprecation import moved

__getattr__ = moved("quantbox.plugins.strategies._universe", "quantbox.universe", card="TOM-1449")
