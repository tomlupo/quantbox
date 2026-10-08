"""Deprecation shims for import paths that moved (internal).

A moved module keeps its old path as a module whose ``__getattr__`` resolves
every name from the new home and warns once per access with a
``DeprecationWarning`` that names the new path. The object returned IS the new
one, so an old import and a new import can never disagree.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from typing import Any


def moved(old: str, *new: str, card: str = "TOM-1618") -> Callable[[str], Any]:
    """A module ``__getattr__`` for ``old`` that resolves names from the ``new`` modules, in order."""

    def __getattr__(name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(f"module {old!r} has no attribute {name!r}")
        for target in new:
            module = importlib.import_module(target)
            if hasattr(module, name):
                warnings.warn(
                    f"{old}.{name} is deprecated: import it from {target} ({card})",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return getattr(module, name)
        raise AttributeError(f"module {old!r} has no attribute {name!r} (it moved to {', '.join(new)})")

    return __getattr__
