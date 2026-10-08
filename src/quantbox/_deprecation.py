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
        found = [
            (t, getattr(importlib.import_module(t), name)) for t in new if hasattr(importlib.import_module(t), name)
        ]
        if not found:
            raise AttributeError(f"module {old!r} has no attribute {name!r} (it moved to {', '.join(new)})")
        # Name the module that DEFINES it, not one that merely imports it.
        target, obj = next(((t, o) for t, o in found if getattr(o, "__module__", t) == t), found[0])
        warnings.warn(f"{old}.{name} is deprecated: import it from {target} ({card})", DeprecationWarning, stacklevel=2)
        return obj

    return __getattr__
