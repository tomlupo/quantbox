"""Deprecation shims for import paths that moved (internal).

A moved module keeps its old path as a module whose ``__getattr__`` resolves
every name from the new home and warns once per access with a
``DeprecationWarning`` that names the new path. The object returned IS the new
one, so an old import and a new import can never disagree.

A module that keeps live code of its own and lost only SOME names passes
``names``: only those old names resolve (each to its new name, which may differ),
and every other missing name raises the usual ``AttributeError``.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable, Mapping
from typing import Any


def moved(old: str, *new: str, card: str = "TOM-1618", names: Mapping[str, str] | None = None) -> Callable[[str], Any]:
    """A module ``__getattr__`` for ``old`` that resolves names from the ``new`` modules, in order.

    ``names`` maps each moved old name to its new name; without it every name moves unchanged.
    """

    def __getattr__(name: str) -> Any:
        if name.startswith("__") or (names is not None and name not in names):
            raise AttributeError(f"module {old!r} has no attribute {name!r}")
        new_name = names[name] if names is not None else name
        found = [
            (t, getattr(importlib.import_module(t), new_name))
            for t in new
            if hasattr(importlib.import_module(t), new_name)
        ]
        if not found:
            raise AttributeError(f"module {old!r} has no attribute {name!r} (it moved to {', '.join(new)})")
        # Name the module that DEFINES it, not one that merely imports it.
        target, obj = next(((t, o) for t, o in found if getattr(o, "__module__", t) == t), found[0])
        what = "it" if new_name == name else new_name
        warnings.warn(
            f"{old}.{name} is deprecated: import {what} from {target} ({card})", DeprecationWarning, stacklevel=2
        )
        return obj

    return __getattr__
