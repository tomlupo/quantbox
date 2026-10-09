"""The ONE way a lower layer reaches a higher one at run time (TOM-1451, ADR-0010).

quantbox is one distribution in four layers: ``core`` < ``plugins`` <
``research``, ``trade``. A module imports only the layers below it. Where core
must still reach an upper-layer module when a command runs (the CLI's research
commands, the registry's builtin plugins, the runner's optional exports), it
names the target as a string and calls :func:`load`. That keeps the import out
of the static graph, so the layer contracts hold, and it turns a missing
third-party package into a :class:`~quantbox.exceptions.MissingExtraError` that
names the extra to install.

Every call site is found with ``rg "_lazy import load|_lazy.load"``. No other
dynamic-import trick is used in core.
"""

from __future__ import annotations

import importlib
from typing import Any

from .exceptions import MissingExtraError

__all__ = ["load"]


def load(target: str, *, extra: str | None = None) -> Any:
    """Import ``"package.module"`` or ``"package.module:attr.path"`` and return the module or attribute.

    ``extra`` names the quantbox extra that carries the target's third-party
    dependencies. When the import fails because a module OUTSIDE quantbox is
    missing, and ``extra`` is given, :class:`MissingExtraError` names that extra.
    A missing quantbox module, or a missing attribute, is a bug and raises as is.
    """
    module_name, _, attr = target.partition(":")
    try:
        obj: Any = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if extra is None or missing == "quantbox" or missing.startswith("quantbox."):
            raise
        raise MissingExtraError(extra, module_name, missing) from exc
    for part in filter(None, attr.split(".")):
        obj = getattr(obj, part)
    return obj
