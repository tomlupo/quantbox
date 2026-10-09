"""Builtin publishers.

The package imports no publisher at import time; ``TelegramPublisher``
resolves on first access through :func:`quantbox._lazy.load` and needs the
``[trade]`` extra (httpx) (TOM-1451).
"""

from __future__ import annotations

from typing import Any

from quantbox._lazy import load

__all__ = ["TelegramPublisher"]

_LAZY = {"TelegramPublisher": "quantbox.plugins.publisher.telegram:TelegramPublisher"}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = load(_LAZY[name], extra="trade")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
