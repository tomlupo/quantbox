"""Builtin pipelines: research (backtest, fund selection) and trade (trading, allocations to orders).

The package holds pipelines of two layers that never import each other
(ADR-0010), so it imports neither at import time. Each name below resolves on
first access through :func:`quantbox._lazy.load`: importing
``BacktestPipeline`` never imports a broker, and ``TradingPipeline`` never
imports the backtest report (TOM-1451).
"""

from __future__ import annotations

from typing import Any

from quantbox._lazy import load

__all__ = ["AllocationsToOrdersPipeline", "BacktestPipeline", "FundSelectionPipeline", "TradingPipeline"]

_LAZY = {
    "AllocationsToOrdersPipeline": ("quantbox.plugins.pipeline.alloc2orders:AllocationsToOrdersPipeline", "trade"),
    "BacktestPipeline": ("quantbox.plugins.pipeline.backtest_pipeline:BacktestPipeline", "research"),
    "FundSelectionPipeline": ("quantbox.plugins.pipeline.fund_selection:FundSelectionPipeline", "research"),
    "TradingPipeline": ("quantbox.plugins.pipeline.trading_pipeline:TradingPipeline", "trade"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    target, extra = _LAZY[name]
    value = load(target, extra=extra)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
