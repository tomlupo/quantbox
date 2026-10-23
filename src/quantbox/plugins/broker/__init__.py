"""Builtin brokers.

The package imports no broker at import time. Each name below resolves on
first access through :func:`quantbox._lazy.load`, so the paper brokers work
without a venue client, and naming one broker never imports the others
(TOM-1451).
"""

from __future__ import annotations

from typing import Any

from quantbox._lazy import load

__all__ = [
    "BinanceBroker",
    "BinanceFuturesBroker",
    "BinancePaperBrokerStub",
    "FuturesPaperBroker",
    "HyperliquidBroker",
    "IBKRBroker",
    "IBKRPaperBrokerStub",
    "KrakenBroker",
    "SimPaperBroker",
]

_PKG = "quantbox.plugins.broker"
_LAZY = {
    "BinanceBroker": f"{_PKG}.binance:BinanceBroker",
    "BinanceFuturesBroker": f"{_PKG}.binance_futures:BinanceFuturesBroker",
    "BinancePaperBrokerStub": f"{_PKG}.binance_stub:PaperBrokerStub",
    "FuturesPaperBroker": f"{_PKG}.futures_paper:FuturesPaperBroker",
    "HyperliquidBroker": f"{_PKG}.hyperliquid:HyperliquidBroker",
    "IBKRBroker": f"{_PKG}.ibkr:IBKRBroker",
    "IBKRPaperBrokerStub": f"{_PKG}.ibkr_stub:PaperBrokerStub",
    "KrakenBroker": f"{_PKG}.kraken:KrakenBroker",
    "SimPaperBroker": f"{_PKG}.sim:SimPaperBroker",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = load(_LAZY[name], extra="trade")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
