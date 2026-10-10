"""Builtin data sources.

The package imports no data source at import time. Each name below resolves on
first access through :func:`quantbox._lazy.load`, so a base install can use
``SyntheticDataPlugin`` or ``LocalFileDataPlugin`` without the ``[data]``
extra, and a source that needs it (Binance, Hyperliquid) raises
``MissingExtraError`` naming ``[data]`` (TOM-1451).
"""

from __future__ import annotations

from typing import Any

from quantbox._lazy import load

__all__ = [
    "BinanceDataFetcher",
    "BinanceDataPlugin",
    "BinanceFuturesDataFetcher",
    "BinanceFuturesDataPlugin",
    "DuckDBParquetData",
    "HyperliquidCachedDataPlugin",
    "HyperliquidDataPlugin",
    "KrakenDataFetcher",
    "KrakenDataPlugin",
    "LocalFileDataPlugin",
    "MarketDataSnapshot",
    "SyntheticDataPlugin",
]

_PKG = "quantbox.plugins.datasources"
_LAZY = {
    "BinanceDataFetcher": f"{_PKG}.binance_data:BinanceDataFetcher",
    "MarketDataSnapshot": f"{_PKG}.binance_data:MarketDataSnapshot",
    "BinanceDataPlugin": f"{_PKG}.binance_data_plugin:BinanceDataPlugin",
    "BinanceFuturesDataFetcher": f"{_PKG}.binance_futures_data:BinanceFuturesDataFetcher",
    "BinanceFuturesDataPlugin": f"{_PKG}.binance_futures_data_plugin:BinanceFuturesDataPlugin",
    "HyperliquidCachedDataPlugin": f"{_PKG}.hyperliquid_cached_data_plugin:HyperliquidCachedDataPlugin",
    "HyperliquidDataPlugin": f"{_PKG}.hyperliquid_data_plugin:HyperliquidDataPlugin",
    "KrakenDataFetcher": f"{_PKG}.kraken_data:KrakenDataFetcher",
    "KrakenDataPlugin": f"{_PKG}.kraken_data_plugin:KrakenDataPlugin",
    "LocalFileDataPlugin": f"{_PKG}.local_file_data:LocalFileDataPlugin",
    "SyntheticDataPlugin": f"{_PKG}.synthetic_data:SyntheticDataPlugin",
    # Backward-compat alias: the old DuckDBParquetData class was replaced by
    # LocalFileDataPlugin (same functionality, better name). This alias keeps
    # existing configs and imports working. Use LocalFileDataPlugin for new code.
    "DuckDBParquetData": f"{_PKG}.local_file_data:LocalFileDataPlugin",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = load(_LAZY[name], extra="data")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
