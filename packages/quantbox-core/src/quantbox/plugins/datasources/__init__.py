from .binance_data import BinanceDataFetcher, MarketDataSnapshot
from .binance_data_plugin import BinanceDataPlugin
from .binance_futures_data import BinanceFuturesDataFetcher
from .binance_futures_data_plugin import BinanceFuturesDataPlugin
from .local_file_data import LocalFileDataPlugin
from .synthetic_data import SyntheticDataPlugin

# Backward compat alias
DuckDBParquetData = LocalFileDataPlugin

__all__ = [
    "BinanceDataFetcher",
    "BinanceDataPlugin",
    "BinanceFuturesDataFetcher",
    "BinanceFuturesDataPlugin",
    "DuckDBParquetData",
    "LocalFileDataPlugin",
    "MarketDataSnapshot",
    "SyntheticDataPlugin",
]
