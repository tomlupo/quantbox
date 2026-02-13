from .binance_data import BinanceDataFetcher, MarketDataSnapshot
from .binance_data_plugin import BinanceDataPlugin
from .binance_futures_data import BinanceFuturesDataFetcher
from .binance_futures_data_plugin import BinanceFuturesDataPlugin
from .hyperliquid_data_plugin import HyperliquidDataPlugin
from .local_file_data import LocalFileDataPlugin
from .synthetic_data import SyntheticDataPlugin

# Backward-compat alias: the old DuckDBParquetData class was replaced by
# LocalFileDataPlugin (same functionality, better name). This alias keeps
# existing configs and imports working. Use LocalFileDataPlugin for new code.
DuckDBParquetData = LocalFileDataPlugin

__all__ = [
    "BinanceDataFetcher",
    "BinanceDataPlugin",
    "BinanceFuturesDataFetcher",
    "BinanceFuturesDataPlugin",
    "DuckDBParquetData",
    "HyperliquidDataPlugin",
    "LocalFileDataPlugin",
    "MarketDataSnapshot",
    "SyntheticDataPlugin",
]
