from .crypto_trend import CryptoTrendStrategy
from .momentum_long_short import MomentumLongShortStrategy
from .carver_trend import CarverTrendStrategy
from .cross_asset_momentum import cross_asset_momentum, CrossAssetMomentumStrategy
from .crypto_regime_trend import CryptoRegimeTrendStrategy
from .weighted_avg_aggregator import WeightedAverageAggregator
from ._universe import select_universe, select_universe_duckdb, DEFAULT_STABLECOINS

__all__ = [
    "CryptoTrendStrategy",
    "MomentumLongShortStrategy",
    "CarverTrendStrategy",
    "cross_asset_momentum",
    "CrossAssetMomentumStrategy",
    "CryptoRegimeTrendStrategy",
    "WeightedAverageAggregator",
    "select_universe",
    "select_universe_duckdb",
    "DEFAULT_STABLECOINS",
]
