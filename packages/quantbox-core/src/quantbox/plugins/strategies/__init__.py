from .crypto_trend import CryptoTrendStrategy
from .momentum_long_short import MomentumLongShortStrategy
from .carver_trend import CarverTrendStrategy
from .cross_asset_momentum import cross_asset_momentum
from .weighted_avg_aggregator import WeightedAverageAggregator

__all__ = [
    "CryptoTrendStrategy",
    "MomentumLongShortStrategy",
    "CarverTrendStrategy",
    "cross_asset_momentum",
    "WeightedAverageAggregator",
]
