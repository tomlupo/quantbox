from ._universe import DEFAULT_STABLECOINS, select_universe, select_universe_duckdb
from .beglobal_strategy import BeGlobalStrategy
from .carver_trend import CarverTrendStrategy
from .cross_asset_momentum import CrossAssetMomentumStrategy, cross_asset_momentum
from .crypto_regime_trend import CryptoRegimeTrendStrategy
from .crypto_trend import CryptoTrendStrategy
from .ml_strategy import MLPredictionStrategy
from .momentum_long_short import MomentumLongShortStrategy
from .portfolio_optimizer import PortfolioOptimizerStrategy
from .weighted_avg_aggregator import WeightedAverageAggregator

__all__ = [
    "BeGlobalStrategy",
    "CryptoTrendStrategy",
    "MomentumLongShortStrategy",
    "CarverTrendStrategy",
    "cross_asset_momentum",
    "CrossAssetMomentumStrategy",
    "CryptoRegimeTrendStrategy",
    "MLPredictionStrategy",
    "PortfolioOptimizerStrategy",
    "WeightedAverageAggregator",
    "select_universe",
    "select_universe_duckdb",
    "DEFAULT_STABLECOINS",
]
