from ._universe import DEFAULT_STABLECOINS, select_universe, select_universe_duckdb
from .altcoin_crash_bounce import AltcoinCrashBounceStrategy
from .beglobal_strategy import BeGlobalStrategy
from .carry import CarryStrategy
from .carver_trend import CarverTrendStrategy
from .cross_asset_momentum import CrossAssetMomentumStrategy, cross_asset_momentum
from .crypto_regime_trend import CryptoRegimeTrendStrategy
from .crypto_trend import CryptoTrendStrategy
from .eth_mean_reversion_24h import EthMeanReversion24h
from .hmm_regime_allocation import HmmRegimeAllocation
from .ml_strategy import MLPredictionStrategy
from .momentum_long_short import MomentumLongShortStrategy
from .portfolio_optimizer import PortfolioOptimizerStrategy
from .static_weights import StaticWeightsStrategy
from .trend_catcher import TrendCatcherStrategy
from .trend_catcher_simple import TrendCatcherSimpleStrategy
from .vol_matched_buy_hold import VolMatchedBuyHoldStrategy
from .weighted_avg_aggregator import WeightedAverageAggregator

__all__ = [
    "AltcoinCrashBounceStrategy",
    "BeGlobalStrategy",
    "CarryStrategy",
    "EthMeanReversion24h",
    "HmmRegimeAllocation",
    "CryptoTrendStrategy",
    "MomentumLongShortStrategy",
    "CarverTrendStrategy",
    "cross_asset_momentum",
    "CrossAssetMomentumStrategy",
    "CryptoRegimeTrendStrategy",
    "MLPredictionStrategy",
    "PortfolioOptimizerStrategy",
    "StaticWeightsStrategy",
    "TrendCatcherStrategy",
    "TrendCatcherSimpleStrategy",
    "VolMatchedBuyHoldStrategy",
    "WeightedAverageAggregator",
    "select_universe",
    "select_universe_duckdb",
    "DEFAULT_STABLECOINS",
]
