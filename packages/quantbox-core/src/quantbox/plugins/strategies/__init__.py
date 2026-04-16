"""Strategy plugins for quantbox.

Strategies compute target portfolio weights from market data. Each
strategy implements the ``StrategyPlugin`` protocol::

    strategy.run(data={"prices": df}, params={...}) -> {"weights": df, ...}

**Generic (any asset class):**

- ``DualMomentumStrategy`` — absolute + relative momentum switching
- ``TrendFollowingStrategy`` — TSMOM signal-to-weight engine
- ``VolTargetingStrategy`` — conditional volatility targeting
- ``CrossAssetMomentumStrategy`` — cross-sectional momentum, top-N, vol-parity
- ``WeightedAverageAggregator`` — weighted-average meta-strategy composition
- ``PortfolioOptimizerStrategy`` — mean-variance / risk-parity optimization

**Crypto-specific:**

- ``CryptoTrendStrategy``, ``CryptoRegimeTrendStrategy``, ``CarverTrendStrategy``
- ``MomentumLongShortStrategy``, ``MLPredictionStrategy``
- ``BeGlobalStrategy``

**Utilities:**

- ``select_universe`` / ``select_universe_duckdb`` — universe filtering
- ``CRYPTO_STABLECOINS`` — default stablecoin exclusion list for crypto
"""

from ._universe import select_universe, select_universe_duckdb
from .beglobal_strategy import BeGlobalStrategy
from .carver_trend import CarverTrendStrategy
from .cross_asset_momentum import (
    CRYPTO_STABLECOINS,
    CrossAssetMomentumStrategy,
    cross_asset_momentum,
)
from .crypto_regime_trend import CryptoRegimeTrendStrategy
from .crypto_trend import CryptoTrendStrategy
from .dual_momentum import DualMomentumStrategy
from .ml_strategy import MLPredictionStrategy
from .momentum_long_short import MomentumLongShortStrategy
from .portfolio_optimizer import PortfolioOptimizerStrategy
from .trend_following import TrendFollowingStrategy
from .vol_targeting import VolTargetingStrategy
from .weighted_avg_aggregator import WeightedAverageAggregator

# Backward compat
DEFAULT_STABLECOINS = CRYPTO_STABLECOINS

__all__ = [
    # Generic strategies (any asset class)
    "DualMomentumStrategy",
    "TrendFollowingStrategy",
    "VolTargetingStrategy",
    "CrossAssetMomentumStrategy",
    "WeightedAverageAggregator",
    "PortfolioOptimizerStrategy",
    # Crypto-specific
    "BeGlobalStrategy",
    "CryptoTrendStrategy",
    "CryptoRegimeTrendStrategy",
    "CarverTrendStrategy",
    "MomentumLongShortStrategy",
    "MLPredictionStrategy",
    # Functional wrapper
    "cross_asset_momentum",
    # Utilities
    "select_universe",
    "select_universe_duckdb",
    "CRYPTO_STABLECOINS",
    "DEFAULT_STABLECOINS",
]
