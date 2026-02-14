"""Built-in plugin registry.

These plugins are shipped inside the core package. External plugins can still be
installed via entry points and will be merged by the PluginRegistry.
"""

from __future__ import annotations

from .broker import (
    BinanceBroker,
    BinanceFuturesBroker,
    BinancePaperBrokerStub,
    FuturesPaperBroker,
    HyperliquidBroker,
    IBKRBroker,
    IBKRPaperBrokerStub,
    SimPaperBroker,
)
from .datasources import (
    BinanceDataPlugin,
    BinanceFuturesDataPlugin,
    HyperliquidDataPlugin,
    LocalFileDataPlugin,
    SyntheticDataPlugin,
)
from .pipeline import AllocationsToOrdersPipeline, BacktestPipeline, FundSelectionPipeline, TradingPipeline
from .publisher import TelegramPublisher
from .rebalancing import FuturesRebalancer, StandardRebalancer
from .risk import StressTestRiskManager, TradingRiskManager
from .strategies import (
    AltcoinCrashBounceStrategy,
    BeGlobalStrategy,
    CarverTrendStrategy,
    CrossAssetMomentumStrategy,
    CryptoRegimeTrendStrategy,
    CryptoTrendStrategy,
    MLPredictionStrategy,
    MomentumLongShortStrategy,
    PortfolioOptimizerStrategy,
)
from .strategies.weighted_avg_aggregator import WeightedAverageAggregator


def _map(*classes):
    return {c.meta.name: c for c in classes}


def builtins() -> dict[str, dict[str, type]]:
    return {
        "pipeline": _map(FundSelectionPipeline, AllocationsToOrdersPipeline, TradingPipeline, BacktestPipeline),
        "data": _map(
            LocalFileDataPlugin, BinanceDataPlugin, BinanceFuturesDataPlugin, HyperliquidDataPlugin, SyntheticDataPlugin
        ),
        "broker": _map(
            SimPaperBroker,
            FuturesPaperBroker,
            IBKRPaperBrokerStub,
            BinancePaperBrokerStub,
            IBKRBroker,
            BinanceBroker,
            BinanceFuturesBroker,
            HyperliquidBroker,
        ),
        "publisher": _map(TelegramPublisher),
        "risk": _map(TradingRiskManager, StressTestRiskManager),
        "strategy": _map(
            AltcoinCrashBounceStrategy,
            BeGlobalStrategy,
            CryptoTrendStrategy,
            CarverTrendStrategy,
            MomentumLongShortStrategy,
            CrossAssetMomentumStrategy,
            CryptoRegimeTrendStrategy,
            MLPredictionStrategy,
            PortfolioOptimizerStrategy,
            WeightedAverageAggregator,
        ),
        "rebalancing": _map(StandardRebalancer, FuturesRebalancer),
    }
