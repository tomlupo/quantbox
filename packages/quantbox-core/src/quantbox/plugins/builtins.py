"""Built-in plugin registry.

These plugins are shipped inside the core package. External plugins can still be
installed via entry points and will be merged by the PluginRegistry.
"""

from __future__ import annotations

from typing import Dict, Type

from ..contracts import PipelinePlugin, BrokerPlugin, DataPlugin, PublisherPlugin, RiskPlugin
from .pipeline import AllocationsToOrdersPipeline, BacktestPipeline, FundSelectionPipeline, TradingPipeline
from .datasources import BinanceDataPlugin, BinanceFuturesDataPlugin, LocalFileDataPlugin
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
from .publisher import TelegramPublisher
from .risk import TradingRiskManager
from .strategies import (
    BeGlobalStrategy,
    CryptoTrendStrategy,
    CarverTrendStrategy,
    MomentumLongShortStrategy,
    CrossAssetMomentumStrategy,
    CryptoRegimeTrendStrategy,
    MLPredictionStrategy,
    PortfolioOptimizerStrategy,
)
from .strategies.weighted_avg_aggregator import WeightedAverageAggregator
from .rebalancing import StandardRebalancer, FuturesRebalancer


def _map(*classes):
    return {c.meta.name: c for c in classes}


def builtins() -> Dict[str, Dict[str, Type]]:
    return {
        "pipeline": _map(FundSelectionPipeline, AllocationsToOrdersPipeline, TradingPipeline, BacktestPipeline),
        "data": _map(LocalFileDataPlugin, BinanceDataPlugin, BinanceFuturesDataPlugin),
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
        "risk": _map(TradingRiskManager),
        "strategy": _map(
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
