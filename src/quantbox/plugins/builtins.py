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
    KrakenBroker,
    SimPaperBroker,
)
from .datasources import (
    BinanceDataPlugin,
    BinanceFuturesDataPlugin,
    HyperliquidCachedDataPlugin,
    HyperliquidDataPlugin,
    KrakenDataPlugin,
    LocalFileDataPlugin,
    SyntheticDataPlugin,
)
from .features import CrossSectionalFeatures, TechnicalFeatures
from .monitor import DrawdownMonitor, SignalDecayMonitor
from .overlays import CorrGrossCapOverlay, RegimeReweightOverlay, ReversalDeriskOverlay
from .pipeline import AllocationsToOrdersPipeline, BacktestPipeline, FundSelectionPipeline, TradingPipeline
from .publisher import TelegramPublisher
from .rebalancing import FuturesRebalancer, StandardRebalancer
from .risk import (
    DrawdownControlRiskManager,
    FactorExposureRiskManager,
    StressTestRiskManager,
    TradingRiskManager,
)
from .strategies import (
    AltcoinCrashBounceStrategy,
    BeGlobalStrategy,
    CarryStrategy,
    CarverTrendStrategy,
    CarverTrendV2Strategy,
    CrossAssetMomentumStrategy,
    CryptoRegimeTrendStrategy,
    CryptoTrendStrategy,
    DualMomentumStrategy,
    EthMeanReversion24h,
    FrozenWeightsStrategy,
    HmmRegimeAllocation,
    MLPredictionStrategy,
    MomentumLongShortStrategy,
    PortfolioOptimizerStrategy,
    StaticWeightsStrategy,
    TrendCatcherSimpleStrategy,
    TrendCatcherStrategy,
    TrendFollowingStrategy,
    VolMatchedBuyHoldStrategy,
    VolTargetingStrategy,
)
from .strategies.weighted_avg_aggregator import WeightedAverageAggregator
from .validation import (
    BenchmarkValidation,
    DeflatedSharpeBLPValidation,
    RegimeValidation,
    StatisticalValidation,
    TurnoverValidation,
    WalkForwardValidation,
)


def _map(*classes):
    return {c.meta.name: c for c in classes}


def builtins() -> dict[str, dict[str, type]]:
    return {
        "pipeline": _map(FundSelectionPipeline, AllocationsToOrdersPipeline, TradingPipeline, BacktestPipeline),
        "data": _map(
            LocalFileDataPlugin,
            BinanceDataPlugin,
            BinanceFuturesDataPlugin,
            HyperliquidDataPlugin,
            HyperliquidCachedDataPlugin,
            KrakenDataPlugin,
            SyntheticDataPlugin,
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
            KrakenBroker,
        ),
        "publisher": _map(TelegramPublisher),
        "risk": _map(
            TradingRiskManager,
            StressTestRiskManager,
            FactorExposureRiskManager,
            DrawdownControlRiskManager,
        ),
        "strategy": _map(
            AltcoinCrashBounceStrategy,
            BeGlobalStrategy,
            CarryStrategy,
            CryptoTrendStrategy,
            CarverTrendStrategy,
            CarverTrendV2Strategy,
            MomentumLongShortStrategy,
            CrossAssetMomentumStrategy,
            CryptoRegimeTrendStrategy,
            DualMomentumStrategy,
            EthMeanReversion24h,
            FrozenWeightsStrategy,
            HmmRegimeAllocation,
            MLPredictionStrategy,
            PortfolioOptimizerStrategy,
            StaticWeightsStrategy,
            TrendCatcherStrategy,
            TrendCatcherSimpleStrategy,
            TrendFollowingStrategy,
            VolMatchedBuyHoldStrategy,
            VolTargetingStrategy,
            WeightedAverageAggregator,
        ),
        "rebalancing": _map(StandardRebalancer, FuturesRebalancer),
        "feature": _map(TechnicalFeatures, CrossSectionalFeatures),
        "monitor": _map(DrawdownMonitor, SignalDecayMonitor),
        "overlay": _map(ReversalDeriskOverlay, RegimeReweightOverlay, CorrGrossCapOverlay),
        "validation": _map(
            WalkForwardValidation,
            StatisticalValidation,
            DeflatedSharpeBLPValidation,
            TurnoverValidation,
            RegimeValidation,
            BenchmarkValidation,
        ),
    }
