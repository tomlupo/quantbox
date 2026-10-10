"""Built-in plugin table: plugin name -> ``"module:Class"``, resolved on use (TOM-1451).

These plugins ship inside quantbox. External plugins are installed via entry
points and merged by :class:`~quantbox.registry.PluginRegistry`.

The table holds strings, not classes, and imports nothing: registering the
builtins must not import every broker and pipeline. The registry imports a
plugin's module the first time its class is asked for
(:func:`quantbox._lazy.load`), so looking up a backtest pipeline never imports
a broker. ``tests/test_layer_edges.py`` checks every entry resolves to a class
whose ``meta.name`` is its key.
"""

from __future__ import annotations

from quantbox._lazy import load

__all__ = ["BUILTIN_PLUGINS", "builtins"]

_P = "quantbox.plugins"

#: kind -> {plugin name: "module:Class"}. Order is the registration order.
BUILTIN_PLUGINS: dict[str, dict[str, str]] = {
    "pipeline": {
        "fund_selection.simple.v1": f"{_P}.pipeline.fund_selection:FundSelectionPipeline",
        "trade.allocations_to_orders.v1": f"{_P}.pipeline.alloc2orders:AllocationsToOrdersPipeline",
        "trade.full_pipeline.v1": f"{_P}.pipeline.trading_pipeline:TradingPipeline",
        "backtest.pipeline.v1": f"{_P}.pipeline.backtest_pipeline:BacktestPipeline",
    },
    "data": {
        "local_file_data": f"{_P}.datasources.local_file_data:LocalFileDataPlugin",
        "binance.live_data.v1": f"{_P}.datasources.binance_data_plugin:BinanceDataPlugin",
        "binance.futures_data.v1": f"{_P}.datasources.binance_futures_data_plugin:BinanceFuturesDataPlugin",
        "hyperliquid.data.v1": f"{_P}.datasources.hyperliquid_data_plugin:HyperliquidDataPlugin",
        "hyperliquid.data.cached.v1": f"{_P}.datasources.hyperliquid_cached_data_plugin:HyperliquidCachedDataPlugin",
        "kraken.live_data.v1": f"{_P}.datasources.kraken_data_plugin:KrakenDataPlugin",
        "data.synthetic.v1": f"{_P}.datasources.synthetic_data:SyntheticDataPlugin",
    },
    "broker": {
        "sim.paper.v1": f"{_P}.broker.sim:SimPaperBroker",
        "sim.futures_paper.v1": f"{_P}.broker.futures_paper:FuturesPaperBroker",
        "ibkr.paper.stub.v1": f"{_P}.broker.ibkr_stub:PaperBrokerStub",
        "binance.paper.stub.v1": f"{_P}.broker.binance_stub:PaperBrokerStub",
        "ibkr.live.v1": f"{_P}.broker.ibkr:IBKRBroker",
        "binance.live.v1": f"{_P}.broker.binance:BinanceBroker",
        "binance.futures.v1": f"{_P}.broker.binance_futures:BinanceFuturesBroker",
        "hyperliquid.perps.v1": f"{_P}.broker.hyperliquid:HyperliquidBroker",
        "kraken.spot.v1": f"{_P}.broker.kraken:KrakenBroker",
    },
    "publisher": {
        "telegram.publisher.v1": f"{_P}.publisher.telegram:TelegramPublisher",
    },
    "risk": {
        "risk.trading_basic.v1": f"{_P}.risk.trading_risk:TradingRiskManager",
        "risk.stress_test.v1": f"{_P}.risk.stress_test_risk:StressTestRiskManager",
        "risk.factor_exposure.v1": f"{_P}.risk.factor_exposure:FactorExposureRiskManager",
        "risk.drawdown_control.v1": f"{_P}.risk.drawdown_control:DrawdownControlRiskManager",
    },
    "strategy": {
        "strategy.altcoin_crash_bounce.v62": f"{_P}.strategies.altcoin_crash_bounce:AltcoinCrashBounceStrategy",
        "strategy.beglobal.v1": f"{_P}.strategies.beglobal_strategy:BeGlobalStrategy",
        "strategy.carry.v1": f"{_P}.strategies.carry:CarryStrategy",
        "strategy.crypto_trend.v1": f"{_P}.strategies.crypto_trend:CryptoTrendStrategy",
        "strategy.carver_trend.v1": f"{_P}.strategies.carver_trend:CarverTrendStrategy",
        "strategy.carver_trend.v2": f"{_P}.strategies.carver_trend_v2:CarverTrendV2Strategy",
        "strategy.momentum_long_short.v1": f"{_P}.strategies.momentum_long_short:MomentumLongShortStrategy",
        "strategy.cross_asset_momentum.v1": f"{_P}.strategies.cross_asset_momentum:CrossAssetMomentumStrategy",
        "strategy.crypto_regime_trend.v1": f"{_P}.strategies.crypto_regime_trend:CryptoRegimeTrendStrategy",
        "strategy.dual_momentum.v1": f"{_P}.strategies.dual_momentum:DualMomentumStrategy",
        "strategy.eth_mean_reversion_24h.v1": f"{_P}.strategies.eth_mean_reversion_24h:EthMeanReversion24h",
        "strategy.frozen_weights.v1": f"{_P}.strategies.frozen_weights:FrozenWeightsStrategy",
        "strategy.hmm_regime_allocation.v1": f"{_P}.strategies.hmm_regime_allocation:HmmRegimeAllocation",
        "strategy.ml_prediction.v1": f"{_P}.strategies.ml_strategy:MLPredictionStrategy",
        "strategy.portfolio_optimizer.v1": f"{_P}.strategies.portfolio_optimizer:PortfolioOptimizerStrategy",
        "strategy.static_weights.v1": f"{_P}.strategies.static_weights:StaticWeightsStrategy",
        "strategy.trend_catcher.v1": f"{_P}.strategies.trend_catcher:TrendCatcherStrategy",
        "strategy.trend_catcher_simple.v1": f"{_P}.strategies.trend_catcher_simple:TrendCatcherSimpleStrategy",
        "strategy.trend_following.v1": f"{_P}.strategies.trend_following:TrendFollowingStrategy",
        "strategy.vol_matched_buy_hold.v1": f"{_P}.strategies.vol_matched_buy_hold:VolMatchedBuyHoldStrategy",
        "strategy.vol_targeting.v1": f"{_P}.strategies.vol_targeting:VolTargetingStrategy",
        "strategy.weighted_avg.v1": f"{_P}.strategies.weighted_avg_aggregator:WeightedAverageAggregator",
    },
    "rebalancing": {
        "rebalancing.standard.v1": f"{_P}.rebalancing.standard_rebalancer:StandardRebalancer",
        "rebalancing.futures.v1": f"{_P}.rebalancing.futures_rebalancer:FuturesRebalancer",
    },
    "feature": {
        "features.technical.v1": f"{_P}.features.technical:TechnicalFeatures",
        "features.cross_sectional.v1": f"{_P}.features.cross_sectional:CrossSectionalFeatures",
    },
    "monitor": {
        "monitor.drawdown.v1": f"{_P}.monitor.drawdown:DrawdownMonitor",
        "monitor.signal_decay.v1": f"{_P}.monitor.signal_decay:SignalDecayMonitor",
    },
    "overlay": {
        "overlay.reversal_derisk.v1": f"{_P}.overlays.reversal_derisk:ReversalDeriskOverlay",
        "overlay.regime_reweight.v1": f"{_P}.overlays.regime_reweight:RegimeReweightOverlay",
        "overlay.corr_gross_cap.v1": f"{_P}.overlays.corr_gross_cap:CorrGrossCapOverlay",
    },
    "validation": {
        "validation.walk_forward.v1": f"{_P}.validation.walk_forward:WalkForwardValidation",
        "validation.statistical.v1": f"{_P}.validation.statistical:StatisticalValidation",
        "validation.deflated_sharpe_blp.v1": f"{_P}.validation.deflated_sharpe_blp:DeflatedSharpeBLPValidation",
        "validation.turnover.v1": f"{_P}.validation.turnover:TurnoverValidation",
        "validation.regime.v1": f"{_P}.validation.regime:RegimeValidation",
        "validation.benchmark.v1": f"{_P}.validation.benchmark:BenchmarkValidation",
    },
}


def builtins() -> dict[str, dict[str, type]]:
    """Every builtin plugin CLASS by kind, imported now (the eager table, for a caller that reads every class)."""
    return {kind: {name: load(target) for name, target in table.items()} for kind, table in BUILTIN_PLUGINS.items()}
