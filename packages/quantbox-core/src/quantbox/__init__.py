"""quantbox — quant research & trading core with plugin marketplace.

Quick start::

    import quantbox

    # Feature computation (wide-format DataFrames)
    from quantbox.features import (
        compute_total_returns,
        compute_tsmom,
        compute_rolling_vol,
        compute_sma,
        compute_zscore_cross_sectional,
        rank_select_top_n,
        inverse_volatility_weights,
    )

    # Strategy plugins
    from quantbox.plugins.strategies import (
        DualMomentumStrategy,
        TrendFollowingStrategy,
        VolTargetingStrategy,
        CrossAssetMomentumStrategy,
        WeightedAverageAggregator,
    )

    # Backtesting (vectorbt-powered)
    from quantbox.plugins.backtesting import backtest

    # Plugin discovery & orchestration
    from quantbox.registry import PluginRegistry
    from quantbox.runner import run_from_config

Modules:

- ``quantbox.features`` — wide-format feature computation (returns,
  momentum, volatility, signals, cross-sectional transforms)
- ``quantbox.indicators`` — single-series technical indicators
  (SMA, EMA, RSI, MACD, Bollinger, ATR) — convenience wrappers
- ``quantbox.plugins.strategies`` — strategy plugins (DualMomentum,
  TrendFollowing, XSMOM, VolTargeting, aggregators)
- ``quantbox.plugins.backtesting`` — vectorbt backtesting engine
- ``quantbox.contracts`` — plugin protocol definitions (StrategyPlugin,
  DataPlugin, BrokerPlugin, PipelinePlugin, etc.)
- ``quantbox.registry`` — plugin discovery via entrypoints
- ``quantbox.runner`` — orchestrate pipelines from config
- ``quantbox.store`` — artifact persistence (Parquet, JSON)
- ``quantbox.performance`` — portfolio performance metrics
"""

__all__ = [
    "contracts",
    "exceptions",
    "features",
    "indicators",
    "performance",
    "plugins",
    "registry",
    "runner",
    "schemas",
    "store",
]
__version__ = "0.2.0"
