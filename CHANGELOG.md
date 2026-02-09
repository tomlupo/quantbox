# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioned per [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Custom exception hierarchy (`quantbox.exceptions`): `QuantboxError`, `ConfigValidationError`, `PluginNotFoundError`, `PluginLoadError`, `DataLoadError`, `BrokerExecutionError`
- Ruff linting config + GitHub Actions CI (`ci.yml`)
- `examples/` folder with 3 runnable scripts
- `.env.example` template for environment variables
- Environment variables and error handling docs in `CLAUDE.md`

### Changed
- `runner.py` raises `ConfigValidationError` and `PluginNotFoundError` instead of bare `ValueError`/`KeyError`
- `cli.py` raises `PluginNotFoundError` instead of `SystemExit` for missing plugins

## [0.1.0] — 2026-02-07

First tagged release. Core framework with full plugin architecture.

### Added
- **Core**: Protocol-based plugin contracts (`contracts.py`), config runner, artifact store, plugin registry with entry-point discovery
- **Strategies**: `crypto_trend.v1`, `momentum_long_short.v1`, `carver_trend.v1`, `cross_asset_momentum.v1`, `crypto_regime_trend.v1`, `weighted_average_aggregator.v1`
- **Pipelines**: `fund_selection.simple.v1`, `trade.full_pipeline.v1`, `trade.allocations_to_orders.v1`, `backtest.pipeline.v1`
- **Data plugins**: `local_file_data`, `binance.live_data.v1`, `binance.futures_data.v1` with DuckDB caching, OHLCV validation, retry
- **Brokers**: `sim.paper.v1`, `sim.futures_paper.v1`, `binance.live.v1`, `binance.futures.v1`, `hyperliquid.perps.v1`, `ibkr.live.v1` (stub)
- **Rebalancing**: `rebalancing.standard.v1`, `rebalancing.futures.v1`
- **Risk**: `risk.trading_basic.v1` (concentration, leverage, drawdown limits)
- **Publisher**: `telegram.publisher.v1`
- **Backtesting**: vectorbt engine (Numba-accelerated) and rsims engine (daily with funding/margin)
- **CLI**: `quantbox plugins list|info|doctor`, `quantbox validate`, `quantbox run`
- **Schemas**: 14 JSON schemas for artifact validation
- **Docs**: CLAUDE.md, CONTRIBUTING_LLM.md, backtesting guide, multi-repo workflow guide

### Breaking changes
- `DataPlugin.load_prices()` replaced by `load_market_data()` returning `Dict[str, DataFrame]`
- `DuckDBParquetData` renamed to `LocalFileDataPlugin` (alias kept for backward compat)

[Unreleased]: https://github.com/tomlupo/quantbox/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tomlupo/quantbox/releases/tag/v0.1.0
