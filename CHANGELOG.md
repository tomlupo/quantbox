# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioned per [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-05-08

### Added
- **Plugin types**: `FeaturePlugin`, `ValidationPlugin`, `MonitorPlugin`, `DatasetPlugin` protocols; corresponding registry slots and entry-point groups (`quantbox.datasets`, `quantbox.capabilities`)
- **Strategy**: `strategy.carry.v1` — funding-rate carry (Mode A)
- **Features**: `features.technical.v1`, `features.cross_sectional.v1`
- **Validation**: `validation.walk_forward.v1`, `validation.statistical.v1`, `validation.turnover.v1`, `validation.regime.v1`, `validation.benchmark.v1`
- **Monitors**: `monitor.drawdown.v1`, `monitor.signal_decay.v1`
- **Risk**: `risk.factor_exposure.v1`, `risk.drawdown_control.v1`
- Capability checker registry with built-in checkers; `_dataset_block` and `_run_capability_checks` run helpers
- `quantbox approve` CLI subcommand for the human approval gate
- `quantbox warehouse` CLI subcommand (init, tables, query, describe, ingest, register-dataset)
- `quantbox plugins doctor` health check for schemas, entry points, and config refs
- Artifact schemas bundled as package data at `src/quantbox/artifact_schemas/` (accessed via `importlib.resources`)
- Plugin manifest bundled at `src/quantbox/plugins/manifest.yaml` with `QUANTBOX_MANIFEST` env override
- Custom exception hierarchy (`quantbox.exceptions`): `QuantboxError`, `ConfigValidationError`, `PluginNotFoundError`, `PluginLoadError`, `DataLoadError`, `BrokerExecutionError`
- Ruff linting config + GitHub Actions CI (`ci.yml`)
- `cookbook/` with `configs/` and `scripts/` (replaces separate `config/` and `examples/`)
- `.env.example` template for environment variables
- `binance.futures_data.v1` and `hyperliquid.data.v1` registered in manifest builtins (were implemented but unregistered)

### Changed
- Run manifest: `data.source_identity` replaced by typed `dataset` block
- Flat `src/` layout replaces `packages/quantbox-core/` workspace structure
- `runner.py` raises `ConfigValidationError` and `PluginNotFoundError` instead of bare `ValueError`/`KeyError`
- `cli.py` raises `PluginNotFoundError` instead of `SystemExit` for missing plugins
- Docs restructured: `guides/` merged into `playbooks/`, `adr/` renamed to `decisions/` (DEC-NNNN prefix), copy-paste templates moved to root `templates/`

### Deprecated
- `dataset_root` and `dataset` config params — use `dataset_id` instead

### Fixed
- `strategy.beglobal`: crash when prices index is non-unique
- `local_file_data`: DuckDB timestamps normalized to UTC midnight
- Configs: `data_dir` param replaced with explicit `*_path` params

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

[Unreleased]: https://github.com/tomlupo/quantbox/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/tomlupo/quantbox/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tomlupo/quantbox/releases/tag/v0.1.0
