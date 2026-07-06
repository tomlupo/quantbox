# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioned per [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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

## v0.3.13 (2026-07-06)

## v0.3.12 (2026-07-03)

### Feat

- **safety**: shared with_retry helper; wrap broker load_markets on transient throttle (#95) (#96)
- **safety**: submission-time intent capture + real Tom-gated recon enforcement (#90) (#92)

### Fix

- **safety**: no synthetic ledger on gated cycle; unique result-binding on failed intent (#90 follow-up) (#94)

## v0.3.11 (2026-07-02)

### Feat

- **safety**: recon-break state machine + append-only order/fill ledger (observe-mode) (#87) (#88)

### Fix

- **safety**: close residual gaps in fill-status + freeze detection (#68 #81) (#91)

## v0.3.10 (2026-07-01)

### Fix

- **safety**: broker fill-confirmation + freeze/quiet-day fail-safe + notify exception inputs (#68 #62 #81 #82) (#85)

## v0.3.9 (2026-06-30)

### Fix

- **kraken**: dust liquidation + quiet-day classification + freeze-invariant hardening (#80)

## v0.3.8 (2026-06-29)

### Feat

- **kraken**: opt-in fail-closed on degraded CMC mcap (require_genuine_mcap) (#79)
- **kraken**: opt-in venue-liquidity Stage-2 screen (screen_volume_source) (#77)

## v0.3.7 (2026-06-25)

### Feat

- **kraken**: spot DataPlugin + BrokerPlugin (ccxt) (#67)

## v0.3.6 (2026-06-20)

### Fix

- **crypto_trend**: DuckDB universe NaN-mask leaked mcap-excluded coins into the book (#60)

## v0.3.5 (2026-06-20)

### Fix

- **binance**: not_tradable opt-out at CMC candidate stage (quantlab parity) (#57)

## v0.3.4 (2026-06-20)

### Fix

- **binance/crypto_trend**: CMC-mcap candidate universe + NaN-weight robustness (#56)

## v0.3.3 (2026-06-20)

### Fix

- **binance**: rank tradable universe by volume not alphabetically (majors silently dropped) (#55)

## v0.3.2 (2026-06-19)

### Fix

- **datasources**: mcap estimator never fabricates (drop uncovered) + complete stablecoin set (#54)
- **broker**: reject non-finite persisted state on load (#53)

## v0.3.1 (2026-06-19)

### Feat

- **datasources**: add CoinMarketCap market-cap rankings source (#52)
- deploy quant subagent roster to .claude/agents/ (#50)
- **claude**: agent-side git-workflow guard (rule + PreToolUse hook) (#48)
- **crypto_trend**: allow regime_ticker=None to disable the donchian-overlay diagnostic (#46)
- **carver**: carver_trend_proper.v1 sizing chain + fine-lot universe guard (default-off) (#40)
- market-wide universe screen + mode-aware point-in-time sourcing (#37)
- rebase integrations/rb onto main (tradfi features)

### Fix

- **rebalancer**: unfreeze stale-position exits + dead-man on suppressed rebalances (#51)
- **rsims**: default margin to 0.0 — stop spurious liquidation breaking vol-invariance (#43)
- **lint**: resolve ruff F401/F841/B007/B023 errors blocking main (#38)

## v0.3.0 (2026-06-02)

### Feat

- **data**: add hyperliquid.data.cached.v1 incremental on-disk cache
- **data**: register hyperliquid.data.cached.v1 in builtins
- **data**: implement incremental load_market_data for cached HL plugin
- **data**: scaffold hyperliquid.data.cached.v1 plugin + cache helpers

## v0.2.5 (2026-06-02)

### Fix

- **broker**: resolve k-prefixed Hyperliquid perps via canonical symbol index

## v0.2.4 (2026-06-02)

## v0.2.3 (2026-05-21)

### Feat

- **report**: retrofit crypto_regime_trend diagnostics + generic primary-variant picker (#30)
- **report**: reusable block registry + 3 generic blocks + cookbook (#28)

### Fix

- **orders**: round up to min_qty on full close-out positions
- **broker**: propagate order failures to pipeline + add min_qty guard

## v0.2.1 (2026-05-19)

### Feat

- **crypto_trend**: SSRN paper parity — strategy fixes + editorial report (#26)
- **strategies**: inject _pipeline_annualize into 4 hardcoded-sqrt strategies (closes #23) (#25)
- **frequency**: Frequency value object + pandas-market-calendars + pipeline injection (closes #20) (#21)
- **strategies**: add frozen_weights plugin + flip cross_asset_momentum annualize default to 252 (#19)
- **canonical**: end-to-end reproductions on bundled synthetic fixture (#17)
- **strategies**: promote 3 lab-side strategies to quantbox core
- **trend_catcher_simple**: close_on_regime_flip param
- **strategies**: trend_catcher_simple plugin (Robuxio PDF rules)
- **runner**: capture installed-package git SHAs in run_manifest.json
- **cli**: quantbox sweep -c <yaml> command
- **analysis**: run_grid orchestrator + parquet loader
- **analysis**: parameter_grid sweep + heatmap helper
- **strategies**: vol-matched buy-and-hold benchmark plugin
- TrendCatcher v2 strategy support + universe-construction fixes
- per-variant report layout + variants runtime support
- template+JSON report architecture with vbt native figures
- add weight heatmap and per-ticker contribution chart to HTML report
- add summary.md + report.html output to every backtest run
- add data frequency param and backtest warmup auto-derive
- **datasources**: add normalize_data_frequency() utility to _utils
- **skills**: add quantbox-core skill for v0.2.0
- **core**: deprecate dataset_root/dataset params in favor of dataset_id
- **core**: replace data.source_identity with typed dataset block in run_manifest.json
- **core**: add _dataset_block and _run_capability_checks helpers
- **core**: register quantbox.datasets and quantbox.capabilities entry-point groups
- **core**: add built-in capability checkers
- **core**: add capability checker registry
- **core**: add DatasetPlugin protocol and DatasetManifest/CoverageReport dataclasses
- enrich run manifest evidence

### Fix

- **ci**: lint + agnostic-quantbox test consequences
- resolve ruff lint errors for CI
- **strategy**: make beglobal robust to non-unique prices index
- **configs**: replace nonexistent data_dir param with explicit *_path params

### Refactor

- drop hardcoded ticker fallback in _universe.py — quantbox stays agnostic
- binance_data / binance_futures_data consume canonical DEFAULT_STABLECOINS
- trend_catcher into core + DEFAULT_STABLECOINS from quantbox-datasets YAML
- merge guides/ into playbooks/, drop guides/ folder
- move doc templates to root templates/, delete methodology/datasets/runbooks from docs/
- flatten packages/quantbox-core/ to standard src/ layout
- dissolve scripts/ — promote approve, move live scripts out, delete dead code
- consolidate config/ and examples/ into cookbook/
- merge recipes/ into docs/playbooks/
- replace contracts/ with schemas/README.md
- rename configs/ → config/, remove stale account config dir
- remove legacy quantlab subtree and clarify adapter policy

## v0.2.0 (2026-02-13)

### Feat

- **carry**: complete strategy.carry.v1 — params_schema, tests, builtins registration
- **carry**: add strategy.carry.v1 — funding-rate carry Mode A
- add validation and monitor plugin support to runner
- register feature plugins in builtins
- register new risk plugins and update manifest
- add risk.drawdown_control.v1 plugin
- add risk.factor_exposure.v1 plugin
- register monitor plugins in builtins registry
- add monitor.signal_decay.v1 plugin
- add monitor.drawdown.v1 plugin
- add features.cross_sectional.v1 plugin
- add features.technical.v1 plugin
- extend PluginRegistry with feature, validation, monitor types
- add FeaturePlugin, ValidationPlugin, MonitorPlugin protocols

### Fix

- **silent-failures**: surface execution failures through Telegram and logs
- **broker/futures_paper**: reject NaN prices in place_orders
- **local_file_data**: normalize DuckDB timestamps to UTC midnight

## v0.1.1 (2026-02-13)

## v0.1.0 (2026-02-07)
