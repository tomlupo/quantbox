# QuantBox

Quant research and trading framework with a plugin architecture. Config-driven pipelines for backtesting, paper trading, and live execution.

## Repositories

| Repo | Purpose | Quantbox pin |
|---|---|---|
| **quantbox** (this) | Library: strategies, plugins, protocols, core runtime | (source) |
| [quantbox-live](https://github.com/tomlupo/quantbox-live) | Production: daily automated trading | `@v0.1.0` (stable tag) |
| [quantbox-lab](https://github.com/tomlupo/quantbox-lab) | Research: backtesting, notebooks, experiments | `@dev` (latest) |

See [multi-repo workflow](docs/guides/multi-repo-workflow.md) for versioning and promotion.

## Install

```bash
uv venv && source .venv/bin/activate
uv sync

# Optional extras for broker adapters:
uv sync --extra ccxt      # Binance, Hyperliquid (via ccxt)
uv sync --extra ibkr      # Interactive Brokers
uv sync --extra binance   # python-binance
uv sync --extra full      # all of the above
```

## Quick start

### Research (fund selection)

```bash
quantbox run -c configs/run_fund_selection.yaml
```

### Backtest

```bash
quantbox run -c configs/run_backtest_crypto_trend.yaml
```

See [backtesting guide](docs/guides/backtesting.md) for engine options and parameters.

### Paper trading

```bash
quantbox run -c configs/run_futures_paper_crypto_trend.yaml
```

### Live trading

Configured via [quantbox-live](https://github.com/tomlupo/quantbox-live).

## Plugins

All plugins are config-driven and discovered automatically. List registered plugins:

```bash
quantbox plugins list
```

### Pipelines

| Name | Description |
|---|---|
| `fund_selection.simple.v1` | Research pipeline: universe screening and allocation |
| `trade.allocations_to_orders.v1` | Converts allocations to broker orders |
| `trade.full_pipeline.v1` | End-to-end trading: strategy → rebalance → execute |
| `backtest.pipeline.v1` | Historical simulation with vectorbt or rsims engine |

### Strategies

| Name | Description |
|---|---|
| `strategy.crypto_trend.v1` | Momentum trend-following for crypto |
| `strategy.carver_trend.v1` | Rob Carver-style trend with vol targeting |
| `strategy.momentum_long_short.v1` | Cross-sectional momentum, long/short |
| `strategy.cross_asset_momentum.v1` | Multi-asset momentum |
| `strategy.crypto_regime_trend.v1` | Regime-aware crypto trend |
| `strategy.weighted_avg.v1` | Meta-strategy aggregator (weighted blend) |

### Data

| Name | Description |
|---|---|
| `local_file_data` | Local Parquet files via DuckDB |
| `binance.live_data.v1` | Binance spot OHLCV + market data |
| `binance.futures_data.v1` | Binance USDM futures + funding rates |

### Brokers

| Name | Description |
|---|---|
| `sim.paper.v1` | Paper simulator (spot) |
| `sim.futures_paper.v1` | Paper simulator (futures, with funding) |
| `ibkr.paper.stub.v1` | IBKR paper trading stub |
| `ibkr.live.v1` | IBKR live execution |
| `binance.paper.stub.v1` | Binance paper trading stub |
| `binance.live.v1` | Binance spot live execution |
| `binance.futures.v1` | Binance USDM futures execution |
| `hyperliquid.perps.v1` | Hyperliquid perpetual futures |

### Rebalancing

| Name | Description |
|---|---|
| `rebalancing.standard.v1` | Standard portfolio rebalancer (spot) |
| `rebalancing.futures.v1` | Futures rebalancer with leverage and margin |

### Risk

| Name | Description |
|---|---|
| `risk.trading_basic.v1` | Leverage, concentration, and notional limits |

### Publishers

| Name | Description |
|---|---|
| `telegram.publisher.v1` | Trade notifications via Telegram |

## Plugin manifest and profiles

The manifest at `plugins/manifest.yaml` defines available profiles:

```yaml
plugins:
  profile: research       # or: trading, trading_full, futures_paper
```

Profiles bundle a set of plugins so you don't repeat them in every config.

## CLI reference

```bash
quantbox plugins list              # list all registered plugins
quantbox plugins list --json       # JSON output
quantbox plugins info --name <id>  # plugin details
quantbox validate -c <config>      # validate config without running
quantbox run -c <config>           # run a pipeline
quantbox run --dry-run -c <config> # dry run (no side effects)
```

## Artifacts

Each run writes to `artifacts/<run_id>/`:
- `run_manifest.json` — run metadata
- `events.jsonl` — structured event log
- Strategy-specific outputs (weights, orders, fills, metrics)

Artifact schemas are in `/schemas/*.schema.json`.

## Development

```bash
make dev       # install dev deps
make dev-full  # install all extras + dev deps
pytest -q      # run tests
```

See [CONTRIBUTING_LLM.md](CONTRIBUTING_LLM.md) for LLM development guidelines.

## Documentation

See [docs/](docs/) for full documentation:
- [Product requirements (PRD)](docs/PRD.md)
- [Backtesting guide](docs/guides/backtesting.md)
- [Multi-repo workflow](docs/guides/multi-repo-workflow.md)
- [Trading bridge](docs/guides/trading-bridge.md)
- [Approval gate](docs/guides/approval-gate.md)
- [Integration guide](docs/guides/quantbox-integration-guide.md)
- [LLM operations reference](docs/reference/llm-operations.md)
- [Broker secrets](docs/reference/broker-secrets.md)
