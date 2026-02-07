# CLAUDE.md — Agent onboarding guide

## What is QuantBox?

Config-driven quant framework. Strategies, data, brokers, and risk are all plugins wired together via YAML configs. Same strategy params work across backtesting, paper trading, and live execution.

## Project layout

```
packages/quantbox-core/src/quantbox/   ← core library
  contracts.py       Protocol definitions (start here)
  runner.py          Config → plugin instantiation → pipeline.run()
  registry.py        Plugin discovery (builtins + entry points)
  cli.py             CLI entry point (quantbox command)
  store.py           Artifact storage (Parquet + JSON)
  schemas.py         Runtime schema validation
  plugins/
    builtins.py      Plugin registration map
    strategies/      Strategy plugins (compute target weights)
    pipeline/        Pipeline plugins (orchestrate full runs)
    datasources/     Data plugins (OHLCV, market cap, funding rates)
    broker/          Broker plugins (paper + live execution)
    rebalancing/     Rebalancing plugins (weights → orders)
    risk/            Risk plugins (pre-trade validation)
    publisher/       Publisher plugins (notifications)
    backtesting/     Backtest engines (vectorbt, rsims)
configs/             Example YAML configs for all pipeline types
schemas/             JSON schemas for artifact validation
plugins/manifest.yaml  Plugin profiles (research, trading, futures_paper)
```

## Key commands

```bash
uv run quantbox plugins list               # list all plugins
uv run quantbox plugins list --json         # JSON output
uv run quantbox plugins info --name <id>    # plugin details
uv run quantbox validate -c <config>        # validate config
uv run quantbox run -c <config>             # run pipeline
uv run quantbox run --dry-run -c <config>   # dry run
uv run pytest -q                            # run tests
```

## Plugin architecture

- All plugins implement Protocols defined in `contracts.py`
- Plugins are `@dataclass` classes with a class-level `meta = PluginMeta(...)` attribute
- Registration: `plugins/builtins.py` builds `{meta.name: class}` dict
- Discovery: `registry.py:PluginRegistry.discover()` merges builtins + entry points
- Runner: `runner.py:run_from_config()` instantiates via `params_init`, calls `pipeline.run()`

## Plugin types and key methods

| Type | Protocol | Key method |
|---|---|---|
| Pipeline | `PipelinePlugin` | `run(mode, asof, params, data, store, broker, risk)` |
| Strategy | `StrategyPlugin` | `compute_weights(market_data, universe, asof, params)` |
| Data | `DataPlugin` | `load_market_data(universe, asof, params) → Dict[str, DataFrame]` |
| Broker | `BrokerPlugin` | `execute_rebalancing(weights)`, `describe()` |
| Rebalancing | `RebalancingPlugin` | `rebalance(targets, positions, params)` |
| Risk | `RiskPlugin` | `check_targets(targets, params)`, `check_orders(orders, params)` |
| Publisher | `PublisherPlugin` | `publish(result, params)` |

## Data format

DataPlugin returns `Dict[str, pd.DataFrame]` of **wide-format** DataFrames:
- Index: date
- Columns: symbol names
- Keys: `"prices"` (required), `"volume"`, `"market_cap"`, `"funding_rates"` (optional)

## Config structure

```yaml
run:
  mode: backtest|paper|live
  asof: "2026-02-06"
  pipeline: "pipeline.name.v1"

plugins:
  pipeline:
    name: "trade.full_pipeline.v1"
    params: { ... }
  strategies:
    - name: "strategy.crypto_trend.v1"
      weight: 1.0
      params: { ... }
  data:
    name: "binance.live_data.v1"
    params_init: { ... }
  broker:
    name: "hyperliquid.perps.v1"
    params_init: { ... }
  rebalancing:
    name: "rebalancing.futures.v1"
    params: { ... }
  risk:
    - name: "risk.trading_basic.v1"
      params: { ... }
```

## Development rules

- Use `uv` as package manager, `uv run` to execute
- Don't use `requests` in core — use `urllib.request` or `httpx`
- `meta` is a class attribute, not an instance attribute
- Prefer additive changes and new plugin versions over breaking changes
- Don't rename existing entry-point IDs
- Add tests for new plugins or core behavior
- See `CONTRIBUTING_LLM.md` for full guidelines

## Multi-repo setup

| Repo | Purpose | Branch/tag |
|---|---|---|
| quantbox (this) | Library | `dev` for development, `main` for releases |
| quantbox-live | Production trading | Pins to tags on `main` (e.g. `@v0.1.0`) |
| quantbox-lab | Research/backtesting | Tracks `@dev` branch |

Develop on `dev` → merge to `main` → tag → bump quantbox-live.
