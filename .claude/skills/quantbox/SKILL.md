---
name: quantbox
description: >
  Operates the QuantBox quant trading framework: running pipelines, creating plugins,
  building configs, and debugging issues. Use when user mentions 'quantbox', 'pipeline',
  'backtest', 'paper trade', 'live trade', 'strategy plugin', 'broker plugin',
  'data plugin', 'risk plugin', 'rebalancing', 'run config', 'target weights',
  'portfolio', or asks to create/run/debug any quantitative trading workflow.
references:
  - strategies
  - pipelines
  - data
  - brokers
  - configs
  - risk
metadata:
  author: quantbox
  version: 2.0.0
---

# QuantBox Platform Skill

Config-driven quant framework. Strategies, data, brokers, and risk are all plugins
wired together via YAML configs. Same config works across backtesting, paper trading,
and live execution by changing `run.mode`.

## Progressive Disclosure Rules

**Do NOT load all reference files into context at once.** Follow this loading order:

1. Read this SKILL.md first (always loaded)
2. Use the decision trees below to identify which domain applies
3. Load only the relevant `references/<domain>/README.md`
4. Load `api.md`, `patterns.md`, or `gotchas.md` only when needed for the specific task

## Pre-Flight Check

Before any pipeline execution, verify the environment:

```bash
uv run quantbox plugins doctor     # check all plugins load correctly
uv run quantbox plugins list       # confirm expected plugins are registered
```

For live brokers, verify credentials are set:
```bash
# Check env vars (do NOT print values)
python -c "import os; print('BINANCE:', bool(os.getenv('API_KEY_BINANCE'))); print('HYPERLIQUID:', bool(os.getenv('HYPERLIQUID_WALLET')))"
```

No env vars are needed for backtesting or paper trading with simulated brokers.

## Decision Trees

Use these to route to the right reference files, then load detailed references on demand.

### "I need to run something"

| Intent | Pipeline | Mode | References to load |
|---|---|---|---|
| Historical backtest | `backtest.pipeline.v1` | backtest | [configs/README](references/configs/README.md), [configs/patterns](references/configs/patterns.md) |
| Screen / rank assets | `fund_selection.simple.v1` | backtest | [configs/README](references/configs/README.md), [configs/patterns](references/configs/patterns.md) |
| Paper trade (spot) | `trade.full_pipeline.v1` | paper | [configs/README](references/configs/README.md), [brokers/README](references/brokers/README.md) |
| Paper trade (futures) | `trade.full_pipeline.v1` | paper | [configs/README](references/configs/README.md), [brokers/README](references/brokers/README.md) |
| Live trade | `trade.full_pipeline.v1` | live | [configs/README](references/configs/README.md), [brokers/README](references/brokers/README.md) |
| Execute pre-computed weights | `trade.allocations_to_orders.v1` | paper/live | [configs/README](references/configs/README.md), [brokers/README](references/brokers/README.md) |

### "I need to create a plugin"

| Plugin type | Protocol | Reference to load |
|---|---|---|
| Strategy (compute weights) | `StrategyPlugin` | [strategies/api](references/strategies/api.md), [strategies/patterns](references/strategies/patterns.md) |
| Pipeline (orchestrate run) | `PipelinePlugin` | [pipelines/api](references/pipelines/api.md) |
| Data source (load OHLCV) | `DataPlugin` | [data/api](references/data/api.md) |
| Broker (execute orders) | `BrokerPlugin` | [brokers/api](references/brokers/api.md) |
| Risk manager (validate) | `RiskPlugin` | [risk/api](references/risk/api.md) |
| Rebalancer (weights to orders) | `RebalancingPlugin` | [pipelines/api](references/pipelines/api.md) |
| Publisher (notifications) | `PublisherPlugin` | [pipelines/api](references/pipelines/api.md) |

### "I need to work with data"

| Intent | Plugin | Reference to load |
|---|---|---|
| Load from local Parquet files | `local_file_data` | [data/README](references/data/README.md) |
| Fetch live Binance spot data | `binance.live_data.v1` | [data/README](references/data/README.md) |
| Fetch Binance futures data | `binance.futures_data.v1` | [data/README](references/data/README.md) |
| Fetch Hyperliquid perps data | `hyperliquid.data.v1` | [data/README](references/data/README.md) |
| Generate synthetic test data | `data.synthetic.v1` | [data/README](references/data/README.md) |

### "I need to choose a strategy"

| Intent | Strategy | Reference to load |
|---|---|---|
| Crypto trend following | `strategy.crypto_trend.v1` | [strategies/README](references/strategies/README.md) |
| Futures trend (Carver-style) | `strategy.carver_trend.v1` | [strategies/README](references/strategies/README.md) |
| Long/short momentum | `strategy.momentum_long_short.v1` | [strategies/README](references/strategies/README.md) |
| Cross-asset momentum | `strategy.cross_asset_momentum.v1` | [strategies/README](references/strategies/README.md) |
| Regime-aware trend | `strategy.crypto_regime_trend.v1` | [strategies/README](references/strategies/README.md) |
| Global multi-asset | `strategy.beglobal.v1` | [strategies/README](references/strategies/README.md) |
| Mean-variance optimization | `strategy.portfolio_optimizer.v1` | [strategies/README](references/strategies/README.md) |
| ML prediction | `strategy.ml_prediction.v1` | [strategies/README](references/strategies/README.md) |
| Blend multiple strategies | `strategy.weighted_avg.v1` | [strategies/README](references/strategies/README.md) |

### "Something went wrong"

| Symptom | Likely cause | Reference to load |
|---|---|---|
| Config validation fails | Invalid YAML structure or plugin name | [configs/gotchas](references/configs/gotchas.md) |
| Plugin not found | Typo or not installed | [configs/gotchas](references/configs/gotchas.md) |
| Data load fails | API keys, network, date range | [data/gotchas](references/data/gotchas.md) |
| Strategy returns empty weights | Data format or lookback issues | [strategies/gotchas](references/strategies/gotchas.md) |
| Broker execution fails | Credentials, balances, or mode mismatch | [brokers/gotchas](references/brokers/gotchas.md) |
| Pipeline crashes | Missing plugin sections in config | [pipelines/gotchas](references/pipelines/gotchas.md) |

## Essential Commands

```bash
uv run quantbox plugins list               # list all registered plugins
uv run quantbox plugins list --json        # JSON output for programmatic use
uv run quantbox plugins info --name <id>   # show plugin metadata and params
uv run quantbox plugins doctor             # diagnostic checks
uv run quantbox validate -c <config>       # validate a YAML config
uv run quantbox run -c <config>            # execute a pipeline
uv run quantbox run --dry-run -c <config>  # preview execution plan
uv run pytest -q                           # run test suite
```

## Project Layout

```
packages/quantbox-core/src/quantbox/   # core library
  contracts.py       # Protocol definitions (start here)
  runner.py          # Config -> plugin instantiation -> pipeline.run()
  registry.py        # Plugin discovery (builtins + entry points)
  cli.py             # CLI entry point
  store.py           # Artifact storage (Parquet + JSON)
  schemas.py         # Runtime schema validation
  exceptions.py      # Structured error hierarchy
  plugins/
    builtins.py      # Plugin registration map
    strategies/      # Strategy plugins
    pipeline/        # Pipeline plugins
    datasources/     # Data plugins
    broker/          # Broker plugins
    rebalancing/     # Rebalancing plugins
    risk/            # Risk plugins
    publisher/       # Publisher plugins
configs/             # Example YAML configs (21 files)
schemas/             # JSON schemas for artifact validation (14 files)
plugins/manifest.yaml  # Plugin profiles (presets)
recipes/             # Step-by-step plugin creation guides
```

## Pipeline Execution Workflow

**Step 1 - Validate:**
```bash
uv run quantbox validate -c configs/<config>.yaml
```

**Step 2 - Preview:**
```bash
uv run quantbox run --dry-run -c configs/<config>.yaml
```

**Step 3 - Execute:**
```bash
uv run quantbox run -c configs/<config>.yaml
```

**Step 4 - Self-verify:**
- Check `artifacts/<run_id>/run_manifest.json` for run metadata
- Check `artifacts/<run_id>/events.jsonl` for event stream
- Inspect `target_weights.parquet` and `orders.parquet` for outputs
- Run `uv run pytest -q` to confirm no regressions

## Plugin Creation Workflow

**Step 1 - Identify type:** Use the "I need to create a plugin" decision tree above

**Step 2 - Load reference:** Read the `api.md` for the relevant plugin type

**Step 3 - Create module:** `packages/quantbox-core/src/quantbox/plugins/<type>/<name>.py`

**Step 4 - Register:**
1. Export from `plugins/<type>/__init__.py`
2. Import in `plugins/builtins.py`, add to `builtins()` dict

**Step 5 - Config:** Add example YAML to `configs/`

**Step 6 - Test:** Add test in `tests/`

**Step 7 - Self-verify:**
```bash
uv run quantbox plugins list            # plugin appears in list
uv run quantbox plugins info --name <id>  # metadata is correct
uv run quantbox validate -c <config>    # config validates
uv run quantbox run --dry-run -c <config>  # dry run succeeds
uv run pytest -q                        # all tests pass
```

## Development Rules

- Use `uv` as package manager; run everything with `uv run`
- `meta` is a **class attribute**, not an instance attribute
- Prefer additive changes and new plugin versions over breaking changes
- Do not rename existing entry-point IDs
- Do not embed secrets in YAML configs; use environment variables
- Do not use `requests` in core; use `httpx` or `urllib.request`
- Add tests for new plugins or core behavior
- Write LLM-friendly docstrings with examples and "LLM Note:" hints

## Product Index

<!-- BEGIN AUTO-GENERATED -->
| Category | Count | Plugin IDs |
|---|---|---|
| Pipelines | 4 | `backtest.pipeline.v1`, `fund_selection.simple.v1`, `trade.allocations_to_orders.v1`, `trade.full_pipeline.v1` |
| Strategies | 9 | `strategy.beglobal.v1`, `strategy.carver_trend.v1`, `strategy.cross_asset_momentum.v1`, `strategy.crypto_regime_trend.v1`, `strategy.crypto_trend.v1`, `strategy.ml_prediction.v1`, `strategy.momentum_long_short.v1`, `strategy.portfolio_optimizer.v1`, `strategy.weighted_avg.v1` |
| Data Sources | 5 | `binance.futures_data.v1`, `binance.live_data.v1`, `data.synthetic.v1`, `hyperliquid.data.v1`, `local_file_data` |
| Brokers | 8 | `binance.futures.v1`, `binance.live.v1`, `binance.paper.stub.v1`, `hyperliquid.perps.v1`, `ibkr.live.v1`, `ibkr.paper.stub.v1`, `sim.futures_paper.v1`, `sim.paper.v1` |
| Risk | 2 | `risk.stress_test.v1`, `risk.trading_basic.v1` |
| Rebalancing | 2 | `rebalancing.futures.v1`, `rebalancing.standard.v1` |
| Publishers | 1 | `telegram.publisher.v1` |

Total: **31** built-in plugins.
<!-- END AUTO-GENERATED -->
