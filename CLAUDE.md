# CLAUDE.md — Agent onboarding guide

> **Modifying QuantBox?** Read [`docs/architecture/principles.md`](docs/architecture/principles.md) and [`docs/architecture/api-layers.md`](docs/architecture/api-layers.md) first. Every architectural decision downstream is shaped by them. This file describes the *current* code; the architecture docs describe the *target design and the rules*. When they conflict, architecture wins.

## What is QuantBox?

A **template-driven SDK with adapters** for quant research and production. Three things, in order:

1. **Conventions** — data layouts, run-artifact shape, lifecycle states, skill API. The owned moat.
2. **Adapters** — thin wrappers around best-of-breed external libraries (vectorbt, riskfolio, ...). The wheel does the wheel's work. A core adapter is added only when ≥2 consumers need the same bridge; single-consumer libraries (mlflow, dvc) are imported directly in the downstream repo.
3. **Skills + templates** — LLM-facing interface and project bootstrap, coupled to the SDK in this repo.

The plugin runtime (`run_from_config`, CLI) is *one* of multiple entry points — see the [layered API](docs/architecture/api-layers.md) (L0–L5). Casual use defaults to L0/L1 (re-exports + convenience helpers). YAML pipelines are L4. Production is L5 with `--strict`.

QuantBox is a **composing framework** — owned and opinionated, but composing external libraries (vectorbt, MLflow, riskfolio, optionally Qlib) rather than competing with them on their turf. See [ADR-0001](docs/decisions/DEC-0001-library-not-framework.md).

## Authoritative docs (read in order)

| # | Doc | When |
|---|---|---|
| 1 | [`docs/architecture/principles.md`](docs/architecture/principles.md) | Read first, every time. The doctrine. |
| 2 | [`docs/architecture/api-layers.md`](docs/architecture/api-layers.md) | The L0–L5 table. Operational rule for "which layer." |
| 3 | [`docs/architecture/plugin-authoring.md`](docs/architecture/plugin-authoring.md) | Plugin types, `meta.status`, registration, naming, testing. |
| 4 | [`docs/architecture/adapters.md`](docs/architecture/adapters.md) | Wrap-don't-rebuild rule. |
| 5 | [`docs/architecture/skills.md`](docs/architecture/skills.md) | LLM-facing API, frontmatter contract, capability-gap branch. |
| 6 | [`docs/architecture/lifecycle.md`](docs/architecture/lifecycle.md) | `meta.status` state machine, reproducibility, promotion. |
| 7 | [`templates/README.md`](templates/README.md) | Copy-paste scaffolds for methodology, dataset, runbook, and decision-record docs. Used by `quantbox new` and consumed by `/promote-lock`. |

For step-by-step modifications, see [`docs/playbooks/`](docs/playbooks/). For historical decisions, see [`docs/decisions/`](docs/decisions/).

## Project layout

```
src/quantbox/              ← installable library (uv add quantbox)
  contracts.py             Protocol definitions (start here)
  runner.py                Config → plugin instantiation → pipeline.run()
  registry.py              Plugin discovery (builtins + entry points)
  cli.py                   CLI entry point (quantbox command)
  store.py                 Artifact storage (Parquet + JSON)
  schemas.py               Runtime schema validation
  artifact_schemas/        JSON schemas for artifacts (bundled as package data)
  plugins/
    manifest.yaml          Default plugin profiles (bundled as package data)
    builtins.py            Plugin registration map
    strategies/            Strategy plugins (compute target weights)
    pipeline/              Pipeline plugins (orchestrate full runs)
    datasources/           Data plugins (OHLCV, market cap, funding rates)
    broker/                Broker plugins (paper + live execution)
    rebalancing/           Rebalancing plugins (weights → orders)
    risk/                  Risk plugins (pre-trade validation)
    publisher/             Publisher plugins (notifications)
    backtesting/           Backtest engines (vectorbt, rsims)
cookbook/
  configs/                 Example YAML pipeline configs (research, trading, paper, live)
  scripts/                 Runnable example scripts (quickstart, custom plugin, artifact inspection)
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
| Strategy | `StrategyPlugin` | `run(data, params)` → dict with `"weights"` (date × symbol) |
| Data | `DataPlugin` | `load_market_data(universe, asof, params) → Dict[str, DataFrame]` |
| Broker | `BrokerPlugin` | `execute_rebalancing(weights)`, `describe()` |
| Rebalancing | `RebalancingPlugin` | `rebalance(targets, positions, params)` |
| Risk | `RiskPlugin` | `check_targets(targets, params)`, `check_orders(orders, params)` |
| Publisher | `PublisherPlugin` | `publish(result, params)` |

## Data format

DataPlugin returns `Dict[str, pd.DataFrame]` of **wide-format** DataFrames:
- Index: date
- Columns: symbol names
- Keys:
  - `"prices"` (required) — close prices
  - `"volume"` — quote-currency dollar volume
  - `"high"` / `"low"` — daily high/low (needed for ATR-based strategies)
  - `"market_cap"` — monthly mcap snapshots (typically forward-filled to daily)
  - `"funding_rates"` — perp funding (futures datasets)
  - `"eligibility_mask"` — boolean wide DataFrame; top-N-by-mcap gate that
    strategies can consume via `data.get("eligibility_mask")` for PIT-correct
    daily universe rotation

  Optional keys are `setdefault`-ed to empty DataFrames by the engine, so
  strategies can always `data.get(key)` safely. Data plugins may emit
  additional non-canonical keys, but only the list above is guaranteed.

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

## Environment variables

| Variable | Required for | Default |
|---|---|---|
| `API_KEY_BINANCE` | Binance spot/futures brokers | — |
| `API_SECRET_BINANCE` | Binance spot/futures brokers | — |
| `HYPERLIQUID_WALLET` | Hyperliquid perps broker | — |
| `HYPERLIQUID_PRIVATE_KEY` | Hyperliquid perps broker | — |
| `TELEGRAM_TOKEN` | Telegram publisher | — |
| `TELEGRAM_CHAT_ID` | Telegram publisher | — |
| `QUANTBOX_MANIFEST` | Custom manifest path | `plugins/manifest.yaml` |

None are needed for backtesting or paper trading with simulated brokers.
Copy `.env.example` to `.env` and fill in only what you need.

## Error handling

Quantbox uses custom exceptions (see `quantbox.exceptions`):

| Exception | When | Recovery |
|---|---|---|
| `ConfigValidationError` | YAML config fails validation | Check `.findings` list for details |
| `PluginNotFoundError` | Plugin name not in registry | Check `.available` for valid names |
| `PluginLoadError` | Entry point import failed | Check dependencies (`uv sync --extra full`) |
| `DataLoadError` | Data plugin can't fetch data | Check API keys, network, date range |
| `BrokerExecutionError` | Order placement failed | Check broker credentials and balances |

## Development rules

- Use `uv` as package manager, `uv run` to execute
- Don't use `requests` in core — use `urllib.request` or `httpx`
- `meta` is a class attribute, not an instance attribute
- Prefer additive changes and new plugin versions over breaking changes
- Don't rename existing entry-point IDs
- Add tests for new plugins or core behavior
- See [`docs/architecture/principles.md`](docs/architecture/principles.md) for LLM-specific guidelines + anti-patterns

**For any architectural change, the rules in [`docs/architecture/principles.md`](docs/architecture/principles.md) take precedence over this file.** Anti-patterns to refuse, decision rules for new features, and the layer-choice doctrine all live there.

## Multi-repo setup

| Repo | Purpose | Branch/tag |
|---|---|---|
| quantbox (this) | Library | `dev` for development, `main` for releases |
| quantbox-live | Production trading | Pins to tags on `main` (e.g. `@v0.1.0`) |
| quantbox-lab | Research/backtesting | Tracks `@dev` branch |

Develop on `dev` → merge to `main` → tag → bump quantbox-live.
