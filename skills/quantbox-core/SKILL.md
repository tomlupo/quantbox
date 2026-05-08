---
name: quantbox-core
description: >
  Operates the QuantBox quant trading framework: running pipelines, creating plugins,
  building configs, and debugging issues. Use when the user mentions "quantbox",
  "pipeline", "backtest", "paper trade", "live trade", "strategy plugin", "broker plugin",
  "data plugin", "risk plugin", "rebalancing", "run config", "target weights", "portfolio",
  "carry", "features", "validation", "monitor", or asks to create/run/debug any
  quantitative trading workflow.
default_layer: L1
escalation_rules:
  - to: L4
    when: "running a long backtest or multi-strategy sweep"
  - to: L5
    when: "scheduling a live trading cron or autoresearch loop"
requires_quantbox_min: "0.2.0"
---

# quantbox-core

Config-driven quant framework. Strategies, data, brokers, features, validation, and risk
are all plugins wired together via YAML configs. The same config drives backtesting,
paper trading, and live execution — change `run.mode` and swap the broker.

---

## Pre-Flight

Before any execution, verify the environment:

```bash
quantbox plugins doctor          # schemas, entry points, config refs
quantbox plugins list            # confirm plugins are registered
quantbox plugins list --json     # machine-readable
```

For live brokers, check credentials are set (never print values):
```bash
python -c "import os; print('BINANCE:', bool(os.getenv('API_KEY_BINANCE'))); print('HYPERLIQUID:', bool(os.getenv('HYPERLIQUID_WALLET')))"
```

No credentials needed for backtesting or paper trading with simulated brokers.

---

## Execution Workflow

```bash
quantbox validate -c cookbook/configs/<config>.yaml   # validate before running
quantbox run --dry-run -c cookbook/configs/<config>.yaml  # preview plan
quantbox run -c cookbook/configs/<config>.yaml            # execute
```

After a run, verify:
- `artifacts/<run_id>/run_manifest.json` — run metadata and dataset block
- `artifacts/<run_id>/events.jsonl` — structured event log
- `artifacts/<run_id>/target_weights.parquet` — strategy output
- `artifacts/<run_id>/orders.parquet` — broker orders

For pipelines with an approval gate:
```bash
quantbox approve --run-dir artifacts/<run_id>   # write approval before live orders
```

---

## Decision Trees

### "I need to run something"

| Intent | Config to use or adapt |
|---|---|
| Historical backtest | `cookbook/configs/run_backtest_crypto_trend.yaml` |
| Screen / rank assets | `cookbook/configs/run_fund_selection.yaml` |
| Paper trade (spot) | `cookbook/configs/run_spot_paper_crypto_trend.yaml` |
| Paper trade (futures) | `cookbook/configs/run_futures_paper_crypto_trend.yaml` |
| Stress test with synthetic data | `cookbook/configs/run_stress_test.yaml` |
| Execute pre-computed weights | `cookbook/configs/run_trade_from_allocations.yaml` |

### "I need to create a plugin"

| Plugin type | Protocol | File location |
|---|---|---|
| Strategy (compute weights) | `StrategyPlugin` | `src/quantbox/plugins/strategies/<name>.py` |
| Pipeline (orchestrate run) | `PipelinePlugin` | `src/quantbox/plugins/pipeline/<name>.py` |
| Data source (load OHLCV) | `DataPlugin` | `src/quantbox/plugins/datasources/<name>.py` |
| Broker (execute orders) | `BrokerPlugin` | `src/quantbox/plugins/broker/<name>.py` |
| Rebalancer (weights→orders) | `RebalancingPlugin` | `src/quantbox/plugins/rebalancing/<name>.py` |
| Risk manager (validate) | `RiskPlugin` | `src/quantbox/plugins/risk/<name>.py` |
| Feature engineer | `FeaturePlugin` | `src/quantbox/plugins/features/<name>.py` |
| Backtester / validator | `ValidationPlugin` | `src/quantbox/plugins/validation/<name>.py` |
| Run monitor | `MonitorPlugin` | `src/quantbox/plugins/monitor/<name>.py` |
| Notification publisher | `PublisherPlugin` | `src/quantbox/plugins/publisher/<name>.py` |

### "Something went wrong"

| Symptom | Likely cause |
|---|---|
| Config validation fails | Invalid YAML structure or unknown plugin name |
| `PluginNotFoundError` | Typo in plugin name or plugin not registered |
| Data load fails | Missing API keys, network, or date range out of bounds |
| Strategy returns empty weights | Insufficient lookback data or universe is empty |
| Broker execution fails | Credentials, insufficient balance, or `mode` mismatch |
| Pipeline crashes mid-run | Missing required plugin section in config |

---

## Project Layout

```
src/quantbox/           # core library
  contracts.py          # Protocol definitions — read first
  runner.py             # Config → plugin instantiation → pipeline.run()
  registry.py           # Plugin discovery (builtins + entry points)
  cli.py                # CLI entry point (typer)
  store.py              # ArtifactStore — Parquet + JSON per run_id
  exceptions.py         # QuantboxError hierarchy
  plugins/
    manifest.yaml       # Builtin plugin list + profiles
    builtins.py         # Plugin class map (name → class)
    strategies/         # StrategyPlugin implementations
    pipeline/           # PipelinePlugin implementations
    datasources/        # DataPlugin implementations
    broker/             # BrokerPlugin implementations
    rebalancing/        # RebalancingPlugin implementations
    risk/               # RiskPlugin implementations
    features/           # FeaturePlugin implementations
    validation/         # ValidationPlugin implementations
    monitor/            # MonitorPlugin implementations
    publisher/          # PublisherPlugin implementations
  artifact_schemas/     # 14 JSON schemas (importlib.resources)

cookbook/
  configs/              # 21 example YAML configs
  scripts/              # standalone utility scripts

docs/
  architecture/         # principles, api-layers, adapters, skills, lifecycle
  playbooks/            # how-tos: add-a-plugin, add-a-skill, promote-a-methodology, …
  decisions/            # DEC-NNNN architecture decision records

skills/                 # LLM-facing API skills (this directory)
templates/              # copy-paste scaffolds for downstream projects
```

---

## Core Mental Model

```
YAML config
    │
    ▼
runner.py               # loads config, resolves plugins from registry
    │
    ▼
PluginRegistry          # builtins (manifest.yaml) + entry-point discovery
    │
    ▼
PipelinePlugin.run()    # orchestrates the run
    ├── DataPlugin        # load_universe(), load_market_data()
    ├── StrategyPlugin[]  # run(data, params) → {"weights": DataFrame}
    ├── RebalancingPlugin # generate_orders(weights, positions, ...) → orders
    ├── BrokerPlugin      # place_orders(orders) → fills
    ├── RiskPlugin[]      # validate(weights, portfolio, ...) → RiskResult
    ├── FeaturePlugin[]   # compute(data, params) → feature DataFrame
    ├── ValidationPlugin[]# validate(results, ...) → ValidationResult
    ├── MonitorPlugin[]   # check(run_result, ...) → MonitorResult
    └── PublisherPlugin[] # publish(result, ...)
    │
    ▼
ArtifactStore           # artifacts/<run_id>/ — Parquet + JSON
```

All plugins are **structural subtypes** — a class is a `StrategyPlugin` if it has
`meta: PluginMeta` as a class attribute and implements the required methods.
No inheritance required.

---

## PluginMeta

Every plugin has a class-level `meta` attribute:

```python
from quantbox.contracts import PluginMeta

class MyStrategyPlugin:
    meta = PluginMeta(
        name="strategy.my_strategy.v1",     # unique, dot-separated, versioned
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.2.0",
        status="research",                  # "research" | "locked" | "production"
        description="What this does in one sentence.",
        tags=("crypto", "trend"),
        outputs=("strategy_weights",),      # artifact names this plugin produces
        params_schema={                     # JSON Schema for params
            "type": "object",
            "properties": {
                "lookback_days": {"type": "integer", "default": 365},
            },
        },
        examples=(
            "plugins:\n  strategies:\n    - name: strategy.my_strategy.v1\n      params:\n        lookback_days: 365",
        ),
    )
```

---

## Plugin Creation Workflow

**Step 1 — Create the module:**

```
src/quantbox/plugins/<type>/<name>.py
```

Implement the Protocol from `quantbox.contracts`. Set `meta` as a class attribute.

**Step 2 — Export from `__init__.py`:**

```python
# src/quantbox/plugins/<type>/__init__.py
from .<name> import MyPlugin
```

**Step 3 — Register in `builtins.py`:**

```python
# src/quantbox/plugins/builtins.py
from .strategies.my_strategy import MyStrategyPlugin
# add to the builtins() dict under the correct key
```

**Step 4 — Add to manifest:**

```yaml
# src/quantbox/plugins/manifest.yaml
plugins:
  builtins:
    strategies:
      - strategy.my_strategy.v1   # add here
```

**Step 5 — Add a cookbook config:**

Copy the closest example in `cookbook/configs/` and adapt.

**Step 6 — Add a test:**

```python
# tests/test_my_strategy.py
from quantbox.plugins.strategies.my_strategy import MyStrategyPlugin

def test_meta():
    assert MyStrategyPlugin.meta.name == "strategy.my_strategy.v1"

def test_run():
    plugin = MyStrategyPlugin()
    # minimal synthetic data → weights DataFrame
```

**Step 7 — Verify:**

```bash
quantbox plugins list              # appears in list
quantbox plugins info --name strategy.my_strategy.v1  # metadata is correct
quantbox validate -c cookbook/configs/<config>.yaml
quantbox run --dry-run -c cookbook/configs/<config>.yaml
pytest -q
```

---

## Config Structure

```yaml
run:
  mode: backtest        # "backtest" | "paper" | "live"
  asof: "2026-02-01"
  pipeline: "backtest.pipeline.v1"

artifacts:
  root: "./artifacts"

plugins:
  profile: research     # optional preset — "research" | "trading" | "trading_full"
                        # | "futures_paper" | "stress_test"

  pipeline:
    name: "backtest.pipeline.v1"
    params:
      engine: vectorbt  # "vectorbt" or "rsims"
      fees: 0.001
      rebalancing_freq: 1
      trading_days: 365
      universe:
        top_n: 100
      prices:
        lookback_days: 365

  strategies:
    - name: "strategy.crypto_trend.v1"
      weight: 1.0
      params:
        lookback_days: 365

  data:
    name: "binance.live_data.v1"
    params_init:
      quote_asset: USDT

  broker:               # omit for backtest.pipeline.v1
    name: "sim.paper.v1"

  risk:
    - name: "risk.trading_basic.v1"
      params:
        max_leverage: 1.0
        max_concentration: 0.25
```

Profiles are presets — set `profile: research` and only override what differs.

---

## Plugin Catalog (v0.2.0)

| Category | Count | Plugin IDs |
|---|---|---|
| Pipelines | 4 | `backtest.pipeline.v1`, `fund_selection.simple.v1`, `trade.allocations_to_orders.v1`, `trade.full_pipeline.v1` |
| Strategies | 10 | `strategy.beglobal.v1`, `strategy.carry.v1`, `strategy.carver_trend.v1`, `strategy.cross_asset_momentum.v1`, `strategy.crypto_regime_trend.v1`, `strategy.crypto_trend.v1`, `strategy.ml_prediction.v1`, `strategy.momentum_long_short.v1`, `strategy.portfolio_optimizer.v1`, `strategy.weighted_avg.v1` |
| Data | 5 | `binance.futures_data.v1`, `binance.live_data.v1`, `data.synthetic.v1`, `hyperliquid.data.v1`, `local_file_data` |
| Brokers | 8 | `binance.futures.v1`, `binance.live.v1`, `binance.paper.stub.v1`, `hyperliquid.perps.v1`, `ibkr.live.v1`, `ibkr.paper.stub.v1`, `sim.futures_paper.v1`, `sim.paper.v1` |
| Risk | 4 | `risk.drawdown_control.v1`, `risk.factor_exposure.v1`, `risk.stress_test.v1`, `risk.trading_basic.v1` |
| Features | 2 | `features.cross_sectional.v1`, `features.technical.v1` |
| Validation | 5 | `validation.benchmark.v1`, `validation.regime.v1`, `validation.statistical.v1`, `validation.turnover.v1`, `validation.walk_forward.v1` |
| Monitors | 2 | `monitor.drawdown.v1`, `monitor.signal_decay.v1` |
| Rebalancing | 2 | `rebalancing.futures.v1`, `rebalancing.standard.v1` |
| Publishers | 1 | `telegram.publisher.v1` |

**Total: 43 built-in plugins.**

---

## Development Rules

- Use `uv` as package manager; run everything with `uv run` or activate `.venv` first
- `meta` is a **class attribute**, not an instance attribute
- Do not rename existing plugin `name` strings — they are public IDs
- Do not embed secrets in YAML; use environment variables
- Do not use `requests` in core; use `httpx` or `urllib.request`
- Prefer adding a new `v2` plugin over breaking an existing one
- Add `tests/` coverage for new plugins and core behavior
- Start with `readonly: true` and `--dry-run` for any broker work
- Run `quantbox plugins doctor` after any structural change

---

## See Also

- `docs/architecture/principles.md` — read before any structural change
- `docs/architecture/api-layers.md` — L0–L5 layering rules
- `docs/playbooks/add-a-plugin.md` — step-by-step plugin authoring
- `docs/playbooks/add-a-skill.md` — authoring a new skill
- `docs/playbooks/promote-a-methodology.md` — research → locked → production
- `skills/quantbox-autoresearch/SKILL.md` — LLM-driven improvement loops (stub)
