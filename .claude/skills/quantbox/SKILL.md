---
name: quantbox
description: >
  Operates the QuantBox quant trading framework: running pipelines, creating plugins,
  building configs, and debugging issues. Use when user mentions 'quantbox', 'pipeline',
  'backtest', 'paper trade', 'live trade', 'strategy plugin', 'broker plugin',
  'data plugin', 'risk plugin', 'rebalancing', 'run config', 'target weights',
  'portfolio', or asks to create/run/debug any quantitative trading workflow.
metadata:
  author: quantbox
  version: 1.0.0
---

# QuantBox Agent Skill

QuantBox is a config-driven quant framework where strategies, data sources, brokers, and risk
managers are all plugins wired together via YAML configs. The same config works across
backtesting, paper trading, and live execution by changing `run.mode`.

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
configs/             # Example YAML configs
schemas/             # JSON schemas for artifact validation
plugins/manifest.yaml  # Plugin profiles (presets)
```

## Essential Commands

```bash
uv run quantbox plugins list               # list all registered plugins
uv run quantbox plugins list --json        # JSON output for programmatic use
uv run quantbox plugins info --name <id>   # show plugin metadata and params
uv run quantbox plugins doctor             # diagnostic checks on all plugins
uv run quantbox validate -c <config>       # validate a YAML config
uv run quantbox run -c <config>            # execute a pipeline
uv run quantbox run --dry-run -c <config>  # preview execution plan
uv run pytest -q                           # run test suite
```

## Workflow 1: Running a Pipeline

**Step 1 - Validate the config first:**
```bash
uv run quantbox validate -c configs/<config>.yaml
```

**Step 2 - Preview with dry-run:**
```bash
uv run quantbox run --dry-run -c configs/<config>.yaml
```

**Step 3 - Execute:**
```bash
uv run quantbox run -c configs/<config>.yaml
```

**Step 4 - Check artifacts:**
Outputs go to `artifacts/<run_id>/` containing:
- `target_weights.parquet` - portfolio weights
- `orders.parquet` - generated orders
- `run_manifest.json` - full run metadata
- `events.jsonl` - event stream

## Workflow 2: Creating a New Plugin

All plugins are `@dataclass` classes with a class-level `meta = PluginMeta(...)` attribute
that implement a Protocol from `contracts.py`. See [references/plugin-protocols.md](references/plugin-protocols.md)
for the exact signatures.

### Steps for a Built-in Plugin

1. **Create module**: `packages/quantbox-core/src/quantbox/plugins/<type>/<name>.py`
2. **Implement protocol**: Use `@dataclass` with `meta = PluginMeta(...)` class attribute
3. **Export**: Add to `plugins/<type>/__init__.py`
4. **Register**: Add to `plugins/builtins.py` imports and `builtins()` dict
5. **Config**: Add example YAML to `configs/`
6. **Test**: Add test in `tests/`
7. **Verify**:
   ```bash
   uv run quantbox plugins list          # confirm plugin appears
   uv run quantbox validate -c <config>  # validate config
   uv run quantbox run --dry-run -c <config>
   uv run pytest -q                      # all tests pass
   ```

### Strategy Plugin Template

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
from quantbox.contracts import PluginMeta

@dataclass
class MyStrategy:
    """Short description of strategy logic.

    LLM Note: Explain the core algorithm and key parameters.
    """

    meta = PluginMeta(
        name="strategy.my_strategy.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="One-line description of strategy",
        tags=("crypto", "trend"),
    )

    # Constructor params (set via params or params_init in config)
    lookback_days: int = 90

    def run(
        self,
        data: dict[str, pd.DataFrame],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prices = data["prices"]

        # Apply param overrides
        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        # --- Strategy logic here ---
        # Compute weights as a wide DataFrame (date index x symbol columns)
        weights = pd.DataFrame(...)

        return {
            "weights": weights,
            "simple_weights": weights.iloc[-1].to_dict(),
        }
```

### Pipeline Plugin Template

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
from quantbox.contracts import (
    BrokerPlugin, DataPlugin, Mode, PluginMeta,
    RiskPlugin, RunResult, StrategyPlugin,
)
from quantbox.store import FileArtifactStore

@dataclass
class MyPipeline:
    meta = PluginMeta(
        name="my.pipeline.v1",
        kind="pipeline",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="What this pipeline does",
        outputs=("target_weights", "run_summary"),
    )

    def run(
        self,
        *,
        mode: Mode,
        asof: str,
        params: Dict[str, Any],
        data: DataPlugin,
        store: FileArtifactStore,
        broker: Optional[BrokerPlugin] = None,
        risk: Optional[List[RiskPlugin]] = None,
        strategies: Optional[List[StrategyPlugin]] = None,
        **kwargs,
    ) -> RunResult:
        # 1. Load data
        universe = data.load_universe(params)
        market_data = data.load_market_data(universe, asof, params)

        # 2. Run strategy
        if strategies:
            result = strategies[0].run(market_data, params)
            weights = result["weights"]
        else:
            # fallback: equal weight
            symbols = list(market_data["prices"].columns)
            n = len(symbols)
            weights = pd.DataFrame(
                {s: [1.0 / n] for s in symbols},
                index=[pd.Timestamp(asof)],
            )

        # 3. Risk checks
        if risk:
            for r in risk:
                findings = r.check_targets(weights, params)
                if findings:
                    ...  # handle violations

        # 4. Store artifacts
        store.put_parquet("target_weights", weights)
        summary = {"mode": mode, "asof": asof, "n_symbols": len(weights.columns)}
        store.put_json("run_summary", summary)

        # 5. Execute (paper/live only)
        if mode in ("paper", "live") and broker:
            ...  # broker execution

        return RunResult(
            run_id=store.run_id,
            pipeline_name=self.meta.name,
            mode=mode,
            asof=asof,
            artifacts={"target_weights": "target_weights.parquet"},
            metrics=summary,
            notes={},
        )
```

## Workflow 3: Building a Config

See [references/config-reference.md](references/config-reference.md) for the full config schema
and [references/plugin-catalog.md](references/plugin-catalog.md) for all available plugins.

### Config Structure

```yaml
run:
  mode: backtest|paper|live     # execution mode
  asof: "2026-02-01"            # reference date
  pipeline: "pipeline.name.v1"  # which pipeline to run

artifacts:
  root: "./artifacts"           # output directory

plugins:
  profile: "research"           # optional preset from manifest.yaml

  pipeline:
    name: "pipeline.name.v1"
    params: { ... }             # runtime params -> pipeline.run()

  data:
    name: "data.source.v1"
    params_init: { ... }        # constructor params -> dataclass fields

  strategies:                   # list of strategies with blend weights
    - name: "strategy.name.v1"
      weight: 1.0
      params: { ... }

  broker:
    name: "broker.name.v1"
    params_init: { ... }

  rebalancing:
    name: "rebalancing.type.v1"
    params: { ... }

  risk:                         # list - all must pass
    - name: "risk.type.v1"
      params: { ... }

  publishers:                   # list - all run after pipeline completes
    - name: "publisher.type.v1"
      params_init: { ... }
```

**Key distinction**: `params` are passed to `plugin.run()` at runtime. `params_init` are
passed to the dataclass constructor when instantiating the plugin.

### Profiles

Profiles in `plugins/manifest.yaml` provide preset plugin configurations:

| Profile | Use Case |
|---|---|
| `research` | Jupyter-style research with local data |
| `trading` | Spot paper trading with simulated broker |
| `trading_full` | Live crypto trading (Binance) |
| `stress_test` | Risk analysis with synthetic data |
| `futures_paper` | Futures paper trading |

## Workflow 4: Debugging

### Common Errors

| Exception | Cause | Fix |
|---|---|---|
| `ConfigValidationError` | Invalid YAML config | Check `.findings` for details |
| `PluginNotFoundError` | Plugin name not in registry | Check `.available` for valid names; run `quantbox plugins list` |
| `PluginLoadError` | Import failed | Check dependencies: `uv sync --extra full` |
| `DataLoadError` | Data fetch failed | Check API keys, network, date range |
| `BrokerExecutionError` | Order placement failed | Check broker credentials and balances |

### Diagnostic Commands

```bash
uv run quantbox plugins doctor             # check all plugins for issues
uv run quantbox plugins doctor --strict    # strict mode
uv run quantbox plugins info --name <id>   # inspect specific plugin
```

### Checking Artifacts

After a run, inspect `artifacts/<run_id>/run_manifest.json` for full metadata,
or query across runs:

```bash
uv run quantbox warehouse query --sql "SELECT * FROM runs ORDER BY created_at DESC LIMIT 5"
```

## Data Format

DataPlugin returns `Dict[str, pd.DataFrame]` of **wide-format** DataFrames:
- **Index**: DatetimeIndex
- **Columns**: symbol names (e.g., "BTC", "ETH", "SPY")
- **Required key**: `"prices"` (close prices)
- **Optional keys**: `"volume"`, `"market_cap"`, `"funding_rates"`

## Development Rules

- Use `uv` as package manager; run everything with `uv run`
- `meta` is a **class attribute**, not an instance attribute
- Prefer additive changes and new plugin versions over breaking changes
- Do not rename existing entry-point IDs
- Do not embed secrets in YAML configs; use environment variables
- Do not use `requests` in core; use `httpx` or `urllib.request`
- Add tests for new plugins or core behavior
- Write LLM-friendly docstrings with examples and "LLM Note:" hints

## Environment Variables

| Variable | Required for |
|---|---|
| `API_KEY_BINANCE` | Binance brokers |
| `API_SECRET_BINANCE` | Binance brokers |
| `HYPERLIQUID_WALLET` | Hyperliquid broker |
| `HYPERLIQUID_PRIVATE_KEY` | Hyperliquid broker |
| `TELEGRAM_TOKEN` | Telegram publisher |
| `TELEGRAM_CHAT_ID` | Telegram publisher |

None needed for backtesting or paper trading with simulated brokers.
