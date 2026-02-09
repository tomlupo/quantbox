# Recipe: Add a pipeline plugin

Two options:

## Option A: Built-in plugin (recommended for core plugins)

### 1. Create the module

Add a new file under `packages/quantbox-core/src/quantbox/plugins/pipeline/`.

```python
# packages/quantbox-core/src/quantbox/plugins/pipeline/my_pipeline.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from quantbox.contracts import (
    BrokerPlugin,
    DataPlugin,
    Mode,
    PluginMeta,
    RiskPlugin,
    RunResult,
    StrategyPlugin,
)
from quantbox.store import FileArtifactStore

logger = logging.getLogger(__name__)


@dataclass
class MyCustomPipeline:
    """Custom pipeline that loads data, runs a strategy, and stores results.

    LLM Note: This pipeline demonstrates the standard pattern. It:
    1. Loads market data from the data plugin
    2. Runs the strategy to get target weights
    3. Optionally executes via broker (paper/live modes)
    4. Stores artifacts (parquet + JSON)
    """

    meta = PluginMeta(
        name="my.custom_pipeline.v1",
        kind="pipeline",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Custom pipeline with data load, strategy, and artifact storage",
        tags=("research",),
        schema_version="v1",
        outputs=("target_weights", "run_summary"),
    )

    def run(
        self,
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
        """Execute the pipeline.

        Args:
            mode: "backtest", "paper", or "live"
            asof: Reference date (YYYY-MM-DD)
            params: Pipeline params from config
            data: Data plugin instance
            store: Artifact store for outputs
            broker: Broker plugin (required for paper/live)
            risk: List of risk plugins
            strategies: List of strategy plugins
        """
        logger.info("Running %s for %s in %s mode", self.meta.name, asof, mode)

        # 1. Load market data
        universe = data.load_universe(params)
        market_data = data.load_market_data(universe, asof, params)
        prices = market_data["prices"]
        logger.info("Loaded %d symbols, %d dates", len(prices.columns), len(prices))

        # 2. Run strategy
        if strategies:
            strategy = strategies[0]
            result = strategy.run(market_data, params)
            weights = result.get("weights", pd.DataFrame())
        else:
            # Default: equal weight
            symbols = list(prices.columns)
            n = len(symbols)
            weights = pd.DataFrame(
                {s: [1.0 / n] for s in symbols},
                index=[pd.Timestamp(asof)],
            )

        # 3. Risk checks
        if risk:
            for r in risk:
                violations = r.check_targets(weights, params)
                if violations:
                    logger.warning("Risk violations: %s", violations)

        # 4. Store artifacts
        store.put_parquet("target_weights", weights)
        summary = {
            "mode": mode,
            "asof": asof,
            "n_symbols": len(weights.columns),
            "strategy": strategies[0].meta.name if strategies else "equal_weight",
        }
        store.put_json("run_summary", summary)

        # 5. Execute (paper/live only)
        if mode in ("paper", "live") and broker:
            broker.execute_rebalancing(weights)

        return RunResult(
            run_id=store.run_id,
            pipeline_name=self.meta.name,
            artifacts={"target_weights": weights, "run_summary": summary},
            metrics=summary,
        )
```

### 2. Export from `__init__.py`

```python
# In plugins/pipeline/__init__.py, add:
from .my_pipeline import MyCustomPipeline
```

### 3. Register in builtins

```python
# In plugins/builtins.py, add to imports:
from .pipeline import MyCustomPipeline

# Add to the "pipeline" line in builtins():
"pipeline": _map(..., MyCustomPipeline),
```

### 4. Add example config

```yaml
# configs/run_my_custom.yaml
run:
  mode: backtest
  asof: "2026-01-31"
  pipeline: "my.custom_pipeline.v1"

artifacts:
  root: "./artifacts"

plugins:
  pipeline:
    name: "my.custom_pipeline.v1"
    params:
      lookback_days: 365
      universe:
        symbols: ["SPY", "QQQ", "IWM", "TLT", "GLD"]

  data:
    name: "local_file_data"
    params_init:
      prices_path: "./data/curated/prices.parquet"

  # Strategies are optional — pipeline falls back to equal weight
  # strategies:
  #   - name: "strategy.portfolio_optimizer.v1"
  #     weight: 1.0
  #     params:
  #       method: "max_sharpe"

  risk: []
  publishers: []
```

### 5. Verify

```bash
uv run quantbox plugins list          # Should show my.custom_pipeline.v1
uv run quantbox validate -c configs/run_my_custom.yaml
uv run quantbox run --dry-run -c configs/run_my_custom.yaml
uv run pytest -q                      # All tests still pass
```

## Option B: External plugin (separate repo/package)

### 1. Create package

```
my-quantbox-pipeline/
  pyproject.toml
  src/my_pipeline/
    __init__.py
    pipeline.py        # MyCustomPipeline class
```

### 2. Register via entry point

```toml
# pyproject.toml
[project.entry-points."quantbox.pipelines"]
"my.custom_pipeline.v1" = "my_pipeline.pipeline:MyCustomPipeline"
```

### 3. Install and verify

```bash
uv pip install -e ./my-quantbox-pipeline
uv run quantbox plugins list          # Should show my.custom_pipeline.v1
uv run quantbox plugins doctor        # Check for issues
```

The plugin will be discovered automatically by quantbox's entry point scanner.

## Key patterns

- Write artifacts using `store.put_parquet(name, df)` and `store.put_json(name, obj)`
- Return `RunResult` with `run_id`, `pipeline_name`, `artifacts`, `metrics`
- Use `mode` to branch behavior (backtest = no execution, paper/live = execute)
- List artifact names in `meta.outputs` for schema validation
