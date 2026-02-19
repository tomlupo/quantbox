# Pipeline Plugin API

## PipelinePlugin Protocol

Source: `packages/quantbox-core/src/quantbox/contracts.py`

```python
class PipelinePlugin(Protocol):
    meta: PluginMeta
    kind: PipelineKind              # "research" or "trading"

    def run(
        self,
        *,
        mode: Mode,                 # "backtest"|"paper"|"live"
        asof: str,                  # Reference date "YYYY-MM-DD"
        params: dict[str, Any],
        data: DataPlugin,
        store: ArtifactStore,
        broker: BrokerPlugin | None,
        risk: list[RiskPlugin],
        strategies: list[StrategyPlugin] | None = None,
        rebalancer: RebalancingPlugin | None = None,
        **kwargs,
    ) -> RunResult: ...
```

## RunResult

```python
@dataclass
class RunResult:
    run_id: str                    # Unique identifier
    pipeline_name: str             # e.g. "my.pipeline.v1"
    mode: Mode                     # "backtest"|"paper"|"live"
    asof: str                      # Reference date
    artifacts: dict[str, str]      # artifact_name -> file_path
    metrics: dict[str, float]      # Numeric metrics
    notes: dict[str, Any]          # Freeform metadata
```

## ArtifactStore Protocol

```python
class ArtifactStore(Protocol):
    def put_parquet(self, name: str, df: pd.DataFrame) -> str: ...
    def put_json(self, name: str, obj: dict) -> str: ...
    def get_path(self, name: str) -> str: ...
    def read_parquet(self, name: str) -> pd.DataFrame: ...
    def read_json(self, name: str) -> dict: ...
    def list_artifacts(self) -> list[str]: ...
    @property
    def run_id(self) -> str: ...
```

## RebalancingPlugin Protocol

```python
class RebalancingPlugin(Protocol):
    meta: PluginMeta

    def generate_orders(
        self,
        *,
        weights: dict[str, float],  # {symbol: target_weight}
        broker: BrokerPlugin,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Returns dict with "orders" DataFrame."""
        ...
```

## PublisherPlugin Protocol

```python
class PublisherPlugin(Protocol):
    meta: PluginMeta

    def publish(self, result: RunResult, params: dict[str, Any]) -> None: ...
```

## Template

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
import pandas as pd
from quantbox.contracts import (
    BrokerPlugin, DataPlugin, Mode, PluginMeta,
    RiskPlugin, RunResult, StrategyPlugin,
)
from quantbox.store import FileArtifactStore

logger = logging.getLogger(__name__)

@dataclass
class MyPipeline:
    """Custom pipeline description.

    LLM Note: Explain what this pipeline does differently.
    """

    meta = PluginMeta(
        name="my.pipeline.v1",
        kind="pipeline",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="What this pipeline does",
        outputs=("target_weights", "run_summary"),
    )

    kind = "research"  # or "trading"

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
        logger.info("Running %s for %s in %s mode", self.meta.name, asof, mode)

        # 1. Load data
        universe = data.load_universe(params)
        market_data = data.load_market_data(universe, asof, params)
        prices = market_data["prices"]

        # 2. Run strategies
        if strategies:
            result = strategies[0].run(market_data, params)
            weights = result["weights"]
        else:
            symbols = list(prices.columns)
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
                    logger.warning("Risk violations: %s", findings)

        # 4. Store artifacts
        store.put_parquet("target_weights", weights)
        summary = {
            "mode": mode,
            "asof": asof,
            "n_symbols": len(weights.columns),
        }
        store.put_json("run_summary", summary)

        # 5. Execute (paper/live only)
        if mode in ("paper", "live") and broker:
            # ... broker execution logic
            pass

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

## Registration

1. Create: `packages/quantbox-core/src/quantbox/plugins/pipeline/my_pipeline.py`
2. Export: Add to `plugins/pipeline/__init__.py`
3. Register in `plugins/builtins.py`:
   ```python
   from .pipeline import MyPipeline
   "pipeline": _map(..., MyPipeline),
   ```

## Key Patterns

- Use `mode` to branch: backtest = no execution, paper/live = execute via broker
- List artifact names in `meta.outputs` so runner can validate schemas
- Write artifacts via `store.put_parquet()` and `store.put_json()`
- Always return `RunResult` with meaningful `metrics`
