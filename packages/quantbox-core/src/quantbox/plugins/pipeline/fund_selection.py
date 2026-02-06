from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd

from quantbox.contracts import PluginMeta, RunResult, Mode, DataPlugin, ArtifactStore, BrokerPlugin, RiskPlugin

@dataclass
class FundSelectionPipeline:
    meta = PluginMeta(
        name="fund_selection.simple.v1",
        kind="pipeline",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Simple ranking pipeline (research)",
        tags=("research","ranking"),
        capabilities=("research",),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "minimum": 1, "default": 5},
                "universe": {"type": "object", "properties": {"symbols": {"type":"array","items":{"type":"string"}}}, "required":["symbols"]},
                "prices": {"type": "object", "properties": {"lookback_days": {"type":"integer","minimum": 30, "default": 365}}},
            },
            "required": ["universe"]
        },
        outputs=("scores","rankings","allocations"),
        examples=(
            "plugins:\n  pipeline:\n    name: fund_selection.simple.v1\n    params:\n      top_n: 5\n      universe:\n        symbols: [SPY, QQQ]\n      prices:\n        lookback_days: 365",
        ),
    )
    kind = "research"

    def run(
        self,
        *,
        mode: Mode,
        asof: str,
        params: Dict[str, Any],
        data: DataPlugin,
        store: ArtifactStore,
        broker: Optional[BrokerPlugin],
        risk: List[RiskPlugin],
        **kwargs,
    ) -> RunResult:
        universe = data.load_universe(params.get("universe", {}))
        store.put_parquet("universe", universe)

        market_data = data.load_market_data(universe, asof, params.get("prices", {}))
        prices_wide = market_data["prices"]
        store.put_parquet("prices", prices_wide.reset_index() if hasattr(prices_wide.index, 'name') else prices_wide)

        returns = prices_wide.pct_change()
        scores = pd.DataFrame({
            "symbol": returns.columns,
            "score": returns.mean().values,
        })
        scores["asof"] = asof

        rankings = scores.sort_values("score", ascending=False).reset_index(drop=True)
        rankings["rank"] = rankings.index + 1

        top_n = int(params.get("top_n", 5))
        alloc = rankings.head(top_n).copy()
        alloc["weight"] = 1.0 / max(top_n, 1)

        a_scores = store.put_parquet("scores", scores)
        a_rank = store.put_parquet("rankings", rankings)
        a_alloc = store.put_parquet("allocations", alloc)

        metrics = {
            "n_universe": float(len(universe)),
            "top_n": float(top_n),
            "mean_score_top": float(alloc["score"].mean()) if len(alloc) else 0.0,
        }

        return RunResult(
            run_id=store.run_id,
            pipeline_name=self.meta.name,
            mode=mode,
            asof=asof,
            artifacts={
                "scores": a_scores,
                "rankings": a_rank,
                "allocations": a_alloc,
            },
            metrics=metrics,
            notes={"kind": "research"},
        )
