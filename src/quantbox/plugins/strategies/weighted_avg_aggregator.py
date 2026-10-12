"""Weighted-average meta-strategy aggregator.

Composes multiple sub-strategy results into a single set of weights
using a weighted average.  This is a StrategyPlugin so it can be
injected via config just like any other strategy.

The aggregation itself is the strategy runner's one aggregation,
:func:`quantbox.strategy_runner.decided_weights` (TOM-1712); this plugin adds
no arithmetic of its own, so a config that injects it and one that does not
decide the same weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quantbox.contracts import PluginMeta, StrategyContext
from quantbox.strategy_runner import decided_weights, latest_weights


@dataclass
class WeightedAverageAggregator:
    """Aggregate sub-strategy weights via weighted average.

    Expects ``data["strategy_results"]`` to be a dict of::

        { strategy_name: {"result": {..., "weights": DataFrame}, "weight": float} }

    Returns a dict with:
    - ``weights``: aggregated weights as a DataFrame (date x ticker)
    - ``simple_weights``: last-row weights as ``{ticker: float}``
    - ``details``: empty (kept for the StrategyPlugin result shape)
    """

    meta = PluginMeta(
        name="strategy.weighted_avg.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Weighted-average meta-strategy aggregator",
        tags=("aggregator", "meta-strategy"),
        params_schema={
            "type": "object",
            "properties": {
                "strategy_weights": {
                    "type": "object",
                    "default": {},
                    "description": "Per-strategy weight overrides {strategy name: weight}.",
                }
            },
        },
    )

    def run(
        self, data: dict[str, Any], params: dict[str, Any], context: StrategyContext | None = None
    ) -> dict[str, Any]:
        strategy_results: dict[str, dict[str, Any]] = data.get("strategy_results", {})
        history = decided_weights(strategy_results, params.get("strategy_weights"))
        return {"weights": history, "simple_weights": latest_weights(history), "details": {}}
