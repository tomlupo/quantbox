"""Static (constant) weights strategy — buy-and-hold / fixed allocation.

Useful as a baseline (e.g. BTC buy-and-hold) and for tests/mockups. Holds the
configured weights at every bar of the backtest window. Weights are renormalised
so they sum to 1.0 across the requested symbols.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantbox.contracts import PluginMeta


@dataclass
class StaticWeightsStrategy:
    meta = PluginMeta(
        name="strategy.static_weights.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.2.0",
        schema_version="v1",
        description="Holds a fixed weight allocation (e.g. BTC buy-and-hold) across all bars.",
        tags=("baseline", "buy-and-hold", "static"),
    )

    weights: dict[str, float] = field(default_factory=dict)
    normalise: bool = True

    @property
    def min_lookback_periods(self) -> int:
        return 1

    def run(self, data: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        prices: pd.DataFrame = data["prices"]
        target = dict(self.weights or {})
        if not target:
            empty = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            return {"weights": empty, "details": {"strategy": "static_weights.v1"}}

        # Restrict to symbols that exist in the price universe.
        held = {k: float(v) for k, v in target.items() if k in prices.columns}
        if self.normalise and held:
            total = sum(abs(v) for v in held.values())
            if total > 0:
                held = {k: v / total for k, v in held.items()}

        w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for sym, val in held.items():
            w[sym] = val

        # Zero out rows where the asset wasn't tradeable yet (no price)
        mask = prices.notna()
        w = w.where(mask, 0.0)

        return {
            "weights": w,
            "details": {
                "strategy": "static_weights.v1",
                "configured_weights": held,
            },
        }
