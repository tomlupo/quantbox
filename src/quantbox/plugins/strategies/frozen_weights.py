"""Frozen-weights strategy — replay a dated weight grid against the universe.

For porting externally-computed strategies (elastic-net trackers, RP optimizer
outputs, regulator filings, published model portfolios) where the weights are
known but the *generator* is too expensive or not available to re-run inside
quantbox.

Different from :class:`StaticWeightsStrategy`, which holds a single constant
weight dict across all bars. This plugin loads a *dated* weight grid (e.g.
weekly) from parquet and forward-fills onto the price index. The pipeline's
rebalance engine (vectorbt buy-and-hold between rebalances) handles the
actual fills — the plugin only outputs target weights.

Weights file format
-------------------
Parquet with DatetimeIndex and ticker columns. Each row is a target weight
vector. Sums do NOT need to be 1.0 (you can express cash residual <1 or
leverage >1) — by default the plugin preserves the source sums.

Examples
--------
::

    plugins:
      strategies:
        - name: "frozen_weights"
          weight: 1.0
          params:
            weights_path: "./data/passive_weights_weekly.parquet"
            rebase_date: "2013-05-31"
            missing_tickers: "fail"
            renormalize: false
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta


@dataclass
class FrozenWeightsStrategy:
    meta = PluginMeta(
        name="strategy.frozen_weights.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.2.0",
        schema_version="v1",
        description=(
            "Replay a dated weight grid loaded from parquet against the "
            "universe. For porting externally-computed strategies "
            "(elastic-net, RP optimizer, model portfolios)."
        ),
        tags=("baseline", "frozen-weights", "external"),
        params_schema={
            "type": "object",
            "properties": {
                "weights_path": {
                    "type": "string",
                    "description": (
                        "Path to a parquet file. Must have a DatetimeIndex "
                        "and ticker columns. Each row is a target weight "
                        "vector."
                    ),
                },
                "rebase_date": {
                    "type": "string",
                    "description": (
                        "ISO date. Weights before this date are forced to "
                        "zero (warmup). Defaults to the first weight date."
                    ),
                },
                "missing_tickers": {
                    "type": "string",
                    "enum": ["fail", "zero"],
                    "default": "fail",
                    "description": (
                        "What to do if a column in the weights file is not "
                        "present in data['prices']: 'fail' raises, 'zero' "
                        "silently drops it."
                    ),
                },
                "renormalize": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "If true, each output row is rescaled to sum to 1.0. "
                        "Default false preserves source sums (so cash residual "
                        "and intentional leverage from the source are kept)."
                    ),
                },
            },
            "required": ["weights_path"],
        },
        inputs=("prices",),
        outputs=("weights",),
    )

    weights_path: str = ""
    rebase_date: str | None = None
    missing_tickers: str = "fail"
    renormalize: bool = False

    @property
    def min_lookback_periods(self) -> int:
        return 1

    def run(self, data: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        if not self.weights_path:
            raise ValueError("strategy.frozen_weights.v1: 'weights_path' is required")

        prices: pd.DataFrame = data["prices"]
        if prices.empty:
            empty = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            return {"weights": empty, "details": {"strategy": "frozen_weights.v1", "n_rebalances": 0}}

        path = Path(self.weights_path)
        if not path.exists():
            raise FileNotFoundError(f"strategy.frozen_weights.v1: weights file not found at {path}")

        weights = pd.read_parquet(path)
        if not isinstance(weights.index, pd.DatetimeIndex):
            weights.index = pd.to_datetime(weights.index)

        missing = [c for c in weights.columns if c not in prices.columns]
        if missing:
            if self.missing_tickers == "fail":
                raise KeyError(
                    f"strategy.frozen_weights.v1: weights file references tickers "
                    f"not in data['prices']: {missing}. Pass missing_tickers='zero' "
                    "to silently drop them."
                )
            weights = weights.drop(columns=missing)

        # Forward-fill weekly/monthly weights onto the daily price grid.
        # Use union(prices.index, weights.index) so weight dates that fall on
        # non-trading days still anchor the carry-forward.
        daily_w = weights.reindex(prices.index.union(weights.index)).ffill().reindex(prices.index).fillna(0.0)

        # Zero out anything before rebase_date (warmup region).
        rebase = pd.Timestamp(self.rebase_date) if self.rebase_date else weights.index.min()
        daily_w.loc[daily_w.index < rebase, :] = 0.0

        if self.renormalize:
            row_sum = daily_w.sum(axis=1).replace(0, np.nan)
            daily_w = daily_w.div(row_sum, axis=0).fillna(0.0)

        # Emit on the FULL prices column space (zeros for non-strategy tickers).
        out = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        out.loc[:, daily_w.columns] = daily_w.values

        # Zero out positions for tickers that aren't yet listed (no price).
        out = out.where(prices.notna(), 0.0)

        return {
            "weights": out,
            "details": {
                "strategy": "frozen_weights.v1",
                "weights_path": str(path),
                "rebase_date": str(rebase.date()) if hasattr(rebase, "date") else str(rebase),
                "n_rebalances": int(len(weights)),
                "tickers": list(daily_w.columns),
            },
        }
