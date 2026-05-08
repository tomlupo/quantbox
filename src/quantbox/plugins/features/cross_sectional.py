"""Cross-sectional feature engineering plugin.

Computes cross-sectional statistics (z-scores and percentile ranks) of
returns across the asset universe at each date. Output is a DataFrame with
``(date, symbol)`` MultiIndex.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quantbox.contracts import PluginMeta

_DEFAULT_HORIZONS = [5, 10, 20, 60]
_DEFAULT_METHODS = ["zscore", "percentile"]


@dataclass
class CrossSectionalFeatures:
    meta = PluginMeta(
        name="features.cross_sectional.v1",
        kind="feature",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Cross-sectional z-scores and percentile ranks of returns "
            "across the asset universe at each date."
        ),
        tags=("cross-sectional", "feature"),
        params_schema={
            "type": "object",
            "properties": {
                "horizons": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "default": _DEFAULT_HORIZONS,
                    "description": "Return horizons (in days) to compute cross-sectional stats for.",
                },
                "methods": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["zscore", "percentile"]},
                    "default": _DEFAULT_METHODS,
                    "description": "Cross-sectional methods to apply.",
                },
            },
        },
    )

    def compute(self, data: dict[str, pd.DataFrame], params: dict[str, Any]) -> pd.DataFrame:
        prices = data["prices"]
        horizons: list[int] = params.get("horizons", _DEFAULT_HORIZONS)
        methods: list[str] = params.get("methods", _DEFAULT_METHODS)

        features = pd.DataFrame(index=prices.index)

        for horizon in horizons:
            returns = prices.pct_change(horizon)

            if "zscore" in methods:
                cross_mean = returns.mean(axis=1)
                cross_std = returns.std(axis=1)
                zscore = returns.sub(cross_mean, axis=0).div(cross_std, axis=0)
                for symbol in prices.columns:
                    features[(f"return_{horizon}d_zscore", symbol)] = zscore[symbol]

            if "percentile" in methods:
                rank = returns.rank(axis=1, method="first")
                count = returns.count(axis=1)
                percentile = rank.div(count, axis=0)
                for symbol in prices.columns:
                    features[(f"return_{horizon}d_percentile", symbol)] = percentile[symbol]

        # Reshape from wide (feature, symbol) columns into stacked (date, symbol) MultiIndex
        features.columns = pd.MultiIndex.from_tuples(features.columns, names=["feature", "symbol"])
        stacked = features.stack(level="symbol", future_stack=True)
        stacked.index.names = ["date", "symbol"]
        return stacked
