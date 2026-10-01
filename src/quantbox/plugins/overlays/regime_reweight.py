"""Scale the whole book by a volatility regime read off a reference series.

The regime is the ratio of short- to long-window realised volatility of a
reference return series: one ``reference_symbol``, or the equal-weight mean
return of every priced instrument when it is unset. On a bar where the ratio
is at or above ``threshold`` the book is in the high-vol regime and every
weight is multiplied by ``high_vol_multiplier``; otherwise by
``low_vol_multiplier``. Bars before both windows have filled are left
untouched (no regime is known yet).

Row ``t`` uses returns up to and including ``t`` and stays on row ``t``; the
run's ``execution.lag_bars`` moves it onto the fill bar (ADR-0004).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quantbox.contracts import PluginMeta


@dataclass
class RegimeReweightOverlay:
    meta = PluginMeta(
        name="overlay.regime_reweight.v1",
        kind="overlay",
        version="0.1.0",
        core_compat=">=0.7",
        description=(
            "Multiplies the whole book by a per-regime factor; the regime is the short/long realised-vol "
            "ratio of a reference symbol (or the equal-weight universe) against a threshold."
        ),
        tags=("overlay", "regime", "volatility"),
        params_schema={
            "type": "object",
            "properties": {
                "reference_symbol": {
                    "type": ["string", "null"],
                    "default": None,
                    "description": "Symbol whose returns define the regime; null = equal-weight mean of all priced symbols.",
                },
                "short_window": {
                    "type": "integer",
                    "minimum": 2,
                    "default": 7,
                    "description": "Bars in the short realised-vol window.",
                },
                "long_window": {
                    "type": "integer",
                    "minimum": 3,
                    "default": 30,
                    "description": "Bars in the long realised-vol window; must exceed short_window.",
                },
                "threshold": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "default": 1.4,
                    "description": "Short/long vol ratio at or above which the bar is in the high-vol regime.",
                },
                "high_vol_multiplier": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.5,
                    "description": "Factor applied to every weight on a high-vol bar.",
                },
                "low_vol_multiplier": {
                    "type": "number",
                    "minimum": 0,
                    "default": 1.0,
                    "description": "Factor applied to every weight on a low-vol bar.",
                },
            },
        },
    )

    def apply(self, weights: pd.DataFrame, data: dict[str, Any], params: dict[str, Any]) -> pd.DataFrame:
        symbol = params.get("reference_symbol")
        short = int(params.get("short_window", 7))
        long = int(params.get("long_window", 30))
        threshold = float(params.get("threshold", 1.4))
        high = float(params.get("high_vol_multiplier", 0.5))
        low = float(params.get("low_vol_multiplier", 1.0))
        if short >= long:
            raise ValueError(f"{self.meta.name}: short_window ({short}) must be below long_window ({long})")

        prices: pd.DataFrame = data["prices"]
        if symbol is not None:
            if symbol not in prices.columns:
                raise ValueError(f"{self.meta.name}: reference_symbol {symbol!r} has no prices")
            ref = prices[symbol].pct_change(fill_method=None)
        else:
            ref = prices.pct_change(fill_method=None).mean(axis=1)
        ratio = ref.rolling(short).std() / ref.rolling(long).std()
        factor = pd.Series(1.0, index=ratio.index)
        known = ratio.notna()
        factor[known & (ratio >= threshold)] = high
        factor[known & (ratio < threshold)] = low
        return weights.mul(factor.reindex(weights.index).fillna(1.0), axis=0)
