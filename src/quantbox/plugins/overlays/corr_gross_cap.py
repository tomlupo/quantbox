"""Cap gross exposure on bars where the held instruments move together.

On each bar, the average pairwise correlation of trailing ``window``-bar
returns is taken over the instruments the book holds (non-zero weight). When
it exceeds ``corr_threshold`` and gross exposure (sum of |weights|) is above
``max_gross``, the whole row is scaled down proportionally to ``max_gross``.
A bar with fewer than two held instruments, or before the window has filled,
is left untouched.

Row ``t`` uses returns up to and including ``t`` and stays on row ``t``; the
run's ``execution.lag_bars`` moves it onto the fill bar (ADR-0004).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta


def mean_pairwise_corr(returns: pd.DataFrame, held: pd.DataFrame, window: int) -> pd.Series:
    """Per bar, the mean trailing-``window`` correlation over pairs of ``held`` columns (NaN: < 1 valid pair)."""
    n_dates, n_cols = returns.shape
    corr = returns.rolling(window).corr().to_numpy().reshape(n_dates, n_cols, n_cols)
    mask = held.to_numpy(dtype=float)
    pairs = mask[:, :, None] * mask[:, None, :] * (1.0 - np.eye(n_cols))[None, :, :]
    valid = ~np.isnan(corr)
    pairs = pairs * valid
    total = (np.where(valid, corr, 0.0) * pairs).sum(axis=(1, 2))
    count = pairs.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    return pd.Series(mean, index=returns.index)


@dataclass
class CorrGrossCapOverlay:
    meta = PluginMeta(
        name="overlay.corr_gross_cap.v1",
        kind="overlay",
        version="0.1.0",
        core_compat=">=0.7",
        description=(
            "Scales a bar's weights down to `max_gross` gross exposure when the mean pairwise "
            "trailing correlation of the held instruments exceeds `corr_threshold`."
        ),
        tags=("overlay", "gross-cap", "correlation", "risk"),
        params_schema={
            "type": "object",
            "properties": {
                "window": {
                    "type": "integer",
                    "minimum": 2,
                    "default": 30,
                    "description": "Bars of trailing returns the pairwise correlations are measured over.",
                },
                "corr_threshold": {
                    "type": "number",
                    "minimum": -1,
                    "maximum": 1,
                    "default": 0.6,
                    "description": "Mean pairwise correlation above which the gross cap applies.",
                },
                "max_gross": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "default": 1.0,
                    "description": "Gross exposure (sum of |weights|) a gated bar is scaled down to.",
                },
            },
        },
    )

    def apply(self, weights: pd.DataFrame, data: dict[str, Any], params: dict[str, Any]) -> pd.DataFrame:
        window = int(params.get("window", 30))
        threshold = float(params.get("corr_threshold", 0.6))
        max_gross = float(params.get("max_gross", 1.0))
        if max_gross <= 0:
            raise ValueError(f"{self.meta.name}: max_gross must be > 0")

        prices = data["prices"].reindex(columns=weights.columns)
        returns = prices.pct_change(fill_method=None)
        held = weights.reindex(returns.index).fillna(0.0).ne(0.0)
        corr = mean_pairwise_corr(returns, held, window).reindex(weights.index)
        gross = weights.abs().sum(axis=1)
        gated = (corr > threshold) & (gross > max_gross)
        scale = pd.Series(1.0, index=weights.index).where(~gated, max_gross / gross)
        return weights.mul(scale, axis=0)
