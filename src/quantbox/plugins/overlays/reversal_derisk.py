"""De-risk an instrument for a few bars after its fast trend signal flips sign.

The mechanism of H22 (trend reversals lead the volatility spike): when the
EWMAC(fast_span, slow_span) of an instrument's close changes sign on bar ``t``,
that instrument's decided weight is multiplied by ``multiplier`` on bars
``t .. t + hold_bars - 1``. A new flip inside the window restarts it. Other
instruments are untouched.

Row ``t`` uses closes up to and including ``t`` and stays on row ``t``; the
run's ``execution.lag_bars`` moves it onto the fill bar (ADR-0004).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta


@dataclass
class ReversalDeriskOverlay:
    meta = PluginMeta(
        name="overlay.reversal_derisk.v1",
        kind="overlay",
        version="0.1.0",
        core_compat=">=0.7",
        description=(
            "Multiplies an instrument's weight by `multiplier` for `hold_bars` bars after its "
            "EWMAC(fast_span, slow_span) changes sign (H22 reversal de-risk)."
        ),
        tags=("overlay", "de-risk", "trend", "reversal"),
        params_schema={
            "type": "object",
            "properties": {
                "fast_span": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 2,
                    "description": "EWMA span (bars) of the fast leg of the watched EWMAC.",
                },
                "slow_span": {
                    "type": "integer",
                    "minimum": 2,
                    "default": 8,
                    "description": "EWMA span (bars) of the slow leg; must exceed fast_span.",
                },
                "multiplier": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.5,
                    "description": "Factor applied to the instrument's weight while the de-risk window is open.",
                },
                "hold_bars": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 5,
                    "description": "Length of the de-risk window in bars, counting the flip bar itself.",
                },
            },
        },
    )

    def apply(self, weights: pd.DataFrame, data: dict[str, Any], params: dict[str, Any]) -> pd.DataFrame:
        fast = int(params.get("fast_span", 2))
        slow = int(params.get("slow_span", 8))
        multiplier = float(params.get("multiplier", 0.5))
        hold = int(params.get("hold_bars", 5))
        if fast >= slow:
            raise ValueError(f"{self.meta.name}: fast_span ({fast}) must be below slow_span ({slow})")
        if hold < 1 or multiplier < 0:
            raise ValueError(f"{self.meta.name}: hold_bars must be >= 1 and multiplier >= 0")

        prices = data["prices"].reindex(columns=weights.columns)
        ewmac = prices.ewm(span=fast, adjust=False).mean() - prices.ewm(span=slow, adjust=False).mean()
        sign = np.sign(ewmac)
        flipped = (sign * sign.shift(1)) < 0  # strictly opposite signs on t-1 and t; NaN is no flip
        window_open = flipped.astype(float).rolling(hold, min_periods=1).max().reindex(weights.index).fillna(0.0) > 0
        return weights.where(~window_open, weights * multiplier)
