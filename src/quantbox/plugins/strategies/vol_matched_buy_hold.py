"""Vol-matched buy-and-hold benchmark strategy.

Synthesises a constant-vol-targeted version of a single asset's buy-and-hold
return stream. Implements the Robuxio TrendCatcher v2 notebook cells 124-127:

    scale_t        = target_annual_vol / (sigma_d * sqrt(trading_days))
    weight[t, ticker] = scale_t
    weight[t, other]  = 0

With ``vol_lookback=None`` (default), ``sigma_d`` is the full-sample daily
return std — non-causal but matches the notebook's benchmark construction
(``annualized_vol_prices`` in cell 125 uses ``daily_returns.std()``). Pass
an integer ``vol_lookback`` for a causal rolling-window version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta


@dataclass
class VolMatchedBuyHoldStrategy:
    meta = PluginMeta(
        name="strategy.vol_matched_buy_hold.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.2.0",
        schema_version="v1",
        description="Buy-and-hold a single asset, scaled to a target annualized volatility.",
        tags=("benchmark", "buy-and-hold", "vol-matched"),
    )

    ticker: str = "BTC"
    target_annual_vol: float = 0.25
    vol_lookback: int | None = None
    trading_days: int = 365

    @property
    def min_lookback_periods(self) -> int:
        return self.vol_lookback if self.vol_lookback is not None else 2

    def run(self, data: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        prices: pd.DataFrame = data["prices"]
        if self.ticker not in prices.columns:
            raise ValueError(
                f"vol_matched_buy_hold: ticker {self.ticker!r} not present in price universe "
                f"(have {len(prices.columns)} columns)"
            )

        rets = prices[self.ticker].ffill().pct_change(fill_method=None)
        sqrt_td = float(np.sqrt(self.trading_days))

        if self.vol_lookback is None:
            sigma_d = float(rets.std())
            if not np.isfinite(sigma_d) or sigma_d <= 0:
                raise ValueError(f"vol_matched_buy_hold: degenerate daily-return std={sigma_d} for {self.ticker!r}")
            scale = self.target_annual_vol / (sigma_d * sqrt_td)
            scale_series = pd.Series(scale, index=prices.index, dtype=float)
        else:
            sigma_d_series = rets.rolling(int(self.vol_lookback)).std()
            scale_series = self.target_annual_vol / (sigma_d_series * sqrt_td)
            scale_series = scale_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        weights[self.ticker] = scale_series

        # Zero out bars where the asset has no price (pre-listing / delisted).
        weights = weights.where(prices.notna(), 0.0)

        return {
            "weights": weights,
            "details": {
                "strategy": "vol_matched_buy_hold.v1",
                "ticker": self.ticker,
                "target_annual_vol": self.target_annual_vol,
                "vol_lookback": self.vol_lookback,
                "trading_days": self.trading_days,
                "scale_first": float(scale_series.iloc[0]) if len(scale_series) else float("nan"),
                "scale_last": float(scale_series.iloc[-1]) if len(scale_series) else float("nan"),
            },
        }
