"""Trend-catcher strategy — MA crossover with BTC regime filter.

Notebook-faithful replication of the Robuxio TrendCatcher v1 strategy
(``research/crypto/robuxio-trend-catcher/robuxio-trend-catcher-v1.ipynb``).

Distinct from :class:`strategy.crypto_regime_trend.v1` in three ways:

1. **Selection by dollar volume among signals** (notebook v1). Surviving
   signals are ranked by ``close * volume`` and the top ``max_positions`` are
   kept. ``crypto_regime_trend.v1`` does universe selection upstream via the
   shared mcap/volume tier system before signals fire.
2. **Fixed 1/max_positions per selected name** (notebook v1). Rows are NOT
   normalised to sum to 1 — if fewer than ``max_positions`` names signal on a
   day, weights sum < 1 and the rest is cash. ``crypto_regime_trend.v1``
   normalises rows to sum to 1.
3. **True 14-day ATR for RW sizing** (notebook v1) using
   ``data["high"]`` / ``data["low"]``. Falls back to a close-only proxy
   (``|close.diff()|``) when high/low aren't available. ``crypto_regime_trend.v1``
   uses ``abs(daily returns)`` as a permanent close-only ATR proxy.

This strategy is the right pick when faithfulness to the v1 paper math is the
goal. Use ``strategy.crypto_regime_trend.v1`` when you want broader research
features (long/short, V2 multi-window ensemble, long+short ensemble variants).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta


@dataclass
class TrendCatcherStrategy:
    meta = PluginMeta(
        name="strategy.trend_catcher.v1",
        kind="strategy",
        version="0.2.0",
        core_compat=">=0.2.0",
        schema_version="v1",
        description="MA-crossover trend-following with BTC regime filter (Robuxio TrendCatcher v1).",
        tags=("crypto", "trend-following", "long-only"),
    )

    regime_window: int = 50
    trend_window: int = 20
    max_positions: int = 10
    sizing: str = "ew"
    atr_window: int = 14
    regime_ticker: str = "BTC"
    # Deprecated legacy params kept so old configs still load. The replication
    # path does not use them — RW is controlled by sizing="rw".
    vol_target: float | None = None
    vol_lookback: int = 60

    @property
    def min_lookback_periods(self) -> int:
        return max(self.regime_window, self.trend_window, self.atr_window) + 1

    def run(self, data: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        # Back-compat: if a config still passes vol_target=<float>, treat that
        # as a request for the RW path (matches the previous semantics where
        # any non-None vol_target switched on inverse-vol sizing).
        if self.vol_target is not None and self.sizing == "ew":
            self.sizing = "rw"

        prices: pd.DataFrame = data["prices"]
        volume = data.get("volume")
        high = data.get("high")
        low = data.get("low")

        elig = data.get("eligibility_mask")
        if isinstance(elig, pd.DataFrame) and not elig.empty:
            elig = elig.reindex(index=prices.index, columns=prices.columns).fillna(False).astype(bool)
        else:
            elig = pd.DataFrame(True, index=prices.index, columns=prices.columns)

        trend_ma = prices.rolling(self.trend_window).mean()
        trend_signal = prices > trend_ma

        btc_col = next(
            (c for c in prices.columns if c in (self.regime_ticker, self.regime_ticker + "USDT")),
            None,
        )
        if btc_col is not None:
            btc_ma = prices[btc_col].rolling(self.regime_window).mean()
            regime = prices[btc_col] > btc_ma
        else:
            regime = pd.Series(True, index=prices.index)

        # Signal = regime ON AND close > MA(trend_window) AND in eligibility mask
        signal = trend_signal.where(elig, False)
        signal = signal.mul(regime.astype(bool), axis=0).fillna(False).astype(bool)

        # Top-N selection: rank by dollar volume (close * volume). Matches the
        # notebook: `(volume*close).where(long_signal).rank(method='first')`.
        if isinstance(volume, pd.DataFrame) and not volume.empty:
            volume = volume.reindex(index=prices.index, columns=prices.columns)
            dollar_volume = prices * volume
        else:
            # No volume → fall back to close as ranking key. Not notebook-faithful
            # but keeps the strategy runnable on close-only synthetic data in tests.
            dollar_volume = prices
        rank_key = dollar_volume.where(signal)
        rank = rank_key.rank(axis=1, method="first", ascending=False)
        selected = (rank <= self.max_positions) & signal

        # Equal-weight path: fixed 1/max_positions per selected name. Row sums
        # are <= 1 (cash on unused slots) — notebook-faithful.
        per_name = 1.0 / float(self.max_positions)
        weights_ew = selected.astype(float) * per_name

        if self.sizing == "rw":
            # Risk-weight via 14-day ATR/close. Mirrors notebook math:
            #   tr = max(high-low, |high - close.shift()|, |low - close.shift()|)
            #   atr = tr.rolling(14).mean() / close
            #   inverse_vol = (1/atr).div(sum_per_row)
            #   weight = (inverse_vol * selected).div(sum_per_row) * weights_ew.sum()
            if (
                isinstance(high, pd.DataFrame)
                and isinstance(low, pd.DataFrame)
                and not high.empty
                and not low.empty
            ):
                high_a = high.reindex(index=prices.index, columns=prices.columns)
                low_a = low.reindex(index=prices.index, columns=prices.columns)
                close_shift = prices.shift(1)
                tr1 = high_a - low_a
                tr2 = (high_a - close_shift).abs()
                tr3 = (low_a - close_shift).abs()
                tr = np.maximum.reduce([tr1.values, tr2.values, tr3.values])
                true_range = pd.DataFrame(tr, index=prices.index, columns=prices.columns)
                atr_source = "true_atr"
            else:
                # No OHLC: close-only proxy. Same rolling window, same inverse-vol
                # normalisation. Documented as a fallback when high/low unavailable.
                true_range = prices.diff().abs()
                atr_source = "close_only_diff"

            atr = true_range.rolling(self.atr_window, min_periods=1).mean() / prices
            inverse = 1.0 / atr.replace(0.0, np.nan)
            inverse_norm = inverse.div(inverse.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
            raw = inverse_norm.where(selected, 0.0)
            raw_sum = raw.sum(axis=1).replace(0, np.nan)
            ew_sum = weights_ew.sum(axis=1)
            weights = raw.div(raw_sum, axis=0).mul(ew_sum, axis=0).fillna(0.0)
        else:
            atr_source = None
            weights = weights_ew

        details = {
            "strategy": "trend_catcher.v1",
            "regime": regime.astype(float),
            "selected_mask": selected,
            "sizing": self.sizing,
            "atr_source": atr_source,
            "diagnostics": {
                "regime_overlay": {
                    "ref_ticker": str(self.regime_ticker),
                    "fast_window": int(self.trend_window),
                    "slow_window": int(self.regime_window),
                },
                "signal_count": {
                    "series": selected.sum(axis=1).astype(int),
                    "cap": int(self.max_positions),
                    "label": "Long signals",
                },
            },
        }
        return {"weights": weights, "details": details}
