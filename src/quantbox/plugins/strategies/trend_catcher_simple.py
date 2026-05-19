"""Simple Trend Catcher — original PDF rules.

Implements the Robuxio TrendCatcher rules as documented in
``docs/TrendCatcher.pdf``:

* Regime filter: BTC > MA50 (long-only) or BTC < MA50 (short-only).
* Entry: close of the daily candle crossing above (long) or below (short) MA20.
* Exit: close crossing back through MA20.
* Universe: top-N by daily volume (default 30).
* Money management: fixed ``position_size`` (default 10%) per position,
  hard cap of ``max_positions`` simultaneously open. When more coins
  signal entry than there are open slots, prioritise by volume rank.

Unlike ``strategy.crypto_regime_trend.v1`` (the v2 ensemble strategy),
this plugin is event-driven, single-window, and uses literal-fraction
position sizes (not normalised over active positions). Designed for
the PDF replication research line, not the v2 notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta


@dataclass
class TrendCatcherSimpleStrategy:
    meta = PluginMeta(
        name="strategy.trend_catcher_simple.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.2.0",
        schema_version="v1",
        description="Event-driven MA20-cross trend follower with BTC>MA50 regime filter (Robuxio PDF rules).",
        tags=("trend-following", "event-driven", "single-window"),
    )

    side: str = "long"
    regime_ticker: str = "BTC"
    regime_filter_ma: int = 50
    signal_ma: int = 20
    universe_top_n_volume: int = 30
    max_positions: int = 10
    position_size: float = 0.10
    close_on_regime_flip: bool = False

    @property
    def min_lookback_periods(self) -> int:
        return max(self.regime_filter_ma, self.signal_ma) + 1

    def run(self, data: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        if self.side not in ("long", "short"):
            raise ValueError(f"side must be 'long' or 'short', got {self.side!r}")

        prices: pd.DataFrame = data["prices"]
        volume: pd.DataFrame = data.get("volume")
        if volume is None:
            raise ValueError("trend_catcher_simple requires data['volume']")
        if self.regime_ticker not in prices.columns:
            raise ValueError(f"regime ticker {self.regime_ticker!r} not in prices")

        # Align volume to prices' index/columns
        volume = volume.reindex(index=prices.index, columns=prices.columns)

        # --- 1. BTC regime filter ----------------------------------------------
        btc_close = prices[self.regime_ticker]
        btc_ma = btc_close.rolling(self.regime_filter_ma).mean()
        if self.side == "long":
            regime_ok = (btc_close > btc_ma).fillna(False)
        else:
            regime_ok = (btc_close < btc_ma).fillna(False)

        # --- 2. Per-coin signal MA + in-trend mask -----------------------------
        sig_ma = prices.rolling(self.signal_ma).mean()
        if self.side == "long":
            in_trend = (prices > sig_ma).fillna(False)
        else:
            in_trend = (prices < sig_ma).fillna(False)

        # --- 3. Top-N-by-volume universe (daily) -------------------------------
        # Rank descending: rank 1 = highest volume. Coins missing volume get NaN
        # which excludes them.
        vol_rank = volume.rank(axis=1, ascending=False, method="first")
        in_universe = (vol_rank <= self.universe_top_n_volume).fillna(False)

        # --- 4. Entry candidates -----------------------------------------------
        # Cross-above (or cross-below for short) happens on the first day a
        # coin enters the in-trend regime AFTER being out of it. Eligibility
        # at that moment requires BTC regime + top-N volume on the same day.
        first_day_of_trend = in_trend & ~in_trend.shift(1, fill_value=False)
        entry_eligible = first_day_of_trend & in_universe
        entry_eligible = entry_eligible.where(regime_ok, False)

        # --- 5. Sticky position via block-id ----------------------------------
        # Each contiguous in_trend run gets a unique block id (cumsum of
        # the gaps). Within each block, we are in position iff the entry
        # signal fired on (or before) the current day inside that block.
        # Vectorised cummax inside each block via groupby.
        block_id = (~in_trend).cumsum()
        position = pd.DataFrame(False, index=prices.index, columns=prices.columns)
        ent_float = entry_eligible.astype(np.int8)
        for col in prices.columns:
            entered_in_block = ent_float[col].groupby(block_id[col]).cummax()
            position[col] = entered_in_block.astype(bool) & in_trend[col]

        # --- 6. Position cap with volume priority -----------------------------
        # If more than ``max_positions`` are held on a given day, keep the
        # top ``max_positions`` by volume rank for that day. This is a daily
        # re-rank rather than a strict event-driven "no displace" rule —
        # documented as a deliberate approximation. Note that already-held
        # positions tend to remain in the top-N (they're typically in the
        # universe by construction), so displacement is rare in practice.
        per_day_count = position.sum(axis=1)
        if (per_day_count > self.max_positions).any():
            ranked = vol_rank.where(position)
            cand_rank = ranked.rank(axis=1, ascending=True, method="first")
            position = position & (cand_rank <= self.max_positions).fillna(False)

        # --- 6b. Optionally close on regime flip ------------------------------
        # When True: positions close immediately on the day BTC regime
        # disagrees, instead of holding until the coin's own MA20 exit.
        # PDF is silent on this; default is the conservative "hold until own
        # exit" interpretation. Set True to test whether bull-rebound losses
        # on the short side are reduced.
        if self.close_on_regime_flip:
            position = position.where(regime_ok, False)

        # --- 7. Weights ---------------------------------------------------------
        sign = 1.0 if self.side == "long" else -1.0
        weights = position.astype(float) * (self.position_size * sign)
        # Zero out bars where the asset wasn't tradeable (no price)
        weights = weights.where(prices.notna(), 0.0)

        return {
            "weights": weights,
            "details": {
                "strategy": "trend_catcher_simple.v1",
                "side": self.side,
                "regime_ticker": self.regime_ticker,
                "regime_filter_ma": self.regime_filter_ma,
                "signal_ma": self.signal_ma,
                "universe_top_n_volume": self.universe_top_n_volume,
                "max_positions": self.max_positions,
                "close_on_regime_flip": self.close_on_regime_flip,
                "position_size": self.position_size,
                "regime_long_days": int(regime_ok.sum()) if self.side == "long" else None,
                "regime_short_days": int(regime_ok.sum()) if self.side == "short" else None,
                "entry_count": int(first_day_of_trend.sum().sum()),
                "eligible_entry_count": int(entry_eligible.sum().sum()),
            },
        }
