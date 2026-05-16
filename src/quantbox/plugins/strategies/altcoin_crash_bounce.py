"""Altcoin Crash Bounce — mean-reversion entry on capitulation.

Long-only strategy that enters altcoin positions after a significant price
crash + volume spike (capitulation signal), with TP/SL/time-decay exits.

Entry:
  - Price drops >= 15% from the 24h rolling high
  - Volume >= 1.3x the 480h (20-day) average volume

Exit (first triggered wins):
  - Take Profit: +12%
  - Stop Loss:   -12%
  - Time Decay:  72h max hold

Sizing: 8% per position, max 15 concurrent, circuit breaker (6 entries/24h).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from quantbox.contracts import PluginMeta

# Canonical non-tradeable-crypto exclusion list — sourced from quantbox-datasets'
# catalog/asset_categories.yaml via quantbox.plugins.strategies._universe. The
# legacy hardcoded list (30 symbols) was a strict subset of the canonical (43);
# switching makes the strategy reject the same names plus the previously-missed
# wrapped/staked-ETH derivatives (STETH, WSTETH, RETH, WBETH, WEETH, EZETH,
# LBTC), newer USD stables (USDS, USDE, RLUSD), EURO stable (EUROC), and promo
# tokens (CHIP, MEGA). None of these were "correctly tradeable" under the
# strategy's intent; backtests touching periods where those tokens were
# top-of-book will produce different curves.
from quantbox.plugins.strategies._universe import DEFAULT_STABLECOINS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def compute_crash_signals(
    prices: pd.DataFrame,
    lookback_periods: int = 24,
    crash_threshold_pct: float = -15.0,
) -> pd.DataFrame:
    rolling_high = prices.rolling(window=lookback_periods, min_periods=lookback_periods).max()
    return (prices - rolling_high) / rolling_high * 100


def compute_volume_ratio(
    volume: pd.DataFrame,
    lookback_periods: int = 480,
) -> pd.DataFrame:
    vol_sma = volume.rolling(window=lookback_periods, min_periods=lookback_periods // 2).mean()
    return volume / vol_sma.replace(0, np.nan)


def generate_entry_signals(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    crash_threshold_pct: float = -15.0,
    volume_spike_ratio: float = 1.3,
    lookback_periods: int = 24,
    volume_lookback_periods: int = 480,
) -> pd.DataFrame:
    crash_pct = compute_crash_signals(prices, lookback_periods, crash_threshold_pct)
    vol_ratio = compute_volume_ratio(volume, volume_lookback_periods)
    return (crash_pct <= crash_threshold_pct) & (vol_ratio >= volume_spike_ratio)


# ---------------------------------------------------------------------------
# Position simulator
# ---------------------------------------------------------------------------


def simulate_positions(
    prices: pd.DataFrame,
    entry_signals: pd.DataFrame,
    take_profit_pct: float = 12.0,
    stop_loss_pct: float = 12.0,
    max_hold_periods: int = 72,
    max_positions: int = 15,
    position_size_pct: float = 8.0,
    circuit_breaker_entries: int = 6,
    circuit_breaker_periods: int = 24,
    slippage_pct: float = 2.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tickers = prices.columns.tolist()
    dates = prices.index
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)
    positions: dict[str, dict[str, Any]] = {}
    recent_entries: list[int] = []
    trades: list[dict] = []

    for i in range(len(dates)):
        current_prices = prices.iloc[i]

        # Check exits
        to_close = []
        for ticker, pos in positions.items():
            price = current_prices.get(ticker, np.nan)
            if pd.isna(price):
                continue
            pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
            hold = i - pos["entry_idx"]
            reason = None
            if pct >= take_profit_pct:
                reason = "take_profit"
            elif pct <= -stop_loss_pct:
                reason = "stop_loss"
            elif hold >= max_hold_periods:
                reason = "time_decay"
            if reason:
                to_close.append(ticker)
                trades.append(
                    {
                        "ticker": ticker,
                        "entry_date": dates[pos["entry_idx"]],
                        "exit_date": dates[i],
                        "entry_price": pos["entry_price"],
                        "exit_price": price,
                        "pct_return": pct,
                        "hold_periods": hold,
                        "exit_reason": reason,
                    }
                )
        for t in to_close:
            del positions[t]

        # Circuit breaker
        recent_entries = [idx for idx in recent_entries if i - idx < circuit_breaker_periods]

        # New entries
        if len(recent_entries) < circuit_breaker_entries and len(positions) < max_positions:
            sigs = entry_signals.iloc[i]
            for ticker in tickers:
                if len(positions) >= max_positions:
                    break
                if ticker in positions or not sigs.get(ticker, False):
                    continue
                price = current_prices.get(ticker, np.nan)
                if pd.isna(price) or price <= 0:
                    continue
                positions[ticker] = {
                    "entry_idx": i,
                    "entry_price": price * (1 + slippage_pct / 100),
                }
                recent_entries.append(i)
                if len(recent_entries) >= circuit_breaker_entries:
                    break

        # Write weights
        for ticker in tickers:
            weights.loc[dates[i], ticker] = position_size_pct / 100 if ticker in positions else 0.0

    return weights, {"trades": trades, "n_trades": len(trades)}


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------


@dataclass
class AltcoinCrashBounceStrategy:
    """Altcoin crash bounce — mean-reversion entry on capitulation.

    Registers as ``strategy.altcoin_crash_bounce.v62`` via quantbox entry points.
    """

    meta = PluginMeta(
        name="strategy.altcoin_crash_bounce.v62",
        kind="strategy",
        version="0.6.2",
        core_compat=">=0.2.0",
        status="research",
        description="Long altcoins after crash + volume spike (capitulation). TP/SL/time exits.",
        tags=("crypto", "mean_reversion", "altcoins", "hourly"),
        params_schema={
            "type": "object",
            "properties": {
                "crash_threshold_pct": {"type": "number", "default": -15.0},
                "volume_spike_ratio": {"type": "number", "default": 1.3},
                "take_profit_pct": {"type": "number", "default": 12.0},
                "stop_loss_pct": {"type": "number", "default": 12.0},
                "max_hold_hours": {"type": "integer", "default": 72},
                "position_size_pct": {"type": "number", "default": 8.0},
                "max_positions": {"type": "integer", "default": 15},
            },
        },
        examples=(
            "plugins:\n"
            "  strategies:\n"
            "    - name: strategy.altcoin_crash_bounce.v62\n"
            "      params:\n"
            "        crash_threshold_pct: -15.0\n"
            "        take_profit_pct: 12.0",
        ),
    )

    crash_threshold_pct: float = -15.0
    volume_spike_ratio: float = 1.3
    lookback_periods: int = 24
    volume_lookback_periods: int = 480
    take_profit_pct: float = 12.0
    stop_loss_pct: float = 12.0
    max_hold_hours: int = 72
    position_size_pct: float = 8.0
    max_positions: int = 15
    circuit_breaker_entries: int = 6
    circuit_breaker_hours: int = 24
    slippage_pct: float = 2.5
    min_daily_volume_usd: float = 500_000
    exclude_tickers: list[str] = field(default_factory=lambda: DEFAULT_STABLECOINS.copy())

    @property
    def min_lookback_periods(self) -> int:
        """Minimum bars of history required before signals are meaningful.

        volume_lookback_periods drives the volume SMA warmup — the binding
        constraint.  max_hold_hours is added so the first full trade cycle
        is visible in the output.
        """
        return self.volume_lookback_periods + self.max_hold_hours

    def _filter_universe(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> list[str]:
        valid = []
        for ticker in prices.columns:
            if ticker in self.exclude_tickers:
                continue
            if ticker in volume.columns:
                avg_daily = volume[ticker].tail(168).sum() / 7
                if avg_daily < self.min_daily_volume_usd:
                    continue
            valid.append(ticker)
        return valid

    def run(
        self,
        data: dict[str, pd.DataFrame],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        prices = data["prices"]
        volume = data.get("volume", pd.DataFrame(index=prices.index))

        valid = self._filter_universe(prices, volume)
        if not valid:
            logger.warning("AltcoinCrashBounce: no valid tickers after filtering")
            return {"weights": pd.DataFrame(index=prices.index), "details": {}}

        p = prices[valid]
        v = volume[[t for t in valid if t in volume.columns]]

        signals = generate_entry_signals(
            p,
            v,
            crash_threshold_pct=self.crash_threshold_pct,
            volume_spike_ratio=self.volume_spike_ratio,
            lookback_periods=self.lookback_periods,
            volume_lookback_periods=self.volume_lookback_periods,
        )

        weights, details = simulate_positions(
            p,
            signals,
            take_profit_pct=self.take_profit_pct,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_periods=self.max_hold_hours,
            max_positions=self.max_positions,
            position_size_pct=self.position_size_pct,
            circuit_breaker_entries=self.circuit_breaker_entries,
            circuit_breaker_periods=self.circuit_breaker_hours,
            slippage_pct=self.slippage_pct,
        )

        trades = details["trades"]
        if trades:
            returns = [t["pct_return"] for t in trades]
            details["stats"] = {
                "n_trades": len(trades),
                "avg_return": float(np.mean(returns)),
                "win_rate": sum(1 for r in returns if r > 0) / len(returns),
                "exits": {
                    r: sum(1 for t in trades if t["exit_reason"] == r)
                    for r in ("take_profit", "stop_loss", "time_decay")
                },
            }

        return {"weights": weights, "details": details}
