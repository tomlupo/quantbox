"""Regression tests: the market-wide universe screen must not look ahead in backtest.

Background
----------
The two-stage universe screen (``select_universe``) ranks coins by market cap
(Stage 1) and by volume (Stage 2). The live Hyperliquid / Binance data plugins
populate those screen inputs (``market_cap`` / ``screen_volume``). For LIVE
trading they use a CoinGecko *snapshot* — today's market cap + today's
market-wide aggregate volume — which is correct, because "today" is the point of
decision.

In a BACKTEST that same snapshot, broadcast onto every historical row, is
look-ahead + survivorship bias: a coin's PAST universe membership gets decided
by its PRESENT size / liquidity. ``resolve_screen_inputs`` fixes this by routing
on the run mode: snapshot only in live/paper, point-in-time sources in backtest.

These tests pin three things:
1. mode routing — backtest never touches the snapshot provider;
2. the look-ahead *guard* actually bites — a snapshot-style screen DOES change
   past membership when the series is truncated;
3. the fixed backtest screen is truncation-invariant — membership up to date *t*
   is unchanged when the input history is truncated at *t* (the definition of
   point-in-time / no look-ahead).
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantbox.plugins.datasources._utils import load_pit_market_cap, resolve_screen_inputs
from quantbox.plugins.strategies._universe import select_universe


class RecordingSnapshotProvider:
    """Test double for ``MarketCapProvider`` that reproduces the look-ahead bug.

    Mirrors ``estimate_market_cap`` / ``estimate_aggregate_volume``: it broadcasts
    a single *latest-observation* snapshot flat across the whole history. Because
    the snapshot is the LAST row of the data it is handed, truncating that data
    changes the snapshot — which is exactly the look-ahead this guards against.
    Also records whether it was called, so we can assert the backtest path never
    invokes it.
    """

    def __init__(self, source_volume: pd.DataFrame):
        self._vol = source_volume  # "today" = the last row of this series
        self.mc_calls = 0
        self.sv_calls = 0

    def estimate_market_cap(self, prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        self.mc_calls += 1
        last = prices.iloc[-1]  # price-path anchor at the latest row (as in L1)
        return pd.DataFrame({c: [float(last[c])] * len(prices) for c in prices.columns}, index=prices.index)

    def estimate_aggregate_volume(self, prices: pd.DataFrame) -> pd.DataFrame:
        self.sv_calls += 1
        last = self._vol.reindex(prices.index).iloc[-1]  # today's market-wide volume, broadcast flat
        return pd.DataFrame({c: [float(last.get(c, 0.0))] * len(prices) for c in prices.columns}, index=prices.index)


def _rotating_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """20 days, 4 coins. Per-venue volume leadership rotates A,B -> C,D at the midpoint.

    Point-in-time, the top-2-by-volume universe is {A,B} early and {C,D} late. A
    snapshot that broadcasts the *final* day's volume would instead force {C,D}
    across the whole history — the look-ahead.
    """
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    prices = pd.DataFrame(100.0, index=dates, columns=["A", "B", "C", "D"])
    early = [100.0] * 10 + [1.0] * 10
    late = [1.0] * 10 + [100.0] * 10
    volume = pd.DataFrame(
        {
            "A": [v + 1 for v in early],  # A slightly > B early
            "B": early,
            "C": [v + 1 for v in late],  # C slightly > D late
            "D": late,
        },
        index=dates,
    )
    return prices, volume


def _membership(prices, volume, mode, provider=None, top_by_volume=2):
    """Resolve screen inputs for *mode* then run the universe screen."""
    mc, sv = resolve_screen_inputs(mode, prices, volume, provider)
    return select_universe(
        prices,
        volume,
        market_cap=mc if not mc.empty else None,
        top_by_mcap=30,
        top_by_volume=top_by_volume,
        exclude_tickers=[],
        volume_is_dollar=True,
        screen_volume=sv if not sv.empty else None,
    )


# ---------------------------------------------------------------------------
# 1. Mode routing
# ---------------------------------------------------------------------------


def test_backtest_mode_never_uses_the_snapshot_provider():
    prices, volume = _rotating_data()
    provider = RecordingSnapshotProvider(volume)
    market_cap, screen_volume = resolve_screen_inputs("backtest", prices, volume, provider)
    assert provider.mc_calls == 0 and provider.sv_calls == 0
    # screen_volume must be empty so Stage 2 ranks on point-in-time per-venue volume
    assert screen_volume.empty
    # market_cap comes from the curated PIT dataset (empty here unless installed),
    # never from the snapshot provider.
    assert provider.mc_calls == 0


def test_unset_mode_defaults_to_backtest_path():
    prices, volume = _rotating_data()
    provider = RecordingSnapshotProvider(volume)
    _, screen_volume = resolve_screen_inputs(None, prices, volume, provider)
    assert provider.mc_calls == 0 and provider.sv_calls == 0
    assert screen_volume.empty


@pytest.mark.parametrize("mode", ["live", "paper"])
def test_live_and_paper_use_the_snapshot_provider(mode):
    prices, volume = _rotating_data()
    provider = RecordingSnapshotProvider(volume)
    market_cap, screen_volume = resolve_screen_inputs(mode, prices, volume, provider)
    assert provider.mc_calls == 1 and provider.sv_calls == 1
    assert not market_cap.empty and not screen_volume.empty


# ---------------------------------------------------------------------------
# 2. The guard bites — a snapshot screen DOES look ahead
# ---------------------------------------------------------------------------


def test_snapshot_screen_is_truncation_variant():
    """Sanity that the truncation test below is non-vacuous: the live/snapshot
    path changes past membership when the history is truncated (look-ahead)."""
    prices, volume = _rotating_data()
    full = _membership(prices, volume, "live", RecordingSnapshotProvider(volume))
    cut = prices.index[5]
    trunc = _membership(prices.loc[:cut], volume.loc[:cut], "live", RecordingSnapshotProvider(volume.loc[:cut]))
    # Full history (snapshot = day-19 volume) marks {C,D} even on day 5; the
    # truncated run (snapshot = day-5 volume) marks {A,B}. They MUST differ —
    # that difference IS the look-ahead.
    assert not full.loc[:cut].equals(trunc)


# ---------------------------------------------------------------------------
# 3. The fix — backtest screen is point-in-time (truncation-invariant)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut", [4, 9, 10, 15, 18])
def test_backtest_universe_membership_is_truncation_invariant(cut):
    """Truncating the input at date t must not change membership up to t."""
    prices, volume = _rotating_data()
    full = _membership(prices, volume, "backtest")
    t = prices.index[cut]
    trunc = _membership(prices.loc[:t], volume.loc[:t], "backtest")
    pd.testing.assert_frame_equal(full.loc[:t], trunc, check_freq=False)


def test_backtest_universe_actually_rotates_point_in_time():
    """Guards against a vacuous invariance (e.g. an all-zero mask): the PIT
    universe really is {A,B} early and {C,D} late."""
    prices, volume = _rotating_data()
    full = _membership(prices, volume, "backtest")
    early = full.iloc[5]
    late = full.iloc[15]
    assert early["A"] == 1.0 and early["B"] == 1.0 and early["C"] == 0.0 and early["D"] == 0.0
    assert late["C"] == 1.0 and late["D"] == 1.0 and late["A"] == 0.0 and late["B"] == 0.0


# ---------------------------------------------------------------------------
# 4. load_pit_market_cap is causal when the curated dataset is available
# ---------------------------------------------------------------------------


def test_load_pit_market_cap_is_empty_or_causal():
    """If quantbox-datasets is not installed, PIT market cap is empty (callers
    fall back to per-venue volume). If it IS installed, the series must be causal:
    truncating the price index must not change earlier rows."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    prices = pd.DataFrame({"BTC": range(100, 130), "ETH": range(50, 80)}, index=dates, dtype=float)
    full = load_pit_market_cap(prices)
    if full.empty:
        pytest.skip("quantbox-datasets not installed; PIT market cap unavailable (clean fallback)")
    t = dates[20]
    trunc = load_pit_market_cap(prices.loc[:t])
    common = [c for c in full.columns if c in trunc.columns]
    pd.testing.assert_frame_equal(full.loc[:t, common], trunc[common], check_freq=False)
