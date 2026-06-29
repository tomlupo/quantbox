"""Opt-in venue-liquidity Stage-2 screen (screen_volume_source).

Proves a single-venue execution book (e.g. live Kraken-USD spot) can rank Stage-2
on per-venue dollar volume, while the default — and any non-opted config — keeps
the market-wide aggregate screen. ccxt is never touched (fetcher stubbed).
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantbox.plugins.datasources._utils import resolve_screen_inputs
from quantbox.plugins.datasources.kraken_data_plugin import KrakenDataPlugin
from quantbox.plugins.strategies._universe import select_universe

_IDX = pd.date_range("2026-01-01", periods=3, freq="D")
_PRICES = pd.DataFrame({"BTC": [1.0, 1.1, 1.2], "ETH": [2.0, 2.1, 2.2]}, index=_IDX)
_VOLUME = pd.DataFrame({"BTC": [10.0, 11.0, 12.0], "ETH": [5.0, 5.5, 6.0]}, index=_IDX)


class _FakeProvider:
    """Stand-in MarketCapProvider with deterministic, distinguishable outputs."""

    def estimate_market_cap(self, prices, volume):
        return pd.DataFrame(1e9, index=prices.index, columns=prices.columns)

    def estimate_aggregate_volume(self, prices):
        # market-wide aggregate volume — must only appear when source == "market"
        return pd.DataFrame(7.7e8, index=prices.index, columns=prices.columns)


def test_default_market_source_keeps_marketwide_screen():
    mcap, sv = resolve_screen_inputs("live", _PRICES, _VOLUME, _FakeProvider())
    assert not mcap.empty  # Stage-1 mcap present
    assert not sv.empty  # Stage-2 = market-wide aggregate
    assert (sv == 7.7e8).all().all()


def test_venue_source_empties_screen_volume_for_per_venue_rank():
    mcap, sv = resolve_screen_inputs("live", _PRICES, _VOLUME, _FakeProvider(), screen_volume_source="venue")
    # Stage-1 market cap UNAFFECTED — still the genuine cross-venue snapshot
    assert not mcap.empty
    assert (mcap == 1e9).all().all()
    # Stage-2: empty screen_volume => select_universe falls back to per-venue volume
    assert sv.empty


def test_default_unchanged_vs_venue_differ():
    _, sv_market = resolve_screen_inputs("live", _PRICES, _VOLUME, _FakeProvider())
    _, sv_venue = resolve_screen_inputs("live", _PRICES, _VOLUME, _FakeProvider(), screen_volume_source="venue")
    assert not sv_market.empty and sv_venue.empty


class _StubFetcher:
    """Captures the screen_volume_source the plugin forwards to the fetcher."""

    def __init__(self):
        self.seen = None

    def get_market_data(self, *, screen_volume_source="market", **kw):
        self.seen = screen_volume_source
        return {
            "prices": _PRICES,
            "volume": _VOLUME,
            "market_cap": pd.DataFrame(),
            "screen_volume": pd.DataFrame(),
        }


def _run(plugin):
    stub = _StubFetcher()
    plugin._fetcher = stub
    plugin.load_market_data(pd.DataFrame({"symbol": ["BTC", "ETH"]}), "2026-01-03", {"mode": "live"})
    return stub.seen


def test_kraken_book_forwards_venue_when_opted_in():
    assert _run(KrakenDataPlugin(screen_volume_source="venue")) == "venue"


def test_non_opted_kraken_book_stays_market_wide():
    # default config (no screen_volume_source) must keep market-wide behaviour
    assert _run(KrakenDataPlugin()) == "market"


@pytest.mark.parametrize("bad", ["veneu", "Venue ", "kraken", "", None])
def test_unknown_source_fails_loud_not_silent_market(bad):
    # a typo must raise, never silently degrade a live venue book to market-wide
    with pytest.raises(ValueError, match="screen_volume_source"):
        resolve_screen_inputs("live", _PRICES, _VOLUME, _FakeProvider(), screen_volume_source=bad)


def test_select_universe_picks_different_names_market_vs_venue():
    """End-to-end proof: the two rankers select different Stage-2 tickers.

    A is liquid on-venue; B is liquid market-wide. All three pass the mcap tier.
    """
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    cols = ["A", "B", "C"]
    prices = pd.DataFrame(1.0, index=idx, columns=cols)
    mcap = pd.DataFrame({"A": 3e9, "B": 2e9, "C": 1e9}, index=idx)  # all top-3
    venue_vol = pd.DataFrame({"A": 100.0, "B": 1.0, "C": 1.0}, index=idx)  # A wins on-venue
    market_vol = pd.DataFrame({"A": 1.0, "B": 100.0, "C": 1.0}, index=idx)  # B wins market-wide

    def picked(mask):
        return set(mask.columns[mask.iloc[-1].astype(bool)])

    venue_sel = select_universe(
        prices,
        venue_vol,
        market_cap=mcap,
        top_by_mcap=3,
        top_by_volume=1,
        volume_is_dollar=True,
        screen_volume=None,  # empty -> per-venue
    )
    market_sel = select_universe(
        prices,
        venue_vol,
        market_cap=mcap,
        top_by_mcap=3,
        top_by_volume=1,
        volume_is_dollar=True,
        screen_volume=market_vol,  # populated -> market-wide
    )
    assert picked(venue_sel) == {"A"}
    assert picked(market_sel) == {"B"}
