"""Regression test for quantbox#120: futures_rebalancer freezes silently when
a held position rotates out of the fetched universe.

Root cause: the strategy/rebalance price-fetch (``load_market_data``) covered
only the NEW screened universe, not ``current_holdings ∪ target_positions``.
A currently-HELD symbol that rotated OUT of today's universe therefore had no
price -> the paper/live broker's market snapshot came back NaN for it -> the
rebalancer flagged the (correct) exit order as ``Invalid (NaN)`` and skipped
it -> if enough held symbols rotated out this way, EVERY order was suppressed
and the book froze on stale positions.

Fix: ``_expand_price_fetch_universe`` unions today's screened universe with
current broker holdings before the price fetch (mirrors the holdings ∪
universe pattern quantbox-live's ``dump_kraken_prices.py`` already uses) —
without touching the strategy-facing ``universe`` used for eligibility/weights.
"""

from __future__ import annotations

import pandas as pd

from quantbox.plugins.pipeline.trading_pipeline import _expand_price_fetch_universe


def _screened_universe(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"symbol": symbols, "rank": range(len(symbols))})


class _BrokerWithPositions:
    def __init__(self, held_symbols: list[str]):
        self._held = held_symbols

    def get_positions(self) -> pd.DataFrame:
        return pd.DataFrame({"symbol": self._held, "qty": [1.0] * len(self._held)})


class _BrokerNoPositions:
    """A broker exposing get_positions() but with nothing on the books."""

    def get_positions(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol", "qty"])


class _BrokerRaises:
    def get_positions(self) -> pd.DataFrame:
        raise RuntimeError("exchange unreachable")


def test_held_symbol_rotated_out_of_universe_is_added_for_price_fetch():
    """The exact live trap: ARB/BNB/PEPE/TRX/UNI held, but today's screen only
    returned 13 other tickers -- the held symbols must still be fetched."""
    universe = _screened_universe(["BTC", "ETH", "SOL"])
    broker = _BrokerWithPositions(["ARB", "BNB", "PEPE", "TRX", "UNI"])

    fetch_universe = _expand_price_fetch_universe(universe, broker, mode="paper")

    assert set(fetch_universe["symbol"]) == {"BTC", "ETH", "SOL", "ARB", "BNB", "PEPE", "TRX", "UNI"}
    # The original screened universe is untouched (no mutation, no dropped rows).
    assert list(universe["symbol"]) == ["BTC", "ETH", "SOL"]


def test_held_symbol_already_in_universe_is_not_duplicated():
    universe = _screened_universe(["BTC", "ETH"])
    broker = _BrokerWithPositions(["ETH"])

    fetch_universe = _expand_price_fetch_universe(universe, broker, mode="live")

    assert sorted(fetch_universe["symbol"]) == ["BTC", "ETH"]
    assert len(fetch_universe) == 2


def test_flat_book_returns_universe_unchanged():
    universe = _screened_universe(["BTC", "ETH"])
    broker = _BrokerNoPositions()

    fetch_universe = _expand_price_fetch_universe(universe, broker, mode="paper")

    assert fetch_universe is universe


def test_backtest_mode_is_not_expanded():
    """Backtest has no live broker holdings to protect; must be a no-op even
    if a broker instance is (unusually) passed."""
    universe = _screened_universe(["BTC"])
    broker = _BrokerWithPositions(["ARB"])

    fetch_universe = _expand_price_fetch_universe(universe, broker, mode="backtest")

    assert fetch_universe is universe


def test_no_broker_is_not_expanded():
    universe = _screened_universe(["BTC"])

    fetch_universe = _expand_price_fetch_universe(universe, None, mode="live")

    assert fetch_universe is universe


def test_broker_without_get_positions_is_not_expanded():
    class _NoPositionsMethod:
        pass

    universe = _screened_universe(["BTC"])
    fetch_universe = _expand_price_fetch_universe(universe, _NoPositionsMethod(), mode="live")

    assert fetch_universe is universe


def test_broker_get_positions_raising_fails_safe_to_unchanged_universe():
    universe = _screened_universe(["BTC"])
    broker = _BrokerRaises()

    fetch_universe = _expand_price_fetch_universe(universe, broker, mode="live")

    assert fetch_universe is universe


def test_extra_universe_columns_are_na_filled_not_dropped():
    """Non-symbol screening columns (e.g. mcap rank) must not crash the union;
    the held-but-rotated-out row gets NA for them, not a KeyError/shape error."""
    universe = _screened_universe(["BTC", "ETH"])
    broker = _BrokerWithPositions(["ARB"])

    fetch_universe = _expand_price_fetch_universe(universe, broker, mode="paper")

    assert list(fetch_universe.columns) == ["symbol", "rank"]
    arb_row = fetch_universe[fetch_universe["symbol"] == "ARB"].iloc[0]
    assert pd.isna(arb_row["rank"])
