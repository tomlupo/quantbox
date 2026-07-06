"""FLATTEN reduce-only gate against the REAL broker position schemas (issue #93).

`TradingPipeline._filter_reduce_only` decides which orders reduce exposure by
reading `broker.get_positions()`. The #92 review (SHIP-WITH-NITS) flagged that
this consumed a non-protocol `get_positions()` whose signed-qty semantics were
never confirmed against the live brokers. These tests feed the clamp the ACTUAL
DataFrames emitted by KrakenBroker (spot, long-only qty>=0) and HyperliquidBroker
(perps, signed qty) and assert the clamp reduces / drops / clamps correctly for
each sign convention.

No network, no capital: Kraken uses its injected fake exchange; Hyperliquid is
built via ``__new__`` with a fake ``fetch_positions``.
"""

from __future__ import annotations

import pandas as pd

from quantbox.plugins.broker.hyperliquid import HyperliquidBroker
from quantbox.plugins.broker.kraken import KrakenBroker
from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline

# --- Kraken fake exchange (mirrors tests/test_kraken_broker.py) -------------


class _FakeKrakenExchange:
    def __init__(self, balance):
        self._balance = balance
        self.markets = {"BTC/USD": {"spot": True, "base": "BTC", "quote": "USD"}}

    def load_markets(self):
        return self.markets

    def fetch_balance(self):
        return self._balance


def _kraken(balance):
    return KrakenBroker(quote_asset="USD", _exchange=_FakeKrakenExchange(balance))


# --- Hyperliquid fake exchange ----------------------------------------------


class _FakeHLExchange:
    def __init__(self, positions):
        self._positions = positions

    def fetch_positions(self):
        return self._positions


def _hl(positions):
    b = object.__new__(HyperliquidBroker)
    b._exchange = _FakeHLExchange(positions)
    b._base_to_coin = {}
    return b


def _hl_pos(symbol, side, contracts):
    return {
        "symbol": symbol,
        "side": side,
        "contracts": contracts,
        "notional": 1000.0,
        "entryPrice": 100.0,
        "unrealizedPnl": 0.0,
    }


def _order(asset, action, qty):
    return {"Asset": asset, "Action": action, "Adjusted Quantity": qty, "Executable": True}


def _clamp(orders, broker):
    return TradingPipeline()._filter_reduce_only(pd.DataFrame(orders), broker)


# ===========================================================================
# Kraken spot: long-only (qty >= 0). A SELL reduces; a BUY increases.
# ===========================================================================


def test_kraken_sell_reduces_long_and_is_kept():
    b = _kraken({"total": {"ZUSD": 1000.0, "XXBT": 0.5}})  # long 0.5 BTC
    out = _clamp([_order("BTC", "Sell", 0.3)], b)
    assert len(out) == 1
    assert out.iloc[0]["Adjusted Quantity"] == 0.3


def test_kraken_sell_is_clamped_to_holding():
    b = _kraken({"total": {"ZUSD": 1000.0, "XXBT": 0.5}})
    out = _clamp([_order("BTC", "Sell", 2.0)], b)  # oversized sell
    assert len(out) == 1
    assert out.iloc[0]["Adjusted Quantity"] == 0.5  # clamped to the 0.5 held


def test_kraken_buy_increases_long_and_is_dropped():
    b = _kraken({"total": {"ZUSD": 1000.0, "XXBT": 0.5}})
    out = _clamp([_order("BTC", "Buy", 0.3)], b)
    assert out.empty


def test_kraken_order_for_unheld_symbol_is_dropped():
    b = _kraken({"total": {"ZUSD": 1000.0}})  # no positions
    out = _clamp([_order("BTC", "Sell", 0.3)], b)
    assert out.empty


# ===========================================================================
# Hyperliquid perps: signed qty. A BUY reduces a SHORT; a SELL reduces a LONG.
# ===========================================================================


def test_hl_buy_reduces_short_and_is_kept():
    b = _hl([_hl_pos("ETH/USDC:USDC", "short", 3.0)])  # qty = -3.0
    out = _clamp([_order("ETH", "Buy", 1.0)], b)
    assert len(out) == 1
    assert out.iloc[0]["Adjusted Quantity"] == 1.0


def test_hl_buy_is_clamped_to_short_magnitude():
    b = _hl([_hl_pos("ETH/USDC:USDC", "short", 3.0)])
    out = _clamp([_order("ETH", "Buy", 10.0)], b)  # oversized cover
    assert len(out) == 1
    assert out.iloc[0]["Adjusted Quantity"] == 3.0  # clamped to |qty|, never flips long


def test_hl_sell_increases_short_and_is_dropped():
    b = _hl([_hl_pos("ETH/USDC:USDC", "short", 3.0)])
    out = _clamp([_order("ETH", "Sell", 1.0)], b)  # would deepen the short
    assert out.empty


def test_hl_sell_reduces_long_and_is_kept():
    b = _hl([_hl_pos("BTC/USDC:USDC", "long", 2.0)])  # qty = +2.0
    out = _clamp([_order("BTC", "Sell", 0.5)], b)
    assert len(out) == 1
    assert out.iloc[0]["Adjusted Quantity"] == 0.5


def test_hl_buy_increases_long_and_is_dropped():
    b = _hl([_hl_pos("BTC/USDC:USDC", "long", 2.0)])
    out = _clamp([_order("BTC", "Buy", 0.5)], b)
    assert out.empty


# ===========================================================================
# Fail-safe: no readable positions → drop everything (send nothing).
# ===========================================================================


def test_fail_safe_drops_all_when_broker_has_no_get_positions():
    class _NoPositions:
        pass

    out = _clamp([_order("BTC", "Sell", 0.3)], _NoPositions())
    assert out.empty
