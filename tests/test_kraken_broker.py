"""Tests for the Kraken spot broker.

ccxt is mocked via an injected fake exchange — these tests NEVER hit live
Kraken and touch no real capital.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantbox.plugins.broker.kraken import KrakenBroker


class _FakeExchange:
    """Minimal ccxt.kraken stand-in."""

    def __init__(self, balance=None):
        self._balance = balance or {"total": {}}
        self.created_orders: list[dict] = []
        self.markets = {
            "BTC/USD": {
                "spot": True,
                "base": "BTC",
                "quote": "USD",
                "precision": {"amount": 3},
                "limits": {"amount": {"min": 0.0001}, "cost": {"min": 5.0}},
            },
            "ETH/USD": {
                "spot": True,
                "base": "ETH",
                "quote": "USD",
                "precision": {"amount": 2},
                "limits": {"amount": {"min": 0.01}, "cost": {"min": 5.0}},
            },
        }

    def load_markets(self):
        return self.markets

    def fetch_balance(self):
        return self._balance

    def fetch_ticker(self, symbol):
        return {"last": 60000.0 if symbol == "BTC/USD" else 3000.0}

    def create_order(self, symbol, type, side, amount, price=None):
        order = {
            "id": f"oid-{len(self.created_orders)}",
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "price": price,
            "average": price or (60000.0 if symbol == "BTC/USD" else 3000.0),
            "filled": amount,
        }
        self.created_orders.append(order)
        return order

    def fetch_my_trades(self, symbol=None, since=None):
        return [
            {
                "symbol": "BTC/USD",
                "side": "buy",
                "amount": 0.01,
                "price": 60000.0,
                "datetime": "2026-06-01T00:00:00Z",
            }
        ]


def _broker(balance=None, readonly=False):
    return KrakenBroker(quote_asset="USD", readonly=readonly, _exchange=_FakeExchange(balance))


# ---------------------------------------------------------------------------
# Balances / cash / positions
# ---------------------------------------------------------------------------


def test_get_cash_normalizes_legacy_quote_code():
    # Balance keyed by Kraken legacy code ZUSD must map to USD.
    b = _broker({"total": {"ZUSD": 1000.0, "XXBT": 0.5}})
    assert b.get_cash() == {"USD": 1000.0}


def test_get_positions_excludes_quote_and_normalizes():
    b = _broker({"total": {"ZUSD": 1000.0, "XXBT": 0.5, "ETH": 2.0}})
    pos = b.get_positions()
    syms = dict(zip(pos["symbol"], pos["qty"], strict=False))
    assert syms == {"BTC": 0.5, "ETH": 2.0}
    assert "USD" not in pos["symbol"].tolist()


def test_get_positions_filters_earn_balances():
    # .S / .F staking balances must NOT count as sellable spot positions.
    b = _broker({"total": {"DOT.S": 100.0, "ETH.F": 5.0, "ETH": 2.0, "ZUSD": 50.0}})
    pos = b.get_positions()
    syms = dict(zip(pos["symbol"], pos["qty"], strict=False))
    assert syms == {"ETH": 2.0}  # earn balances dropped, only spot ETH remains


def test_get_positions_empty():
    b = _broker({"total": {"ZUSD": 100.0}})
    pos = b.get_positions()
    assert pos.empty
    assert list(pos.columns) == ["symbol", "qty"]


# ---------------------------------------------------------------------------
# Market snapshot
# ---------------------------------------------------------------------------


def test_market_snapshot():
    b = _broker()
    snap = b.get_market_snapshot(["BTC", "ETH"])
    row = snap.set_index("symbol").loc["BTC"]
    assert row["mid"] == 60000.0
    assert row["min_qty"] == 0.0001
    assert row["step_size"] == pytest.approx(0.001)  # 3 decimals -> 1e-3
    assert row["min_notional"] == 5.0


def test_market_snapshot_unknown_symbol():
    b = _broker()
    snap = b.get_market_snapshot(["NOPE"])
    assert snap.iloc[0]["mid"] is None


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


def test_place_market_order():
    b = _broker()
    orders = pd.DataFrame([{"symbol": "BTC", "side": "buy", "qty": 0.01}])
    fills = b.place_orders(orders)
    assert fills.iloc[0]["status"] == "FILLED"
    assert fills.iloc[0]["price"] == 60000.0
    assert b._exchange.created_orders[0]["type"] == "market"
    assert b._exchange.created_orders[0]["side"] == "buy"


def test_place_limit_order_when_price_given():
    b = _broker()
    orders = pd.DataFrame([{"symbol": "ETH", "side": "sell", "qty": 1.0, "price": 3100.0}])
    fills = b.place_orders(orders)
    assert fills.iloc[0]["status"] == "FILLED"
    assert b._exchange.created_orders[0]["type"] == "limit"
    assert b._exchange.created_orders[0]["price"] == 3100.0


def test_readonly_blocks_orders():
    b = _broker(readonly=True)
    with pytest.raises(PermissionError):
        b.place_orders(pd.DataFrame([{"symbol": "BTC", "side": "buy", "qty": 0.01}]))


def test_below_min_qty_skips_not_fails():
    # A sub-minimum / sub-precision order is a clean SKIP, never a FAILURE — it
    # placed no order and moved no capital, so it must not inflate total_failed.
    b = _broker()
    orders = pd.DataFrame([{"symbol": "BTC", "side": "buy", "qty": 0.00001}])  # < min 0.0001
    fills = b.place_orders(orders)
    assert fills.iloc[0]["status"] == "SKIPPED"
    assert b._exchange.created_orders == []


def test_skip_does_not_block_other_orders():
    # A sub-min dust order must NOT abort the rest of the batch: the legitimate
    # BTC buy still fills even though the dust sell is skipped.
    b = _broker()
    orders = pd.DataFrame(
        [
            {"symbol": "BTC", "side": "sell", "qty": 0.00001},  # dust -> SKIPPED
            {"symbol": "BTC", "side": "buy", "qty": 0.01},  # real -> FILLED
        ]
    )
    fills = b.place_orders(orders)
    statuses = fills["status"].tolist()
    assert statuses == ["SKIPPED", "FILLED"]
    assert len(b._exchange.created_orders) == 1  # only the real buy reached Kraken


def test_negative_qty_rejected():
    b = _broker()
    fills = b.place_orders(pd.DataFrame([{"symbol": "BTC", "side": "buy", "qty": -1.0}]))
    assert fills.iloc[0]["status"] == "FAILED"


def test_get_positions_excludes_stablecoin_dust():
    # A USDC residue alongside a USD quote is cash-equivalent dust, NOT a
    # liquidatable position — it must never appear in get_positions (else the
    # rebalancer emits a guaranteed sub-minimum sell). Real positions remain.
    b = _broker({"total": {"ZUSD": 278.0, "USDC": 0.001442, "ETH": 2.0}})
    pos = b.get_positions()
    syms = dict(zip(pos["symbol"], pos["qty"], strict=False))
    assert syms == {"ETH": 2.0}
    assert "USDC" not in pos["symbol"].tolist()


def test_unknown_market_fails():
    b = _broker()
    fills = b.place_orders(pd.DataFrame([{"symbol": "DOGE", "side": "buy", "qty": 100.0}]))
    assert fills.iloc[0]["status"] == "FAILED"


def test_empty_orders_returns_empty():
    b = _broker()
    fills = b.place_orders(pd.DataFrame(columns=["symbol", "side", "qty"]))
    assert fills.empty


# ---------------------------------------------------------------------------
# Fills
# ---------------------------------------------------------------------------


def test_fetch_fills_normalizes_symbol():
    b = _broker()
    fills = b.fetch_fills("2026-06-01")
    assert fills.iloc[0]["symbol"] == "BTC"
    assert fills.iloc[0]["side"] == "buy"
    assert fills.iloc[0]["qty"] == 0.01
