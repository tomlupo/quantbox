"""Fill-confirmation hardening for live ccxt brokers (issue #68).

Both KrakenBroker.place_orders and HyperliquidBroker.place_orders used to report
EVERY accepted order as ``status=FILLED`` and fall back to the requested qty when
ccxt returned ``filled=0/absent``. That can make the book believe a live order
filled when it is actually still open / partial / rejected — a silent wrong-state
on real capital. These tests pin the honest classification: an unconfirmed or
failed placement must NOT report FILLED.
"""

from __future__ import annotations

import pandas as pd

from quantbox.plugins.broker import _fills
from quantbox.plugins.broker._fills import classify_fill, resolve_fill

# ---------------------------------------------------------------------------
# Pure classifier
# ---------------------------------------------------------------------------


def test_closed_full_fill_is_filled():
    verdict, qty, price = classify_fill(
        {"status": "closed", "filled": 2.0, "average": 100.0}, 2.0
    )
    assert verdict == _fills.FILL_FILLED
    assert qty == 2.0
    assert price == 100.0


def test_closed_without_filled_field_assumes_full_fill():
    # A 'closed' market order that omits ``filled`` == fully filled: fall back to
    # the requested qty (this is the ONLY case where requested==filled is assumed).
    verdict, qty, _ = classify_fill({"status": "closed", "average": 5.0}, 3.0)
    assert verdict == _fills.FILL_FILLED
    assert qty == 3.0


def test_closed_but_zero_filled_is_not_a_fill():
    # Contradictory (closed yet filled==0): never claim a fill.
    verdict, qty, _ = classify_fill({"status": "closed", "filled": 0.0}, 3.0)
    assert verdict == _fills.FILL_UNFILLED
    assert qty == 0.0


def test_open_with_partial_fill_is_partial():
    verdict, qty, _ = classify_fill({"status": "open", "filled": 0.4}, 1.0)
    assert verdict == _fills.FILL_PARTIAL
    assert qty == 0.4


def test_open_with_zero_fill_is_unfilled():
    verdict, qty, _ = classify_fill({"status": "open", "filled": 0.0}, 1.0)
    assert verdict == _fills.FILL_UNFILLED


def test_rejected_and_canceled_are_unfilled():
    for status in ("rejected", "canceled", "cancelled", "expired"):
        verdict, qty, _ = classify_fill({"status": status}, 1.0)
        assert verdict == _fills.FILL_UNFILLED, status
        assert qty == 0.0


def test_rejected_with_partial_fill_is_partial():
    verdict, qty, _ = classify_fill({"status": "canceled", "filled": 0.25}, 1.0)
    assert verdict == _fills.FILL_PARTIAL
    assert qty == 0.25


def test_missing_status_with_zero_filled_is_unfilled():
    # The exact pre-fix landmine: ccxt returns filled=0 and the old code assumed
    # FILLED with the requested qty. Now it is honestly UNFILLED.
    verdict, qty, _ = classify_fill({"filled": 0.0}, 1.0)
    assert verdict == _fills.FILL_UNFILLED
    assert qty == 0.0


def test_missing_status_with_partial_remaining_is_partial():
    verdict, qty, _ = classify_fill({"filled": 0.6, "remaining": 0.4}, 1.0)
    assert verdict == _fills.FILL_PARTIAL
    assert qty == 0.6


def test_no_status_no_filled_is_unknown():
    verdict, qty, _ = classify_fill({"id": "abc"}, 1.0)
    assert verdict == _fills.FILL_UNKNOWN


def test_empty_order_is_unknown():
    assert classify_fill(None, 1.0)[0] == _fills.FILL_UNKNOWN
    assert classify_fill({}, 1.0)[0] == _fills.FILL_UNKNOWN


# ---------------------------------------------------------------------------
# resolve_fill: emitted (status, qty, price, reason) + one follow-up fetch
# ---------------------------------------------------------------------------


def test_resolve_unknown_confirmed_by_refetch():
    status, qty, _, reason = resolve_fill(
        {"id": "x"}, 1.0, refetch=lambda: {"status": "closed", "filled": 1.0}
    )
    assert status == "FILLED"
    assert qty == 1.0
    assert reason == ""


def test_resolve_unknown_unconfirmed_fails_safe_to_failed():
    # Refetch still can't confirm -> NEVER claim a fill.
    status, qty, _, reason = resolve_fill({"id": "x"}, 1.0, refetch=lambda: {"id": "x"})
    assert status == "FAILED"
    assert qty == 0.0
    assert "not confirmed" in reason


def test_resolve_unknown_refetch_raises_fails_safe():
    def _boom():
        raise RuntimeError("429 rate limited")

    status, qty, _, _ = resolve_fill({"id": "x"}, 1.0, refetch=_boom)
    assert status == "FAILED"
    assert qty == 0.0


def test_resolve_open_order_is_failed_not_filled():
    status, qty, _, _ = resolve_fill({"status": "open", "filled": 0.0}, 1.0)
    assert status == "FAILED"
    assert qty == 0.0


def test_resolve_partial_reports_real_qty():
    status, qty, _, reason = resolve_fill({"status": "open", "filled": 0.3}, 1.0)
    assert status == "PARTIAL"
    assert qty == 0.3
    assert "partial" in reason.lower()


# ---------------------------------------------------------------------------
# KrakenBroker.place_orders end-to-end (no network: fake ccxt exchange)
# ---------------------------------------------------------------------------


class _FakeKrakenExchange:
    """Minimal ccxt.kraken stand-in; create_order returns a scripted result."""

    def __init__(self, order_result):
        self._order_result = order_result
        self.markets = {
            "BTC/USD": {
                "spot": True,
                "base": "BTC",
                "quote": "USD",
                "precision": {"amount": 8},
                "limits": {"amount": {"min": 0.0001}, "cost": {"min": 1.0}},
            }
        }

    def load_markets(self):
        return self.markets

    def market(self, symbol):
        return self.markets[symbol]

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.8f}"

    def create_order(self, **kwargs):
        return self._order_result

    def fetch_order(self, order_id, symbol):
        # Default: still unconfirmable (no status / no filled).
        return {"id": order_id}


def _kraken_with(order_result):
    from quantbox.plugins.broker.kraken import KrakenBroker

    broker = KrakenBroker.__new__(KrakenBroker)
    broker._exchange = _FakeKrakenExchange(order_result)
    broker._markets = broker._exchange.markets
    broker.quote_asset = "USD"
    broker.readonly = False
    return broker


def _one_buy():
    return pd.DataFrame([{"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 0.0}])


def test_kraken_confirmed_fill_reports_filled():
    broker = _kraken_with({"id": "1", "status": "closed", "filled": 1.0, "average": 60000.0})
    out = broker.place_orders(_one_buy())
    row = out.iloc[0]
    assert row["status"] == "FILLED"
    assert row["qty"] == 1.0
    assert row["price"] == 60000.0


def test_kraken_unconfirmed_placement_not_reported_filled():
    # ccxt returns an accepted order with NO fill evidence; the follow-up
    # fetch_order also can't confirm -> must be FAILED, never FILLED.
    broker = _kraken_with({"id": "2"})
    out = broker.place_orders(_one_buy())
    row = out.iloc[0]
    assert row["status"] == "FAILED"
    assert row["qty"] == 0.0


def test_kraken_partial_fill_reported_honestly():
    broker = _kraken_with({"id": "3", "status": "open", "filled": 0.4, "average": 60000.0})
    out = broker.place_orders(_one_buy())
    row = out.iloc[0]
    assert row["status"] == "PARTIAL"
    assert row["qty"] == 0.4


def test_kraken_rejected_order_not_reported_filled():
    broker = _kraken_with({"id": "4", "status": "rejected"})
    out = broker.place_orders(_one_buy())
    row = out.iloc[0]
    assert row["status"] == "FAILED"
    assert row["qty"] == 0.0
