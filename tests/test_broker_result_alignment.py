"""A broker result is never booked to another order (PR #310 review, TOM-2396).

The executor binds the k-th broker result row of a (symbol, side) to the k-th
order of that key it submitted. A flip sends two orders of one key: a
reduce-only close, then an open. The review of PR #310 reproduced the break:
Binance futures returned NO row for a rejected order, so the open's fill was
booked to the close (close target 0.02 -> filled 0.03, open -> timeout,
total_failed 0).

Two rules close the class:

1. Every broker returns one result row per submitted order, a rejection or a
   skip included. Binance futures (a rejected order) and futures paper (an
   invalid price) did not.
2. The executor fails closed when a key's result rows do not number its
   orders: each order of the key books FAILED with an UNKNOWN fill, its ledger
   intent closes ``failed``, and no fill is booked to any of them.

Every test here is red at 9c6cd84 and green after the fix.
"""

from __future__ import annotations

import pandas as pd

from quantbox.plugins.rebalancing.futures_rebalancer import FuturesRebalancer
from quantbox.trade.executor import execute_orders
from quantbox.trade.lots import LotRules
from quantbox.trade.reconciliation.ledger import OrderFillLedger

PRICE = 2000.0


def _flip_orders(venue: str = "binance_futures") -> pd.DataFrame:
    """ETH long 0.02 to short 0.03 on a $100 book: a 0.02 close leg, then a 0.03 open leg, both sells."""
    nav = 100.0
    cur, tgt = 0.02, -0.03
    cw, tw = cur * PRICE / nav, tgt * PRICE / nav
    row = {
        "Asset": "ETH",
        "Current Quantity": cur,
        "Current Value": cur * PRICE,
        "Current Weight": cw,
        "Target Quantity": tgt,
        "Target Value": tgt * PRICE,
        "Target Weight": tw,
        "Weight Delta": tw - cw,
        "Delta Quantity": tgt - cur,
        "Price": PRICE,
        "Trade Action": "Sell",
    }
    rules = LotRules.from_snapshot({"step_size": 0.001, "min_notional": 5.0}, venue=venue)
    orders = FuturesRebalancer()._create_executable_orders(
        rebalancing_df=pd.DataFrame([row]),
        min_trade_size=0.01,
        min_notional=5.0,
        min_notional_map={"ETH": 5.0},
        lot_rules={"ETH": rules},
        default_lot_rules=LotRules.from_snapshot({}, venue=venue),
    )
    assert list(orders["Executable"]) == [True, True], orders.to_string()
    return orders


def _ledger_by_target(ledger: OrderFillLedger) -> dict[float, dict]:
    return {
        round(slot["intent"]["target_qty"], 6): (slot["result"] or {})
        for slot in ledger.match_intents_to_results().values()
        if slot["intent"] is not None
    }


def _binance_rejecting_reduce_only():
    """The real Binance futures ``place_orders``; the venue rejects the reduce-only close and fills the open."""
    from quantbox.plugins.broker.binance_futures import BinanceFuturesBroker

    broker = BinanceFuturesBroker.__new__(BinanceFuturesBroker)
    broker.quote_currency = "USDT"

    def place_order(symbol, side, quantity, order_type="market", reduce_only=False):
        if reduce_only:
            return None  # ccxt.InvalidOrder (or any refusal) -> None, as place_order does
        return {"id": "OPEN-1", "status": "closed", "filled": quantity, "amount": quantity, "average": PRICE}

    broker.place_order = place_order
    broker.get_positions = lambda: pd.DataFrame([{"symbol": "ETH", "qty": 0.02}])
    return broker


def test_guard_binance_flip_with_a_rejected_close_books_close_failed_and_open_filled(tmp_path):
    ledger = OrderFillLedger(book_key="t", root=tmp_path)
    report = execute_orders(
        broker=_binance_rejecting_reduce_only(),
        orders_df=_flip_orders(),
        trading_enabled=True,
        alert=lambda m, *, site=None: True,
        ledger=ledger,
        cycle_id="c1",
    )
    results = _ledger_by_target(ledger)
    assert results[0.02]["status"] == "failed", results  # the rejected reduce-only close
    assert results[0.03]["status"] == "filled", results  # the open leg
    assert results[0.03]["filled_qty"] == 0.03
    assert report["summary"]["total_failed"] == 1
    assert report["summary"]["total_executed"] == 1


def test_binance_futures_returns_one_row_per_order_including_a_rejection():
    frame = pd.DataFrame(
        [
            {"symbol": "ETH", "side": "sell", "qty": 0.02, "price": PRICE, "reduce_only": True},
            {"symbol": "ETH", "side": "sell", "qty": 0.03, "price": PRICE, "reduce_only": False},
        ]
    )
    rows = _binance_rejecting_reduce_only().place_orders(frame)
    assert list(rows["status"]) == ["FAILED", "FILLED"]
    assert list(rows["symbol"]) == ["ETH", "ETH"]


def test_futures_paper_returns_one_row_per_order_on_an_invalid_price():
    from quantbox.plugins.broker.futures_paper import FuturesPaperBroker

    broker = FuturesPaperBroker(margin_balance=100_000.0)
    broker.prices = {"BTC": 60_000.0}
    frame = pd.DataFrame(
        [
            {"symbol": "NOPRICE", "side": "buy", "qty": 1.0, "price": 0.0},
            {"symbol": "BTC", "side": "buy", "qty": 0.01, "price": 60_000.0},
        ]
    )
    rows = broker.place_orders(frame)
    assert list(rows["symbol"]) == ["NOPRICE", "BTC"]
    assert list(rows["status"]) == ["SKIPPED", "FILLED"]


class _DropsRejections:
    """A third-party broker that omits a rejected order's row: the shape the executor must not trust."""

    def get_positions(self):
        return pd.DataFrame([{"symbol": "ETH", "qty": 0.02}])

    def place_orders(self, frame):
        rows = [
            {
                "symbol": o["symbol"],
                "side": o["side"],
                "qty": o["qty"],
                "price": o["price"],
                "order_id": f"X{i}",
                "status": "FILLED",
                "error": "",
            }
            for i, (_, o) in enumerate(frame.iterrows())
            if not bool(o["reduce_only"])
        ]
        return pd.DataFrame(rows)


def test_guard_executor_fails_closed_when_a_key_has_fewer_results_than_orders(tmp_path):
    ledger = OrderFillLedger(book_key="t", root=tmp_path)
    alerts: list = []
    report = execute_orders(
        broker=_DropsRejections(),
        orders_df=_flip_orders(),
        trading_enabled=True,
        alert=lambda m, *, site=None: alerts.append(site) or True,
        ledger=ledger,
        cycle_id="c1",
    )
    results = _ledger_by_target(ledger)
    # Neither order is credited with the fill: neither can be told apart.
    assert results[0.02]["status"] == "failed", results
    assert results[0.03]["status"] == "failed", results
    assert all(r.get("filled_qty") is None for r in results.values())
    assert report["summary"]["total_executed"] == 0
    assert report["summary"]["total_failed"] == 2
    eth = [d for d in report["orders_details"] if d["symbol"] == "ETH"]
    assert [d["fill_status"] for d in eth] == ["UNKNOWN", "UNKNOWN"]
    assert [r["order_id"] for r in report["unbound_results"]] == ["X1"]
    assert "order_failures" in alerts  # loud
    assert "zero_fills" not in alerts  # and not mislabelled as a dead broker


def test_aligned_results_bind_as_before(tmp_path):
    """One row per order: nothing is unbound, and both legs book their own outcome."""
    from quantbox.trade.executor import unbound_keys

    frame = pd.DataFrame([{"symbol": "ETH", "side": "sell"}, {"symbol": "ETH", "side": "sell"}])
    assert unbound_keys(frame, frame) == set()
    assert unbound_keys(frame, frame.iloc[:1]) == {("ETH", "sell")}
