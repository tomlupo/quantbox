"""An unbound broker result is still tracked and still loud (TOM-2422, follow-up of PR #310).

When a (symbol, side) gets more or fewer result rows than it had orders, the
executor binds none of them (TOM-2396). Two holes were left:

1. A WORKING or UNKNOWN row of such a key is a live order at the venue. It was
   only listed under ``unbound_results``, never queued, so its later fill was
   lost. It now joins the working-order store with its order id, and the
   failure alert names the ids.
2. Result rows for a key with NO submitted order booked no FAILED row, so only
   a log line said so. They now raise an alert (site ``orphan_results``).

Every test here is red at b83e1ff and green after the fix.
"""

from __future__ import annotations

import pandas as pd

from quantbox.trade.executor import execute_orders
from quantbox.trade.reconciliation.working_orders import WorkingOrderStore

#: A flip (an ETH reduce-only close, then an ETH open, both sells) and a BTC buy.
ORDERS = pd.DataFrame(
    [
        {"Asset": "ETH", "Action": "Sell", "Adjusted Quantity": 0.02, "Price": 2000.0, "Executable": True,
         "reduce_only": True, "opens_exposure": False},
        {"Asset": "ETH", "Action": "Sell", "Adjusted Quantity": 0.03, "Price": 2000.0, "Executable": True,
         "reduce_only": False, "opens_exposure": True},
        {"Asset": "BTC", "Action": "Buy", "Adjusted Quantity": 0.001, "Price": 60000.0, "Executable": True,
         "reduce_only": False, "opens_exposure": True},
    ]
)  # fmt: skip


def _row(symbol, side, status, qty=0.0, oid=None):
    return {"symbol": symbol, "side": side, "qty": qty, "price": 0.0, "order_id": oid, "status": status, "error": ""}


class _Broker:
    def __init__(self, rows):
        self.rows = rows

    def get_positions(self):
        return pd.DataFrame([{"symbol": "ETH", "qty": 0.02}])

    def place_orders(self, frame):
        return pd.DataFrame(self.rows)


def _run(rows, tmp_path):
    alerts: list[tuple[str, str]] = []
    store = WorkingOrderStore(book_key="t", root=tmp_path)
    report = execute_orders(
        broker=_Broker(rows),
        orders_df=ORDERS,
        trading_enabled=True,
        alert=lambda m, *, site=None: alerts.append((site, m)) or True,
        cycle_id="c1",
        working_store=store,
    )
    return report, store.load(), dict(alerts)


def test_guard_an_unbound_working_row_is_queued_with_its_order_id(tmp_path):
    # Two ETH orders, ONE ETH row back: the venue still works W1, and no one can say which order it is.
    rows = [_row("ETH", "sell", "WORKING", oid="W1"), _row("BTC", "buy", "FILLED", 0.001, "B1")]
    report, queued, alerts = _run(rows, tmp_path)
    assert [(q["symbol"], q["side"], q["order_id"], q["order_ref"]) for q in queued] == [("ETH", "sell", "W1", None)]
    assert report["unbound_results"][0]["queued"] is True
    assert "W1" in alerts["order_failures"]  # the failure alert names the venue order id
    assert report["summary"]["total_failed"] == 2 and report["summary"]["total_executed"] == 1


def test_guard_an_unbound_unknown_row_is_queued_and_one_without_an_id_is_not(tmp_path):
    rows = [
        _row("ETH", "sell", "UNKNOWN", oid="U1"),
        _row("ETH", "sell", "FAILED"),
        _row("ETH", "sell", "UNKNOWN"),  # three rows for two orders: unbound
        _row("BTC", "buy", "FILLED", 0.001, "B1"),
    ]
    report, queued, alerts = _run(rows, tmp_path)
    assert [q["order_id"] for q in queued] == ["U1"]
    assert "U1" in alerts["order_failures"]


def test_guard_result_rows_for_a_key_with_no_submitted_order_alert(tmp_path):
    rows = [
        _row("ETH", "sell", "FILLED", 0.02, "E1"),
        _row("ETH", "sell", "FILLED", 0.03, "E2"),
        _row("BTC", "buy", "FILLED", 0.001, "B1"),
        _row("SOL", "buy", "WORKING", oid="S9"),  # this run sent no SOL order
    ]
    report, queued, alerts = _run(rows, tmp_path)
    assert "orphan_results" in alerts
    assert "SOL" in alerts["orphan_results"] and "S9" in alerts["orphan_results"]
    assert report["orphan_results"] == ["BUY SOL: 1 row(s), order ids S9"]
    assert [q["order_id"] for q in queued] == ["S9"]  # still live at the venue: tracked
    assert report["summary"]["total_executed"] == 3  # the bound keys book as before
    assert report["summary"]["total_failed"] == 0
