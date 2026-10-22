"""``ibkr.live.v1`` reports the order's own status, never a hard-coded FILLED (TOM-1978).

IB is faked at its API object: no network, no ib_insync needed. Each fake
``ib.sleep`` advances the order to the next scripted ``OrderStatus``, the way
ib_insync's event loop delivers status messages while it sleeps.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from quantbox.plugins.broker import ibkr
from quantbox.plugins.broker._fills import STATUS_WORKING


def _status(status: str, filled: float, avg: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(status=status, filled=filled, remaining=100.0 - filled, avgFillPrice=avg)


class _FakeIB:
    def __init__(self, script: list[SimpleNamespace], log: list[str] | None = None):
        self.script = list(script)
        self.log = [SimpleNamespace(message=m) for m in (log or [])]
        self.trade = None

    def connect(self, *args, **kwargs):
        return None

    def qualifyContracts(self, *contracts):
        return list(contracts)

    def placeOrder(self, contract, order):
        order.orderId = 42
        self.trade = SimpleNamespace(contract=contract, order=order, orderStatus=self.script[0], fills=[], log=self.log)
        return self.trade

    def sleep(self, seconds):
        if len(self.script) > 1:
            self.script.pop(0)
        if self.trade is not None:
            self.trade.orderStatus = self.script[0]


def _broker(monkeypatch, script, log=None):
    fake = _FakeIB(script, log)
    monkeypatch.setattr(ibkr, "IB", lambda: fake, raising=False)
    monkeypatch.setattr(ibkr, "Stock", lambda sym, exch, ccy: SimpleNamespace(symbol=sym), raising=False)
    monkeypatch.setattr(
        ibkr,
        "MarketOrder",
        lambda action, qty: SimpleNamespace(action=action, totalQuantity=qty, lmtPrice=1.7976931348623157e308),
        raising=False,
    )
    monkeypatch.setattr(
        ibkr,
        "LimitOrder",
        lambda action, qty, px: SimpleNamespace(action=action, totalQuantity=qty, lmtPrice=px),
        raising=False,
    )
    return ibkr.IBKRBroker()


ORDER = pd.DataFrame([{"symbol": "SPY", "side": "buy", "qty": 100.0, "price": 0.0}])


@pytest.mark.parametrize(
    ("script", "log", "status", "qty", "price"),
    [
        # A partial fill: IB still works the rest of the order.
        ([_status("Submitted", 40.0, 500.0)], None, "PARTIAL", 40.0, 500.0),
        # A working order: accepted, nothing filled when the window closes.
        ([_status("Submitted", 0.0)] * 5, None, STATUS_WORKING, 0.0, None),
        # A rejection: IB reports an order it will not work as Inactive.
        ([_status("Inactive", 0.0)], ["Order rejected - reason: no trading permissions"], "FAILED", 0.0, None),
        # Not yet acknowledged at the first read, then filled in full on the re-read.
        ([_status("PendingSubmit", 0.0), _status("Filled", 100.0, 501.0)], None, "FILLED", 100.0, 501.0),
    ],
    ids=["partial", "working", "rejected", "settles-filled"],
)
def test_ibkr_reports_the_orders_own_status(monkeypatch, script, log, status, qty, price):
    row = _broker(monkeypatch, script, log).place_orders(ORDER).iloc[0]

    assert row["status"] == status
    assert row["qty"] == pytest.approx(qty)
    assert row["requested_qty"] == pytest.approx(100.0)
    assert row["order_id"] == "42"
    assert row["price"] < 1e300  # IB's unset lmtPrice on a market order never reaches the row
    if price is not None:
        assert row["price"] == pytest.approx(price)
    if status == "FAILED":
        assert "no trading permissions" in row["error"]
