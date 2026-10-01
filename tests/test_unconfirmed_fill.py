"""An unconfirmed order never books as a full fill (TOM-1336).

Two holes let an order the venue never confirmed book as a FULL fill:

* the trading pipeline read a fill row with no ``status`` as ``FILLED``;
* the Binance futures broker emitted no status at all and, when ccxt omitted
  ``filled``, reported the REQUESTED quantity as the filled one.

The rule pinned here: a missing status is UNKNOWN, a missing ``filled`` is an
unknown quantity (never the requested one), every broker's order result goes
through the shared fill vocabulary in ``_fills``, and an UNKNOWN outcome flows
into the failure / working-order handling instead of the fill path.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantbox.plugins.broker import _fills
from quantbox.plugins.broker._fills import STATUS_UNKNOWN, classify_fill, resolve_fill

NO_STATUS = {"id": "OID-1", "filled": 1.0, "amount": 1.0, "average": 100.0}
NO_FILLED = {"id": "OID-2", "status": "closed", "amount": 1.0, "average": 100.0}


# ---------------------------------------------------------------------------
# The shared classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [NO_STATUS, NO_FILLED], ids=["no-status", "no-filled"])
def test_classifier_reads_a_missing_field_as_unknown_not_filled(order):
    verdict, qty, _ = classify_fill(order, 1.0)
    assert verdict == _fills.FILL_UNKNOWN
    assert qty == 0.0


def test_an_unrecognised_status_is_unknown():
    verdict, qty, _ = classify_fill({"status": "weird", "filled": 1.0}, 1.0)
    assert verdict == _fills.FILL_UNKNOWN
    assert qty == 0.0


@pytest.mark.parametrize("order", [NO_STATUS, NO_FILLED], ids=["no-status", "no-filled"])
def test_resolver_emits_unknown_and_books_no_fill_when_unconfirmable(order):
    status, qty, _, reason = resolve_fill(order, 1.0, refetch=lambda: None, confirm_delay=0)
    assert status == STATUS_UNKNOWN
    assert qty == 0.0
    assert "not confirmed" in reason


def test_resolver_still_books_a_fill_the_venue_confirms_on_refetch():
    # The normal-day path: the create reply is ambiguous, the re-read is not.
    status, qty, _, _ = resolve_fill(
        NO_FILLED, 1.0, refetch=lambda: {"status": "closed", "filled": 1.0, "average": 100.0}, confirm_delay=0
    )
    assert status == "FILLED"
    assert qty == 1.0


# ---------------------------------------------------------------------------
# Each live broker adapter
# ---------------------------------------------------------------------------


class _FakeExchange:
    """fetch_order echoes the scripted reply: the venue cannot confirm either."""

    def __init__(self, refetched):
        self.refetched = refetched
        self.fetch_calls = 0

    def fetch_order(self, order_id, symbol):
        self.fetch_calls += 1
        return self.refetched


def _one_buy():
    return pd.DataFrame([{"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 0.0}])


def _binance_futures(result, refetched):
    from quantbox.plugins.broker.binance_futures import BinanceFuturesBroker

    broker = BinanceFuturesBroker.__new__(BinanceFuturesBroker)
    broker._exchange = _FakeExchange(refetched)
    broker.quote_currency = "USDT"
    broker.place_order = lambda *a, **k: dict(result)
    return broker


def _hyperliquid(result, refetched):
    from quantbox.plugins.broker.hyperliquid import HyperliquidBroker

    broker = HyperliquidBroker.__new__(HyperliquidBroker)
    broker._exchange = _FakeExchange(refetched)
    broker._get_market_symbol = lambda s: f"{s}/USDC:USDC"
    broker.place_order = lambda *a, **k: dict(result)
    broker._notify_order_outcome = lambda *a, **k: False
    return broker


def _kraken(result, refetched):
    from quantbox.plugins.broker.kraken import KrakenBroker

    broker = KrakenBroker.__new__(KrakenBroker)
    broker._exchange = _FakeExchange(refetched)
    broker.quote_asset = "USD"
    broker.readonly = False
    broker._market_symbol = lambda s: f"{s}/USD"
    broker._place_one = lambda *a, **k: dict(result)
    return broker


BROKERS = {"binance_futures": _binance_futures, "hyperliquid": _hyperliquid, "kraken": _kraken}


@pytest.mark.parametrize("make", list(BROKERS.values()), ids=list(BROKERS))
@pytest.mark.parametrize("order", [NO_STATUS, NO_FILLED], ids=["no-status", "no-filled"])
def test_broker_reports_an_unconfirmed_result_as_unknown_with_no_fill(make, order):
    broker = make(order, refetched={k: v for k, v in order.items() if k in ("id",)})
    row = broker.place_orders(_one_buy()).iloc[0]
    assert row["status"] == STATUS_UNKNOWN
    assert row["qty"] == 0.0
    assert row["order_id"] == order["id"]


def test_binance_futures_goes_through_the_shared_resolver_and_refetches():
    # Proof the adapter does not classify on its own: an ambiguous create reply
    # is re-read from the venue, and the venue's answer is what gets booked.
    broker = _binance_futures(NO_FILLED, refetched={"id": "OID-2", "status": "closed", "filled": 0.4, "average": 99.0})
    row = broker.place_orders(_one_buy()).iloc[0]
    assert broker._exchange.fetch_calls == 1
    assert row["status"] == "PARTIAL"
    assert row["qty"] == pytest.approx(0.4)
    assert row["price"] == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# The pipeline: a missing status is UNKNOWN, and UNKNOWN is not dropped
# ---------------------------------------------------------------------------


class _RowBroker:
    def __init__(self, row):
        self.row = row
        self.messages: list[str] = []

    def notify(self, message):
        self.messages.append(message)
        return True

    def place_orders(self, orders):
        return pd.DataFrame([dict(self.row)])


class _Store:
    def __init__(self):
        self.recorded: list[dict] = []

    def record(self, **kw):
        self.recorded.append(kw)


def _orders_df():
    return pd.DataFrame(
        [
            {
                "Asset": "BTC",
                "Action": "Buy",
                "Adjusted Quantity": 1.0,
                "Price": 100.0,
                "Notional Value": 100.0,
                "Min Notional": 10.0,
                "Order Status": "To be placed",
                "Executable": True,
            }
        ]
    )


def _run(row, store=None):
    from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline

    broker = _RowBroker(row)
    report = TradingPipeline()._execute_orders(
        broker=broker,
        orders_df=_orders_df(),
        stable_coin="USDT",
        trading_enabled=True,
        mode="live",
        working_store=store,
    )
    return report, broker


@pytest.mark.parametrize(
    "row",
    [
        {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 100.0, "order_id": "OID-9"},
        {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 100.0, "order_id": "OID-9", "status": None},
        {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 100.0, "order_id": "OID-9", "status": "UNKNOWN"},
        {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 100.0, "order_id": "OID-9", "status": "weird"},
    ],
    ids=["missing", "none", "unknown", "unrecognised"],
)
def test_pipeline_books_no_fill_for_an_unconfirmed_row(row):
    store = _Store()
    report, broker = _run(row, store)

    assert report["summary"]["total_executed"] == 0
    assert not [d for d in report["orders_details"] if d.get("status") in ("FILLED", "PARTIAL")]
    # Not silently dropped: it is a failure the operator is told about...
    assert report["summary"]["total_failed"] == 1
    assert any("BTC" in m and "UNCONFIRMED" in m.upper() for m in broker.messages)
    # ...and, carrying an order id, it is queued so the next cycle books the
    # venue's real outcome through the working-order path.
    assert [r["order_id"] for r in store.recorded] == ["OID-9"]


def test_pipeline_still_books_an_explicit_fill():
    report, _ = _run({"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 100.0, "status": "FILLED"})
    assert report["summary"]["total_executed"] == 1
    assert report["summary"]["total_failed"] == 0


# ---------------------------------------------------------------------------
# The working-order resolver: an UNKNOWN late outcome stays queued
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("late_status", [STATUS_UNKNOWN, None, ""], ids=["unknown", "none", "empty"])
def test_working_order_resolver_keeps_an_unknown_outcome_queued(tmp_path, late_status):
    from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline
    from quantbox.reconciliation.working_orders import WorkingOrderStore

    store = WorkingOrderStore(book_key="book-a", root=tmp_path)
    store.record(symbol="BTC", side="buy", order_id="OID1", requested_qty=1.0, cycle_id="c1", order_ref="r1")

    class _B:
        def fetch_order_result(self, order_id, symbol):
            return {"status": late_status, "qty": 0.0, "price": 0.0, "error": "not confirmed"}

    resolved = TradingPipeline()._resolve_working_orders(store, _B())
    assert resolved == []
    assert [r["order_id"] for r in store.load()] == ["OID1"]


# ---------------------------------------------------------------------------
# Brokers whose rows ARE executions must say so — the pipeline no longer
# assumes it. Without an explicit status their fills would read UNKNOWN.
# ---------------------------------------------------------------------------


def _sim():
    from quantbox.plugins.broker.sim import SimPaperBroker

    return SimPaperBroker()


def _futures_paper():
    from quantbox.plugins.broker.futures_paper import FuturesPaperBroker

    return FuturesPaperBroker()


def _binance_stub():
    from quantbox.plugins.broker.binance_stub import PaperBrokerStub

    return PaperBrokerStub()


def _ibkr_stub():
    from quantbox.plugins.broker.ibkr_stub import PaperBrokerStub

    return PaperBrokerStub()


SIMULATED = {"sim": _sim, "futures_paper": _futures_paper, "binance_stub": _binance_stub, "ibkr_stub": _ibkr_stub}


@pytest.mark.parametrize("make", list(SIMULATED.values()), ids=list(SIMULATED))
def test_simulated_broker_rows_carry_an_explicit_filled_status(make, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = make().place_orders(pd.DataFrame([{"symbol": "BTC", "side": "buy", "qty": 0.01, "price": 100.0}]))
    assert not out.empty
    assert list(out["status"]) == ["FILLED"]
