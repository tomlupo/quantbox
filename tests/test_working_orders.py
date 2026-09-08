"""Working orders: the cross-cycle path that books a late fill.

A priced order reaches the venue as a LIMIT order and may rest on the book long
after the run that placed it exited. The bounded in-run confirmation window
cannot wait for that (a real Kraken order rested 6m04s against a ~9s window on
2026-09-08 and filled in full, and was reported FAILED). These tests pin the two
halves of the fix: the order is reported WORKING rather than FAILED, and a later
cycle resolves it against the venue and books what actually happened.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline
from quantbox.reconciliation.working_orders import WorkingOrderStore

# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def test_store_roundtrip_and_drop(tmp_path):
    store = WorkingOrderStore(book_key="book-a", root=tmp_path)
    store.record(symbol="DOGE", side="buy", order_id="OID1", requested_qty=94.25, cycle_id="c1")
    store.record(symbol="ENA", side="buy", order_id="OID2", requested_qty=42.0, cycle_id="c1")
    assert {r["order_id"] for r in store.load()} == {"OID1", "OID2"}

    store.drop({"OID1"})
    remaining = store.load()
    assert [r["order_id"] for r in remaining] == ["OID2"]
    assert remaining[0]["symbol"] == "ENA"


def test_store_is_namespaced_per_book(tmp_path):
    a = WorkingOrderStore(book_key="book-a", root=tmp_path)
    b = WorkingOrderStore(book_key="book-b", root=tmp_path)
    a.record(symbol="DOGE", side="buy", order_id="OID1", requested_qty=1.0, cycle_id="c1")
    assert a.load() and b.load() == []


def test_store_read_is_fail_soft_but_visible(tmp_path):
    store = WorkingOrderStore(book_key="book-a", root=tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not json")
    # A corrupt queue must not stop the book trading.
    assert store.load() == []


def test_store_rejects_a_book_key_that_escapes_the_root(tmp_path):
    with pytest.raises(ValueError, match="single safe path segment"):
        WorkingOrderStore(book_key="../escape", root=tmp_path)


# ---------------------------------------------------------------------------
# Resolution against the venue
# ---------------------------------------------------------------------------


class _Broker:
    """Broker stub whose fetch_order_result is scripted per order id."""

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def fetch_order_result(self, order_id, symbol):
        self.calls.append((order_id, symbol))
        return self.outcomes.get(order_id)


class _NoResolveBroker:
    pass


class _Ledger:
    def __init__(self):
        self.results = []

    def record_result(self, **kw):
        self.results.append(kw)
        return kw


def _queued(tmp_path, **overrides):
    store = WorkingOrderStore(book_key="book-a", root=tmp_path)
    rec = {
        "symbol": "DOGE",
        "side": "buy",
        "order_id": "OID1",
        "requested_qty": 94.25,
        "cycle_id": "c1",
        "order_ref": "ref-1",
    }
    rec.update(overrides)
    store.record(**rec)
    return store


def test_a_late_fill_is_booked_and_dropped_from_the_queue(tmp_path):
    # The real 2026-09-08 case: reported WORKING, filled 6 minutes later.
    store = _queued(tmp_path)
    broker = _Broker({"OID1": {"status": "FILLED", "qty": 94.25, "price": 0.0897354, "error": ""}})
    ledger = _Ledger()

    resolved = TradingPipeline()._resolve_working_orders(store, broker, ledger=ledger)

    assert len(resolved) == 1
    assert resolved[0]["status"] == "FILLED"
    assert resolved[0]["quantity"] == pytest.approx(94.25)
    assert store.load() == []  # no longer tracked
    # ...and it binds to the intent that produced it, not an orphan record.
    assert ledger.results[0]["order_ref"] == "ref-1"
    assert ledger.results[0]["filled_qty"] == pytest.approx(94.25)
    assert ledger.results[0]["resolved_late"] is True


def test_an_unreadable_venue_keeps_the_order_queued(tmp_path):
    # "Could not check" must exit differently from "checked and found nothing":
    # discarding here would throw away a real fill on a transient API error.
    store = _queued(tmp_path)
    broker = _Broker({"OID1": None})

    resolved = TradingPipeline()._resolve_working_orders(store, broker)

    assert resolved == []
    assert [r["order_id"] for r in store.load()] == ["OID1"]


def test_an_order_still_working_stays_queued(tmp_path):
    store = _queued(tmp_path)
    broker = _Broker({"OID1": {"status": "WORKING", "qty": 0.0, "price": 0.0, "error": "still working"}})

    resolved = TradingPipeline()._resolve_working_orders(store, broker)

    assert resolved == []
    assert [r["order_id"] for r in store.load()] == ["OID1"]


def test_a_late_failure_is_booked_as_a_failure_not_a_fill(tmp_path):
    store = _queued(tmp_path)
    broker = _Broker({"OID1": {"status": "FAILED", "qty": 0.0, "price": 0.0, "error": "canceled"}})

    resolved = TradingPipeline()._resolve_working_orders(store, broker)

    assert [r["status"] for r in resolved] == ["FAILED"]
    assert resolved[0]["quantity"] == 0.0
    assert store.load() == []


def test_a_broker_that_cannot_resolve_leaves_the_queue_intact(tmp_path):
    store = _queued(tmp_path)

    resolved = TradingPipeline()._resolve_working_orders(store, _NoResolveBroker())

    assert resolved == []
    assert [r["order_id"] for r in store.load()] == ["OID1"]


def test_an_order_unresolved_past_the_retention_cap_is_dropped_loudly(tmp_path, caplog):
    store = _queued(tmp_path)
    stale = store.load()
    stale[0]["recorded_at"] = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    store.save(stale)
    broker = _Broker({"OID1": None})

    with caplog.at_level("ERROR"):
        TradingPipeline()._resolve_working_orders(store, broker)

    assert store.load() == []  # queue cannot grow without bound
    assert "reconcile it by hand" in caplog.text


def test_an_empty_queue_does_not_touch_the_broker(tmp_path):
    store = WorkingOrderStore(book_key="book-a", root=tmp_path)
    broker = _Broker({})
    assert TradingPipeline()._resolve_working_orders(store, broker) == []
    assert broker.calls == []


# ---------------------------------------------------------------------------
# Store selection
# ---------------------------------------------------------------------------


def test_working_store_is_none_without_a_book_key(tmp_path):
    assert TradingPipeline()._working_store({"data_dir": str(tmp_path)}) is None


def test_working_store_uses_the_reconciliation_book_key_when_present(tmp_path):
    store = TradingPipeline()._working_store({"reconciliation": {"book_key": "carver-HL", "data_dir": str(tmp_path)}})
    assert store is not None
    assert store.book_key == "carver-HL"
