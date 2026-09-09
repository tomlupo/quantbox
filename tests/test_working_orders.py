"""Working orders: the cross-cycle path that books a late fill.

A priced order reaches the venue as a LIMIT order and may rest on the book long
after the run that placed it exited. The bounded in-run confirmation window
cannot wait for that (a real Kraken order rested 6m04s against a ~9s window on
2026-09-08 and filled in full, and was reported FAILED). These tests pin the two
halves of the fix: the order is reported WORKING rather than FAILED, and a later
cycle resolves it against the venue and books what actually happened.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline
from quantbox.reconciliation import working_orders as working_orders_mod
from quantbox.reconciliation.working_orders import (
    WorkingOrderResolution,
    WorkingOrderStore,
    resolve_working_orders,
)

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
    stale[0]["recorded_at"] = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
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


# ---------------------------------------------------------------------------
# The resolver as a FUNCTION — the surface the out-of-cycle follow-up job uses.
#
# The pipeline tests above exercise the same code through Stage 6c's delegate,
# which is the positive control that one implementation really serves both
# callers: break the function and both halves of this file go red.
# ---------------------------------------------------------------------------


def test_the_pipeline_stage_delegates_to_the_shared_function(monkeypatch, tmp_path):
    """No second copy of the semantics: Stage 6c is a call, not a reimplementation."""
    calls = []

    def _spy(store, broker, *, ledger=None, max_age_days=7, now=None):
        calls.append((store, broker, ledger, max_age_days))
        return WorkingOrderResolution(queued=0, checked=True)

    monkeypatch.setattr(working_orders_mod, "resolve_working_orders", _spy)
    store = _queued(tmp_path)
    assert TradingPipeline()._resolve_working_orders(store, _Broker({}), ledger=None) == []
    assert len(calls) == 1
    assert calls[0][3] == TradingPipeline._WORKING_MAX_AGE_DAYS


def test_every_queued_order_lands_in_exactly_one_bucket(tmp_path):
    store = WorkingOrderStore(book_key="book-a", root=tmp_path)
    for oid, sym in (("F", "DOGE"), ("W", "ENA"), ("U", "UNI")):
        store.record(symbol=sym, side="sell", order_id=oid, requested_qty=1.0, cycle_id="c1", order_ref=f"ref-{oid}")
    broker = _Broker(
        {
            "F": {"status": "FILLED", "qty": 1.0, "price": 2.0},
            "W": {"status": "WORKING"},
            "U": None,  # venue unreadable
        }
    )

    res = resolve_working_orders(store, broker)

    assert [r["order_id"] for r in res.resolved] == ["F"]
    assert [r["order_id"] for r in res.still_working] == ["W"]
    assert [r["order_id"] for r in res.unreadable] == ["U"]
    assert res.queued == 3
    assert res.checked is True
    # Both non-terminal buckets stay on the queue; only the booked one leaves.
    assert {r["order_id"] for r in store.load()} == {"W", "U"}
    assert {r["order_id"] for r in res.unresolved} == {"W", "U"}


def test_an_unreadable_venue_is_not_a_clean_check(tmp_path):
    """The distinction the whole queue exists for: 'could not check' != 'nothing found'."""
    store = _queued(tmp_path)
    res = resolve_working_orders(store, _Broker({"OID1": None}))
    assert res.resolved == []
    assert res.still_working == []  # NOT filed as 'still working'
    assert len(res.unreadable) == 1  # filed as its own outcome
    assert res.unresolved  # so a caller cannot report 'clean'


def test_a_broker_that_cannot_resolve_reports_that_it_did_not_look(tmp_path):
    store = _queued(tmp_path)
    res = resolve_working_orders(store, _NoResolveBroker())
    assert res.checked is False  # we did not look
    assert len(res.unreadable) == 1
    assert store.load()  # and nothing was discarded


def test_an_empty_queue_is_a_complete_check(tmp_path):
    store = WorkingOrderStore(book_key="book-a", root=tmp_path)
    res = resolve_working_orders(store, _Broker({}))
    assert res.checked is True and res.queued == 0 and res.unresolved == []


def test_a_venue_that_raises_is_unreadable_not_terminal(tmp_path):
    """A ccxt exception must never be read as 'the order is gone'."""

    class _Exploding:
        def fetch_order_result(self, order_id, symbol):
            raise RuntimeError("kraken 520")

    store = _queued(tmp_path)
    res = resolve_working_orders(store, _Exploding())
    assert len(res.unreadable) == 1
    assert res.resolved == []
    assert store.load()  # kept for the next pass


def test_older_than_finds_the_order_still_open_an_hour_later(tmp_path):
    """The operational signal Tom asked for: still open after the threshold."""
    store = _queued(tmp_path)
    now = datetime.now(timezone.utc)
    recs = store.load()
    recs[0]["recorded_at"] = (now - timedelta(minutes=90)).isoformat()
    store.save(recs)

    res = resolve_working_orders(store, _Broker({"OID1": {"status": "WORKING"}}), now=now)

    assert res.older_than(3600, now=now)  # 90m > 1h -> flagged
    assert res.older_than(4 * 3600, now=now) == []  # 90m < 4h -> not yet


def test_a_fresh_working_order_is_not_flagged_as_stuck(tmp_path):
    """Positive control on the other side: the threshold can say 'no'."""
    store = _queued(tmp_path)
    now = datetime.now(timezone.utc)
    res = resolve_working_orders(store, _Broker({"OID1": {"status": "WORKING"}}), now=now)
    assert res.still_working and res.older_than(3600, now=now) == []


def test_an_unparseable_age_is_flagged_rather_than_assumed_young(tmp_path):
    """An age we cannot read must not read as 'young enough to ignore'."""
    store = _queued(tmp_path)
    recs = store.load()
    recs[0]["recorded_at"] = "not-a-timestamp"
    store.save(recs)
    res = resolve_working_orders(store, _Broker({"OID1": {"status": "WORKING"}}))
    assert len(res.older_than(3600)) == 1


def test_an_unparseable_age_is_never_expired_away(tmp_path):
    """...but the same unreadable timestamp must NOT drop a real fill."""
    store = _queued(tmp_path)
    recs = store.load()
    recs[0]["recorded_at"] = "not-a-timestamp"
    store.save(recs)
    res = resolve_working_orders(store, _Broker({"OID1": None}), max_age_days=0)
    assert res.expired == []
    assert store.load()  # still queued


def test_the_retention_cap_is_configurable_by_the_caller(tmp_path):
    store = _queued(tmp_path)
    recs = store.load()
    recs[0]["recorded_at"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    store.save(recs)

    kept = resolve_working_orders(store, _Broker({"OID1": None}), max_age_days=7)
    assert kept.expired == [] and store.load()

    dropped = resolve_working_orders(store, _Broker({"OID1": None}), max_age_days=1)
    assert len(dropped.expired) == 1 and store.load() == []


def test_a_late_fill_is_booked_against_the_carried_order_ref(tmp_path):
    store = _queued(tmp_path)
    ledger = _Ledger()
    res = resolve_working_orders(
        store, _Broker({"OID1": {"status": "FILLED", "qty": 94.25, "price": 0.09}}), ledger=ledger
    )
    assert len(res.resolved) == 1
    assert ledger.results[0]["order_ref"] == "ref-1"
    assert ledger.results[0]["status"] == "filled"
    assert ledger.results[0]["resolved_late"] is True


def test_a_ledger_write_failure_does_not_lose_the_resolution(tmp_path):
    class _BrokenLedger:
        def record_result(self, **kw):
            raise OSError("disk full")

    store = _queued(tmp_path)
    res = resolve_working_orders(
        store, _Broker({"OID1": {"status": "FILLED", "qty": 1.0, "price": 2.0}}), ledger=_BrokenLedger()
    )
    assert len(res.resolved) == 1  # still reported to the caller


def test_expired_cuts_across_the_outcome_buckets_rather_than_being_a_fourth(tmp_path):
    """An aged-out order is filed under the outcome the venue gave it AND under
    `expired`. The docstring says so; this is what stops that claim drifting."""
    store = _queued(tmp_path)
    recs = store.load()
    recs[0]["recorded_at"] = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    store.save(recs)

    res = resolve_working_orders(store, _Broker({"OID1": {"status": "WORKING"}}))

    assert len(res.expired) == 1
    assert len(res.still_working) == 1  # counted in its outcome bucket TOO
    assert res.expired[0]["order_id"] == res.still_working[0]["order_id"]
    assert store.load() == []  # and it is the unresolved order that gets dropped


def test_the_resolution_exposes_no_dropped_attribute(tmp_path):
    """Pins the absence the docstring used to claim as a field. Documenting an
    attribute that does not exist is a contract a caller cannot use."""
    res = resolve_working_orders(WorkingOrderStore(book_key="book-a", root=tmp_path), _Broker({}))
    assert not hasattr(res, "dropped")
