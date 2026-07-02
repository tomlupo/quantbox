"""Append-only order/fill ledger tests (issue #87, deliverable #1)."""

from __future__ import annotations

import json

from quantbox.reconciliation import KIND_INTENT, KIND_RESULT, OrderFillLedger


def _ledger(tmp_path, clock=None):
    return OrderFillLedger(book_key="carver-HL", root=tmp_path, clock=clock)


def test_per_book_path_namespacing(tmp_path):
    led = _ledger(tmp_path)
    assert led.path == tmp_path / "carver-HL" / "orders.jsonl"
    assert led.path.parent.is_dir()


def test_empty_book_key_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        OrderFillLedger(book_key="", root=tmp_path)


def test_intent_then_result_appended_in_order(tmp_path):
    ticks = iter(["2026-01-01T00:00:00+00:00", "2026-01-01T00:00:05+00:00"])
    led = _ledger(tmp_path, clock=lambda: next(ticks))
    led.record_intent(
        cycle_id="c1",
        symbol="DOGE",
        side="buy",
        order_ref="r1",
        target_qty=100.0,
        target_wt=0.5,
        limit_px=0.1,
    )
    led.record_result(order_ref="r1", status="filled", filled_qty=100.0, avg_px=0.1)

    lines = led.path.read_text().strip().splitlines()
    assert len(lines) == 2
    intent = json.loads(lines[0])
    result = json.loads(lines[1])
    assert intent["kind"] == KIND_INTENT
    assert intent["intended"] is True
    assert intent["symbol"] == "DOGE" and intent["order_ref"] == "r1"
    assert result["kind"] == KIND_RESULT
    assert result["status"] == "filled" and result["order_ref"] == "r1"


def test_append_only_never_rewrites(tmp_path):
    led = _ledger(tmp_path)
    for i in range(3):
        led.record_intent(cycle_id=f"c{i}", symbol="X", side="buy", order_ref=f"r{i}")
    assert len(led.read_all()) == 3
    led.record_intent(cycle_id="c3", symbol="X", side="buy", order_ref="r3")
    # Earlier lines are untouched; only a new line is appended.
    assert len(led.read_all()) == 4


def test_match_intents_to_results_detects_missed_fill(tmp_path):
    led = _ledger(tmp_path)
    # r1 filled; r2 submitted but NO result -> a missed fill only the ledger sees.
    led.record_intent(cycle_id="c1", symbol="DOGE", side="buy", order_ref="r1")
    led.record_result(order_ref="r1", status="filled", filled_qty=1.0, avg_px=0.1)
    led.record_intent(cycle_id="c1", symbol="ETH", side="buy", order_ref="r2")

    matched = led.match_intents_to_results()
    assert matched["r1"]["result"]["status"] == "filled"
    assert matched["r2"]["intent"] is not None
    assert matched["r2"]["result"] is None  # missed fill signature


def test_corrupt_trailing_line_is_skipped(tmp_path):
    led = _ledger(tmp_path)
    led.record_intent(cycle_id="c1", symbol="X", side="buy", order_ref="r1")
    with open(led.path, "a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")
    assert len(led.read_all()) == 1


def test_intents_for_cycle_filters(tmp_path):
    led = _ledger(tmp_path)
    led.record_intent(cycle_id="c1", symbol="A", side="buy", order_ref="r1")
    led.record_intent(cycle_id="c2", symbol="B", side="buy", order_ref="r2")
    got = led.intents_for_cycle("c1")
    assert [r["symbol"] for r in got] == ["A"]
