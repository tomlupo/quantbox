"""Submission-time intent capture + REAL pre-execution enforcement (issue #90).

Covers the two #90 deliverables end-to-end at the pipeline-helper level:

1. Intent is captured INSIDE ``_execute_orders`` at the broker submission call —
   on disk (fsynced) BEFORE ``place_orders`` runs — so the ledger is a
   crash-durable audit trail, and Stage 7b reads it back rather than
   reconstructing it.
2. Enforcement is REAL and crash-durable: the state machine persists its state,
   and the NEXT cycle's pre-execution gate reads it back to HALT (send nothing)
   or FLATTEN (reduce-only) BEFORE orders are executed. Enforce is Tom-gated
   behind an explicit ``enforce_acknowledged`` flag; observe stays the default.
"""

from __future__ import annotations

import json

import pandas as pd

from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline


class _FakeBroker:
    def __init__(self, positions=None, fills_fn=None, place_hook=None):
        self._positions = positions or []
        self._fills_fn = fills_fn if fills_fn is not None else _fills_all("FILLED")
        self._place_hook = place_hook
        self.placed: list[pd.DataFrame] = []
        self.alerts: list[str] = []

    def get_positions(self):
        return pd.DataFrame(self._positions, columns=["symbol", "qty"]) if self._positions else pd.DataFrame()

    def get_market_snapshot(self, symbols):
        return pd.DataFrame({"symbol": list(symbols), "mid": [1.0] * len(list(symbols))})

    def notify(self, msg):
        self.alerts.append(msg)
        return True

    def place_orders(self, orders):
        if self._place_hook is not None:
            self._place_hook(orders)
        self.placed.append(orders.copy())
        return self._fills_fn(orders)


def _fills_all(status):
    def _fn(orders):
        rows = []
        for _, o in orders.iterrows():
            filled = status in ("FILLED", "PARTIAL")
            rows.append(
                {
                    "symbol": o["symbol"],
                    "side": o["side"],
                    "qty": float(o["qty"]) if filled else 0.0,
                    "price": float(o["price"]),
                    "status": status,
                    "fee": 0.0,
                }
            )
        return pd.DataFrame(rows)

    return _fn


def _order(asset, action, qty=10.0, price=1.0):
    return {"Asset": asset, "Action": action, "Adjusted Quantity": qty, "Price": price, "Executable": True}


def _drive_cycle(pipe, params, broker, orders, final_weights, run_id, asof="d", portfolio_value=1000.0):
    """Replicate run()'s recon wiring: preflight → execute → evaluate."""
    ctx = pipe._recon_load(params, run_id, asof)
    pre = pipe._recon_preflight(ctx, broker) if ctx is not None else {}
    applied = bool(pre.get("applied"))
    report = pipe._execute_orders(
        broker=broker,
        orders_df=pd.DataFrame(orders),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
        ledger=ctx.ledger if ctx is not None else None,
        cycle_id=ctx.cycle_id if ctx is not None else None,
        gate_orders_allowed=pre.get("orders_allowed", True) if applied else True,
        gate_reduce_only=pre.get("reduce_only", False) if applied else False,
    )
    notes = pipe._run_reconciliation(
        params=params,
        broker=broker,
        final_weights=final_weights,
        orders_df=pd.DataFrame(orders),
        execution_report=report,
        portfolio_value=portfolio_value,
        stable_coin="USDC",
        asof=asof,
        run_id=run_id,
        intent_captured=ctx is not None,
    )
    return report, notes, pre


def _ledger_records(tmp_path, book="carver-HL"):
    p = tmp_path / book / "orders.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# Deliverable 1: submission-time intent capture
# ---------------------------------------------------------------------------
def test_intent_is_on_disk_before_place_orders(tmp_path):
    """The INTENT record must be flushed BEFORE broker.place_orders is called —
    that ordering is what makes the ledger crash-durable."""
    seen_intent_at_submit = {}

    def _hook(orders):
        # When the broker is called, the intent for these orders must ALREADY be
        # on disk (record_intent ran + fsynced before place_orders).
        recs = _ledger_records(tmp_path)
        intents = [r for r in recs if r.get("kind") == "intent"]
        seen_intent_at_submit["n"] = len(intents)
        seen_intent_at_submit["symbols"] = {r["symbol"] for r in intents}
        # And NO result yet (place_orders hasn't returned).
        seen_intent_at_submit["results"] = [r for r in recs if r.get("kind") == "result"]

    broker = _FakeBroker(fills_fn=_fills_all("FILLED"), place_hook=_hook)
    params = {"reconciliation": {"book_key": "carver-HL", "data_dir": str(tmp_path), "mode": "observe"}}
    pipe = TradingPipeline()
    _drive_cycle(pipe, params, broker, [_order("DOGE", "Buy")], {"DOGE": 0.5}, run_id="r1")

    assert seen_intent_at_submit["n"] == 1
    assert seen_intent_at_submit["symbols"] == {"DOGE"}
    assert seen_intent_at_submit["results"] == []  # result only after the fill


def test_capture_records_intent_and_result_and_stage7b_reads_not_rewrites(tmp_path):
    """Stage 7 captures intent+result; Stage 7b must READ them back, not double-write."""
    broker = _FakeBroker(fills_fn=_fills_all("FILLED"))
    params = {"reconciliation": {"book_key": "carver-HL", "data_dir": str(tmp_path), "mode": "observe"}}
    pipe = TradingPipeline()
    _report, notes, _pre = _drive_cycle(pipe, params, broker, [_order("DOGE", "Buy")], {"DOGE": 0.5}, run_id="r1")

    recs = _ledger_records(tmp_path)
    intents = [r for r in recs if r.get("kind") == "intent"]
    results = [r for r in recs if r.get("kind") == "result"]
    # Exactly ONE intent + ONE result — no duplicate from a post-exec rewrite.
    assert len(intents) == 1
    assert len(results) == 1
    assert results[0]["status"] == "filled"
    assert notes["intent_captured_at_submission"] is True


def test_missed_fill_captured_as_timeout(tmp_path):
    """A submitted order the broker returns nothing for is closed as a timeout —
    the missed-fill class the ledger exists to prove."""
    broker = _FakeBroker(fills_fn=lambda orders: pd.DataFrame())  # zero fills
    params = {"reconciliation": {"book_key": "carver-HL", "data_dir": str(tmp_path), "mode": "observe"}}
    pipe = TradingPipeline()
    _report, notes, _pre = _drive_cycle(pipe, params, broker, [_order("DOGE", "Buy")], {"DOGE": 0.5}, run_id="r1")

    results = [r for r in _ledger_records(tmp_path) if r.get("kind") == "result"]
    assert len(results) == 1 and results[0]["status"] == "timeout"
    assert notes["missed_fills"] == ["DOGE"]


def test_replay_from_ledger_after_crash_before_stage7b(tmp_path):
    """CRASH-DURABILITY REPLAY: if the run dies after Stage 7 but before Stage 7b,
    the ledger alone still proves intent + outcome. Reconstruct the reconciliation
    view purely from the persisted JSONL and confirm the missed fill is provable."""
    from quantbox.reconciliation import OrderFillLedger

    # Stage 7 only: capture intent + (missing) result, then "crash" — no Stage 7b.
    broker = _FakeBroker(fills_fn=lambda orders: pd.DataFrame())  # order never fills
    ctx_params = {"reconciliation": {"book_key": "carver-HL", "data_dir": str(tmp_path), "mode": "observe"}}
    pipe = TradingPipeline()
    ctx = pipe._recon_load(ctx_params, run_id="r1", asof="d")
    pipe._execute_orders(
        broker=broker,
        orders_df=pd.DataFrame([_order("DOGE", "Buy"), _order("ETH", "Buy")]),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
        ledger=ctx.ledger,
        cycle_id=ctx.cycle_id,
    )
    # <<< process dies here — Stage 7b never runs >>>

    # Replay: a FRESH ledger object reading the same file recovers the full audit.
    replay = OrderFillLedger(book_key="carver-HL", root=str(tmp_path))
    joined = replay.match_intents_to_results()
    # Both orders have an intent; both resolved to a timeout (never filled) — the
    # missed-fill signature the exchange alone could not prove.
    assert len(joined) == 2
    for _ref, slot in joined.items():
        assert slot["intent"] is not None
        assert slot["result"] is not None and slot["result"]["status"] == "timeout"


# ---------------------------------------------------------------------------
# Deliverable 2: real, Tom-gated pre-execution enforcement
# ---------------------------------------------------------------------------
def _enforce_params(tmp_path, ack=True, **tol):
    cfg = {
        "book_key": "carver-HL",
        "data_dir": str(tmp_path),
        "mode": "enforce",
        "tolerances": tol,
    }
    if ack:
        cfg["enforce_acknowledged"] = True
    return {"reconciliation": cfg}


def test_enforce_without_ack_is_refused(tmp_path):
    """mode=enforce WITHOUT enforce_acknowledged is refused → forced to observe."""
    broker = _FakeBroker(fills_fn=_fills_all("FAILED"))
    params = _enforce_params(tmp_path, ack=False, halt_failed_streak=1)
    pipe = TradingPipeline()
    _report, notes, pre = _drive_cycle(pipe, params, broker, [_order("DOGE", "Buy")], {"DOGE": 0.5}, run_id="r1")
    assert notes["enforce_refused"] is True
    assert notes["mode"] == "observe"
    assert pre["enforced"] is False
    assert pre["applied"] is False  # never gates on an unacknowledged config


def test_enforce_halt_gates_next_cycle_before_execution(tmp_path):
    """Cycle 1 fails → persists HALT. Cycle 2 (enforce+ack) reads HALT back and
    blocks ALL orders BEFORE execution — broker.place_orders is never called."""
    params = _enforce_params(tmp_path, ack=True, halt_failed_streak=1)
    pipe = TradingPipeline()

    # Cycle 1: a failed order drives the machine to HALT (halt_failed_streak=1).
    b1 = _FakeBroker(fills_fn=_fills_all("FAILED"))
    _r1, n1, _p1 = _drive_cycle(pipe, params, b1, [_order("DOGE", "Buy")], {"DOGE": 0.5}, run_id="r1")
    assert n1["to_state"] == "halt"

    # Cycle 2: enforce reads persisted HALT and gates pre-execution.
    b2 = _FakeBroker(fills_fn=_fills_all("FILLED"))
    r2, _n2, p2 = _drive_cycle(pipe, params, b2, [_order("BTC", "Buy")], {"BTC": 0.5}, run_id="r2")
    assert p2["applied"] is True
    assert p2["orders_allowed"] is False
    assert r2.get("recon_gated") == "halt"
    assert b2.placed == []  # NOTHING was sent to the broker
    # Preflight HALT alert fired.
    assert any("HALT" in a for a in b2.alerts)


def test_enforce_flatten_keeps_only_reducing_orders(tmp_path):
    """A persisted FLATTEN makes the next cycle reduce-only: an opening BUY is
    dropped and a position-reducing SELL is kept (clamped)."""
    # A phantom holding (never intended) is a hard FLATTEN break. Drive cycle 1 to
    # FLATTEN, then check cycle 2's reduce-only gate.
    params = _enforce_params(tmp_path, ack=True)
    pipe = TradingPipeline()

    # Cycle 1: broker holds XRP we never intended → phantom → FLATTEN.
    b1 = _FakeBroker(positions=[{"symbol": "XRP", "qty": 100.0}], fills_fn=_fills_all("FILLED"))
    _r1, n1, _p1 = _drive_cycle(pipe, params, b1, [_order("DOGE", "Buy")], {"DOGE": 0.5}, run_id="r1")
    assert n1["to_state"] == "flatten"

    # Cycle 2: we hold LONG SOL (qty 5). Orders: BUY DOGE (opens → dropped),
    # SELL SOL (reduces → kept, clamped to holding).
    b2 = _FakeBroker(positions=[{"symbol": "SOL", "qty": 5.0}], fills_fn=_fills_all("FILLED"))
    r2, _n2, p2 = _drive_cycle(
        pipe,
        params,
        b2,
        [_order("DOGE", "Buy", qty=10.0), _order("SOL", "Sell", qty=3.0)],
        {"DOGE": 0.5},
        run_id="r2",
    )
    assert p2["applied"] is True and p2["reduce_only"] is True
    assert r2.get("recon_gated") == "flatten"
    # Only the reducing SOL sell reached the broker.
    assert len(b2.placed) == 1
    sent = b2.placed[0]
    assert set(sent["symbol"]) == {"SOL"}
    assert sent.iloc[0]["side"] == "sell"


def test_failed_intent_write_does_not_leave_a_dangling_result(tmp_path):
    """If record_intent raises (disk/permission), the order still sends but NO
    result is recorded against the never-written intent — the ledger must never
    hold a result with no matching intent (review BLOCKER)."""
    broker = _FakeBroker(fills_fn=_fills_all("FILLED"))
    params = {"reconciliation": {"book_key": "carver-HL", "data_dir": str(tmp_path), "mode": "observe"}}
    pipe = TradingPipeline()
    ctx = pipe._recon_load(params, run_id="r1", asof="d")

    # Make the FIRST intent write fail, the rest succeed.
    real_record_intent = ctx.ledger.record_intent
    calls = {"n": 0}

    def _flaky_intent(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return real_record_intent(**kw)

    ctx.ledger.record_intent = _flaky_intent

    report = pipe._execute_orders(
        broker=broker,
        orders_df=pd.DataFrame([_order("DOGE", "Buy"), _order("ETH", "Buy")]),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
        ledger=ctx.ledger,
        cycle_id=ctx.cycle_id,
    )
    # Both orders still executed (ledger never blocks execution).
    assert report["summary"]["total_executed"] == 2

    recs = _ledger_records(tmp_path)
    intents = {r["order_ref"] for r in recs if r.get("kind") == "intent"}
    results = [r for r in recs if r.get("kind") == "result"]
    # Every RESULT references an intent that was actually written — no dangling.
    assert intents  # ETH intent was written
    for r in results:
        assert r["order_ref"] in intents
    # The order whose intent write failed has neither intent nor result on disk.
    assert len(intents) == 1  # only ETH


def test_reduce_only_drops_unknown_action_fail_closed(tmp_path):
    """An unrecognised order action under a FLATTEN gate must be DROPPED, not
    treated as an exposure-reducing sell (review BLOCKER)."""
    params = _enforce_params(tmp_path, ack=True)
    pipe = TradingPipeline()

    # Cycle 1 → FLATTEN via a phantom holding.
    b1 = _FakeBroker(positions=[{"symbol": "XRP", "qty": 100.0}], fills_fn=_fills_all("FILLED"))
    _r1, n1, _p1 = _drive_cycle(pipe, params, b1, [_order("DOGE", "Buy")], {"DOGE": 0.5}, run_id="r1")
    assert n1["to_state"] == "flatten"

    # Cycle 2: hold LONG SOL; an order with a garbage action must be dropped even
    # though SOL is held (a naive "non-buy == sell" would have sent it).
    b2 = _FakeBroker(positions=[{"symbol": "SOL", "qty": 5.0}], fills_fn=_fills_all("FILLED"))
    r2, _n2, p2 = _drive_cycle(
        pipe,
        params,
        b2,
        [{"Asset": "SOL", "Action": "Rebalance", "Adjusted Quantity": 3.0, "Price": 1.0, "Executable": True}],
        {"SOL": 0.5},
        run_id="r2",
    )
    assert p2["reduce_only"] is True
    assert r2.get("recon_reduce_only_noop") is True  # nothing valid to reduce
    assert b2.placed == []  # the unknown-action order was NOT sent


def test_enforce_drops_order_when_intent_capture_fails(tmp_path):
    """Under enforce, an order whose intent write fails must NOT be submitted —
    we never trade live without the crash-durable intent record (review BLOCKER)."""
    params = _enforce_params(tmp_path, ack=True)
    pipe = TradingPipeline()
    ctx = pipe._recon_load(params, run_id="r1", asof="d")

    real = ctx.ledger.record_intent
    calls = {"n": 0}

    def _flaky(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return real(**kw)

    ctx.ledger.record_intent = _flaky
    broker = _FakeBroker(fills_fn=_fills_all("FILLED"))
    report = pipe._execute_orders(
        broker=broker,
        orders_df=pd.DataFrame([_order("DOGE", "Buy"), _order("ETH", "Buy")]),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
        ledger=ctx.ledger,
        cycle_id=ctx.cycle_id,
        capture_fail_closed=True,
    )
    # Only ONE order reached the broker — the un-captured one was dropped.
    assert len(broker.placed) == 1
    sent = broker.placed[0]
    assert set(sent["symbol"]) == {"ETH"}  # DOGE (first, failed capture) dropped
    assert report.get("recon_capture_dropped") == 1
    assert report["summary"]["total_failed"] == 1


def test_enforce_state_persist_failure_raises_and_alerts(tmp_path):
    """Under enforce, a total state-persist failure must fail LOUD (hard alert +
    error), not silently continue from stale state (review BLOCKER)."""
    import quantbox.plugins.pipeline.trading_pipeline as tp

    params = _enforce_params(tmp_path, ack=True, halt_failed_streak=1)
    pipe = TradingPipeline()
    broker = _FakeBroker(fills_fn=_fills_all("FAILED"))

    # Force the atomic write to always fail.
    orig = tp._atomic_write_text

    def _boom(path, text):
        raise OSError("read-only filesystem")

    tp._atomic_write_text = _boom
    try:
        _report, notes, _pre = _drive_cycle(pipe, params, broker, [_order("DOGE", "Buy")], {"DOGE": 0.5}, run_id="r1")
    finally:
        tp._atomic_write_text = orig

    # The wrapper caught the raised error and surfaced it (run stays alive).
    assert "error" in notes
    assert "persist failed under enforce" in notes["error"]
    # A hard operator alert fired.
    assert any("state persist FAILED" in a for a in broker.alerts)


def test_observe_state_persist_failure_is_non_fatal(tmp_path):
    """In observe mode a persist failure is a warning, not fatal, and does not
    alert as an enforcement failure."""
    import quantbox.plugins.pipeline.trading_pipeline as tp

    params = {"reconciliation": {"book_key": "carver-HL", "data_dir": str(tmp_path), "mode": "observe"}}
    pipe = TradingPipeline()
    broker = _FakeBroker(fills_fn=_fills_all("FILLED"))
    orig = tp._atomic_write_text
    tp._atomic_write_text = lambda p, t: (_ for _ in ()).throw(OSError("boom"))
    try:
        _report, notes, _pre = _drive_cycle(pipe, params, broker, [_order("DOGE", "Buy")], {"DOGE": 0.5}, run_id="r1")
    finally:
        tp._atomic_write_text = orig
    # No 'error' from a raise — observe swallows the persist failure.
    assert "error" not in notes
    assert not any("persist FAILED" in a for a in broker.alerts)


def test_corrupt_state_fails_closed_to_halt_under_enforce(tmp_path):
    """The persisted state IS the enforcement authority: a corrupt state file must
    fail CLOSED to HALT under enforce, never silently reset to NORMAL (review
    BLOCKER)."""
    book_dir = tmp_path / "carver-HL"
    book_dir.mkdir(parents=True)
    (book_dir / "recon_state.json").write_text("{ this is not valid json ")

    params = _enforce_params(tmp_path, ack=True)
    pipe = TradingPipeline()
    # Preflight must read the corrupt state as HALT and gate this cycle.
    ctx = pipe._recon_load(params, run_id="r1", asof="d")
    pre = pipe._recon_preflight(ctx, _FakeBroker())
    assert pre["applied"] is True
    assert pre["orders_allowed"] is False

    # And a full cycle sends nothing.
    b = _FakeBroker(fills_fn=_fills_all("FILLED"))
    r, _n, p = _drive_cycle(pipe, params, b, [_order("BTC", "Buy")], {"BTC": 0.5}, run_id="r1")
    assert p["orders_allowed"] is False
    assert r.get("recon_gated") == "halt"
    assert b.placed == []


def test_corrupt_state_in_observe_does_not_gate(tmp_path):
    """In observe mode a corrupt state is only an observability loss — it resets
    to NORMAL and never gates orders."""
    book_dir = tmp_path / "carver-HL"
    book_dir.mkdir(parents=True)
    (book_dir / "recon_state.json").write_text("garbage")

    params = {"reconciliation": {"book_key": "carver-HL", "data_dir": str(tmp_path), "mode": "observe"}}
    pipe = TradingPipeline()
    b = _FakeBroker(fills_fn=_fills_all("FILLED"))
    r, _n, p = _drive_cycle(pipe, params, b, [_order("BTC", "Buy")], {"BTC": 0.5}, run_id="r1")
    assert p["applied"] is False
    assert "recon_gated" not in r
    assert len(b.placed) == 1


def test_state_write_is_atomic_no_partial_file(tmp_path):
    """The state file is replaced atomically — after a persist it is always valid
    JSON with the expected keys (temp + fsync + os.replace)."""
    params = _enforce_params(tmp_path, ack=True, halt_failed_streak=1)
    pipe = TradingPipeline()
    b = _FakeBroker(fills_fn=_fills_all("FAILED"))
    _drive_cycle(pipe, params, b, [_order("DOGE", "Buy")], {"DOGE": 0.5}, run_id="r1")
    state = json.loads((tmp_path / "carver-HL" / "recon_state.json").read_text())
    assert state["state"] == "halt"
    assert "degraded_cycles" in state and "streaks" in state
    # No leftover temp files in the book dir.
    assert not list((tmp_path / "carver-HL").glob(".*tmp*"))


def test_preflight_exception_fails_closed_under_enforce(tmp_path):
    """If preflight itself raises under acknowledged enforce, run()'s guard must
    fail CLOSED. We assert the decision helper _enforce_requested drives that."""
    pipe = TradingPipeline()
    enforce_params = _enforce_params(tmp_path, ack=True)
    observe_params = {"reconciliation": {"book_key": "b", "data_dir": str(tmp_path), "mode": "observe"}}
    no_recon = {}
    enforce_no_ack = _enforce_params(tmp_path, ack=False)
    assert pipe._enforce_requested(enforce_params) is True
    assert pipe._enforce_requested(observe_params) is False
    assert pipe._enforce_requested(no_recon) is False
    assert pipe._enforce_requested(enforce_no_ack) is False


def test_observe_mode_never_applies_the_gate(tmp_path):
    """Even with a persisted HALT, OBSERVE mode never gates — orders flow."""
    # Cycle 1 (enforce+ack) → HALT persisted.
    params_enf = _enforce_params(tmp_path, ack=True, halt_failed_streak=1)
    pipe = TradingPipeline()
    b1 = _FakeBroker(fills_fn=_fills_all("FAILED"))
    _r1, n1, _p1 = _drive_cycle(pipe, params_enf, b1, [_order("DOGE", "Buy")], {"DOGE": 0.5}, run_id="r1")
    assert n1["to_state"] == "halt"

    # Cycle 2 in OBSERVE against the same book: gate computed but NOT applied.
    params_obs = {
        "reconciliation": {
            "book_key": "carver-HL",
            "data_dir": str(tmp_path),
            "mode": "observe",
            "tolerances": {"halt_failed_streak": 1},
        }
    }
    b2 = _FakeBroker(fills_fn=_fills_all("FILLED"))
    r2, _n2, p2 = _drive_cycle(pipe, params_obs, b2, [_order("BTC", "Buy")], {"BTC": 0.5}, run_id="r2")
    assert p2["applied"] is False
    assert "recon_gated" not in r2
    assert len(b2.placed) == 1  # order flowed normally
