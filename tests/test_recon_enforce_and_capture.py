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
