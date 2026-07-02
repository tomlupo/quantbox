"""Pipeline wiring for reconciliation (issue #87), OBSERVE-mode only.

Proves _run_reconciliation records the ledger + classifies breaks + alerts, and
that in observe mode it never reports a gating action (zero order-behavior
change). Also proves it is a no-op when no `reconciliation` config block exists.
"""

from __future__ import annotations

import pandas as pd

from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline


class _FakeBroker:
    """Broker exposing get_positions/get_market_snapshot/notify, like the live ones."""

    def __init__(self, positions):
        self._positions = positions
        self.alerts: list[str] = []

    def get_positions(self):
        return pd.DataFrame(self._positions)

    def get_market_snapshot(self, symbols):
        return pd.DataFrame({"symbol": symbols, "mid": [1.0] * len(symbols)})

    def notify(self, msg):
        self.alerts.append(msg)
        return True


def _orders_df():
    return pd.DataFrame(
        [
            {"Asset": "DOGE", "Action": "Buy", "Adjusted Quantity": 10.0, "Price": 1.0, "Executable": True},
        ]
    )


def _exec_report(status="FAILED"):
    return {
        "orders_details": [
            {"symbol": "DOGE", "side": "buy", "status": status, "executed_quantity": 0.0, "executed_price": 0.0}
        ],
        "summary": {},
    }


def test_duplicate_symbol_side_results_bind_one_to_one(tmp_path):
    """N same-(symbol,side) orders in a cycle must each bind to a DISTINCT result,
    not all to the first — else a fill is duplicated or a missed one is masked."""
    import json

    pipe = TradingPipeline()
    two_doge = pd.DataFrame(
        [
            {"Asset": "DOGE", "Action": "Buy", "Adjusted Quantity": 5.0, "Price": 1.0, "Executable": True},
            {"Asset": "DOGE", "Action": "Buy", "Adjusted Quantity": 5.0, "Price": 1.0, "Executable": True},
        ]
    )
    report = {
        "orders_details": [
            {"symbol": "DOGE", "side": "buy", "status": "FILLED", "executed_quantity": 5.0, "executed_price": 1.0},
            {"symbol": "DOGE", "side": "buy", "status": "FAILED", "executed_quantity": 0.0, "executed_price": 0.0},
        ],
        "summary": {},
    }
    params = {"reconciliation": {"book_key": "carver-HL", "data_dir": str(tmp_path), "mode": "observe"}}
    pipe._run_reconciliation(
        params=params,
        broker=_FakeBroker([]),
        final_weights={"DOGE": 0.5},
        orders_df=two_doge,
        execution_report=report,
        portfolio_value=1000.0,
        stable_coin="USDC",
        asof="d1",
        run_id="r",
    )
    results = [
        json.loads(x)
        for x in (tmp_path / "carver-HL" / "orders.jsonl").read_text().splitlines()
        if json.loads(x).get("kind") == "result"
    ]
    statuses = sorted(r["status"] for r in results)
    # Exactly one filled + one failed — not two filled (duplicate) or a phantom miss.
    assert statuses == ["failed", "filled"]


def test_no_config_is_noop(tmp_path):
    pipe = TradingPipeline()
    notes = pipe._run_reconciliation(
        params={},  # no `reconciliation` block
        broker=_FakeBroker([]),
        final_weights={"DOGE": 0.5},
        orders_df=_orders_df(),
        execution_report=_exec_report(),
        portfolio_value=1000.0,
        stable_coin="USDC",
        asof="2026-01-01",
        run_id="run1",
    )
    assert notes == {}


def test_observe_records_ledger_and_classifies_without_gating(tmp_path):
    pipe = TradingPipeline()
    broker = _FakeBroker(
        # DOGE tiny vs 50% target (huge drift) + ETH phantom (no intent).
        [{"symbol": "DOGE", "qty": 1.0}, {"symbol": "ETH", "qty": 100.0}]
    )
    params = {
        "reconciliation": {
            "book_key": "carver-HL",
            "data_dir": str(tmp_path),
            "mode": "observe",
        }
    }
    notes = pipe._run_reconciliation(
        params=params,
        broker=broker,
        final_weights={"DOGE": 0.5},
        orders_df=_orders_df(),
        execution_report=_exec_report(status="FAILED"),
        portfolio_value=1000.0,
        stable_coin="USDC",
        asof="2026-01-01",
        run_id="run1",
    )

    # Ledger written per-book.
    ledger_path = tmp_path / "carver-HL" / "orders.jsonl"
    assert ledger_path.exists()
    assert ledger_path.read_text().strip()  # has intent + result

    # Breaks classified (phantom ETH is hard) and would-be action is protective.
    assert notes["book_key"] == "carver-HL"
    assert notes["mode"] == "observe"
    assert notes["n_breaks"] >= 1
    assert notes["would_be_action"] in ("halt", "flatten")

    # OBSERVE guarantee: no gating.
    assert notes["enforced"] is False
    assert notes["orders_allowed"] is True
    assert notes["reduce_only"] is False

    # Alerted on the transition.
    assert broker.alerts and "OBSERVE" in broker.alerts[0]


def test_state_persists_across_cycles(tmp_path):
    pipe = TradingPipeline()
    broker = _FakeBroker([{"symbol": "DOGE", "qty": 1.0}])
    params = {
        "reconciliation": {
            "book_key": "carver-HL",
            "data_dir": str(tmp_path),
            "mode": "observe",
            "tolerances": {"halt_failed_streak": 3},
        }
    }
    # Three cycles of failed DOGE orders -> streak accrues in persisted state.
    for _ in range(3):
        notes = pipe._run_reconciliation(
            params=params,
            broker=broker,
            final_weights={"DOGE": 0.001},
            orders_df=_orders_df(),
            execution_report=_exec_report(status="FAILED"),
            portfolio_value=1000.0,
            stable_coin="USDC",
            asof="2026-01-01",
            run_id="run1",
        )
    assert (tmp_path / "carver-HL" / "recon_state.json").exists()
    # By cycle 3 the consecutive-failed streak is hard -> would HALT.
    assert notes["would_be_action"] == "halt"
    assert notes["orders_allowed"] is True  # still observe-only


def test_missed_fill_is_detected_and_counted(tmp_path):
    """Intent submitted but NO result observed = the missed-fill class the ledger
    exists to prove; it must feed the failure streak and be flagged in notes."""
    import json

    pipe = TradingPipeline()
    params = {"reconciliation": {"book_key": "carver-HL", "data_dir": str(tmp_path), "mode": "observe"}}
    notes = pipe._run_reconciliation(
        params=params,
        broker=_FakeBroker([]),
        final_weights={"DOGE": 0.5},
        orders_df=_orders_df(),
        execution_report={"orders_details": [], "summary": {}},  # broker returned nothing
        portfolio_value=1000.0,
        stable_coin="USDC",
        asof="2026-01-01",
        run_id="run1",
    )
    assert notes["missed_fills"] == ["DOGE"]
    # A timeout result was written to close the ledger match.
    lines = [json.loads(x) for x in (tmp_path / "carver-HL" / "orders.jsonl").read_text().splitlines()]
    assert any(r.get("kind") == "result" and r.get("status") == "timeout" for r in lines)


def test_stale_streak_does_not_escalate_when_symbol_untraded(tmp_path):
    """A symbol that fails once then is no longer traded must not keep a stale
    streak that falsely escalates to HALT (codex finding #3)."""
    pipe = TradingPipeline()
    params = {
        "reconciliation": {
            "book_key": "carver-HL",
            "data_dir": str(tmp_path),
            "mode": "observe",
            "tolerances": {"halt_failed_streak": 2},
        }
    }
    # Cycle 1: DOGE fails once.
    pipe._run_reconciliation(
        params=params,
        broker=_FakeBroker([]),
        final_weights={"DOGE": 0.5},
        orders_df=_orders_df(),
        execution_report=_exec_report("FAILED"),
        portfolio_value=1000.0,
        stable_coin="USDC",
        asof="d1",
        run_id="r",
    )
    # Cycles 2 & 3: DOGE no longer traded (empty order set) -> streak must drop.
    for _ in range(2):
        notes = pipe._run_reconciliation(
            params=params,
            broker=_FakeBroker([]),
            final_weights={},
            orders_df=pd.DataFrame(columns=["Asset", "Action", "Adjusted Quantity", "Price", "Executable"]),
            execution_report={"orders_details": [], "summary": {}},
            portfolio_value=1000.0,
            stable_coin="USDC",
            asof="d",
            run_id="r",
        )
    assert notes["would_be_action"] != "halt"


def test_recon_never_crashes_the_run(tmp_path):
    """A bad config must be caught by the guard, not propagate out of the run."""
    pipe = TradingPipeline()
    params = {
        "reconciliation": {
            "book_key": "carver-HL",
            "data_dir": str(tmp_path),
            "tolerances": {"not_a_real_key": 123},  # TypeError in BookTolerances
        }
    }
    notes = pipe._run_reconciliation(
        params=params,
        broker=_FakeBroker([]),
        final_weights={"DOGE": 0.5},
        orders_df=_orders_df(),
        execution_report=_exec_report("FAILED"),
        portfolio_value=1000.0,
        stable_coin="USDC",
        asof="d1",
        run_id="r",
    )
    assert "error" in notes  # swallowed, run survives


def test_enforce_mode_is_refused_no_false_safety(tmp_path):
    """mode='enforce' must NOT be honored in this post-execution path — it is
    forced back to observe and flagged, so no config can claim orders are gated
    when they have already been sent (independent-review BLOCKER)."""
    pipe = TradingPipeline()
    params = {
        "reconciliation": {
            "book_key": "carver-HL",
            "data_dir": str(tmp_path),
            "mode": "enforce",  # requested — must be refused here
        }
    }
    notes = pipe._run_reconciliation(
        params=params,
        broker=_FakeBroker([{"symbol": "DOGE", "qty": 1.0}, {"symbol": "ETH", "qty": 100.0}]),
        final_weights={"DOGE": 0.5},
        orders_df=_orders_df(),
        execution_report=_exec_report("FAILED"),
        portfolio_value=1000.0,
        stable_coin="USDC",
        asof="d1",
        run_id="r",
    )
    assert notes["enforce_refused"] is True
    assert notes["mode"] == "observe"
    assert notes["enforced"] is False
    # Even with hard breaks present, orders are never reported as gated.
    assert notes["orders_allowed"] is True
    assert notes["reduce_only"] is False


def test_carryover_position_is_not_phantom_but_never_intended_is(tmp_path):
    """Phantom must be defined by the LEDGER (intent), not final_weights.

    A position intended in a PRIOR cycle (legitimate carryover) is not phantom
    even when absent from this cycle's targets; a holding never intended is."""
    pipe = TradingPipeline()
    params = {"reconciliation": {"book_key": "carver-HL", "data_dir": str(tmp_path), "mode": "observe"}}

    # Cycle 1: we intend + trade ETH (records an ETH intent in the ledger).
    eth_orders = pd.DataFrame(
        [{"Asset": "ETH", "Action": "Buy", "Adjusted Quantity": 5.0, "Price": 1.0, "Executable": True}]
    )
    pipe._run_reconciliation(
        params=params,
        broker=_FakeBroker([]),
        final_weights={"ETH": 0.5},
        orders_df=eth_orders,
        execution_report={
            "orders_details": [
                {"symbol": "ETH", "side": "buy", "status": "FILLED", "executed_quantity": 5.0, "executed_price": 1.0}
            ],
            "summary": {},
        },
        portfolio_value=1000.0,
        stable_coin="USDC",
        asof="d1",
        run_id="r1",
    )

    # Cycle 2: target is only DOGE now; the book still holds ETH (carryover) and
    # an unexpected XRP that was NEVER intended.
    notes = pipe._run_reconciliation(
        params=params,
        broker=_FakeBroker([{"symbol": "ETH", "qty": 5.0}, {"symbol": "XRP", "qty": 50.0}]),
        final_weights={"DOGE": 0.5},
        orders_df=_orders_df(),
        execution_report=_exec_report("FILLED"),
        portfolio_value=1000.0,
        stable_coin="USDC",
        asof="d2",
        run_id="r2",
    )
    phantom_syms = {b["symbol"] for b in notes["breaks"] if b["class"] == "phantom_position"}
    assert "ETH" not in phantom_syms  # carryover with a prior intent → NOT phantom
    assert "XRP" in phantom_syms  # never intended → phantom


def _phantom_syms(notes) -> set[str]:
    return {b["symbol"] for b in notes.get("breaks", []) if b["class"] == "phantom_position"}


def test_phantom_uses_recent_intent_window_not_all_history(tmp_path):
    """A holding intended only OUTSIDE the recent window is phantom; a holding
    intended within it (carryover) is not. Round-1 used final_weights (reflagged
    every carryover); round-2 used ALL history (a once-traded name could never be
    phantom again). This pins the scoped fix (#88)."""
    pipe = TradingPipeline()
    dd = str(tmp_path)

    def _cycle(cycle_id, lookback, held, target, order_asset):
        return pipe._run_reconciliation(
            params={
                "reconciliation": {
                    "book_key": "b",
                    "data_dir": dd,
                    "cycle_id": cycle_id,
                    "mode": "observe",
                    "tolerances": {"phantom_lookback": lookback},
                }
            },
            broker=_FakeBroker([{"symbol": s, "qty": 10.0} for s in held]),
            final_weights=target,
            orders_df=pd.DataFrame(
                [{"Asset": order_asset, "Action": "Buy", "Adjusted Quantity": 1.0, "Price": 1.0, "Executable": True}]
            ),
            execution_report={
                "orders_details": [
                    {"symbol": order_asset, "side": "buy", "status": "FILLED", "executed_quantity": 1.0, "executed_price": 1.0}
                ],
                "summary": {},
            },
            portfolio_value=100.0,
            stable_coin="USDC",
            asof="2026-01-01",
            run_id=cycle_id,
        )

    # Cycle 'old': we intended DOGE (submitted an order). Not held yet.
    _cycle("old", lookback=1, held=[], target={"DOGE": 0.5}, order_asset="DOGE")
    # Cycle 'cur' (lookback=1 → recent window = {cur} only): DOGE is now HELD but
    # was intended only in 'old', outside the window → PHANTOM.
    cur = _cycle("cur", lookback=1, held=["DOGE"], target={"BTC": 0.5}, order_asset="BTC")
    assert "DOGE" in _phantom_syms(cur)

    # Fresh book: DOGE intended in the immediately-prior cycle, lookback=2 covers
    # it → the same held DOGE is explained carryover, NOT phantom.
    dd2 = str(tmp_path / "book2")
    pipe2 = TradingPipeline()

    def _cycle2(cycle_id, held, target, order_asset):
        return pipe2._run_reconciliation(
            params={"reconciliation": {"book_key": "b2", "data_dir": dd2, "cycle_id": cycle_id, "mode": "observe", "tolerances": {"phantom_lookback": 2}}},
            broker=_FakeBroker([{"symbol": s, "qty": 10.0} for s in held]),
            final_weights=target,
            orders_df=pd.DataFrame([{"Asset": order_asset, "Action": "Buy", "Adjusted Quantity": 1.0, "Price": 1.0, "Executable": True}]),
            execution_report={"orders_details": [{"symbol": order_asset, "side": "buy", "status": "FILLED", "executed_quantity": 1.0, "executed_price": 1.0}], "summary": {}},
            portfolio_value=100.0, stable_coin="USDC", asof="2026-01-01", run_id=cycle_id,
        )

    _cycle2("prev", held=[], target={"DOGE": 0.5}, order_asset="DOGE")
    cur2 = _cycle2("cur", held=["DOGE"], target={"BTC": 0.5}, order_asset="BTC")
    assert "DOGE" not in _phantom_syms(cur2)  # recent carryover, not phantom
