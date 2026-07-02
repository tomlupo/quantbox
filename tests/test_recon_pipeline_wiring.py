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
