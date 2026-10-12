"""The reconciliation cycle through its own interface: ``open_cycle`` then ``CycleRun.close`` (TOM-1455).

The cycle used to live in ``TradingPipeline``'s private methods, and its
config was loaded twice per run (once for the gate, once for the close) against
its own "built once" docstring. These tests pin what the package now owns: one
load per cycle, the gate failing CLOSED under acknowledged enforce and open in
observe, and a close that never fails the run except on an unwritable
enforcement authority.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantbox.trade.reconciliation import CycleRun, ReconCycle, ReconEnforcementError, open_cycle
from quantbox.trade.reconciliation import cycle as cycle_mod


class _Alerts:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    def __call__(self, message, *, site):
        self.sent.append((site, message))
        return True


class _Broker:
    def get_positions(self):
        return pd.DataFrame(columns=["symbol", "qty"])

    def get_market_snapshot(self, symbols):
        return pd.DataFrame({"symbol": list(symbols), "mid": [1.0] * len(symbols)})


def _observe(tmp_path, **extra):
    return {"reconciliation": {"book_key": "b", "data_dir": str(tmp_path), "mode": "observe", **extra}}


def _enforce(tmp_path, **extra):
    return {
        "reconciliation": {
            "book_key": "b",
            "data_dir": str(tmp_path),
            "mode": "enforce",
            "enforce_acknowledged": True,
            **extra,
        }
    }


def _close(run: CycleRun, alerts: _Alerts, report: dict | None = None) -> dict:
    return run.close(
        broker=_Broker(),
        final_weights={},
        orders_df=pd.DataFrame(),
        execution_report=report if report is not None else {"orders_details": [], "summary": {}},
        portfolio_value=1000.0,
        stable_coin="USDC",
        alert=alerts,
    )


def test_a_cycle_loads_its_config_once(monkeypatch, tmp_path):
    calls = []
    real = ReconCycle.load.__func__

    def _counting(cls, params, *, run_id, asof):
        calls.append(run_id)
        return real(cls, params, run_id=run_id, asof=asof)

    monkeypatch.setattr(ReconCycle, "load", classmethod(_counting))
    alerts = _Alerts()
    run = open_cycle(_observe(tmp_path), run_id="r1", asof="d", alert=alerts)
    notes = _close(run, alerts)
    assert calls == ["r1"]
    assert notes["book_key"] == "b"
    assert notes["intent_captured_at_submission"] is True  # armed: the execution stage had the ledger


def test_no_reconciliation_block_is_a_no_op(tmp_path):
    alerts = _Alerts()
    run = open_cycle({}, run_id="r1", asof="d", alert=alerts)
    assert run.ledger is None and run.cycle_id is None
    assert run.orders_allowed is True and run.reduce_only is False
    assert _close(run, alerts) == {}


def test_a_load_fault_under_acknowledged_enforce_fails_closed(tmp_path):
    alerts = _Alerts()
    run = open_cycle(_enforce(tmp_path, tolerances={"not_a_key": 1}), run_id="r1", asof="d", alert=alerts)
    assert run.orders_allowed is False  # HALT: no order this cycle
    assert run.preflight["fail_closed"] is True
    assert run.ledger is None  # nothing to capture to
    notes = _close(run, alerts, {"recon_gated": "halt", "orders_details": [], "summary": {}})
    assert "error" in notes  # the fault is on the record, not raised
    assert notes["preflight_gate"]["fail_closed"] is True
    assert notes["recon_gated"] == "halt"


def test_a_load_fault_in_observe_fails_open(tmp_path):
    alerts = _Alerts()
    run = open_cycle(_observe(tmp_path, tolerances={"not_a_key": 1}), run_id="r1", asof="d", alert=alerts)
    assert run.orders_allowed is True and run.preflight == {}
    assert "error" in _close(run, alerts)


def test_enforce_without_a_run_id_fails_the_run_at_close(tmp_path):
    """The gate fails closed, and the close re-raises: a run with no run-unique cycle id under enforce stops."""
    alerts = _Alerts()
    run = open_cycle(_enforce(tmp_path), run_id="", asof="d", alert=alerts)
    assert run.orders_allowed is False
    with pytest.raises(ReconEnforcementError, match="run-unique run_id"):
        _close(run, alerts)


def test_a_gate_fault_after_a_good_load_disarms_capture_and_the_close_reconstructs(monkeypatch, tmp_path):
    def _boom(self, *, alert):
        raise RuntimeError("gate fault")

    monkeypatch.setattr(ReconCycle, "gate", _boom)
    alerts = _Alerts()
    run = open_cycle(_observe(tmp_path), run_id="r1", asof="d", alert=alerts)
    assert run.loaded is not None and run.armed is False
    assert run.ledger is None  # the execution stage gets no ledger
    notes = _close(run, alerts)
    assert notes["intent_captured_at_submission"] is False


def test_an_unwritable_state_under_enforce_escapes_the_close_and_alerts(monkeypatch, tmp_path):
    def _boom(path, text):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(cycle_mod, "atomic_write_text", _boom)
    alerts = _Alerts()
    run = open_cycle(_enforce(tmp_path), run_id="r1", asof="d", alert=alerts)
    with pytest.raises(ReconEnforcementError, match="persist failed under enforce"):
        _close(run, alerts)
    assert [site for site, _ in alerts.sent] == ["recon_persist"]


def test_any_other_close_fault_is_on_the_record_not_raised(monkeypatch, tmp_path):
    def _boom(self, **kw):
        raise RuntimeError("ledger disk full")

    monkeypatch.setattr(ReconCycle, "close", _boom)
    alerts = _Alerts()
    run = open_cycle(_enforce(tmp_path), run_id="r1", asof="d", alert=alerts)
    notes = _close(run, alerts)
    assert notes["error"] == "ledger disk full"
    assert notes["preflight_gate"]["enforced"] is True


def test_the_gate_is_attached_to_the_close_notes(tmp_path):
    alerts = _Alerts()
    run = open_cycle(_observe(tmp_path), run_id="r1", asof="d", alert=alerts)
    notes = _close(run, alerts)
    assert notes["preflight_gate"] == {
        "gate_from_state": "normal",
        "orders_allowed": True,
        "reduce_only": False,
        "enforced": False,
        "applied": False,
    }
    assert notes["gate_applied"] is False
    assert "recon_gated" not in notes
