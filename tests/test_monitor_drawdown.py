"""Tests for DrawdownMonitor plugin (monitor.drawdown.v1).

Covers drawdown threshold alerts, halt action escalation, max_loss checks,
within-limits scenarios, and plugin metadata. Self-contained.
"""

from __future__ import annotations

import pytest

from quantbox.contracts import PluginMeta, RunResult

from quantbox.plugins.monitor.drawdown import DrawdownMonitor


def _make_result(max_drawdown: float = -0.10, total_return: float = 0.05) -> RunResult:
    """Build a RunResult with the given metrics."""
    return RunResult(
        run_id="run-001",
        pipeline_name="test.pipe.v1",
        mode="backtest",
        asof="2026-01-01",
        artifacts={},
        metrics={"max_drawdown": max_drawdown, "total_return": total_return},
        notes={},
    )


class TestDrawdownMonitor:
    """Test suite for the DrawdownMonitor monitor plugin."""

    @pytest.fixture()
    def monitor(self) -> DrawdownMonitor:
        return DrawdownMonitor()

    # ----------------------------------------------------------------
    # 1. Alert triggered when drawdown exceeds threshold
    # ----------------------------------------------------------------
    def test_drawdown_exceeds_threshold(self, monitor: DrawdownMonitor) -> None:
        result = _make_result(max_drawdown=-0.25)
        alerts = monitor.check(result, history=None, params={"max_drawdown": -0.20})

        dd_alerts = [a for a in alerts if a["rule"] == "max_drawdown_exceeded"]
        assert len(dd_alerts) == 1
        assert dd_alerts[0]["level"] == "warn"
        assert "-0.25" in dd_alerts[0]["detail"] or "-25" in dd_alerts[0]["detail"]

    # ----------------------------------------------------------------
    # 2. Halt action produces error level
    # ----------------------------------------------------------------
    def test_halt_action_produces_error(self, monitor: DrawdownMonitor) -> None:
        result = _make_result(max_drawdown=-0.25)
        alerts = monitor.check(
            result, history=None, params={"max_drawdown": -0.20, "action": "halt"}
        )

        dd_alerts = [a for a in alerts if a["rule"] == "max_drawdown_exceeded"]
        assert len(dd_alerts) == 1
        assert dd_alerts[0]["level"] == "error"
        assert dd_alerts[0]["action"] == "halt"

    # ----------------------------------------------------------------
    # 3. No alert when within limits
    # ----------------------------------------------------------------
    def test_no_alert_within_limits(self, monitor: DrawdownMonitor) -> None:
        result = _make_result(max_drawdown=-0.10, total_return=0.05)
        alerts = monitor.check(
            result, history=None, params={"max_drawdown": -0.20, "max_loss": -0.30}
        )
        assert alerts == []

    # ----------------------------------------------------------------
    # 4. Max loss alert triggered
    # ----------------------------------------------------------------
    def test_max_loss_exceeded(self, monitor: DrawdownMonitor) -> None:
        result = _make_result(max_drawdown=-0.10, total_return=-0.35)
        alerts = monitor.check(
            result, history=None, params={"max_drawdown": -0.05, "max_loss": -0.30}
        )

        loss_alerts = [a for a in alerts if a["rule"] == "max_loss_exceeded"]
        assert len(loss_alerts) == 1
        assert loss_alerts[0]["level"] == "warn"

    # ----------------------------------------------------------------
    # 5. Max loss with halt action
    # ----------------------------------------------------------------
    def test_max_loss_halt(self, monitor: DrawdownMonitor) -> None:
        result = _make_result(max_drawdown=-0.10, total_return=-0.35)
        alerts = monitor.check(
            result,
            history=None,
            params={"max_drawdown": -0.05, "max_loss": -0.30, "action": "halt"},
        )

        loss_alerts = [a for a in alerts if a["rule"] == "max_loss_exceeded"]
        assert len(loss_alerts) == 1
        assert loss_alerts[0]["level"] == "error"

    # ----------------------------------------------------------------
    # 6. Default params used when not specified
    # ----------------------------------------------------------------
    def test_defaults_no_alert(self, monitor: DrawdownMonitor) -> None:
        result = _make_result(max_drawdown=-0.15, total_return=0.0)
        alerts = monitor.check(result, history=None, params={})
        assert alerts == []

    # ----------------------------------------------------------------
    # 7. Missing metrics gracefully handled
    # ----------------------------------------------------------------
    def test_missing_metrics_no_alert(self, monitor: DrawdownMonitor) -> None:
        result = RunResult(
            run_id="run-002",
            pipeline_name="test.pipe.v1",
            mode="backtest",
            asof="2026-01-01",
            artifacts={},
            metrics={},
            notes={},
        )
        alerts = monitor.check(result, history=None, params={})
        assert alerts == []

    # ----------------------------------------------------------------
    # 8. Meta name and kind correct
    # ----------------------------------------------------------------
    def test_meta_name(self) -> None:
        assert DrawdownMonitor.meta.name == "monitor.drawdown.v1"

    def test_meta_kind(self) -> None:
        assert DrawdownMonitor.meta.kind == "monitor"

    def test_meta_version(self) -> None:
        assert DrawdownMonitor.meta.version == "0.1.0"
