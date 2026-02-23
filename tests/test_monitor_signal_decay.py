"""Tests for SignalDecayMonitor plugin (monitor.signal_decay.v1).

Covers declining Sharpe detection, healthy series, short history,
no history, and plugin metadata. Self-contained.
"""

from __future__ import annotations

import pytest

from quantbox.contracts import RunResult

from quantbox.plugins.monitor.signal_decay import SignalDecayMonitor


def _make_result(sharpe: float, run_id: str = "run-001") -> RunResult:
    """Build a RunResult with a given Sharpe metric."""
    return RunResult(
        run_id=run_id,
        pipeline_name="test.pipe.v1",
        mode="backtest",
        asof="2026-01-01",
        artifacts={},
        metrics={"sharpe": sharpe},
        notes={},
    )


class TestSignalDecayMonitor:
    """Test suite for the SignalDecayMonitor monitor plugin."""

    @pytest.fixture()
    def monitor(self) -> SignalDecayMonitor:
        return SignalDecayMonitor()

    # ----------------------------------------------------------------
    # 1. Declining Sharpe series triggers alert
    # ----------------------------------------------------------------
    def test_declining_sharpe_triggers_alert(self, monitor: SignalDecayMonitor) -> None:
        history = [_make_result(sharpe=s, run_id=f"run-{i}") for i, s in enumerate([0.3, 0.2, 0.1, 0.2, 0.1])]
        result = _make_result(sharpe=0.1)

        alerts = monitor.check(result, history=history, params={"min_sharpe": 0.5, "window": 5})

        decay_alerts = [a for a in alerts if a["rule"] == "signal_decay"]
        assert len(decay_alerts) == 1
        assert decay_alerts[0]["level"] == "warn"

    # ----------------------------------------------------------------
    # 2. Healthy Sharpe series: no alert
    # ----------------------------------------------------------------
    def test_healthy_sharpe_no_alert(self, monitor: SignalDecayMonitor) -> None:
        history = [_make_result(sharpe=s, run_id=f"run-{i}") for i, s in enumerate([1.2, 1.3, 1.1, 1.4, 1.0])]
        result = _make_result(sharpe=1.2)

        alerts = monitor.check(result, history=history, params={"min_sharpe": 0.5, "window": 5})
        assert alerts == []

    # ----------------------------------------------------------------
    # 3. No history returns empty
    # ----------------------------------------------------------------
    def test_no_history_returns_empty(self, monitor: SignalDecayMonitor) -> None:
        result = _make_result(sharpe=0.1)
        alerts = monitor.check(result, history=None, params={"min_sharpe": 0.5, "window": 5})
        assert alerts == []

    # ----------------------------------------------------------------
    # 4. History too short returns empty
    # ----------------------------------------------------------------
    def test_short_history_returns_empty(self, monitor: SignalDecayMonitor) -> None:
        history = [_make_result(sharpe=0.1, run_id="run-0")]
        result = _make_result(sharpe=0.1)

        alerts = monitor.check(result, history=history, params={"min_sharpe": 0.5, "window": 5})
        assert alerts == []

    # ----------------------------------------------------------------
    # 5. Uses last `window` entries from history
    # ----------------------------------------------------------------
    def test_uses_last_window_entries(self, monitor: SignalDecayMonitor) -> None:
        # First 5 entries are poor, last 3 are excellent
        history = [
            _make_result(sharpe=0.1, run_id="run-0"),
            _make_result(sharpe=0.1, run_id="run-1"),
            _make_result(sharpe=0.1, run_id="run-2"),
            _make_result(sharpe=0.1, run_id="run-3"),
            _make_result(sharpe=0.1, run_id="run-4"),
            _make_result(sharpe=2.0, run_id="run-5"),
            _make_result(sharpe=2.0, run_id="run-6"),
            _make_result(sharpe=2.0, run_id="run-7"),
        ]
        result = _make_result(sharpe=2.0)

        alerts = monitor.check(result, history=history, params={"min_sharpe": 0.5, "window": 3})
        assert alerts == []

    # ----------------------------------------------------------------
    # 6. Default params work
    # ----------------------------------------------------------------
    def test_default_params(self, monitor: SignalDecayMonitor) -> None:
        # Default: min_sharpe=0.5, window=5
        history = [_make_result(sharpe=0.1, run_id=f"run-{i}") for i in range(5)]
        result = _make_result(sharpe=0.1)

        alerts = monitor.check(result, history=history, params={})
        decay_alerts = [a for a in alerts if a["rule"] == "signal_decay"]
        assert len(decay_alerts) == 1

    # ----------------------------------------------------------------
    # 7. History entries missing sharpe metric are skipped
    # ----------------------------------------------------------------
    def test_missing_sharpe_entries_skipped(self, monitor: SignalDecayMonitor) -> None:
        good = _make_result(sharpe=1.0, run_id="run-good")
        no_sharpe = RunResult(
            run_id="run-no-sharpe",
            pipeline_name="test.pipe.v1",
            mode="backtest",
            asof="2026-01-01",
            artifacts={},
            metrics={},
            notes={},
        )
        # 3 good + 2 missing = only 3 usable, less than window=5
        history = [good, no_sharpe, good, no_sharpe, good]
        result = _make_result(sharpe=1.0)

        alerts = monitor.check(result, history=history, params={"min_sharpe": 0.5, "window": 5})
        assert alerts == []

    # ----------------------------------------------------------------
    # 8. Meta correct
    # ----------------------------------------------------------------
    def test_meta_name(self) -> None:
        assert SignalDecayMonitor.meta.name == "monitor.signal_decay.v1"

    def test_meta_kind(self) -> None:
        assert SignalDecayMonitor.meta.kind == "monitor"

    def test_meta_version(self) -> None:
        assert SignalDecayMonitor.meta.version == "0.1.0"
