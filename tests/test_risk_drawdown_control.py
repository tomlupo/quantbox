"""Tests for DrawdownControlRiskManager plugin (risk.drawdown_control.v1).

Covers halt on severe drawdown, scale warning on moderate drawdown,
no findings when mild, check_orders delegation, and plugin metadata.
Self-contained.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantbox.plugins.risk.drawdown_control import DrawdownControlRiskManager


class TestDrawdownControlRiskManager:
    """Test suite for the DrawdownControlRiskManager risk plugin."""

    @pytest.fixture()
    def rm(self) -> DrawdownControlRiskManager:
        return DrawdownControlRiskManager()

    @staticmethod
    def _make_targets() -> pd.DataFrame:
        return pd.DataFrame([
            {"symbol": "BTC", "weight": 0.5},
            {"symbol": "ETH", "weight": 0.5},
        ])

    # ----------------------------------------------------------------
    # 1. Halt on severe drawdown
    # ----------------------------------------------------------------
    def test_halt_on_severe_drawdown(self, rm: DrawdownControlRiskManager) -> None:
        targets = self._make_targets()
        params = {
            "current_drawdown": -0.30,
            "max_drawdown": -0.20,
            "action": "halt",
        }
        findings = rm.check_targets(targets, params)

        halt_findings = [f for f in findings if f["rule"] == "drawdown_halt"]
        assert len(halt_findings) == 1
        assert halt_findings[0]["level"] == "error"

    # ----------------------------------------------------------------
    # 2. Warn on drawdown breach (default action)
    # ----------------------------------------------------------------
    def test_warn_on_drawdown_breach(self, rm: DrawdownControlRiskManager) -> None:
        targets = self._make_targets()
        params = {
            "current_drawdown": -0.25,
            "max_drawdown": -0.20,
        }
        findings = rm.check_targets(targets, params)

        warn_findings = [f for f in findings if f["rule"] == "drawdown_halt"]
        assert len(warn_findings) == 1
        assert warn_findings[0]["level"] == "warn"

    # ----------------------------------------------------------------
    # 3. Scale warning on moderate drawdown
    # ----------------------------------------------------------------
    def test_scale_warning_on_moderate_drawdown(self, rm: DrawdownControlRiskManager) -> None:
        targets = self._make_targets()
        params = {
            "current_drawdown": -0.15,
            "max_drawdown": -0.20,
            "scale_threshold": -0.10,
            "scale_factor": 0.5,
        }
        findings = rm.check_targets(targets, params)

        scale_findings = [f for f in findings if f["rule"] == "drawdown_scale"]
        assert len(scale_findings) == 1
        assert scale_findings[0]["level"] == "info"
        assert "0.5" in scale_findings[0]["detail"]

    # ----------------------------------------------------------------
    # 4. No findings when drawdown is mild
    # ----------------------------------------------------------------
    def test_no_findings_mild_drawdown(self, rm: DrawdownControlRiskManager) -> None:
        targets = self._make_targets()
        params = {
            "current_drawdown": -0.05,
            "max_drawdown": -0.20,
        }
        findings = rm.check_targets(targets, params)
        assert findings == []

    # ----------------------------------------------------------------
    # 5. Scale threshold defaults to max_drawdown
    # ----------------------------------------------------------------
    def test_scale_threshold_defaults_to_max_drawdown(self, rm: DrawdownControlRiskManager) -> None:
        targets = self._make_targets()
        # current_drawdown = -0.25 is worse than max_drawdown = -0.20
        # scale_threshold defaults to max_drawdown = -0.20
        # Since current is worse than max_drawdown, we get halt/warn but also
        # scale check: current_drawdown < scale_threshold? -0.25 < -0.20 = yes
        # But halted takes precedence when action=halt
        params = {
            "current_drawdown": -0.25,
            "max_drawdown": -0.20,
            "action": "halt",
        }
        findings = rm.check_targets(targets, params)

        halt_findings = [f for f in findings if f["rule"] == "drawdown_halt"]
        assert len(halt_findings) == 1
        # No scale finding when halted
        scale_findings = [f for f in findings if f["rule"] == "drawdown_scale"]
        assert len(scale_findings) == 0

    # ----------------------------------------------------------------
    # 6. Scale without halt: drawdown breaches but action is warn
    # ----------------------------------------------------------------
    def test_scale_and_warn_together(self, rm: DrawdownControlRiskManager) -> None:
        targets = self._make_targets()
        params = {
            "current_drawdown": -0.25,
            "max_drawdown": -0.20,
            "scale_threshold": -0.15,
            "scale_factor": 0.3,
            "action": "warn",
        }
        findings = rm.check_targets(targets, params)

        halt_findings = [f for f in findings if f["rule"] == "drawdown_halt"]
        scale_findings = [f for f in findings if f["rule"] == "drawdown_scale"]
        assert len(halt_findings) == 1
        assert halt_findings[0]["level"] == "warn"
        assert len(scale_findings) == 1
        assert "0.3" in scale_findings[0]["detail"]

    # ----------------------------------------------------------------
    # 7. check_orders returns empty
    # ----------------------------------------------------------------
    def test_check_orders_returns_empty(self, rm: DrawdownControlRiskManager) -> None:
        orders = pd.DataFrame([{"symbol": "BTC", "side": "buy", "qty": 1, "price": 50000}])
        findings = rm.check_orders(orders, {})
        assert findings == []

    # ----------------------------------------------------------------
    # 8. Missing current_drawdown returns empty
    # ----------------------------------------------------------------
    def test_missing_current_drawdown(self, rm: DrawdownControlRiskManager) -> None:
        targets = self._make_targets()
        findings = rm.check_targets(targets, {"max_drawdown": -0.20})
        assert findings == []

    # ----------------------------------------------------------------
    # 9. Default params
    # ----------------------------------------------------------------
    def test_default_max_drawdown(self, rm: DrawdownControlRiskManager) -> None:
        targets = self._make_targets()
        params = {"current_drawdown": -0.25}
        findings = rm.check_targets(targets, params)

        halt_findings = [f for f in findings if f["rule"] == "drawdown_halt"]
        assert len(halt_findings) == 1

    # ----------------------------------------------------------------
    # 10. Meta correct
    # ----------------------------------------------------------------
    def test_meta_name(self) -> None:
        assert DrawdownControlRiskManager.meta.name == "risk.drawdown_control.v1"

    def test_meta_kind(self) -> None:
        assert DrawdownControlRiskManager.meta.kind == "risk"

    def test_meta_version(self) -> None:
        assert DrawdownControlRiskManager.meta.version == "0.1.0"
