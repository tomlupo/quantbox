"""Tests for risk plugin discovery via builtins registry.

Verifies that all four risk plugins are registered and discoverable.
"""

from __future__ import annotations

from quantbox.plugins.builtins import builtins
from quantbox.plugins.risk.drawdown_control import DrawdownControlRiskManager
from quantbox.plugins.risk.factor_exposure import FactorExposureRiskManager
from quantbox.plugins.risk.stress_test_risk import StressTestRiskManager
from quantbox.plugins.risk.trading_risk import TradingRiskManager


class TestRiskDiscovery:
    """Verify all risk plugins appear in the builtins registry."""

    def test_risk_section_exists(self) -> None:
        registry = builtins()
        assert "risk" in registry

    def test_trading_basic_registered(self) -> None:
        registry = builtins()
        assert "risk.trading_basic.v1" in registry["risk"]
        assert registry["risk"]["risk.trading_basic.v1"] is TradingRiskManager

    def test_stress_test_registered(self) -> None:
        registry = builtins()
        assert "risk.stress_test.v1" in registry["risk"]
        assert registry["risk"]["risk.stress_test.v1"] is StressTestRiskManager

    def test_factor_exposure_registered(self) -> None:
        registry = builtins()
        assert "risk.factor_exposure.v1" in registry["risk"]
        assert registry["risk"]["risk.factor_exposure.v1"] is FactorExposureRiskManager

    def test_drawdown_control_registered(self) -> None:
        registry = builtins()
        assert "risk.drawdown_control.v1" in registry["risk"]
        assert registry["risk"]["risk.drawdown_control.v1"] is DrawdownControlRiskManager

    def test_risk_count(self) -> None:
        registry = builtins()
        assert len(registry["risk"]) == 4

    def test_risk_init_exports(self) -> None:
        from quantbox.plugins.risk import (
            DrawdownControlRiskManager as DDC,
            FactorExposureRiskManager as FE,
            StressTestRiskManager as ST,
            TradingRiskManager as TR,
        )

        assert TR is TradingRiskManager
        assert ST is StressTestRiskManager
        assert FE is FactorExposureRiskManager
        assert DDC is DrawdownControlRiskManager
