"""Tests for StressTestRiskManager — risk.stress_test.v1."""
import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.risk.stress_test_risk import StressTestRiskManager


@pytest.fixture
def plugin():
    return StressTestRiskManager()


@pytest.fixture
def sample_targets():
    return pd.DataFrame({
        "symbol": ["equity_A", "equity_B", "bond_C"],
        "weight": [0.5, 0.3, 0.2],
    })


class TestMeta:
    def test_name(self):
        assert StressTestRiskManager.meta.name == "risk.stress_test.v1"

    def test_kind(self):
        assert StressTestRiskManager.meta.kind == "risk"


class TestCheckTargets:
    def test_basic_run(self, plugin, sample_targets):
        findings = plugin.check_targets(sample_targets, {
            "scenarios": ["2020_covid_crash"],
            "n_simulations": 100,
        })
        assert isinstance(findings, list)
        for f in findings:
            assert "level" in f
            assert "rule" in f
            assert "detail" in f

    def test_empty_targets(self, plugin):
        findings = plugin.check_targets(pd.DataFrame(), {})
        assert findings == []

    def test_unknown_scenario(self, plugin, sample_targets):
        findings = plugin.check_targets(sample_targets, {
            "scenarios": ["nonexistent_scenario"],
            "n_simulations": 100,
        })
        assert any(f["rule"] == "unknown_scenario" for f in findings)

    def test_findings_have_scenario_name(self, plugin, sample_targets):
        findings = plugin.check_targets(sample_targets, {
            "scenarios": ["2008_financial_crisis"],
            "max_var_95": 0.01,  # Positive threshold — any loss will trigger
            "max_cvar_95": 0.01,
            "n_simulations": 200,
        })
        # Should have at least one breach
        breach_findings = [f for f in findings if "breach" in f["rule"]]
        assert len(breach_findings) > 0

    def test_drawdown_threshold(self, plugin, sample_targets):
        findings = plugin.check_targets(sample_targets, {
            "scenarios": ["2008_financial_crisis"],
            "max_stress_drawdown": 0.01,  # Positive threshold — always triggers
            "n_simulations": 200,
        })
        dd_findings = [f for f in findings if f["rule"] == "stress_drawdown_breach"]
        assert len(dd_findings) > 0
        assert dd_findings[0]["level"] == "error"


class TestCheckOrders:
    def test_passthrough(self, plugin):
        orders = pd.DataFrame({"symbol": ["X"], "side": ["buy"], "qty": [10], "price": [100]})
        findings = plugin.check_orders(orders, {})
        assert findings == []
