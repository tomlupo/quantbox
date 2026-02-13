"""Tests for quantbox.simulation.stress_testing — StressTestEngine."""

import numpy as np
import pandas as pd
import pytest

from quantbox.simulation.stress_testing import (
    HISTORICAL_SCENARIOS,
    HistoricalScenario,
    StressScenario,
    StressTestEngine,
    StressTestResult,
)


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    return pd.DataFrame(
        {
            "equity_fund": rng.normal(0.0003, 0.012, 500),
            "bond_fund": rng.normal(0.0001, 0.005, 500),
            "gold": rng.normal(0.0002, 0.010, 500),
        },
        index=dates,
    )


@pytest.fixture
def engine(sample_returns):
    weights = {"equity_fund": 0.6, "bond_fund": 0.3, "gold": 0.1}
    return StressTestEngine(returns=sample_returns, weights=weights)


class TestHistoricalScenarios:
    def test_all_scenarios_defined(self):
        assert len(HISTORICAL_SCENARIOS) == 8

    def test_scenario_values(self):
        crisis = HISTORICAL_SCENARIOS[HistoricalScenario.FINANCIAL_CRISIS_2008]
        assert crisis.equity_shock == -0.55
        assert crisis.duration_days == 350

    def test_covid_scenario(self):
        covid = HISTORICAL_SCENARIOS[HistoricalScenario.COVID_CRASH_2020]
        assert covid.equity_shock == -0.34
        assert covid.duration_days == 23


class TestStressTestEngine:
    def test_run_historical_scenario(self, engine):
        result = engine.run_historical_scenario(
            HistoricalScenario.COVID_CRASH_2020,
            n_simulations=200,
            random_state=42,
        )
        assert isinstance(result, StressTestResult)
        assert result.portfolio_impact < 0  # Should show loss
        assert result.var_95 < 0
        assert result.cvar_95 <= result.var_95

    def test_run_by_string(self, engine):
        result = engine.run_historical_scenario(
            "2008_financial_crisis",
            n_simulations=200,
            random_state=42,
        )
        assert result.scenario.name == "2008 Financial Crisis"

    def test_run_custom_scenario(self, engine):
        scenario = StressScenario(
            name="Custom Crash",
            description="Test scenario",
            equity_shock=-0.30,
            bond_shock=0.05,
            volatility_multiplier=3.0,
            correlation_stress=1.5,
            duration_days=20,
        )
        result = engine.run_custom_scenario(scenario, n_simulations=200, random_state=42)
        assert result.portfolio_impact < 0

    def test_asset_impacts(self, engine):
        result = engine.run_historical_scenario(
            HistoricalScenario.BLACK_MONDAY_1987,
            n_simulations=200,
            random_state=42,
        )
        assert "equity_fund" in result.asset_impacts
        assert "bond_fund" in result.asset_impacts

    def test_max_drawdown_negative(self, engine):
        result = engine.run_historical_scenario(
            HistoricalScenario.FINANCIAL_CRISIS_2008,
            n_simulations=200,
            random_state=42,
        )
        assert result.max_drawdown < 0

    def test_sensitivity_analysis(self, engine):
        shock_range = np.array([-0.1, -0.2, -0.3])
        results = engine.sensitivity_analysis(
            shock_variable="equity",
            shock_range=shock_range,
            n_simulations=100,
            random_state=42,
        )
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        assert "portfolio_impact" in results.columns
        # More severe shocks should produce worse impact
        assert results.iloc[2]["portfolio_impact"] < results.iloc[0]["portfolio_impact"]

    def test_sensitivity_volatility(self, engine):
        shock_range = np.array([1.0, 2.0, 4.0])
        results = engine.sensitivity_analysis(
            shock_variable="volatility",
            shock_range=shock_range,
            n_simulations=100,
            random_state=42,
        )
        assert len(results) == 3

    def test_compare_scenarios(self, engine):
        scenarios = [
            HistoricalScenario.COVID_CRASH_2020,
            HistoricalScenario.VOLMAGEDDON_2018,
        ]
        comparison = engine.compare_scenarios(scenarios, n_simulations=200)
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "portfolio_impact" in comparison.columns

    def test_var_calculation(self, engine):
        var_results = engine.var_calculation(
            confidence_levels=[0.95, 0.99],
            horizon_days=1,
            method="historical",
        )
        assert 0.95 in var_results
        assert 0.99 in var_results
        assert var_results[0.99] <= var_results[0.95]  # 99% VaR is more extreme

    def test_cvar_calculation(self, engine):
        cvar_results = engine.cvar_calculation(confidence_levels=[0.95])
        assert 0.95 in cvar_results

    def test_cvar_worse_than_var(self, engine):
        var = engine.var_calculation(confidence_levels=[0.95], method="historical")
        cvar = engine.cvar_calculation(confidence_levels=[0.95])
        assert cvar[0.95] <= var[0.95]


class TestStressScenario:
    def test_dataclass(self):
        s = StressScenario(
            name="Test",
            description="desc",
            equity_shock=-0.2,
            bond_shock=0.0,
            volatility_multiplier=2.0,
            correlation_stress=1.5,
            duration_days=30,
        )
        assert s.recovery_days == 0
        assert s.custom_shocks == {}


class TestReverseStressTest:
    def test_finds_shock_level(self, engine):
        shock_level, result = engine.reverse_stress_test(
            target_loss=-0.05,
            shock_variable="equity",
            tolerance=0.02,
            max_iterations=20,
            n_simulations=200,
        )
        assert shock_level < 0
        assert isinstance(result, StressTestResult)
