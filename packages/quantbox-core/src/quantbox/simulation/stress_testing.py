"""Stress testing scenarios and risk analysis.

Provides historical and custom stress scenarios, sensitivity analysis,
reverse stress testing, and VaR/CVaR calculations.

Ported from quantlabnew/src/market-simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class HistoricalScenario(Enum):
    """Pre-defined historical stress scenarios."""

    FINANCIAL_CRISIS_2008 = "2008_financial_crisis"
    COVID_CRASH_2020 = "2020_covid_crash"
    DOT_COM_BUST_2000 = "2000_dotcom_bust"
    BLACK_MONDAY_1987 = "1987_black_monday"
    EURO_CRISIS_2011 = "2011_euro_crisis"
    TAPER_TANTRUM_2013 = "2013_taper_tantrum"
    FLASH_CRASH_2010 = "2010_flash_crash"
    VOLMAGEDDON_2018 = "2018_volmageddon"


@dataclass
class StressScenario:
    """Definition of a stress scenario."""

    name: str
    description: str
    equity_shock: float
    bond_shock: float
    volatility_multiplier: float
    correlation_stress: float
    duration_days: int
    recovery_days: int = 0
    custom_shocks: dict = field(default_factory=dict)


HISTORICAL_SCENARIOS = {
    HistoricalScenario.FINANCIAL_CRISIS_2008: StressScenario(
        name="2008 Financial Crisis",
        description="Lehman Brothers collapse and global financial crisis",
        equity_shock=-0.55,
        bond_shock=0.10,
        volatility_multiplier=4.0,
        correlation_stress=1.8,
        duration_days=350,
        recovery_days=700,
    ),
    HistoricalScenario.COVID_CRASH_2020: StressScenario(
        name="COVID-19 Crash",
        description="Pandemic-induced market crash",
        equity_shock=-0.34,
        bond_shock=0.05,
        volatility_multiplier=5.0,
        correlation_stress=2.0,
        duration_days=23,
        recovery_days=140,
    ),
    HistoricalScenario.DOT_COM_BUST_2000: StressScenario(
        name="Dot-Com Bust",
        description="Technology bubble burst",
        equity_shock=-0.49,
        bond_shock=0.15,
        volatility_multiplier=2.0,
        correlation_stress=1.3,
        duration_days=650,
        recovery_days=1500,
    ),
    HistoricalScenario.BLACK_MONDAY_1987: StressScenario(
        name="Black Monday 1987",
        description="Single-day market crash",
        equity_shock=-0.22,
        bond_shock=0.03,
        volatility_multiplier=6.0,
        correlation_stress=2.5,
        duration_days=1,
        recovery_days=400,
    ),
    HistoricalScenario.EURO_CRISIS_2011: StressScenario(
        name="European Debt Crisis",
        description="Sovereign debt crisis in Europe",
        equity_shock=-0.20,
        bond_shock=-0.05,
        volatility_multiplier=2.5,
        correlation_stress=1.5,
        duration_days=180,
        recovery_days=300,
    ),
    HistoricalScenario.TAPER_TANTRUM_2013: StressScenario(
        name="Taper Tantrum",
        description="Fed tapering announcement shock",
        equity_shock=-0.06,
        bond_shock=-0.08,
        volatility_multiplier=1.5,
        correlation_stress=1.2,
        duration_days=60,
        recovery_days=90,
    ),
    HistoricalScenario.FLASH_CRASH_2010: StressScenario(
        name="Flash Crash 2010",
        description="Rapid intraday market crash",
        equity_shock=-0.09,
        bond_shock=0.02,
        volatility_multiplier=3.0,
        correlation_stress=1.8,
        duration_days=1,
        recovery_days=1,
    ),
    HistoricalScenario.VOLMAGEDDON_2018: StressScenario(
        name="Volmageddon",
        description="February 2018 volatility spike",
        equity_shock=-0.10,
        bond_shock=-0.02,
        volatility_multiplier=4.0,
        correlation_stress=1.6,
        duration_days=10,
        recovery_days=60,
    ),
}


@dataclass
class StressTestResult:
    """Results from stress testing."""

    scenario: StressScenario
    portfolio_impact: float
    asset_impacts: dict[str, float]
    var_95: float
    var_99: float
    cvar_95: float
    max_drawdown: float
    time_to_recovery: int | None = None
    stressed_prices: np.ndarray | None = None
    stressed_returns: np.ndarray | None = None


class StressTestEngine:
    """Engine for stress testing portfolios."""

    def __init__(
        self,
        returns: pd.DataFrame | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.returns = returns
        self.weights = weights or {}
        self.asset_names = list(returns.columns) if returns is not None else []

    def run_historical_scenario(
        self,
        scenario: HistoricalScenario | str,
        n_simulations: int = 1000,
        random_state: int | None = None,
    ) -> StressTestResult:
        if isinstance(scenario, str):
            scenario = HistoricalScenario(scenario)
        stress_def = HISTORICAL_SCENARIOS[scenario]
        return self.run_custom_scenario(stress_def, n_simulations, random_state)

    def run_custom_scenario(
        self,
        scenario: StressScenario,
        n_simulations: int = 1000,
        random_state: int | None = None,
    ) -> StressTestResult:
        rng = np.random.default_rng(random_state)
        n_assets = len(self.asset_names)
        n_steps = scenario.duration_days

        if self.returns is not None:
            base_stds = self.returns.std().values
            base_corr = self.returns.corr().values
        else:
            base_stds = np.ones(n_assets) * 0.01
            base_corr = np.eye(n_assets)

        stressed_stds = base_stds * scenario.volatility_multiplier
        stressed_corr = self._stress_correlation_matrix(base_corr, scenario.correlation_stress)
        L = np.linalg.cholesky(stressed_corr)

        Z = rng.standard_normal((n_simulations, n_steps, n_assets))
        stressed_returns = np.zeros((n_simulations, n_steps, n_assets))

        for t in range(n_steps):
            correlated_shocks = Z[:, t, :] @ L.T
            stressed_returns[:, t, :] = stressed_stds * correlated_shocks

        for i, asset in enumerate(self.asset_names):
            if asset in scenario.custom_shocks:
                shock = scenario.custom_shocks[asset]
            elif "equity" in asset.lower() or "stock" in asset.lower():
                shock = scenario.equity_shock
            elif "bond" in asset.lower() or "fixed" in asset.lower():
                shock = scenario.bond_shock
            else:
                shock = scenario.equity_shock
            daily_shock = shock / n_steps
            stressed_returns[:, :, i] += daily_shock

        weights_arr = np.array([self.weights.get(a, 1 / n_assets) for a in self.asset_names])
        portfolio_returns = (stressed_returns * weights_arr).sum(axis=2)
        cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1) - 1
        terminal_returns = cumulative_returns[:, -1]

        var_95 = float(np.percentile(terminal_returns, 5))
        var_99 = float(np.percentile(terminal_returns, 1))
        tail = terminal_returns[terminal_returns <= var_95]
        cvar_95 = float(np.mean(tail)) if len(tail) > 0 else var_95

        max_dd = self._calculate_max_drawdown(cumulative_returns)

        asset_impacts = {}
        for i, asset in enumerate(self.asset_names):
            asset_ret = stressed_returns[:, :, i]
            asset_cum = np.cumprod(1 + asset_ret, axis=1) - 1
            asset_impacts[asset] = float(np.mean(asset_cum[:, -1]))

        return StressTestResult(
            scenario=scenario,
            portfolio_impact=float(np.mean(terminal_returns)),
            asset_impacts=asset_impacts,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            max_drawdown=float(np.mean(max_dd)),
            stressed_returns=stressed_returns,
        )

    def sensitivity_analysis(
        self,
        shock_variable: str,
        shock_range: np.ndarray,
        n_simulations: int = 1000,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        results = []
        for shock_level in shock_range:
            if shock_variable == "equity":
                scenario = StressScenario(
                    name=f"Equity shock {shock_level:.1%}",
                    description="Equity sensitivity test",
                    equity_shock=shock_level,
                    bond_shock=0.0,
                    volatility_multiplier=1.0,
                    correlation_stress=1.0,
                    duration_days=21,
                )
            elif shock_variable == "volatility":
                scenario = StressScenario(
                    name=f"Volatility shock {shock_level:.1f}x",
                    description="Volatility sensitivity test",
                    equity_shock=0.0,
                    bond_shock=0.0,
                    volatility_multiplier=shock_level,
                    correlation_stress=1.0,
                    duration_days=21,
                )
            elif shock_variable == "correlation":
                scenario = StressScenario(
                    name=f"Correlation stress {shock_level:.1f}x",
                    description="Correlation sensitivity test",
                    equity_shock=0.0,
                    bond_shock=0.0,
                    volatility_multiplier=1.0,
                    correlation_stress=shock_level,
                    duration_days=21,
                )
            else:
                raise ValueError(f"Unknown shock variable: {shock_variable}")

            result = self.run_custom_scenario(scenario, n_simulations, random_state)
            results.append(
                {
                    "shock_level": shock_level,
                    "portfolio_impact": result.portfolio_impact,
                    "var_95": result.var_95,
                    "var_99": result.var_99,
                    "cvar_95": result.cvar_95,
                    "max_drawdown": result.max_drawdown,
                }
            )
        return pd.DataFrame(results)

    def reverse_stress_test(
        self,
        target_loss: float,
        shock_variable: str = "equity",
        tolerance: float = 0.01,
        max_iterations: int = 50,
        n_simulations: int = 1000,
    ) -> tuple[float, StressTestResult]:
        """Find shock level required to achieve target loss (binary search)."""
        if shock_variable == "equity":
            low, high = -0.8, 0.0
        elif shock_variable == "volatility":
            low, high = 1.0, 10.0
        else:
            low, high = 1.0, 5.0

        result = None
        mid = (low + high) / 2
        for _ in range(max_iterations):
            mid = (low + high) / 2
            if shock_variable == "equity":
                scenario = StressScenario(
                    name="Reverse stress",
                    description="Reverse stress test",
                    equity_shock=mid,
                    bond_shock=0.0,
                    volatility_multiplier=1.5,
                    correlation_stress=1.3,
                    duration_days=21,
                )
            elif shock_variable == "volatility":
                scenario = StressScenario(
                    name="Reverse stress",
                    description="Reverse stress test",
                    equity_shock=-0.1,
                    bond_shock=0.0,
                    volatility_multiplier=mid,
                    correlation_stress=1.3,
                    duration_days=21,
                )
            else:
                scenario = StressScenario(
                    name="Reverse stress",
                    description="Reverse stress test",
                    equity_shock=-0.1,
                    bond_shock=0.0,
                    volatility_multiplier=1.5,
                    correlation_stress=mid,
                    duration_days=21,
                )

            result = self.run_custom_scenario(scenario, n_simulations)
            current_loss = result.portfolio_impact

            if abs(current_loss - target_loss) < tolerance:
                return mid, result
            if current_loss > target_loss:
                if shock_variable == "equity":
                    high = mid
                else:
                    low = mid
            else:
                if shock_variable == "equity":
                    low = mid
                else:
                    high = mid

        return mid, result  # type: ignore[return-value]

    def var_calculation(
        self,
        confidence_levels: list[float] | None = None,
        horizon_days: int = 1,
        method: str = "historical",
        n_simulations: int = 10000,
    ) -> dict[float, float]:
        """Calculate Value at Risk."""
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        if self.returns is None:
            raise ValueError("Historical returns required for VaR calculation")

        weights_arr = np.array([self.weights.get(a, 1 / len(self.asset_names)) for a in self.asset_names])
        portfolio_returns = (self.returns.values * weights_arr).sum(axis=1)

        if horizon_days > 1:
            scaled_returns = pd.Series(portfolio_returns).rolling(horizon_days).sum().dropna().values
        else:
            scaled_returns = portfolio_returns

        var_results = {}
        for level in confidence_levels:
            alpha = 1 - level
            if method == "historical":
                var_results[level] = float(np.percentile(scaled_returns, alpha * 100))
            elif method == "parametric":
                from scipy.stats import norm

                mu = np.mean(scaled_returns)
                sigma = np.std(scaled_returns)
                var_results[level] = float(mu + sigma * norm.ppf(alpha))
            elif method == "monte_carlo":
                rng = np.random.default_rng(42)
                mu = np.mean(scaled_returns)
                sigma = np.std(scaled_returns)
                simulated = rng.normal(mu, sigma, n_simulations)
                var_results[level] = float(np.percentile(simulated, alpha * 100))
            else:
                raise ValueError(f"Unknown method: {method}")
        return var_results

    def cvar_calculation(
        self,
        confidence_levels: list[float] | None = None,
        horizon_days: int = 1,
    ) -> dict[float, float]:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        if self.returns is None:
            raise ValueError("Historical returns required")

        weights_arr = np.array([self.weights.get(a, 1 / len(self.asset_names)) for a in self.asset_names])
        portfolio_returns = (self.returns.values * weights_arr).sum(axis=1)

        if horizon_days > 1:
            scaled_returns = pd.Series(portfolio_returns).rolling(horizon_days).sum().dropna().values
        else:
            scaled_returns = portfolio_returns

        cvar_results = {}
        for level in confidence_levels:
            alpha = 1 - level
            var = np.percentile(scaled_returns, alpha * 100)
            tail = scaled_returns[scaled_returns <= var]
            cvar_results[level] = float(np.mean(tail)) if len(tail) > 0 else float(var)
        return cvar_results

    def compare_scenarios(
        self,
        scenarios: list[HistoricalScenario | StressScenario],
        n_simulations: int = 1000,
    ) -> pd.DataFrame:
        results = []
        for scenario in scenarios:
            if isinstance(scenario, HistoricalScenario):
                result = self.run_historical_scenario(scenario, n_simulations)
            else:
                result = self.run_custom_scenario(scenario, n_simulations)
            results.append(
                {
                    "scenario": result.scenario.name,
                    "portfolio_impact": result.portfolio_impact,
                    "var_95": result.var_95,
                    "var_99": result.var_99,
                    "cvar_95": result.cvar_95,
                    "max_drawdown": result.max_drawdown,
                    "duration_days": result.scenario.duration_days,
                }
            )
        return pd.DataFrame(results).set_index("scenario")

    def _stress_correlation_matrix(self, corr: np.ndarray, stress_factor: float) -> np.ndarray:
        n = len(corr)
        stressed = corr.copy()
        for i in range(n):
            for j in range(n):
                if i != j:
                    sign = np.sign(corr[i, j])
                    magnitude = min(abs(corr[i, j]) * stress_factor, 0.999)
                    stressed[i, j] = sign * magnitude

        eigvals, eigvecs = np.linalg.eigh(stressed)
        eigvals = np.maximum(eigvals, 1e-8)
        stressed = eigvecs @ np.diag(eigvals) @ eigvecs.T

        d = np.sqrt(np.diag(stressed))
        stressed = stressed / np.outer(d, d)
        np.fill_diagonal(stressed, 1.0)
        return stressed

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> np.ndarray:
        wealth = 1 + cumulative_returns
        running_max = np.maximum.accumulate(wealth, axis=1)
        drawdowns = (wealth - running_max) / running_max
        return np.min(drawdowns, axis=1)


__all__ = [
    "HistoricalScenario",
    "StressScenario",
    "StressTestResult",
    "StressTestEngine",
    "HISTORICAL_SCENARIOS",
]
