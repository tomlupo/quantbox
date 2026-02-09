"""Stress-test risk manager plugin.

Runs configured stress scenarios against proposed target weights and
raises findings when portfolio risk metrics exceed thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta
from quantbox.simulation.stress_testing import (
    HistoricalScenario,
    StressTestEngine,
    HISTORICAL_SCENARIOS,
)


@dataclass
class StressTestRiskManager:
    """Validate targets against stress scenarios.

    Uses the StressTestEngine to run historical and custom scenarios.
    Produces findings when VaR, CVaR, or max drawdown exceed configured
    thresholds.
    """

    meta = PluginMeta(
        name="risk.stress_test.v1",
        kind="risk",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Stress-test risk plugin. Runs Monte Carlo stress scenarios "
            "(2008 crisis, COVID crash, etc.) and flags portfolios whose "
            "VaR, CVaR, or drawdown breach thresholds."
        ),
        tags=("risk", "stress-test", "simulation"),
        capabilities=("backtest", "paper", "live"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "scenarios": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["2008_financial_crisis", "2020_covid_crash"],
                    "description": (
                        "Scenario IDs to test. Use HistoricalScenario enum values "
                        "(e.g. '2008_financial_crisis', '2020_covid_crash', "
                        "'1987_black_monday')."
                    ),
                },
                "max_var_95": {
                    "type": "number",
                    "default": -0.20,
                    "description": "Max acceptable 95% VaR (negative = loss). Breach triggers warning.",
                },
                "max_cvar_95": {
                    "type": "number",
                    "default": -0.30,
                    "description": "Max acceptable 95% CVaR. Breach triggers warning.",
                },
                "max_stress_drawdown": {
                    "type": "number",
                    "default": -0.40,
                    "description": "Max acceptable stress drawdown. Breach triggers error.",
                },
                "n_simulations": {
                    "type": "integer",
                    "minimum": 100,
                    "default": 1000,
                    "description": "Monte Carlo paths per scenario.",
                },
                "historical_returns": {
                    "type": "string",
                    "description": (
                        "Path to a Parquet file with historical returns "
                        "(wide format: date x symbols). If omitted, uses "
                        "baseline parameters from the stress scenario."
                    ),
                },
            },
        },
        examples=(
            "plugins:\n  risk:\n    - name: risk.stress_test.v1\n      params:\n"
            "        scenarios:\n          - 2008_financial_crisis\n"
            "          - 2020_covid_crash\n        max_var_95: -0.20\n"
            "        max_cvar_95: -0.30\n        n_simulations: 1000",
        ),
    )

    def check_targets(
        self, targets: pd.DataFrame, params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Run stress scenarios on proposed target weights.

        Parameters
        ----------
        targets : pd.DataFrame
            Must contain ``symbol`` and ``weight`` columns.
        params : dict
            Plugin configuration (scenarios, thresholds, etc.).

        Returns
        -------
        list[dict]
            Findings with keys ``level``, ``rule``, ``detail``.
        """
        findings: List[Dict[str, Any]] = []
        if targets is None or targets.empty:
            return findings

        weights_col = targets["weight"].astype(float) if "weight" in targets.columns else pd.Series(dtype=float)
        symbols_col = targets["symbol"].tolist() if "symbol" in targets.columns else [f"asset_{i}" for i in range(len(targets))]
        if weights_col.empty:
            return findings

        weights = {sym: float(w) for sym, w in zip(symbols_col, weights_col)}

        # Load historical returns if provided
        returns_df = None
        hist_path = params.get("historical_returns")
        if hist_path:
            try:
                returns_df = pd.read_parquet(hist_path)
            except Exception:
                pass

        engine = StressTestEngine(returns=returns_df, weights=weights)

        scenario_ids = params.get("scenarios", ["2008_financial_crisis", "2020_covid_crash"])
        max_var = float(params.get("max_var_95", -0.20))
        max_cvar = float(params.get("max_cvar_95", -0.30))
        max_dd = float(params.get("max_stress_drawdown", -0.40))
        n_sims = int(params.get("n_simulations", 1000))

        for sid in scenario_ids:
            try:
                result = engine.run_historical_scenario(sid, n_simulations=n_sims)
            except (KeyError, ValueError):
                findings.append({
                    "level": "warn",
                    "rule": "unknown_scenario",
                    "detail": f"Scenario '{sid}' not found. Skipping.",
                })
                continue

            scenario_name = result.scenario.name

            if result.var_95 < max_var:
                findings.append({
                    "level": "warn",
                    "rule": "stress_var_breach",
                    "detail": (
                        f"[{scenario_name}] VaR 95% = {result.var_95:.2%} "
                        f"exceeds threshold {max_var:.2%}."
                    ),
                })

            if result.cvar_95 < max_cvar:
                findings.append({
                    "level": "warn",
                    "rule": "stress_cvar_breach",
                    "detail": (
                        f"[{scenario_name}] CVaR 95% = {result.cvar_95:.2%} "
                        f"exceeds threshold {max_cvar:.2%}."
                    ),
                })

            if result.max_drawdown < max_dd:
                findings.append({
                    "level": "error",
                    "rule": "stress_drawdown_breach",
                    "detail": (
                        f"[{scenario_name}] Max drawdown = {result.max_drawdown:.2%} "
                        f"exceeds threshold {max_dd:.2%}."
                    ),
                })

        return findings

    def check_orders(
        self, orders: pd.DataFrame, params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Stress testing is target-level only; orders pass through."""
        return []
