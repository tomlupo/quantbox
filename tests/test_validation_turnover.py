"""Tests for TurnoverValidation plugin (validation.turnover.v1).

Covers turnover computation, cost-adjusted returns, breakeven cost, and plugin metadata.
Self-contained -- no conftest.py required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.validation.turnover import TurnoverValidation


class TestTurnoverValidation:
    """Test suite for the TurnoverValidation plugin."""

    @pytest.fixture
    def plugin(self) -> TurnoverValidation:
        return TurnoverValidation()

    @staticmethod
    def _make_dates(n: int = 100) -> pd.DatetimeIndex:
        return pd.date_range("2023-01-01", periods=n, freq="D")

    # ================================================================
    # 1. Static weights produce near-zero turnover
    # ================================================================
    def test_static_weights_zero_turnover(self, plugin: TurnoverValidation) -> None:
        dates = self._make_dates()
        n = len(dates)
        returns = pd.DataFrame({"strategy": np.random.default_rng(42).normal(0.001, 0.01, n)}, index=dates)
        # Constant weights
        weights = pd.DataFrame({"BTC": [0.5] * n, "ETH": [0.5] * n}, index=dates)

        result = plugin.validate(returns, weights, None, {})

        assert result["metrics"]["daily_turnover"] < 1e-10
        assert result["metrics"]["annual_turnover"] < 1e-6

    # ================================================================
    # 2. Random weights produce high turnover
    # ================================================================
    def test_random_weights_high_turnover(self, plugin: TurnoverValidation) -> None:
        dates = self._make_dates()
        n = len(dates)
        rng = np.random.default_rng(42)
        returns = pd.DataFrame({"strategy": rng.normal(0.001, 0.01, n)}, index=dates)
        # Random weights each day
        w1 = rng.uniform(0, 1, n)
        w2 = 1 - w1
        weights = pd.DataFrame({"BTC": w1, "ETH": w2}, index=dates)

        result = plugin.validate(returns, weights, None, {})

        assert result["metrics"]["daily_turnover"] > 0.05
        assert result["metrics"]["annual_turnover"] > 10.0

    # ================================================================
    # 3. Result has required top-level keys
    # ================================================================
    def test_result_has_required_keys(self, plugin: TurnoverValidation) -> None:
        dates = self._make_dates()
        n = len(dates)
        returns = pd.DataFrame({"strategy": np.zeros(n)}, index=dates)
        weights = pd.DataFrame({"BTC": [0.5] * n}, index=dates)

        result = plugin.validate(returns, weights, None, {})

        assert "findings" in result
        assert "metrics" in result
        assert "passed" in result

    # ================================================================
    # 4. Metrics contain expected turnover keys
    # ================================================================
    def test_metrics_contain_expected_keys(self, plugin: TurnoverValidation) -> None:
        dates = self._make_dates()
        n = len(dates)
        rng = np.random.default_rng(42)
        returns = pd.DataFrame({"strategy": rng.normal(0.001, 0.01, n)}, index=dates)
        weights = pd.DataFrame({"BTC": [0.5] * n, "ETH": [0.5] * n}, index=dates)

        result = plugin.validate(returns, weights, None, {})
        metrics = result["metrics"]

        assert "daily_turnover" in metrics
        assert "annual_turnover" in metrics
        assert "cost_adjusted_sharpe" in metrics
        assert "breakeven_cost_bps" in metrics

    # ================================================================
    # 5. Cost-adjusted Sharpe is less than or equal to unadjusted
    # ================================================================
    def test_cost_adjusted_sharpe_leq_unadjusted(self, plugin: TurnoverValidation) -> None:
        dates = self._make_dates(200)
        n = len(dates)
        rng = np.random.default_rng(42)
        returns = pd.DataFrame({"strategy": rng.normal(0.002, 0.01, n)}, index=dates)
        w1 = rng.uniform(0.3, 0.7, n)
        weights = pd.DataFrame({"BTC": w1, "ETH": 1 - w1}, index=dates)

        result = plugin.validate(returns, weights, None, {"cost_bps": 20})
        metrics = result["metrics"]

        # Cost adjustment should reduce Sharpe (or keep equal if no turnover)
        assert metrics["cost_adjusted_sharpe"] <= metrics.get("raw_sharpe", float("inf"))

    # ================================================================
    # 6. Breakeven cost is non-negative
    # ================================================================
    def test_breakeven_cost_non_negative(self, plugin: TurnoverValidation) -> None:
        dates = self._make_dates(200)
        n = len(dates)
        rng = np.random.default_rng(42)
        returns = pd.DataFrame({"strategy": rng.normal(0.002, 0.01, n)}, index=dates)
        w1 = rng.uniform(0.3, 0.7, n)
        weights = pd.DataFrame({"BTC": w1, "ETH": 1 - w1}, index=dates)

        result = plugin.validate(returns, weights, None, {})
        assert result["metrics"]["breakeven_cost_bps"] >= 0.0

    # ================================================================
    # 7. Plugin meta attributes
    # ================================================================
    def test_plugin_meta_attributes(self) -> None:
        meta = TurnoverValidation.meta

        assert meta.name == "validation.turnover.v1"
        assert meta.kind == "validation"
        assert meta.version == "0.1.0"
        assert "validation" in meta.tags
