"""Tests for StatisticalValidation plugin (validation.statistical.v1).

Covers deflated Sharpe, bootstrap CI, haircut Sharpe, and plugin metadata.
Self-contained -- no conftest.py required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.validation.statistical import StatisticalValidation


class TestStatisticalValidation:
    """Test suite for the StatisticalValidation plugin."""

    @pytest.fixture
    def plugin(self) -> StatisticalValidation:
        return StatisticalValidation()

    @staticmethod
    def _make_returns(n_days: int = 500, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        rets = rng.normal(0.0005, 0.02, size=n_days)
        return pd.DataFrame({"returns": rets}, index=dates)

    @staticmethod
    def _empty_weights() -> pd.DataFrame:
        return pd.DataFrame()

    # ================================================================
    # 1. Result has required top-level keys
    # ================================================================
    def test_result_has_required_keys(self, plugin: StatisticalValidation) -> None:
        returns = self._make_returns()
        result = plugin.validate(returns, self._empty_weights(), None, {})

        assert "findings" in result
        assert "metrics" in result
        assert "passed" in result

    # ================================================================
    # 2. Metrics contain expected statistical values
    # ================================================================
    def test_metrics_contain_expected_keys(self, plugin: StatisticalValidation) -> None:
        returns = self._make_returns()
        result = plugin.validate(returns, self._empty_weights(), None, {})

        metrics = result["metrics"]
        assert "deflated_sharpe" in metrics
        assert "sharpe_ci_lower" in metrics
        assert "sharpe_ci_upper" in metrics
        assert "haircut_sharpe" in metrics
        assert "observed_sharpe" in metrics

    # ================================================================
    # 3. CI lower <= CI upper
    # ================================================================
    def test_ci_ordering(self, plugin: StatisticalValidation) -> None:
        returns = self._make_returns()
        result = plugin.validate(returns, self._empty_weights(), None, {})

        metrics = result["metrics"]
        assert metrics["sharpe_ci_lower"] <= metrics["sharpe_ci_upper"]

    # ================================================================
    # 4. Haircut Sharpe <= observed Sharpe (in absolute value)
    # ================================================================
    def test_haircut_leq_observed(self, plugin: StatisticalValidation) -> None:
        returns = self._make_returns()
        params = {"n_strategies_tested": 10}
        result = plugin.validate(returns, self._empty_weights(), None, params)

        metrics = result["metrics"]
        assert metrics["haircut_sharpe"] <= metrics["observed_sharpe"]

    # ================================================================
    # 5. With single strategy, haircut applies minimal penalty (2%)
    # ================================================================
    def test_single_strategy_minimal_haircut(self, plugin: StatisticalValidation) -> None:
        returns = self._make_returns()
        params = {"n_strategies_tested": 1}
        result = plugin.validate(returns, self._empty_weights(), None, params)

        metrics = result["metrics"]
        expected = metrics["observed_sharpe"] * 0.98  # 1 * 0.02 penalty
        assert abs(metrics["haircut_sharpe"] - expected) < 1e-10

    # ================================================================
    # 6. More trials increases multiple-testing penalty
    # ================================================================
    def test_more_strategies_reduces_haircut(self, plugin: StatisticalValidation) -> None:
        returns = self._make_returns()
        result_1 = plugin.validate(returns, self._empty_weights(), None, {"n_strategies_tested": 1})
        result_50 = plugin.validate(returns, self._empty_weights(), None, {"n_strategies_tested": 50})

        assert result_50["metrics"]["haircut_sharpe"] <= result_1["metrics"]["haircut_sharpe"]

    # ================================================================
    # 7. Bootstrap with custom confidence level
    # ================================================================
    def test_custom_confidence(self, plugin: StatisticalValidation) -> None:
        returns = self._make_returns()
        result_90 = plugin.validate(returns, self._empty_weights(), None, {"confidence": 0.90})
        result_99 = plugin.validate(returns, self._empty_weights(), None, {"confidence": 0.99})

        # Wider confidence => wider interval
        width_90 = result_90["metrics"]["sharpe_ci_upper"] - result_90["metrics"]["sharpe_ci_lower"]
        width_99 = result_99["metrics"]["sharpe_ci_upper"] - result_99["metrics"]["sharpe_ci_lower"]
        assert width_99 >= width_90

    # ================================================================
    # 8. Plugin meta attributes
    # ================================================================
    def test_plugin_meta_attributes(self) -> None:
        meta = StatisticalValidation.meta

        assert meta.name == "validation.statistical.v1"
        assert meta.kind == "validation"
        assert meta.version == "0.1.0"
        assert "validation" in meta.tags
