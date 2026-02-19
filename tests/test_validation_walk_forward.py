"""Tests for WalkForwardValidation plugin (validation.walk_forward.v1).

Covers fold splitting, Sharpe computation, degradation detection, and plugin metadata.
Self-contained -- no conftest.py required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.validation.walk_forward import WalkForwardValidation


class TestWalkForwardValidation:
    """Test suite for the WalkForwardValidation plugin."""

    @pytest.fixture
    def plugin(self) -> WalkForwardValidation:
        return WalkForwardValidation()

    @staticmethod
    def _make_returns(n_days: int = 500, seed: int = 42) -> pd.DataFrame:
        """Generate random daily returns as a single-column DataFrame."""
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
    def test_result_has_required_keys(self, plugin: WalkForwardValidation) -> None:
        returns = self._make_returns()
        result = plugin.validate(returns, self._empty_weights(), None, {})

        assert "findings" in result
        assert "metrics" in result
        assert "passed" in result

    # ================================================================
    # 2. Metrics contain expected Sharpe values
    # ================================================================
    def test_metrics_contain_sharpe_values(self, plugin: WalkForwardValidation) -> None:
        returns = self._make_returns()
        result = plugin.validate(returns, self._empty_weights(), None, {})

        metrics = result["metrics"]
        assert "is_sharpe_mean" in metrics
        assert "oos_sharpe_mean" in metrics
        assert "sharpe_degradation" in metrics

    # ================================================================
    # 3. Custom params are respected
    # ================================================================
    def test_custom_params(self, plugin: WalkForwardValidation) -> None:
        returns = self._make_returns()
        params = {"n_splits": 3, "train_ratio": 0.6, "trading_days": 252}
        result = plugin.validate(returns, self._empty_weights(), None, params)

        assert "metrics" in result
        assert isinstance(result["metrics"]["is_sharpe_mean"], float)

    # ================================================================
    # 4. Sharpe degradation is computed correctly as a fraction
    # ================================================================
    def test_sharpe_degradation_formula(self, plugin: WalkForwardValidation) -> None:
        returns = self._make_returns()
        result = plugin.validate(returns, self._empty_weights(), None, {})

        metrics = result["metrics"]
        is_sharpe = metrics["is_sharpe_mean"]
        oos_sharpe = metrics["oos_sharpe_mean"]
        degradation = metrics["sharpe_degradation"]

        if abs(is_sharpe) > 1e-10:
            expected = (oos_sharpe - is_sharpe) / abs(is_sharpe)
            assert abs(degradation - expected) < 1e-10

    # ================================================================
    # 5. Finding generated when OOS Sharpe is negative
    # ================================================================
    def test_finding_on_negative_oos_sharpe(self, plugin: WalkForwardValidation) -> None:
        rng = np.random.default_rng(99)
        dates = pd.date_range("2023-01-01", periods=500, freq="D")
        # Negative drift to ensure negative OOS Sharpe
        rets = rng.normal(-0.005, 0.02, size=500)
        returns = pd.DataFrame({"returns": rets}, index=dates)

        result = plugin.validate(returns, self._empty_weights(), None, {})

        assert result["passed"] is False
        assert len(result["findings"]) > 0

    # ================================================================
    # 6. Passed is True when performance is strong
    # ================================================================
    def test_passed_when_strong_performance(self, plugin: WalkForwardValidation) -> None:
        rng = np.random.default_rng(7)
        dates = pd.date_range("2023-01-01", periods=500, freq="D")
        # Strong positive drift
        rets = rng.normal(0.005, 0.01, size=500)
        returns = pd.DataFrame({"returns": rets}, index=dates)

        result = plugin.validate(returns, self._empty_weights(), None, {})

        assert result["passed"] is True

    # ================================================================
    # 7. Plugin meta attributes
    # ================================================================
    def test_plugin_meta_attributes(self) -> None:
        meta = WalkForwardValidation.meta

        assert meta.name == "validation.walk_forward.v1"
        assert meta.kind == "validation"
        assert meta.version == "0.1.0"
        assert "validation" in meta.tags
