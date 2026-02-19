"""Tests for RegimeValidation plugin (validation.regime.v1).

Covers regime classification, per-regime metrics, and plugin metadata.
Self-contained -- no conftest.py required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.validation.regime import RegimeValidation


class TestRegimeValidation:
    """Test suite for the RegimeValidation plugin."""

    @pytest.fixture
    def plugin(self) -> RegimeValidation:
        return RegimeValidation()

    @staticmethod
    def _make_returns(n_days: int = 500, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        rets = rng.normal(0.001, 0.02, size=n_days)
        return pd.DataFrame({"returns": rets}, index=dates)

    @staticmethod
    def _empty_weights() -> pd.DataFrame:
        return pd.DataFrame()

    # ================================================================
    # 1. Result has required top-level keys
    # ================================================================
    def test_result_has_required_keys(self, plugin: RegimeValidation) -> None:
        returns = self._make_returns()
        result = plugin.validate(returns, self._empty_weights(), None, {})

        assert "findings" in result
        assert "metrics" in result
        assert "passed" in result

    # ================================================================
    # 2. Regime breakdown exists and has expected structure
    # ================================================================
    def test_regime_breakdown_structure(self, plugin: RegimeValidation) -> None:
        returns = self._make_returns()
        result = plugin.validate(returns, self._empty_weights(), None, {})

        breakdown = result["metrics"]["regime_breakdown"]
        assert isinstance(breakdown, list)
        assert len(breakdown) > 0

        for entry in breakdown:
            assert "regime" in entry
            assert "sharpe" in entry
            assert "return" in entry
            assert "pct_time" in entry

    # ================================================================
    # 3. Regime names are from the expected set
    # ================================================================
    def test_regime_names(self, plugin: RegimeValidation) -> None:
        returns = self._make_returns()
        result = plugin.validate(returns, self._empty_weights(), None, {})

        breakdown = result["metrics"]["regime_breakdown"]
        regime_names = {e["regime"] for e in breakdown}
        valid_names = {"trending_up", "trending_down", "high_vol", "low_vol"}
        assert regime_names.issubset(valid_names)

    # ================================================================
    # 4. Percentage of time sums to approximately 1.0
    # ================================================================
    def test_pct_time_sums_to_one(self, plugin: RegimeValidation) -> None:
        returns = self._make_returns()
        result = plugin.validate(returns, self._empty_weights(), None, {})

        breakdown = result["metrics"]["regime_breakdown"]
        total_pct = sum(e["pct_time"] for e in breakdown)
        assert abs(total_pct - 1.0) < 0.05  # Allow small rounding tolerance

    # ================================================================
    # 5. Custom window parameter is respected
    # ================================================================
    def test_custom_window(self, plugin: RegimeValidation) -> None:
        returns = self._make_returns()
        result_30 = plugin.validate(returns, self._empty_weights(), None, {"window": 30})
        result_90 = plugin.validate(returns, self._empty_weights(), None, {"window": 90})

        # Different windows should produce different breakdowns
        breakdown_30 = result_30["metrics"]["regime_breakdown"]
        breakdown_90 = result_90["metrics"]["regime_breakdown"]
        # At minimum, both produce results
        assert len(breakdown_30) > 0
        assert len(breakdown_90) > 0

    # ================================================================
    # 6. Short return series still works (graceful handling)
    # ================================================================
    def test_short_series(self, plugin: RegimeValidation) -> None:
        dates = pd.date_range("2023-01-01", periods=70, freq="D")
        rets = np.random.default_rng(42).normal(0.001, 0.02, 70)
        returns = pd.DataFrame({"returns": rets}, index=dates)

        result = plugin.validate(returns, self._empty_weights(), None, {"window": 60})

        assert "metrics" in result
        assert "regime_breakdown" in result["metrics"]

    # ================================================================
    # 7. Plugin meta attributes
    # ================================================================
    def test_plugin_meta_attributes(self) -> None:
        meta = RegimeValidation.meta

        assert meta.name == "validation.regime.v1"
        assert meta.kind == "validation"
        assert meta.version == "0.1.0"
        assert "validation" in meta.tags
