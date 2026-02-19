"""Tests for BenchmarkValidation plugin (validation.benchmark.v1).

Covers benchmark metrics (alpha, beta, tracking error, IR, R-squared),
graceful skip when no benchmark, and plugin metadata.
Self-contained -- no conftest.py required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.validation.benchmark import BenchmarkValidation


class TestBenchmarkValidation:
    """Test suite for the BenchmarkValidation plugin."""

    @pytest.fixture
    def plugin(self) -> BenchmarkValidation:
        return BenchmarkValidation()

    @staticmethod
    def _make_returns(n_days: int = 300, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        rets = rng.normal(0.001, 0.02, size=n_days)
        return pd.DataFrame({"returns": rets}, index=dates)

    @staticmethod
    def _make_benchmark(n_days: int = 300, seed: int = 99) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        rets = rng.normal(0.0008, 0.015, size=n_days)
        return pd.DataFrame({"benchmark": rets}, index=dates)

    @staticmethod
    def _empty_weights() -> pd.DataFrame:
        return pd.DataFrame()

    # ================================================================
    # 1. Graceful skip when benchmark is None
    # ================================================================
    def test_no_benchmark_graceful_skip(self, plugin: BenchmarkValidation) -> None:
        returns = self._make_returns()
        result = plugin.validate(returns, self._empty_weights(), None, {})

        assert result == {"findings": [], "metrics": {}, "passed": True}

    # ================================================================
    # 2. Result has required top-level keys with benchmark
    # ================================================================
    def test_result_has_required_keys(self, plugin: BenchmarkValidation) -> None:
        returns = self._make_returns()
        benchmark = self._make_benchmark()
        result = plugin.validate(returns, self._empty_weights(), benchmark, {})

        assert "findings" in result
        assert "metrics" in result
        assert "passed" in result

    # ================================================================
    # 3. Metrics contain expected benchmark comparison keys
    # ================================================================
    def test_metrics_contain_expected_keys(self, plugin: BenchmarkValidation) -> None:
        returns = self._make_returns()
        benchmark = self._make_benchmark()
        result = plugin.validate(returns, self._empty_weights(), benchmark, {})

        metrics = result["metrics"]
        assert "beta" in metrics
        assert "alpha" in metrics
        assert "tracking_error" in metrics
        assert "information_ratio" in metrics
        assert "r_squared" in metrics

    # ================================================================
    # 4. Beta of strategy against itself is approximately 1.0
    # ================================================================
    def test_beta_self_is_one(self, plugin: BenchmarkValidation) -> None:
        returns = self._make_returns()
        result = plugin.validate(returns, self._empty_weights(), returns.copy(), {})

        assert abs(result["metrics"]["beta"] - 1.0) < 1e-6

    # ================================================================
    # 5. R-squared of strategy against itself is approximately 1.0
    # ================================================================
    def test_r_squared_self_is_one(self, plugin: BenchmarkValidation) -> None:
        returns = self._make_returns()
        result = plugin.validate(returns, self._empty_weights(), returns.copy(), {})

        assert abs(result["metrics"]["r_squared"] - 1.0) < 1e-6

    # ================================================================
    # 6. R-squared is between 0 and 1
    # ================================================================
    def test_r_squared_in_range(self, plugin: BenchmarkValidation) -> None:
        returns = self._make_returns()
        benchmark = self._make_benchmark()
        result = plugin.validate(returns, self._empty_weights(), benchmark, {})

        r2 = result["metrics"]["r_squared"]
        assert 0.0 <= r2 <= 1.0

    # ================================================================
    # 7. Tracking error is non-negative
    # ================================================================
    def test_tracking_error_non_negative(self, plugin: BenchmarkValidation) -> None:
        returns = self._make_returns()
        benchmark = self._make_benchmark()
        result = plugin.validate(returns, self._empty_weights(), benchmark, {})

        assert result["metrics"]["tracking_error"] >= 0.0

    # ================================================================
    # 8. Alpha is annualized
    # ================================================================
    def test_alpha_is_annualized(self, plugin: BenchmarkValidation) -> None:
        returns = self._make_returns()
        benchmark = self._make_benchmark()

        result_365 = plugin.validate(returns, self._empty_weights(), benchmark, {"trading_days": 365})
        result_252 = plugin.validate(returns, self._empty_weights(), benchmark, {"trading_days": 252})

        # Different annualization factors produce different alpha values
        assert result_365["metrics"]["alpha"] != result_252["metrics"]["alpha"]

    # ================================================================
    # 9. Plugin meta attributes
    # ================================================================
    def test_plugin_meta_attributes(self) -> None:
        meta = BenchmarkValidation.meta

        assert meta.name == "validation.benchmark.v1"
        assert meta.kind == "validation"
        assert meta.version == "0.1.0"
        assert "validation" in meta.tags
