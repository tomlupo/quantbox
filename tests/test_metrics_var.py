"""Tests for VaR/CVaR functions in backtesting/metrics.py."""

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.backtesting.metrics import (
    compute_backtest_metrics,
    compute_cvar,
    compute_portfolio_cvar,
    compute_portfolio_var,
    compute_var,
)


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    return pd.Series(rng.normal(0.0003, 0.012, 500), index=dates)


@pytest.fixture
def multi_asset_returns():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    return pd.DataFrame(
        {
            "A": rng.normal(0.0003, 0.012, 500),
            "B": rng.normal(0.0001, 0.008, 500),
            "C": rng.normal(0.0002, 0.010, 500),
        },
        index=dates,
    )


class TestComputeVaR:
    def test_historical_var(self, sample_returns):
        var = compute_var(sample_returns, confidence_level=0.95)
        assert var < 0  # Loss

    def test_var_99_more_extreme(self, sample_returns):
        var_95 = compute_var(sample_returns, confidence_level=0.95)
        var_99 = compute_var(sample_returns, confidence_level=0.99)
        assert var_99 <= var_95

    def test_parametric_var(self, sample_returns):
        try:
            var = compute_var(sample_returns, method="parametric")
            assert var < 0
        except ImportError:
            pytest.skip("scipy not available")

    def test_monte_carlo_var(self, sample_returns):
        var = compute_var(sample_returns, method="monte_carlo")
        assert var < 0

    def test_multi_day_horizon(self, sample_returns):
        var_1d = compute_var(sample_returns, horizon_days=1)
        var_5d = compute_var(sample_returns, horizon_days=5)
        # Multi-day VaR should be more extreme than single day
        assert var_5d < var_1d

    def test_invalid_method(self, sample_returns):
        with pytest.raises(ValueError, match="Unknown VaR method"):
            compute_var(sample_returns, method="invalid")


class TestComputeCVaR:
    def test_cvar_worse_than_var(self, sample_returns):
        var = compute_var(sample_returns, confidence_level=0.95)
        cvar = compute_cvar(sample_returns, confidence_level=0.95)
        assert cvar <= var

    def test_cvar_is_negative(self, sample_returns):
        cvar = compute_cvar(sample_returns, confidence_level=0.95)
        assert cvar < 0

    def test_multi_day_horizon(self, sample_returns):
        cvar_1d = compute_cvar(sample_returns, horizon_days=1)
        cvar_5d = compute_cvar(sample_returns, horizon_days=5)
        assert cvar_5d < cvar_1d


class TestPortfolioVaR:
    def test_portfolio_var(self, multi_asset_returns):
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}
        var_dict = compute_portfolio_var(multi_asset_returns, weights)
        assert 0.95 in var_dict
        assert 0.99 in var_dict
        assert var_dict[0.99] <= var_dict[0.95]

    def test_custom_confidence(self, multi_asset_returns):
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}
        var_dict = compute_portfolio_var(
            multi_asset_returns,
            weights,
            confidence_levels=[0.90],
        )
        assert 0.90 in var_dict


class TestPortfolioCVaR:
    def test_portfolio_cvar(self, multi_asset_returns):
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}
        cvar_dict = compute_portfolio_cvar(multi_asset_returns, weights)
        assert 0.95 in cvar_dict

    def test_cvar_worse_than_var(self, multi_asset_returns):
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}
        var_dict = compute_portfolio_var(multi_asset_returns, weights, confidence_levels=[0.95])
        cvar_dict = compute_portfolio_cvar(multi_asset_returns, weights, confidence_levels=[0.95])
        assert cvar_dict[0.95] <= var_dict[0.95]


class TestBacktestMetricsVaR:
    def test_var_in_metrics(self, sample_returns):
        metrics = compute_backtest_metrics(sample_returns)
        assert "var_95" in metrics
        assert "cvar_95" in metrics
        assert metrics["var_95"] < 0
        assert metrics["cvar_95"] <= metrics["var_95"]
