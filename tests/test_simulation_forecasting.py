"""Tests for quantbox.simulation.forecasting — ReturnForecaster."""
import numpy as np
import pandas as pd
import pytest

from quantbox.simulation.forecasting import (
    ReturnForecaster,
    Horizon,
    ForecastResult,
    MultiHorizonForecast,
)


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    return pd.DataFrame({
        "SPY": rng.normal(0.0003, 0.012, 500),
        "TLT": rng.normal(0.0001, 0.005, 500),
        "GLD": rng.normal(0.0002, 0.008, 500),
    }, index=dates)


@pytest.fixture
def forecaster(sample_returns):
    return ReturnForecaster(sample_returns, risk_free_rate=0.04)


class TestHorizonEnum:
    def test_values(self):
        assert Horizon.DAILY.value == 1
        assert Horizon.ANNUAL.value == 252
        assert Horizon.TEN_YEAR.value == 2520


class TestForecastSingleHorizon:
    def test_historical(self, forecaster):
        result = forecaster.forecast_single_horizon("SPY", Horizon.MONTHLY, method="historical")
        assert isinstance(result, ForecastResult)
        assert result.asset == "SPY"
        assert result.horizon == 21
        assert result.method == "historical"
        assert isinstance(result.expected_return, float)
        assert isinstance(result.volatility, float)
        assert result.volatility > 0

    def test_bootstrap(self, forecaster):
        result = forecaster.forecast_single_horizon(
            "SPY", 21, method="bootstrap", n_simulations=1000,
        )
        assert result.method == "bootstrap"
        assert result.volatility > 0

    def test_confidence_intervals(self, forecaster):
        result = forecaster.forecast_single_horizon(
            "TLT", Horizon.QUARTERLY, method="historical",
        )
        assert 0.95 in result.confidence_intervals
        lower, upper = result.confidence_intervals[0.95]
        assert lower < upper

    def test_percentiles(self, forecaster):
        result = forecaster.forecast_single_horizon("SPY", 21, method="historical")
        assert 5 in result.percentiles
        assert 50 in result.percentiles
        assert 95 in result.percentiles
        assert result.percentiles[5] < result.percentiles[95]

    def test_to_dict(self, forecaster):
        result = forecaster.forecast_single_horizon("SPY", 21, method="historical")
        d = result.to_dict()
        assert "asset" in d
        assert "expected_return" in d
        assert "ci_95_lower" in d

    def test_integer_horizon(self, forecaster):
        result = forecaster.forecast_single_horizon("SPY", 10, method="historical")
        assert result.horizon == 10


class TestForecastMultiHorizon:
    def test_default_horizons(self, forecaster):
        # Use short horizons that fit within 500 data points
        short_horizons = [Horizon.DAILY, Horizon.WEEKLY, Horizon.MONTHLY, Horizon.QUARTERLY]
        mhf = forecaster.forecast_multi_horizon("SPY", horizons=short_horizons, method="historical")
        assert isinstance(mhf, MultiHorizonForecast)
        assert mhf.asset == "SPY"
        assert Horizon.DAILY.value in mhf.forecasts
        assert Horizon.QUARTERLY.value in mhf.forecasts

    def test_custom_horizons(self, forecaster):
        mhf = forecaster.forecast_multi_horizon(
            "TLT", horizons=[Horizon.WEEKLY, Horizon.MONTHLY], method="historical",
        )
        assert len(mhf.forecasts) == 2

    def test_term_structure(self, forecaster):
        mhf = forecaster.forecast_multi_horizon(
            "SPY", horizons=[Horizon.DAILY, Horizon.MONTHLY, Horizon.ANNUAL],
            method="historical",
        )
        ts = mhf.term_structure
        assert isinstance(ts, pd.DataFrame)
        assert "horizon_days" in ts.columns
        assert "annualized_return" in ts.columns
        assert len(ts) == 3

    def test_get_horizon(self, forecaster):
        mhf = forecaster.forecast_multi_horizon(
            "SPY", horizons=[Horizon.MONTHLY], method="historical",
        )
        f = mhf.get_horizon(Horizon.MONTHLY)
        assert f.horizon == 21


class TestForecastAllAssets:
    def test_all_assets(self, forecaster):
        results = forecaster.forecast_all_assets(
            horizons=[Horizon.MONTHLY], method="historical",
        )
        assert "SPY" in results
        assert "TLT" in results
        assert "GLD" in results


class TestFanChart:
    def test_fan_chart_data(self, forecaster):
        data = forecaster.generate_fan_chart_data(
            "SPY", max_horizon=50, step=5, n_simulations=1000,
        )
        assert isinstance(data, pd.DataFrame)
        assert "horizon" in data.columns
        assert "p50" in data.columns
        assert "p5" in data.columns
        assert "p95" in data.columns
        # p5 < p50 < p95
        assert (data["p5"] <= data["p50"]).all()
        assert (data["p50"] <= data["p95"]).all()


class TestParametricForecast:
    def test_requires_scipy(self, forecaster):
        try:
            result = forecaster.forecast_single_horizon(
                "SPY", 21, method="parametric",
            )
            assert result.method == "parametric"
        except ImportError:
            pytest.skip("scipy not available")


class TestMeanReversionForecast:
    def test_mean_reversion(self, forecaster):
        try:
            result = forecaster.expected_return_with_mean_reversion(
                "SPY", horizon=63, long_term_return=0.08, mean_reversion_speed=0.3,
            )
            assert result.method == "mean_reversion"
            assert isinstance(result.expected_return, float)
        except ImportError:
            pytest.skip("scipy not available")


class TestBayesianShrinkage:
    def test_bayesian(self, forecaster):
        try:
            result = forecaster.bayesian_shrinkage_forecast(
                "SPY", horizon=63, prior_return=0.06, prior_weight=0.5,
            )
            assert result.method == "bayesian_shrinkage"
            # Result should be between pure prior and pure historical
        except ImportError:
            pytest.skip("scipy not available")
