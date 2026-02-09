"""Tests for quantbox.simulation.correlation — CorrelationEngine."""
import numpy as np
import pandas as pd
import pytest

from quantbox.simulation.correlation import (
    CorrelationEngine,
    CorrelationResult,
    generate_random_correlation_matrix,
)


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)
    n_obs = 500
    dates = pd.bdate_range("2020-01-01", periods=n_obs)
    # Correlated returns
    corr = np.array([[1.0, 0.6, -0.2], [0.6, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    L = np.linalg.cholesky(corr)
    Z = rng.standard_normal((n_obs, 3))
    returns = Z @ L.T * 0.01
    return pd.DataFrame(returns, index=dates, columns=["A", "B", "C"])


class TestStaticCorrelation:
    def test_shape(self, sample_returns):
        engine = CorrelationEngine(sample_returns)
        result = engine.static_correlation()
        assert result.current_correlation.shape == (3, 3)

    def test_diagonal_is_one(self, sample_returns):
        engine = CorrelationEngine(sample_returns)
        result = engine.static_correlation()
        np.testing.assert_allclose(np.diag(result.current_correlation), 1.0)

    def test_symmetric(self, sample_returns):
        engine = CorrelationEngine(sample_returns)
        result = engine.static_correlation()
        np.testing.assert_allclose(result.current_correlation, result.current_correlation.T)

    def test_to_dataframe(self, sample_returns):
        engine = CorrelationEngine(sample_returns)
        result = engine.static_correlation()
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["A", "B", "C"]


class TestRollingCorrelation:
    def test_shape(self, sample_returns):
        engine = CorrelationEngine(sample_returns)
        result = engine.rolling_correlation(window=60)
        n_valid = len(sample_returns) - 60 + 1
        assert result.correlation_history.shape == (n_valid, 3, 3)

    def test_current_is_last(self, sample_returns):
        engine = CorrelationEngine(sample_returns)
        result = engine.rolling_correlation(window=60)
        np.testing.assert_array_equal(result.current_correlation, result.correlation_history[-1])


class TestEWMACorrelation:
    def test_shape(self, sample_returns):
        engine = CorrelationEngine(sample_returns)
        result = engine.ewma_correlation(lambda_param=0.94)
        assert result.correlation_history is not None
        assert result.current_correlation.shape == (3, 3)

    def test_psd(self, sample_returns):
        """EWMA correlation should be positive semi-definite."""
        engine = CorrelationEngine(sample_returns)
        result = engine.ewma_correlation()
        eigvals = np.linalg.eigvalsh(result.current_correlation)
        assert np.all(eigvals >= -1e-10)


class TestLedoitWolf:
    def test_shrinkage_produces_valid_corr(self, sample_returns):
        engine = CorrelationEngine(sample_returns)
        result = engine.ledoit_wolf_shrinkage()
        # Valid correlation: diagonal=1, symmetric, PSD
        np.testing.assert_allclose(np.diag(result.current_correlation), 1.0, atol=1e-10)
        np.testing.assert_allclose(result.current_correlation, result.current_correlation.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(result.current_correlation)
        assert np.all(eigvals >= -1e-10)

    def test_method_name(self, sample_returns):
        engine = CorrelationEngine(sample_returns)
        result = engine.ledoit_wolf_shrinkage()
        assert result.method.startswith("ledoit_wolf_")


class TestCorrelationStress:
    def test_stressed_corr_psd(self, sample_returns):
        engine = CorrelationEngine(sample_returns)
        stressed = engine.correlation_stress(stress_factor=1.5)
        eigvals = np.linalg.eigvalsh(stressed)
        assert np.all(eigvals >= -1e-10)

    def test_stressed_correlations_increase(self, sample_returns):
        engine = CorrelationEngine(sample_returns)
        base = engine.static_correlation().current_correlation
        stressed = engine.correlation_stress(stress_factor=2.0)
        # Off-diagonal absolute values should generally increase
        mask = ~np.eye(3, dtype=bool)
        assert np.mean(np.abs(stressed[mask])) >= np.mean(np.abs(base[mask])) - 0.1


class TestCorrelationForecast:
    def test_static_forecast(self, sample_returns):
        engine = CorrelationEngine(sample_returns)
        forecast = engine.forecast_correlation(n_steps=10, method="static")
        assert forecast.shape == (10, 3, 3)

    def test_ewma_forecast(self, sample_returns):
        engine = CorrelationEngine(sample_returns)
        forecast = engine.forecast_correlation(n_steps=5, method="ewma")
        assert forecast.shape == (5, 3, 3)


class TestRegimeDetection:
    def test_output_shape(self, sample_returns):
        engine = CorrelationEngine(sample_returns)
        regimes = engine.correlation_regime_detection(window=60)
        assert "average_correlation" in regimes.columns
        assert "regime_change" in regimes.columns


class TestGenerateRandomCorrelationMatrix:
    def test_valid_correlation_matrix(self):
        corr = generate_random_correlation_matrix(5, random_state=42)
        assert corr.shape == (5, 5)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(corr)
        assert np.all(eigvals >= -1e-10)

    def test_concentrated_eigenvalues(self):
        corr = generate_random_correlation_matrix(10, eigenvalue_concentration=0.9, random_state=42)
        eigvals = np.linalg.eigvalsh(corr)
        assert eigvals[-1] / np.sum(eigvals) > 0.5  # First eigenvalue dominates

    def test_deterministic(self):
        c1 = generate_random_correlation_matrix(4, random_state=123)
        c2 = generate_random_correlation_matrix(4, random_state=123)
        np.testing.assert_array_equal(c1, c2)
