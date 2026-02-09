"""Tests for quantbox.simulation.models — 5 stochastic price models."""
import numpy as np
import pytest

from quantbox.simulation.models import (
    GBM, GBMParams,
    JumpDiffusion, JumpDiffusionParams,
    MeanReversion, MeanReversionParams,
    GARCH, GARCHParams,
    RegimeSwitching,
    ModelParameters,
)


class TestGBM:
    def test_simulate_shape(self):
        model = GBM(GBMParams(mu=0.08, sigma=0.20))
        prices = model.simulate(S0=100, n_steps=252, n_paths=50, random_state=42)
        assert prices.shape == (50, 253)

    def test_simulate_starts_at_S0(self):
        model = GBM()
        prices = model.simulate(S0=42.0, n_steps=10, n_paths=5, random_state=1)
        np.testing.assert_allclose(prices[:, 0], 42.0)

    def test_simulate_returns_shape(self):
        model = GBM()
        returns = model.simulate_returns(n_steps=100, n_paths=20, random_state=7)
        assert returns.shape == (20, 100)

    def test_positive_prices(self):
        model = GBM(GBMParams(mu=-0.5, sigma=0.8))
        prices = model.simulate(S0=100, n_steps=500, n_paths=100, random_state=42)
        assert np.all(prices > 0)

    def test_fit_from_returns(self):
        rng = np.random.default_rng(99)
        returns = rng.normal(0.0003, 0.012, 252)
        fitted = GBM.fit(returns)
        assert isinstance(fitted, GBM)
        assert fitted.params.sigma > 0

    def test_deterministic_with_seed(self):
        model = GBM()
        p1 = model.simulate(100, 50, 10, random_state=42)
        p2 = model.simulate(100, 50, 10, random_state=42)
        np.testing.assert_array_equal(p1, p2)


class TestJumpDiffusion:
    def test_simulate_shape(self):
        model = JumpDiffusion()
        prices = model.simulate(S0=100, n_steps=100, n_paths=30, random_state=42)
        assert prices.shape == (30, 101)

    def test_positive_prices(self):
        model = JumpDiffusion(JumpDiffusionParams(
            mu=0.05, sigma=0.25, jump_intensity=10, jump_mean=-0.03, jump_std=0.04,
        ))
        prices = model.simulate(S0=100, n_steps=252, n_paths=100, random_state=42)
        assert np.all(prices > 0)

    def test_simulate_returns_shape(self):
        model = JumpDiffusion()
        returns = model.simulate_returns(n_steps=50, n_paths=10, random_state=42)
        assert returns.shape == (10, 50)


class TestMeanReversion:
    def test_simulate_shape(self):
        model = MeanReversion()
        prices = model.simulate(S0=100, n_steps=200, n_paths=25, random_state=42)
        assert prices.shape == (25, 201)

    def test_positive_prices(self):
        model = MeanReversion(MeanReversionParams(theta=0.5, long_term_mean=4.6, sigma=0.15))
        prices = model.simulate(S0=100, n_steps=252, n_paths=50, random_state=42)
        assert np.all(prices > 0)


class TestGARCH:
    def test_simulate_shape(self):
        model = GARCH()
        prices = model.simulate(S0=100, n_steps=100, n_paths=20, random_state=42)
        assert prices.shape == (20, 101)

    def test_simulate_returns_shape(self):
        model = GARCH()
        returns = model.simulate_returns(n_steps=100, n_paths=20, random_state=42)
        assert returns.shape == (20, 100)

    def test_time_varying_volatility(self):
        """GARCH returns should have heteroskedastic clustering."""
        model = GARCH(GARCHParams(omega=0.00001, alpha=0.15, beta=0.80, sigma=0.20))
        returns = model.simulate_returns(n_steps=1000, n_paths=1, random_state=42)
        # Rolling variance should vary (not constant like GBM)
        rolling_var = np.array([np.var(returns[0, max(0, i-20):i+1]) for i in range(20, 1000)])
        assert np.std(rolling_var) > 0


class TestRegimeSwitching:
    def test_simulate_shape(self):
        model = RegimeSwitching()
        prices = model.simulate(S0=100, n_steps=252, n_paths=30, random_state=42)
        assert prices.shape == (30, 253)

    def test_positive_prices(self):
        model = RegimeSwitching()
        prices = model.simulate(S0=100, n_steps=500, n_paths=50, random_state=42)
        assert np.all(prices > 0)

    def test_simulate_returns_shape(self):
        model = RegimeSwitching()
        returns = model.simulate_returns(n_steps=100, n_paths=10, random_state=42)
        assert returns.shape == (10, 100)


class TestModelParameters:
    def test_default_values(self):
        p = ModelParameters()
        assert p.mu == 0.08
        assert p.sigma == 0.20
        assert abs(p.dt - 1/252) < 1e-10

    def test_gbm_params_inherits(self):
        p = GBMParams(mu=0.1, sigma=0.3)
        assert p.mu == 0.1
        assert p.sigma == 0.3
