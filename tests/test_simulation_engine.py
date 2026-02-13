"""Tests for quantbox.simulation.engine — MarketSimulator orchestrator."""

import numpy as np
import pandas as pd
import pytest

from quantbox.simulation.engine import (
    MarketSimulator,
    SimulationConfig,
    generate_correlated_returns,
)
from quantbox.simulation.models import GBM, GBMParams


class TestSimulationConfig:
    def test_defaults(self):
        cfg = SimulationConfig()
        assert cfg.n_paths == 10000
        assert cfg.n_steps == 252
        assert cfg.random_state is None


class TestMarketSimulator:
    def test_single_asset(self):
        sim = MarketSimulator()
        sim.add_asset("SPY", GBM(GBMParams(mu=0.08, sigma=0.18)), initial_price=450)
        result = sim.simulate(SimulationConfig(n_paths=100, n_steps=50, random_state=42))
        assert result.prices.shape == (1, 100, 51)
        assert result.asset_names == ["SPY"]

    def test_multi_asset(self):
        sim = MarketSimulator()
        sim.add_asset("SPY", GBM(GBMParams(mu=0.08, sigma=0.18)), initial_price=450)
        sim.add_asset("TLT", GBM(GBMParams(mu=0.03, sigma=0.10)), initial_price=100)
        result = sim.simulate(SimulationConfig(n_paths=50, n_steps=30, random_state=42))
        assert result.prices.shape == (2, 50, 31)
        assert result.returns.shape == (2, 50, 30)

    def test_correlated_simulation(self):
        sim = MarketSimulator()
        sim.add_asset("A", GBM(), initial_price=100)
        sim.add_asset("B", GBM(), initial_price=100)
        corr = np.array([[1.0, 0.9], [0.9, 1.0]])
        sim.set_correlation_matrix(corr)
        result = sim.simulate(SimulationConfig(n_paths=5000, n_steps=100, random_state=42))
        # Returns should be highly correlated
        r_A = result.returns[0].flatten()
        r_B = result.returns[1].flatten()
        empirical_corr = np.corrcoef(r_A, r_B)[0, 1]
        assert empirical_corr > 0.7  # Wide tolerance for MC

    def test_initial_prices(self):
        sim = MarketSimulator()
        sim.add_asset("X", GBM(), initial_price=42.5)
        result = sim.simulate(SimulationConfig(n_paths=10, n_steps=5, random_state=1))
        np.testing.assert_allclose(result.prices[0, :, 0], 42.5)

    def test_from_historical_data(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", periods=300)
        prices = pd.DataFrame(
            {
                "SPY": 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 300)),
                "TLT": 50 * np.cumprod(1 + rng.normal(0.0001, 0.008, 300)),
            },
            index=dates,
        )
        sim = MarketSimulator.from_historical_data(prices, model_type="gbm")
        assert "SPY" in sim.models
        assert "TLT" in sim.models
        assert sim.correlation_matrix is not None

    def test_builder_pattern(self):
        sim = MarketSimulator()
        ret = sim.add_asset("A", GBM(), 100)
        assert ret is sim  # Returns self for chaining


class TestSimulationResult:
    @pytest.fixture
    def result(self):
        sim = MarketSimulator()
        sim.add_asset("A", GBM(GBMParams(mu=0.08, sigma=0.18)), 100)
        sim.add_asset("B", GBM(GBMParams(mu=0.05, sigma=0.12)), 50)
        return sim.simulate(SimulationConfig(n_paths=500, n_steps=100, random_state=42))

    def test_get_terminal_prices(self, result):
        tp = result.get_terminal_prices()
        assert isinstance(tp, pd.DataFrame)
        assert list(tp.columns) == ["A", "B"]
        assert len(tp) == 500

    def test_get_terminal_returns(self, result):
        tr = result.get_terminal_returns()
        assert isinstance(tr, pd.DataFrame)
        assert list(tr.columns) == ["A", "B"]

    def test_get_path_statistics(self, result):
        stats = result.get_path_statistics()
        assert isinstance(stats, pd.DataFrame)
        assert "mean_return" in stats.columns
        assert "var_95" in stats.columns
        assert "sharpe_ratio" in stats.columns

    def test_get_percentile_paths(self, result):
        pp = result.get_percentile_paths("A")
        assert isinstance(pp, pd.DataFrame)
        assert "p50" in pp.columns
        assert len(pp) == 101  # n_steps + 1


class TestGenerateCorrelatedReturns:
    def test_shape(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        returns = generate_correlated_returns(
            n_assets=2,
            n_steps=50,
            n_paths=100,
            means=np.array([0.0003, 0.0001]),
            stds=np.array([0.01, 0.008]),
            correlation_matrix=corr,
            random_state=42,
        )
        assert returns.shape == (2, 100, 50)
