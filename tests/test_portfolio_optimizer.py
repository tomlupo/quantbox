"""Tests for the portfolio optimizer strategy plugin."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.strategies.portfolio_optimizer import (
    PortfolioOptimizerStrategy,
    _PortfolioAnalyzer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """4 assets, 500 days of synthetic random-walk prices."""
    rng = np.random.default_rng(42)
    n_days, n_assets = 500, 4
    symbols = ["AAAA", "BBBB", "CCCC", "DDDD"]
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    log_returns = rng.normal(0.0003, 0.015, (n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=symbols)


@pytest.fixture
def sample_data(sample_prices: pd.DataFrame) -> dict:
    return {"prices": sample_prices}


@pytest.fixture
def analyzer(sample_prices: pd.DataFrame) -> _PortfolioAnalyzer:
    returns = sample_prices.pct_change().dropna()
    return _PortfolioAnalyzer(returns, risk_free_rate=0.02, trading_days=252)


# ---------------------------------------------------------------------------
# _PortfolioAnalyzer tests
# ---------------------------------------------------------------------------

class TestPortfolioAnalyzer:
    def test_equal_weight_sums_to_one(self, analyzer: _PortfolioAnalyzer) -> None:
        w = analyzer.equal_weight()
        assert pytest.approx(w.sum(), abs=1e-10) == 1.0

    def test_equal_weight_values(self, analyzer: _PortfolioAnalyzer) -> None:
        w = analyzer.equal_weight()
        expected = 1.0 / analyzer.n_assets
        np.testing.assert_allclose(w, expected)

    def test_risk_parity_sums_to_one(self, analyzer: _PortfolioAnalyzer) -> None:
        w = analyzer.risk_parity()
        assert pytest.approx(w.sum(), abs=1e-10) == 1.0

    def test_inverse_vol_equals_risk_parity(self, analyzer: _PortfolioAnalyzer) -> None:
        np.testing.assert_array_equal(analyzer.inverse_vol(), analyzer.risk_parity())

    def test_max_sharpe_sums_to_one(self, analyzer: _PortfolioAnalyzer) -> None:
        w = analyzer.optimize_sharpe()
        assert pytest.approx(w.sum(), abs=1e-6) == 1.0

    def test_min_variance_sums_to_one(self, analyzer: _PortfolioAnalyzer) -> None:
        w = analyzer.optimize_min_variance()
        assert pytest.approx(w.sum(), abs=1e-6) == 1.0

    def test_max_sharpe_respects_bounds(self, analyzer: _PortfolioAnalyzer) -> None:
        w = analyzer.optimize_sharpe(min_weight=0.05, max_weight=0.5)
        assert w.min() >= 0.05 - 1e-6
        assert w.max() <= 0.5 + 1e-6

    def test_min_variance_respects_bounds(self, analyzer: _PortfolioAnalyzer) -> None:
        w = analyzer.optimize_min_variance(min_weight=0.1, max_weight=0.4)
        assert w.min() >= 0.1 - 1e-6
        assert w.max() <= 0.4 + 1e-6

    def test_portfolio_performance_returns_tuple(self, analyzer: _PortfolioAnalyzer) -> None:
        w = analyzer.equal_weight()
        ret, vol, sharpe = analyzer.portfolio_performance(w)
        assert isinstance(ret, float)
        assert isinstance(vol, float)
        assert vol > 0

    def test_var_positive(self, analyzer: _PortfolioAnalyzer) -> None:
        w = analyzer.equal_weight()
        assert analyzer.var(w) > 0

    def test_cvar_gte_var(self, analyzer: _PortfolioAnalyzer) -> None:
        w = analyzer.equal_weight()
        assert analyzer.cvar(w) >= analyzer.var(w) - 1e-10

    def test_statistics_shape(self, analyzer: _PortfolioAnalyzer) -> None:
        stats = analyzer.statistics()
        assert stats.shape == (4, 3)
        assert list(stats.columns) == ["ann_return", "ann_vol", "sharpe"]


# ---------------------------------------------------------------------------
# Strategy run() tests
# ---------------------------------------------------------------------------

class TestPortfolioOptimizerStrategy:
    @pytest.mark.parametrize("method", [
        "max_sharpe", "min_variance", "equal_weight", "risk_parity", "inverse_vol",
    ])
    def test_run_all_methods(self, sample_data: dict, method: str) -> None:
        strategy = PortfolioOptimizerStrategy(method=method)
        result = strategy.run(sample_data)
        w = result["weights"]
        assert isinstance(w, pd.DataFrame)
        assert w.shape[0] == 1
        assert pytest.approx(w.iloc[0].sum(), abs=1e-4) == 1.0

    def test_run_returns_correct_structure(self, sample_data: dict) -> None:
        result = PortfolioOptimizerStrategy().run(sample_data)
        assert "weights" in result
        assert "simple_weights" in result
        assert "details" in result
        assert "statistics" in result["details"]
        assert "var" in result["details"]
        assert "cvar" in result["details"]

    def test_run_with_params_override(self, sample_data: dict) -> None:
        strategy = PortfolioOptimizerStrategy()
        result = strategy.run(sample_data, params={"method": "equal_weight"})
        w = result["weights"].iloc[0]
        expected = 1.0 / len(w)
        np.testing.assert_allclose(w.values, expected, atol=1e-10)

    def test_rolling_mode_shape(self, sample_data: dict) -> None:
        strategy = PortfolioOptimizerStrategy(
            method="equal_weight", rolling=True, lookback=100, output_periods=20,
        )
        result = strategy.run(sample_data)
        assert result["weights"].shape[0] == 20
        assert result["weights"].shape[1] == 4

    def test_rolling_weights_sum_to_one(self, sample_data: dict) -> None:
        strategy = PortfolioOptimizerStrategy(
            method="risk_parity", rolling=True, lookback=100, output_periods=10,
        )
        result = strategy.run(sample_data)
        row_sums = result["weights"].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-10)

    def test_invalid_method_raises(self, sample_data: dict) -> None:
        strategy = PortfolioOptimizerStrategy(method="bogus")
        with pytest.raises(ValueError, match="Unknown method"):
            strategy.run(sample_data)

    def test_meta_attributes(self) -> None:
        assert PortfolioOptimizerStrategy.meta.name == "strategy.portfolio_optimizer.v1"
        assert PortfolioOptimizerStrategy.meta.kind == "strategy"
