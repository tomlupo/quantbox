"""Tests for quantbox.plugins.backtesting.optimizer module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.backtesting.optimizer import optimize


@pytest.fixture
def prices():
    """Synthetic price data for 5 assets, 400 days."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2024-01-01", periods=400)
    data = {}
    for sym in ["A", "B", "C", "D", "E"]:
        data[sym] = 100 * np.exp(np.cumsum(rng.randn(400) * 0.01 + 0.0002))
    return pd.DataFrame(data, index=dates)


def _equal_weight_fn(prices: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Simple equal-weight strategy that ignores params."""
    n = len(prices.columns)
    return pd.DataFrame(
        1.0 / n,
        index=prices.index,
        columns=prices.columns,
    )


def _momentum_fn(prices: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Momentum strategy parameterized by lookback."""
    lookback = params.get("lookback", 20)
    rets = prices.pct_change(lookback).iloc[lookback:]
    # Go long top 2 by momentum
    top_n = params.get("top_n", 2)
    weights = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    for i in range(len(rets)):
        row = rets.iloc[i]
        top = row.nlargest(top_n).index
        weights.iloc[i, weights.columns.get_indexer(top)] = 1.0 / top_n
    return weights


class TestGridSearch:
    def test_basic(self, prices):
        result = optimize(
            prices,
            _equal_weight_fn,
            {"dummy": [1, 2, 3]},
            method="grid",
            metric="sharpe",
        )
        assert "best_params" in result
        assert "best_metric" in result
        assert "all_results" in result
        assert len(result["all_results"]) == 3

    def test_best_params_returned(self, prices):
        result = optimize(
            prices,
            _momentum_fn,
            {"lookback": [10, 20, 50], "top_n": [2, 3]},
            method="grid",
            metric="sharpe",
        )
        assert "lookback" in result["best_params"]
        assert "top_n" in result["best_params"]
        assert len(result["all_results"]) == 6  # 3 * 2 combos

    def test_metric_is_numeric(self, prices):
        result = optimize(
            prices,
            _equal_weight_fn,
            {"dummy": [1]},
            method="grid",
            metric="sharpe",
        )
        assert isinstance(result["best_metric"], (int, float))

    def test_empty_result_on_failure(self, prices):
        def bad_fn(p, params):
            raise ValueError("fail")

        result = optimize(prices, bad_fn, {"x": [1]}, method="grid", metric="sharpe")
        assert result["best_params"] == {}
        assert result["all_results"].empty


class TestWalkForward:
    def test_basic(self, prices):
        result = optimize(
            prices,
            _equal_weight_fn,
            {"dummy": [1, 2]},
            method="walk_forward",
            metric="sharpe",
            train_size=200,
            test_size=100,
        )
        assert "best_params" in result
        assert "oos_results" in result
        assert len(result["oos_results"]) >= 1

    def test_oos_metrics_present(self, prices):
        result = optimize(
            prices,
            _equal_weight_fn,
            {"dummy": [1]},
            method="walk_forward",
            metric="sharpe",
            train_size=150,
            test_size=100,
        )
        if not result["oos_results"].empty:
            assert "oos_sharpe" in result["oos_results"].columns

    def test_insufficient_data(self, prices):
        result = optimize(
            prices,
            _equal_weight_fn,
            {"dummy": [1]},
            method="walk_forward",
            metric="sharpe",
            train_size=300,
            test_size=200,
        )
        assert result["oos_results"].empty or len(result["oos_results"]) <= 1
