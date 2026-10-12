"""One strategy aggregation, owned by the strategy runner (TOM-1712).

There were four: backtest_pipeline._aggregate_weights_history,
trading_pipeline._aggregate_strategies (last row, with a silent manual fallback),
trading_pipeline._aggregate_history, and the weighted-average aggregator plugin.
Nothing made them agree. These tests go through the runner's interface,
`decided_weights` + `latest_weights`, and through the aggregator plugin that now
returns the same thing.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

import quantbox.plugins.pipeline.backtest_pipeline as backtest_pipeline
import quantbox.plugins.pipeline.trading_pipeline as trading_pipeline
from quantbox.plugins.strategies.weighted_avg_aggregator import WeightedAverageAggregator
from quantbox.strategy_runner import decided_weights, latest_weights

IDX = pd.date_range("2026-01-01", periods=3, freq="D")


def _result(frame: pd.DataFrame | None, weight: float = 1.0) -> dict:
    return {"result": {"weights": frame}, "weight": weight}


def _w(**cols: list[float]) -> pd.DataFrame:
    return pd.DataFrame(cols, index=IDX)


def test_one_strategy_is_scaled_by_its_account_weight():
    history = decided_weights({"a": _result(_w(BTC=[0.2, 0.4, 0.6]), weight=0.5)})
    pd.testing.assert_frame_equal(history, _w(BTC=[0.1, 0.2, 0.3]))
    assert latest_weights(history) == {"BTC": 0.3}


def test_strategies_are_weighted_and_summed_per_ticker():
    results = {
        "trend": _result(_w(BTC=[0.5, 0.5, 0.5], ETH=[0.1, 0.2, 0.3]), weight=0.6),
        "carry": _result(_w(BTC=[-0.5, 0.0, 0.5]), weight=0.4),
    }
    history = decided_weights(results)
    assert latest_weights(history) == pytest.approx({"BTC": 0.6 * 0.5 + 0.4 * 0.5, "ETH": 0.6 * 0.3})
    assert history.loc[IDX[0], "BTC"] == pytest.approx(0.6 * 0.5 - 0.4 * 0.5)


def test_strategy_weights_override_the_config_weight():
    results = {"a": _result(_w(BTC=[1.0, 1.0, 1.0]), weight=1.0), "b": _result(_w(ETH=[1.0, 1.0, 1.0]), weight=1.0)}
    assert latest_weights(decided_weights(results, {"b": 0.25})) == {"BTC": 1.0, "ETH": 0.25}


def test_an_empty_strategy_before_a_live_one_keeps_the_weights_paired():
    """The backtest copy keyed the frames by `names[:len(frames)]`: an empty first
    strategy shifted every later strategy onto its neighbour's account weight."""
    results = {
        "empty": _result(pd.DataFrame(), weight=0.9),
        "a": _result(_w(BTC=[1.0, 1.0, 1.0]), weight=0.3),
        "b": _result(_w(ETH=[1.0, 1.0, 1.0]), weight=0.7),
        "none": _result(None, weight=5.0),
    }
    assert latest_weights(decided_weights(results)) == pytest.approx({"BTC": 0.3, "ETH": 0.7})


def test_no_strategy_with_weights_is_an_empty_history():
    history = decided_weights({"a": _result(pd.DataFrame()), "b": _result(None)})
    assert history.empty
    assert latest_weights(history) == {}


def test_the_aggregator_plugin_returns_the_runner_aggregation():
    results = {
        "trend": _result(_w(BTC=[0.5, 0.5, 0.5], ETH=[0.1, 0.2, 0.3]), weight=0.6),
        "carry": _result(_w(BTC=[-0.5, 0.0, 0.5]), weight=0.4),
    }
    out = WeightedAverageAggregator().run({"strategy_results": results}, {"strategy_weights": {"carry": 0.1}})
    expected = decided_weights(results, {"carry": 0.1})
    pd.testing.assert_frame_equal(out["weights"], expected)
    assert out["simple_weights"] == latest_weights(expected)


def test_one_aggregation_in_src():
    """The search sweep the card asks for: no pipeline keeps a copy of its own."""
    for module in (trading_pipeline, backtest_pipeline):
        source = inspect.getsource(module)
        for name in ("_aggregate_strategies", "_aggregate_history", "_aggregate_weights_history"):
            assert name not in source, f"{module.__name__} still has {name}"
        assert "decided_weights(" in source
