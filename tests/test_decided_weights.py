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


# ── through the trading pipeline's run(): the built-in path and an injected aggregator ──


class _Store:
    run_id = "testrun"

    def __init__(self):
        self.written: dict = {}

    def put_parquet(self, name, df):
        self.written[name] = df
        return f"artifacts/{name}.parquet"

    def put_json(self, name, obj):
        self.written[name] = obj
        return f"artifacts/{name}.json"


class _Data:
    def load_universe(self, params):
        return pd.DataFrame({"symbol": ["BTC", "ETH"]})

    def load_market_data(self, universe, asof, params):
        return {"prices": _w(BTC=[100.0, 101.0, 102.0], ETH=[10.0, 10.5, 11.0])}


def _strategy(name: str, frame: pd.DataFrame):
    class _S:
        meta = type("M", (), {"name": name})()

        def run(self, *, data, params, context=None):
            return {"weights": frame}

    return _S()


def _run_pipeline(pipe=None, **kwargs) -> dict:
    store = _Store()
    (pipe or trading_pipeline.TradingPipeline()).run(
        mode="backtest",
        asof=str(IDX[-1].date()),
        params={"strategies": [{"name": "a", "weight": 0.6}, {"name": "b", "weight": 0.4}]},
        data=_Data(),
        store=store,
        broker=None,
        risk=[],
        strategies=[
            _strategy("a", _w(BTC=[0.5, 0.5, 0.5], ETH=[0.1, 0.2, 0.3])),
            _strategy("b", _w(BTC=[-0.5, 0.0, 0.5])),
        ],
        **kwargs,
    )
    agg = store.written["aggregated_weights"]
    return dict(zip(agg["symbol"], agg["weight"], strict=True))


def test_the_pipeline_and_an_injected_aggregator_decide_the_same_weights():
    built_in = _run_pipeline()
    injected = _run_pipeline(aggregator=WeightedAverageAggregator())
    assert built_in == pytest.approx({"BTC": 0.6 * 0.5 + 0.4 * 0.5, "ETH": 0.6 * 0.3})
    assert injected == pytest.approx(built_in)


def test_a_run_notifier_serves_that_run_only():
    """TOM-1720: run(notifier=...) opens that run's channel and never sticks to the pipeline."""
    from quantbox.trade.notifier import RecordingNotifier

    pipe = trading_pipeline.TradingPipeline()
    fake = RecordingNotifier()
    _run_pipeline(pipe, notifier=fake)
    assert pipe._alert("during", site="frozen") is True
    assert fake.messages == ["during"]
    assert pipe.notifier is None and trading_pipeline.TradingPipeline.notifier is None
    _run_pipeline(pipe)  # the next run, no notifier: the fake is gone
    assert pipe._alerts is None
