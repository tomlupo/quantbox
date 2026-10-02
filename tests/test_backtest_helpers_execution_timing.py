"""`backtest()` and `optimize()` take the same execution timing as the pipeline (TOM-1337).

Same toy as ``test_execution_timing``: one asset `A` jumps +10% on bar ``J``.
A weight decided on ``J-1`` fills at close[J], after the jump, so it earns
nothing; same-bar (which would earn it) is refused (docs/adr/0005).
"""

from __future__ import annotations

import pandas as pd
import pytest
from test_execution_timing import JUMP, J, _prices, _run_pipeline, _weights_decided_on

from quantbox.plugins.backtesting import _lag_for_engine, backtest, optimize


def _decided_fn(prices, params):
    return _weights_decided_on(params["decided_on"]).loc[prices.index]


@pytest.mark.parametrize(("lag", "expected"), [(None, 0.0), (1, 0.0), (2, 0.0)])
def test_backtest_agrees_with_the_pipeline_on_the_toy(tmp_path, lag, expected):
    kwargs = {} if lag is None else {"lag_bars": lag}
    result = backtest(_prices(), _weights_decided_on(J - 1), fees=0.0, **kwargs)
    params = {} if lag is None else {"execution": {"lag_bars": lag}}
    pipeline_return = _run_pipeline(tmp_path, params, decided_on=J - 1)[0].metrics["total_return"]
    assert result["metrics"]["total_return"] == pytest.approx(expected, abs=1e-9)
    assert result["metrics"]["total_return"] == pytest.approx(pipeline_return, abs=1e-9)


def test_backtest_default_a_decision_two_bars_before_the_jump_earns_it():
    result = backtest(_prices(), _weights_decided_on(J - 2), fees=0.0)
    assert result["metrics"]["total_return"] == pytest.approx(JUMP, abs=1e-9)


def test_each_strategy_of_a_dict_is_lagged_on_the_engine_grid():
    """A dict of weights is lagged per strategy, one bar on the price grid each.

    (Through ``backtest()`` a dict trips ``compute_backtest_metrics`` on grouped
    returns, before and after this change, so the lag is checked at its seam.)
    """
    prices = _prices()
    sparse = _weights_decided_on(J - 1).iloc[[0, J - 1]]  # decided on two dates only
    lagged = _lag_for_engine(prices, {"late": sparse, "same": _weights_decided_on(J - 2)}, 1)
    assert pd.isna(lagged["late"]["A"].iloc[J - 1])  # no decision here: the engine ffills it
    assert lagged["late"]["A"].iloc[J] == 1.0  # one PRICE bar later, not one weights row later
    assert lagged["same"]["A"].iloc[J - 1] == 1.0
    with pytest.raises(ValueError, match="lag_bars must be >= 1"):
        _lag_for_engine(prices, sparse, 0)


def test_backtest_records_the_execution_timing():
    assert backtest(_prices(), _weights_decided_on(J - 1), fees=0.0)["execution"]["lag_bars"] == 1
    assert "next-bar (lag_bars=1)" in backtest(_prices(), _weights_decided_on(J - 1))["execution"]["description"]


@pytest.mark.parametrize("bad", [0, -1, 1.0, "1", True])
def test_backtest_refuses_a_malformed_lag(bad):
    with pytest.raises(ValueError):
        backtest(_prices(), _weights_decided_on(J - 1), lag_bars=bad)


@pytest.mark.parametrize(("lag", "expected"), [(None, 0.0), (2, 0.0)])
def test_optimize_agrees_with_the_pipeline_on_the_toy(tmp_path, lag, expected):
    kwargs = {} if lag is None else {"lag_bars": lag}
    result = optimize(_prices(), _decided_fn, {"decided_on": [J - 1]}, metric="total_return", fees=0.0, **kwargs)
    params = {} if lag is None else {"execution": {"lag_bars": lag}}
    pipeline_return = _run_pipeline(tmp_path, params, decided_on=J - 1)[0].metrics["total_return"]
    assert result["best_metric"] == pytest.approx(expected, abs=1e-9)
    assert result["best_metric"] == pytest.approx(pipeline_return, abs=1e-9)
    assert result["execution"]["lag_bars"] == (1 if lag is None else lag)


def test_optimize_refuses_same_bar():
    with pytest.raises(ValueError, match="lag_bars must be >= 1"):
        optimize(_prices(), _decided_fn, {"decided_on": [J - 1, J - 2, J - 3]}, fees=0.0, lag_bars=0)


@pytest.mark.parametrize("helper", ["bt.run", "from_signals_with_costs"])
def test_the_signal_helpers_trade_next_bar_and_refuse_same_bar(helper):
    """The L1 signal helpers used to fill at the signal's own close (ADR-0005).

    A signal switched on at ``J-1`` must buy at close[J], after the jump.
    """
    if helper == "bt.run":
        import quantbox.bt as qbt

        def go(**kw):
            return qbt.run(_prices(), _weights_decided_on(J - 1), fees=0.0, slippage=0.0, **kw).portfolio
    else:
        from quantbox.adapters.vectorbt import from_signals_with_costs

        def go(**kw):
            return from_signals_with_costs(_prices(), _weights_decided_on(J - 1), fees=0.0, slippage=0.0, **kw)

    assert float(go().total_return().iloc[0]) == pytest.approx(0.0, abs=1e-9)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="lag_bars must be >= 1"):
            go(lag_bars=bad)


@pytest.mark.parametrize("name", ["run_vectorbt", "fixed_commission_backtest_with_funding"])
def test_the_same_bar_engine_primitives_are_not_exported(name):
    import quantbox.plugins.backtesting as b

    assert name not in b.__all__
    with pytest.raises(ImportError, match="backtest"):
        getattr(b, name)


def test_optimize_walk_forward_records_the_execution_timing():
    result = optimize(
        _prices(), _decided_fn, {"decided_on": [J - 1]}, method="walk_forward", train_size=10, test_size=10, fees=0.0
    )
    assert result["execution"]["lag_bars"] == 1


def _hold_prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame({"A": [100.0, 100.0, 110.0, 121.0, 133.1]}, index=idx)


@pytest.mark.parametrize(("lag", "expected"), [(1, 0.331), (2, 0.21)])
def test_buy_and_hold_enters_on_the_lagged_bar(lag, expected):
    """``rebalancing_freq=None`` trades once: the decision on bar 0 fills at close[lag].

    Lagging the weights alone left the one scheduled trade on bar 0, where the
    lagged weight is flat, so a next-bar buy-and-hold never entered and returned 0%.
    """
    prices = _hold_prices()
    weights = pd.DataFrame({"A": 1.0}, index=prices.index)
    result = backtest(prices, weights, fees=0.0, rebalancing_freq=None, lag_bars=lag)
    assert result["metrics"]["total_return"] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("lag", [None, 1, 2])
def test_buy_and_hold_agrees_with_the_pipeline(tmp_path, lag):
    """Both doors enter a buy-and-hold book at close[lag_bars]; the toy's later jump is earned."""
    kwargs = {} if lag is None else {"lag_bars": lag}
    result = backtest(_prices(), _weights_decided_on(0), fees=0.0, rebalancing_freq=None, **kwargs)
    params = {"rebalancing_freq": None}
    if lag is not None:
        params["execution"] = {"lag_bars": lag}
    pipeline_return = _run_pipeline(tmp_path, params, decided_on=0)[0].metrics["total_return"]
    assert result["metrics"]["total_return"] == pytest.approx(JUMP, abs=1e-9)
    assert pipeline_return == pytest.approx(JUMP, abs=1e-9)
