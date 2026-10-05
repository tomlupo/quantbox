"""The rebalancing policy in live trading: refusals, aliases and the unchanged default path (TOM-1518).

The parity of the decision itself (backtest seam vs trading pipeline, bar by
bar) is ``tests/test_live_policy_parity.py``. This file holds the rest of what
the card asks: the live rebalancers carry no tranche copy, the deprecated
``tranches`` keys are the policy's tranche cadence, one trade floor is declared
once, and a config without a policy trades exactly as before.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantbox.engine.policy import decide_rebalance, next_execution_bar
from quantbox.plugins.broker.sim import SimPaperBroker
from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline
from quantbox.plugins.rebalancing.futures_rebalancer import FuturesRebalancer
from quantbox.plugins.rebalancing.standard_rebalancer import StandardRebalancer
from quantbox.store import FileArtifactStore

DAY = pd.Timedelta("1D")


def _history(end: str, n: int = 20) -> pd.DataFrame:
    idx = pd.date_range(end=end, periods=n, freq="D")
    a = np.linspace(0.2, 0.6, n)
    return pd.DataFrame({"A": a, "B": 0.8 - a}, index=idx)


# ----------------------------------------------------------------------
# decide_rebalance: what a live run can and cannot know
# ----------------------------------------------------------------------


def test_weekly_is_considered_on_the_last_bar_of_the_week_only():
    """Sunday 2024-03-10 ends a Monday-Sunday week: the next bar (Monday) is in a later week."""
    sunday = _history("2024-03-10")
    on = decide_rebalance({"policy": "periodic", "frequency": "weekly"}, sunday, next_bar=pd.Timestamp("2024-03-11"))
    assert on.considered and on.placed
    assert on.targets.to_dict() == pytest.approx(sunday.iloc[-1].to_dict())

    saturday = _history("2024-03-09")
    off = decide_rebalance({"policy": "periodic", "frequency": "weekly"}, saturday, next_bar=pd.Timestamp("2024-03-10"))
    assert not off.considered and not off.placed
    assert off.trades == ()


def test_a_period_end_frequency_without_the_next_bar_is_refused():
    with pytest.raises(ValueError, match="next execution bar"):
        decide_rebalance({"policy": "periodic", "frequency": "monthly"}, _history("2024-03-31"))


@pytest.mark.parametrize("frequency", [5, None], ids=["every-5th-bar", "buy-and-hold"])
def test_a_frequency_anchored_to_the_first_bar_of_the_data_is_refused(frequency):
    """The live history window moves every run, so the n-th bar (or the first bar) moves with it."""
    with pytest.raises(ValueError, match="live"):
        decide_rebalance({"policy": "periodic", "frequency": frequency}, _history("2024-03-31"))


def test_a_trigger_without_the_held_book_is_refused():
    with pytest.raises(ValueError, match="held book"):
        decide_rebalance({"policy": "band", "band": 0.02}, _history("2024-03-31"))


def test_tranche_targets_are_the_mean_of_the_last_n_decisions():
    hist = _history("2024-03-31")
    out = decide_rebalance({"policy": "tranche", "tranches": 3}, hist)
    assert out.targets.to_dict() == pytest.approx(hist.iloc[-3:].mean().to_dict())


def test_a_corridor_hit_trades_the_whole_book_and_a_quiet_book_trades_nothing():
    hist = _history("2024-03-31")
    target = hist.iloc[-1]
    policy = {"policy": "corridor", "width": 0.05}
    quiet = decide_rebalance(policy, hist, target + pd.Series({"A": 0.01, "B": -0.01}))
    assert quiet.considered and not quiet.placed and quiet.hit is False
    hit = decide_rebalance(policy, hist, target + pd.Series({"A": 0.10, "B": 0.0}))
    assert hit.placed and set(hit.trades) == {"A", "B"}  # B is inside its corridor and trades too


def test_next_execution_bar_skips_a_market_holiday():
    """Thu 2024-03-28 (NYSE): Good Friday and the weekend are closed, so the next session is Mon 1 April."""
    assert next_execution_bar(pd.Timestamp("2024-03-28"), DAY, "NYSE") == pd.Timestamp("2024-04-01")
    assert next_execution_bar(pd.Timestamp("2024-03-28"), DAY, "24/7") == pd.Timestamp("2024-03-29")


# ----------------------------------------------------------------------
# The live rebalancers: target weights -> orders, no tranche copy
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", [StandardRebalancer, FuturesRebalancer])
def test_a_rebalancer_refuses_tranches_handed_to_it_directly(cls):
    with pytest.raises(ValueError, match="rebalancing_policy"):
        cls()._apply_risk_transforms({"A": 0.5}, {"tranches": 3})


@pytest.mark.parametrize("cls", [StandardRebalancer, FuturesRebalancer])
def test_a_rebalancer_with_tranches_one_still_runs(cls):
    """Both quantbox-live configs declare `tranches: 1` on the rebalancer: still accepted."""
    assert cls()._apply_risk_transforms({"A": 0.5}, {"tranches": 1}) == {"A": 0.5}


# ----------------------------------------------------------------------
# The trading pipeline
# ----------------------------------------------------------------------

ASOF = "2024-03-30"


def _prices() -> pd.DataFrame:
    idx = pd.date_range(end=ASOF, periods=30, freq="D")
    rng = np.random.default_rng(1)
    a = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, 30))
    return pd.DataFrame({"A": a, "B": 50.0}, index=idx)


class _Data:
    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame({"symbol": ["A", "B"]})

    def load_market_data(self, universe: Any, asof: str, params: dict[str, Any]) -> dict[str, pd.DataFrame]:
        return {"prices": _prices()}


class _Strategy:
    meta = type("M", (), {"name": "strategy.ramp.v1"})()

    def run(self, data: Any, params: Any = None, context: Any = None) -> dict[str, Any]:
        idx = data["prices"].index
        a = np.linspace(0.1, 0.7, len(idx))
        return {"weights": pd.DataFrame({"A": a, "B": 0.2}, index=idx)}


class _Spy(StandardRebalancer):
    """Records every generate_orders call (the weights it was asked to size)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, float]] = []
        self.params_seen: list[dict[str, Any]] = []

    def generate_orders(self, *, weights, broker, params):
        self.calls.append(dict(weights))
        self.params_seen.append(dict(params))
        return super().generate_orders(weights=weights, broker=broker, params=params)


def _run(tmp_path, params: dict[str, Any], rebal_params: dict[str, Any] | None = None, broker=None):
    spy = _Spy()
    full = {
        "strategies": [{"name": "strategy.ramp.v1", "weight": 1.0, "params": {}}],
        "stable_coin_symbol": "USD",
        "trading_enabled": False,
        "_rebalancer_cfg": {"params": dict(rebal_params or {})},
        **params,
    }
    result = TradingPipeline().run(
        mode="paper",
        asof=ASOF,
        params=full,
        data=_Data(),
        store=FileArtifactStore(str(tmp_path / "run"), "run"),
        broker=broker or SimPaperBroker(cash=10_000.0, quote_currency="USD"),
        risk=[],
        strategies=[_Strategy()],
        rebalancer=spy,
    )
    return result, spy


def test_without_a_policy_the_run_trades_the_last_decided_row_as_before(tmp_path):
    """Both quantbox-live books: no rebalancing_policy, rebalancer tranches 1 -> one sizing pass, last row."""
    result, spy = _run(tmp_path, {}, {"tranches": 1, "min_trade_size": 0.01})
    assert "rebalancing_policy" not in result.notes
    assert len(spy.calls) == 1
    assert spy.calls[0] == pytest.approx({"A": 0.7, "B": 0.2})
    assert spy.params_seen[0]["min_trade_size"] == 0.01  # the legacy order floor is untouched
    assert "tranches" not in spy.params_seen[0]


@pytest.mark.parametrize(
    ("params", "rebal_params", "key"),
    [
        ({"risk": {"tranches": 3}}, {}, "risk.tranches"),
        ({}, {"tranches": 3}, "rebalancing.params.tranches"),
    ],
    ids=["risk.tranches", "rebalancer.tranches"],
)
def test_the_deprecated_tranches_keys_are_the_policy_tranche_cadence(tmp_path, params, rebal_params, key):
    with pytest.warns(DeprecationWarning, match=key.replace(".", r"\.")):
        result, spy = _run(tmp_path, params, rebal_params)
    record = result.notes["rebalancing_policy"]
    assert record["policy"]["cadence"] == "tranche"
    assert record["policy"]["tranches"] == 3
    expected_a = float(np.linspace(0.1, 0.7, 30)[-3:].mean())
    assert record["targets"]["A"] == pytest.approx(expected_a)
    assert spy.calls[-1]["A"] == pytest.approx(expected_a)
    assert all("tranches" not in p for p in spy.params_seen)


def test_tranches_declared_twice_are_refused(tmp_path):
    with pytest.raises(ValueError, match="both declare tranches"):
        _run(tmp_path, {"risk": {"tranches": 3}}, {"tranches": 2})
    with pytest.raises(ValueError, match="both declare tranches"):
        _run(tmp_path, {"rebalancing_policy": {"policy": "tranche", "tranches": 4}, "risk": {"tranches": 2}})


@pytest.mark.parametrize("where", ["pipeline", "rebalancer"])
def test_min_trade_size_beside_a_declared_policy_is_refused(tmp_path, where):
    params = {"rebalancing_policy": {"policy": "periodic", "min_trade": 0.02}}
    rebal: dict[str, Any] = {}
    if where == "pipeline":
        params["min_trade_size"] = 0.01
    else:
        rebal["min_trade_size"] = 0.01
    with pytest.raises(ValueError, match="min_trade"):
        _run(tmp_path, params, rebal)


def test_a_declared_policy_owns_the_trade_floor(tmp_path):
    """The rebalancer's own min_trade_size is 0 under a declared policy: one floor, the policy's."""
    _, spy = _run(tmp_path, {"rebalancing_policy": {"policy": "periodic", "min_trade": 0.02}})
    assert spy.params_seen and all(p["min_trade_size"] == 0.0 for p in spy.params_seen)


def test_a_bar_that_is_not_considered_sends_no_orders_and_holds_the_book(tmp_path):
    """Monthly on 2024-03-30 (a Saturday, 24/7 data): March ends tomorrow, so nothing trades today."""
    broker = SimPaperBroker(cash=5_000.0, quote_currency="USD", positions={"A": 10.0})
    result, spy = _run(tmp_path, {"rebalancing_policy": {"policy": "periodic", "frequency": "monthly"}}, broker=broker)
    record = result.notes["rebalancing_policy"]
    assert record["considered"] is False and record["placed"] is False
    assert len(spy.calls) == 1  # one valuation pass, no sizing pass
    assert result.metrics["n_orders"] == 0.0
    targets = pd.read_parquet(result.artifacts["targets"])
    assert set(targets["symbol"]) == {"A"}  # the held book, not the decided row


class _Levered:
    meta = type("M", (), {"name": "strategy.levered.v1"})()

    def run(self, data: Any, params: Any = None, context: Any = None) -> dict[str, Any]:
        return {"weights": pd.DataFrame({"A": 2.0, "B": 0.0}, index=data["prices"].index)}


@pytest.mark.parametrize("path", ["rebalancer", "pipeline"])
def test_the_trigger_reads_the_risk_transformed_targets_as_the_backtest_does(tmp_path, path):
    """Round 1 of #244: raw target A=2.0 under max_leverage 1 is A=1.0, which the book already holds.

    The backtest caps every decided row before the seam, so a band of 0.5 sees
    no drift. Live must too: a trigger on the raw 2.0 would fire a rebalance
    the order generator then clamps back to the held book.
    """
    price_a = float(_prices()["A"].iloc[-1])
    broker = SimPaperBroker(cash=0.0, quote_currency="USD", positions={"A": 10_000.0 / price_a})
    params = {
        "strategies": [{"name": "strategy.levered.v1", "weight": 1.0, "params": {}}],
        "stable_coin_symbol": "USD",
        "trading_enabled": False,
        "rebalancing_policy": {"policy": "band", "band": 0.5},
        "risk": {"max_leverage": 1.0},
        "_rebalancer_cfg": {"params": {"max_leverage": 1.0}},
    }
    result = TradingPipeline().run(
        mode="paper",
        asof=ASOF,
        params=params,
        data=_Data(),
        store=FileArtifactStore(str(tmp_path / path), "run"),
        broker=broker,
        risk=[],
        strategies=[_Levered()],
        rebalancer=StandardRebalancer() if path == "rebalancer" else None,
    )
    record = result.notes["rebalancing_policy"]
    assert record["policy_targets"]["A"] == pytest.approx(1.0)
    assert record["placed"] is False, record["reason"]


def test_an_untraded_instrument_is_held_by_the_policy():
    orders = pd.DataFrame(
        {
            "Asset": ["A", "B"],
            "Order Status": ["To be placed", "To be placed"],
            "Reason": ["", ""],
            "Adjusted Quantity": [1.0, 2.0],
            "Executable": [True, True],
        }
    )
    out = TradingPipeline._hold_untraded(orders, ("A",))
    assert out["Executable"].tolist() == [True, False]
    assert out.loc[1, "Order Status"] == "Held by policy"
    assert out.loc[1, "Adjusted Quantity"] == 0.0
