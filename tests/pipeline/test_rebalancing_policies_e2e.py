"""End-to-end smoke: a backtest run with a rebalancing policy and group limits (TOM-1450 3d-2).

It lives under ``tests/pipeline/`` because CI's smoke job runs
``pytest tests/pipeline/ -m pipeline_smoke`` (TOM-1500); in
``tests/test_rebalancing_policies.py`` that job never collected it. The unit
cases of the policies and group limits stay there.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

UNIVERSE = pd.DataFrame({"symbol": ["A", "B", "C"], "asset_class": ["equity", "equity", "bond"]})


def _daily_247(start: str, end: str, tickers: tuple[str, ...], seed: int) -> pd.DataFrame:
    """A 24/7 daily panel (crypto-style: weekends and US holidays print)."""
    idx = pd.date_range(start, end, freq="D")
    steps = np.random.default_rng(seed).normal(0.0005, 0.02, size=(len(idx), len(tickers)))
    return pd.DataFrame(100.0 * np.cumprod(1.0 + steps, axis=0), index=idx, columns=list(tickers))


class _Fixed:
    meta = type("M", (), {"name": "strategy.fixed.v1"})()

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def run(self, data: Any, params: Any = None, context: Any = None) -> dict[str, Any]:
        return {"weights": self.frame}


class _Data:
    def __init__(self, prices: pd.DataFrame, universe: pd.DataFrame):
        self.prices, self.universe = prices, universe

    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        return self.universe

    def load_market_data(self, universe: Any, asof: str, params: dict[str, Any]) -> dict[str, pd.DataFrame]:
        return {"prices": self.prices}


@pytest.mark.pipeline_smoke
def test_a_backtest_run_with_a_policy_and_group_limits_end_to_end(tmp_path):
    from quantbox.instrument_calendar import validate_data_validation
    from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline
    from quantbox.store import FileArtifactStore

    prices = _daily_247("2024-01-01", "2024-06-30", tickers=("A", "B", "C"), seed=9)
    decided = pd.DataFrame({"A": 0.5, "B": 0.3, "C": 0.2}, index=prices.index)
    store = FileArtifactStore(str(tmp_path), "run")
    params = {
        "engine": "rsims",
        "fees": 0.0,
        "strategies": [{"name": "strategy.fixed.v1", "weight": 1.0}],
        "rebalancing_policy": {"policy": "periodic", "frequency": "monthly", "calendar": "NYSE"},
        "group_limits": {"by": "asset_class", "limits": {"equity": {"max": 0.6}}},
    }
    result = BacktestPipeline().run(
        mode="backtest",
        asof="2024-06-30",
        params=params,
        data=_Data(prices, UNIVERSE),
        store=store,
        broker=None,
        risk=[],
        strategies=[_Fixed(decided)],
    )
    validation = json.loads((store.root / "data_validation.json").read_text())
    assert validate_data_validation(validation) == []
    assert validation["rebalancing"]["policy"]["calendar"] == "NYSE"
    assert validation["groups"]["limits"] == {"equity": {"min": 0.0, "max": 0.6}}
    traded = store.read_parquet("traded_weights").set_index("date")
    assert traded.iloc[-1][["A", "B"]].sum() == pytest.approx(0.6)
    assert traded.iloc[-1]["C"] == pytest.approx(0.4)
    schedule = store.read_parquet("rebalance_schedule")
    assert [d.strftime("%Y-%m-%d") for d in schedule["decision_date"]][:3] == ["2024-01-31", "2024-02-29", "2024-03-28"]
    assert result.metrics["rebalance_placed"] == 5.0
    assert result.notes["rebalancing"]["policy"] == "periodic"
