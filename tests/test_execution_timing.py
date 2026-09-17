"""Execution timing (`execution.lag_bars`) and venue constraints (`venue.allow_shorts`).

THE TOY. One asset `A` is flat at 100, jumps to 110 on bar ``J`` and is flat
after; `USD` is constant. The jump is the ONLY return in the series, so a
book's total return says exactly whether it held `A` across the jump.

Both engines are same-bar primitives (row ``t`` fills at ``close[t]``), so:

* a weight DECIDED on bar ``J-1`` (last bar before the jump), traded
  same-bar, buys at close[J-1]=100 and earns the +10%;
* under the default ``lag_bars: 1`` that same decision fills at close[J]=110
  — AFTER the jump — and earns ~0;
* a weight decided on bar ``J-2`` fills at close[J-1] under the default and
  earns the jump.

Every entry point that turns decided weights into a book must agree on this:
the config-run pipeline (vectorbt ``from_orders`` branch, vectorbt order-func
branch, rsims) and the parameter sweep.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantbox.analysis.parameter_grid import sweep
from quantbox.execution import (
    apply_execution_lag,
    exposure_metrics,
    resolve_allow_shorts,
    resolve_lag_bars,
    resolve_sweep_lag_bars,
)
from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline
from quantbox.store import FileArtifactStore

N = 40
J = 20
JUMP = 0.10


def _prices() -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=N, freq="D").values)
    a = np.full(N, 100.0)
    a[J:] = 100.0 * (1 + JUMP)
    return pd.DataFrame({"A": a, "USD": 100.0}, index=idx)


def _weights_decided_on(bar: int, *, sign: float = 1.0) -> pd.DataFrame:
    """Fully in `A` from the decision bar onward, flat before."""
    w = pd.DataFrame(0.0, index=_prices().index, columns=["A", "USD"])
    w.iloc[bar:, 0] = sign
    return w


class _FixedWeights:
    """Strategy stub: returns the weights it was built with (both call shapes)."""

    meta = type("M", (), {"name": "strategy.fixed.v1"})()

    def __init__(self, decided_on: int = J - 1, sign: float = 1.0, **_: Any):
        self.decided_on = int(decided_on)
        self.sign = float(sign)

    def run(self, data: Any, params: Any = None) -> dict[str, Any]:
        return {"weights": _weights_decided_on(self.decided_on, sign=self.sign)}


class _Data:
    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame({"symbol": ["A", "USD"]})

    def load_market_data(self, universe: Any, asof: str, params: dict[str, Any]) -> dict[str, pd.DataFrame]:
        return {"prices": _prices()}


def _run_pipeline(tmp_path, params: dict[str, Any], *, decided_on: int = J - 1, sign: float = 1.0):
    store = FileArtifactStore(str(tmp_path), "run")
    full = {"fees": 0.0, "strategies": [{"name": "strategy.fixed.v1", "weight": 1.0}], **params}
    result = BacktestPipeline().run(
        mode="backtest",
        asof="2024-02-09",
        params=full,
        data=_Data(),
        store=store,
        broker=None,
        risk=[],
        strategies=[_FixedWeights(decided_on, sign)],
    )
    return result, store


# The three engine branches the pipeline can take.
BRANCHES = {
    "vectorbt_from_orders": {"engine": "vectorbt"},
    "vectorbt_order_func": {"engine": "vectorbt", "threshold": 0.0},
    "rsims": {"engine": "rsims"},
}


# ----------------------------------------------------------------------
# The test that would have caught it
# ----------------------------------------------------------------------


@pytest.mark.parametrize("branch", BRANCHES)
def test_default_is_next_bar_a_decision_on_the_last_bar_before_the_jump_earns_nothing(tmp_path, branch):
    result, _ = _run_pipeline(tmp_path, BRANCHES[branch], decided_on=J - 1)
    assert result.metrics["total_return"] == pytest.approx(0.0, abs=1e-9)
    assert result.metrics["execution_lag_bars"] == 1.0


@pytest.mark.parametrize("branch", BRANCHES)
def test_default_a_decision_one_bar_earlier_earns_the_jump(tmp_path, branch):
    result, _ = _run_pipeline(tmp_path, BRANCHES[branch], decided_on=J - 2)
    assert result.metrics["total_return"] == pytest.approx(JUMP, abs=1e-9)


@pytest.mark.parametrize("branch", BRANCHES)
def test_lag_zero_reproduces_the_historical_same_bar_behaviour_exactly(tmp_path, branch):
    """PINNED back-compat: `lag_bars: 0` is the pre-change engine, bit for bit.

    Same-bar, the J-1 decision buys at close[J-1] and earns the whole jump, and
    the engine receives the decided weights unshifted.
    """
    result, store = _run_pipeline(tmp_path, {**BRANCHES[branch], "execution": {"lag_bars": 0}}, decided_on=J - 1)
    assert result.metrics["total_return"] == pytest.approx(JUMP, abs=1e-9)
    traded = store.read_parquet("traded_weights").set_index("date")
    decided = store.read_parquet("weights_history").set_index("date")
    pd.testing.assert_frame_equal(traded, decided, check_freq=False)


def test_lag_two_waits_two_bars(tmp_path):
    late, _ = _run_pipeline(tmp_path / "a", {"execution": {"lag_bars": 2}}, decided_on=J - 2)
    early, _ = _run_pipeline(tmp_path / "b", {"execution": {"lag_bars": 2}}, decided_on=J - 3)
    assert late.metrics["total_return"] == pytest.approx(0.0, abs=1e-9)
    assert early.metrics["total_return"] == pytest.approx(JUMP, abs=1e-9)


@pytest.mark.parametrize(("lag", "expected"), [(None, 0.0), (1, 0.0), (0, JUMP)])
def test_the_sweep_path_agrees_with_the_pipeline_on_the_same_toy(tmp_path, lag, expected):
    grid = sweep(
        strategy_cls=_FixedWeights,
        base_params={},
        sweep_params={"decided_on": [J - 1]},
        data={"prices": _prices()},
        backtest_kwargs={"fees": 0.0, "rebalancing_freq": 1},
        metrics=["total_return"],
        lag_bars=lag,
    )
    sweep_return = float(grid["total_return"].iloc[0])
    params = {} if lag is None else {"execution": {"lag_bars": lag}}
    pipeline_return = _run_pipeline(tmp_path, params, decided_on=J - 1)[0].metrics["total_return"]
    assert sweep_return == pytest.approx(expected, abs=1e-9)
    assert sweep_return == pytest.approx(pipeline_return, abs=1e-9)


def test_a_lagged_weight_never_lands_on_a_bar_without_a_price():
    """The lag is applied BEFORE the missing-price mask: a position decided on the
    last priced bar of a delisted asset must not be carried onto unpriced bars."""
    prices = _prices()
    prices.iloc[35:, 0] = np.nan  # `A` stops trading after bar 34
    weights = _weights_decided_on(0)
    _, traded = BacktestPipeline._align_for_engine(prices, weights, 1)
    assert traded["A"].iloc[34] == 1.0
    assert (traded["A"].iloc[35:] == 0.0).all()


def test_variants_flow_takes_the_same_lag(tmp_path):
    store = FileArtifactStore(str(tmp_path), "run")
    result = BacktestPipeline().run(
        mode="backtest",
        asof="2024-02-09",
        params={
            "fees": 0.0,
            "variants": [
                {"name": "late", "strategy": {"name": "late"}},
                {"name": "early", "strategy": {"name": "early"}},
            ],
        },
        data=_Data(),
        store=store,
        broker=None,
        risk=[],
        variant_plugins={"late": _FixedWeights(J - 1), "early": _FixedWeights(J - 2)},
    )
    assert result.metrics["late__total_return"] == pytest.approx(0.0, abs=1e-9)
    assert result.metrics["early__total_return"] == pytest.approx(JUMP, abs=1e-9)
    assert result.notes["execution"]["lag_bars"] == 1
    assert "traded_weights" in result.artifacts


def test_variant_level_execution_override_is_refused(tmp_path):
    with pytest.raises(ValueError, match="run-level"):
        BacktestPipeline().run(
            mode="backtest",
            asof="2024-02-09",
            params={
                "variants": [{"name": "v", "strategy": {"name": "v"}, "overrides": {"execution": {"lag_bars": 0}}}]
            },
            data=_Data(),
            store=FileArtifactStore(str(tmp_path), "run"),
            broker=None,
            risk=[],
            variant_plugins={"v": _FixedWeights()},
        )


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------


def test_lag_bars_defaults_to_one():
    assert resolve_lag_bars(None) == 1
    assert resolve_lag_bars({}) == 1


@pytest.mark.parametrize(
    "bad",
    [{"lag_bars": -1}, {"lag_bars": 1.0}, {"lag_bars": "1"}, {"lag_bars": True}, {"lag_bar": 0}, 0, [1]],
)
def test_malformed_execution_block_is_refused_not_defaulted(bad):
    with pytest.raises(ValueError, match="execution"):
        resolve_lag_bars(bad)


def test_pipeline_refuses_a_typo_before_loading_data(tmp_path):
    class _NoData:
        def load_universe(self, params):
            raise AssertionError("data must not be loaded for a malformed execution block")

    with pytest.raises(ValueError, match="unknown key"):
        BacktestPipeline().run(
            mode="backtest",
            asof="2024-02-09",
            params={"execution": {"lag_bar": 0}},
            data=_NoData(),
            store=FileArtifactStore(str(tmp_path), "run"),
            broker=None,
            risk=[],
        )


def test_validate_config_reports_execution_and_venue_problems():
    from quantbox.validate import validate_config

    def cfg(params):
        return {
            "run": {"mode": "backtest", "asof": "2024-01-01"},
            "artifacts": {},
            "plugins": {"pipeline": {"name": "backtest.pipeline.v1", "params": params}, "data": {"name": "x"}},
        }

    assert validate_config(cfg({})) == []
    assert [f.level for f in validate_config(cfg({"execution": {"lag_bars": 0}}))] == ["warning"]
    assert [f.level for f in validate_config(cfg({"execution": {"lag_bars": -1}}))] == ["error"]
    assert [f.level for f in validate_config(cfg({"venue": {"allow_short": False}}))] == ["error"]


def test_shift_signal_is_a_deprecated_alias_of_lag_bars():
    assert resolve_sweep_lag_bars(None, None) == 1
    assert resolve_sweep_lag_bars(2, None) == 2
    with pytest.warns(DeprecationWarning, match="shift_signal"):
        assert resolve_sweep_lag_bars(None, 0) == 0
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="contradicts"):
        resolve_sweep_lag_bars(1, 0)


def test_apply_execution_lag_shape():
    w = _weights_decided_on(3)
    lagged = apply_execution_lag(w, 1)
    assert lagged.iloc[0].tolist() == [0.0, 0.0]
    assert lagged.iloc[4, 0] == 1.0 and lagged.iloc[3, 0] == 0.0
    assert apply_execution_lag(w, 0) is w
    assert apply_execution_lag(w, 1, fill_leading=None).iloc[0].isna().all()


# ----------------------------------------------------------------------
# Recorded and shown
# ----------------------------------------------------------------------


def test_default_run_records_and_states_its_timing(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="quantbox.execution"):
        result, store = _run_pipeline(tmp_path, {})
    assert "SAME-BAR" not in caplog.text
    assert result.notes["execution"] == {
        "lag_bars": 1,
        "fill": "close",
        "same_bar": False,
        "description": result.notes["execution"]["description"],
    }
    assert "next-bar (lag_bars=1)" in result.notes["execution"]["description"]
    assert "**Execution timing:** next-bar (lag_bars=1)" in (store.root / "summary.md").read_text()
    report = json.loads((store.root / "report_data.json").read_text())
    assert report["execution"].startswith("next-bar (lag_bars=1)")
    assert report["reproducibility"]["engine_config"]["execution"]["lag_bars"] == 1
    assert "metaItem('Execution', D.execution" in (store.root / "report.html").read_text()


def test_same_bar_run_is_loud_and_recorded(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="quantbox.execution"):
        result, store = _run_pipeline(tmp_path, {"execution": {"lag_bars": 0}})
    assert "execution.lag_bars=0 (SAME-BAR)" in caplog.text
    assert result.notes["execution"]["same_bar"] is True
    assert result.metrics["execution_lag_bars"] == 0.0
    assert "**Execution timing:** SAME-BAR (lag_bars=0)" in (store.root / "summary.md").read_text()


def test_run_manifest_carries_execution_and_venue(tmp_path):
    """Through the real runner: `run_manifest.json` has `execution.lag_bars`."""
    import yaml

    from quantbox.registry import PluginRegistry
    from quantbox.runner import run_from_config

    prices = _prices().rename_axis("date").reset_index().melt("date", var_name="symbol", value_name="close")
    prices_path = tmp_path / "prices.parquet"
    prices.to_parquet(prices_path, index=False)
    cfg = yaml.safe_load(f"""
run: {{mode: backtest, asof: "2024-02-09", pipeline: backtest.pipeline.v1}}
artifacts: {{root: "{tmp_path / "artifacts"}"}}
plugins:
  pipeline:
    name: backtest.pipeline.v1
    params:
      fees: 0.0
      venue: {{allow_shorts: false}}
      universe: {{symbols: [A, USD]}}
  strategies:
    - name: strategy.cross_asset_momentum.v1
      weight: 1.0
      params: {{windows: [5, 10], long_only: true, position_size: 0.5}}
  data:
    name: local_file_data
    params_init: {{prices_path: "{prices_path}"}}
""")
    result = run_from_config(cfg, PluginRegistry.discover())
    manifest = json.loads((tmp_path / "artifacts" / result.run_id / "run_manifest.json").read_text())
    assert manifest["execution"]["lag_bars"] == 1
    assert manifest["execution"]["same_bar"] is False
    assert manifest["venue"] == {"declared": True, "allow_shorts": False}
    assert manifest["metrics"]["execution_lag_bars"] == 1.0


# ----------------------------------------------------------------------
# Venue
# ----------------------------------------------------------------------


def test_venue_resolution():
    assert resolve_allow_shorts(None, None) == (False, False)
    assert resolve_allow_shorts(None, {"allow_short": True}) == (True, False)
    assert resolve_allow_shorts({"allow_shorts": True}, {}) == (True, True)
    assert resolve_allow_shorts({"allow_shorts": False}, {"allow_short": False}) == (False, True)
    with pytest.raises(ValueError, match="contradicts"):
        resolve_allow_shorts({"allow_shorts": False}, {"allow_short": True})
    for bad in ({"allow_shorts": "no"}, {"allow_short": False}, {}, False):
        with pytest.raises(ValueError, match="venue"):
            resolve_allow_shorts(bad, {})


def _long_short_weights() -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=_prices().index, columns=["A", "USD"])
    w["A"] = 0.5
    w["USD"] = -0.5
    return w


def test_allow_shorts_false_clips_before_transforms_and_does_not_relever_longs():
    pipe = BacktestPipeline()
    out = pipe._apply_venue_and_risk(_long_short_weights(), {"tranches": 3, "max_leverage": 1.0}, False, True)
    assert (out["USD"] == 0.0).all()
    # Long side untouched: 0.5 stays 0.5 — NOT scaled up to 1.0 to refill the gross.
    assert out["A"].tolist() == pytest.approx([0.5] * N)


def test_clip_happens_before_tranching_when_venue_is_declared():
    """A short that flips long must not be averaged in as a negative by the tranche mean."""
    w = pd.DataFrame({"A": [-1.0, 1.0, 1.0]}, index=pd.date_range("2024-01-01", periods=3))
    pipe = BacktestPipeline()
    declared = pipe._apply_venue_and_risk(w, {"tranches": 2}, False, True)
    legacy = pipe._apply_venue_and_risk(w, {"tranches": 2}, False, False)
    assert declared["A"].tolist() == pytest.approx([0.0, 0.5, 1.0])  # mean of CLIPPED targets
    assert legacy["A"].tolist() == pytest.approx([0.0, 0.0, 1.0])  # legacy: clip of the mean, kept bit-for-bit


def test_shorts_traded_without_a_venue_block_warn_and_are_measured(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="quantbox.execution"):
        result, _ = _run_pipeline(tmp_path, {"risk": {"allow_short": True}}, decided_on=0, sign=-1.0)
    assert "no `venue:` block is declared" in caplog.text
    assert result.metrics["traded_short_gross_share"] == pytest.approx(1.0)
    assert result.metrics["traded_mean_net_exposure"] < 0
    assert result.notes["venue"] == {"declared": False, "allow_shorts": True}


def test_declared_short_venue_is_quiet(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="quantbox.execution"):
        result, _ = _run_pipeline(tmp_path, {"venue": {"allow_shorts": True}}, decided_on=0, sign=-1.0)
    assert "VENUE" not in caplog.text
    assert result.metrics["traded_short_gross_share"] == pytest.approx(1.0)


def test_silently_clipped_shorts_are_no_longer_silent(tmp_path, caplog):
    """The legacy default (`risk.allow_short` false) clips shorts; the run must say so."""
    with caplog.at_level(logging.WARNING, logger="quantbox.execution"):
        result, _ = _run_pipeline(tmp_path, {}, decided_on=0, sign=-1.0)
    assert "are CLIPPED to 0 by risk.allow_short=false" in caplog.text
    assert result.metrics["target_short_gross_share"] == pytest.approx(1.0)
    assert result.metrics["traded_short_gross_share"] == 0.0
    assert result.metrics["traded_flat_bar_share"] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Traded weights and the metrics read from them
# ----------------------------------------------------------------------


def test_traded_weights_are_what_the_engine_received_and_weights_history_keeps_its_meaning(tmp_path):
    result, store = _run_pipeline(tmp_path, {"risk": {"max_leverage": 0.5}}, decided_on=J - 2)
    decided = store.read_parquet("weights_history").set_index("date")
    traded = store.read_parquet("traded_weights").set_index("date")
    # weights_history: the strategy's decision, pre-transform, unlagged (meaning unchanged).
    assert decided["A"].iloc[J - 2] == 1.0 and decided["A"].iloc[J - 3] == 0.0
    # traded_weights: leverage-capped AND lagged one bar.
    assert traded["A"].iloc[J - 2] == 0.0 and traded["A"].iloc[J - 1] == 0.5
    # Exposure metrics describe the traded book, not the targets.
    held_bars = N - (J - 1)
    assert result.metrics["traded_mean_gross_exposure"] == pytest.approx(0.5 * held_bars / N)
    assert result.metrics["traded_flat_bar_share"] == pytest.approx((J - 1) / N)
    assert result.metrics["traded_mean_turnover"] == pytest.approx(0.5 / N)


def test_exposure_metrics_definitions():
    w = pd.DataFrame({"A": [0.0, 1.0, 1.0, -0.5], "B": [0.0, -1.0, 0.0, 0.0]})
    m = exposure_metrics(w, "traded")
    assert m["traded_mean_gross_exposure"] == pytest.approx((0 + 2 + 1 + 0.5) / 4)
    assert m["traded_mean_net_exposure"] == pytest.approx((0 + 0 + 1 - 0.5) / 4)
    assert m["traded_short_gross_share"] == pytest.approx(1.5 / 3.5)
    assert m["traded_mean_turnover"] == pytest.approx((0 + 2 + 1 + 1.5) / 4)
    assert m["traded_flat_bar_share"] == pytest.approx(0.25)
    assert exposure_metrics(pd.DataFrame(), "x")["x_flat_bar_share"] == 1.0


# ----------------------------------------------------------------------
# Positive control: the LIVE path must not start lagging orders
# ----------------------------------------------------------------------


@pytest.mark.parametrize("module", ["trading_pipeline", "alloc2orders"])
def test_live_trading_path_never_touches_the_backtest_execution_lag(module):
    """The backtest lag simulates the delay between deciding and filling. Live
    trading HAS that delay for real; applying the lag there would trade a bar
    stale. Asked of the AST (imports + referenced names), not of the text."""
    import ast
    from pathlib import Path

    import quantbox.plugins.pipeline as pkg

    tree = ast.parse((Path(pkg.__file__).parent / f"{module}.py").read_text())
    imported = {
        (n.module or "") if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in n.names
    }
    assert len(imported) > 3, "AST walk saw no imports — the control is blind"
    assert not {m for m in imported if m.endswith("execution") or "backtest_pipeline" in m}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    assert not names & {"apply_execution_lag", "resolve_lag_bars", "lag_bars", "_align_for_engine"}
