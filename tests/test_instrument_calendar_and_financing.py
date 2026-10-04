"""Per-instrument calendar, financing of cash, and the rebalance schedule as bars (TOM-1429, ADR-0007).

Two defects, both found by replicating Moskowitz-Ooi-Pedersen TSMOM against an
independent reference (quantbox-lab ``replication-tsmom-mop2012``,
``results/engine-parity``):

1. A missing price ZEROED the traded weight. On a holiday rebalance bar (1
   January, where other instruments print) that meant flat for the whole
   holding period: 478 of 5920 instrument-months, -0.10 Sharpe.
2. vectorbt cannot take cash below zero, so a book whose net exposure is above
   1 had its last buys cut silently: 58 of 409 rebalances, -0.03 Sharpe.

Every test drives the real pipeline (or the real engine) on a toy whose answer
is known in closed form.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantbox.financing import BORROW, LEND, resolve_financing
from quantbox.frequency import rebalancing_dates
from quantbox.instrument_calendar import apply_instrument_calendar
from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline
from quantbox.store import FileArtifactStore


class _Fixed:
    """Strategy stub: the same decided weight every bar, per ticker."""

    meta = type("M", (), {"name": "strategy.fixed.v1"})()

    def __init__(self, weights: dict[str, float], index: pd.Index):
        self.frame = pd.DataFrame({k: float(v) for k, v in weights.items()}, index=index)

    def run(self, data: Any, params: Any = None) -> dict[str, Any]:
        return {"weights": self.frame}


class _Data:
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices

    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame({"symbol": list(self.prices.columns)})

    def load_market_data(self, universe: Any, asof: str, params: dict[str, Any]) -> dict[str, pd.DataFrame]:
        return {"prices": self.prices}


def _run(tmp_path, prices: pd.DataFrame, weights: dict[str, float], **params: Any):
    store = FileArtifactStore(str(tmp_path), "run")
    full = {"fees": 0.0, "strategies": [{"name": "strategy.fixed.v1", "weight": 1.0}], **params}
    result = BacktestPipeline().run(
        mode="backtest",
        asof=str(prices.index[-1].date()),
        params=full,
        data=_Data(prices),
        store=store,
        broker=None,
        risk=[],
        strategies=[_Fixed(weights, prices.index)],
    )
    return result, store


def _validation(store: FileArtifactStore) -> dict[str, Any]:
    return json.loads((store.root / "data_validation.json").read_text())


# ----------------------------------------------------------------------
# (a) The per-instrument calendar
# ----------------------------------------------------------------------


def _holiday_panel() -> pd.DataFrame:
    """Business days Dec 2023 - Feb 2024. `A` does not print on 1 January (B does); A is +10% on 10 January."""
    idx = pd.bdate_range("2023-12-01", "2024-02-29")
    a = pd.Series(100.0, index=idx)
    a[a.index >= "2024-01-10"] = 110.0
    a[pd.Timestamp("2024-01-01")] = np.nan
    return pd.DataFrame({"A": a, "B": 50.0}, index=idx)


def test_a_holiday_on_the_rebalance_bar_keeps_the_target_for_the_whole_holding_period(tmp_path):
    """BMS rebalance on 1 January, a holiday for A only. The January target (0.5 A) must be HELD
    across the +10% move. Before TOM-1429 the missing price zeroed it: flat all January, 0% return."""
    prices = _holiday_panel()
    result, store = _run(tmp_path, prices, {"A": 0.5, "B": 0.0}, rebalancing_freq="BMS")

    traded = store.read_parquet("traded_weights").set_index("date")
    assert traded.loc[pd.Timestamp("2024-01-01"), "A"] == 0.5
    assert result.metrics["total_return"] == pytest.approx(0.5 * 0.10, rel=1e-9)

    report = _validation(store)
    assert report["schema"] == "quantbox/data-validation@1"
    a = report["calendar"]["instruments"]["A"]
    assert a["ffilled_bars"] == 1 and a["targeted_outside_window_bars"] == 0
    assert report["calendar"]["totals"]["ffilled_bars"] == 1
    assert result.metrics["calendar_ffilled_bars"] == 1.0
    # The run's notes carry the summary the manifest records.
    assert result.notes["data_validation"]["calendar"]["ffilled_bars"] == 1
    assert result.notes["data_validation"]["file"] == "data_validation.json"


def test_an_instrument_listed_mid_sample_is_flat_before_listing_and_the_override_is_counted(tmp_path, caplog):
    """`B` lists on 15 January. The strategy targets it from the start: those bars are outside B's
    life window, so the weight is 0 there — no back-filled tradable position — and every one of
    them is counted and logged, never dropped without a record."""
    idx = pd.bdate_range("2024-01-01", "2024-02-29")
    listing = pd.Timestamp("2024-01-15")
    b = pd.Series(np.linspace(20.0, 30.0, len(idx)), index=idx).where(idx >= listing)
    prices = pd.DataFrame({"A": 100.0, "B": b}, index=idx)

    with caplog.at_level(logging.WARNING, logger="quantbox.instrument_calendar"):
        result, store = _run(tmp_path, prices, {"A": 0.0, "B": 0.5}, rebalancing_freq=1)

    traded = store.read_parquet("traded_weights").set_index("date")
    assert (traded.loc[traded.index < listing, "B"] == 0.0).all()
    assert (traded.loc[traded.index >= listing, "B"] == 0.5).all()

    # Lag 1: row 0 has no decision behind it; rows 1..(listing-1) carried a 0.5 target outside the window.
    expected = int((idx < listing).sum()) - 1
    b_report = _validation(store)["calendar"]["instruments"]["B"]
    assert b_report["targeted_outside_window_bars"] == expected
    assert b_report["max_abs_weight_outside_window"] == 0.5
    assert b_report["first_valid"].startswith("2024-01-15")
    assert result.metrics["calendar_targeted_outside_window_bars"] == float(expected)
    assert "OUTSIDE the instrument's life window" in caplog.text

    # The engine held no B before listing: the book's value did not move with B's (back-filled) mark.
    pf = store.read_parquet("portfolio_daily").set_index("date")["portfolio_value"]
    assert (pf.loc[pf.index < listing] == pf.iloc[0]).all()


def test_a_delisted_instrument_goes_flat_after_its_last_price():
    idx = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"A": [1.0, 2, 3, 4, 5, np.nan, np.nan, np.nan, np.nan, np.nan], "B": 1.0}, index=idx)
    weights = pd.DataFrame({"A": 0.7, "B": 0.0}, index=idx)
    out = apply_instrument_calendar(prices, weights)
    assert (out.weights["A"].iloc[:5] == 0.7).all() and (out.weights["A"].iloc[5:] == 0.0).all()
    assert out.report["instruments"]["A"]["targeted_outside_window_bars"] == 5
    assert out.report["instruments"]["A"]["ffilled_bars"] == 0
    assert not out.prices.isna().any().any()


def test_a_short_history_column_is_kept_not_dropped():
    """The 50%-coverage drop is gone: a coin with 20% history is traded inside its window."""
    idx = pd.bdate_range("2024-01-01", periods=50)
    prices = pd.DataFrame({"A": 1.0, "NEW": pd.Series(2.0, index=idx).where(idx >= idx[40])}, index=idx)
    weights = pd.DataFrame({"A": 0.5, "NEW": 0.5}, index=idx)
    out = apply_instrument_calendar(prices, weights)
    assert list(out.weights.columns) == ["A", "NEW"]
    assert (out.weights["NEW"].iloc[40:] == 0.5).all()


# ----------------------------------------------------------------------
# (b) Financing: rate + spread, and the refusal without it
# ----------------------------------------------------------------------


def _levered_panel() -> pd.DataFrame:
    """Calendar days; A wanders deterministically, CASH is a T-bill TR index at ~4%/yr with a wobble."""
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(7)
    a = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx)))
    cash_r = 0.04 / 365 + rng.normal(0, 1e-5, len(idx))
    cash_r[0] = 0.0
    return pd.DataFrame({"A": a, "CASH": 100.0 * np.cumprod(1 + cash_r)}, index=idx)


@pytest.mark.parametrize(
    ("w_a", "borrow_bps", "lend_bps"),
    [(1.5, 100.0, 0.0), (0.4, 0.0, 50.0)],
    ids=["borrow-1.5x", "lend-0.4x"],
)
def test_financing_holds_the_full_book_and_charges_rate_plus_spread(tmp_path, w_a, borrow_bps, lend_bps):
    """Daily rebalance, so each bar's return is closed-form:
    r_t = w * rA_t + (1 - w) * (rCASH_t -/+ spread * dt), the spread signed by who lends."""
    prices = _levered_panel()
    venue = {
        "allow_shorts": False,
        "financing": {"rate": "CASH", "borrow_spread_bps": borrow_bps, "lend_spread_bps": lend_bps},
    }
    result, store = _run(tmp_path, prices, {"A": w_a, "CASH": 0.0}, rebalancing_freq=1, venue=venue)

    traded = store.read_parquet("traded_weights").set_index("date")
    assert (traded["A"].iloc[1:] == w_a).all()
    assert LEND not in traded.columns and BORROW not in traded.columns  # the saved book is the real one
    assert result.metrics["engine_underfilled_rebalances"] == 0.0
    assert result.notes["financing"]["modelled"] is True
    assert result.notes["venue"]["financing"]["rate"] == {"ticker": "CASH"}

    returns = store.read_parquet("returns").set_index(store.read_parquet("returns").columns[0])["returns"]
    r_a = prices["A"].pct_change()
    r_cash = prices["CASH"].pct_change()
    dt = 1.0 / 365.0
    cash_w = 1.0 - w_a
    spread = borrow_bps if cash_w < 0 else -lend_bps
    expected = w_a * r_a + cash_w * (r_cash + spread / 1e4 * dt)
    # Bar 0 is flat (lag 1), bar 1 is the first fill; from bar 2 every return is the held book's.
    np.testing.assert_allclose(returns.iloc[2:].to_numpy(), expected.iloc[2:].to_numpy(), rtol=1e-9, atol=1e-12)


def test_without_financing_a_vectorbt_book_above_net_one_is_refused_not_cut(tmp_path):
    prices = _levered_panel()
    with pytest.raises(ValueError, match=r"venue\.financing") as exc:
        _run(tmp_path, prices, {"A": 1.5, "CASH": 0.0}, rebalancing_freq=1, venue={"allow_shorts": False})
    assert "net exposure above 1" in str(exc.value)


def test_without_financing_an_unlevered_vectorbt_book_runs_and_is_filled(tmp_path):
    result, _ = _run(tmp_path, _levered_panel(), {"A": 1.0, "CASH": 0.0}, rebalancing_freq=1)
    assert result.metrics["engine_underfilled_rebalances"] == 0.0


def test_without_financing_rsims_is_not_refused_it_has_no_cash_floor(tmp_path):
    result, store = _run(tmp_path, _levered_panel(), {"A": 1.5, "CASH": 0.0}, engine="rsims")
    traded = store.read_parquet("traded_weights").set_index("date")
    assert (traded["A"].iloc[1:] == 1.5).all()
    assert result.notes["financing"] == {"modelled": False}


def test_the_fill_gap_measurement_sees_a_cut_book():
    """The counter the financing test reads as 0 must be able to fail: hand vectorbt a 1.5x book
    with no cash legs and it cuts the buy."""
    from quantbox.plugins.backtesting.vectorbt_engine import rebalance_fill_gaps
    from quantbox.plugins.backtesting.vectorbt_engine import run as run_vectorbt

    prices = _levered_panel()[["A"]]
    weights = pd.DataFrame({"A": 1.5}, index=prices.index)
    weights.iloc[0] = 0.0
    pf = run_vectorbt(prices, weights, rebalancing_freq=1)
    gaps = rebalance_fill_gaps(pf, weights, prices.index)
    assert gaps["gap"].iloc[1:].min() > 0.4  # held ~1.0, target 1.5


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"borrow_spread_bps": 10}, "'rate' is required"),
        ({"rate": True}, "ticker .* or an annual rate"),
        ({"rate": "X", "borrow_spread_bps": -1}, ">= 0"),
        ({"rate": "X", "spread": 1}, "unknown key"),
        ("LT12", "must be a mapping"),
    ],
)
def test_a_malformed_financing_block_is_refused(block, message):
    with pytest.raises(ValueError, match=message):
        resolve_financing(block)


def test_plan_refuses_a_malformed_financing_block_before_any_data():
    with pytest.raises(ValueError, match="'rate' is required"):
        BacktestPipeline().plan({"venue": {"allow_shorts": True, "financing": {"lend_spread_bps": 5}}})


def test_a_rate_ticker_that_starts_after_the_backtest_is_refused(tmp_path):
    prices = _levered_panel()
    prices.loc[prices.index < prices.index[10], "CASH"] = np.nan
    venue = {"allow_shorts": False, "financing": {"rate": "CASH"}}
    with pytest.raises(ValueError, match="financing rate there is unknown"):
        _run(tmp_path, prices, {"A": 1.5}, rebalancing_freq=1, venue=venue)


# ----------------------------------------------------------------------
# The schedule is bars
# ----------------------------------------------------------------------


def test_a_calendar_rebalance_date_that_is_not_a_bar_snaps_forward_to_the_next_bar():
    bars = pd.bdate_range("2023-12-01", "2024-03-29").drop(pd.Timestamp("2024-01-01"))  # no 1 January row
    bms = rebalancing_dates(bars, "BMS")
    # 1 January is the BMS date and not a bar: January's rebalance is the first bar after it.
    assert list(bms) == list(pd.to_datetime(["2023-12-01", "2024-01-02", "2024-02-01", "2024-03-01"]))
    # W-SUN on weekday data: every Sunday becomes the Monday after (it used to be no rebalance at all).
    weekdays = pd.bdate_range("2024-01-02", "2024-03-29")
    weekly = rebalancing_dates(weekdays, "1W")
    assert len(weekly) == 12 and all(d.dayofweek == 0 for d in weekly)
    assert weekly.isin(weekdays).all()
