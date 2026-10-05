"""Instrument and execution calendars, decision vs execution timing, leverage and financing (TOM-1429, ADR-0007).

Found by replicating Moskowitz-Ooi-Pedersen TSMOM against an independent
reference (quantbox-lab ``replication-tsmom-mop2012``, ``results/engine-parity``):

1. A missing price ZEROED the traded weight. On a holiday rebalance bar (1
   January, where other instruments print) that meant flat for the whole
   holding period: 478 of 5920 instrument-months, -0.10 Sharpe.
2. vectorbt cannot take cash below zero, so a book whose net exposure is above
   1 had its last buys cut silently: 58 of 409 rebalances, -0.03 Sharpe.
3. (review round 1) Keeping the target on a bar the instrument did not print
   FILLED it at the stale forward-filled close: decided at the 29 Dec close,
   filled at that same close on 1 January, it booked the 2 January move.

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

from quantbox.execution_schedule import schedule_book
from quantbox.financing import BORROW, LEND, resolve_financing
from quantbox.frequency import rebalancing_dates
from quantbox.instrument_calendar import execution_bars, instrument_calendar, validate_data_validation
from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline
from quantbox.store import FileArtifactStore


class _Fixed:
    """Strategy stub: a decided-weights frame, or the same weight every bar per ticker."""

    meta = type("M", (), {"name": "strategy.fixed.v1"})()

    def __init__(self, weights: dict[str, float] | pd.DataFrame, index: pd.Index):
        if isinstance(weights, pd.DataFrame):
            self.frame = weights
        else:
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


def _run(tmp_path, prices: pd.DataFrame, weights: dict[str, float] | pd.DataFrame, **params: Any):
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


def _schedule(store: FileArtifactStore) -> pd.DataFrame:
    return store.read_parquet("rebalance_schedule")


def _book(prices: pd.DataFrame, weights: pd.DataFrame, *, freq: Any = 1, lag: int = 1, calendar: str = "majority"):
    cal = instrument_calendar(prices)
    bars = execution_bars(cal, calendar, prices[calendar] if calendar in prices.columns else None)
    return cal, schedule_book(weights, cal, bars, freq, lag, leverage="normalize")


def _step(index: pd.Index, ticker_weights: dict[str, float], from_date: str, columns: list[str]) -> pd.DataFrame:
    """Flat until *from_date* (the DECISION bar), then *ticker_weights* from it onward."""
    w = pd.DataFrame(0.0, index=index, columns=columns)
    for k, v in ticker_weights.items():
        w.loc[w.index >= pd.Timestamp(from_date), k] = v
    return w


ENGINES = ["vectorbt", "rsims"]


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


def test_a_holiday_on_the_rebalance_bar_keeps_the_position_for_the_whole_holding_period(tmp_path):
    """BMS rebalance on 1 January, a holiday for A only. The position (0.5 A) must be HELD
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


def _blocker_panel() -> pd.DataFrame:
    """The review's exact shape: A is not priced on 1 January (B is), A is +10% on 2 January."""
    idx = pd.bdate_range("2023-12-01", "2024-01-31")
    a = pd.Series(100.0, index=idx)
    a[a.index >= "2024-01-02"] = 110.0
    a[pd.Timestamp("2024-01-01")] = np.nan
    return pd.DataFrame({"A": a, "B": 50.0}, index=idx)


@pytest.mark.parametrize("engine", ENGINES)
def test_an_order_never_fills_at_a_price_the_instrument_did_not_print(tmp_path, engine):
    """BLOCKER (review round 1). Decided at the 29 Dec close, lagged to 1 January — a bar A did not
    print. Filling there would use the 29 Dec close and book the 2 January +10%. The order waits for
    A's next printed bar (2 January, at 110): the run books 0, and the deferral is counted."""
    prices = _blocker_panel()
    decided = _step(prices.index, {"A": 1.0}, "2023-12-29", ["A", "B"])
    result, store = _run(tmp_path, prices, decided, rebalancing_freq=1, engine=engine)

    assert result.metrics["total_return"] == pytest.approx(0.0, abs=1e-12)
    traded = store.read_parquet("traded_weights").set_index("date")
    assert traded.loc[pd.Timestamp("2024-01-01"), "A"] == 0.0  # the previous weight is carried
    assert traded.loc[pd.Timestamp("2024-01-02"), "A"] == 1.0
    report = _validation(store)
    assert report["calendar"]["instruments"]["A"]["deferred_trades"] == 1
    assert report["calendar"]["instruments"]["B"]["deferred_trades"] == 0
    assert result.metrics["calendar_deferred_trades"] == 1.0
    sched = _schedule(store).set_index("execution_date")
    assert sched.loc[pd.Timestamp("2024-01-01"), "deferred_instruments"] == "A"
    assert sched.loc[pd.Timestamp("2024-01-01"), "decision_date"] == pd.Timestamp("2023-12-29")


def test_an_instrument_listed_mid_sample_is_flat_before_listing_and_the_override_is_counted(tmp_path, caplog):
    """`B` lists on 15 January. The strategy targets it from the start: those bars are outside B's
    life window, so the weight is 0 there — no back-filled tradable position — and every one of
    them is counted and logged, never dropped without a record."""
    idx = pd.bdate_range("2024-01-01", "2024-02-29")
    listing = pd.Timestamp("2024-01-15")
    b = pd.Series(np.linspace(20.0, 30.0, len(idx)), index=idx).where(idx >= listing)
    prices = pd.DataFrame({"A": 100.0, "B": b}, index=idx)

    with caplog.at_level(logging.WARNING):
        result, store = _run(tmp_path, prices, {"A": 0.0, "B": 0.5}, rebalancing_freq=1)

    traded = store.read_parquet("traded_weights").set_index("date")
    assert (traded.loc[traded.index < listing, "B"] == 0.0).all()
    assert (traded.loc[traded.index >= listing, "B"] == 0.5).all()

    # Lag 1: bar 0 executes nothing; bars 1..(listing-1) execute a 0.5 target outside the window.
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
    cal, book = _book(prices, weights)
    assert (book.weights["A"].iloc[1:5] == 0.7).all() and (book.weights["A"].iloc[5:] == 0.0).all()
    assert book.report["instruments"]["A"]["targeted_outside_window_bars"] == 5
    assert cal.report["instruments"]["A"]["ffilled_bars"] == 0
    assert not cal.prices.isna().any().any()


def test_a_short_history_column_is_kept_and_the_old_drop_is_counted():
    """The 50%-coverage drop is gone: a coin with 20% history is traded inside its window, and the
    column the old rule would have dropped is COUNTED, so the result change is visible."""
    idx = pd.bdate_range("2024-01-01", periods=50)
    prices = pd.DataFrame({"A": 1.0, "NEW": pd.Series(2.0, index=idx).where(idx >= idx[40])}, index=idx)
    weights = pd.DataFrame({"A": 0.5, "NEW": 0.5}, index=idx)
    cal, book = _book(prices, weights)
    assert list(book.weights.columns) == ["A", "NEW"]
    assert (book.weights["NEW"].iloc[40:] == 0.5).all()
    assert cal.report["legacy_coverage_drop"]["count"] == 1
    assert cal.report["legacy_coverage_drop"]["columns"] == ["NEW"]


# ----------------------------------------------------------------------
# The execution calendar
# ----------------------------------------------------------------------


H = pd.Timestamp("2024-01-10")  # the holiday row


def _three_panel(*, closed: tuple[str, ...]) -> pd.DataFrame:
    """A, B, C on business days of January 2024; the instruments in *closed* do not print on H.
    Every instrument is +10% on the bar after H, flat otherwise."""
    idx = pd.bdate_range("2024-01-01", "2024-01-31")
    after = idx > H
    prices = pd.DataFrame({k: np.where(after, 110.0, 100.0) for k in ("A", "B", "C")}, index=idx)
    for k in closed:
        prices.loc[H, k] = np.nan
    return prices


@pytest.mark.parametrize("engine", ENGINES)
def test_a_holiday_most_instruments_skip_is_not_an_execution_bar(tmp_path, engine):
    """Only C prints on H: under the default `majority` calendar H is not an execution bar. A decision
    taken the bar before H (C: 0 -> 1) executes one EXECUTION bar later — the bar after H, at 110 —
    not on the raw H row, where C printed 100 and the run would book the +10%."""
    prices = _three_panel(closed=("A", "B"))
    decided = _step(prices.index, {"C": 1.0}, "2024-01-09", ["A", "B", "C"])
    result, store = _run(tmp_path, prices, decided, rebalancing_freq=1, engine=engine)

    assert result.metrics["total_return"] == pytest.approx(0.0, abs=1e-12)
    sched = _schedule(store).set_index("decision_date")
    assert sched.loc[pd.Timestamp("2024-01-09"), "execution_date"] == pd.Timestamp("2024-01-11")
    assert H not in set(sched.index) and H not in set(sched["execution_date"])
    ex = _validation(store)["execution_calendar"]
    assert ex["calendar"] == "majority"
    assert ex["total_bars"] - ex["execution_bars"] == 1
    assert ex["non_execution_bars_by_year"] == {"2024": 1}
    assert result.notes["execution"]["calendar"] == "majority"
    assert result.notes["data_validation"]["execution_calendar"]["execution_bars"] == ex["execution_bars"]


def test_a_lag_counts_execution_bars_not_raw_rows(tmp_path):
    """lag_bars 2 from the bar two before H: raw rows land on H (a holiday row); execution bars land on
    the bar after H."""
    prices = _three_panel(closed=("A", "B"))
    decided = _step(prices.index, {"C": 1.0}, "2024-01-08", ["A", "B", "C"])
    result, store = _run(tmp_path, prices, decided, rebalancing_freq=1, execution={"lag_bars": 2})
    assert result.metrics["total_return"] == pytest.approx(0.0, abs=1e-12)
    sched = _schedule(store).set_index("decision_date")
    assert sched.loc[pd.Timestamp("2024-01-08"), "execution_date"] == pd.Timestamp("2024-01-11")


@pytest.mark.parametrize("engine", ENGINES)
def test_a_single_instrument_holiday_on_a_majority_bar_defers_only_that_instrument(tmp_path, engine):
    """A is closed on H; B and C print, so H IS an execution bar. The decision before H (A and B to 0.3)
    executes on H for B, and is deferred for A alone to its next print (the bar after H, at 110): the run
    books B's +10% on 0.3 and nothing on A."""
    prices = _three_panel(closed=("A",))
    decided = _step(prices.index, {"A": 0.3, "B": 0.3}, "2024-01-09", ["A", "B", "C"])
    result, store = _run(tmp_path, prices, decided, rebalancing_freq=1, engine=engine)

    assert result.metrics["total_return"] == pytest.approx(0.3 * 0.10, rel=1e-9)
    sched = _schedule(store).set_index("execution_date")
    assert sched.loc[H, "deferred_instruments"] == "A"
    inst = _validation(store)["calendar"]["instruments"]
    assert (inst["A"]["deferred_trades"], inst["B"]["deferred_trades"], inst["C"]["deferred_trades"]) == (1, 0, 0)


def test_a_ticker_execution_calendar_follows_that_series(tmp_path):
    """`execution.calendar: REF` — a reference series in the prices that carries no weight. REF is
    closed on H while A and B print: H is not an execution bar, and the decision before it executes
    on the bar after H. Under the default `majority` the same decision fills on H (A printed 100)."""
    prices = _three_panel(closed=())
    prices = prices.rename(columns={"C": "REF"})
    prices.loc[H, "REF"] = np.nan
    decided = _step(prices.index, {"A": 1.0}, "2024-01-09", ["A", "B"])

    result, store = _run(tmp_path, prices, decided, rebalancing_freq=1, execution={"calendar": "REF"})
    assert result.metrics["total_return"] == pytest.approx(0.0, abs=1e-12)
    ex = _validation(store)["execution_calendar"]
    assert ex["calendar"] == "REF" and ex["total_bars"] - ex["execution_bars"] == 1

    majority, _ = _run(tmp_path / "m", prices, decided, rebalancing_freq=1)
    assert majority.metrics["total_return"] == pytest.approx(0.10, rel=1e-9)


def test_an_unknown_execution_calendar_ticker_is_refused(tmp_path):
    prices = _three_panel(closed=())
    with pytest.raises(ValueError, match="not a ticker in the loaded prices"):
        _run(tmp_path, prices, {"A": 1.0}, execution={"calendar": "NOPE"})


def test_a_month_ending_on_a_holiday_decides_on_the_last_execution_bar(tmp_path):
    """31 January 2024 (a Wednesday): only C prints, so it is not an execution bar. Monthly (`ME`)
    decides on 30 January — the last execution bar of the month — and executes on 1 February; it never
    decides on the raw 31 January row, nor snaps into February."""
    idx = pd.bdate_range("2024-01-01", "2024-02-29")
    prices = pd.DataFrame({k: 100.0 for k in ("A", "B", "C")}, index=idx)
    prices.loc[pd.Timestamp("2024-01-31"), ["A", "B"]] = np.nan
    result, store = _run(tmp_path, prices, {"A": 0.5, "B": 0.5, "C": 0.0}, rebalancing_freq="ME")
    sched = _schedule(store)
    jan = sched[sched["decision_date"].dt.month == 1].iloc[0]
    assert jan["decision_date"] == pd.Timestamp("2024-01-30")
    assert jan["execution_date"] == pd.Timestamp("2024-02-01")


def test_period_end_offsets_snap_back_to_the_last_bar_of_the_period():
    bars = pd.bdate_range("2023-12-01", "2024-03-29").drop(pd.Timestamp("2024-01-31"))
    me = rebalancing_dates(bars, "ME")
    assert list(me) == list(pd.to_datetime(["2023-12-29", "2024-01-30", "2024-02-29", "2024-03-29"]))


@pytest.mark.parametrize("engine", ENGINES)
def test_input_staleness_at_each_decision_is_counted(tmp_path, engine):
    """A does not print for three bars from H while B and C do (each is an execution bar). Deciding
    every bar, A's input is 1, 2 and 3 bars old on those decisions: counted, not blocking."""
    idx = pd.bdate_range("2024-01-01", "2024-01-31")
    prices = pd.DataFrame({k: 100.0 for k in ("A", "B", "C")}, index=idx)
    stale_rows = idx[(idx >= H)][:3]
    prices.loc[stale_rows, "A"] = np.nan
    result, store = _run(tmp_path, prices, {"A": 0.3, "B": 0.3, "C": 0.3}, rebalancing_freq=1, engine=engine)

    st = _validation(store)["staleness"]
    assert st["stale_decisions"] == 3 and st["max_bars"] == 3
    assert st["instrument_decisions"] == 3 * len(idx)
    a = _validation(store)["calendar"]["instruments"]["A"]
    assert a["stale_decisions"] == 3 and a["max_staleness_bars"] == 3
    assert result.metrics["decision_stale_inputs"] == 3.0
    assert result.notes["data_validation"]["staleness"]["max_bars"] == 3


# ----------------------------------------------------------------------
# (b) Leverage and financing
# ----------------------------------------------------------------------


def _levered_panel() -> pd.DataFrame:
    """Calendar days; A wanders deterministically, CASH is a T-bill TR index at ~4%/yr with a wobble."""
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(7)
    a = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx)))
    cash_r = 0.04 / 365 + rng.normal(0, 1e-5, len(idx))
    cash_r[0] = 0.0
    return pd.DataFrame({"A": a, "CASH": 100.0 * np.cumprod(1 + cash_r)}, index=idx)


def _returns(store: FileArtifactStore) -> pd.Series:
    frame = store.read_parquet("returns")
    return frame.set_index(frame.columns[0])["returns"]


def _month_end_panel() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", "2024-04-30")  # 31 Mar 2024 is a Sunday; 29 Mar the last bar of March
    return pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)


def test_a_weight_row_stamped_after_the_decision_bar_is_a_stale_decision(tmp_path, caplog):
    """The strategy stamps its weights on CALENDAR month-ends. March's lands on Sunday 31 Mar, after the
    decision bar (Friday 29 Mar) and before the next execution bar: no decision ever sees it, and February's
    weights are held through April. Counted and warned, not refused."""
    prices = _month_end_panel()
    stamps = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"])
    sparse = pd.DataFrame({"A": [1.0, 0.0, 1.0, 0.0], "B": [0.0, 1.0, 0.0, 1.0]}, index=stamps)
    weights = sparse.reindex(prices.index.union(stamps))  # NaN ("said nothing") off the stamps
    with caplog.at_level(logging.WARNING):
        result, store = _run(tmp_path, prices, weights, rebalancing_freq="ME")
    age = _validation(store)["weight_age"]
    assert age["decisions"] == 3, age  # 30 Apr is past the last bar: never executed, not counted
    assert age["stale_decisions"] == 1
    assert age["first_stale"] == ["2024-03-29T00:00:00"]
    assert age["first_missed_stamps"] == ["2024-03-31T00:00:00"]
    assert age["stale_held_bars_max"] == 22  # executed 1 Apr, held to the end (no later decision executes)
    assert result.metrics["decision_stale_weights"] == 1.0
    assert "MISSED their period" in caplog.text and "2024-03-29" in caplog.text
    assert validate_data_validation(_validation(store)) == []


def test_weights_on_every_bar_are_never_stale(tmp_path, caplog):
    prices = _month_end_panel()
    weights = pd.DataFrame({"A": np.linspace(0.1, 0.9, len(prices)), "B": 0.0}, index=prices.index)
    weights["B"] = 1.0 - weights["A"]
    with caplog.at_level(logging.WARNING):
        result, store = _run(tmp_path, prices, weights, rebalancing_freq="ME")
    assert _validation(store)["weight_age"]["stale_decisions"] == 0
    assert result.metrics["decision_stale_weights"] == 0.0
    assert "MISSED their period" not in caplog.text


# Review round 1 on 4f6e4a1 (probes in /tmp/rev-wa-probe): two years of weekday bars, a monthly signal
# stamped on CALENDAR month-ends — 7 of them fall on a weekend — or on business month-ends (correct).
_IDX = pd.bdate_range("2023-01-01", "2024-12-31")
_FULL = pd.date_range(_IDX[0], _IDX[-1], freq="D")


def _signal(stamps: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    sig = pd.DataFrame({"A": rng.uniform(0, 1, len(stamps))}, index=stamps)
    sig["B"] = 1 - sig["A"]
    return sig


def _calendar_me() -> pd.DataFrame:  # the lab shape: forward-filled on a 7-day panel
    return _signal(pd.date_range(_IDX[0], _IDX[-1], freq="ME")).reindex(_FULL).ffill()


def _business_me() -> pd.DataFrame:  # stamped on execution bars: correct
    return _signal(pd.date_range(_IDX[0], _IDX[-1], freq="BME")).reindex(_FULL).ffill()


def _masked() -> pd.DataFrame:  # the calendar-ME signal times a weekday vol scaler (held over weekends)
    rng = np.random.default_rng(1)
    scale = pd.Series(rng.uniform(0.8, 1.2, len(_FULL)), index=_FULL).where(_FULL.weekday < 5).ffill()
    return _calendar_me().mul(scale, axis=0)


def _scaler_7day() -> pd.Series:
    return pd.Series(np.random.default_rng(2).uniform(0.8, 1.2, len(_FULL)), index=_FULL)


def _stamp_plus_drift() -> pd.DataFrame:  # the calendar-ME signal times a 7-DAY scaler: a step with drift on top
    return _calendar_me().mul(_scaler_7day(), axis=0)


def _constant_x_scaler() -> pd.DataFrame:  # no stamping at all: 50/50 times a 7-day vol scaler
    return pd.DataFrame({"A": 0.5, "B": 0.5}, index=_FULL).mul(_scaler_7day(), axis=0)


def _weekly_sunday() -> pd.DataFrame:  # a crypto-style weekly signal stamped on Sundays, forward-filled
    return _signal(pd.date_range(_FULL[0], _FULL[-1], freq="W-SUN")).reindex(_FULL).ffill()


def _sparse_union() -> pd.DataFrame:
    return _signal(pd.date_range(_IDX[0], _IDX[-1], freq="ME")).reindex(
        _IDX.union(pd.date_range(_IDX[0], _IDX[-1], freq="ME"))
    )


@pytest.mark.parametrize(
    ("weights", "freq", "stale"),
    [
        pytest.param(_calendar_me, "ME", 7, id="lab-shape-ME"),
        pytest.param(_calendar_me, "BME", 7, id="lab-shape-BME"),  # was 0: the BME period ended before Sunday
        pytest.param(_masked, "ME", 7, id="masked-by-a-daily-scaler"),  # was 0: any in-period move blinded it
        pytest.param(_sparse_union, "ME", 7, id="sparse-stamps"),
        pytest.param(_constant_x_scaler, "ME", 0, id="scaler-ME"),  # was 10: daily variation, not a stamp
        pytest.param(_constant_x_scaler, "W-FRI", 0, id="scaler-W-FRI"),  # was 104 of 104
        # a Sunday weekly signal opens the NEXT period after a Friday decision: that period's decision trades it
        # (fa87769: ME 10, QE 6). Left: a weekly stamp ON a Sunday month-end — in the period, never traded.
        pytest.param(_weekly_sunday, "ME", 4, id="w-sun-signal-ME"),
        pytest.param(_weekly_sunday, "QE", 3, id="w-sun-signal-QE"),
        # KNOWN FALSE NEGATIVE (ADR-0007 1c): a stamped step with daily drift on top reads as drift
        pytest.param(_stamp_plus_drift, "ME", 0, id="step-plus-drift-missed"),
        pytest.param(_business_me, "ME", 0, id="correct-ME"),
        # no calendar period, not measured (never a 0 that reads as clean)
        pytest.param(_calendar_me, 1, None, id="lab-shape-int-1"),
        pytest.param(_calendar_me, None, None, id="lab-shape-buy-and-hold"),
        pytest.param(_business_me, 5, None, id="int-5"),
        pytest.param(_business_me, "2W-FRI", None, id="2W-FRI"),
        pytest.param(_business_me, 21, None, id="int-21"),
        pytest.param(_weekly_sunday, 21, None, id="w-sun-signal-int-21"),
        pytest.param(_constant_x_scaler, 1, None, id="scaler-int-1"),
        pytest.param(_constant_x_scaler, None, None, id="scaler-buy-and-hold"),
    ],
)
def test_weight_age_counts_missed_stamps_on_calendar_schedules(weights, freq, stale):
    """On a calendar schedule a decision is stale when the strategy stamps a weight STEP on a non-execution bar
    after it, holds it to the next execution bar, and the step's period gets no decision after it — however
    the book moves on execution bars; weights that keep moving over the gap are not a stamp. Any other
    schedule is not measured."""
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=_IDX)
    w = weights()
    cal = instrument_calendar(prices)
    book = schedule_book(
        w.reindex(_IDX),
        cal,
        execution_bars(cal, "majority"),
        freq,
        1,
        leverage="normalize",
        weight_rows=w,
    )
    age = book.report["weight_age"]
    if stale is None:
        assert age["measured"] is False and age["reason"] and "stale_decisions" not in age, age
    else:
        assert age["measured"] is True and age["stale_decisions"] == stale


@pytest.mark.parametrize("freq", ["W-FRI", "ME"])
def test_a_daily_signal_on_a_crypto_and_equity_panel_is_not_stale(freq):
    """Prices on a 7-day index, the equity's calendar executes: a daily signal moves on every weekend row.
    Daily variation the schedule cannot trade, not a mis-stamp (round 2 flagged 104 of 104 on W-FRI)."""
    prices = pd.DataFrame({"A": 100.0, "BTC": 100.0}, index=_FULL)
    prices.loc[_FULL.weekday >= 5, "A"] = np.nan
    cal = instrument_calendar(prices)
    signal = pd.DataFrame({"A": np.random.default_rng(3).uniform(0, 1, len(_FULL))}, index=_FULL)
    signal["BTC"] = 1 - signal["A"]
    book = schedule_book(
        signal,
        cal,
        execution_bars(cal, "A", prices["A"]),
        freq,
        1,
        leverage="normalize",
        weight_rows=signal,
    )
    assert book.report["weight_age"]["stale_decisions"] == 0


def test_a_decision_past_the_last_bar_is_not_counted():
    """Data end on Friday 29 Mar 2024: March's decision never executes, so the Sunday stamp after it is not
    a stale decision (it was counted before)."""
    idx = pd.bdate_range("2023-01-01", "2024-03-29")
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)
    w = _calendar_me().loc[:"2024-03-31"]
    cal = instrument_calendar(prices)
    book = schedule_book(
        w.reindex(idx),
        cal,
        execution_bars(cal, "majority"),
        "ME",
        1,
        leverage="normalize",
        weight_rows=w,
    )
    age = book.report["weight_age"]
    assert book.report["timing"]["decisions_past_last_bar"] == 1
    assert age["decisions"] == book.report["timing"]["executed_decisions"]
    assert "2024-03-29T00:00:00" not in age["first_stale"]
    assert age["stale_decisions"] == 3  # Apr, Sep, Dec 2023


def test_a_strategy_with_rows_only_on_its_stamps_shrinks_the_panel_and_says_so(tmp_path, caplog):
    """The price/weight intersection keeps only the bars that carry a weight row: a strategy writing rows on
    its month-end stamps alone runs on those few bars. Not changed here (a separate card) — counted, warned."""
    prices = _month_end_panel()
    stamps = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"])
    weights = pd.DataFrame({"A": [1.0, 0.0, 1.0, 0.0], "B": [0.0, 1.0, 0.0, 1.0]}, index=stamps)
    with caplog.at_level(logging.WARNING):
        result, store = _run(tmp_path, prices, weights, rebalancing_freq="ME")
    al = _validation(store)["index_alignment"]
    assert al["bars_used"] == 3 and al["weight_rows_dropped"] == 1
    # the shrunk calendar has no March bar: March's Sunday stamp is a step whose period gets no decision
    age = _validation(store)["weight_age"]
    assert age["stale_decisions"] == 1 and age["first_missed_stamps"] == ["2024-03-31T00:00:00"]
    assert al["warmup_price_bars_dropped"] == 22  # January before the first stamp
    assert al["price_bars_dropped"] == len(prices) - 22 - 3
    assert result.metrics["index_price_bars_dropped"] == float(al["price_bars_dropped"])
    assert "INDEX:" in caplog.text
    assert validate_data_validation(_validation(store)) == []


@pytest.mark.parametrize(
    ("weights", "timing"),
    [
        pytest.param(
            lambda full: (
                pd.DataFrame(
                    {"A": [0.5, 1.0, 0.0, 1.0, 0.0], "B": [0.5, 0.0, 1.0, 0.0, 1.0]},
                    index=pd.DatetimeIndex(["2024-01-01", "2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"]),
                )
                .reindex(full)
                .ffill()
            ),
            True,
            id="month-end-stamps-sunday-march",
        ),
        pytest.param(
            lambda full: pd.DataFrame({"A": 0.5, "B": 0.5}, index=full).mul(
                pd.Series(np.random.default_rng(4).uniform(0.8, 1.2, len(full)), index=full), axis=0
            ),
            False,
            id="7-day-drift",
        ),
    ],
)
def test_weight_rows_on_non_price_dates_are_data_not_an_index_warning(tmp_path, caplog, weights, timing):
    """A 7-day weight panel on weekday prices: its weekend rows are counted, never warned by INDEX (round 3:
    a drifting 7-day book was told 208 rows CHANGE the weights). A missed weekend STAMP is TIMING's."""
    prices = _month_end_panel()
    full = pd.date_range("2024-01-01", "2024-04-30", freq="D")
    with caplog.at_level(logging.WARNING):
        _, store = _run(tmp_path, prices, weights(full), rebalancing_freq="ME")
    al = _validation(store)["index_alignment"]
    assert al["weight_rows_dropped"] == len(full) - len(prices) and al["price_bars_dropped"] == 0
    assert "INDEX:" not in caplog.text
    assert ("MISSED their period" in caplog.text) is timing


def test_an_unmeasured_schedule_has_no_weight_age_metric(tmp_path):
    """An int schedule has no period: weight_age says measured false, and no metric reads 0 as clean."""
    prices = _month_end_panel()
    result, store = _run(tmp_path, prices, {"A": 0.5, "B": 0.5}, rebalancing_freq=5)
    age = _validation(store)["weight_age"]
    assert age["measured"] is False and "not a calendar offset" in age["reason"] and "stale_decisions" not in age
    assert "decision_stale_weights" not in result.metrics
    assert validate_data_validation(_validation(store)) == []


def test_a_strategy_with_a_row_on_every_bar_drops_nothing(tmp_path, caplog):
    prices = _month_end_panel()
    with caplog.at_level(logging.WARNING):
        result, store = _run(tmp_path, prices, {"A": 0.5, "B": 0.5})
    al = _validation(store)["index_alignment"]
    assert al["price_bars_dropped"] == al["weight_rows_dropped"] == al["warmup_price_bars_dropped"] == 0
    assert "INDEX:" not in caplog.text


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
        "leverage": "borrow",
        "financing": {"rate": "CASH", "borrow_spread_bps": borrow_bps, "lend_spread_bps": lend_bps},
    }
    result, store = _run(tmp_path, prices, {"A": w_a, "CASH": 0.0}, rebalancing_freq=1, venue=venue)

    traded = store.read_parquet("traded_weights").set_index("date")
    assert (traded["A"].iloc[1:] == w_a).all()
    assert LEND not in traded.columns and BORROW not in traded.columns  # the saved book is the real one
    assert result.metrics["engine_underfilled_rebalances"] == 0.0
    assert result.notes["financing"]["modelled"] is True
    assert result.notes["venue"]["financing"]["rate"] == {"ticker": "CASH"}
    assert result.notes["venue"]["leverage"] == "borrow"

    returns = _returns(store)
    r_a = prices["A"].pct_change()
    r_cash = prices["CASH"].pct_change()
    dt = 1.0 / 365.0
    cash_w = 1.0 - w_a
    spread = borrow_bps if cash_w < 0 else -lend_bps
    expected = w_a * r_a + cash_w * (r_cash + spread / 1e4 * dt)
    # Bar 0 is flat (lag 1), bar 1 is the first fill; from bar 2 every return is the held book's.
    np.testing.assert_allclose(returns.iloc[2:].to_numpy(), expected.iloc[2:].to_numpy(), rtol=1e-9, atol=1e-12)


def test_normalize_is_the_default_and_scales_a_levered_decision_to_net_one(tmp_path, caplog):
    """No venue.leverage on vectorbt = normalize: every decided 1.5 row is scaled by exactly 2/3 in the
    DECISION (TOM-1520), each one counted, and the held book is A at 1.0 — its return is A's, no cut buys."""
    prices = _levered_panel()
    with caplog.at_level(logging.WARNING):
        result, store = _run(tmp_path, prices, {"A": 1.5, "CASH": 0.0}, rebalancing_freq=1)

    traded = store.read_parquet("traded_weights").set_index("date")
    np.testing.assert_allclose(traded["A"].iloc[1:].to_numpy(), 1.0, rtol=1e-12)
    n_rebalances = len(prices) - 1  # the last decision has no execution bar
    decision = _validation(store)["decision"]
    assert decision["rules"]["leverage"] == "normalize"
    assert decision["rows_normalised"] == len(prices) == decision["rows_above_net_1"]
    assert decision["scale_mean"] == pytest.approx(2 / 3) and decision["scale_min"] == pytest.approx(2 / 3)
    assert decision["max_net_exposure_decided"] == pytest.approx(1.5)
    assert decision["max_net_exposure_final"] == pytest.approx(1.0)
    lev = _validation(store)["leverage"]
    assert lev["mode"] == "normalize" and lev["rebalances"] == n_rebalances
    assert lev["scaled_rebalances"] == 0 and lev["cash_capped_rebalances"] == 0  # the seam only measures
    assert lev["max_net_exposure_decided"] == pytest.approx(1.0)  # the seam receives final targets
    assert result.metrics["leverage_normalised_rows"] == float(len(prices))
    assert result.notes["venue"]["leverage"] == "normalize" and result.notes["venue"]["financing"] is None
    assert "SCALED to net 1" in caplog.text
    assert result.metrics["engine_underfilled_rebalances"] == 0.0
    returns = _returns(store)
    r_a = prices["A"].pct_change()
    np.testing.assert_allclose(returns.iloc[2:].to_numpy(), r_a.iloc[2:].to_numpy(), rtol=1e-9, atol=1e-12)


def test_borrow_without_financing_runs_at_an_assumed_zero_rate_and_says_so(tmp_path, caplog):
    prices = _levered_panel()
    with caplog.at_level(logging.WARNING):
        result, store = _run(
            tmp_path,
            prices,
            {"A": 1.5, "CASH": 0.0},
            rebalancing_freq=1,
            venue={"allow_shorts": False, "leverage": "borrow"},
        )
    traded = store.read_parquet("traded_weights").set_index("date")
    assert (traded["A"].iloc[1:] == 1.5).all()
    assert result.metrics["engine_underfilled_rebalances"] == 0.0
    fin = result.notes["venue"]["financing"]
    assert fin["assumed"] is True and fin["rate"] == {"annual": 0.0}
    assert result.notes["financing"]["modelled"] is True
    assert "ASSUMED FREE" in caplog.text
    returns = _returns(store)
    r_a = prices["A"].pct_change()
    np.testing.assert_allclose(returns.iloc[2:].to_numpy(), 1.5 * r_a.iloc[2:].to_numpy(), rtol=1e-9, atol=1e-12)


def _rotation(w_b: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Review round 2's probe: the book rotates from B to A; the execution bar (4 Jan) is one B does
    not print, so B's sale is DEFERRED to 5 Jan, the bar A rises 10%."""
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    prices = pd.DataFrame(
        {
            "A": [100, 100, 100, 100, 110, 110, 110, 110.0],
            "B": [100, 100, 100, np.nan, 100, 100, 100, 100.0],
            "C": [100.0] * 8,
        },
        index=idx,
    )
    decided = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    decided.loc[: idx[1], "B"] = w_b
    decided.loc[idx[2] :, "A"] = 1.0
    return prices, decided


def _total_return(store: FileArtifactStore) -> float:
    return float((1 + _returns(store)).prod() - 1)


@pytest.mark.parametrize(
    ("w_b", "venue", "a_held", "expected", "counter"),
    [
        # B's deferred 0.5 leaves cash for 0.5 of A: A's buy is capped at it -> half of A's 10%.
        (0.5, None, 0.5, 0.05, "cash_capped_rebalances"),
        # B's deferred 1.0 leaves NO cash: A's buy is capped to 0 — held, recorded and counted, not cut by the engine.
        (1.0, None, 0.0, 0.0, "cash_capped_rebalances"),
        # borrow holds both (net 2): A's 10% is booked, the held net 2 is counted and warned on.
        (1.0, {"allow_shorts": False, "leverage": "borrow"}, 1.0, 0.10, "rebalances_above_net_1"),
    ],
)
def test_leverage_applies_to_the_book_held_after_a_deferral(tmp_path, caplog, w_b, venue, a_held, expected, counter):
    """A deferred cell keeps its weight, so a FINAL target (net 1) can still need cash the account does not
    have. Without borrow the buys are capped at the cash plus the sell proceeds (TOM-1520; it used to be the
    held-book normalisation, same numbers here). traded_weights is that held book, and the engine holds it."""
    prices, decided = _rotation(w_b)
    params: dict[str, Any] = {"rebalancing_freq": 1}
    if venue is not None:
        params["venue"] = venue
    with caplog.at_level(logging.WARNING):
        result, store = _run(tmp_path, prices, decided, **params)
    traded = store.read_parquet("traded_weights").set_index("date")
    jan4 = pd.Timestamp("2024-01-04")
    assert traded.loc[jan4, "B"] == pytest.approx(w_b)  # deferred: B could not trade
    assert traded.loc[jan4, "A"] == pytest.approx(a_held)
    assert result.metrics["engine_underfilled_rebalances"] == 0.0  # the engine held traded_weights
    assert _total_return(store) == pytest.approx(expected, abs=1e-9)
    assert _validation(store)["leverage"][counter] == 1
    if venue is not None:
        assert _validation(store)["leverage"]["max_net_exposure_held"] == pytest.approx(2.0)
        assert "ASSUMED FREE" in caplog.text


def test_a_short_cover_is_a_buy_and_shares_the_cash_cap():
    """Deferred B is 1.0 (held over from a 1.0 / -0.2 book, net 0.8); covering the -0.2 short in C RAISES
    the held net like any buy. The 0.2 of cash left is shared by the cover and the buy of A, scaled by the
    same factor 0.2 / 0.7 (TOM-1520; the held-book normalisation used to make both wait): held net 1."""
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame(
        {"A": [100.0] * 6, "B": [100, 100, 100, np.nan, 100, 100.0], "C": [100.0] * 6, "D": [100.0] * 6},
        index=idx,
    )
    decided = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    decided.loc[: idx[1], ["B", "C"]] = [1.0, -0.2]  # net 0.8
    decided.loc[idx[2] :, "A"] = 0.5  # rotate: sell B (deferred), cover C, buy A
    cal, book = _book(prices, decided)
    jan4 = pd.Timestamp("2024-01-04")
    scale = 0.2 / 0.7
    assert book.weights.loc[jan4, "B"] == pytest.approx(1.0)  # deferred
    assert book.weights.loc[jan4, "C"] == pytest.approx(-0.2 + 0.2 * scale)
    assert book.weights.loc[jan4, "A"] == pytest.approx(0.5 * scale)
    assert (book.weights.sum(axis=1) <= 1.0 + 1e-9).all()
    assert book.report["leverage"]["cash_capped_rebalances"] == 1
    assert book.report["leverage"]["cash_cap_scale_min"] == pytest.approx(scale)


def test_the_fill_check_runs_on_a_threshold_run_too(tmp_path):
    """The underfill measurement is the backstop for a book that asks for more cash than it has, so a
    threshold run measures it as well (on the bars the engine traded)."""
    prices, decided = _rotation(0.5)
    result, _ = _run(tmp_path, prices, decided, rebalancing_freq=1, threshold=0.01)
    assert "engine_underfilled_rebalances" in result.metrics
    assert result.metrics["engine_underfilled_rebalances"] == 0.0


@pytest.mark.parametrize("leverage", ["borrow", "normalize"])
def test_a_deferred_drifted_position_never_starves_the_cash_legs(tmp_path, leverage):
    """A is closed on a third of the bars while B moves: its deferred position DRIFTS away from its
    old target. The cash legs take the residual of what is really held, so no buy is ever cut."""
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(11)
    prices = pd.DataFrame(
        {k: 100.0 * np.cumprod(1 + rng.normal(0.0, 0.03, len(idx))) for k in ("A", "B", "C")}, index=idx
    )
    prices.loc[prices.index[1::3], "A"] = np.nan
    decided = pd.DataFrame({"A": np.where(np.arange(len(idx)) % 2, 1.2, -0.8), "B": 0.6, "C": -0.3}, index=idx)
    venue = {"allow_shorts": True, "leverage": leverage, "financing": {"rate": 0.02}}
    result, store = _run(tmp_path, prices, decided, rebalancing_freq=1, venue=venue)
    assert result.metrics["calendar_deferred_trades"] > 10
    assert result.metrics["engine_underfilled_rebalances"] == 0.0
    assert result.metrics["engine_max_fill_gap"] < 1e-9


def test_without_financing_an_unlevered_vectorbt_book_runs_and_is_filled(tmp_path):
    result, store = _run(tmp_path, _levered_panel(), {"A": 1.0, "CASH": 0.0}, rebalancing_freq=1)
    assert result.metrics["engine_underfilled_rebalances"] == 0.0
    assert _validation(store)["leverage"]["scaled_rebalances"] == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_the_default_leverage_is_normalize_on_every_engine(tmp_path, engine):
    """One default for every engine (ADR-0008, TOM-1450): until then rsims borrowed by default."""
    result, store = _run(tmp_path, _levered_panel(), {"A": 1.5, "CASH": 0.0}, engine=engine)
    traded = store.read_parquet("traded_weights").set_index("date")
    assert (traded["A"].iloc[1:] == 1.0).all()
    assert result.notes["venue"]["leverage"] == "normalize"
    assert result.notes["venue"]["financing"] is None


def test_rsims_borrows_when_declared_it_has_no_cash_floor(tmp_path):
    venue = {"allow_shorts": False, "leverage": "borrow"}
    result, store = _run(tmp_path, _levered_panel(), {"A": 1.5, "CASH": 0.0}, engine="rsims", venue=venue)
    traded = store.read_parquet("traded_weights").set_index("date")
    assert (traded["A"].iloc[1:] == 1.5).all()
    assert result.notes["venue"]["leverage"] == "borrow"
    assert result.notes["venue"]["financing"]["assumed"] is True
    assert result.notes["financing"]["modelled"] is False  # a margin simulator needs no cash legs


def test_the_cash_legs_trade_without_fees(tmp_path):
    """fees 1%: buying 0.4 A costs 0.4%. If the 0.6 LEND leg paid fees too, the run would lose 1%."""
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)
    venue = {"allow_shorts": False, "financing": {"rate": 0.0}}
    result, _ = _run(tmp_path, prices, {"A": 0.4, "B": 0.0}, rebalancing_freq=1, fees=0.01, venue=venue)
    assert result.metrics["total_return"] == pytest.approx(-0.004, abs=2e-4)


def test_a_rate_ticker_that_stops_printing_early_is_counted(tmp_path, caplog):
    prices = _levered_panel()
    prices.loc[prices.index > prices.index[50], "CASH"] = np.nan
    venue = {"allow_shorts": False, "financing": {"rate": "CASH"}}
    with caplog.at_level(logging.WARNING):
        result, _ = _run(tmp_path, prices, {"A": 0.5, "CASH": 0.0}, rebalancing_freq=1, venue=venue)
    assert result.notes["financing"]["rate_stale_bars_at_end"] == 9
    assert result.metrics["financing_rate_stale_bars_at_end"] == 9.0
    assert "stops printing" in caplog.text


def test_the_fill_gap_measurement_sees_a_cut_book():
    """The counter the financing tests read as 0 must be able to fail: hand vectorbt a 1.5x book
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


def test_plan_refuses_a_malformed_financing_or_leverage_before_any_data():
    with pytest.raises(ValueError, match="'rate' is required"):
        BacktestPipeline().plan({"venue": {"allow_shorts": True, "financing": {"lend_spread_bps": 5}}})
    with pytest.raises(ValueError, match="venue.leverage must be one of"):
        BacktestPipeline().plan({"venue": {"allow_shorts": True, "leverage": "yes"}})
    with pytest.raises(ValueError, match="execution.calendar must be"):
        BacktestPipeline().plan({"execution": {"calendar": 3}})


def test_a_rate_ticker_that_starts_after_the_backtest_is_refused(tmp_path):
    prices = _levered_panel()
    prices.loc[prices.index < prices.index[10], "CASH"] = np.nan
    venue = {"allow_shorts": False, "financing": {"rate": "CASH"}}
    with pytest.raises(ValueError, match="financing rate there is unknown"):
        _run(tmp_path, prices, {"A": 1.5}, rebalancing_freq=1, venue=venue)


# ----------------------------------------------------------------------
# The schedule is bars; data_validation.json has a schema
# ----------------------------------------------------------------------


def test_a_calendar_rebalance_date_that_is_not_a_bar_snaps_to_a_bar():
    bars = pd.bdate_range("2023-12-01", "2024-03-29").drop(pd.Timestamp("2024-01-01"))  # no 1 January row
    bms = rebalancing_dates(bars, "BMS")
    # 1 January is the BMS date and not a bar: January's rebalance is the first bar after it.
    assert list(bms) == list(pd.to_datetime(["2023-12-01", "2024-01-02", "2024-02-01", "2024-03-01"]))
    # W-SUN on weekday data is a period END: every week decides on its last bar, the Friday
    # (it used to be no rebalance at all).
    weekdays = pd.bdate_range("2024-01-02", "2024-03-29")
    weekly = rebalancing_dates(weekdays, "1W")
    assert len(weekly) == 13 and all(d.dayofweek == 4 for d in weekly)
    assert weekly.isin(weekdays).all()


def test_data_validation_json_matches_its_schema(tmp_path):
    _, store = _run(tmp_path, _blocker_panel(), {"A": 0.5, "B": 0.0}, rebalancing_freq="BMS")
    report = _validation(store)
    assert validate_data_validation(report) == []
    broken = {k: v for k, v in report.items() if k != "staleness"}
    assert validate_data_validation(broken)  # the schema can fail
