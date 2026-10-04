"""The two ways the seam turns DECIDED weights into a traded book, both through one lag.

:func:`simulate_book` — the scheduled book (``quantbox run``, every variant):
instrument and execution calendars, decision vs execution bars, deferral,
``venue.leverage``, financing legs (docs/adr/0007), then the adapter.

:func:`simulate_weights` — the bar-grid book (``backtest()``, ``optimize()``,
the sweep): the weights are lagged on the engine's own bar grid and the
adapter trades them on its schedule.

Both move every decision with :func:`quantbox.engine._lag.lag_positions`
(through :mod:`quantbox.engine.schedule` and :func:`~quantbox.engine._lag.lag_frame`),
so a run never depends on which door it came through for WHEN it trades.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

from quantbox.execution import ExecutionTiming, timing_record
from quantbox.financing import CASH_LEGS, Financing, add_cash_legs
from quantbox.instrument_calendar import (
    DATA_VALIDATION_SCHEMA,
    execution_bars,
    execution_calendar_report,
    instrument_calendar,
)

from ._lag import lag_buy_and_hold, lag_frame
from .base import Costs, EngineAdapter, TradedBook
from .registry import get_engine
from .schedule import schedule_book

logger = logging.getLogger(__name__)


def simulate_book(
    prices: pd.DataFrame,
    decided: pd.DataFrame,
    *,
    engine: str | EngineAdapter,
    timing: ExecutionTiming,
    costs: Costs = Costs(),
    rebalancing_freq: Any = 1,
    threshold: float | None = None,
    leverage: str,
    financing: Financing | None = None,
    funding: pd.DataFrame | None = None,
    engine_params: Mapping[str, Any] | None = None,
    trading_days: int = 365,
    where: str = "",
) -> TradedBook:
    """Decided weights -> the scheduled, lagged, financed book, simulated (docs/adr/0007, 0008).

    *decided* is the strategy's book after overlays and risk transforms, NOT
    lagged. The result's ``weights`` / ``prices`` are the REAL book (what is
    held after each bar's orders, saved as ``traded_weights``); the engine
    itself may also have traded the financing cash legs.
    """
    adapter = get_engine(engine)
    staged = _schedule(prices, decided, adapter, timing, rebalancing_freq, leverage, financing, where)
    book = adapter.run(
        staged["engine_prices"],
        staged["engine_weights"],
        costs=costs,
        orders=staged["orders"],
        threshold=threshold,
        funding=funding,
        cash_legs=CASH_LEGS if staged["legs"] else (),
        params=engine_params,
        trading_days=trading_days,
    )
    book.weights = staged["weights"]
    book.prices = staged["prices"]
    book.schedule = staged["schedule"]
    book.financing = staged["financing"]
    book.book_metrics = staged["metrics"]
    book.data_validation = staged["data_validation"]
    book.execution = timing_record(timing)
    return book


def simulate_weights(
    prices: pd.DataFrame,
    decided: pd.DataFrame | dict[str, pd.DataFrame],
    *,
    engine: str | EngineAdapter,
    timing: ExecutionTiming,
    costs: Costs = Costs(),
    rebalancing_freq: Any = 1,
    threshold: float | None = None,
    funding: pd.DataFrame | None = None,
    engine_params: Mapping[str, Any] | None = None,
    trading_days: int = 365,
    leading: str = "flat",
) -> TradedBook | None:
    """Decided weights -> the lagged book on the engine's bar grid, simulated.

    ``leading="flat"`` (``backtest()``, ``optimize()``): the grid is
    ``prices.index | weights.index``; a sparse frame (rebalance dates only) is
    put on it first, so the lag moves a decision one BAR, not one rebalance;
    bars before the first decision are flat, and a buy-and-hold book
    (``rebalancing_freq=None``) enters on the bar its first decision fills.
    A dict of frames is one strategy each.

    ``leading="drop"`` (the sweep): bars with no decision behind them (the lag's
    first bars, a strategy's warm-up) are dropped and prices are carried over
    gaps; the book may carry MultiIndex columns (one strategy slice each, the
    ticker last). Returns None when fewer than two bars remain.
    """
    adapter = get_engine(engine)
    if leading == "flat":
        weights = _lag_on_grid(prices, decided, timing)
        grid = prices.index
        for w in weights.values() if isinstance(weights, dict) else [weights]:
            grid = grid.union(w.index)
        schedule = lag_buy_and_hold(pd.to_datetime(grid), rebalancing_freq, timing.lag_bars, timing.same_bar)
        engine_prices = prices
    elif leading == "drop":
        if isinstance(decided, dict):
            raise ValueError("leading='drop' takes one weights frame (MultiIndex columns for several slices)")
        lagged = lag_frame(decided, timing.lag_bars, same_bar=timing.same_bar, fill_leading=None)
        common = lagged.dropna(how="all").index.intersection(prices.index)
        if len(common) < 2:
            return None
        weights = lagged.reindex(common)
        tickers = weights.columns.get_level_values(-1).unique()
        missing = [t for t in tickers if t not in prices.columns]
        if missing:  # never manufacture a price column: an engine would trade a flat book on it
            raise ValueError(f"All tickers in weights must be present in prices (missing {missing})")
        engine_prices = prices.reindex(common).reindex(columns=tickers).ffill().bfill()
        schedule = rebalancing_freq
    else:
        raise ValueError(f"leading must be 'flat' or 'drop', got {leading!r}")
    book = adapter.run(
        engine_prices,
        weights,
        costs=costs,
        rebalancing_freq=schedule,
        threshold=threshold,
        funding=funding,
        params=engine_params,
        trading_days=trading_days,
    )
    book.prices = engine_prices
    book.execution = timing_record(timing)
    return book


def _lag_on_grid(
    prices: pd.DataFrame, weights: dict[str, pd.DataFrame] | pd.DataFrame, timing: ExecutionTiming
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """Lag each frame on ``prices.index | its own index`` — one BAR, not one row of a sparse frame.

    Cells stay NaN, so the engine's own NaN policy is unchanged; only the
    decision moves ``lag_bars`` bars later.
    """

    def one(w: pd.DataFrame) -> pd.DataFrame:
        return lag_frame(w.reindex(prices.index.union(w.index)), timing.lag_bars, same_bar=timing.same_bar)

    if isinstance(weights, dict):
        return {name: one(w) for name, w in weights.items()}
    return one(weights)


def _schedule(
    prices_wide: pd.DataFrame,
    weights: pd.DataFrame,
    adapter: EngineAdapter,
    timing: ExecutionTiming,
    rebalancing_freq: Any,
    leverage: str,
    financing: Financing | None,
    where: str,
) -> dict[str, Any]:
    """Decided weights -> the book the engine trades, for every engine the same way (docs/adr/0007).

    1. each instrument's calendar (:func:`quantbox.instrument_calendar.instrument_calendar`)
       and the EXECUTION calendar (``execution.calendar``,
       :func:`~quantbox.instrument_calendar.execution_bars`);
    2. decision bars, the execution lag counted in execution bars, per-instrument
       deferral of unprinted orders, ``venue.leverage``, input staleness
       (:func:`quantbox.engine.schedule.schedule_book`);
    3. financing: with ``venue.financing`` (or ``venue.leverage: borrow`` on an
       engine that does not model margin, at an assumed rate of 0) the
       LEND/BORROW cash legs are appended (:func:`quantbox.financing.add_cash_legs`).
    """
    common_idx = prices_wide.index.intersection(weights.index)
    common_cols = [c for c in weights.columns if c in prices_wide.columns]
    if not common_cols:
        raise ValueError("No overlapping tickers between prices and weights")
    alignment = _index_alignment(prices_wide.index, weights, common_idx, where)

    cal = instrument_calendar(prices_wide.loc[common_idx, common_cols])
    reference = None
    if timing.calendar in prices_wide.columns:
        reference = prices_wide[timing.calendar]
    exec_bars = execution_bars(cal, timing.calendar, reference)
    book = schedule_book(
        weights.loc[common_idx, common_cols],
        cal,
        exec_bars,
        rebalancing_freq,
        timing.lag_bars,
        engine=adapter,
        leverage=leverage,
        weight_rows=weights[common_cols],
        same_bar=timing.same_bar,
    )
    bt_prices, bt_weights, orders = cal.prices, book.weights, book.orders

    exec_report = execution_calendar_report(exec_bars, timing.calendar)
    calendar = cal.report
    for name, extra in book.report["instruments"].items():
        calendar["instruments"].setdefault(name, {}).update(extra)
    timing_report = dict(book.report["timing"])
    lev = book.report["leverage"]
    metrics: dict[str, float] = {
        "calendar_ffilled_bars": float(calendar["totals"]["ffilled_bars"]),
        "calendar_targeted_outside_window_bars": float(timing_report["targeted_outside_window_bars"]),
        "calendar_deferred_trades": float(timing_report["deferred_trades"]),
        "calendar_legacy_coverage_drop": float(calendar["legacy_coverage_drop"]["count"]),
        "execution_calendar_bars": float(exec_report["execution_bars"]),
        "execution_calendar_bar_share": exec_report["execution_bars"] / max(exec_report["total_bars"], 1),
        "decision_stale_inputs": float(book.report["staleness"]["stale_decisions"]),
        "decision_max_staleness_bars": float(book.report["staleness"]["max_bars"]),
        "index_price_bars_dropped": float(alignment["price_bars_dropped"]),
        "index_weight_rows_dropped": float(alignment["weight_rows_dropped"]),
        "leverage_rebalances_above_net_1": float(lev["rebalances_above_net_1"]),
        "leverage_scaled_rebalances": float(lev["scaled_rebalances"]),
        "leverage_scale_mean": lev["scale_mean"],
        "leverage_scale_min": lev["scale_min"],
        "leverage_scaled_after_deferral": float(lev["scaled_after_deferral"]),
        "leverage_buys_zeroed_rebalances": float(lev["buys_zeroed_rebalances"]),
        "leverage_max_net_exposure_held": lev["max_net_exposure_held"],
    }
    age = book.report["weight_age"]
    if age["measured"]:  # no metric at all when unmeasured: a 0 would read as clean
        metrics["decision_stale_weights"] = float(age["stale_decisions"])
        metrics["decision_stale_weights_held_bars_max"] = float(age["stale_held_bars_max"])
    legs = financing is not None and not (financing.assumed and adapter.models_margin)
    if not legs:
        # A margin simulator at an assumed rate of 0 is what it already does: no legs needed.
        eng_prices, eng_weights, eng_orders = bt_prices, bt_weights, orders
        fin_record = {"modelled": False} if financing is None else {"modelled": False, **financing.record()}
    else:
        eng_prices, eng_weights, fin_record = add_cash_legs(bt_prices, bt_weights, financing, prices_wide)
        fin_record = {"modelled": True, **fin_record}
        eng_orders = orders.copy()
        traded_bar = orders.any(axis=1)
        for leg in CASH_LEGS:
            eng_orders[leg] = traded_bar
        metrics.update(
            financing_mean_cash_weight=fin_record["mean_cash_weight"],
            financing_min_cash_weight=fin_record["min_cash_weight"],
            financing_borrow_bar_share=fin_record["borrow_bar_share"],
            financing_rate_stale_bars_at_end=float(fin_record["rate_stale_bars_at_end"]),
        )
    if financing is not None and financing.assumed and lev["rebalances_above_net_1"]:
        logger.warning(
            "FINANCING — %s%d rebalance(s) HOLD net exposure above 1 (max %.4f) under venue.leverage: borrow "
            "with NO venue.financing block: borrowing is ASSUMED FREE (rate 0), recorded as "
            "venue.financing.assumed in the manifest. Declare venue.financing to price it.",
            where,
            lev["rebalances_above_net_1"],
            lev["max_net_exposure_held"],
        )
    return {
        "prices": bt_prices,
        "weights": bt_weights,
        "engine_prices": eng_prices,
        "engine_weights": eng_weights,
        "orders": eng_orders,
        "legs": legs,
        "schedule": book.schedule,
        "financing": fin_record,
        "metrics": metrics,
        "data_validation": {
            "schema": DATA_VALIDATION_SCHEMA,
            "calendar": calendar,
            "execution_calendar": exec_report,
            "timing": timing_report,
            "staleness": book.report["staleness"],
            "weight_age": book.report["weight_age"],
            "index_alignment": alignment,
            "leverage": lev,
        },
    }


def _index_alignment(price_index: pd.Index, weights: pd.DataFrame, common_idx: pd.Index, where: str) -> dict[str, Any]:
    """What the price/weight index INTERSECTION drops (``data_validation.json`` ``index_alignment``).

    The backtest runs on the bars where both prices and a strategy weight
    row exist. Price bars before the strategy's first row are its warm-up
    (counted, not warned); any OTHER dropped price bar — a strategy that
    writes rows only on its stamp dates shrinks the whole panel to them — is
    warned. Weight rows on dates with no price bar (the weekend rows of a
    7-day panel) are counted only: a weight step stamped there is the weight
    age's to judge (``TIMING:``).
    """
    first_row = weights.index.min() if len(weights.index) else None
    dropped_prices = price_index.difference(common_idx)
    warmup = dropped_prices[dropped_prices < first_row] if first_row is not None else dropped_prices[:0]
    shrink = dropped_prices.difference(warmup)
    dropped_rows = weights.index.difference(common_idx)
    record = {
        "price_bars": int(len(price_index)),
        "weight_rows": int(len(weights.index)),
        "bars_used": int(len(common_idx)),
        "warmup_price_bars_dropped": int(len(warmup)),
        "price_bars_dropped": int(len(shrink)),
        "first_price_bars_dropped": [pd.Timestamp(t).isoformat() for t in shrink[:5]],
        "weight_rows_dropped": int(len(dropped_rows)),
    }
    if len(shrink):
        logger.warning(
            "INDEX: %sthe backtest runs only on the %d bar(s) that carry a strategy weight row: %d price bar(s) "
            "after the strategy's first row have none and were DROPPED (first %s). Write a weight row on every "
            "price bar.",
            where,
            len(common_idx),
            len(shrink),
            record["first_price_bars_dropped"],
        )
    return record
