"""THE book function — decided weights -> the scheduled, lagged, financed book, executed (docs/adr/0007, 0008).

:func:`simulate` is the one way quantbox turns DECIDED weights into a traded
book, for every door (``quantbox run`` and its variants, ``backtest()``,
``optimize()``, the sweep) and every engine:

1. the price/weight alignment and each instrument's calendar;
2. the schedule (:func:`quantbox.engine.schedule.schedule_book`): the NaN
   policy, decision bars, the lag, deferral, ``venue.leverage``, the threshold —
   a held book and a per-cell ORDERS mask;
3. the financing cash legs (:mod:`quantbox.financing`), when declared;
4. the adapter, which only executes the orders (:meth:`EngineAdapter.execute`).

``execution.schedule`` picks the calendar. ``calendar`` (the default) is the
scheduled book of ADR-0007. ``bars`` is the SAME function on a degenerate
calendar: every price bar is an execution bar on which every instrument
prints (no deferral), and ``venue.leverage`` is measured, never applied. It is
not a second code path.

A book may carry several strategy slices — a dict of frames, or MultiIndex
columns with the ticker last (the sweep). Each slice is scheduled on its own;
the engine runs them in one batch.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

from quantbox.execution import SCHEDULES, ExecutionTiming, timing_record
from quantbox.financing import CASH_LEGS, DEFAULT_LEVERAGE, Financing, add_cash_legs
from quantbox.instrument_calendar import (
    DATA_VALIDATION_SCHEMA,
    InstrumentCalendar,
    execution_bars,
    execution_calendar_report,
    instrument_calendar,
)

from .base import Costs, EngineAdapter, TradedBook
from .groups import GroupLimits, apply_group_limits
from .policy import RebalancePolicy, policy_execution_bars, schedule_policy
from .registry import get_engine
from .schedule import NAN_POLICY, ScheduledBook, materialise_nan, schedule_book

logger = logging.getLogger(__name__)

__all__ = ["NAN_POLICY", "materialise_nan", "simulate"]


def simulate(
    prices: pd.DataFrame,
    decided: pd.DataFrame | dict[str, pd.DataFrame],
    *,
    engine: str | EngineAdapter,
    timing: ExecutionTiming,
    costs: Costs = Costs(),
    rebalancing_freq: Any = 1,
    threshold: float | None = None,
    policy: Mapping[str, Any] | RebalancePolicy | None = None,
    groups: GroupLimits | None = None,
    leverage: str | None = None,
    financing: Financing | None = None,
    funding: pd.DataFrame | None = None,
    engine_params: Mapping[str, Any] | None = None,
    trading_days: int = 365,
    where: str = "",
) -> TradedBook:
    """Decided weights -> the traded book, through the one schedule, on *engine* (see the module docstring).

    *decided* is the strategy's book after overlays and risk transforms, NOT
    lagged: row ``t`` was decided with data through ``close[t]``. A dict of
    frames, or MultiIndex columns (the ticker last), is one strategy slice each.
    ``timing.schedule`` is ``calendar`` or ``bars``. *policy* is the
    rebalancing policy (:mod:`quantbox.engine.policy`: periodic, tranche, band,
    corridor; a mapping or a resolved :class:`RebalancePolicy`); without it
    *rebalancing_freq* and *threshold* are the schedule (periodic, or a band).
    Declaring both is refused. Every engine gets the same schedule.
    *groups* (bound to a universe, :class:`quantbox.engine.groups.GroupLimits`)
    keeps each group's gross weight inside its limits on every decided row,
    before the schedule.
    *leverage* is ``venue.leverage``; None = :data:`quantbox.financing.DEFAULT_LEVERAGE`
    on the calendar, and nothing is applied on ``bars``. *financing* (one slice
    only) appends the LEND/BORROW cash legs. The result's ``weights`` /
    ``prices`` / ``orders`` are the REAL book (what is held after each bar's
    orders, saved as ``traded_weights``); the engine itself may also have
    traded the financing cash legs.
    """
    adapter = get_engine(engine)
    _refuse_uncharged_costs(adapter, costs, where)
    if timing.schedule not in SCHEDULES:
        raise ValueError(f"execution.schedule must be one of {list(SCHEDULES)}, got {timing.schedule!r}")
    bars = timing.schedule == "bars"
    if bars and (leverage is not None or financing is not None):
        raise ValueError(
            "execution.schedule: bars applies no venue.leverage and no venue.financing; drop them or use "
            "execution.schedule: calendar"
        )
    pol = schedule_policy(policy, rebalancing_freq, threshold)
    if groups is not None and groups.membership is None:
        raise ValueError("group_limits are not bound to a universe: pass GroupLimits.bind(universe)")
    slices, level_names = _slices(decided)
    if financing is not None and len(slices) > 1:
        raise ValueError("venue.financing takes a book with one strategy slice")
    staged = _stage(
        prices,
        slices,
        timing,
        pol,
        groups,
        "none" if bars else (leverage or DEFAULT_LEVERAGE),
        financing,
        adapter,
        where,
    )
    book = adapter.execute(
        staged["engine_prices"],
        _join(staged["engine_weights"], level_names),
        _join(staged["engine_orders"], level_names),
        costs,
        funding,
        engine_params,
        cash_legs=CASH_LEGS if staged["legs"] else (),
        trading_days=trading_days,
    )
    book.weights = _join(staged["weights"], level_names)
    book.orders = _join(staged["orders"], level_names)
    book.prices = staged["prices"]
    book.schedule = staged["schedule"]
    book.financing = staged["financing"]
    book.book_metrics = staged["metrics"]
    book.data_validation = staged["data_validation"]
    book.execution = timing_record(timing)
    return book


def _refuse_uncharged_costs(adapter: EngineAdapter, costs: Costs, where: str) -> None:
    """A non-zero cost the engine does not charge is refused, never dropped (TOM-1500).

    Every :class:`Costs` field is checked, so a field added later is refused on
    every engine until its adapter says it charges it.
    """
    charged = adapter.charged_costs()
    uncharged = {
        f.name: getattr(costs, f.name)
        for f in dataclasses.fields(Costs)
        if getattr(costs, f.name) and f.name not in charged
    }
    if uncharged:
        raise ValueError(
            f"{where}engine {adapter.name!r} cannot model the cost(s) {uncharged}; it charges "
            f"{sorted(charged) or 'none'}. Set them to 0 or pick an engine that charges them."
        )


def _slices(decided: pd.DataFrame | dict[str, pd.DataFrame]) -> tuple[list[tuple[Any, pd.DataFrame]], list | None]:
    """``[(key, frame with ticker columns), ...]`` and the slice level names (None for a one-slice book)."""
    if isinstance(decided, dict):
        if not decided:
            raise ValueError("weights: an empty dict has no strategy to simulate")
        return list(decided.items()), ["strategy"]
    if decided.columns.nlevels == 1:
        return [(None, decided)], None
    levels = list(range(decided.columns.nlevels - 1))
    out = []
    for key, frame in decided.T.groupby(level=levels[0] if len(levels) == 1 else levels, sort=False):
        w = frame.T
        w.columns = w.columns.get_level_values(-1)
        out.append((key, w))
    return out, list(decided.columns.names[:-1])


def _join(frames: list[tuple[Any, pd.DataFrame]], level_names: list | None) -> pd.DataFrame:
    """The slices back into one frame: MultiIndex columns (the slice levels, the ticker last) when several."""
    if level_names is None:
        return frames[0][1]
    joined = pd.concat({key: frame for key, frame in frames}, axis=1)
    joined.columns = joined.columns.set_names([*level_names, frames[0][1].columns.name])
    return joined


def _missing_tickers(prices: pd.DataFrame, slices: list[tuple[Any, pd.DataFrame]], bars: bool, where: str) -> None:
    """A weight on a ticker with no price column at all: refused on ``bars``, dropped LOUDLY on the calendar.

    The calendar keeps the instruments the prices carry (ADR-0007) and has
    always dropped such a column; it now says so (``WEIGHTS:``). The bar grid
    refuses it, as the helpers did before the one book function (review #235):
    an engine would otherwise trade a flat book on it.
    """
    missing = sorted(
        {str(c) for _, w in slices for c in w.columns if c not in prices.columns and (w[c].fillna(0.0) != 0).any()}
    )
    if not missing:
        return
    if bars:
        raise ValueError(f"All tickers in weights must be present in prices (missing {missing})")
    logger.warning(
        "WEIGHTS: %sthe strategy holds weight on %d ticker(s) with NO price column, DROPPED from the book (its "
        "weight is not traded): %s. Load prices for them or stop targeting them.",
        where,
        len(missing),
        missing,
    )


def _refuse_held_without_price(carried: pd.DataFrame, held: pd.DataFrame) -> None:
    """``schedule: bars``: refuse a weight HELD on a bar before its ticker's first price (review #235).

    The calendar forces such a target to 0 and says so (``targeted_outside_window``);
    the bar grid has no life window, so it refuses instead of trading a price
    that does not exist yet. A weight of 0 before the first print is a normal warm-up.
    """
    p = carried.reindex(index=held.index, columns=held.columns)
    bad = (held.to_numpy() != 0) & p.isna().to_numpy()
    if bad.any():
        rows, cols = bad.nonzero()
        first = [(str(held.index[r]), held.columns[c]) for r, c in list(zip(rows, cols, strict=True))[:5]]
        raise ValueError(
            f"Weight held on {int(bad.sum())} bar(s) with no price yet for its ticker (first {first}): "
            "a book cannot hold an instrument before it prints. Write 0 there, or start the panel later."
        )


def _bar_grid(cal: InstrumentCalendar) -> InstrumentCalendar:
    """``schedule: bars`` — the degenerate calendar: every bar prints for every instrument, all inside its window."""
    every = pd.DataFrame(True, index=cal.observed.index, columns=cal.observed.columns)
    return InstrumentCalendar(prices=cal.prices, observed=every, inside=every, report=cal.report)


def _stage(
    prices_wide: pd.DataFrame,
    slices: list[tuple[Any, pd.DataFrame]],
    timing: ExecutionTiming,
    policy: RebalancePolicy,
    groups: GroupLimits | None,
    leverage: str,
    financing: Financing | None,
    adapter: EngineAdapter,
    where: str,
) -> dict[str, Any]:
    """Decided weights -> the book the engine trades, for every engine the same way (docs/adr/0007, 0008).

    1. each instrument's calendar (:func:`quantbox.instrument_calendar.instrument_calendar`)
       and the EXECUTION calendar (``execution.calendar``,
       :func:`~quantbox.instrument_calendar.execution_bars`) — on ``schedule: bars``
       the degenerate one (:func:`_bar_grid`);
    2. per strategy slice: the NaN policy, the group limits (when declared,
       :func:`quantbox.engine.groups.apply_group_limits`), decision bars (narrowed
       to the policy's market sessions), the execution lag counted in execution
       bars, per-instrument deferral of unprinted orders, ``venue.leverage``,
       the rebalancing policy, input staleness
       (:func:`quantbox.engine.schedule.schedule_book`);
    3. financing: with ``venue.financing`` (or ``venue.leverage: borrow`` on an
       engine that does not model margin, at an assumed rate of 0) the
       LEND/BORROW cash legs are appended (:func:`quantbox.financing.add_cash_legs`).
    """
    _missing_tickers(prices_wide, slices, timing.schedule == "bars", where)
    tickers = list(dict.fromkeys(c for _, w in slices for c in w.columns))
    weight_index = slices[0][1].index
    for _, w in slices[1:]:
        weight_index = weight_index.union(w.index)
    common_idx = prices_wide.index.intersection(weight_index)
    common_cols = [c for c in tickers if c in prices_wide.columns]
    if not common_cols:
        raise ValueError("No overlapping tickers between prices and weights")
    alignment = _index_alignment(prices_wide.index, weight_index, common_idx, where)

    cal = instrument_calendar(prices_wide.loc[common_idx, common_cols])
    if timing.schedule == "bars":
        cal = _bar_grid(cal)
        exec_bars = pd.Series(True, index=cal.observed.index)
    else:
        reference = prices_wide[timing.calendar] if timing.calendar in prices_wide.columns else None
        exec_bars = execution_bars(cal, timing.calendar, reference)
    exec_bars = policy_execution_bars(exec_bars, policy)
    books: list[tuple[Any, ScheduledBook]] = []
    group_reports: list[dict[str, Any]] = []
    for key, w in slices:
        on_bars = w.reindex(index=common_idx, columns=common_cols)
        if groups is not None:
            on_bars, group_report = apply_group_limits(materialise_nan(on_bars), groups)
            group_reports.append(group_report)
        sb = schedule_book(
            on_bars,
            cal,
            exec_bars,
            1,  # the legacy rebalancing_freq default: the schedule is *policy*
            timing.lag_bars,
            leverage=leverage,
            weight_rows=w.reindex(columns=common_cols),
            same_bar=timing.same_bar,
            policy=policy,
        )
        if timing.schedule == "bars":
            _refuse_held_without_price(prices_wide.loc[common_idx, common_cols].ffill(), sb.weights)
        books.append((key, sb))
    bt_prices = cal.prices
    weights = [(key, sb.weights) for key, sb in books]
    orders = [(key, sb.orders) for key, sb in books]

    exec_report = execution_calendar_report(exec_bars, timing.calendar)
    one = books[0][1] if len(books) == 1 else None
    legs = financing is not None and not (financing.assumed and adapter.models_margin)
    fin_record: dict[str, Any] = {"modelled": False} if financing is None else {"modelled": False, **financing.record()}
    eng_prices, eng_weights, eng_orders = bt_prices, weights, orders
    metrics: dict[str, float] = {}
    data_validation = None
    schedule = None
    if one is not None:
        calendar = cal.report
        for name, extra in one.report["instruments"].items():
            calendar["instruments"].setdefault(name, {}).update(extra)
        timing_report = dict(one.report["timing"])
        lev = one.report["leverage"]
        metrics = {
            "calendar_ffilled_bars": float(calendar["totals"]["ffilled_bars"]),
            "calendar_targeted_outside_window_bars": float(timing_report["targeted_outside_window_bars"]),
            "calendar_deferred_trades": float(timing_report["deferred_trades"]),
            "calendar_legacy_coverage_drop": float(calendar["legacy_coverage_drop"]["count"]),
            "execution_calendar_bars": float(exec_report["execution_bars"]),
            "execution_calendar_bar_share": exec_report["execution_bars"] / max(exec_report["total_bars"], 1),
            "decision_stale_inputs": float(one.report["staleness"]["stale_decisions"]),
            "decision_max_staleness_bars": float(one.report["staleness"]["max_bars"]),
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
        age = one.report["weight_age"]
        if age["measured"]:  # no metric at all when unmeasured: a 0 would read as clean
            metrics["decision_stale_weights"] = float(age["stale_decisions"])
            metrics["decision_stale_weights_held_bars_max"] = float(age["stale_held_bars_max"])
        if "threshold" in one.report:
            metrics["threshold_skipped_rebalances"] = float(one.report["threshold"]["skipped_rebalances"])
        if "rebalancing" in one.report:
            reb = one.report["rebalancing"]
            metrics["rebalance_placed"] = float(reb["placed_rebalances"])
            metrics["rebalance_skipped"] = float(reb["skipped_rebalances"])
            metrics["rebalance_partial"] = float(reb["partial_rebalances"])
        if group_reports:
            metrics["group_limit_rows_adjusted"] = float(group_reports[0]["rows_adjusted"])
        if legs:
            p, w, fin = add_cash_legs(bt_prices, one.weights, financing, prices_wide)
            fin_record = {"modelled": True, **fin}
            o = one.orders.copy()
            traded_bar = one.orders.any(axis=1)
            for leg in CASH_LEGS:
                o[leg] = traded_bar
            eng_prices, eng_weights, eng_orders = p, [(None, w)], [(None, o)]
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
        schedule = one.schedule
        data_validation = {
            "schema": DATA_VALIDATION_SCHEMA,
            "calendar": calendar,
            "execution_calendar": exec_report,
            "timing": timing_report,
            "staleness": one.report["staleness"],
            "weight_age": one.report["weight_age"],
            "index_alignment": alignment,
            "leverage": lev,
        }
        if "threshold" in one.report:
            data_validation["threshold"] = one.report["threshold"]
        if "rebalancing" in one.report:
            data_validation["rebalancing"] = one.report["rebalancing"]
        if group_reports:
            data_validation["groups"] = group_reports[0]
    return {
        "prices": bt_prices,
        "weights": weights,
        "orders": orders,
        "engine_prices": eng_prices,
        "engine_weights": eng_weights,
        "engine_orders": eng_orders,
        "legs": legs,
        "schedule": schedule,
        "financing": fin_record,
        "metrics": metrics,
        "data_validation": data_validation,
    }


def _index_alignment(price_index: pd.Index, weight_index: pd.Index, common_idx: pd.Index, where: str) -> dict[str, Any]:
    """What the price/weight index INTERSECTION drops (``data_validation.json`` ``index_alignment``).

    The backtest runs on the bars where both prices and a strategy weight
    row exist. Price bars before the strategy's first row are its warm-up
    (counted, not warned); any OTHER dropped price bar — a strategy that
    writes rows only on its stamp dates shrinks the whole panel to them — is
    warned. Weight rows on dates with no price bar (the weekend rows of a
    7-day panel) are counted only: a weight step stamped there is the weight
    age's to judge (``TIMING:``).
    """
    first_row = weight_index.min() if len(weight_index) else None
    dropped_prices = price_index.difference(common_idx)
    warmup = dropped_prices[dropped_prices < first_row] if first_row is not None else dropped_prices[:0]
    shrink = dropped_prices.difference(warmup)
    dropped_rows = weight_index.difference(common_idx)
    record = {
        "price_bars": int(len(price_index)),
        "weight_rows": int(len(weight_index)),
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
