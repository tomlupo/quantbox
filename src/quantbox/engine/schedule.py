"""Decision vs execution timing: which bar a weight is DECIDED on and which bar it FILLS on (TOM-1429).

The engine seam (:func:`quantbox.engine.simulate`) turns the strategy's
decided weights into the book the engine trades here, the same way for every
engine — the seam owns the schedule, an adapter only executes it (docs/adr/0008):

0. **NaN policy**, once for every engine (:func:`materialise_nan`): a NaN
   cell HOLDS the last decided target; a leading NaN is 0.
1. **Decision bars** are the rebalance schedule on the EXECUTION calendar
   (:func:`quantbox.frequency.rebalancing_dates` over
   :func:`quantbox.instrument_calendar.execution_bars`): "monthly" is the last
   execution bar of the month, never a raw holiday row. Every engine follows
   it. The weight decided on bar ``d`` is the strategy's row
   ``d``: it was computed from data stamped on or before ``d``.
2. **Execution bar** = the decision bar plus ``lag_bars`` bars OF THE
   EXECUTION CALENDAR, not raw index rows — a lag of 1 on a union index would
   otherwise land on a holiday row.
3. **Per instrument**, an order cannot fill at a price the instrument did not
   print: on an execution bar where it is inside its life window but did not
   print, it keeps its previous weight and its order is DEFERRED to its own
   next printed bar (``deferred_trades``). A later decision that reaches the
   instrument first supersedes the deferred one.
4. **Leverage** (``venue.leverage``), on the HELD book after deferral
   (:func:`_apply_leverage`): ``normalize`` scales the cells ordered on a bar
   whose held net exposure would exceed 1 down to net 1, proportionally — a
   deferred cell keeps its weight; ``borrow`` keeps it and the financing legs
   (:mod:`quantbox.financing`) carry the borrowing.
5. **Threshold** (``threshold``, optional): a scheduled rebalance is placed
   only when the held book has DRIFTED more than ``threshold`` (absolute
   weight) from its target on some ordered instrument (:func:`_apply_threshold`).
   The drift is the cost-free price drift of the weights held after the last
   rebalance the trigger placed. An engine charging costs holds a slightly
   different book, so a run with costs can trigger on slightly different bars
   than vectorbt's in-engine threshold did (the declared caveat).
6. **Input staleness**: for each instrument on each decision bar, the bars
   since its last real print — the age of the forward-filled price the signal
   saw. Recorded, never blocking (TOM-1430 gates on it).

``schedule: bars`` is this same function on a DEGENERATE calendar
(:mod:`quantbox.engine.book`): every price bar is an execution bar, every
instrument prints on it (no deferral), and ``venue.leverage`` is ``none`` —
measured, never applied.

The result is the held book (``weights``: what the portfolio holds after each
bar's orders, saved as ``traded_weights``), the per-cell ``orders`` mask the
engines honour (no order where it is False), and the ``rebalance_schedule``
artifact (``decision_date``, ``execution_date``, ``deferred_instruments``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.execution import SameBarOverride
from quantbox.financing import LEVERAGE_MODES, NET_EXPOSURE_TOLERANCE
from quantbox.frequency import _period_end, parse_rebalance_offset, rebalancing_dates
from quantbox.instrument_calendar import InstrumentCalendar

from ._lag import lag_positions

logger = logging.getLogger(__name__)

#: The seam's NaN policy for every engine: a NaN weight cell HOLDS the last decided target.
NAN_POLICY = "hold"
#: ``venue.leverage`` modes the scheduler applies: the config's two, and ``none`` (``schedule: bars``).
SCHEDULE_LEVERAGE = (*LEVERAGE_MODES, "none")


def materialise_nan(weights: pd.DataFrame) -> pd.DataFrame:
    """THE NaN policy (docs/adr/0008): a NaN cell holds the last target; a leading NaN is 0 (idempotent)."""
    return weights.ffill().fillna(0.0)


@dataclass(frozen=True)
class ScheduledBook:
    """The book the engine trades: held weights, the orders mask, the schedule and its report."""

    weights: pd.DataFrame
    orders: pd.DataFrame
    schedule: pd.DataFrame
    report: dict[str, Any]


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0


_BUSINESS_TO_CALENDAR = {
    pd.offsets.BusinessMonthEnd: lambda o: pd.offsets.MonthEnd(),
    pd.offsets.BQuarterEnd: lambda o: pd.offsets.QuarterEnd(startingMonth=o.startingMonth),
    pd.offsets.BYearEnd: lambda o: pd.offsets.YearEnd(month=o.month),
}


def _calendar_period(rebalancing_freq: Any) -> tuple[pd.DateOffset | None, str]:
    """The calendar period a schedule rebalances once in, or ``(None, why not)``.

    Only a single period-end offset (``ME``, ``BME``, ``QE``, ``YE``, ``W-SUN``,
    ...) defines one; a business offset's period is its calendar one (``BME``
    -> the month), so a Sunday month-end is still in its month.
    """
    if not isinstance(rebalancing_freq, (str, pd.DateOffset)):
        return None, f"rebalancing_freq {rebalancing_freq!r} is not a calendar offset: no rebalance period"
    offset = parse_rebalance_offset(rebalancing_freq)
    if not _period_end(offset) or offset.n != 1:
        return None, f"rebalancing_freq {rebalancing_freq!r} is not a single period-end offset: no rebalance period"
    period = _BUSINESS_TO_CALENDAR.get(type(offset), lambda o: o)(offset)
    return period, ""


def _decision_weight_age(
    weight_rows: pd.DataFrame,
    columns: pd.Index,
    index: pd.Index,
    exec_idx: pd.Index,
    decisions: pd.Index,
    dec_dates: pd.Index,
    exe_rows: np.ndarray,
    rebalancing_freq: Any,
) -> dict[str, Any]:
    """Decisions that MISSED a weight step the strategy stamped late in their period (``weight_age``).

    The rule is about where weights are STAMPED. A decision bar ``d`` trades
    the strategy's row on (or forward-filled onto) ``d``. On a CALENDAR
    schedule (one period-end offset), ``d`` is STALE when the strategy writes
    a STEP on a non-execution bar after ``d`` and before the next execution
    bar — weights that differ from the ones ``d`` traded and stay unchanged up
    to that next execution bar (the strategy forward-fills its own stamp: a
    calendar month-end that falls on a weekend, from a wider panel) — whose
    calendar period gets NO decision of its own at or after it: the next
    decision lies in a later period, so that period's weights are never
    traded. Normally the step's period is ``d``'s own; it is a later, empty
    one when the price/weight intersection left that period without a bar. A
    step that opens the NEXT period (a Sunday weekly signal after a Friday
    month-end) is traded by that period's decision: never counted.

    Not measured (``measured: false`` with the reason, never a 0 that reads
    as clean): an ``int`` or explicit-date schedule, buy-and-hold —
    there is no period to say whose step it is. Narrow on purpose: weights
    that keep moving over the non-execution bars (a 7-day vol scaler) are
    daily variation, never counted — which also hides a stamped step with
    drift on top (ADR-0007 1c). Executed decisions only. *weight_rows* is the
    strategy's book on its OWN rows; equal = within 1e-12 everywhere.
    ``held_bars`` of a stale decision = bars from its execution to the next
    decision's execution. Recorded, never refused (TOM-1430 gates).
    """
    dec = pd.DatetimeIndex(dec_dates)
    rule = (
        "calendar schedules only: stale when the strategy stamps a weight step on a non-execution bar after the "
        "decision bar, holds it unchanged to the next execution bar, and the step's calendar period gets no "
        "decision after it"
    )
    period, why = _calendar_period(rebalancing_freq)
    if period is None:
        return {"rule": rule, "measured": False, "reason": why, "decisions": int(len(dec))}
    raw = weight_rows.reindex(columns=columns).sort_index().ffill()
    stamps = pd.DatetimeIndex(raw.index)
    vals = np.nan_to_num(raw.to_numpy(dtype=float), nan=0.0)
    exec_dt = pd.DatetimeIndex(exec_idx)
    all_dec = pd.DatetimeIndex(decisions)
    stale_rows: list[int] = []
    missed: list[str] = []
    for k, d in enumerate(dec):
        nxt = exec_dt.searchsorted(d, side="right")
        if nxt >= len(exec_dt):
            continue
        lo = stamps.searchsorted(d, side="right")
        hi = stamps.searchsorted(exec_dt[nxt], side="left")
        if hi <= lo:
            continue
        traded = vals[lo - 1] if lo > 0 else np.zeros(vals.shape[1])
        window = vals[lo:hi]
        moved = np.flatnonzero((np.abs(window - traded) > 1e-12).any(axis=1))
        if not len(moved):
            continue
        step_at = pd.Timestamp(stamps[lo + moved[0]])
        later = all_dec.searchsorted(d, side="right")
        step_period = period.rollforward(step_at.normalize())
        if later < len(all_dec) and period.rollforward(all_dec[later].normalize()) == step_period:
            continue  # the step's own period has a decision after it, which trades it
        if (np.abs(window[moved[0] :] - window[moved[0]]) > 1e-12).any():
            continue  # keeps moving up to the next execution bar: daily variation, not a stamped step
        stale_rows.append(k)
        missed.append(step_at.isoformat())
    ends = np.append(np.asarray(exe_rows, dtype=int)[1:], len(index))
    held = np.asarray([ends[k] - exe_rows[k] for k in stale_rows], dtype=int)
    return {
        "rule": rule,
        "measured": True,
        "decisions": int(len(dec)),
        "stale_decisions": len(stale_rows),
        "first_stale": [pd.Timestamp(dec[k]).isoformat() for k in stale_rows[:5]],
        "first_missed_stamps": missed[:5],
        "stale_held_bars_max": int(held.max()) if len(held) else 0,
        "stale_held_bars_total": int(held.sum()),
    }


def _apply_leverage(
    target_cells: np.ndarray, orders: np.ndarray, leverage: str, decided_net: np.ndarray, index: pd.Index
) -> dict[str, Any]:
    """Apply ``venue.leverage`` to the HELD book, bar by bar, in place on *target_cells*.

    What the engine holds after a bar's orders is the ordered cells at their
    targets plus every untouched (deferred) cell at its previous weight — not
    the decided row. ``normalize`` therefore scales only the cells ORDERED on
    a bar whose held net exposure would exceed 1, until it is 1 (a deferred
    cell cannot trade); when the deferred cells alone are already at or above
    1, every buy ordered on that bar is set to 0 (no ordered cell rises above
    its previous weight; sells and new shorts still go through). ``borrow`` and
    ``none`` (``schedule: bars``) change nothing and only measure. All count, on the held book.
    """
    tol = NET_EXPOSURE_TOLERANCE
    n_inst = orders.shape[1]
    held_prev = np.zeros(n_inst)
    order_rows = np.flatnonzero(orders.any(axis=1))
    above = 0
    after_deferral = 0
    zeroed: list[int] = []
    scales: list[float] = []
    max_unscaled = 0.0
    max_held = 0.0
    for r in order_rows:
        o = orders[r]
        t = target_cells[r]
        deferred_net = float(held_prev[~o].sum())
        ordered_net = float(t[o].sum())
        net = deferred_net + ordered_net
        max_unscaled = max(max_unscaled, net)
        if net > 1.0 + tol:
            above += 1
            if leverage == "normalize":
                if (~o).any():
                    after_deferral += 1
                if deferred_net < 1.0 - tol and ordered_net > 0:
                    scale = (1.0 - deferred_net) / ordered_net
                    t[o] = t[o] * scale
                    scales.append(scale)
                else:
                    # No room: no ordered cell may RISE above its weight (a buy, a short
                    # cover included). The held net then cannot exceed the last bar's.
                    t[o] = np.minimum(t[o], held_prev[o])
                    zeroed.append(int(r))
        held_prev[o] = t[o]
        max_held = max(max_held, float(held_prev.sum()))
    scaled = len(scales) + len(zeroed) if leverage == "normalize" else 0
    sc = np.asarray(scales)
    return {
        "mode": leverage,
        "rebalances": int(len(order_rows)),
        "rebalances_above_net_1": int(above),
        "max_net_exposure_decided": float(decided_net.max()) if len(decided_net) else 0.0,
        "max_net_exposure_unscaled": float(max_unscaled),
        "max_net_exposure_held": float(max_held),
        "scaled_rebalances": int(scaled),
        "scaled_after_deferral": int(after_deferral) if leverage == "normalize" else 0,
        "buys_zeroed_rebalances": len(zeroed),
        "buys_zeroed_dates": [pd.Timestamp(index[r]).isoformat() for r in zeroed],
        "scale_mean": float(sc.mean()) if len(sc) else 1.0,
        "scale_min": float(sc.min()) if len(sc) else 1.0,
        "scale_max": float(sc.max()) if len(sc) else 1.0,
    }


def _apply_threshold(
    target_cells: np.ndarray, orders: np.ndarray, prices: np.ndarray, threshold: float, index: pd.Index
) -> dict[str, Any]:
    """Drop the scheduled rebalances whose held book has not drifted past *threshold* (in place on *orders*).

    The held book is tracked cost-free: after a placed rebalance, each
    ordered cell holds its target and each untouched cell its drifted weight;
    between rebalances every weight drifts with its price, against a cash
    remainder of ``1 - sum(weights)``. On a scheduled bar the rebalance is
    placed when ``max |drifted - target|`` over the ORDERED cells exceeds
    *threshold* (a cell with no order keeps its weight, so it cannot trigger),
    and then trades every ordered cell to its target — vectorbt's band rule.
    A bar that does not trigger has all its orders removed. It runs after
    ``venue.leverage``, on the targets the engine would trade.

    The caveat (docs/adr/0008): an engine that charges costs holds slightly
    less than this cost-free book, so its in-engine drift differs by roughly
    the costs paid, and a bar near the band edge can fall the other way. Both
    engines trade on the bars this function keeps.
    """
    n_inst = orders.shape[1]
    held = np.zeros(n_inst)
    last: int | None = None
    order_rows = np.flatnonzero(orders.any(axis=1))
    skipped: list[int] = []
    for r in order_rows:
        o = orders[r]
        if last is None:
            drifted = held.copy()
        else:
            growth = prices[r] / prices[last]
            value = held * np.where(np.isfinite(growth), growth, 1.0)
            total = (1.0 - held.sum()) + value.sum()
            drifted = value / total if total != 0 else value
        if np.abs(drifted[o] - target_cells[r, o]).max() > threshold:
            held = drifted
            held[o] = target_cells[r, o]
            last = int(r)
        else:
            orders[r] = False
            skipped.append(int(r))
    return {
        "threshold": float(threshold),
        "rule": "cost-free price drift of the held weights; a scheduled rebalance is placed when an ordered "
        "instrument drifted more than the threshold from its target",
        "scheduled_rebalances": int(len(order_rows)),
        "placed_rebalances": int(len(order_rows) - len(skipped)),
        "skipped_rebalances": len(skipped),
        "first_skipped": [pd.Timestamp(index[r]).isoformat() for r in skipped[:5]],
    }


def schedule_book(
    decided: pd.DataFrame,
    cal: InstrumentCalendar,
    exec_bars: pd.Series,
    rebalancing_freq: Any,
    lag_bars: int,
    *,
    leverage: str,
    threshold: float | None = None,
    weight_rows: pd.DataFrame | None = None,
    same_bar: SameBarOverride | None = None,
) -> ScheduledBook:
    """Decided weights -> the traded book on *cal*'s bars (see the module docstring).

    *decided* is the strategy's book after overlays and risk transforms, NOT
    lagged; it is reindexed to *cal*'s bars and instruments, and its NaN cells
    read by :func:`materialise_nan`. Every engine gets this one schedule.
    ``lag_bars`` 0 runs only with the *same_bar* override the resolver
    granted; the lag itself is :func:`quantbox.engine._lag.lag_positions`.
    *threshold* (absolute weight, optional) drops a scheduled rebalance whose
    held book has not drifted that far (:func:`_apply_threshold`).
    *weight_rows* is the strategy's book on ITS OWN rows (weekend rows of a
    wider panel included), for the decision weight age; default *decided*.
    """
    if leverage not in SCHEDULE_LEVERAGE:
        raise ValueError(f"venue.leverage must be one of {list(LEVERAGE_MODES)}, got {leverage!r}")
    if threshold is not None and (isinstance(threshold, bool) or not float(threshold) >= 0):
        raise ValueError(f"threshold must be a number >= 0 (absolute weight drift), got {threshold!r}")
    index, columns = cal.observed.index, cal.observed.columns
    raw_decided = decided if weight_rows is None else weight_rows  # the strategy's own rows
    decided = materialise_nan(decided.reindex(index=index, columns=columns))
    observed = cal.observed.to_numpy()
    inside = cal.inside.to_numpy()
    n_bars, n_inst = observed.shape

    exec_idx = index[exec_bars.reindex(index, fill_value=False).to_numpy()]
    decisions = rebalancing_dates(exec_idx, rebalancing_freq)
    k = exec_idx.get_indexer(decisions)
    e = lag_positions(k, lag_bars, same_bar)  # the execution bar, counted in execution-calendar bars
    executed = e < len(exec_idx)
    dec_dates = decisions[executed]
    exe_dates = exec_idx[e[executed]]
    dec_rows = index.get_indexer(dec_dates)
    exe_rows = index.get_indexer(exe_dates)

    # Targets: the decided row, 0 outside the instrument's window at the execution bar.
    targets = decided.to_numpy(dtype=float)[dec_rows] if len(dec_rows) else np.zeros((0, n_inst))
    inside_exe = inside[exe_rows] if len(exe_rows) else np.zeros((0, n_inst), dtype=bool)
    outside_targeted = (targets != 0) & ~inside_exe
    max_outside = np.where(outside_targeted, np.abs(targets), 0.0).max(axis=0) if len(targets) else np.zeros(n_inst)
    targets = np.where(inside_exe, targets, 0.0)

    decided_net = targets.sum(axis=1) if len(targets) else np.zeros(0)

    # Orders: every instrument on every execution bar, except the unprinted ones, which are deferred.
    target_cells = np.full((n_bars, n_inst), np.nan)
    orders = np.zeros((n_bars, n_inst), dtype=bool)
    target_cells[exe_rows] = targets
    orders[exe_rows] = True
    scheduled = set(int(r) for r in exe_rows)
    unprinted = inside & ~observed
    printed_at = [np.flatnonzero(inside[:, j] & observed[:, j]) for j in range(n_inst)]
    deferred = np.zeros(n_inst, dtype=int)
    deferred_names: list[str] = []
    for i, r in enumerate(exe_rows):
        cols = np.flatnonzero(unprinted[r])
        deferred_names.append(";".join(sorted(str(columns[j]) for j in cols)))
        for j in cols:
            orders[r, j] = False
            deferred[j] += 1
            # r is inside the window and unprinted, so a later print exists: the window ends on one.
            p = int(printed_at[j][np.searchsorted(printed_at[j], r)])
            if p in scheduled:
                continue  # a later decision executes there and supersedes this one
            orders[p, j] = True
            target_cells[p, j] = targets[i, j]  # later deferrals to the same bar overwrite: the newest wins

    leverage_report = _apply_leverage(target_cells, orders, leverage, decided_net, index)
    threshold_report = (
        _apply_threshold(target_cells, orders, cal.prices.to_numpy(dtype=float), float(threshold), index)
        if threshold is not None
        else None
    )
    held = pd.DataFrame(np.where(orders, target_cells, np.nan), index=index, columns=columns).ffill().fillna(0.0)
    orders_df = pd.DataFrame(orders, index=index, columns=columns)

    # Staleness of the inputs at each decision: bars since the instrument's last real print.
    all_dec_rows = index.get_indexer(decisions)
    last_print = pd.DataFrame(
        np.where(observed, np.arange(n_bars)[:, None], np.nan), index=index, columns=columns
    ).ffill()
    age = np.arange(n_bars)[:, None] - last_print.to_numpy()
    dec_age = age[all_dec_rows] if len(all_dec_rows) else np.zeros((0, n_inst))
    dec_inside = inside[all_dec_rows] if len(all_dec_rows) else np.zeros((0, n_inst), dtype=bool)
    ages = dec_age[dec_inside]
    stale = (dec_age > 0) & dec_inside

    weight_age = _decision_weight_age(
        raw_decided, columns, index, exec_idx, decisions, dec_dates, exe_rows, rebalancing_freq
    )

    schedule = pd.DataFrame(
        {
            "decision_date": pd.DatetimeIndex(dec_dates),
            "execution_date": pd.DatetimeIndex(exe_dates),
            "deferred_instruments": deferred_names,
        }
    )
    per_instrument = {
        str(c): {
            "deferred_trades": int(deferred[j]),
            "targeted_outside_window_bars": int(outside_targeted[:, j].sum()) if len(targets) else 0,
            "max_abs_weight_outside_window": float(max_outside[j]),
            "stale_decisions": int(stale[:, j].sum()),
            "max_staleness_bars": int(np.nanmax(np.where(dec_inside[:, j], dec_age[:, j], 0))) if len(dec_age) else 0,
        }
        for j, c in enumerate(columns)
    }
    n_out = int(outside_targeted.sum())
    report = {
        "timing": {
            "decision_calendar": "execution",
            "lag_bars": int(lag_bars),
            "lag_counted_in": "execution-calendar bars",
            "decisions": int(len(decisions)),
            "executed_decisions": int(len(dec_dates)),
            "decisions_past_last_bar": int((~executed).sum()),
            "deferred_trades": int(deferred.sum()),
            "targeted_outside_window_bars": n_out,
            "instruments_targeted_outside_window": sorted(
                c for c, v in per_instrument.items() if v["targeted_outside_window_bars"]
            ),
        },
        "staleness": {
            "instrument_decisions": int(len(ages)),
            "stale_decisions": int(stale.sum()),
            "max_bars": int(ages.max()) if len(ages) else 0,
            "p95_bars": _percentile(ages, 95),
        },
        "weight_age": weight_age,
        "leverage": leverage_report,
        "instruments": per_instrument,
    }
    if threshold_report is not None:  # absent without a threshold: the run's files stay as they were
        report["threshold"] = threshold_report
    if weight_age.get("stale_decisions"):
        logger.warning(
            "TIMING: %d of %d decision(s) MISSED their period's weights, stamped on a non-execution bar after the "
            "decision bar (a weekend calendar month-end, say): the previous weights were held for up to %d "
            "bar(s) instead. Stamp weights on the decision bars (the execution calendar). Decisions: %s; missed "
            "stamps: %s",
            weight_age["stale_decisions"],
            weight_age["decisions"],
            weight_age["stale_held_bars_max"],
            weight_age["first_stale"],
            weight_age["first_missed_stamps"],
        )
    if n_out:
        logger.warning(
            "CALENDAR: the strategy targeted %d instrument-rebalance(s) OUTSIDE the instrument's life window "
            "(before its first or after its last price) — forced to 0, counted in data_validation.json: %s",
            n_out,
            report["timing"]["instruments_targeted_outside_window"],
        )
    if leverage == "normalize" and leverage_report["scaled_rebalances"]:
        logger.warning(
            "LEVERAGE: %d of %d rebalance(s) would have HELD net exposure above 1 (max %.4f) and were SCALED to "
            "net 1 (venue.leverage: normalize, the default; scale mean %.4f, min %.4f; %d after a deferral). "
            "Declare venue.leverage: borrow with venue.financing to hold the levered book.",
            leverage_report["scaled_rebalances"],
            leverage_report["rebalances"],
            leverage_report["max_net_exposure_unscaled"],
            leverage_report["scale_mean"],
            leverage_report["scale_min"],
            leverage_report["scaled_after_deferral"],
        )
    if leverage_report["buys_zeroed_rebalances"]:
        logger.warning(
            "LEVERAGE: on %d rebalance(s) the DEFERRED positions alone held net exposure at or above 1, so every "
            "buy ordered on that bar was set to 0 (venue.leverage: normalize): %s",
            leverage_report["buys_zeroed_rebalances"],
            leverage_report["buys_zeroed_dates"],
        )
    if report["staleness"]["stale_decisions"]:
        logger.info(
            "Staleness: %d of %d instrument-decision(s) saw a forward-filled input (max %d bar(s), p95 %.1f)",
            report["staleness"]["stale_decisions"],
            report["staleness"]["instrument_decisions"],
            report["staleness"]["max_bars"],
            report["staleness"]["p95_bars"],
        )
    return ScheduledBook(weights=held, orders=orders_df, schedule=schedule, report=report)
