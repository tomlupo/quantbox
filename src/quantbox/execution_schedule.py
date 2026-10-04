"""Decision vs execution timing: which bar a weight is DECIDED on and which bar it FILLS on (TOM-1429).

The backtest pipeline turns the strategy's decided weights into the book the
engine trades here, the same way for every engine:

1. **Decision bars** are the rebalance schedule on the EXECUTION calendar
   (:func:`quantbox.frequency.rebalancing_dates` over
   :func:`quantbox.instrument_calendar.execution_bars`): "monthly" is the last
   execution bar of the month, never a raw holiday row. rsims decides on
   every execution bar. The weight decided on bar ``d`` is the strategy's row
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
5. **Input staleness**: for each instrument on each decision bar, the bars
   since its last real print — the age of the forward-filled price the signal
   saw. Recorded, never blocking (TOM-1430 gates on it).

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

from quantbox.execution import materialise_nan_policy
from quantbox.financing import LEVERAGE_MODES, NET_EXPOSURE_TOLERANCE
from quantbox.frequency import rebalancing_dates
from quantbox.instrument_calendar import InstrumentCalendar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduledBook:
    """The book the engine trades: held weights, the orders mask, the schedule and its report."""

    weights: pd.DataFrame
    orders: pd.DataFrame
    schedule: pd.DataFrame
    report: dict[str, Any]


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0


def _decision_weight_age(
    decided: pd.DataFrame, columns: pd.Index, index: pd.Index, decisions: pd.Index, rebalancing_freq: Any, engine: str
) -> dict[str, Any]:
    """Age of the strategy weights each decision trades (``data_validation.json`` ``weight_age``).

    The weights a decision bar ``d`` trades are the strategy's newest row on or
    before ``d``, forward-filled onto it (by the strategy or by the engine
    seam). They are measured by when they last CHANGED, on the strategy's own
    rows — a strategy that forward-fills its own output repeats old values, so
    "a row exists" says nothing. A decision is STALE when the weights it trades
    last changed before the start of its rebalance period AND the strategy
    changes them later in the same period: that period's weights were stamped
    after its decision bar (a calendar month-end that falls on a weekend, say),
    and the previous period's were traded. A strategy that holds its weights
    constant, or writes fresh ones on every bar, is never stale.

    The period: for a single period-end offset (``ME``, ``W-FRI``, ...) the
    calendar period that ends on or after ``d``; otherwise (``int`` / explicit
    dates / rsims, which decides on every execution bar) the span between the
    previous and the next decision. Age, of the stale decisions: bars of
    *index* in ``(change, d]`` and periods = decisions in ``(change, d]``.
    Recorded, never refused (TOM-1430 gates).
    """
    from quantbox.frequency import _period_end, parse_rebalance_offset

    dec = pd.DatetimeIndex(decisions)
    raw = decided.reindex(columns=columns).sort_index().ffill()
    valid = raw.notna().any(axis=1).to_numpy()
    vals = raw.fillna(0.0).to_numpy(dtype=float)
    moved = np.zeros(len(raw), dtype=bool)
    if len(raw):
        moved[0] = valid[0]
        moved[1:] = valid[1:] & ((np.abs(np.diff(vals, axis=0)) > 1e-12).any(axis=1) | ~valid[:-1])
    changes = pd.DatetimeIndex(raw.index[moved])

    n = len(dec)
    prev = np.concatenate([[pd.Timestamp.min.value], dec.asi8[:-1]]) if n else np.zeros(0, dtype=np.int64)
    nxt = np.concatenate([dec.asi8[1:], [pd.Timestamp.max.value]]) if n else np.zeros(0, dtype=np.int64)
    start, end, end_inclusive = prev, nxt, False
    offset = None
    if engine != "rsims" and isinstance(rebalancing_freq, (str, pd.DateOffset)):
        offset = parse_rebalance_offset(rebalancing_freq)
    if offset is not None and _period_end(offset) and getattr(offset, "n", 1) == 1 and n:
        naive = dec.tz_localize(None) if dec.tz is not None else dec
        ends = pd.DatetimeIndex([offset.rollforward(d.normalize()) for d in naive])
        starts = ends - offset
        if dec.tz is not None:
            ends, starts = ends.tz_localize(dec.tz), starts.tz_localize(dec.tz)
        start, end, end_inclusive = starts.asi8, (ends + pd.Timedelta(days=1)).asi8 - 1, True

    ch = changes.asi8
    pos = np.searchsorted(ch, dec.asi8, side="right") - 1
    has = pos >= 0
    last = np.where(has, ch[np.clip(pos, 0, None)] if len(ch) else 0, 0)
    old = has & (last <= start)
    later = np.searchsorted(ch, end, side="right" if end_inclusive else "left") - np.searchsorted(
        ch, dec.asi8, side="right"
    )
    stale = old & (later > 0)

    s_dec = dec[stale]
    s_last = pd.DatetimeIndex(last[stale]).tz_localize(dec.tz) if dec.tz is not None else pd.DatetimeIndex(last[stale])
    bars = index.searchsorted(s_dec, side="right") - index.searchsorted(s_last, side="right")
    periods = dec.searchsorted(s_dec, side="right") - dec.searchsorted(s_last, side="right")
    return {
        "rule": "stale when the traded weights last changed before the start of the decision's rebalance period "
        "and the strategy changes them later in that period",
        "decisions": int(n),
        "decisions_with_weights": int(has.sum()),
        "stale_decisions": int(stale.sum()),
        "max_bars": int(bars.max()) if len(bars) else 0,
        "p95_bars": _percentile(np.asarray(bars, dtype=float), 95),
        "max_periods": int(periods.max()) if len(periods) else 0,
        "first_stale": [pd.Timestamp(d).isoformat() for d in s_dec[:5]],
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
    its previous weight; sells and new shorts still go through). ``borrow`` changes nothing
    and only measures. Both count, on the held book.
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


def schedule_book(
    decided: pd.DataFrame,
    cal: InstrumentCalendar,
    exec_bars: pd.Series,
    rebalancing_freq: Any,
    lag_bars: int,
    *,
    engine: str,
    leverage: str,
    weight_rows: pd.DataFrame | None = None,
) -> ScheduledBook:
    """Decided weights -> the traded book on *cal*'s bars (see the module docstring).

    *decided* is the strategy's book after overlays and risk transforms, NOT
    lagged; it is reindexed to *cal*'s bars and instruments. ``lag_bars`` 0
    is reachable only under the same-bar override the resolver granted.
    *weight_rows* is the strategy's book on ITS OWN rows (weekend rows of a
    wider panel included), for the decision weight age; default *decided*.
    """
    if leverage not in LEVERAGE_MODES:
        raise ValueError(f"venue.leverage must be one of {list(LEVERAGE_MODES)}, got {leverage!r}")
    index, columns = cal.observed.index, cal.observed.columns
    raw_decided = decided if weight_rows is None else weight_rows  # the strategy's own rows
    decided = materialise_nan_policy(decided.reindex(index=index, columns=columns), engine)
    observed = cal.observed.to_numpy()
    inside = cal.inside.to_numpy()
    n_bars, n_inst = observed.shape

    exec_idx = index[exec_bars.reindex(index, fill_value=False).to_numpy()]
    decisions = exec_idx if engine == "rsims" else rebalancing_dates(exec_idx, rebalancing_freq)
    k = exec_idx.get_indexer(decisions)
    e = k + int(lag_bars)
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

    weight_age = _decision_weight_age(raw_decided, columns, index, decisions, rebalancing_freq, engine)

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
    if weight_age["stale_decisions"]:
        logger.warning(
            "TIMING: %d of %d decision(s) traded a STALE strategy weight row — the weights on the decision bar "
            "date from before its rebalance period and the strategy changed them only AFTER the decision bar "
            "(max %d bar(s), %d period(s) old): the previous period's weights were traded. Stamp weights on "
            "the decision bars (the execution calendar). First: %s",
            weight_age["stale_decisions"],
            weight_age["decisions"],
            weight_age["max_bars"],
            weight_age["max_periods"],
            weight_age["first_stale"],
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
