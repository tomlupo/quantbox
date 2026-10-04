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


def _count_between(stamps: pd.DatetimeIndex, lo: pd.Timestamp, hi: pd.Timestamp) -> int:
    """Stamps strictly inside (lo, hi)."""
    return int(stamps.searchsorted(hi, side="left") - stamps.searchsorted(lo, side="right"))


def _decision_weight_age(
    weight_rows: pd.DataFrame,
    columns: pd.Index,
    index: pd.Index,
    exec_idx: pd.Index,
    decisions: pd.Index,
    dec_dates: pd.Index,
    exe_rows: np.ndarray,
    price_index: pd.Index | None = None,
) -> dict[str, Any]:
    """Decisions that MISSED a weight step the strategy stamped off the execution calendar (``weight_age``).

    The rule is about where weights are STAMPED. A decision bar ``d`` trades
    the strategy's row on (or forward-filled onto) ``d``. A decision is STALE
    when, strictly between ``d`` and the next execution bar, the strategy
    writes a STEP on a non-execution bar — weights that differ from the ones
    ``d`` traded and then stay unchanged up to that next execution bar (the
    strategy forward-fills its own stamp: a calendar month-end that falls on a
    weekend, from a wider panel) — AND the next decision comes after that
    execution bar, so the missed step is not simply traded there (it always is
    on a daily schedule or rsims) — or the price/weight intersection dropped
    price bars from that gap (*price_index*), so the "next execution bar" is
    a later stamp, not the next bar. Schedule-agnostic. Executed decisions
    only.

    Narrow on purpose (review round 2): weights that keep moving over the
    non-execution bars (a 7-day vol scaler, a daily signal on a crypto+equity
    panel) are daily variation, not a mis-stamp, and are never counted — which
    also hides a stamped step with daily drift on top (ADR-0007 1c, known
    false negative). *weight_rows* is the strategy's book on its OWN rows
    (forward-filled on them); equal = within 1e-12 in every instrument.
    ``held_bars`` of a stale decision = bars from its execution to the next
    decision's execution. Recorded, never refused (TOM-1430 gates).
    """
    raw = weight_rows.reindex(columns=columns).sort_index().ffill()
    stamps = pd.DatetimeIndex(raw.index)
    vals = np.nan_to_num(raw.to_numpy(dtype=float), nan=0.0)
    exec_dt = pd.DatetimeIndex(exec_idx)
    all_dec = pd.DatetimeIndex(decisions)
    bars = pd.DatetimeIndex(index)
    prices = bars if price_index is None else pd.DatetimeIndex(price_index)
    dec = pd.DatetimeIndex(dec_dates)
    stale_rows: list[int] = []
    missed: list[str] = []
    for k, d in enumerate(dec):
        nxt = exec_dt.searchsorted(d, side="right")
        if nxt >= len(exec_dt):
            continue
        nxt_bar = exec_dt[nxt]
        later = all_dec.searchsorted(d, side="right")
        shrunk = _count_between(prices, d, nxt_bar) > _count_between(bars, d, nxt_bar)
        if later < len(all_dec) and all_dec[later] <= nxt_bar and not shrunk:
            continue  # the next decision trades whatever was stamped in the gap
        lo = stamps.searchsorted(d, side="right")
        hi = stamps.searchsorted(nxt_bar, side="left")
        if hi <= lo:
            continue
        traded = vals[lo - 1] if lo > 0 else np.zeros(vals.shape[1])
        window = vals[lo:hi]
        moved = np.flatnonzero((np.abs(window - traded) > 1e-12).any(axis=1))
        if not len(moved):
            continue
        step = window[moved[0]]
        if (np.abs(window[moved[0] :] - step) > 1e-12).any():
            continue  # keeps moving up to the next execution bar: daily variation, not a stamped step
        stale_rows.append(k)
        missed.append(pd.Timestamp(stamps[lo + moved[0]]).isoformat())
    ends = np.append(np.asarray(exe_rows, dtype=int)[1:], len(index))
    held = np.asarray([ends[k] - exe_rows[k] for k in stale_rows], dtype=int)
    return {
        "rule": "stale when the strategy stamps a weight step on a non-execution bar between the decision bar and "
        "the next execution bar, holds it unchanged to that bar, and the next decision comes later",
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
    price_index: pd.Index | None = None,
) -> ScheduledBook:
    """Decided weights -> the traded book on *cal*'s bars (see the module docstring).

    *decided* is the strategy's book after overlays and risk transforms, NOT
    lagged; it is reindexed to *cal*'s bars and instruments. ``lag_bars`` 0
    is reachable only under the same-bar override the resolver granted.
    *weight_rows* is the strategy's book on ITS OWN rows (weekend rows of a
    wider panel included), for the decision weight age; default *decided*.
    *price_index* is the price panel's index BEFORE the price/weight
    intersection, so the weight age can tell a real calendar gap from one the
    intersection made; default *cal*'s bars.
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

    weight_age = _decision_weight_age(
        raw_decided, columns, index, exec_idx, decisions, dec_dates, exe_rows, price_index
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
    if weight_age["stale_decisions"]:
        logger.warning(
            "TIMING: %d of %d decision(s) MISSED weights the strategy stamped on a non-execution bar before the "
            "next execution bar (a weekend calendar month-end, say): the previous weights were held for up to %d "
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
