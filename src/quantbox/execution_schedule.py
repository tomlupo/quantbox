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
4. **Leverage** (``venue.leverage``): ``normalize`` scales a decision whose
   net exposure is above 1 down to net 1, proportionally; ``borrow`` keeps it
   and the financing legs (:mod:`quantbox.financing`) carry the borrowing.
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


def schedule_book(
    decided: pd.DataFrame,
    cal: InstrumentCalendar,
    exec_bars: pd.Series,
    rebalancing_freq: Any,
    lag_bars: int,
    *,
    engine: str,
    leverage: str,
) -> ScheduledBook:
    """Decided weights -> the traded book on *cal*'s bars (see the module docstring).

    *decided* is the strategy's book after overlays and risk transforms, NOT
    lagged; it is reindexed to *cal*'s bars and instruments. ``lag_bars`` 0
    is reachable only under the same-bar override the resolver granted.
    """
    if leverage not in LEVERAGE_MODES:
        raise ValueError(f"venue.leverage must be one of {list(LEVERAGE_MODES)}, got {leverage!r}")
    index, columns = cal.observed.index, cal.observed.columns
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

    # Leverage: normalize scales a decision with net > 1 to net 1, proportionally.
    net = targets.sum(axis=1) if len(targets) else np.zeros(0)
    over = net > 1.0 + NET_EXPOSURE_TOLERANCE
    scales = np.ones(len(net))
    if leverage == "normalize":
        scales[over] = 1.0 / net[over]
        targets = targets * scales[:, None]
    leverage_report = {
        "mode": leverage,
        "rebalances": int(len(net)),
        "rebalances_above_net_1": int(over.sum()),
        "max_net_exposure_decided": float(net.max()) if len(net) else 0.0,
        "scaled_rebalances": int(over.sum()) if leverage == "normalize" else 0,
        "scale_mean": float(scales[over].mean()) if leverage == "normalize" and over.any() else 1.0,
        "scale_min": float(scales[over].min()) if leverage == "normalize" and over.any() else 1.0,
        "scale_max": float(scales[over].max()) if leverage == "normalize" and over.any() else 1.0,
    }

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
        "leverage": leverage_report,
        "instruments": per_instrument,
    }
    if n_out:
        logger.warning(
            "CALENDAR: the strategy targeted %d instrument-rebalance(s) OUTSIDE the instrument's life window "
            "(before its first or after its last price) — forced to 0, counted in data_validation.json: %s",
            n_out,
            report["timing"]["instruments_targeted_outside_window"],
        )
    if leverage == "normalize" and over.any():
        logger.warning(
            "LEVERAGE: %d of %d rebalance(s) decided net exposure above 1 (max %.4f) and were SCALED to net 1 "
            "(venue.leverage: normalize, the default; scale mean %.4f, min %.4f). Declare venue.leverage: borrow "
            "with venue.financing to hold the levered book.",
            int(over.sum()),
            len(net),
            leverage_report["max_net_exposure_decided"],
            leverage_report["scale_mean"],
            leverage_report["scale_min"],
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
