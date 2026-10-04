"""Per-instrument trading calendar — what a missing price MEANS to a backtest (TOM-1429).

A wide prices panel mixes instruments with different calendars: a holiday in
one market is a bar in another (1 January prints for some indices and not for
others), and an index may be monthly before it turns daily. A NaN in the panel
is therefore one of two different facts, and the engine seam must not confuse
them:

- **inside the instrument's life window** (its first valid price through its
  last valid price) it is a bar the instrument did not print — a holiday, a
  gap. The position EXISTS across it. The price is forward-filled explicitly,
  the target weight is kept, and the bar is counted (``ffilled_bars``).
- **outside the life window** (before listing, after delisting) the
  instrument cannot be held. The weight is forced to 0, and prices are filled
  only so the engine has a number to mark a flat book at — never a tradable
  position. A weight the strategy targeted there is counted and logged
  (``targeted_outside_window_bars``), never dropped without a record.

Until v0.9.0 the pipeline zeroed the weight on EVERY missing price
(``weights.where(prices.notna(), 0)``), so a target set on a holiday rebalance
bar was flat for the whole holding period (478 of 5920 instrument-months in the
TSMOM replication, -0.10 Sharpe), and it silently dropped instruments with
under 50% price coverage. :func:`apply_instrument_calendar` replaces both.

It is engine-agnostic: :class:`~quantbox.plugins.pipeline.backtest_pipeline.BacktestPipeline`
calls it once, after the execution lag and before the engine branch, so every
engine receives the same book. The report it returns is the ``calendar``
section of ``data_validation.json`` (:data:`DATA_VALIDATION_SCHEMA`), the shape
the data-validation step (TOM-1430) consumes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: ``data_validation.json`` — one section per check; this module owns ``calendar``.
DATA_VALIDATION_SCHEMA = "quantbox/data-validation@1"


@dataclass(frozen=True)
class CalendarAlignment:
    """The engine-ready book: prices without NaN, weights zero outside each life window."""

    prices: pd.DataFrame
    weights: pd.DataFrame
    report: dict[str, Any]


def _stamp(ts: Any) -> str | None:
    return None if ts is None or pd.isna(ts) else pd.Timestamp(ts).isoformat()


def life_windows(prices: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame: True where the bar lies between the column's first and last valid price (inclusive)."""
    observed = prices.notna()
    return observed.cummax() & observed.iloc[::-1].cummax().iloc[::-1]


def apply_instrument_calendar(prices: pd.DataFrame, weights: pd.DataFrame) -> CalendarAlignment:
    """Apply each instrument's life window to an aligned ``prices`` / ``weights`` pair.

    *weights* are the TRADED weights (already lagged) on the same index and
    columns as *prices*. Returns the prices and weights the engine receives, and
    the calendar report. Rows where no instrument has a price are dropped (there
    is nothing to mark), and so are columns that never print — both counted. A
    NaN weight inside a window is left NaN: the engine's own NaN policy
    (:func:`quantbox.execution.materialise_nan_policy`) resolves it afterwards.
    """
    if not prices.index.equals(weights.index) or list(prices.columns) != list(weights.columns):
        raise ValueError("apply_instrument_calendar: prices and weights must share index and columns")

    observed = prices.notna()
    targeted = weights.notna() & weights.ne(0.0)

    never_priced = [c for c in prices.columns if not observed[c].any()]
    never_priced_targeted = {str(c): int(targeted[c].sum()) for c in never_priced}
    keep_cols = [c for c in prices.columns if c not in never_priced]

    empty_rows = ~observed[keep_cols].any(axis=1) if keep_cols else pd.Series(True, index=prices.index)
    rows = ~empty_rows
    # Windows and overridden targets on the FULL index, so a target on a dropped empty row
    # outside the window (a single-asset book after delisting) is still counted.
    inside_full = life_windows(prices[keep_cols])
    outside_targeted = targeted[keep_cols] & ~inside_full
    px = prices.loc[rows, keep_cols]
    w = weights.loc[rows, keep_cols]
    obs = observed.loc[rows, keep_cols]
    inside = inside_full.loc[rows]
    ffilled = inside & ~obs

    out_weights = w.where(inside, 0.0)
    # ffill: holidays inside the window, and the last price after delisting (weight 0 there).
    # bfill reaches only bars BEFORE listing, where the weight is 0: a mark, never a position.
    out_prices = px.ffill().bfill()

    instruments: dict[str, dict[str, Any]] = {}
    for c in keep_cols:
        col_obs = obs[c]
        n_out = int(outside_targeted[c].sum())
        instruments[str(c)] = {
            "first_valid": _stamp(col_obs.idxmax()) if col_obs.any() else None,
            "last_valid": _stamp(col_obs[::-1].idxmax()) if col_obs.any() else None,
            "window_bars": int(inside[c].sum()),
            "observed_bars": int(col_obs.sum()),
            "ffilled_bars": int(ffilled[c].sum()),
            "targeted_outside_window_bars": n_out,
            "max_abs_weight_outside_window": float(weights[c].where(outside_targeted[c]).abs().max()) if n_out else 0.0,
        }
    for c, n in never_priced_targeted.items():
        instruments[c] = {
            "first_valid": None,
            "last_valid": None,
            "window_bars": 0,
            "observed_bars": 0,
            "ffilled_bars": 0,
            "targeted_outside_window_bars": n,
            "max_abs_weight_outside_window": float(weights[c].where(targeted[c]).abs().max()) if n else 0.0,
        }

    targeted_outside = sorted(k for k, v in instruments.items() if v["targeted_outside_window_bars"])
    report = {
        "policy": {
            "inside_window": "ffill price, keep target weight",
            "outside_window": "weight forced to 0; price filled for marking only",
        },
        "n_bars": int(rows.sum()),
        "n_instruments": len(instruments),
        "rows_without_any_price": int(empty_rows.sum()),
        "totals": {
            "ffilled_bars": int(sum(v["ffilled_bars"] for v in instruments.values())),
            "targeted_outside_window_bars": int(sum(v["targeted_outside_window_bars"] for v in instruments.values())),
            "instruments_targeted_outside_window": targeted_outside,
            "instruments_without_prices": sorted(str(c) for c in never_priced),
        },
        "instruments": instruments,
    }
    n_out = report["totals"]["targeted_outside_window_bars"]
    if n_out:
        logger.warning(
            "CALENDAR: the strategy targeted %d instrument-bar(s) OUTSIDE the instrument's life window "
            "(before its first or after its last price) — forced to 0, counted in data_validation.json: %s",
            n_out,
            targeted_outside or report["totals"]["instruments_without_prices"],
        )
    logger.info(
        "Calendar: %d holiday/gap bar(s) forward-filled inside life windows across %d instrument(s)",
        report["totals"]["ffilled_bars"],
        len(instruments),
    )
    return CalendarAlignment(prices=out_prices, weights=out_weights, report=report)


def calendar_summary(report: dict[str, Any]) -> dict[str, Any]:
    """The run-manifest-sized part of a calendar report: totals, no per-instrument rows."""
    return {
        "n_bars": report["n_bars"],
        "n_instruments": report["n_instruments"],
        "rows_without_any_price": report["rows_without_any_price"],
        **report["totals"],
    }
