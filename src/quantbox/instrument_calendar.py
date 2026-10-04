"""Per-instrument and execution calendars — what a missing price MEANS to a backtest (TOM-1429).

A wide prices panel mixes instruments with different calendars: a holiday in
one market is a bar in another (1 January prints for some indices and not for
others), and an index may be monthly before it turns daily. A NaN in the panel
is therefore one of two different facts, and the engine seam must not confuse
them:

- **inside the instrument's life window** (its first valid price through its
  last valid price) it is a bar the instrument did not print — a holiday, a
  gap. The position EXISTS across it. The price is forward-filled explicitly,
  to MARK the position and never to fill an order, and the bar is counted
  (``ffilled_bars``). An order for the instrument waits for its next printed
  bar (:mod:`quantbox.engine.schedule`, ``deferred_trades``): filling at
  the stale close would let a decision taken at the 29 Dec close fill at that
  same close on 1 January and book the 2 January move (same-bar look-ahead,
  docs/adr/0005).
- **outside the life window** (before listing, after delisting) the
  instrument cannot be held. Its target is forced to 0, and prices are filled
  only so the engine has a number to mark a flat book at. A weight the
  strategy targeted there is counted and logged, never dropped without a
  record. A trailing feed gap cannot be told from a delisting: both end the
  window at the last print.

On top of the per-instrument calendars sits ONE **execution calendar**
(``execution.calendar``, :func:`execution_bars`): the bars a decision can be
taken and an order placed on. Rebalance decisions are scheduled on it and the
execution lag counts its bars (docs/adr/0007).

Until v0.9.0 the pipeline zeroed the weight on EVERY missing price
(``weights.where(prices.notna(), 0)``), so a target set on a holiday rebalance
bar was flat for the whole holding period, and it silently dropped instruments
with under 50% price coverage (now counted, ``legacy_coverage_drop``).

Engine-agnostic: the backtest pipeline calls it once, before the engine
branch. The report is the ``calendar`` section of ``data_validation.json``
(:data:`DATA_VALIDATION_SCHEMA`), the shape the data-validation step (TOM-1430)
consumes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: ``data_validation.json`` (``artifact_schemas/data_validation.schema.json``).
DATA_VALIDATION_SCHEMA = "quantbox/data-validation@1"

#: The rule the pipeline applied until TOM-1429: drop a column with fewer priced bars
#: than max(30, 50% of the bars). Now only COUNTED, so the result change is visible.
LEGACY_MIN_OBS_FLOOR = 30
LEGACY_MIN_OBS_SHARE = 0.5

#: ``execution.calendar`` values other than a ticker.
EXECUTION_CALENDARS = ("majority", "union", "intersection")
DEFAULT_EXECUTION_CALENDAR = "majority"


@dataclass(frozen=True)
class InstrumentCalendar:
    """A prices panel resolved against each instrument's calendar.

    ``prices`` has no NaN (forward-filled inside each window, filled outside it
    for marking only); ``observed`` is True where the instrument really printed
    and ``inside`` where the bar lies in its life window — all three on the same
    index (bars where nothing printed are dropped) and columns (instruments that
    never print are dropped).
    """

    prices: pd.DataFrame
    observed: pd.DataFrame
    inside: pd.DataFrame
    report: dict[str, Any]


def _stamp(ts: Any) -> str:
    return pd.Timestamp(ts).isoformat()


def life_windows(prices: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame: True where the bar lies between the column's first and last valid price (inclusive)."""
    observed = prices.notna()
    return observed.cummax() & observed.iloc[::-1].cummax().iloc[::-1]


def instrument_calendar(prices: pd.DataFrame) -> InstrumentCalendar:
    """Resolve *prices* against each instrument's life window; the report counts every fill and drop."""
    observed_all = prices.notna()
    min_obs = max(LEGACY_MIN_OBS_FLOOR, int(len(prices) * LEGACY_MIN_OBS_SHARE))
    legacy_dropped = sorted(str(c) for c in prices.columns if int(observed_all[c].sum()) < min_obs)
    never_priced = [c for c in prices.columns if not observed_all[c].any()]
    keep_cols = [c for c in prices.columns if c not in never_priced]
    rows = observed_all[keep_cols].any(axis=1) if keep_cols else pd.Series(False, index=prices.index)

    px = prices.loc[rows, keep_cols]
    observed = observed_all.loc[rows, keep_cols]
    inside = life_windows(px)
    ffilled = inside & ~observed
    # ffill: holidays inside the window, the last price after delisting (target 0 there).
    # bfill reaches only bars BEFORE listing, where the target is 0: a mark, never a position.
    out_prices = px.ffill().bfill()

    instruments: dict[str, dict[str, Any]] = {}
    for c in keep_cols:
        col = observed[c]
        instruments[str(c)] = {
            "first_valid": _stamp(col.idxmax()),
            "last_valid": _stamp(col[::-1].idxmax()),
            "window_bars": int(inside[c].sum()),
            "observed_bars": int(col.sum()),
            "ffilled_bars": int(ffilled[c].sum()),
        }
    for c in never_priced:
        instruments[str(c)] = {
            "first_valid": None,
            "last_valid": None,
            "window_bars": 0,
            "observed_bars": 0,
            "ffilled_bars": 0,
        }
    report = {
        "policy": {
            "inside_window": "price forward-filled to mark; an order waits for the instrument's next printed bar",
            "outside_window": "target forced to 0; price filled for marking only",
        },
        "n_bars": int(rows.sum()),
        "n_instruments": len(instruments),
        "rows_without_any_price": int((~rows).sum()),
        "totals": {
            "ffilled_bars": int(ffilled.to_numpy().sum()),
            "instruments_without_prices": sorted(str(c) for c in never_priced),
        },
        # Columns the pre-TOM-1429 rule would have DROPPED (now traded inside their window): counted only.
        "legacy_coverage_drop": {
            "rule": f"fewer than max({LEGACY_MIN_OBS_FLOOR}, {LEGACY_MIN_OBS_SHARE:.0%} of {len(prices)} bars) "
            f"= {min_obs} priced bars",
            "count": len(legacy_dropped),
            "columns": legacy_dropped,
        },
        "instruments": instruments,
    }
    if legacy_dropped:
        logger.warning(
            "CALENDAR: %d column(s) would have been DROPPED before TOM-1429 (under %d priced bars) and are now "
            "traded inside their life window — results differ from earlier releases: %s",
            len(legacy_dropped),
            min_obs,
            legacy_dropped,
        )
    logger.info(
        "Calendar: %d holiday/gap bar(s) forward-filled inside life windows across %d instrument(s)",
        report["totals"]["ffilled_bars"],
        len(instruments),
    )
    return InstrumentCalendar(prices=out_prices, observed=observed, inside=inside, report=report)


def resolve_execution_calendar(value: Any) -> str:
    """``execution.calendar`` -> ``majority`` (default) | ``union`` | ``intersection`` | a ticker."""
    if value is None:
        return DEFAULT_EXECUTION_CALENDAR
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"execution.calendar must be one of {list(EXECUTION_CALENDARS)} or a ticker in the prices, got {value!r}"
        )
    return value


def execution_bars(cal: InstrumentCalendar, calendar: str, reference: pd.Series | None = None) -> pd.Series:
    """Boolean Series over ``cal``'s bars: True where a decision is taken and an order may be placed.

    ``majority``: at least half of the instruments inside their life window
    print; ``union``: any of them prints; ``intersection``: every one inside
    its window prints; a ticker: that series (*reference*, its column of the
    LOADED prices, which need not carry a weight) prints. PnL is still marked
    on every bar; only decisions and orders wait for these.
    """
    printed = (cal.observed & cal.inside).sum(axis=1)
    alive = cal.inside.sum(axis=1)
    if calendar == "majority":
        bars = (printed > 0) & (2 * printed >= alive)
    elif calendar == "union":
        bars = printed > 0
    elif calendar == "intersection":
        bars = (alive > 0) & (printed == alive)
    else:
        if reference is None:
            raise ValueError(
                f"execution.calendar: {calendar!r} is not one of {list(EXECUTION_CALENDARS)} and not a ticker in "
                "the loaded prices — add the reference series to the universe (it need not carry a weight)"
            )
        bars = reference.reindex(cal.observed.index).notna()
    return bars.astype(bool)


def execution_calendar_report(bars: pd.Series, calendar: str) -> dict[str, Any]:
    """The ``execution_calendar`` section of ``data_validation.json``."""
    years = pd.DatetimeIndex(bars.index).year
    by_year = (~bars).groupby(years).sum()
    return {
        "calendar": calendar,
        "execution_bars": int(bars.sum()),
        "total_bars": int(len(bars)),
        "non_execution_bars_by_year": {str(y): int(n) for y, n in by_year.items()},
    }


def calendar_summary(report: dict[str, Any]) -> dict[str, Any]:
    """The run-manifest-sized part of a calendar report: totals, no per-instrument rows."""
    return {
        "n_bars": report["n_bars"],
        "n_instruments": report["n_instruments"],
        "rows_without_any_price": report["rows_without_any_price"],
        "legacy_coverage_drop": report["legacy_coverage_drop"]["count"],
        **report["totals"],
    }


def validate_data_validation(doc: dict[str, Any]) -> list[str]:
    """Every way *doc* fails ``quantbox/data-validation@1``, as messages; ``[]`` means it validates."""
    import json
    from importlib.resources import files

    import jsonschema

    path = files("quantbox").joinpath("artifact_schemas").joinpath("data_validation.schema.json")
    validator = jsonschema.Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in validator.iter_errors(doc)
    ]
