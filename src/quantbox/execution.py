"""Execution timing and venue constraints for backtests — ONE convention.

Both engines (vectorbt, rsims) are same-bar primitives: the weight row handed
to them for bar ``t`` is filled at ``close[t]``. Whether that is look-ahead
depends entirely on WHEN the row was decided. Strategies in this repo decide
``weights[t]`` with data through ``close[t]``, so handing them to an engine
unshifted trades on information the fill price already contains.

This module owns the one answer, shared by every entry point that turns
strategy weights into a simulated book (``backtest.pipeline.v1`` and
``analysis.parameter_grid``):

    execution:
      lag_bars: 1      # weights decided with data through bar t trade at the
                       # CLOSE of bar t + lag_bars. Default and minimum 1.

``lag_bars: 0`` (same-bar) is REFUSED, not warned (docs/adr/0005): a fill at
the close the signal was computed from is an order nobody could have placed,
so no number it produces is a backtest. Execution is at least one bar later —
unless the SAME block carries the explicit override (docs/adr/0006)::

    execution:
      lag_bars: 0
      same_bar: {allow: true, reason: "monthly-only data: ..."}

which is an allowance against best practice, recorded with its reason, and
classifies the run as RESEARCH, never a backtest (:func:`run_record`).

Venue constraints live here too, because they answer the same question
("could this book have existed?"):

    venue:
      allow_shorts: false   # negative TARGET weights are clipped to 0 before
                            # any risk transform; longs are NOT re-levered.
      financing: {...}      # what borrowed / idle cash costs (quantbox.financing,
                            # docs/adr/0007); resolved by resolve_financing.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from quantbox.financing import FINANCING_SCHEMA, LEVERAGE_SCHEMA
from quantbox.instrument_calendar import DEFAULT_EXECUTION_CALENDAR, EXECUTION_CALENDARS, resolve_execution_calendar

logger = logging.getLogger(__name__)

DEFAULT_LAG_BARS = 1
MIN_LAG_BARS = 1
#: ``execution.schedule``: the scheduled book (``calendar``) or the same book with every price bar executing.
SCHEDULES = ("calendar", "bars")

#: How the override is spelled in an error message — every refusal names it.
SAME_BAR_OVERRIDE = 'execution.same_bar: {allow: true, reason: "<why same-bar is closer to reality here>"}'

EXECUTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "lag_bars": {
            "type": "integer",
            "minimum": 0,
            "default": DEFAULT_LAG_BARS,
            "description": (
                "Execution lag in bars. Weights decided with data through bar t are filled at the "
                "CLOSE of bar t+lag_bars. Default and minimum 1 (next-bar). 0 (same-bar) is refused "
                "unless `same_bar` grants it: the fill price would be part of the information set "
                "that chose the weight."
            ),
        },
        "same_bar": {
            "type": "object",
            "additionalProperties": False,
            "required": ["allow", "reason"],
            "properties": {
                "allow": {"type": "boolean"},
                "reason": {"type": "string", "minLength": 1},
            },
            "description": (
                "The explicit override that lets lag_bars 0 run (docs/adr/0006), against best "
                "practice — e.g. monthly-only data, where the period's close is the only price. "
                "Valid only with lag_bars 0; the reason is recorded in run_manifest.json and the run "
                "is classified run.kind: research, not a backtest."
            ),
        },
        "calendar": {
            "type": "string",
            "default": DEFAULT_EXECUTION_CALENDAR,
            "description": (
                f"The EXECUTION calendar (docs/adr/0007): one of {list(EXECUTION_CALENDARS)} or a ticker in the "
                "loaded prices. A bar is an execution bar when at least half (majority) / any (union) / every "
                "(intersection) instrument inside its life window prints on it, or when the ticker prints. "
                "Rebalance decisions are scheduled on it and lag_bars counts its bars; PnL is marked on every bar."
            ),
        },
        "schedule": {
            "enum": list(SCHEDULES),
            "default": "calendar",
            "description": (
                "How decided weights become a traded book (docs/adr/0008). calendar (default): the scheduled "
                "book of ADR-0007 — instrument and execution calendars, deferral of unprinted orders, "
                "venue.leverage. bars: the same book on a degenerate calendar — every price bar is an "
                "execution bar, no deferral, no venue.leverage (declaring venue.leverage or venue.financing "
                "with it is refused)."
            ),
        },
    },
}

VENUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "allow_shorts": {
            "type": "boolean",
            "description": (
                "Whether the venue can hold short positions. false: negative target weights are "
                "clipped to 0 BEFORE tranching / leverage cap, and the long side is NOT re-normalised "
                "(the book simply carries less gross). true: shorts pass through. When `venue` is "
                "absent the legacy `risk.allow_short` (default false) decides, and the run warns if "
                "shorts are present either way. Must not contradict an explicit `risk.allow_short`."
            ),
        },
        "financing": FINANCING_SCHEMA,
        "leverage": LEVERAGE_SCHEMA,
    },
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class SameBarOverride:
    """A granted ``execution.same_bar`` override: the one thing that lets ``lag_bars`` be 0.

    Only :func:`resolve_execution` builds one from config, after checking it;
    :func:`apply_execution_lag` asks for it by type, so a bare ``0`` never fills.
    """

    reason: str


@dataclass(frozen=True)
class ExecutionTiming:
    """A resolved ``execution:`` block."""

    lag_bars: int
    same_bar: SameBarOverride | None = None
    #: ``execution.calendar`` (docs/adr/0007): decisions are scheduled on it and the lag counts its bars.
    calendar: str = DEFAULT_EXECUTION_CALENDAR
    #: ``execution.schedule``: ``calendar`` (the scheduled book) or ``bars`` (every price bar executes).
    schedule: str = "calendar"


def resolve_execution(execution_cfg: Any) -> ExecutionTiming:
    """Validate an ``execution:`` block and return the timing it declares.

    Raises ``ValueError`` on anything that is not exactly the declared shape —
    an unknown key is refused rather than ignored, because a typo
    (``lag_bar: 0``) silently falling back to a default is how a timing
    convention gets believed instead of applied.
    """
    if execution_cfg is None:
        return ExecutionTiming(DEFAULT_LAG_BARS)
    if not isinstance(execution_cfg, Mapping):
        raise ValueError(f"execution must be a mapping like {{lag_bars: 1}}, got {execution_cfg!r}")
    unknown = sorted(set(execution_cfg) - {"lag_bars", "same_bar", "calendar", "schedule"})
    if unknown:
        raise ValueError(
            f"execution: unknown key(s) {unknown}; the keys are 'lag_bars', 'same_bar', 'calendar' and 'schedule'"
        )
    lag = execution_cfg.get("lag_bars", DEFAULT_LAG_BARS)
    same_bar = _resolve_same_bar(execution_cfg.get("same_bar"))
    _check_lag(lag, same_bar)
    if same_bar is not None and lag != 0:
        raise ValueError(
            f"execution.same_bar is valid only with lag_bars 0, got lag_bars={lag!r}: the override "
            "would classify a next-bar run as research. Delete the same_bar block."
        )
    schedule = execution_cfg.get("schedule", "calendar")
    if schedule not in SCHEDULES:
        raise ValueError(f"execution.schedule must be one of {list(SCHEDULES)}, got {schedule!r}")
    return ExecutionTiming(int(lag), same_bar, resolve_execution_calendar(execution_cfg.get("calendar")), schedule)


def check_schedule_venue(timing: ExecutionTiming, venue_cfg: Any) -> None:
    """Refuse ``venue.leverage`` / ``venue.financing`` under ``execution.schedule: bars``.

    The bar grid applies no ``venue.leverage`` and appends no financing legs
    (docs/adr/0008): a block that asks for either would be silently ignored.
    """
    if timing.schedule != "bars" or not isinstance(venue_cfg, Mapping):
        return
    declared = sorted(k for k in ("leverage", "financing") if venue_cfg.get(k) is not None)
    if declared:
        raise ValueError(
            f"execution.schedule: bars applies no venue.leverage and no venue.financing, but the config declares "
            f"venue.{' and venue.'.join(declared)}. Drop it, or use execution.schedule: calendar (the default)."
        )


def resolve_lag_bars(execution_cfg: Any) -> int:
    """:func:`resolve_execution`, ``lag_bars`` only — for a caller that never fills a book."""
    return resolve_execution(execution_cfg).lag_bars


def _resolve_same_bar(block: Any) -> SameBarOverride | None:
    """``execution.same_bar`` -> the granted override, or None (absent, or ``allow: false``)."""
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"execution.same_bar must be a mapping like {{allow: true, reason: ...}}, got {block!r}")
    unknown = sorted(set(block) - {"allow", "reason"})
    if unknown:
        raise ValueError(f"execution.same_bar: unknown key(s) {unknown}; the keys are 'allow' and 'reason'")
    allow = block.get("allow")
    if not isinstance(allow, bool):
        raise ValueError(f"execution.same_bar.allow must be true or false, got {allow!r}")
    reason = block.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(
            f"execution.same_bar.reason must be a non-empty string, got {reason!r}: same-bar is an "
            "allowance against best practice, and the reason is what the manifest records for it."
        )
    return SameBarOverride(reason.strip()) if allow else None


def _check_lag(lag: Any, same_bar: SameBarOverride | None = None) -> None:
    """Raise unless ``lag`` is an integer >= 1, or 0 under a granted override — THE gate every entry point shares."""
    if not _is_int(lag):
        raise ValueError(f"execution.lag_bars must be an integer >= {MIN_LAG_BARS}, got {lag!r}")
    if lag == 0 and isinstance(same_bar, SameBarOverride):
        return
    if lag < MIN_LAG_BARS:
        raise ValueError(
            f"execution.lag_bars must be >= {MIN_LAG_BARS}, got {lag}: weights decided on bar t fill at "
            "the close of bar t+1 at the earliest. A smaller lag fills at (0, same-bar) or before "
            "(negative) the close the signal was computed from — look-ahead — and is refused, not "
            "warned (docs/adr/0005-next-bar-is-mandatory.md). Same-bar runs ONLY with the explicit "
            f"override {SAME_BAR_OVERRIDE} next to lag_bars: 0 (backtest()/optimize(): "
            "allow_same_bar=True, same_bar_reason=...), and is then a RESEARCH run, not a backtest "
            "(docs/adr/0006-same-bar-explicit-override.md)."
        )


def helper_execution(
    lag_bars: int | None, allow_same_bar: bool, same_bar_reason: str | None, schedule: str = "calendar"
) -> ExecutionTiming:
    """The ``backtest()`` / ``optimize()`` keywords as an ``execution:`` block, through the same resolver."""
    cfg: dict[str, Any] = {"schedule": schedule}
    if lag_bars is not None:
        cfg["lag_bars"] = lag_bars
    if allow_same_bar or same_bar_reason is not None:
        cfg["same_bar"] = {"allow": bool(allow_same_bar), "reason": same_bar_reason}
    return resolve_execution(cfg)


def resolve_sweep_lag_bars(lag_bars: int | None, shift_signal: int | None) -> int:
    """Resolve the sweep path's lag: ``lag_bars`` wins, ``shift_signal`` is a deprecated alias."""
    if shift_signal is not None:
        warnings.warn(
            "shift_signal is deprecated; use execution.lag_bars (same meaning, one convention "
            "shared with `quantbox run`).",
            DeprecationWarning,
            stacklevel=3,
        )
        alias = resolve_lag_bars({"lag_bars": shift_signal})
        if lag_bars is not None and resolve_lag_bars({"lag_bars": lag_bars}) != alias:
            raise ValueError(f"execution.lag_bars={lag_bars} contradicts deprecated shift_signal={shift_signal}")
        return alias
    if lag_bars is None:
        return DEFAULT_LAG_BARS
    return resolve_lag_bars({"lag_bars": lag_bars})


def apply_execution_lag(
    weights: pd.DataFrame,
    lag_bars: int,
    *,
    same_bar: SameBarOverride | None = None,
    fill_leading: float | None = 0.0,
) -> pd.DataFrame:
    """Shift decided weights forward by ``lag_bars`` rows — the engine seam's lag, for a caller outside it.

    Row ``t`` of the result is what an engine trades at ``close[t]``: the
    weights decided at ``t - lag_bars``. The first ``lag_bars`` rows have no
    decision behind them; they are set to ``fill_leading`` (0.0 = flat, the
    default) or left NaN with ``fill_leading=None``.

    A lag below 1 is refused, so a caller that skips :func:`resolve_execution`
    cannot hand an engine a same-bar book; ``0`` passes only with the
    :class:`SameBarOverride` that resolver granted. The lag itself is applied by
    :func:`quantbox.engine.lag_frame` (docs/adr/0008).
    """
    from quantbox.engine._lag import lag_frame

    return lag_frame(weights, lag_bars, same_bar=same_bar, fill_leading=fill_leading)


def materialise_nan_policy(weights: pd.DataFrame, engine: str | None = None) -> pd.DataFrame:
    """Make the seam's NaN policy explicit in a weights frame: a NaN cell HOLDS the last target (leading NaN -> 0).

    A NaN weight cell means "the strategy said nothing for this bar". The
    policy is decided ONCE, in the engine seam (:func:`quantbox.engine.materialise_nan`),
    for every engine (docs/adr/0008); *engine* is accepted for old callers and
    does not change the answer. ``engine=None`` returns the frame untouched.
    """
    if engine is None:
        return weights
    from quantbox.engine.book import materialise_nan

    return materialise_nan(weights)


def describe_execution(lag_bars: int, same_bar: SameBarOverride | None = None) -> str:
    """The execution timing in words — for logs, summary.md, the report header."""
    if lag_bars == 0:
        reason = same_bar.reason if same_bar is not None else "no override recorded"
        return (
            "same-bar (lag_bars=0): weights decided on bar t fill at the close of bar t — RESEARCH run, "
            f"not a backtest; allowed by execution.same_bar: {reason}"
        )
    bars = "bar" if lag_bars == 1 else "bars"
    return (
        f"next-bar (lag_bars={lag_bars}): weights decided on bar t fill at the close of bar t+{lag_bars} {bars} later"
    )


def execution_record(lag_bars: int, same_bar: SameBarOverride | None = None) -> dict[str, Any]:
    """The block written to ``run_manifest.json`` / ``RunResult.notes['execution']``."""
    _check_lag(lag_bars, same_bar)
    record: dict[str, Any] = {
        "lag_bars": int(lag_bars),
        "fill": "close",
        "same_bar": lag_bars == 0,
        "description": describe_execution(lag_bars, same_bar),
    }
    if same_bar is not None:
        record["same_bar_reason"] = same_bar.reason
    return record


def timing_record(timing: ExecutionTiming) -> dict[str, Any]:
    """:func:`execution_record` of a resolved timing; ``schedule`` is recorded only when it is ``bars``."""
    record = execution_record(timing.lag_bars, timing.same_bar)
    if timing.schedule != "calendar":
        record["schedule"] = timing.schedule
    return record


def run_record(execution: Mapping[str, Any]) -> dict[str, str]:
    """The run@1 ``run`` block an execution record implies: ``{"kind": "research" | "backtest"}``.

    A same-bar run is RESEARCH (docs/adr/0006): it may inform a question, it
    is never presented as a backtest result. Every reader that shows a result
    (the manifest, ``config explain``, the finding-report export, the gates)
    reads this one classification.
    """
    return {"kind": "research" if execution.get("same_bar") else "backtest"}


# ----------------------------------------------------------------------
# Venue
# ----------------------------------------------------------------------


def resolve_allow_shorts(venue_cfg: Any, risk_cfg: Mapping[str, Any] | None) -> tuple[bool, bool]:
    """Return ``(allow_shorts, venue_declared)``.

    ``venue.allow_shorts`` is authoritative when present. Otherwise the legacy
    ``risk.allow_short`` (default ``False``) decides. An explicit
    ``risk.allow_short`` that contradicts ``venue.allow_shorts`` is refused:
    one fact, two keys, they must agree.
    """
    risk_cfg = risk_cfg or {}
    legacy = bool(risk_cfg.get("allow_short", False))
    if venue_cfg is None:
        return legacy, False
    if not isinstance(venue_cfg, Mapping):
        raise ValueError(f"venue must be a mapping like {{allow_shorts: false}}, got {venue_cfg!r}")
    unknown = sorted(set(venue_cfg) - {"allow_shorts", "financing", "leverage"})
    if unknown:
        raise ValueError(f"venue: unknown key(s) {unknown}; the keys are 'allow_shorts', 'financing' and 'leverage'")
    if "allow_shorts" not in venue_cfg:
        raise ValueError("venue: 'allow_shorts' is required when a venue block is declared")
    allow = venue_cfg["allow_shorts"]
    if not isinstance(allow, bool):
        raise ValueError(f"venue.allow_shorts must be true or false, got {allow!r}")
    if "allow_short" in risk_cfg and bool(risk_cfg["allow_short"]) != allow:
        raise ValueError(
            f"venue.allow_shorts={allow} contradicts risk.allow_short={risk_cfg['allow_short']}; "
            "declare the venue once (drop risk.allow_short)."
        )
    return allow, True


def clip_shorts(weights: pd.DataFrame) -> pd.DataFrame:
    """Clip negative weights to 0. The long side is left exactly as it was (no re-levering)."""
    return weights.clip(lower=0)


def exposure_metrics(weights: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Exposure / turnover statistics of a weights frame, keys ``{prefix}_*``.

    - ``mean_gross_exposure``: mean over bars of sum(|w|)
    - ``mean_net_exposure``: mean over bars of sum(w)
    - ``short_gross_share``: sum(|w| where w<0) / sum(|w|) over the whole frame (0 when flat)
    - ``mean_turnover``: mean over bars of sum(|w[t] - w[t-1]|) (first bar vs a flat book)
    - ``flat_bar_share``: share of bars with zero gross
    """
    w = weights.select_dtypes(include="number").fillna(0.0)
    if w.empty:
        return {
            f"{prefix}_mean_gross_exposure": 0.0,
            f"{prefix}_mean_net_exposure": 0.0,
            f"{prefix}_short_gross_share": 0.0,
            f"{prefix}_mean_turnover": 0.0,
            f"{prefix}_flat_bar_share": 1.0,
        }
    gross = w.abs().sum(axis=1)
    total_gross = float(gross.sum())
    short_gross = float(w.clip(upper=0).abs().sum().sum())
    from quantbox.metrics import turnover_series

    turnover = turnover_series(w, from_flat=True)
    return {
        f"{prefix}_mean_gross_exposure": float(gross.mean()),
        f"{prefix}_mean_net_exposure": float(w.sum(axis=1).mean()),
        f"{prefix}_short_gross_share": short_gross / total_gross if total_gross > 0 else 0.0,
        f"{prefix}_mean_turnover": float(turnover.mean()),
        f"{prefix}_flat_bar_share": float((gross == 0).mean()),
    }


def warn_on_shorts(
    *,
    target_short_share: float,
    traded_short_share: float,
    venue_declared: bool,
    allow_shorts: bool,
    where: str,
) -> list[str]:
    """Log (and return) the loud lines about shorts a reader must not miss."""
    lines: list[str] = []
    if target_short_share > 0 and not allow_shorts:
        source = "venue.allow_shorts=false" if venue_declared else "risk.allow_short=false (the default)"
        lines.append(
            f"VENUE — {where}: target weights are {target_short_share:.1%} SHORT by gross, and those shorts "
            f"are CLIPPED to 0 by {source}. The traded book is long-only and is NOT the book the strategy "
            "designed; the long side is not re-levered."
        )
    if traded_short_share > 0 and not venue_declared:
        lines.append(
            f"VENUE — {where}: the traded book is {traded_short_share:.1%} SHORT by gross and no `venue:` "
            "block is declared. On a spot market these positions cannot exist. Declare "
            "`venue: {allow_shorts: true}` (perps/margin) or `venue: {allow_shorts: false}` (spot)."
        )
    for line in lines:
        logger.warning(line)
    return lines
