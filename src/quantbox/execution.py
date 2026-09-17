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
                       # CLOSE of bar t + lag_bars. Default 1. 0 = same-bar.

``lag_bars: 0`` stays possible — historical numbers must be reproducible on
purpose — but only when written explicitly, and then it is loud
(:func:`warn_if_same_bar`) and recorded in ``run_manifest.json``.

Venue constraints live here too, because they answer the same question
("could this book have existed?"):

    venue:
      allow_shorts: false   # negative TARGET weights are clipped to 0 before
                            # any risk transform; longs are NOT re-levered.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_LAG_BARS = 1

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
                "CLOSE of bar t+lag_bars. Default 1 (next-bar). 0 = same-bar fill: the fill price is "
                "part of the information set that chose the weight (look-ahead for any close-based "
                "signal); allowed only when written explicitly, logged as a warning and recorded in "
                "run_manifest.json."
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
    },
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def resolve_lag_bars(execution_cfg: Any) -> int:
    """Validate an ``execution:`` block and return ``lag_bars`` (default 1).

    Raises ``ValueError`` on anything that is not exactly the declared shape —
    an unknown key is refused rather than ignored, because a typo
    (``lag_bar: 0``) silently falling back to a default is how a timing
    convention gets believed instead of applied.
    """
    if execution_cfg is None:
        return DEFAULT_LAG_BARS
    if not isinstance(execution_cfg, Mapping):
        raise ValueError(f"execution must be a mapping like {{lag_bars: 1}}, got {execution_cfg!r}")
    unknown = sorted(set(execution_cfg) - {"lag_bars"})
    if unknown:
        raise ValueError(f"execution: unknown key(s) {unknown}; the only key is 'lag_bars'")
    lag = execution_cfg.get("lag_bars", DEFAULT_LAG_BARS)
    if not _is_int(lag):
        raise ValueError(f"execution.lag_bars must be an integer >= 0, got {lag!r}")
    if lag < 0:
        raise ValueError(f"execution.lag_bars must be >= 0 (a negative lag trades on the future), got {lag}")
    return int(lag)


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
    fill_leading: float | None = 0.0,
) -> pd.DataFrame:
    """Shift decided weights forward by ``lag_bars`` rows — THE execution lag.

    Row ``t`` of the result is what the engine trades at ``close[t]``: the
    weights decided at ``t - lag_bars``. The first ``lag_bars`` rows have no
    decision behind them; they are set to ``fill_leading`` (0.0 = flat, the
    default) or left NaN with ``fill_leading=None``.

    ``lag_bars == 0`` returns the frame unchanged (same-bar; see module doc).
    """
    if lag_bars == 0:
        return weights
    lagged = weights.shift(lag_bars)
    if fill_leading is not None:
        lagged.iloc[:lag_bars] = fill_leading
    return lagged


def describe_execution(lag_bars: int) -> str:
    """The execution timing in words — for logs, summary.md, the report header."""
    if lag_bars == 0:
        return (
            "SAME-BAR (lag_bars=0): weights decided on bar t fill at close[t] — "
            "look-ahead for any signal that uses close[t]"
        )
    bars = "bar" if lag_bars == 1 else "bars"
    return (
        f"next-bar (lag_bars={lag_bars}): weights decided on bar t fill at the close of bar t+{lag_bars} {bars} later"
    )


def warn_if_same_bar(lag_bars: int, *, where: str) -> None:
    if lag_bars == 0:
        logger.warning(
            "EXECUTION TIMING — %s runs with execution.lag_bars=0 (SAME-BAR): weights decided on bar t "
            "are filled at close[t]. For any signal built from close[t] this is look-ahead and the "
            "result is NOT tradeable. Use only to reproduce a historical same-bar number.",
            where,
        )


def execution_record(lag_bars: int) -> dict[str, Any]:
    """The block written to ``run_manifest.json`` / ``RunResult.notes['execution']``."""
    return {
        "lag_bars": int(lag_bars),
        "fill": "close",
        "same_bar": lag_bars == 0,
        "description": describe_execution(lag_bars),
    }


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
    unknown = sorted(set(venue_cfg) - {"allow_shorts"})
    if unknown:
        raise ValueError(f"venue: unknown key(s) {unknown}; the only key is 'allow_shorts'")
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
    turnover = w.diff().fillna(w).abs().sum(axis=1)
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
