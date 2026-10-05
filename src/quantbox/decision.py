"""The decision layer: decided weights -> FINAL target weights, before any rebalancing policy reads them (TOM-1520).

Tom, 2026-10-05: "Normalizacja jest częścią strategii, bo pokazuje, co byśmy
robili w realnym świecie." The order is::

    signals -> weights -> [this module] -> TARGET weights -> rebalancing policy -> execution

The target weights are final here, and the same in a backtest and in live
trading: ``backtest.pipeline.v1``, ``trade.full_pipeline.v1``, ``backtest()``,
``optimize()`` and the sweep all call :func:`final_targets`. The rebalancing
policy (:mod:`quantbox.engine.policy`: cadence x trigger, ``min_trade``) then
reads them, and the engine seam (:mod:`quantbox.engine`) executes them. "Konto
bez lewara nie otrzymuje wag, gdzie wymagany jest lewar": an account without
leverage never receives weights that need it.

ONE ordered transform, row by row:

1. **short clip** — without ``allow_short`` a negative weight becomes 0; the
   long side is not re-levered;
2. **gross cap** — a row whose gross ``sum |w|`` is above ``max_leverage`` is
   scaled down to it, proportionally;
3. **group limits** (:mod:`quantbox.engine.groups`) — each group's gross
   inside its ``[min, max]``;
4. **normalisation** (``venue.leverage``) — ``normalize`` scales a row whose
   NET exposure ``sum w`` is above 1 (by more than
   :data:`quantbox.financing.NET_EXPOSURE_TOLERANCE`) down to net 1,
   proportionally; ``borrow`` keeps it (a levered variant declares its
   leverage; the financing legs carry it); ``none`` (``execution.schedule:
   bars``) only measures it.

Each step only scales DOWN or clips, so a later step never undoes an earlier
one: a capped row stays under the cap, a group maximum still holds after
normalisation (a group minimum can fall below its bound by the normalisation
scale, as before).

**NaN cells.** The seam's NaN policy (HOLD: a NaN cell keeps the last decided
target, :func:`quantbox.engine.materialise_nan`) is applied first, so a row's
gross and net count what the book would really hold. A cell that came in NaN
and that no step changed goes back to NaN, so the rows the strategy did not
write stay unwritten (the seam reindexes onto the price bars and fills them
there, as it always did).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.engine.groups import GroupLimits, apply_group_limits
from quantbox.engine.schedule import materialise_nan
from quantbox.financing import LEVERAGE_MODES, NET_EXPOSURE_TOLERANCE

logger = logging.getLogger(__name__)

#: ``venue.leverage`` values the decision reads: the config's two, and ``none`` (``schedule: bars``: measured only).
DECISION_LEVERAGE = (*LEVERAGE_MODES, "none")
#: The steps of :func:`final_targets`, in the order they run.
ORDER = ("short_clip", "gross_cap", "group_limits", "normalise")


@dataclass(frozen=True)
class DecisionRules:
    """What :func:`final_targets` applies: the short clip, the gross cap, the group limits and ``venue.leverage``."""

    allow_short: bool = True
    #: The gross cap; ``None`` = no cap.
    max_leverage: float | None = None
    #: Bound to a universe (:meth:`GroupLimits.bind`); ``None`` = no group limits.
    groups: GroupLimits | None = None
    leverage: str = "normalize"

    def __post_init__(self) -> None:
        if self.leverage not in DECISION_LEVERAGE:
            raise ValueError(f"venue.leverage must be one of {list(LEVERAGE_MODES)}, got {self.leverage!r}")
        if self.groups is not None and self.groups.membership is None:
            raise ValueError("group_limits are not bound to a universe: pass GroupLimits.bind(universe)")

    def record(self) -> dict[str, Any]:
        """JSON-safe: what ``data_validation.json`` and the run notes record."""
        return {
            "order": list(ORDER),
            "allow_short": bool(self.allow_short),
            "max_leverage": None if self.max_leverage is None else float(self.max_leverage),
            "group_limits": None if self.groups is None else self.groups.record(),
            "leverage": self.leverage,
        }


def risk_caps(weights: pd.DataFrame, *, allow_short: bool, max_leverage: float | None) -> pd.DataFrame:
    """Steps 1-2 of :func:`final_targets` on *weights* as given: the short clip, then the gross cap.

    The live rebalancers call this through :func:`risk_caps_row` on the targets
    they size, so re-applying it to final targets changes nothing (a clipped
    row has no short, a capped row is under its cap).
    """
    w = weights if allow_short else weights.clip(lower=0)
    if max_leverage is None:
        return w
    gross = w.abs().sum(axis=1)
    scale = (float(max_leverage) / gross).clip(upper=1.0)
    return w.mul(scale, axis=0)


def risk_caps_row(weights: Mapping[str, float], *, allow_short: bool, max_leverage: float | None) -> dict[str, float]:
    """:func:`risk_caps` on one row ``{symbol: weight}``."""
    if not weights:
        return {}
    row = pd.DataFrame([dict(weights)], dtype=float)
    out = risk_caps(row, allow_short=allow_short, max_leverage=max_leverage).iloc[0]
    return {str(k): float(v) for k, v in out.items()}


def final_targets(weights: pd.DataFrame, rules: DecisionRules) -> tuple[pd.DataFrame, dict[str, Any]]:
    """*weights* (date x symbol, as decided) -> the FINAL target weights, and the decision report.

    See the module docstring for the order. The report (``data_validation.json``
    ``decision``) counts the rows each step changed and the normalisation
    scales; ``groups`` holds the group-limit report when limits are declared.
    """
    if weights.empty:
        return weights.copy(), _report(rules, 0, 0, 0, np.zeros(0), np.zeros(0), np.zeros(0), None)
    raw = weights.astype(float)
    filled = materialise_nan(raw)
    clipped = filled if rules.allow_short else filled.clip(lower=0)
    short_rows = int((clipped.to_numpy() != filled.to_numpy()).any(axis=1).sum())
    capped = risk_caps(clipped, allow_short=True, max_leverage=rules.max_leverage)
    cap_rows = int((capped.to_numpy() != clipped.to_numpy()).any(axis=1).sum())
    group_report = None
    grouped = capped
    if rules.groups is not None:
        grouped, group_report = apply_group_limits(capped, rules.groups)
    arr = grouped.to_numpy(dtype=float, copy=True)
    net_before = arr.sum(axis=1)
    above = net_before > 1.0 + NET_EXPOSURE_TOLERANCE
    scales = np.ones(len(arr))
    if rules.leverage == "normalize" and above.any():
        scales[above] = 1.0 / net_before[above]
        arr[above] = arr[above] * scales[above][:, None]
    out = pd.DataFrame(arr, index=grouped.index, columns=grouped.columns)
    # A cell the strategy left NaN, and that no step changed, stays NaN (the seam fills it on the price bars).
    same = out.eq(filled)
    untouched = raw.isna() & same & same.shift(1, fill_value=True)
    out = out.mask(untouched)
    report = _report(
        rules,
        len(arr),
        short_rows,
        cap_rows,
        net_before,
        arr.sum(axis=1),
        scales[above] if rules.leverage == "normalize" else np.zeros(0),
        group_report,
        above_rows=int(above.sum()),
    )
    return out, report


def _report(
    rules: DecisionRules,
    rows: int,
    short_rows: int,
    cap_rows: int,
    net_before: np.ndarray,
    net_after: np.ndarray,
    scales: np.ndarray,
    group_report: dict[str, Any] | None,
    *,
    above_rows: int = 0,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rules": rules.record(),
        "rows": int(rows),
        "rows_short_clipped": int(short_rows),
        "rows_gross_capped": int(cap_rows),
        "rows_above_net_1": int(above_rows),
        "rows_normalised": int(len(scales)),
        "scale_mean": float(scales.mean()) if len(scales) else 1.0,
        "scale_min": float(scales.min()) if len(scales) else 1.0,
        "max_net_exposure_decided": float(net_before.max()) if len(net_before) else 0.0,
        "max_net_exposure_final": float(net_after.max()) if len(net_after) else 0.0,
    }
    if group_report is not None:
        out["groups"] = group_report
    return out


def final_book(
    decided: pd.DataFrame | dict[str, pd.DataFrame], rules: DecisionRules
) -> tuple[pd.DataFrame | dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """:func:`final_targets` on every strategy slice of a book (a frame, a dict of frames, or MultiIndex
    columns with the ticker last), and each slice's report, in slice order."""
    if isinstance(decided, dict):
        pairs = {k: final_targets(w, rules) for k, w in decided.items()}
        return {k: v[0] for k, v in pairs.items()}, [v[1] for v in pairs.values()]
    if decided.columns.nlevels == 1:
        out, report = final_targets(decided, rules)
        return out, [report]
    levels = list(range(decided.columns.nlevels - 1))
    parts: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    for _key, frame in decided.T.groupby(level=levels[0] if len(levels) == 1 else levels, sort=False):
        w = frame.T
        cols = w.columns
        w.columns = cols.get_level_values(-1)
        out, report = final_targets(w, rules)
        out.columns = cols
        parts.append(out)
        reports.append(report)
    return pd.concat(parts, axis=1).reindex(columns=decided.columns), reports


def log_normalisation(report: Mapping[str, Any], *, where: str = "") -> None:
    """The loud lines for rows the gross cap and ``normalize`` scaled (one each per run, not per row)."""
    if report.get("rows_gross_capped"):
        logger.warning(
            "RISK: %s%d of %d decided row(s) had gross exposure above max_leverage %s and were scaled down to it.",
            where,
            report["rows_gross_capped"],
            report["rows"],
            report["rules"]["max_leverage"],
        )
    if report.get("rows_normalised"):
        logger.warning(
            "LEVERAGE: %s%d of %d decided row(s) had net exposure above 1 (max %.4f) and were SCALED to net 1 in "
            "the decision (venue.leverage: normalize, the default; scale mean %.4f, min %.4f). Declare "
            "venue.leverage: borrow (with venue.financing) to hold the levered book.",
            where,
            report["rows_normalised"],
            report["rows"],
            report["max_net_exposure_decided"],
            report["scale_mean"],
            report["scale_min"],
        )
