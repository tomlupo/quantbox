"""Group limits — asset-class (or any category) limits on the DECIDED book, before execution (TOM-1450 3d).

The groups come from universe metadata: a column of the data plugin's
``load_universe()`` frame (``asset_class``, ``category``, ...) maps each
symbol to its group. A limit bounds a group's GROSS weight (``sum |w|``,
fractions of the portfolio value; for a long-only book its sum of weights) on
every row of the decided book::

    group_limits:
      by: asset_class
      limits:
        equity: {max: 0.6}
        bond:   {min: 0.2, max: 0.5}
      excess: redistribute     # | cash

On a row that breaks a limit, each group's gross ``g`` becomes
``t = clip(s * g, min, max)`` with one scale ``s`` for the row, solved so that
the total is preserved: ``redistribute`` (the default) keeps the row's gross,
so weight cut from a capped group goes to the others pro rata (and to cash
only when every group is at its max); ``cash`` keeps the cut as cash. A raised
minimum is paid for by the other groups pro rata, in both modes. Inside a
group every instrument is scaled by ``t / g``, so the strategy's mix within
the group and every sign are kept. A row that is within its limits is
unchanged.

Refused loudly (``ValueError``), never bent:

- a malformed limit (min above max, a negative bound, an unknown key),
  a limit on a group no symbol belongs to (a typo), a ``by`` column the universe
  does not carry, a weighted symbol with no group;
- INFEASIBLE on a row: the minimums need more gross than the row holds, or a
  group with a minimum holds nothing on the row (there is no instrument to
  scale up). The message names the first rows.

A flat row (all cash) has no gross to allocate: it is counted (``flat_rows``),
not refused. The limits bind the decided book; the seam's ``venue.leverage:
normalize`` then only scales a row DOWN, proportionally, so a maximum still
holds; a minimum on a row above net 1 can fall below its bound by that scale.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

EXCESS = ("redistribute", "cash")
_TOL = 1e-12

#: ``backtest.pipeline.v1`` ``group_limits``: for validate and config explain.
GROUP_LIMITS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Group (asset-class) limits on the decided weights before execution, the same on every engine: "
        "{by: <universe metadata column>, limits: {<group>: {min, max}}, excess: redistribute | cash}. A group's "
        "gross weight is kept inside [min, max] on every row; an infeasible limit refuses the run "
        "(quantbox.engine.groups)."
    ),
    "required": ["by", "limits"],
    "additionalProperties": False,
    "properties": {
        "by": {
            "type": "string",
            "description": "The universe column that names each symbol's group (asset_class, category, ...).",
        },
        "limits": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"min": {"type": "number", "minimum": 0}, "max": {"type": "number", "minimum": 0}},
            },
            "description": "{group: {min, max}}: gross weight bounds, fractions of the portfolio value.",
        },
        "excess": {
            "type": "string",
            "enum": list(EXCESS),
            "default": "redistribute",
            "description": (
                "Weight cut from a capped group: redistribute = to the other groups pro rata (the row's gross is "
                "kept); cash = held as cash."
            ),
        },
    },
}


@dataclass(frozen=True)
class GroupLimits:
    """Resolved group limits (:func:`resolve_group_limits`); :meth:`bind` adds the symbol -> group map."""

    by: str
    #: ``{group: (min, max)}``; ``max`` may be ``inf``.
    limits: Mapping[str, tuple[float, float]]
    excess: str = "redistribute"
    #: ``{symbol: group}`` from the universe metadata; None until :meth:`bind`.
    membership: Mapping[str, str] | None = field(default=None)

    def bind(self, universe: pd.DataFrame) -> GroupLimits:
        """These limits with the symbol -> group map read from *universe* (``symbol`` + the ``by`` column)."""
        if not isinstance(universe, pd.DataFrame) or "symbol" not in universe.columns:
            raise ValueError("group_limits: the universe has no `symbol` column to read the groups from")
        if self.by not in universe.columns:
            raise ValueError(
                f"group_limits.by: the universe carries no {self.by!r} column (it has {list(universe.columns)}); "
                "the data plugin's load_universe() must return it as metadata"
            )
        rows = universe[["symbol", self.by]].dropna().drop_duplicates()
        clash = rows["symbol"][rows["symbol"].duplicated()].tolist()
        if clash:
            raise ValueError(f"group_limits: symbol(s) {sorted(map(str, clash))} sit in more than one {self.by}")
        membership = {str(s): str(g) for s, g in zip(rows["symbol"], rows[self.by], strict=True)}
        unknown = sorted(set(self.limits) - set(membership.values()))
        if unknown:
            raise ValueError(
                f"group_limits.limits: {unknown} is not a group of the universe's {self.by!r} "
                f"(groups: {sorted(set(membership.values()))})"
            )
        return replace(self, membership=membership)

    def record(self) -> dict[str, Any]:
        """JSON-safe: ``data_validation.json``, ``quantbox config explain`` and the run notes."""
        return {
            "by": self.by,
            "excess": self.excess,
            "limits": {
                g: {"min": lo, **({"max": hi} if np.isfinite(hi) else {})} for g, (lo, hi) in self.limits.items()
            },
        }


def resolve_group_limits(spec: Mapping[str, Any] | GroupLimits) -> GroupLimits:
    """A declared ``group_limits`` -> :class:`GroupLimits` (unbound), refused when malformed."""
    if isinstance(spec, GroupLimits):
        return spec
    if not isinstance(spec, Mapping):
        raise ValueError(f"group_limits must be a mapping {{by, limits, excess}}, got {spec!r}")
    extra = sorted(set(spec) - {"by", "limits", "excess"})
    if extra:
        raise ValueError(f"group_limits does not take {extra}; it takes ['by', 'limits', 'excess']")
    by = spec.get("by")
    if not isinstance(by, str) or not by:
        raise ValueError("group_limits.by must name the universe metadata column of the groups (asset_class, ...)")
    excess = spec.get("excess", "redistribute")
    if excess not in EXCESS:
        raise ValueError(f"group_limits.excess must be one of {list(EXCESS)}, got {excess!r}")
    raw = spec.get("limits")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("group_limits.limits needs at least one {group: {min, max}}")
    limits: dict[str, tuple[float, float]] = {}
    for group, lim in raw.items():
        if not isinstance(lim, Mapping) or not lim or set(lim) - {"min", "max"}:
            raise ValueError(f"group_limits.limits[{group!r}] takes min, max (one or both), got {lim!r}")
        try:
            lo = float(lim.get("min", 0.0))
            hi = float(lim.get("max", np.inf))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"group_limits.limits[{group!r}]: min and max must be numbers, got {lim!r}") from exc
        if any(isinstance(v, bool) for v in lim.values()) or not (lo >= 0 and hi >= 0):
            raise ValueError(f"group_limits.limits[{group!r}]: min and max must be >= 0, got {lim!r}")
        if lo > hi:
            raise ValueError(f"group_limits.limits[{group!r}]: min {lo} is above its max {hi}: no weight satisfies it")
        limits[str(group)] = (lo, hi)
    return GroupLimits(by=by, limits=limits, excess=excess)


def _scale_for(g: np.ndarray, lo: np.ndarray, hi: np.ndarray, total: float) -> float:
    """The ``s`` with ``sum clip(s * g, lo, hi) == total`` (monotone in s: bisection)."""

    def f(s: float) -> float:
        return float(np.clip(s * g, lo, hi).sum())

    s_lo, s_hi = 0.0, 1.0
    for _ in range(200):
        if f(s_hi) >= total:
            break
        s_lo, s_hi = s_hi, s_hi * 2.0
    for _ in range(200):
        mid = 0.5 * (s_lo + s_hi)
        if f(mid) >= total:
            s_hi = mid
        else:
            s_lo = mid
        if s_hi - s_lo <= 1e-16 * max(s_hi, 1.0):
            break
    return s_hi


def apply_group_limits(weights: pd.DataFrame, limits: GroupLimits) -> tuple[pd.DataFrame, dict[str, Any]]:
    """*weights* with every row's groups inside their limits, and the report (see the module docstring).

    *weights* hold no NaN (the seam materialises its NaN policy first). Raises
    ``ValueError`` on a weighted symbol with no group and on an INFEASIBLE row.
    """
    if limits.membership is None:
        raise ValueError("group_limits are not bound to a universe: call GroupLimits.bind(universe) first")
    cols = [str(c) for c in weights.columns]
    w = weights.to_numpy(dtype=float, copy=True)
    weighted = np.abs(w).sum(axis=0) > 0
    ungrouped = sorted(c for c, on in zip(cols, weighted, strict=True) if on and c not in limits.membership)
    if ungrouped:
        raise ValueError(
            f"group_limits: weighted symbol(s) {ungrouped} have no {limits.by} in the universe metadata; every "
            "symbol the strategy weights needs a group"
        )
    member = np.array([limits.membership.get(c, "") for c in cols])
    groups = sorted({g for g in member if g} | set(limits.limits))
    lo = np.array([limits.limits.get(g, (0.0, np.inf))[0] for g in groups])
    hi = np.array([limits.limits.get(g, (0.0, np.inf))[1] for g in groups])
    onehot = np.stack([member == g for g in groups], axis=1).astype(float)  # (n_inst, n_groups)
    gross = np.abs(w) @ onehot  # (n_bars, n_groups)
    total = gross.sum(axis=1)
    flat = total <= _TOL
    breaks = ((gross > hi + _TOL) | (gross < lo - _TOL)).any(axis=1) & ~flat
    index = weights.index
    infeasible: list[str] = []
    adjusted = 0
    cut_to_cash = 0.0
    for r in np.flatnonzero(breaks):
        g = gross[r]
        empty_min = (g <= _TOL) & (lo > _TOL)
        lo_r = np.where(g > _TOL, lo, 0.0)
        hi_r = np.where(g > _TOL, hi, 0.0)
        if limits.excess == "redistribute":
            target_total = min(float(total[r]), float(hi_r.sum()))
        else:
            target_total = float(total[r] - np.maximum(g - hi, 0.0).sum())
        if empty_min.any() or lo.sum() > target_total + _TOL:
            why = (
                f"group(s) {[groups[k] for k in np.flatnonzero(empty_min)]} hold nothing to scale up to their min"
                if empty_min.any()
                else f"the minimums need {lo.sum():.4f} gross, the row holds {target_total:.4f}"
            )
            infeasible.append(f"{pd.Timestamp(index[r]).date()}: {why}")
            continue
        s = _scale_for(g, lo_r, hi_r, target_total)
        t = np.clip(s * g, lo_r, hi_r)
        factor = np.divide(t, g, out=np.ones_like(g), where=g > _TOL)
        w[r] *= onehot @ factor
        cut_to_cash = max(cut_to_cash, float(total[r] - t.sum()))
        adjusted += 1
    if infeasible:
        raise ValueError(
            f"group_limits INFEASIBLE on {len(infeasible)} row(s) of the decided book (first: {infeasible[:3]}): "
            f"no weights satisfy {limits.record()['limits']} there. Loosen the limits or change the book."
        )
    after = np.abs(w) @ onehot
    report = {
        **limits.record(),
        "rows": int(len(w)),
        "rows_adjusted": int(adjusted),
        "flat_rows": int(flat.sum()),
        "max_gross_cut_to_cash": float(cut_to_cash),
        "groups": {
            g: {
                "max_gross_decided": float(gross[:, k].max()) if len(gross) else 0.0,
                "max_gross_after": float(after[:, k].max()) if len(after) else 0.0,
                "min_gross_after_invested": float(after[~flat, k].min()) if (~flat).any() else 0.0,
            }
            for k, g in enumerate(groups)
        },
    }
    return pd.DataFrame(w, index=weights.index, columns=weights.columns), report
