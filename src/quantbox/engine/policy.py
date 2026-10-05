"""Rebalancing policies — WHEN the seam places orders, and on which cells (TOM-1450 3d, docs/adr/0008).

A policy is part of the seam's schedule (:mod:`quantbox.engine.schedule`): it
produces the per-cell ORDERS mask (and, for ``tranche``, the targets) that
every engine executes. No adapter knows which policy ran.

Every policy has a ``frequency`` — the bars on which a rebalance is CONSIDERED
— and an optional market ``calendar``:

- ``frequency``: ``daily`` / ``weekly`` / ``monthly`` / ``quarterly`` /
  ``yearly``, or any ``rebalancing_freq`` form (an int n = every n-th execution
  bar, a pandas offset such as ``"W-FRI"`` or ``"ME"``, a list of dates, or
  null = buy-and-hold). ``weekly`` is the last execution bar of each calendar
  week (Monday to Sunday), ``monthly`` the last of each month
  (:func:`quantbox.frequency.rebalancing_dates`).
- ``calendar``: a pandas-market-calendars name (``NYSE``, ``XLON``, ...). The
  execution bars are narrowed to that market's SESSIONS, so a decision falls on
  the last session of the period (Good Friday moves a month-end to Thursday)
  and the execution lag is counted in sessions. Without it, the execution
  calendar of the data (``execution.calendar``) alone decides.

The four policies:

``periodic``
    Trade every instrument to its target on every considered bar. Today's
    ``rebalancing_freq`` is this policy.
``tranche``
    Stagger the book across ``tranches`` = N tranches. Tranche k is refreshed
    to the decided weights on every N-th considered bar (offset k) and holds
    them for N decisions; the book is their mean, traded on every considered
    bar. Every tranche starts at the first decision. A tranche holds its
    TARGET weights between refreshes — it is not a separate sub-account that
    drifts. (``risk.tranches`` is a different thing: a rolling mean of the
    decided weights over N BARS, before the seam.)
``band``
    A considered rebalance is placed only when the held book has drifted more
    than ``band`` (absolute weight) from its target on some ordered
    instrument; it then trades EVERY ordered instrument to target. Today's
    ``threshold`` is this policy.
``corridor``
    Each instrument has its own corridor around its target, ``width`` =
    ``[below, above]`` (absolute weight; one number = symmetric), per
    instrument in ``bounds``. The corridors are only the TRIGGER: when any
    instrument's held weight is outside its corridor on a considered bar,
    EVERY ordered instrument trades back to target (TOM-1513, Tom 2026-10-05:
    "a corridor hit triggers the whole rebalance"). A target of 0 with a held
    weight is always a hit: an exit is never left inside a corridor.

``band`` and ``corridor`` take the same action; they differ in the trigger
only (one absolute width for every instrument vs a corridor per instrument).

``band`` and ``corridor`` read the held book. The seam tracks it cost-free
(the price drift of the weights held since each placed order, against a cash
remainder), so a run with costs can trigger on slightly different bars than
an engine-internal band would. Both engines trade on the bars the seam keeps
(docs/adr/0008, the threshold caveat).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: The rebalancing policies, in the order the docs list them.
POLICIES = ("periodic", "tranche", "band", "corridor")

#: Frequency names -> the ``rebalancing_freq`` they mean. Period-END offsets: the last bar of the period.
FREQUENCY_NAMES: dict[str, Any] = {
    "daily": 1,
    "weekly": "W-SUN",
    "monthly": "ME",
    "quarterly": "QE",
    "yearly": "YE",
}

_COMMON = ("policy", "frequency", "calendar")
#: The keys each policy takes; any other key is refused.
POLICY_KEYS: dict[str, tuple[str, ...]] = {
    "periodic": _COMMON,
    "tranche": (*_COMMON, "tranches"),
    "band": (*_COMMON, "band"),
    "corridor": (*_COMMON, "width", "bounds"),
}

_FREQUENCY_DOC = (
    "The bars a rebalance is CONSIDERED on: daily | weekly | monthly | quarterly | yearly (the LAST execution bar "
    "of each period; weekly = Monday-Sunday weeks), or any rebalancing_freq form: an int n (every n-th execution "
    "bar), a pandas offset ('W-FRI', 'ME', 'BMS'), a list of dates, or null (buy-and-hold: the first bar only). "
    "Default 1: every execution bar."
)

#: ``backtest.pipeline.v1`` ``rebalancing_policy``: every policy and its keys, for validate and config explain.
POLICY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "How decided weights become orders, in the engine seam, the same on every engine (docs/adr/0008, "
        "TOM-1450). periodic: trade every instrument to target on every considered bar. tranche: the book is "
        "the mean of N staggered tranches, each refreshed every N-th decision. band: trade the whole book when "
        "a held weight drifted more than `band` from target. corridor: trade the whole book when a held weight "
        "left its own [target - below, target + above] corridor. Replaces rebalancing_freq + "
        "threshold (declare one or the other, not both)."
    ),
    "required": ["policy"],
    "additionalProperties": False,
    "properties": {
        "policy": {"type": "string", "enum": list(POLICIES), "description": "periodic | tranche | band | corridor."},
        "frequency": {"type": ["integer", "string", "array", "null"], "default": 1, "description": _FREQUENCY_DOC},
        "calendar": {
            "type": ["string", "null"],
            "default": None,
            "description": (
                "A pandas-market-calendars name (NYSE, XLON, ...): decisions and the execution lag use that "
                "market's sessions only, so a month-end on a holiday moves to the session before it. null = the "
                "data's execution calendar alone."
            ),
        },
        "tranches": {
            "type": "integer",
            "minimum": 2,
            "description": "tranche only, required: N tranches; each is refreshed on every N-th decision.",
        },
        "band": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "band only, required: absolute weight drift that triggers a whole-book rebalance.",
        },
        "width": {
            "type": ["number", "array"],
            "items": {"type": "number", "minimum": 0},
            "minItems": 2,
            "maxItems": 2,
            "minimum": 0,
            "description": (
                "corridor only, required: the default corridor [below, above] around each target, absolute "
                "weight (one number = symmetric)."
            ),
        },
        "bounds": {
            "type": "object",
            "additionalProperties": {
                "type": ["number", "array"],
                "items": {"type": "number", "minimum": 0},
                "minItems": 2,
                "maxItems": 2,
                "minimum": 0,
            },
            "default": {},
            "description": "corridor only: per-instrument corridors {symbol: [below, above] | width}, over `width`.",
        },
    },
    "allOf": [
        {
            "if": {"properties": {"policy": {"const": name}}},
            "then": {
                "properties": {k: False for k in ("tranches", "band", "width", "bounds") if k not in keys},
                **({"required": [k for k in ("tranches", "band", "width") if k in keys]} if name != "periodic" else {}),
            },
        }
        for name, keys in POLICY_KEYS.items()
    ],
}


@dataclass(frozen=True)
class RebalancePolicy:
    """A resolved rebalancing policy (see the module docstring); build it with :func:`resolve_policy`."""

    policy: str = "periodic"
    #: As declared: a frequency name or a ``rebalancing_freq`` form.
    frequency: Any = 1
    calendar: str | None = None
    tranches: int = 1
    band: float | None = None
    #: corridor: the default ``(below, above)``.
    width: tuple[float, float] | None = None
    #: corridor: ``{symbol: (below, above)}``.
    bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    #: False: built from the legacy ``rebalancing_freq`` / ``threshold`` keys (the run's files stay as they were).
    declared: bool = True

    @property
    def rebalancing_freq(self) -> Any:
        """The frequency as :func:`quantbox.frequency.rebalancing_dates` reads it (a name resolved)."""
        if isinstance(self.frequency, str) and self.frequency in FREQUENCY_NAMES:
            return FREQUENCY_NAMES[self.frequency]
        return self.frequency

    def corridor_arrays(self, columns: pd.Index) -> tuple[np.ndarray, np.ndarray]:
        """``(below, above)`` per column: ``bounds`` where given, else ``width``."""
        assert self.width is not None
        below = np.array([self.bounds.get(str(c), self.width)[0] for c in columns], dtype=float)
        above = np.array([self.bounds.get(str(c), self.width)[1] for c in columns], dtype=float)
        return below, above

    def record(self) -> dict[str, Any]:
        """JSON-safe: what ``data_validation.json``, ``quantbox config explain`` and the run notes record."""
        freq = self.frequency
        if isinstance(freq, (list, tuple, pd.Index)):
            freq = [pd.Timestamp(d).isoformat() for d in freq]
        elif isinstance(freq, pd.DateOffset):
            freq = freq.freqstr
        out: dict[str, Any] = {"policy": self.policy, "frequency": freq, "calendar": self.calendar}
        if self.policy == "tranche":
            out["tranches"] = self.tranches
        if self.policy == "band":
            out["band"] = self.band
        if self.policy == "corridor":
            out["width"] = list(self.width or ())
            out["bounds"] = {k: list(v) for k, v in self.bounds.items()}
        return out


def _check_frequency(freq: Any) -> None:
    from quantbox.frequency import parse_rebalance_offset

    if freq is None or (isinstance(freq, str) and freq in FREQUENCY_NAMES):
        return
    if isinstance(freq, bool):
        raise ValueError("rebalancing_policy.frequency: a bool is not a schedule")
    if isinstance(freq, (int, np.integer)):
        if freq < 1:
            raise ValueError(f"rebalancing_policy.frequency: an int must be >= 1 (every n-th bar), got {freq}")
        return
    if isinstance(freq, (str, pd.DateOffset)):
        parse_rebalance_offset(freq)  # raises on "1m" (ambiguous) and on an unparseable offset
        return
    if isinstance(freq, (list, tuple, pd.Index)):
        pd.DatetimeIndex(freq)
        return
    raise ValueError(
        f"rebalancing_policy.frequency: expected {sorted(FREQUENCY_NAMES)}, an int, an offset, a list of dates or "
        f"null, got {freq!r}"
    )


def _check_calendar(calendar: Any) -> str | None:
    if calendar is None:
        return None
    from quantbox.frequency import _ALWAYS_OPEN_CALENDARS

    if not isinstance(calendar, str) or not calendar:
        raise ValueError(f"rebalancing_policy.calendar must be a market calendar name, got {calendar!r}")
    if calendar in _ALWAYS_OPEN_CALENDARS:
        return calendar
    import pandas_market_calendars as mcal

    if calendar not in mcal.get_calendar_names():
        raise ValueError(
            f"rebalancing_policy.calendar {calendar!r} is not a pandas-market-calendars name (NYSE, XLON, ...)"
        )
    return calendar


def _pair(value: Any, what: str) -> tuple[float, float]:
    """``(below, above)`` from a number or a 2-list, both >= 0."""
    if isinstance(value, bool):
        raise ValueError(f"{what} must be a number or [below, above], got {value!r}")
    if isinstance(value, (int, float, np.number)):
        pair = (float(value), float(value))
    elif isinstance(value, (list, tuple)) and len(value) == 2 and not any(isinstance(v, bool) for v in value):
        try:
            pair = (float(value[0]), float(value[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{what} must be a number or [below, above], got {value!r}") from exc
    else:
        raise ValueError(f"{what} must be a number or [below, above], got {value!r}")
    if not (pair[0] >= 0 and pair[1] >= 0):
        raise ValueError(f"{what} must be >= 0 on both sides (absolute weight), got {value!r}")
    return pair


def resolve_policy(spec: Mapping[str, Any] | RebalancePolicy) -> RebalancePolicy:
    """A declared ``rebalancing_policy`` -> :class:`RebalancePolicy`, refused (``ValueError``) when malformed."""
    if isinstance(spec, RebalancePolicy):
        return spec
    if not isinstance(spec, Mapping):
        raise ValueError(f"rebalancing_policy must be a mapping with a `policy` key, got {spec!r}")
    name = spec.get("policy")
    if name not in POLICIES:
        raise ValueError(f"rebalancing_policy.policy must be one of {list(POLICIES)}, got {name!r}")
    extra = sorted(set(spec) - set(POLICY_KEYS[name]))
    if extra:
        raise ValueError(f"rebalancing_policy {name!r} does not take {extra}; it takes {list(POLICY_KEYS[name])}")
    freq = spec.get("frequency", 1)
    _check_frequency(freq)
    calendar = _check_calendar(spec.get("calendar"))
    kw: dict[str, Any] = {"policy": name, "frequency": freq, "calendar": calendar}
    if name == "tranche":
        n = spec.get("tranches")
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError(f"rebalancing_policy tranche needs tranches: an int >= 2 (1 is periodic), got {n!r}")
        kw["tranches"] = int(n)
    if name == "band":
        band = spec.get("band")
        if isinstance(band, bool) or not isinstance(band, (int, float, np.number)) or not band > 0:
            raise ValueError(f"rebalancing_policy band needs band: a number > 0 (absolute weight drift), got {band!r}")
        kw["band"] = float(band)
    if name == "corridor":
        if spec.get("width") is None:
            raise ValueError("rebalancing_policy corridor needs width: [below, above] (or one number), absolute weight")
        kw["width"] = _pair(spec["width"], "rebalancing_policy.width")
        bounds = spec.get("bounds") or {}
        if not isinstance(bounds, Mapping):
            raise ValueError(f"rebalancing_policy.bounds must be a mapping {{symbol: [below, above]}}, got {bounds!r}")
        kw["bounds"] = {str(k): _pair(v, f"rebalancing_policy.bounds[{k!r}]") for k, v in bounds.items()}
    return RebalancePolicy(**kw)


def legacy_policy(rebalancing_freq: Any, threshold: float | None) -> RebalancePolicy:
    """``rebalancing_freq`` + ``threshold`` (the pre-policy keys) as the policy they always were: periodic or band."""
    _check_frequency(rebalancing_freq)
    if threshold is None:
        return RebalancePolicy(policy="periodic", frequency=rebalancing_freq, declared=False)
    if isinstance(threshold, bool) or not float(threshold) >= 0:
        raise ValueError(f"threshold must be a number >= 0 (absolute weight drift), got {threshold!r}")
    return RebalancePolicy(policy="band", frequency=rebalancing_freq, band=float(threshold), declared=False)


def schedule_policy(
    policy: Mapping[str, Any] | RebalancePolicy | None,
    rebalancing_freq: Any = 1,
    threshold: float | None = None,
) -> RebalancePolicy:
    """THE policy of a call: a declared one, or the legacy keys. Both together are refused (one schedule)."""
    if policy is None:
        return legacy_policy(rebalancing_freq, threshold)
    if threshold is not None or not (isinstance(rebalancing_freq, int) and rebalancing_freq == 1):
        raise ValueError(
            "declare the schedule ONCE: rebalancing_policy, or the legacy rebalancing_freq / threshold, not both "
            "(rebalancing_policy.frequency and band replace them)"
        )
    return resolve_policy(policy)


def market_sessions(index: pd.Index, calendar: str) -> np.ndarray:
    """Boolean per bar of *index*: True when the bar's DATE is a session of the market *calendar*."""
    from quantbox.frequency import _ALWAYS_OPEN_CALENDARS

    idx = pd.DatetimeIndex(index)
    if calendar in _ALWAYS_OPEN_CALENDARS or not len(idx):
        return np.ones(len(idx), dtype=bool)
    import pandas_market_calendars as mcal

    naive = idx.tz_convert(None) if idx.tz is not None else idx
    days = mcal.get_calendar(calendar).valid_days(start_date=naive.min().normalize(), end_date=naive.max())
    days = pd.DatetimeIndex(days).tz_convert(None).normalize() if days.tz is not None else days.normalize()
    return np.asarray(naive.normalize().isin(days), dtype=bool)


def policy_execution_bars(exec_bars: pd.Series, policy: RebalancePolicy) -> pd.Series:
    """The execution bars a policy decides and fills on: *exec_bars*, narrowed to its market's sessions."""
    if policy.calendar is None:
        return exec_bars.astype(bool)
    return exec_bars.astype(bool) & pd.Series(market_sessions(exec_bars.index, policy.calendar), index=exec_bars.index)


def blend_tranches(targets: np.ndarray, n: int) -> np.ndarray:
    """Row i -> the mean of rows i, i-1, ..., i-n+1 (rows before 0 are row 0): N tranches, one refreshed per row."""
    if n <= 1 or not len(targets):
        return targets
    rows = np.arange(len(targets))
    out = np.zeros_like(targets, dtype=float)
    for j in range(n):
        out += targets[np.maximum(rows - j, 0)]
    return out / n


def apply_drift_trigger(
    policy: RebalancePolicy,
    target_cells: np.ndarray,
    orders: np.ndarray,
    prices: np.ndarray,
    columns: pd.Index,
) -> dict[str, Any]:
    """``band`` / ``corridor``: keep a bar's orders only on a HIT, in place on *orders*.

    The held book is tracked cost-free: after a bar with orders, each ordered
    cell holds its target and each untouched cell its drifted weight; between
    such bars every weight drifts with its price against a cash remainder of
    ``1 - sum(weights)``. A bar is a hit when one ordered cell is outside its
    trigger: for ``band``, ``|drifted - target|`` above the band (vectorbt's
    band rule; today's ``threshold``); for ``corridor``, the drifted weight
    outside the cell's own ``[target - below, target + above]``, or an exit
    to 0 from a held weight. On a hit EVERY ordered cell trades to target
    (TOM-1513: a corridor hit rebalances the whole book); otherwise the bar's
    orders are all dropped. A cell with no order cannot trigger.
    """
    n_inst = orders.shape[1]
    held = np.zeros(n_inst)
    last: int | None = None
    order_rows = np.flatnonzero(orders.any(axis=1))
    skipped: list[int] = []
    partial = 0
    if policy.policy == "corridor":
        below, above = policy.corridor_arrays(columns)
    for r in order_rows:
        o = orders[r].copy()
        if last is None:
            drifted = held.copy()
        else:
            growth = prices[r] / prices[last]
            value = held * np.where(np.isfinite(growth), growth, 1.0)
            total = (1.0 - held.sum()) + value.sum()
            drifted = value / total if total != 0 else value
        target = target_cells[r]
        if policy.policy == "band":
            hit = np.abs(drifted - target) > float(policy.band or 0.0)
        else:
            hit = (drifted < target - below) | (drifted > target + above) | ((target == 0) & (drifted != 0))
        keep = o if (hit & o).any() else np.zeros_like(o)
        if keep.any():
            if not keep[o].all():
                partial += 1
            held = drifted
            held[keep] = target[keep]
            last = int(r)
        else:
            skipped.append(int(r))
        orders[r] = keep
    return {
        "scheduled_rebalances": int(len(order_rows)),
        "placed_rebalances": int(len(order_rows) - len(skipped)),
        "skipped_rebalances": len(skipped),
        "partial_rebalances": int(partial),
        "first_skipped_rows": skipped[:5],
    }
