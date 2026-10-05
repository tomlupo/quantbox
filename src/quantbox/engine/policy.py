"""Rebalancing policies — WHEN the seam places orders, and on which cells (TOM-1450 3d, TOM-1513, docs/adr/0008).

A policy is part of the seam's schedule (:mod:`quantbox.engine.schedule`): it
produces the per-cell ORDERS mask (and, for ``tranche``, the targets) that
every engine executes. No adapter knows which policy ran.

A policy is a CADENCE times a TRIGGER (TOM-1513):

- the cadence sets the targets on every considered bar: ``periodic`` (the
  decided row) or ``tranche`` (the mean of N staggered tranches);
- the trigger decides whether a considered bar trades: ``none`` (always),
  ``band`` or ``corridor`` (only on a hit, and then the whole book).

``{cadence: tranche, tranches: 5, frequency: daily, trigger: corridor, width:
0.02}`` is robo's "tranches 5 + 2% corridor" as one config. The single-key
spellings of TOM-1450 keep working and map to this form (:data:`SINGLE_KEY`):
``periodic`` and ``tranche`` have no trigger, ``band`` and ``corridor`` have
the periodic cadence.

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

The cadences and the triggers (named here by their single-key spellings):

``periodic``
    Trade every instrument to its target on every considered bar. Today's
    ``rebalancing_freq`` is this policy.
``tranche``
    Stagger the book across ``tranches`` = N tranches. Tranche k is refreshed
    to the decided weights on every N-th considered bar (offset k) and holds
    them for N decisions; the book is their mean, traded on every considered
    bar. Every tranche starts at the first decision. A tranche holds its
    TARGET weights between refreshes — it is not a separate sub-account that
    drifts. ``risk.tranches: N`` of ``backtest.pipeline.v1`` is a deprecated
    alias of this cadence (TOM-1513: one tranche concept).
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

#: The single-key spellings (TOM-1450), in the order the docs list them.
POLICIES = ("periodic", "tranche", "band", "corridor")
#: How the targets are set on a considered bar.
CADENCES = ("periodic", "tranche")
#: Whether a considered bar trades: always (none), or only on a hit (band, corridor), then the whole book.
TRIGGERS = ("none", "band", "corridor")
#: Each single-key spelling as (cadence, trigger).
SINGLE_KEY: dict[str, tuple[str, str]] = {
    "periodic": ("periodic", "none"),
    "tranche": ("tranche", "none"),
    "band": ("periodic", "band"),
    "corridor": ("periodic", "corridor"),
}

#: Frequency names -> the ``rebalancing_freq`` they mean. Period-END offsets: the last bar of the period.
FREQUENCY_NAMES: dict[str, Any] = {
    "daily": 1,
    "weekly": "W-SUN",
    "monthly": "ME",
    "quarterly": "QE",
    "yearly": "YE",
}

_COMMON = ("frequency", "calendar", "min_trade")
#: The keys each cadence and each trigger takes, on top of the common ones; any other key is refused.
CADENCE_KEYS: dict[str, tuple[str, ...]] = {"periodic": (), "tranche": ("tranches",)}
TRIGGER_KEYS: dict[str, tuple[str, ...]] = {"none": (), "band": ("band",), "corridor": ("width", "bounds")}
#: The keys of a cadence or trigger that must be declared.
_REQUIRED = {"tranches", "band", "width"}
_SPECIFIC = ("tranches", "band", "width", "bounds")


def _keys(cadence: str, trigger: str) -> tuple[str, ...]:
    return (*_COMMON, *CADENCE_KEYS[cadence], *TRIGGER_KEYS[trigger])


#: The keys each single-key spelling takes; any other key is refused.
POLICY_KEYS: dict[str, tuple[str, ...]] = {name: ("policy", *_keys(*ct)) for name, ct in SINGLE_KEY.items()}

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
        "TOM-1450, TOM-1513). A policy is a cadence x a trigger: {cadence: periodic | tranche, trigger: none | "
        "band | corridor}, e.g. {cadence: tranche, tranches: 5, frequency: daily, trigger: corridor, width: 0.02}. "
        "Cadence periodic: the targets are the decided row on every considered bar. Cadence tranche: the "
        "targets are the mean of N staggered tranches, each refreshed every N-th decision. Trigger none: every "
        "considered bar trades. Trigger band: the whole book trades when a held weight drifted more than `band` "
        "from target. Trigger corridor: the whole book trades when a held weight left its own [target - below, "
        "target + above] corridor. The single-key `policy` (periodic | tranche | band | corridor) is the same "
        "thing spelled once: declare `policy` or `cadence`, not both. Replaces rebalancing_freq + threshold "
        "(declare one or the other, not both)."
    ),
    "oneOf": [{"required": ["policy"]}, {"required": ["cadence"]}],
    "additionalProperties": False,
    "properties": {
        "policy": {
            "type": "string",
            "enum": list(POLICIES),
            "description": (
                "The single-key spelling: periodic | tranche | band | corridor = cadence periodic | tranche, or "
                "cadence periodic with trigger band | corridor. Not with cadence / trigger."
            ),
        },
        "cadence": {
            "type": "string",
            "enum": list(CADENCES),
            "description": "periodic: the decided row on every considered bar | tranche: the mean of `tranches` tranches.",
        },
        "trigger": {
            "type": ["string", "null"],
            "enum": [*TRIGGERS, None],
            "default": "none",
            "description": (
                "none (null): every considered bar trades | band: a held weight more than `band` from target | "
                "corridor: a held weight outside its [target - below, target + above]. On a hit the WHOLE book "
                "trades to target."
            ),
        },
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
        "min_trade": {
            "type": "number",
            "minimum": 0,
            "default": 0,
            "description": (
                "Every policy, optional: on a rebalance, a trade with |target - held| below min_trade (absolute "
                "weight) is dropped; sells always execute; when the buys exceed cash + sell proceeds they are "
                "scaled down proportionally, so without venue.leverage: borrow the held net never goes above 1. "
                "A trade under min_trade cannot trigger a band or corridor. 0 (the default) is off."
            ),
        },
        "tranches": {
            "type": "integer",
            "minimum": 2,
            "description": "cadence tranche only, required: N tranches; each is refreshed on every N-th decision.",
        },
        "band": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "trigger band only, required: absolute weight drift that triggers a whole-book rebalance.",
        },
        "width": {
            "type": ["number", "array"],
            "items": {"type": "number", "minimum": 0},
            "minItems": 2,
            "maxItems": 2,
            "minimum": 0,
            "description": (
                "trigger corridor only, required: the default corridor [below, above] around each target, "
                "absolute weight (one number = symmetric)."
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
            "description": (
                "trigger corridor only: per-instrument corridors {symbol: [below, above] | width}, over `width`."
            ),
        },
    },
    "allOf": [
        # The single-key spelling: its own keys, never cadence / trigger.
        {"if": {"required": ["policy"]}, "then": {"properties": {"cadence": False, "trigger": False}}},
        *(
            {
                "if": {"required": ["policy"], "properties": {"policy": {"const": name}}},
                "then": {
                    "properties": {k: False for k in _SPECIFIC if k not in keys},
                    **({"required": sorted(_REQUIRED & set(keys))} if _REQUIRED & set(keys) else {}),
                },
            }
            for name, keys in POLICY_KEYS.items()
        ),
        # cadence x trigger: each half takes its own keys.
        *(
            {
                "if": {"required": ["cadence"], "properties": {"cadence": {"const": cadence}}},
                "then": {
                    "properties": {k: False for k in CADENCE_KEYS["tranche"] if k not in keys},
                    **({"required": sorted(_REQUIRED & set(keys))} if _REQUIRED & set(keys) else {}),
                },
            }
            for cadence, keys in CADENCE_KEYS.items()
        ),
        *(
            {
                "if": {
                    "required": ["cadence"],
                    **(
                        {"properties": {"trigger": {"const": trigger}}, "required": ["cadence", "trigger"]}
                        if trigger != "none"
                        else {
                            "not": {"required": ["trigger"], "properties": {"trigger": {"enum": ["band", "corridor"]}}}
                        }
                    ),
                },
                "then": {
                    "properties": {
                        k: False for k in (*TRIGGER_KEYS["band"], *TRIGGER_KEYS["corridor"]) if k not in keys
                    },
                    **({"required": sorted(_REQUIRED & set(keys))} if _REQUIRED & set(keys) else {}),
                },
            }
            for trigger, keys in TRIGGER_KEYS.items()
        ),
    ],
}


@dataclass(frozen=True)
class RebalancePolicy:
    """A resolved rebalancing policy (see the module docstring); build it with :func:`resolve_policy`."""

    cadence: str = "periodic"
    trigger: str = "none"
    #: As declared: a frequency name or a ``rebalancing_freq`` form.
    frequency: Any = 1
    calendar: str | None = None
    tranches: int = 1
    band: float | None = None
    #: corridor: the default ``(below, above)``.
    width: tuple[float, float] | None = None
    #: corridor: ``{symbol: (below, above)}``.
    bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    #: Every policy: the smallest trade (absolute weight) a rebalance places; 0 = off (:func:`place_orders`).
    min_trade: float = 0.0
    #: False: built from the legacy ``rebalancing_freq`` / ``threshold`` keys (the run's files stay as they were).
    declared: bool = True

    @property
    def policy(self) -> str:
        """The single-key name of this cadence x trigger, or ``"<cadence>+<trigger>"`` when it has none."""
        for name, ct in SINGLE_KEY.items():
            if ct == (self.cadence, self.trigger):
                return name
        return f"{self.cadence}+{self.trigger}"

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
        out: dict[str, Any] = {
            "policy": self.policy,
            "cadence": self.cadence,
            "trigger": self.trigger,
            "frequency": freq,
            "calendar": self.calendar,
        }
        if self.cadence == "tranche":
            out["tranches"] = self.tranches
        if self.trigger == "band":
            out["band"] = self.band
        if self.trigger == "corridor":
            out["width"] = list(self.width or ())
            out["bounds"] = {k: list(v) for k, v in self.bounds.items()}
        if self.min_trade > 0:  # off by default: a run without it records what it always did
            out["min_trade"] = self.min_trade
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
        raise ValueError(f"rebalancing_policy must be a mapping with a `cadence` (or `policy`) key, got {spec!r}")
    if "policy" in spec:
        name = spec.get("policy")
        if "cadence" in spec or "trigger" in spec:
            raise ValueError(
                "rebalancing_policy: declare `policy` (the single-key spelling) or `cadence` / `trigger`, not both"
            )
        if name not in POLICIES:
            raise ValueError(f"rebalancing_policy.policy must be one of {list(POLICIES)}, got {name!r}")
        cadence, trigger = SINGLE_KEY[name]
        allowed, what = POLICY_KEYS[name], repr(name)
    else:
        if "cadence" not in spec:
            raise ValueError(
                "rebalancing_policy needs `cadence` (periodic | tranche, with an optional `trigger`) or the "
                f"single-key `policy` ({' | '.join(POLICIES)}), got {dict(spec)!r}"
            )
        cadence = spec.get("cadence")
        trigger = "none" if spec.get("trigger") is None else spec.get("trigger")
        if cadence not in CADENCES:
            raise ValueError(f"rebalancing_policy.cadence must be one of {list(CADENCES)}, got {cadence!r}")
        if trigger not in TRIGGERS:
            raise ValueError(f"rebalancing_policy.trigger must be one of {list(TRIGGERS)} (or null), got {trigger!r}")
        allowed, what = ("cadence", "trigger", *_keys(cadence, trigger)), f"cadence {cadence!r} x trigger {trigger!r}"
    extra = sorted(set(spec) - set(allowed))
    if extra:
        raise ValueError(f"rebalancing_policy {what} does not take {extra}; it takes {list(allowed)}")
    freq = spec.get("frequency", 1)
    _check_frequency(freq)
    calendar = _check_calendar(spec.get("calendar"))
    kw: dict[str, Any] = {"cadence": cadence, "trigger": trigger, "frequency": freq, "calendar": calendar}
    min_trade = spec.get("min_trade", 0)
    if isinstance(min_trade, bool) or not isinstance(min_trade, (int, float, np.number)) or not min_trade >= 0:
        raise ValueError(f"rebalancing_policy.min_trade must be a number >= 0 (absolute weight), got {min_trade!r}")
    kw["min_trade"] = float(min_trade)
    if cadence == "tranche":
        n = spec.get("tranches")
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError(f"rebalancing_policy tranche needs tranches: an int >= 2 (1 is periodic), got {n!r}")
        kw["tranches"] = int(n)
    if trigger == "band":
        band = spec.get("band")
        if isinstance(band, bool) or not isinstance(band, (int, float, np.number)) or not band > 0:
            raise ValueError(f"rebalancing_policy band needs band: a number > 0 (absolute weight drift), got {band!r}")
        kw["band"] = float(band)
    if trigger == "corridor":
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
        return RebalancePolicy(frequency=rebalancing_freq, declared=False)
    if isinstance(threshold, bool) or not float(threshold) >= 0:
        raise ValueError(f"threshold must be a number >= 0 (absolute weight drift), got {threshold!r}")
    return RebalancePolicy(trigger="band", frequency=rebalancing_freq, band=float(threshold), declared=False)


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


#: Buys within this much of the available cash are not scaled (float noise, not a shortfall).
_CASH_TOLERANCE = 1e-12


def place_orders(
    policy: RebalancePolicy,
    target_cells: np.ndarray,
    orders: np.ndarray,
    prices: np.ndarray,
    columns: pd.Index,
    leverage: str,
) -> dict[str, Any]:
    """The trigger and ``min_trade``, against the held book: in place on *orders* and *target_cells*.

    The seam runs this when the policy has a trigger or a ``min_trade`` above
    0; otherwise every considered order stands as scheduled.

    The held book is tracked cost-free: after a bar with orders, each ordered
    cell holds its target and each untouched cell its drifted weight; between
    such bars every weight drifts with its price against a cash remainder of
    ``1 - sum(weights)``.

    **Trigger.** A bar is a hit when one ordered cell is outside its trigger:
    for ``band``, ``|drifted - target|`` above the band (vectorbt's band rule;
    today's ``threshold``); for ``corridor``, the drifted weight outside the
    cell's own ``[target - below, target + above]``, or an exit to 0 from a
    held weight. On a hit EVERY ordered cell trades to target (TOM-1513: a
    corridor hit rebalances the whole book); otherwise the bar's orders are
    all dropped. A cell with no order cannot trigger. Without a trigger every
    considered bar is placed.

    **min_trade** (TOM-1513), on a placed bar, in one deterministic pass:

    1. ``delta = target - held`` for every ordered cell;
    2. a cell with ``|delta| < min_trade`` does not trade (it keeps its held
       weight); such a cell cannot trigger either, so a dust exit never
       rebalances the book;
    3. sells always execute; when the buys exceed the cash plus the sell
       proceeds, every buy is scaled down by the same factor. Without
       ``venue.leverage: borrow`` the held net therefore never goes above 1.

    A bar on which every trade is under ``min_trade`` is skipped.
    """
    n_inst = orders.shape[1]
    held = np.zeros(n_inst)
    last: int | None = None
    order_rows = np.flatnonzero(orders.any(axis=1))
    skipped: list[int] = []
    partial = 0
    dropped = 0
    scales: list[float] = []
    min_trade = float(policy.min_trade)
    cap_buys = min_trade > 0 and leverage != "borrow"
    if policy.trigger == "corridor":
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
        target = target_cells[r]  # a view: a scaled buy is written into target_cells
        trades = o & (np.abs(target - drifted) >= min_trade) if min_trade > 0 else o
        if policy.trigger == "none":
            placed = True
        else:
            if policy.trigger == "band":
                hit = np.abs(drifted - target) > float(policy.band or 0.0)
            else:
                hit = (drifted < target - below) | (drifted > target + above) | ((target == 0) & (drifted != 0))
            placed = bool((hit & trades).any())
        keep = trades if placed else np.zeros_like(o)
        if keep.any():
            if cap_buys:
                delta = np.where(keep, target - drifted, 0.0)
                buys = delta > 0
                need = float(delta[buys].sum())
                room = max((1.0 - float(drifted.sum())) - float(delta[delta < 0].sum()), 0.0)
                if need > room + _CASH_TOLERANCE:
                    scale = room / need
                    target[buys] = drifted[buys] + delta[buys] * scale
                    scales.append(scale)
            if not keep[o].all():
                partial += 1
                dropped += int((o & ~keep).sum())
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
        "min_trade": {
            "dropped_trades": int(dropped),
            "scaled_rebalances": len(scales),
            "buy_scale_min": float(min(scales)) if scales else 1.0,
        },
    }
