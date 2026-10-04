"""Frequency value object — single source of truth for bar cadence + calendar.

Resolves the historical sprawl of `prices.frequency` (ccxt string), `rebalancing_freq`
(int/pandas-offset/list), `trading_days` (int, default 365), and `annualize` (float,
default 252) by deriving everything from one `bar_size + calendar` pair, exchange-aware
via `pandas-market-calendars`.

Used internally by `backtest.pipeline.v1` and `trade.full_pipeline.v1` (both via
`resolve_pipeline_frequency`) to:
  - derive default `trading_days` from `frequency.bars_per_year()` (backtest)
  - inject `_pipeline_annualize` into each strategy's params so strategies don't
    need their own (potentially drifting) defaults — identically in backtest and
    paper/live (TOM-1338)

Strategies that need annualization should declare `annualize: float | None = None`
and consume it via `params.get("_pipeline_annualize", 252.0)` as a fallback —
explicit per-strategy values still win, and a drift warning fires in the pipeline
when they disagree with the derived value.

Example
-------
::

    from quantbox.frequency import Frequency

    f = Frequency.parse({"bar_size": "1d", "calendar": "NYSE"})
    f.bars_per_year()        # → ~250.0 (US equity trading days)

    f = Frequency.parse({"bar_size": "1h", "calendar": "24/7"})
    f.bars_per_year()        # → 8760.0 (crypto)

    f = Frequency.parse("4h")  # shorthand; calendar defaults to "24/7"
    f.bars_per_year()        # → 2190.0
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

# Calendars where every day trades 24h — no exchange holidays apply.
_ALWAYS_OPEN_CALENDARS = frozenset({"24/7", "always_open", "ALWAYS_OPEN"})

# Reference year used for `bars_per_year` calendar lookups. 2023 = non-leap, no
# unusual US holiday closures, broadly representative across calendars.
_REFERENCE_YEAR_START = "2023-01-01"
_REFERENCE_YEAR_END = "2023-12-31"

# Semantic spellings of a bar size → ccxt interval. The ONE table: the data
# sources' `normalize_data_frequency` and `_parse_bar_size` both read it, so a
# `prices.frequency` that loads data also resolves an annualisation (TOM-1338).
FREQUENCY_ALIASES: dict[str, str] = {
    "daily": "1d",
    "day": "1d",
    "d": "1d",
    "1day": "1d",
    "hourly": "1h",
    "hour": "1h",
    "h": "1h",
    "1hour": "1h",
    "4hourly": "4h",
    "4hour": "4h",
    "weekly": "1w",
    "week": "1w",
    "w": "1w",
    "monthly": "1M",
    "month": "1M",
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
}


@dataclass(frozen=True, slots=True)
class Frequency:
    """Bar cadence + calendar — single source of truth for frequency-derived values."""

    bar_size: pd.Timedelta
    calendar: str = "24/7"

    @classmethod
    def parse(cls, spec: str | dict | Frequency) -> Frequency:
        """Parse user input into a Frequency.

        Accepts:
          - Frequency instance (returned unchanged)
          - dict ``{'bar_size': '1h', 'calendar': 'NYSE'}``
          - str ``'1d'``, ``'1h'``, ``'4h'``, ``'1M'`` — calendar defaults to ``'24/7'``
        """
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, dict):
            bar = _parse_bar_size(spec.get("bar_size", "1d"))
            cal = spec.get("calendar", "24/7")
            return cls(bar_size=bar, calendar=cal)
        if isinstance(spec, str):
            return cls(bar_size=_parse_bar_size(spec), calendar="24/7")
        raise TypeError(f"Frequency.parse: unsupported type {type(spec).__name__}")

    def bars_per_year(self) -> float:
        """Exchange-aware bars per year.

        For 24/7 calendars: 365 * bars_per_day.
        For exchange calendars: trading_days * bars_per_session (intraday) or
            trading_days / days_per_bar (daily+).
        """
        # 24/7 markets: every calendar day trades, full 24h
        if self.calendar in _ALWAYS_OPEN_CALENDARS:
            bars_per_day = pd.Timedelta(days=1) / self.bar_size
            return 365.0 * float(bars_per_day)

        # Exchange calendars: use pandas-market-calendars for trading-day count
        # + session length. Cache the calendar lookup since it's pure.
        trading_days, mean_session = _calendar_reference(self.calendar)

        if self.bar_size >= pd.Timedelta(days=1):
            # Multi-day bars: divide trading days by bar size in days
            days_per_bar = self.bar_size / pd.Timedelta(days=1)
            return float(trading_days / days_per_bar)

        # Intraday bars: trading_days × bars per session
        bars_per_session = mean_session / self.bar_size
        return float(trading_days * bars_per_session)

    def valid_dates(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
        """Trading dates in the window per the selected calendar."""
        if self.calendar in _ALWAYS_OPEN_CALENDARS:
            return pd.date_range(start=start, end=end, freq="D")
        return mcal.get_calendar(self.calendar).valid_days(start_date=start, end_date=end)

    def to_pandas_offset(self) -> pd.DateOffset:
        """Pandas DateOffset equivalent of ``bar_size``."""
        return pd.tseries.frequencies.to_offset(self.bar_size)

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable dict form."""
        return {"bar_size": _bar_size_to_str(self.bar_size), "calendar": self.calendar}

    def __str__(self) -> str:
        return f"Frequency(bar_size={_bar_size_to_str(self.bar_size)!r}, calendar={self.calendar!r})"


def resolve_pipeline_frequency(params: dict[str, Any], prices_params: dict[str, Any]) -> Frequency:
    """Resolve a pipeline run's `Frequency` from its params — the ONE resolver.

    Both ``backtest.pipeline.v1`` and ``trade.full_pipeline.v1`` call this, and
    inject its ``bars_per_year()`` into every strategy's params as
    ``_pipeline_annualize``, so a strategy is annualised identically in its
    backtest and in paper/live (TOM-1338).

    Accepts (in priority order):
      1. ``params['frequency']`` — full spec, str or dict
         - dict: ``{'bar_size': '1h', 'calendar': 'NYSE'}``
         - str: ``'1h'`` (calendar defaults to '24/7')
      2. ``prices.frequency`` + optional ``params['market_calendar']`` shorthand
      3. Default: ``Frequency('1d', '24/7')`` — 365 bars/yr, crypto-friendly
    """
    explicit = params.get("frequency")
    if explicit is not None:
        return Frequency.parse(explicit)

    bar_size = prices_params.get("frequency", "1d")
    calendar = params.get("market_calendar", "24/7")
    return Frequency.parse({"bar_size": bar_size, "calendar": calendar})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_rebalance_offset(spec: str | pd.DateOffset) -> pd.DateOffset:
    """Strict parser for rebalance-frequency strings (e.g. ``"1D"``, ``"1W"``, ``"1M"``).

    Two reasons this exists separately from `Frequency.parse`:

    1. The historical `get_rebalancing_dates` parser in vectorbt_engine treats
       lowercase ``"1m"`` as MONTHS, while ccxt and pandas's own
       ``to_offset("1m")`` both treat it as MINUTES. This silent disagreement is
       the single most dangerous footgun in the codebase. This parser **rejects
       ``"1m"`` (lowercase) explicitly** with a clear error pointing at the two
       unambiguous spellings (``"1M"`` for months, ``"1min"`` for minutes).
    2. Rebalance specs are conceptually different from data bar size — they're
       always pandas DateOffsets, never Timedeltas in disguise (you can't have
       a 30-day rebalance "month" if your goal is to rebalance on calendar
       month-ends with day-of-month variation).

    Accepts:
      - ``pd.DateOffset`` → returned unchanged
      - str → routed through ``pd.tseries.frequencies.to_offset()``

    Raises:
      - ``ValueError`` on ambiguous ``"1m"`` / ``"5m"`` etc with a clear suggestion
      - ``ValueError`` on unparseable strings (propagated from pandas)

    For int / list / None forms of `rebalancing_freq`, callers should handle
    those directly — this parser is only for the string→offset path.
    """
    if isinstance(spec, pd.DateOffset):
        return spec
    if not isinstance(spec, str) or not spec:
        raise TypeError(f"parse_rebalance_offset: expected str|DateOffset, got {type(spec).__name__}")

    # The "1m" ambiguity: pandas means minutes, the historical quantbox
    # vectorbt_engine parser means months. Reject both lowercase 'm' suffixes
    # to force the caller to disambiguate.
    if spec.endswith("m") and not spec.endswith(("min", "ms")):
        raise ValueError(
            f"parse_rebalance_offset: {spec!r} is ambiguous — "
            f"lowercase 'm' historically meant MONTHS in this codebase but pandas "
            f"interprets it as MINUTES. Use {spec[:-1] + 'M'!r} for months or "
            f"{spec[:-1] + 'min'!r} for minutes to disambiguate."
        )

    return pd.tseries.frequencies.to_offset(spec)


def rebalancing_dates(dates: pd.Index, rebalancing_freq: Any) -> pd.DatetimeIndex:
    """The BARS a rebalancing schedule trades on — every date is a member of *dates*.

    ``None`` → buy-and-hold (the first bar); ``int`` n → every n-th bar; ``str``
    / ``pd.DateOffset`` → calendar dates from :func:`parse_rebalance_offset`;
    ``list`` → explicit dates. A calendar or explicit date that is not a bar
    (a weekend month-end, a holiday ``BMS``) is snapped FORWARD to the first bar
    on or after it — the first moment an order could trade — and dates past the
    last bar are dropped. Until TOM-1429 such a date was simply missing from the
    engine's mask, so that period was never rebalanced (``"1W"`` = Sundays never
    traded on weekday data at all). Engine-free, so every engine reads the same
    schedule.
    """
    dates = pd.DatetimeIndex(dates)
    if rebalancing_freq is None:
        return pd.DatetimeIndex(dates[:1])
    if isinstance(rebalancing_freq, bool):
        raise ValueError("rebalancing_freq: a bool is not a schedule")
    if isinstance(rebalancing_freq, int):
        return dates[::rebalancing_freq]
    if isinstance(rebalancing_freq, (str, pd.DateOffset)):
        offset = parse_rebalance_offset(rebalancing_freq)
        wanted = pd.date_range(start=dates[0], end=dates[-1], freq=offset) if len(dates) else pd.DatetimeIndex([])
    elif isinstance(rebalancing_freq, (list, tuple, pd.Index)):
        wanted = pd.DatetimeIndex(rebalancing_freq)
    else:
        raise ValueError(
            f"rebalancing_dates: unsupported rebalancing_freq type "
            f"{type(rebalancing_freq).__name__!r}; expected int|str|list|DateOffset|None"
        )
    if dates.tz is not None and wanted.tz is None:
        wanted = wanted.tz_localize(dates.tz)
    pos = dates.searchsorted(wanted, side="left")
    pos = np.unique(pos[pos < len(dates)])
    return dates[pos]


def _parse_bar_size(s: str | pd.Timedelta) -> pd.Timedelta:
    """Accept ccxt format (``'1d'``, ``'4h'``, ``'5m'``, ``'1M'``) and pandas-offset
    style. ``'1M'`` is treated as one calendar month (~30 days); for precise
    monthly bars use the exchange calendar via ``bars_per_year()``.
    """
    if isinstance(s, pd.Timedelta):
        return s
    if not isinstance(s, str) or not s:
        raise TypeError(f"_parse_bar_size: expected str|Timedelta, got {type(s).__name__}")
    # Semantic aliases ("daily", "hourly", ...) the data sources already accept for
    # `prices.frequency`. Looked up case-insensitively, but a canonical spec is
    # never case-folded: "1M" (month) and "1m" (minute) are different bars.
    s = FREQUENCY_ALIASES.get(s.strip().lower(), s)
    if s.endswith("M"):
        # Calendar month — pd.Timedelta has no months; approximate at 30 days.
        n = int(s[:-1]) if s[:-1] else 1
        return pd.Timedelta(days=30 * n)
    try:
        return pd.Timedelta(s)
    except ValueError as exc:
        raise ValueError(
            f"_parse_bar_size: cannot parse {s!r}. Use ccxt format ('1d','4h','5m','1M') "
            f"or a pandas Timedelta-compatible string."
        ) from exc


def _bar_size_to_str(td: pd.Timedelta) -> str:
    """Inverse of `_parse_bar_size`, best-effort."""
    if td % pd.Timedelta(days=1) == pd.Timedelta(0):
        n = td // pd.Timedelta(days=1)
        return f"{int(n)}d"
    if td % pd.Timedelta(hours=1) == pd.Timedelta(0):
        n = td // pd.Timedelta(hours=1)
        return f"{int(n)}h"
    if td % pd.Timedelta(minutes=1) == pd.Timedelta(0):
        n = td // pd.Timedelta(minutes=1)
        return f"{int(n)}m"
    return str(td)


@lru_cache(maxsize=32)
def _calendar_reference(calendar_name: str) -> tuple[int, pd.Timedelta]:
    """Return (trading_days_in_reference_year, mean_session_length) for a calendar.

    Cached — pandas-market-calendars schedule lookup is non-trivial.
    """
    cal = mcal.get_calendar(calendar_name)
    schedule = cal.schedule(start_date=_REFERENCE_YEAR_START, end_date=_REFERENCE_YEAR_END)
    if schedule.empty:
        raise ValueError(
            f"Calendar {calendar_name!r} produced empty schedule for {_REFERENCE_YEAR_START}..{_REFERENCE_YEAR_END}"
        )
    n_days = len(schedule)
    mean_session = (schedule["market_close"] - schedule["market_open"]).mean()
    return n_days, mean_session
