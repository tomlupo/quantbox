"""The ONE metrics module: every statistic that DESCRIBES a run is computed here.

Sharpe (and excess Sharpe vs a benchmark), information ratio, tracking error,
CAGR, volatility, max drawdown, top-N drawdowns, turnover, VaR / CVaR, and
per-bar returns compounded to periods (:func:`period_returns`, TOM-3698), and the
correlation with a reference series on a declared frequency (:func:`correlation`,
TOM-3699). Every
backtest path, the run metrics, the finding-report export, the validation
plugins and :mod:`quantbox.performance` (which quantbox-live imports) call
these functions instead of carrying their own copy (TOM-1448). The one
aggregate is :func:`compute_backtest_metrics`.

The descriptive research statistics live here too (TOM-1596): IC and ICIR,
beta to a benchmark, hit rate, active share and risk decomposition.

Metrics DESCRIBE; :mod:`quantbox.inference` TESTS (Newey-West, factor
regression, DSR, bootstrap) and :mod:`quantbox.gates` DECIDES (TOM-1618).
``newey_west_tstat``, ``newey_west_auto_lags``, ``hac_ols`` and
``require_finite`` moved to :mod:`quantbox.inference`; since 0.13.0 the old
names here raise ``ImportError`` naming that home (TOM-1457).

Core: numpy and pandas only (scipy is imported lazily for parametric VaR).
``quantbox.plugins.backtesting.metrics`` re-exports the same objects for the
callers that still import that path.

Conventions, stated once:

- ``bars_per_year`` (``trading_days`` in the older signatures) is the
  annualisation factor; a pipeline takes it from the run's ``Frequency``.
- Sharpe is ``mean(r - rf) / std(r - rf, ddof=1) * sqrt(bars_per_year)``, and
  0.0 when there are fewer than two returns or no variance.
- Excess Sharpe vs a benchmark series IS the information ratio: the Sharpe of
  ``r - benchmark``. :func:`information_ratio` computes it through
  :func:`sharpe_ratio`, so the two names can never disagree.
- Drawdowns are negative fractions. ``start_is_peak`` says whether the equity
  before the first return (1.0) counts as a peak. Every drawdown OUTPUT names
  its sign (TOM-1627): ``max_drawdown`` is signed (<= 0) and ``max_drawdown_abs``
  is the same drawdown as a positive depth (>= 0) — :func:`drawdown_fields`.
- Turnover is two-sided: ``sum(|w[t] - w[t-1]|)`` per bar.
- A research statistic with no answer (too few observations, no variance) is
  NaN, never 0.0 — the Sharpe's 0.0 convention above predates it and stays.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

# The framework's one "cancelled to floating-point noise" threshold and test;
# the measurement behind DEGENERATE_RTOL is in quantbox._numerics.
from ._numerics import DEGENERATE_RTOL  # noqa: F401 — public re-export, imported from here before TOM-1618
from ._numerics import flat as _flat
from ._removed import removed_names

__all__ = [
    "DEGENERATE_RTOL",
    "TRADING_DAYS_PER_YEAR",
    "Correlation",
    "Window",
    "active_share",
    "annual_turnover",
    "beta",
    "compute_backtest_metrics",
    "compute_cvar",
    "compute_drawdown_series",
    "compute_portfolio_cvar",
    "compute_portfolio_var",
    "compute_rolling_sharpe",
    "compute_var",
    "correlation",
    "drawdown_fields",
    "hit_rate",
    "icir",
    "information_coefficient",
    "information_ratio",
    "max_drawdown",
    "period_returns",
    "risk_contributions",
    "sharpe_ratio",
    "top_drawdowns",
    "tracking_error",
    "turnover_series",
]

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 365  # crypto default; callers can override

# Inference that lived here until TOM-1618 (newey_west_tstat, ...): removed in
# 0.13.0, each old name raises ImportError naming quantbox.inference (TOM-1457).
__getattr__ = removed_names(__name__)


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    bars_per_year: float,
    *,
    risk_free: float | pd.Series = 0.0,
) -> float:
    """Annualised Sharpe ratio of a per-bar return series.

    ``risk_free`` is either an ANNUAL rate (a float, converted to a per-bar rate
    by compounding) or a per-bar return SERIES (a cash leg or a benchmark),
    aligned on the returns' index. With a benchmark series this is the excess
    Sharpe vs that benchmark, which is the information ratio.
    """
    if not isinstance(returns, pd.Series) and not isinstance(risk_free, pd.Series):
        # A plain array (the validation plugins' hot loops): numpy, no index. A NaN
        # in the array gives a NaN Sharpe — a corrupted series is not reported as 0.
        x = np.asarray(returns, dtype=float).ravel()
        if risk_free:
            x = x - ((1 + risk_free) ** (1 / bars_per_year) - 1)
        if len(x) < 2:
            return 0.0
        std = float(np.std(x, ddof=1))
        return 0.0 if std == 0 else float(np.mean(x) / std * np.sqrt(bars_per_year))
    r = returns if isinstance(returns, pd.Series) else pd.Series(np.asarray(returns, dtype=float))
    if isinstance(risk_free, pd.Series):
        excess = r - risk_free.reindex(r.index).fillna(0.0)
    else:
        excess = r - ((1 + risk_free) ** (1 / bars_per_year) - 1)
    std = excess.std()  # pandas skips NaN
    return float(excess.mean() / std * np.sqrt(bars_per_year)) if std > 0 else 0.0


def tracking_error(returns: pd.Series | np.ndarray, benchmark: pd.Series | np.ndarray, bars_per_year: float) -> float:
    """Annualised standard deviation (ddof=1) of ``returns - benchmark``."""
    r, b = _aligned(returns, benchmark)
    active = r - b
    std = active.std() if isinstance(active, pd.Series) else np.std(active, ddof=1)
    return float(std * np.sqrt(bars_per_year))


def information_ratio(
    returns: pd.Series | np.ndarray, benchmark: pd.Series | np.ndarray, bars_per_year: float
) -> float:
    """Annualised mean active return over tracking error — the excess Sharpe vs ``benchmark``."""
    r, b = _aligned(returns, benchmark)
    return sharpe_ratio(r - b, bars_per_year)


def compute_backtest_metrics(
    pf_or_returns: Any,
    *,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
    benchmark: pd.Series | None = None,
    weights: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute a standard set of performance metrics.

    Parameters
    ----------
    pf_or_returns : vbt.Portfolio | pd.Series | pd.DataFrame
        Either a vectorbt Portfolio object (uses ``.returns()``),
        or a Series / single-column DataFrame of period returns.
    trading_days : int
        Annualization factor (default 365 for crypto).
    risk_free_rate : float
        Annual risk-free rate for Sharpe / Sortino (default 0).
    benchmark : pd.Series, optional
        Per-bar benchmark returns. Adds ``excess_sharpe``,
        ``information_ratio`` (the same number) and ``tracking_error``.
    weights : pd.DataFrame, optional
        Held weights (date x symbol). Adds ``annual_turnover`` (two-sided,
        mean per-bar turnover x ``trading_days``, first bar excluded).

    Returns
    -------
    dict
        Keys: total_return, cagr, sharpe, sortino, max_drawdown (signed,
        <= 0), max_drawdown_abs (the same drawdown as a positive depth, >= 0),
        max_drawdown_duration_days, annual_volatility, calmar,
        win_rate, profit_factor, var_95, cvar_95 — plus the optional
        keys above only when their input is given.
    """
    returns = _extract_returns(pf_or_returns)
    if returns.empty:
        return {}

    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] - 1  # every return compounds, the first included (TOM-262)

    n_days = (returns.index[-1] - returns.index[0]).days or 1
    years = n_days / 365.25
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0

    ann_vol = returns.std() * np.sqrt(trading_days)
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess = returns - daily_rf

    sharpe = sharpe_ratio(returns, trading_days, risk_free=risk_free_rate)
    downside = excess[excess < 0].std()
    sortino = (excess.mean() / downside * np.sqrt(trading_days)) if downside > 0 else 0.0

    dd = compute_drawdown_series(cum)
    max_dd = dd.min()
    dd_dur = _max_drawdown_duration(dd)

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = hit_rate(returns)
    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")

    var_95 = float(np.percentile(returns, 5))
    tail = returns[returns <= var_95]
    cvar_95 = float(np.mean(tail)) if len(tail) > 0 else var_95

    out = {
        "total_return": float(total_ret),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        **drawdown_fields(max_dd),
        "max_drawdown_duration_days": int(dd_dur),
        "annual_volatility": float(ann_vol),
        "calmar": float(calmar),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "var_95": var_95,
        "cvar_95": cvar_95,
    }
    if benchmark is not None:
        out["information_ratio"] = information_ratio(returns, benchmark, trading_days)
        out["excess_sharpe"] = out["information_ratio"]
        out["tracking_error"] = tracking_error(returns, benchmark, trading_days)
    if weights is not None:
        out["annual_turnover"] = annual_turnover(weights, trading_days)
    return out


def compute_drawdown_series(equity: Any) -> Any:
    """Compute drawdown series from an equity (cumulative) curve.

    Parameters
    ----------
    equity : pd.Series | pd.DataFrame | np.ndarray
        Cumulative equity or cumulative return series (must be > 0). A numpy
        array is read along its LAST axis (one path per row).

    Returns
    -------
    Same type as ``equity``
        Drawdown as negative fractions (e.g. -0.10 = 10% drawdown).
    """
    peak = np.maximum.accumulate(equity, axis=-1) if isinstance(equity, np.ndarray) else equity.cummax()
    return (equity - peak) / peak


def drawdown_fields(drawdown: float) -> dict[str, float]:
    """A drawdown as the two names every drawdown output carries (TOM-1627).

    ``max_drawdown`` is SIGNED, <= 0 (-0.25 = a 25% drawdown), the package's
    convention; ``max_drawdown_abs`` is the same drawdown as a positive depth,
    >= 0 (0.25). *drawdown* is a signed drawdown, e.g. :func:`max_drawdown`.
    """
    signed = float(drawdown)
    return {"max_drawdown": signed, "max_drawdown_abs": abs(signed)}


def max_drawdown(returns: pd.Series | np.ndarray, *, start_is_peak: bool = False) -> float:
    """Deepest drawdown of the compounded equity, SIGNED: a negative fraction (0.0 when none).

    Its positive depth is ``max_drawdown_abs`` (:func:`drawdown_fields`).

    ``start_is_peak=False`` (the run metrics' convention) measures from the
    equity after the first return; ``True`` also counts the starting equity
    1.0 as a peak, so a loss on the first row is a drawdown (the gates'
    convention).
    """
    r = np.asarray(returns, dtype=float).ravel()
    eq = np.cumprod(1.0 + r)
    if start_is_peak:
        eq = np.concatenate([[1.0], eq])
    if eq.size == 0:
        return 0.0
    return float(np.min(compute_drawdown_series(eq)))


def top_drawdowns(returns: pd.Series, n: int = 5) -> pd.DataFrame:
    """The ``n`` deepest non-overlapping drawdown episodes, deepest first.

    The starting equity 1.0 counts as a peak. Columns: ``peak`` (last date at
    the peak level; NaT when the peak is the starting equity), ``trough``,
    ``recovery`` (first date back at the peak; NaT if never) and ``depth``
    (negative fraction).
    """
    r = pd.Series(returns).dropna()
    eq = (1.0 + r).cumprod()
    eq_full = np.concatenate([[1.0], eq.to_numpy()])
    dd = compute_drawdown_series(eq_full)[1:]
    dates = r.index
    rows = []
    i = 0
    while i < len(dd):
        if dd[i] >= 0:
            i += 1
            continue
        start = i
        while i < len(dd) and dd[i] < 0:
            i += 1
        seg = dd[start:i]
        t = start + int(np.argmin(seg))
        rows.append(
            {
                "peak": dates[start - 1] if start > 0 else pd.NaT,
                "trough": dates[t],
                "recovery": dates[i] if i < len(dd) else pd.NaT,
                "depth": float(seg.min()),
            }
        )
    out = pd.DataFrame(rows, columns=["peak", "trough", "recovery", "depth"])
    return out.sort_values("depth", kind="stable").head(n).reset_index(drop=True)


def turnover_series(weights: pd.DataFrame, *, from_flat: bool = False) -> pd.Series:
    """Two-sided turnover per bar: ``sum(|w[t] - w[t-1]|)`` over the numeric columns.

    ``from_flat=True`` measures the first bar against a flat book (its turnover
    is its gross); otherwise the first bar's turnover is 0.0. NaN weights are
    taken as given — fill them first when a missing weight means flat.
    """
    w = weights.select_dtypes(include="number")
    moves = w.diff()
    if from_flat:
        moves = moves.fillna(w)
    return moves.abs().sum(axis=1)


def annual_turnover(weights: pd.DataFrame, bars_per_year: float) -> float:
    """Mean two-sided turnover per bar (first bar excluded) times ``bars_per_year``."""
    per_bar = turnover_series(weights).iloc[1:]
    return float(per_bar.mean() * bars_per_year) if len(per_bar) else 0.0


def compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 30,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Compute rolling annualized Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    window : int
        Rolling window size in periods.
    trading_days : int
        Annualization factor.

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio.
    """
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    return (roll_mean / roll_std) * np.sqrt(trading_days)


# ---------------------------------------------------------------------------
# Research statistics (TOM-1596)
# ---------------------------------------------------------------------------


def information_coefficient(signal: pd.DataFrame, forward_returns: pd.DataFrame, *, min_assets: int = 2) -> pd.Series:
    """Per-date cross-sectional Spearman rank IC of ``signal`` vs ``forward_returns``.

    Both are wide (date x symbol) and aligned on dates and symbols. On each
    date only the symbols with BOTH a signal and a forward return count; ties
    take their average rank (as ``scipy.stats.spearmanr``). A date with fewer
    than ``min_assets`` such symbols, or with no spread in either rank, has no
    IC and is left out. Lining the forward return up with the bar the signal
    is traded on (next-bar, ADR-0005) is the caller's job.
    """
    s, f = signal.align(forward_returns, join="inner")
    both = s.notna() & f.notna()
    rs = s.where(both).rank(axis=1)
    rf = f.where(both).rank(axis=1)
    ds = rs.sub(rs.mean(axis=1), axis=0)
    df = rf.sub(rf.mean(axis=1), axis=0)
    num = (ds * df).sum(axis=1)
    den = np.sqrt((ds**2).sum(axis=1) * (df**2).sum(axis=1))
    ic = num / den.where(den > 0)
    ic = ic.where(both.sum(axis=1) >= max(min_assets, 2))
    return ic.dropna().astype(float)


def icir(ic: pd.Series) -> float:
    """IC information ratio: mean IC over its standard deviation (ddof=1), NOT annualised.

    It is the per-period Sharpe of the IC series (:func:`sharpe_ratio` at one
    bar a year). NaN with fewer than two ICs or no variance.
    """
    x = pd.Series(ic, dtype=float).dropna()
    if len(x) < 2 or _flat(x.to_numpy()):
        return float("nan")
    return sharpe_ratio(x, 1.0)


def beta(returns: pd.Series | np.ndarray, benchmark: pd.Series | np.ndarray) -> float:
    """Beta of ``returns`` to ``benchmark``: ``cov(r, b) / var(b)`` (the OLS slope with an intercept).

    Series align by date (inner), arrays by position; rows missing either
    side are dropped. NaN with fewer than two rows or a flat benchmark.
    """
    r, b = _aligned(returns, benchmark)
    pair = pd.DataFrame({"r": np.asarray(r, dtype=float), "b": np.asarray(b, dtype=float)}).dropna()
    if len(pair) < 2 or _flat(pair["b"].to_numpy()):
        return float("nan")
    c = np.cov(pair["r"].to_numpy(), pair["b"].to_numpy(), ddof=1)
    return float(c[0, 1] / c[1, 1])


def hit_rate(returns: pd.Series | np.ndarray, benchmark: pd.Series | np.ndarray | None = None) -> float:
    """Share of periods with ``returns > 0`` — or ``returns > benchmark`` when one is given.

    Strict: a flat period is not a hit. Missing periods are left out of the
    count. Resample first for a monthly hit rate. NaN when no period is left.
    """
    if benchmark is None:
        x = pd.Series(np.asarray(returns, dtype=float)).dropna()
        wins = x > 0
    else:
        r, b = _aligned(returns, benchmark)
        pair = pd.DataFrame({"r": np.asarray(r, dtype=float), "b": np.asarray(b, dtype=float)}).dropna()
        wins = pair["r"] > pair["b"]
    return float(wins.mean()) if len(wins) else float("nan")


def active_share(weights: pd.DataFrame, reference: pd.DataFrame | pd.Series | Mapping[str, float]) -> float:
    """Active share vs a reference book: ``0.5 * sum(|w - ref|)`` per date, averaged over dates.

    ``reference`` is a book (date x symbol, aligned on ``weights``' dates) or
    one static set of weights. A symbol only one side holds counts in full. A
    date on which either book carries no weight at all is left out. The level
    is whatever ``weights`` holds (proxies, sleeves or names).
    """
    if isinstance(reference, pd.DataFrame):
        ref = reference.reindex(weights.index)
    else:
        ref = pd.DataFrame([pd.Series(reference, dtype=float)] * len(weights.index), index=weights.index)
    cols = weights.columns.union(ref.columns)
    a = weights.reindex(columns=cols).fillna(0.0)
    b = ref.reindex(columns=cols).fillna(0.0)
    held = weights.notna().any(axis=1) & ref.notna().any(axis=1)
    per_date = (a - b).abs().sum(axis=1) * 0.5
    return float(per_date[held].mean()) if held.any() else float("nan")


def risk_contributions(
    weights: pd.Series | Mapping[str, float],
    returns: pd.DataFrame | None = None,
    *,
    cov: pd.DataFrame | None = None,
) -> pd.Series:
    """Each asset's (or sleeve's) share of the portfolio variance: ``w_i (S w)_i / w' S w``.

    ``S`` is ``cov``, or the sample covariance of ``returns`` (date x asset) on
    the rows where every asset has a return. Pass exactly one. The shares sum
    to 1; a hedge's share is negative. NaN everywhere when the portfolio has no
    variance.
    """
    if (returns is None) == (cov is None):
        raise ValueError("risk_contributions needs exactly one of `returns` or `cov`")
    w = pd.Series(weights, dtype=float)
    if returns is not None:
        cov = returns[list(w.index)].dropna().cov()
    s = cov.reindex(index=w.index, columns=w.index).to_numpy(dtype=float)
    marginal = s @ w.to_numpy()
    total = float(w.to_numpy() @ marginal)
    if not total > 0:
        return pd.Series(np.nan, index=w.index)
    return pd.Series(w.to_numpy() * marginal / total, index=w.index)


# ---------------------------------------------------------------------------
# Period returns (TOM-3698)
# ---------------------------------------------------------------------------

# pandas period-end alias -> the offset that takes a period end back to its first day
_PERIOD_BEGIN = {
    "ME": pd.offsets.MonthBegin(1),
    "QE": pd.offsets.QuarterBegin(1, startingMonth=1),
    "YE": pd.offsets.YearBegin(1),
}


@dataclass(frozen=True)
class Window:
    """An inclusive window of calendar days, ``start``..``end``; ``None`` leaves that side open.

    Bounds compare by day: ``end="2020-01-31"`` keeps a row stamped 2020-01-31 16:00.
    A naive bound on a tz-aware index is read in the index's zone.
    """

    start: Any = None
    end: Any = None

    def bounds(self, tz: Any = None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        lo, hi = _day(self.start, tz), _day(self.end, tz)
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"window start {lo.date()} is after its end {hi.date()}")
        return lo, hi

    def clip(self, series: pd.Series) -> pd.Series:
        """The rows of ``series`` dated inside the window, in their own order."""
        idx = _dated(series.index)
        lo, hi = self.bounds(idx.tz)
        days = idx.normalize()
        keep = np.ones(len(idx), dtype=bool)
        if lo is not None:
            keep &= days >= lo
        if hi is not None:
            keep &= days <= hi
        return series.set_axis(idx)[keep]


def period_returns(
    returns: pd.Series,
    freq: str,
    start: Any = None,
    end: Any = None,
    *,
    keep_partial: bool = False,
) -> pd.Series:
    """Per-bar returns compounded to months (``ME``), quarters (``QE``) or years (``YE``)
    inside the inclusive :class:`Window` ``start``..``end``, stamped at the period end.

    A period counts only when the data shows both of its ends:

    - its opening: ``returns`` (before the window) holds a finite row dated before the
      period begins, and ``start`` cuts none of the period's business days. A series'
      first row is where the data begins, not evidence that it was the period's first
      trading day, so a series' first period is always dropped;
    - its close: the window's last finite row is on or after the period's last business
      day (Mon-Fri), so a month ending on a Saturday is complete on its Friday. A holiday
      on that weekday drops a complete period; a partial one is never kept.

    A counted period with a NaN/Inf day, or with no row at all, is NaN, not 0%. A period
    uses only rows dated inside it, and the keep test reads no row after the window.

    ``keep_partial=True`` skips these rules: exactly ``(1 + r).resample(freq).prod() - 1``
    on the window (NaN skipped, an empty period 0.0), which the report's monthly table uses.
    """
    # The partial-period masking follows qis perfstats/returns.py to_quarterly_returns
    # (MIT, Copyright (c) 2024 Artur Sepp): a period is complete only when the input
    # reaches its last sub-period. No qis code is copied; the test here is by business day.
    if freq not in _PERIOD_BEGIN:
        raise ValueError(f"freq must be one of {sorted(_PERIOD_BEGIN)}, got {freq!r}")
    window = Window(start, end)
    inside = window.clip(returns)
    compounded = (1 + inside).resample(freq).prod() - 1
    if keep_partial:
        return compounded
    finite = np.isfinite(inside.to_numpy(dtype=float))
    every = returns.set_axis(_dated(returns.index))
    every_finite = np.isfinite(every.to_numpy(dtype=float))
    if not finite.any():
        return compounded.iloc[:0]
    rows = inside.resample(freq).size()
    nonfinite = pd.Series(~finite, index=inside.index).resample(freq).sum()
    compounded = compounded.where((rows > 0) & (nonfinite == 0))

    ends = compounded.index
    day, bday = pd.offsets.Day(1), pd.offsets.BDay(1)
    begins = ends - _PERIOD_BEGIN[freq]
    first_bday = ((begins - day) + bday).normalize()
    last_bday = ((ends + day) - bday).normalize()
    keep = (begins > every.index[every_finite].min().normalize()) & (
        last_bday <= inside.index[finite].max().normalize()
    )
    lo, _ = window.bounds(ends.tz)
    if lo is not None:
        keep &= first_bday >= lo
    # every rule drops only leading or trailing periods, so the kept ones are one run
    pos = np.flatnonzero(keep)
    return compounded.iloc[pos[0] : pos[-1] + 1] if len(pos) else compounded.iloc[:0]


class Correlation(NamedTuple):
    """Pearson ``rho`` of two return series, and the ``n_overlap`` pairs it was measured on."""

    rho: float
    n_overlap: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None


def correlation(
    returns: pd.Series,
    reference: pd.Series,
    freq: str | None = None,
    start: Any = None,
    end: Any = None,
) -> Correlation:
    """Pearson correlation of ``returns`` with ``reference`` on a declared frequency (TOM-3699).

    ``freq=None`` pairs the rows as given, by date. ``"ME"``, ``"QE"`` or ``"YE"`` first
    compounds EACH series through :func:`period_returns`, so only periods whose both ends
    the data shows pair, stamped at the period end. ``start``/``end`` keep an inclusive
    :class:`Window`, the same way :func:`period_returns` reads them.

    Pairwise masking: a date (or period) where either side is NaN/Inf is left out of
    both; ``n_overlap`` counts the pairs left, ``first``/``last`` their dates. ``rho`` is
    NaN with fewer than two pairs or a flat side, never 0.0.

    Both indexes must be dates in one timezone (or both naive), each date once. Pearson is
    unit-free, but compounding is not: with ``freq`` a return <= -1 is refused, because
    ``1 + r <= 0`` cannot compound and a percent-unit file (1.5 meaning 1.5%) shows it.
    A sign-flipped replication gives a rho near -1, not near 1.
    """
    sides = []
    for name, s in (("returns", returns), ("reference", reference)):
        idx = _dated(s.index)
        if idx.has_duplicates:
            raise ValueError(f"{name}: duplicate date {idx[idx.duplicated()][0]} — one row per date")
        sides.append(s.set_axis(idx).astype(float).sort_index(kind="stable"))
    r, ref = sides
    if str(r.index.tz) != str(ref.index.tz):
        raise ValueError(
            f"returns are dated in {r.index.tz}, the reference in {ref.index.tz}: "
            "one timezone (or both naive), or the periods never meet"
        )
    if freq is None:
        r, ref = Window(start, end).clip(r), Window(start, end).clip(ref)
    else:
        for name, s in (("returns", r), ("reference", ref)):
            if (s.to_numpy() <= -1).any():
                raise ValueError(
                    f"{name} holds a return <= -1 (-100%), which cannot be compounded to {freq}: "
                    "percent units? divide by 100"
                )
        r, ref = period_returns(r, freq, start, end), period_returns(ref, freq, start, end)
    pair = pd.concat([r.rename("r"), ref.rename("b")], axis=1, join="inner").sort_index()
    pair = pair[np.isfinite(pair.to_numpy()).all(axis=1)]
    n = len(pair)
    first, last = (pair.index[0], pair.index[-1]) if n else (None, None)
    a, b = pair["r"].to_numpy(), pair["b"].to_numpy()
    if n < 2 or _flat(a) or _flat(b):
        return Correlation(float("nan"), n, first, last)
    # rounding can land a hair outside [-1, 1]; an identical series is exactly 1.0
    rho = float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))
    return Correlation(rho, n, first, last)


def _day(bound: Any, tz: Any) -> pd.Timestamp | None:
    if bound is None:
        return None
    t = pd.Timestamp(bound)
    if t.tzinfo is None and tz is not None:
        t = t.tz_localize(tz)
    elif t.tzinfo is not None:
        if tz is None:
            raise ValueError(f"window bound {bound!r} carries a timezone, the series does not")
        t = t.tz_convert(tz)
    return t.normalize()


def _dated(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex):
        return index
    if pd.api.types.is_numeric_dtype(index):
        # to_datetime would read a row number as epoch nanoseconds: 1970 dates nobody wrote
        raise ValueError(f"period returns need dates on the index, got a numeric {type(index).__name__}")
    return pd.DatetimeIndex(index)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aligned(returns: Any, benchmark: Any) -> tuple[Any, Any]:
    """Two return series on one index: Series align by date, arrays by position (truncated)."""
    if isinstance(returns, pd.Series) and isinstance(benchmark, pd.Series):
        return returns.align(benchmark, join="inner")
    r_arr = np.asarray(returns, dtype=float).ravel()
    b_arr = np.asarray(benchmark, dtype=float).ravel()
    n = min(len(r_arr), len(b_arr))
    return r_arr[:n], b_arr[:n]


def _extract_returns(pf_or_returns: Any) -> pd.Series:
    """Get a flat returns Series from various input types."""
    if isinstance(pf_or_returns, pd.Series):
        return pf_or_returns
    if isinstance(pf_or_returns, pd.DataFrame):
        if pf_or_returns.shape[1] == 1:
            return pf_or_returns.iloc[:, 0]
        raise ValueError("DataFrame must have a single column of returns")
    # Assume vectorbt Portfolio
    try:
        return pf_or_returns.returns()
    except Exception as exc:
        raise TypeError(
            f"Cannot extract returns from {type(pf_or_returns).__name__}. "
            "Pass a vbt.Portfolio, pd.Series, or single-column pd.DataFrame."
        ) from exc


# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------


def compute_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
    method: str = "historical",
) -> float:
    """Value at Risk.

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    confidence_level : float
        e.g. 0.95 or 0.99.
    horizon_days : int
        Holding period (1 = single-period).
    method : str
        ``'historical'``, ``'parametric'``, or ``'monte_carlo'``.

    Returns
    -------
    float
        VaR as a negative float (loss).
    """
    if horizon_days > 1:
        scaled = returns.rolling(horizon_days).sum().dropna()
    else:
        scaled = returns

    alpha = 1 - confidence_level

    if method == "historical":
        return float(np.percentile(scaled, alpha * 100))
    elif method == "parametric":
        from scipy.stats import norm

        mu = float(np.mean(scaled))
        sigma = float(np.std(scaled))
        return mu + sigma * norm.ppf(alpha)
    elif method == "monte_carlo":
        rng = np.random.default_rng(42)
        mu = float(np.mean(scaled))
        sigma = float(np.std(scaled))
        simulated = rng.normal(mu, sigma, 10000)
        return float(np.percentile(simulated, alpha * 100))
    else:
        raise ValueError(f"Unknown VaR method: {method}")


def compute_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """Conditional VaR (Expected Shortfall).

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    confidence_level : float
        e.g. 0.95 or 0.99.
    horizon_days : int
        Holding period.

    Returns
    -------
    float
        CVaR as a negative float (loss).
    """
    if horizon_days > 1:
        scaled = returns.rolling(horizon_days).sum().dropna().values
    else:
        scaled = returns.values

    alpha = 1 - confidence_level
    var_threshold = np.percentile(scaled, alpha * 100)
    tail = scaled[scaled <= var_threshold]
    return float(np.mean(tail)) if len(tail) > 0 else float(var_threshold)


def compute_portfolio_var(
    returns: pd.DataFrame,
    weights: dict[str, float],
    confidence_levels: list | None = None,
    horizon_days: int = 1,
    method: str = "historical",
) -> dict[float, float]:
    """Portfolio VaR with asset weights.

    Parameters
    ----------
    returns : pd.DataFrame
        Wide DataFrame (date x symbols).
    weights : dict
        ``{symbol: weight}`` dict.
    confidence_levels : list[float]
        Confidence levels.
    horizon_days : int
        Holding period.
    method : str
        ``'historical'``, ``'parametric'``, or ``'monte_carlo'``.

    Returns
    -------
    dict
        ``{confidence_level: var_value}``.
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    weight_arr = np.array([weights.get(c, 0.0) for c in returns.columns])
    port_returns = pd.Series((returns.values * weight_arr).sum(axis=1), index=returns.index)

    return {level: compute_var(port_returns, level, horizon_days, method) for level in confidence_levels}


def compute_portfolio_cvar(
    returns: pd.DataFrame,
    weights: dict[str, float],
    confidence_levels: list | None = None,
    horizon_days: int = 1,
) -> dict[float, float]:
    """Portfolio CVaR (Expected Shortfall) with asset weights.

    Parameters
    ----------
    returns : pd.DataFrame
        Wide DataFrame (date x symbols).
    weights : dict
        ``{symbol: weight}`` dict.
    confidence_levels : list[float]
        Confidence levels.
    horizon_days : int
        Holding period.

    Returns
    -------
    dict
        ``{confidence_level: cvar_value}``.
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    weight_arr = np.array([weights.get(c, 0.0) for c in returns.columns])
    port_returns = pd.Series((returns.values * weight_arr).sum(axis=1), index=returns.index)

    return {level: compute_cvar(port_returns, level, horizon_days) for level in confidence_levels}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _max_drawdown_duration(dd: pd.Series) -> int:
    """Return the longest drawdown duration in calendar days."""
    in_dd = dd < 0
    if not in_dd.any():
        return 0
    groups = (~in_dd).cumsum()
    groups = groups[in_dd]
    if groups.empty:
        return 0
    durations = groups.groupby(groups).apply(lambda g: (g.index[-1] - g.index[0]).days)
    return int(durations.max())
