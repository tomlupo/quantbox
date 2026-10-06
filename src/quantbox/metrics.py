"""The ONE metrics module: every run statistic quantbox reports is computed here.

Sharpe (and excess Sharpe vs a benchmark), information ratio, tracking error,
CAGR, volatility, max drawdown, top-N drawdowns, turnover, VaR / CVaR. Every
backtest path, the run metrics, the finding-report export, the validation
plugins and :mod:`quantbox.performance` (which quantbox-live imports) call
these functions instead of carrying their own copy (TOM-1448).

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
  before the first return (1.0) counts as a peak.
- Turnover is two-sided: ``sum(|w[t] - w[t-1]|)`` per bar.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 365  # crypto default; callers can override


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
        Keys: total_return, cagr, sharpe, sortino, max_drawdown,
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
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")

    var_95 = float(np.percentile(returns, 5))
    tail = returns[returns <= var_95]
    cvar_95 = float(np.mean(tail)) if len(tail) > 0 else var_95

    out = {
        "total_return": float(total_ret),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
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


def max_drawdown(returns: pd.Series | np.ndarray, *, start_is_peak: bool = False) -> float:
    """Deepest drawdown of the compounded equity, as a negative fraction (0.0 when none).

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
