"""
Performance metrics for backtesting results.

Extracts standard risk/return metrics from a vectorbt Portfolio object
or a plain returns Series/DataFrame.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 365  # crypto default; callers can override


def compute_backtest_metrics(
    pf_or_returns: Any,
    *,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
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

    Returns
    -------
    dict
        Keys: total_return, cagr, sharpe, sortino, max_drawdown,
        max_drawdown_duration_days, annual_volatility, calmar,
        win_rate, profit_factor.
    """
    returns = _extract_returns(pf_or_returns)
    if returns.empty:
        return {}

    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] / cum.iloc[0] - 1

    n_days = (returns.index[-1] - returns.index[0]).days or 1
    years = n_days / 365.25
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0

    ann_vol = returns.std() * np.sqrt(trading_days)
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess = returns - daily_rf

    sharpe = (excess.mean() / excess.std() * np.sqrt(trading_days)) if excess.std() > 0 else 0.0
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

    return {
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
    }


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute drawdown series from an equity (cumulative) curve.

    Parameters
    ----------
    equity : pd.Series
        Cumulative equity or cumulative return series (must be > 0).

    Returns
    -------
    pd.Series
        Drawdown as negative fractions (e.g. -0.10 = 10% drawdown).
    """
    peak = equity.cummax()
    return (equity - peak) / peak


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
    except Exception:
        raise TypeError(
            f"Cannot extract returns from {type(pf_or_returns).__name__}. "
            "Pass a vbt.Portfolio, pd.Series, or single-column pd.DataFrame."
        )


def _max_drawdown_duration(dd: pd.Series) -> int:
    """Return the longest drawdown duration in calendar days."""
    in_dd = dd < 0
    if not in_dd.any():
        return 0
    groups = (~in_dd).cumsum()
    groups = groups[in_dd]
    if groups.empty:
        return 0
    durations = groups.groupby(groups).apply(
        lambda g: (g.index[-1] - g.index[0]).days
    )
    return int(durations.max())
