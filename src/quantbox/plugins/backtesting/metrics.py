"""
Performance metrics for backtesting results.

Delegates to vectorbt's native stats when a Portfolio object is available,
falls back to manual computation for plain returns Series/DataFrame inputs.

Trade-level analytics are exposed via ``compute_trade_metrics`` and
``trade_summary`` — these require a vbt.Portfolio object.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 365  # crypto default; callers can override


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def compute_backtest_metrics(
    pf_or_returns: Any,
    *,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute a standard set of performance metrics.

    When *pf_or_returns* is a ``vbt.Portfolio``, delegates to its native
    accessors (Sharpe, Sortino, Calmar, Omega, drawdowns, trades).  Falls
    back to manual computation for plain Series/DataFrame inputs.

    Parameters
    ----------
    pf_or_returns : vbt.Portfolio | pd.Series | pd.DataFrame
        Either a vectorbt Portfolio object or period returns.
    trading_days : int
        Annualization factor (default 365 for crypto).
    risk_free_rate : float
        Annual risk-free rate for Sharpe / Sortino (default 0).

    Returns
    -------
    dict
        Keys: total_return, cagr, sharpe, sortino, max_drawdown,
        max_drawdown_duration_days, annual_volatility, calmar, omega,
        win_rate, profit_factor, best_trade, worst_trade,
        avg_winning_trade, avg_losing_trade, total_trades,
        total_closed_trades, var_95, cvar_95.
    """
    if _is_portfolio(pf_or_returns):
        return _metrics_from_portfolio(pf_or_returns, trading_days=trading_days)
    return _metrics_from_returns(
        _extract_returns(pf_or_returns),
        trading_days=trading_days,
        risk_free_rate=risk_free_rate,
    )


def _is_portfolio(obj: Any) -> bool:
    """Check if obj is a vbt.Portfolio without importing vectorbt."""
    return type(obj).__name__ == "Portfolio" and hasattr(obj, "sharpe_ratio")


def _metrics_from_portfolio(pf: Any, *, trading_days: int) -> dict[str, float]:
    """Extract metrics using vbt.Portfolio native methods."""
    returns = pf.returns()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0] if returns.shape[1] == 1 else returns.mean(axis=1)

    cum = (1 + returns).cumprod()
    total_ret = float(cum.iloc[-1] / cum.iloc[0] - 1)
    n_days = (returns.index[-1] - returns.index[0]).days or 1
    years = n_days / 365.25
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0
    ann_vol = float(returns.std() * np.sqrt(trading_days))

    # Native vbt ratios
    sharpe = _safe_float(pf.sharpe_ratio())
    sortino = _safe_float(pf.sortino_ratio())
    calmar = _safe_float(pf.calmar_ratio())
    omega = _safe_float(pf.omega_ratio())
    max_dd = _safe_float(pf.max_drawdown())

    # Drawdown duration from native drawdown records
    dd_dur = 0
    try:
        dd_records = pf.drawdowns.records_readable
        if not dd_records.empty:
            durations = (
                dd_records["End Timestamp"] - dd_records["Peak Timestamp"]
            ).dt.days
            dd_dur = int(durations.max()) if not durations.empty else 0
    except Exception:
        dd_dur = _max_drawdown_duration(compute_drawdown_series(cum))

    # Trade metrics from native trade records
    trade_metrics = _trade_metrics_from_portfolio(pf)

    # VaR / CVaR (no vbt native — keep manual)
    var_95 = float(np.percentile(returns, 5))
    tail = returns[returns <= var_95]
    cvar_95 = float(np.mean(tail)) if len(tail) > 0 else var_95

    return {
        "total_return": total_ret,
        "cagr": float(cagr),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(-abs(max_dd)),  # negative convention
        "max_drawdown_duration_days": dd_dur,
        "annual_volatility": ann_vol,
        "calmar": calmar,
        "omega": omega,
        **trade_metrics,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }


def _trade_metrics_from_portfolio(pf: Any) -> dict[str, float]:
    """Extract trade-level metrics from vbt.Portfolio.trades."""
    try:
        trades = pf.trades.records_readable
    except Exception:
        return {"win_rate": 0.0, "profit_factor": 0.0}

    closed = trades[trades["Status"] == "Closed"]
    total = len(trades)
    total_closed = len(closed)

    if total_closed == 0:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "avg_winning_trade": 0.0,
            "avg_losing_trade": 0.0,
            "total_trades": float(total),
            "total_closed_trades": 0.0,
        }

    wins = closed[closed["PnL"] > 0]
    losses = closed[closed["PnL"] <= 0]
    win_rate = len(wins) / total_closed if total_closed > 0 else 0.0
    pf_ratio = (
        float(wins["PnL"].sum() / abs(losses["PnL"].sum()))
        if len(losses) > 0 and losses["PnL"].sum() != 0
        else float("inf")
    )
    return {
        "win_rate": win_rate,
        "profit_factor": pf_ratio,
        "best_trade": float(closed["Return"].max()) if total_closed > 0 else 0.0,
        "worst_trade": float(closed["Return"].min()) if total_closed > 0 else 0.0,
        "avg_winning_trade": float(wins["Return"].mean()) if len(wins) > 0 else 0.0,
        "avg_losing_trade": float(losses["Return"].mean()) if len(losses) > 0 else 0.0,
        "total_trades": float(total),
        "total_closed_trades": float(total_closed),
    }


def _metrics_from_returns(
    returns: pd.Series,
    *,
    trading_days: int,
    risk_free_rate: float,
) -> dict[str, float]:
    """Fallback: compute metrics from plain returns (no vbt.Portfolio)."""
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

    var_95 = float(np.percentile(returns, 5))
    tail = returns[returns <= var_95]
    cvar_95 = float(np.mean(tail)) if len(tail) > 0 else var_95

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
        "var_95": var_95,
        "cvar_95": cvar_95,
    }


# ---------------------------------------------------------------------------
# Trade analytics (new — requires vbt.Portfolio)
# ---------------------------------------------------------------------------


def compute_trade_metrics(pf: Any) -> pd.DataFrame:
    """Detailed per-trade analytics from a vbt.Portfolio.

    Returns a DataFrame with one row per closed trade, including PnL,
    return, duration, direction, entry/exit timestamps and prices.
    """
    if not _is_portfolio(pf):
        raise TypeError("compute_trade_metrics requires a vbt.Portfolio object")
    trades = pf.trades.records_readable
    closed = trades[trades["Status"] == "Closed"].copy()
    if closed.empty:
        return pd.DataFrame()
    closed["Duration"] = (
        closed["Exit Timestamp"] - closed["Entry Timestamp"]
    ).dt.days
    return closed


def trade_summary(pf: Any, by: str = "Column") -> pd.DataFrame:
    """Aggregate trade stats grouped by column (asset/strategy).

    Parameters
    ----------
    pf : vbt.Portfolio
        Portfolio with trade records.
    by : str
        Column to group by (default "Column" = per-asset).

    Returns
    -------
    pd.DataFrame
        Columns: count, win_rate, avg_pnl, avg_return, avg_duration_days,
        best_return, worst_return, profit_factor.
    """
    trades = compute_trade_metrics(pf)
    if trades.empty:
        return pd.DataFrame()

    def _agg(g: pd.DataFrame) -> pd.Series:
        wins = g[g["PnL"] > 0]
        losses = g[g["PnL"] <= 0]
        pf_ratio = (
            float(wins["PnL"].sum() / abs(losses["PnL"].sum()))
            if len(losses) > 0 and losses["PnL"].sum() != 0
            else float("inf")
        )
        return pd.Series({
            "count": len(g),
            "win_rate": len(wins) / len(g) if len(g) > 0 else 0.0,
            "avg_pnl": g["PnL"].mean(),
            "avg_return": g["Return"].mean(),
            "avg_duration_days": g["Duration"].mean(),
            "best_return": g["Return"].max(),
            "worst_return": g["Return"].min(),
            "profit_factor": pf_ratio,
        })

    return trades.groupby(by).apply(_agg, include_groups=False)


# ---------------------------------------------------------------------------
# Drawdown analytics (enhanced — vbt native when available)
# ---------------------------------------------------------------------------


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


def drawdown_table(pf: Any, top_n: int = 10) -> pd.DataFrame:
    """Top drawdowns from a vbt.Portfolio's native drawdown records.

    Parameters
    ----------
    pf : vbt.Portfolio
        Portfolio object.
    top_n : int
        Number of worst drawdowns to return.

    Returns
    -------
    pd.DataFrame
        Columns: peak_date, valley_date, recovery_date, depth, duration_days, status.
    """
    if not _is_portfolio(pf):
        raise TypeError("drawdown_table requires a vbt.Portfolio object")

    dd = pf.drawdowns.records_readable
    if dd.empty:
        return pd.DataFrame()

    dd = dd.copy()
    dd["depth"] = (dd["Valley Value"] - dd["Peak Value"]) / dd["Peak Value"]
    dd["duration_days"] = (dd["End Timestamp"] - dd["Peak Timestamp"]).dt.days

    result = dd.nsmallest(top_n, "depth")[
        ["Peak Timestamp", "Valley Timestamp", "End Timestamp", "depth", "duration_days", "Status"]
    ].copy()
    result.columns = ["peak_date", "valley_date", "recovery_date", "depth", "duration_days", "status"]
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------


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
    except Exception as exc:
        raise TypeError(
            f"Cannot extract returns from {type(pf_or_returns).__name__}. "
            "Pass a vbt.Portfolio, pd.Series, or single-column pd.DataFrame."
        ) from exc


def _safe_float(val: Any) -> float:
    """Safely convert a vbt scalar (possibly aggregated) to float."""
    if isinstance(val, pd.Series):
        val = val.mean()
    try:
        f = float(val)
        return f if np.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


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
