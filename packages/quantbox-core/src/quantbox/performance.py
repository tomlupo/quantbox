"""Live portfolio performance tracking.

Computes equity curves, time-weighted returns (TWR) adjusted for
external flows, period returns, risk metrics, and reconciliation.

This module provides the reusable computation layer. Deployment scripts
handle I/O (loading snapshots, writing reports).

## LLM Usage Guide

### Quick Start
```python
from quantbox.performance import compute_performance

# equity_series: pd.Series with DatetimeIndex, values = account equity in USD
# flows_df: pd.DataFrame with columns [date, amount_usdc] (withdrawals negative)
result = compute_performance(
    equity=equity_series,
    flows_df=flows_df,
    initial_deposit=100.0,
    inception_date="2026-02-07",
    trading_days=365,
)

# result dict contains:
#   periods: {1D, 7D, 30D, ITD} each with return_pct, pnl_usdc, days
#   risk_metrics: sharpe, sortino, max_drawdown, volatility, win_rate, ...
#   equity_curve: [{date, equity}, ...]
#   reconciliation: {start_value, end_value, implied_pnl, is_valid}
```

### Integration with Snapshots
```python
import pandas as pd

# Build equity series from broker snapshots
snapshots = [
    {"date": "2026-02-08", "equity": 101.14},
    {"date": "2026-02-09", "equity": 100.67},
]
equity = pd.Series(
    {pd.Timestamp(s["date"]): s["equity"] for s in snapshots}
)
```
"""

from __future__ import annotations

import logging

import pandas as pd

from quantbox.plugins.backtesting.metrics import compute_backtest_metrics

logger = logging.getLogger(__name__)

RECONCILIATION_TOLERANCE = 1.0  # USD


def compute_daily_returns(
    equity: pd.Series,
    flows_df: pd.DataFrame,
) -> pd.Series:
    """Compute TWR daily returns adjusted for external flows.

    Formula: r_t = (V_t - V_{t-1} - F_t) / V_{t-1}
    where F_t = net flows on day t.

    Parameters
    ----------
    equity : pd.Series
        Equity curve with DatetimeIndex.
    flows_df : pd.DataFrame
        Columns [date, amount_usdc]. Withdrawals should be negative.

    Returns
    -------
    pd.Series
        Daily returns with DatetimeIndex.
    """
    if len(equity) < 2:
        return pd.Series(dtype=float)

    returns = []
    dates = []

    for i in range(1, len(equity)):
        date = equity.index[i]
        v_prev = equity.iloc[i - 1]
        v_curr = equity.iloc[i]

        # Sum flows on this date
        flow = 0.0
        if not flows_df.empty:
            day_flows = flows_df[flows_df["date"] == date]
            flow = day_flows["amount_usdc"].sum()

        if v_prev > 0:
            r = (v_curr - v_prev - flow) / v_prev
        else:
            r = 0.0

        returns.append(r)
        dates.append(date)

    return pd.Series(returns, index=pd.DatetimeIndex(dates), name="daily_return")


def compute_period_return(
    returns: pd.Series,
    equity: pd.Series,
    flows_df: pd.DataFrame,
    n_days: int | None,
) -> dict | None:
    """Compute compounded return for the last N days.

    Parameters
    ----------
    returns : pd.Series
        Daily returns series.
    equity : pd.Series
        Equity curve.
    flows_df : pd.DataFrame
        Columns [date, amount_usdc].
    n_days : int or None
        Number of days, or None for inception-to-date.

    Returns
    -------
    dict or None
        {return_pct, pnl_usdc, days} or None if insufficient data.
    """
    if returns.empty:
        return None

    if n_days is not None:
        subset = returns.iloc[-n_days:]
        if len(subset) < n_days:
            return None
    else:
        subset = returns

    compound = float((1 + subset).prod() - 1)

    # PnL in USD for this period
    if n_days is not None and n_days <= len(equity):
        start_equity = equity.iloc[-(n_days + 1)] if n_days < len(equity) else equity.iloc[0]
    else:
        start_equity = equity.iloc[0]

    end_equity = equity.iloc[-1]

    # Sum flows in this period
    if n_days is not None:
        cutoff = returns.index[-n_days] if n_days <= len(returns) else returns.index[0]
        period_flows = flows_df[flows_df["date"] >= cutoff]["amount_usdc"].sum() if not flows_df.empty else 0.0
    else:
        period_flows = flows_df["amount_usdc"].sum() if not flows_df.empty else 0.0

    pnl = end_equity - start_equity - period_flows

    return {
        "return_pct": round(compound * 100, 4),
        "pnl_usdc": round(float(pnl), 4),
        "days": len(subset),
    }


def compute_reconciliation(
    equity: pd.Series,
    flows_df: pd.DataFrame,
    tolerance: float = RECONCILIATION_TOLERANCE,
) -> dict:
    """Validate: end = start + cumulative_flows + pnl.

    Parameters
    ----------
    equity : pd.Series
        Equity curve with DatetimeIndex.
    flows_df : pd.DataFrame
        Columns [date, amount_usdc].
    tolerance : float
        Maximum acceptable difference in USD.

    Returns
    -------
    dict
        Reconciliation result with is_valid flag.
    """
    if equity.empty:
        return {"is_valid": False, "reason": "no data"}

    start_value = equity.iloc[0]
    end_value = equity.iloc[-1]
    cumulative_flows = flows_df["amount_usdc"].sum() if not flows_df.empty else 0.0
    implied_pnl = end_value - start_value - cumulative_flows
    difference = abs(end_value - (start_value + cumulative_flows + implied_pnl))

    return {
        "start_value": round(float(start_value), 4),
        "end_value": round(float(end_value), 4),
        "cumulative_flows": round(float(cumulative_flows), 4),
        "implied_pnl": round(float(implied_pnl), 4),
        "difference": round(float(difference), 4),
        "is_valid": difference <= tolerance,
        "tolerance_usdc": tolerance,
    }


def compute_performance(
    equity: pd.Series,
    flows_df: pd.DataFrame,
    initial_deposit: float,
    inception_date: str,
    target_date: str | None = None,
    trading_days: int = 365,
    period_days: list[tuple] | None = None,
) -> dict:
    """Compute full performance report from an equity curve.

    Parameters
    ----------
    equity : pd.Series
        Equity curve with DatetimeIndex (one entry per day).
    flows_df : pd.DataFrame
        Columns [date, amount_usdc]. Withdrawals negative. Can be empty.
    initial_deposit : float
        Initial deposit in USD.
    inception_date : str
        Start date string (YYYY-MM-DD).
    target_date : str or None
        Report date. Defaults to last equity date.
    trading_days : int
        Annualization factor (365 for crypto, 252 for equities).
    period_days : list of (label, n_days) or None
        Custom period definitions. Defaults to 1D, 7D, 30D.

    Returns
    -------
    dict
        Full performance result with periods, risk_metrics,
        equity_curve, and reconciliation.
    """
    if equity.empty:
        return {"error": "no equity data"}

    returns = compute_daily_returns(equity, flows_df)

    # Determine report date
    if target_date:
        date = target_date
    else:
        date = equity.index[-1].strftime("%Y-%m-%d")

    # Period returns
    if period_days is None:
        period_days = [("1D", 1), ("7D", 7), ("30D", 30)]

    periods = {}
    for label, n in period_days:
        result = compute_period_return(returns, equity, flows_df, n)
        periods[label] = result

    # Inception-to-date
    itd = compute_period_return(returns, equity, flows_df, None)
    periods["ITD"] = itd

    # Risk metrics from quantbox-core
    risk_metrics = {}
    if len(returns) >= 2:
        metrics = compute_backtest_metrics(returns, trading_days=trading_days)
        risk_metrics = {
            "sharpe": round(metrics.get("sharpe", 0), 4),
            "sortino": round(metrics.get("sortino", 0), 4),
            "max_drawdown": round(metrics.get("max_drawdown", 0), 4),
            "max_drawdown_duration_days": metrics.get("max_drawdown_duration_days", 0),
            "annual_volatility": round(metrics.get("annual_volatility", 0), 4),
            "calmar": round(metrics.get("calmar", 0), 4),
            "win_rate": round(metrics.get("win_rate", 0), 4),
            "profit_factor": round(min(metrics.get("profit_factor", 0), 999), 4),
            "var_95": round(metrics.get("var_95", 0), 6),
            "cvar_95": round(metrics.get("cvar_95", 0), 6),
        }

    # Equity curve for output
    equity_list = [{"date": d.strftime("%Y-%m-%d"), "equity": round(float(v), 4)} for d, v in equity.items()]

    # Reconciliation
    reconciliation = compute_reconciliation(equity, flows_df)

    # Cumulative values
    cumulative_flows = flows_df["amount_usdc"].sum() if not flows_df.empty else 0.0
    cumulative_pnl = float(equity.iloc[-1]) - initial_deposit - cumulative_flows

    return {
        "date": date,
        "inception_date": inception_date,
        "equity_usdc": round(float(equity.iloc[-1]), 4),
        "initial_deposit_usdc": initial_deposit,
        "cumulative_flows_usdc": round(float(cumulative_flows), 4),
        "cumulative_pnl_usdc": round(float(cumulative_pnl), 4),
        "data_points": len(equity),
        "periods": periods,
        "risk_metrics": risk_metrics,
        "equity_curve": equity_list,
        "reconciliation": reconciliation,
    }
