"""
Quantbox backtesting engines.

Two engines are provided:

* **vectorbt** — Numba-accelerated, supports periodic + threshold rebalancing,
  multi-strategy grouping.  Best for fast iteration on spot/equity strategies.
* **rsims** — Pure numpy/pandas daily simulator with perp funding rates, margin,
  leverage caps, no-trade buffers, and forced liquidation.  Best for futures /
  perp strategy research.

Quick start::

    from quantbox.plugins.backtesting import backtest

    result = backtest(prices, weights, fees=0.001, rebalancing_freq='1W')
    print(result["metrics"])
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .metrics import (
    compute_backtest_metrics,
    compute_cvar,
    compute_drawdown_series,
    compute_portfolio_cvar,
    compute_portfolio_var,
    compute_rolling_sharpe,
    compute_var,
)
from .optimizer import optimize
from .rsims_engine import fixed_commission_backtest_with_funding, positions_from_no_trade_buffer
from .vectorbt_engine import run as run_vectorbt

__all__ = [
    "backtest",
    "optimize",
    "run_vectorbt",
    "fixed_commission_backtest_with_funding",
    "positions_from_no_trade_buffer",
    "compute_backtest_metrics",
    "compute_cvar",
    "compute_drawdown_series",
    "compute_portfolio_cvar",
    "compute_portfolio_var",
    "compute_rolling_sharpe",
    "compute_var",
]


def backtest(
    prices: pd.DataFrame,
    weights: dict[str, pd.DataFrame] | pd.DataFrame,
    *,
    fees: float = 0.001,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    rebalancing_freq: int | str | list | None = 1,
    threshold: float | None = None,
    use_numba: bool = True,
    trading_days: int = 365,
) -> dict[str, Any]:
    """High-level backtest using the vectorbt engine.

    Parameters
    ----------
    prices : pd.DataFrame
        Asset prices (index=dates, columns=tickers).
    weights : dict | pd.DataFrame
        Target weights.
    fees : float
        Proportional fee rate.
    fixed_fees : float
        Fixed fee per order.
    slippage : float
        Slippage rate.
    rebalancing_freq : None | int | str | list
        Rebalancing schedule.
    threshold : float | None
        Deviation threshold for rebalancing bands.
    use_numba : bool
        Enable Numba JIT.
    trading_days : int
        Annualization factor for metrics (365 for crypto).

    Returns
    -------
    dict
        ``"vbt_portfolio"`` — the vbt.Portfolio object,
        ``"metrics"`` — dict of performance metrics,
        ``"returns"`` — daily returns Series.
    """
    pf = run_vectorbt(
        prices,
        weights,
        rebalancing_freq=rebalancing_freq,
        threshold=threshold,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        use_numba=use_numba,
    )
    metrics = compute_backtest_metrics(pf, trading_days=trading_days)
    return {
        "vbt_portfolio": pf,
        "metrics": metrics,
        "returns": pf.returns(),
    }
