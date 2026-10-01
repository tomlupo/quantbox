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

``backtest()`` and ``optimize()`` follow the one execution-timing convention
(:mod:`quantbox.execution`): weights decided on bar ``t`` fill at the close of
bar ``t + lag_bars``, default 1 (next-bar), exactly as ``quantbox run -c``.
``lag_bars=0`` reproduces the old same-bar numbers, loudly.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from quantbox.execution import apply_execution_lag, execution_record, resolve_lag_bars, warn_if_same_bar

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


def _lag_for_engine(
    prices: pd.DataFrame,
    weights: dict[str, pd.DataFrame] | pd.DataFrame,
    lag_bars: int,
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """Apply the execution lag on the engine's own bar grid.

    The vectorbt engine trades on ``prices.index | weights.index`` and
    forward-fills weights onto it, so a sparse weights frame (rebalance dates
    only) is first put on that grid — otherwise ``shift(1)`` would lag by one
    REBALANCE, not one bar. Cells stay NaN, so the engine's forward-fill is
    unchanged; only the decision moves ``lag_bars`` bars later.
    """
    if lag_bars == 0:
        return weights

    def one(w: pd.DataFrame) -> pd.DataFrame:
        return apply_execution_lag(w.reindex(prices.index.union(w.index)), lag_bars)

    if isinstance(weights, dict):
        return {name: one(w) for name, w in weights.items()}
    return one(weights)


def _backtest(
    prices: pd.DataFrame,
    weights: dict[str, pd.DataFrame] | pd.DataFrame,
    *,
    lag_bars: int,
    fees: float,
    fixed_fees: float,
    slippage: float,
    rebalancing_freq: int | str | list | None,
    threshold: float | None,
    use_numba: bool,
    trading_days: int,
) -> dict[str, Any]:
    """``backtest()`` with an already-resolved ``lag_bars`` and no warning (``optimize()`` warns once)."""
    pf = run_vectorbt(
        prices,
        _lag_for_engine(prices, weights, lag_bars),
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
        "execution": execution_record(lag_bars),
    }


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
    lag_bars: int | None = None,
) -> dict[str, Any]:
    """High-level backtest using the vectorbt engine.

    Parameters
    ----------
    prices : pd.DataFrame
        Asset prices (index=dates, columns=tickers).
    weights : dict | pd.DataFrame
        Target weights, as DECIDED: row ``t`` uses data through ``close[t]``.
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
    lag_bars : int | None
        Execution lag (:mod:`quantbox.execution`): weights decided on bar ``t``
        fill at the close of bar ``t + lag_bars``. ``None`` = the default, 1
        (next-bar), the same as ``quantbox run -c``. ``0`` = same-bar, logged
        as a look-ahead warning; use it only to reproduce an old number.

    Returns
    -------
    dict
        ``"vbt_portfolio"`` — the vbt.Portfolio object,
        ``"metrics"`` — dict of performance metrics,
        ``"returns"`` — daily returns Series,
        ``"execution"`` — the execution timing used (as ``run_manifest.json``).
    """
    lag = resolve_lag_bars(None if lag_bars is None else {"lag_bars": lag_bars})
    warn_if_same_bar(lag, where="backtest()")
    return _backtest(
        prices,
        weights,
        lag_bars=lag,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        rebalancing_freq=rebalancing_freq,
        threshold=threshold,
        use_numba=use_numba,
        trading_days=trading_days,
    )
