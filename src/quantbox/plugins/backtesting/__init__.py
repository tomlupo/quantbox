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
``lag_bars=0`` (same-bar) is refused (docs/adr/0005) unless the call also
passes ``allow_same_bar=True, same_bar_reason="..."`` — the explicit override
(docs/adr/0006); the result then says ``run: {kind: research}``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from quantbox.execution import (
    ExecutionTiming,
    apply_execution_lag,
    helper_execution,
    lag_buy_and_hold,
    run_record,
    timing_record,
)
from quantbox.metrics import (
    compute_backtest_metrics,
    compute_cvar,
    compute_drawdown_series,
    compute_portfolio_cvar,
    compute_portfolio_var,
    compute_rolling_sharpe,
    compute_var,
)

from .optimizer import optimize
from .rsims_engine import positions_from_no_trade_buffer

__all__ = [
    "backtest",
    "optimize",
    "positions_from_no_trade_buffer",
    "compute_backtest_metrics",
    "compute_cvar",
    "compute_drawdown_series",
    "compute_portfolio_cvar",
    "compute_portfolio_var",
    "compute_rolling_sharpe",
    "compute_var",
]


# The engine primitives fill row t at close[t] and expect weights ALREADY
# lagged; exported here, a quick calculation reached them with raw weights and
# came out same-bar. They stay importable from their modules for the pipeline
# and engine tests, and asking for them here says where the lagged path is.
_ENGINE_PRIMITIVES = {
    "run_vectorbt": "quantbox.plugins.backtesting.vectorbt_engine.run",
    "fixed_commission_backtest_with_funding": (
        "quantbox.plugins.backtesting.rsims_engine.fixed_commission_backtest_with_funding"
    ),
}


def __getattr__(name: str) -> Any:
    if name in _ENGINE_PRIMITIVES:
        raise ImportError(
            f"{name} is no longer exported from {__name__} (docs/adr/0005-next-bar-is-mandatory.md): "
            "it fills at the close the weights were decided on. Use backtest(prices, weights) — "
            f"next-bar, lag_bars >= 1 — or, for weights you have lagged yourself, {_ENGINE_PRIMITIVES[name]}."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _lag_for_engine(
    prices: pd.DataFrame,
    weights: dict[str, pd.DataFrame] | pd.DataFrame,
    lag_bars: int | ExecutionTiming,
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """Apply the execution lag on the engine's own bar grid.

    The vectorbt engine trades on ``prices.index | weights.index`` and
    forward-fills weights onto it, so a sparse weights frame (rebalance dates
    only) is first put on that grid — otherwise ``shift(1)`` would lag by one
    REBALANCE, not one bar. Cells stay NaN, so the engine's forward-fill is
    unchanged; only the decision moves ``lag_bars`` bars later.
    """

    timing = lag_bars if isinstance(lag_bars, ExecutionTiming) else ExecutionTiming(lag_bars)

    def one(w: pd.DataFrame) -> pd.DataFrame:
        return apply_execution_lag(w.reindex(prices.index.union(w.index)), timing.lag_bars, same_bar=timing.same_bar)

    if isinstance(weights, dict):
        return {name: one(w) for name, w in weights.items()}
    return one(weights)


def _backtest(
    prices: pd.DataFrame,
    weights: dict[str, pd.DataFrame] | pd.DataFrame,
    *,
    timing: ExecutionTiming,
    fees: float,
    fixed_fees: float,
    slippage: float,
    rebalancing_freq: int | str | list | None,
    threshold: float | None,
    use_numba: bool,
    trading_days: int,
) -> dict[str, Any]:
    """``backtest()`` with an already-resolved timing (``optimize()`` resolves it once per call)."""
    from .vectorbt_engine import run as run_vectorbt

    grid = prices.index
    for w in weights.values() if isinstance(weights, dict) else [weights]:
        grid = grid.union(w.index)  # the engine's own bar grid
    pf = run_vectorbt(
        prices,
        _lag_for_engine(prices, weights, timing),
        rebalancing_freq=lag_buy_and_hold(pd.to_datetime(grid), rebalancing_freq, timing.lag_bars),
        threshold=threshold,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        use_numba=use_numba,
    )
    metrics = compute_backtest_metrics(pf, trading_days=trading_days)
    execution = timing_record(timing)
    return {
        "vbt_portfolio": pf,
        "metrics": metrics,
        "returns": pf.returns(),
        "execution": execution,
        "run": run_record(execution),
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
    allow_same_bar: bool = False,
    same_bar_reason: str | None = None,
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
        Rebalancing schedule. ``None`` = buy-and-hold: one trade, at
        ``close[lag_bars]`` (:func:`quantbox.execution.lag_buy_and_hold`).
    threshold : float | None
        Deviation threshold for rebalancing bands.
    use_numba : bool
        Enable Numba JIT.
    trading_days : int
        Annualization factor for metrics (365 for crypto).
    lag_bars : int | None
        Execution lag (:mod:`quantbox.execution`): weights decided on bar ``t``
        fill at the close of bar ``t + lag_bars``. ``None`` = the default, 1
        (next-bar), the same as ``quantbox run -c``; also the minimum — ``0``
        (same-bar) raises ``ValueError`` unless the override below is given.
    allow_same_bar, same_bar_reason : bool, str | None
        The explicit same-bar override (docs/adr/0006), the keywords of
        ``execution.same_bar: {allow, reason}``: only with ``lag_bars=0`` and a
        non-empty reason. The result is then RESEARCH, not a backtest.

    Returns
    -------
    dict
        ``"vbt_portfolio"`` — the vbt.Portfolio object,
        ``"metrics"`` — dict of performance metrics,
        ``"returns"`` — daily returns Series,
        ``"execution"`` — the execution timing used (as ``run_manifest.json``);
        ``"run"`` — ``{"kind": "backtest" | "research"}`` (as ``run_manifest.json``).
    """
    return _backtest(
        prices,
        weights,
        timing=helper_execution(lag_bars, allow_same_bar, same_bar_reason),
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        rebalancing_freq=rebalancing_freq,
        threshold=threshold,
        use_numba=use_numba,
        trading_days=trading_days,
    )
