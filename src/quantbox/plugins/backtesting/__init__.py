"""
Quantbox backtesting engines.

Two engines are provided, behind one seam (:mod:`quantbox.engine`, docs/adr/0008):

* **vectorbt** — Numba-accelerated, supports periodic + threshold rebalancing,
  multi-strategy grouping.  Best for fast iteration on spot/equity strategies.
* **rsims** — Pure numpy/pandas daily simulator with perp funding rates, margin,
  leverage caps, no-trade buffers, and forced liquidation.  Best for futures /
  perp strategy research.

Quick start::

    from quantbox.plugins.backtesting import backtest

    result = backtest(prices, weights, fees=0.001, rebalancing_freq='1W')
    print(result["metrics"])
    result = backtest(prices, weights, engine="rsims")  # the same call, the other engine

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

from quantbox.execution import ExecutionTiming, helper_execution, run_record
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
    """The seam's lag on the engine's own bar grid (:func:`quantbox.engine.book._lag_on_grid`)."""
    from quantbox.engine.book import _lag_on_grid

    timing = lag_bars if isinstance(lag_bars, ExecutionTiming) else ExecutionTiming(lag_bars)
    return _lag_on_grid(prices, weights, timing)


def _backtest(
    prices: pd.DataFrame,
    weights: dict[str, pd.DataFrame] | pd.DataFrame,
    *,
    timing: ExecutionTiming,
    engine: str,
    fees: float,
    fixed_fees: float,
    slippage: float,
    rebalancing_freq: int | str | list | None,
    threshold: float | None,
    engine_params: dict[str, Any] | None,
    trading_days: int,
) -> dict[str, Any]:
    """``backtest()`` with an already-resolved timing (``optimize()`` resolves it once per call)."""
    from quantbox.engine import Costs, simulate_weights

    book = simulate_weights(
        prices,
        weights,
        engine=engine,
        timing=timing,
        costs=Costs(fees=fees, fixed_fees=fixed_fees, slippage=slippage),
        rebalancing_freq=rebalancing_freq,
        threshold=threshold,
        engine_params=engine_params,
        trading_days=trading_days,
    )
    return {
        "engine": book.engine,
        "book": book,
        "native": book.native,
        book.native_key: book.native,  # the pre-seam name: vbt_portfolio (vectorbt), rsims_results (rsims)
        "metrics": book.metrics,
        "returns": book.returns,
        "execution": book.execution,
        "run": run_record(book.execution),
    }


def backtest(
    prices: pd.DataFrame,
    weights: dict[str, pd.DataFrame] | pd.DataFrame,
    *,
    engine: str = "vectorbt",
    fees: float = 0.001,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    rebalancing_freq: int | str | list | None = 1,
    threshold: float | None = None,
    use_numba: bool | None = None,
    engine_params: dict[str, Any] | None = None,
    trading_days: int = 365,
    lag_bars: int | None = None,
    allow_same_bar: bool = False,
    same_bar_reason: str | None = None,
) -> dict[str, Any]:
    """High-level backtest through the engine seam (:mod:`quantbox.engine`, docs/adr/0008).

    Parameters
    ----------
    prices : pd.DataFrame
        Asset prices (index=dates, columns=tickers).
    weights : dict | pd.DataFrame
        Target weights, as DECIDED: row ``t`` uses data through ``close[t]``.
    engine : str
        The engine adapter: ``"vectorbt"`` (default, the ``[vectorbt]`` extra)
        or ``"rsims"``. The rest of the call does not change with it.
    fees : float
        Proportional fee rate.
    fixed_fees : float
        Fixed fee per order (vectorbt).
    slippage : float
        Slippage rate (vectorbt).
    rebalancing_freq : None | int | str | list
        Rebalancing schedule (vectorbt; rsims trades every bar). ``None`` =
        buy-and-hold: one trade, at ``close[lag_bars]``
        (:func:`quantbox.engine.lag_buy_and_hold`).
    threshold : float | None
        Deviation threshold for rebalancing bands (vectorbt).
    use_numba : bool | None
        Numba JIT (vectorbt); shorthand for ``engine_params={"use_numba": ...}``.
    engine_params : dict | None
        The adapter's own parameters (rsims: ``trade_buffer``, ``initial_cash``,
        ``margin``, ``capitalise_profits``, ``equity_basis``); a key the
        adapter does not own is refused.
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
        ``"engine"``; ``"book"`` — the :class:`~quantbox.engine.TradedBook`
        (returns, value, turnover, trades, native); ``"native"`` — the engine's
        own object (a ``vbt.Portfolio`` on vectorbt), also under its pre-seam
        key (``"vbt_portfolio"`` on vectorbt, ``"rsims_results"`` on rsims);
        ``"metrics"`` — dict of performance metrics; ``"returns"`` — per-bar
        returns; ``"execution"`` — the execution timing used (as
        ``run_manifest.json``); ``"run"`` — ``{"kind": "backtest" | "research"}``.
    """
    params = dict(engine_params or {})
    if use_numba is not None:
        params["use_numba"] = use_numba
    return _backtest(
        prices,
        weights,
        timing=helper_execution(lag_bars, allow_same_bar, same_bar_reason),
        engine=engine,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        rebalancing_freq=rebalancing_freq,
        threshold=threshold,
        engine_params=params,
        trading_days=trading_days,
    )
