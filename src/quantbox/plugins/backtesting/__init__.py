"""
Quantbox backtesting engines.

Two engines are provided, behind one seam (:mod:`quantbox.engine`, docs/adr/0008).
The seam owns the rebalancing schedule (periodic + threshold); both engines
execute it:

* **vectorbt** — Numba-accelerated, multi-strategy grouping.  Best for fast
  iteration on spot/equity strategies.
* **rsims** — Pure numpy/pandas daily simulator with perp funding rates, margin,
  leverage caps, no-trade buffers, and forced liquidation.  Best for futures /
  perp strategy research.

Quick start::

    from quantbox.plugins.backtesting import backtest

    result = backtest(prices, weights, fees=0.001, rebalancing_freq='1W')
    print(result["metrics"])
    result = backtest(prices, weights, engine="rsims")  # the same call, the other engine

``backtest()`` and ``optimize()`` build the book with the one book function,
:func:`quantbox.engine.simulate` — the same schedule as ``quantbox run -c``:
the instrument and execution calendars (``schedule="calendar"``, the default;
``schedule="bars"`` makes every price bar an execution bar), the rebalancing
schedule and threshold on every engine, ``venue.leverage`` (``leverage=``,
default normalize), and the execution timing (:mod:`quantbox.execution`):
weights decided on bar ``t`` fill at the close of bar ``t + lag_bars``,
default 1 (next-bar).
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


def _on_price_bars(
    prices: pd.DataFrame, weights: dict[str, pd.DataFrame] | pd.DataFrame
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """Put a SPARSE weights frame (rebalance dates only) on the price bars, from its first row on.

    The helpers take weights stamped only on rebalance dates; the book function
    runs on the bars that carry both a price and a weight row. Each row
    is carried forward (the seam's HOLD policy) onto every price bar after it,
    so a sparse frame is the book it describes, and a row stamped on a date
    with no price bar is decided on the next price bar. A dense frame is
    unchanged.
    """

    def one(w: pd.DataFrame) -> pd.DataFrame:
        if w.index.isin(prices.index).all() and prices.index[prices.index >= w.index.min()].isin(w.index).all():
            return w
        grid = prices.index.union(w.index)
        bars = prices.index[prices.index >= w.index.min()] if len(w.index) else prices.index[:0]
        return w.reindex(grid).ffill().reindex(bars)

    if isinstance(weights, dict):
        return {name: one(w) for name, w in weights.items()}
    return one(weights)


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
    leverage: str | None = None,
    policy: dict[str, Any] | None = None,
    group_limits: dict[str, Any] | None = None,
    universe: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """``backtest()`` with an already-resolved timing (``optimize()`` resolves it once per call)."""
    from quantbox.engine import Costs, get_engine, simulate
    from quantbox.engine.groups import resolve_group_limits
    from quantbox.financing import resolve_leverage

    engine = get_engine(engine)  # first: a missing [vectorbt] extra is named before anything else runs
    groups = None
    if group_limits is not None:
        if universe is None:
            raise ValueError("group_limits needs universe=: a frame with `symbol` and the group column it names")
        groups = resolve_group_limits(group_limits).bind(universe)
    book = simulate(
        prices,
        _on_price_bars(prices, weights),
        engine=engine,
        timing=timing,
        costs=Costs(fees=fees, fixed_fees=fixed_fees, slippage=slippage),
        rebalancing_freq=rebalancing_freq,
        threshold=threshold,
        policy=policy,
        groups=groups,
        leverage=None if leverage is None else resolve_leverage(leverage),
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
    schedule: str = "calendar",
    leverage: str | None = None,
    policy: dict[str, Any] | None = None,
    group_limits: dict[str, Any] | None = None,
    universe: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """High-level backtest through the engine seam (:func:`quantbox.engine.simulate`, docs/adr/0008).

    Parameters
    ----------
    prices : pd.DataFrame
        Asset prices (index=dates, columns=tickers).
    weights : dict | pd.DataFrame
        Target weights, as DECIDED: row ``t`` uses data through ``close[t]``.
        A sparse frame (rebalance dates only) is carried onto the price bars
        after its first row; a NaN cell holds the last target (every engine).
        A dict is one strategy each.
    engine : str
        The engine adapter: ``"vectorbt"`` (default, the ``[vectorbt]`` extra)
        or ``"rsims"``. The rest of the call does not change with it.
    fees : float
        Proportional fee rate.
    fixed_fees : float
        Fixed fee per order (every engine).
    slippage : float
        Slippage rate (every engine).
    rebalancing_freq : None | int | str | list
        Rebalancing schedule on the execution calendar, the same on every
        engine (:func:`quantbox.frequency.rebalancing_dates`). ``None`` =
        buy-and-hold: one decision, on the first bar, filled ``lag_bars`` later.
    threshold : float | None
        Rebalancing band (absolute weight), on every engine: a scheduled
        rebalance is placed only when a held weight drifted more than this
        from its target. The seam measures the drift cost-free, so with costs
        it can differ slightly from an in-engine band (docs/adr/0008).
    use_numba : bool | None
        Numba JIT (vectorbt); shorthand for ``engine_params={"use_numba": ...}``.
    engine_params : dict | None
        The adapter's own parameters (rsims: ``trade_buffer``, ``initial_cash``,
        ``margin``, ``capitalise_profits``, ``equity_basis``; vectorbt:
        ``use_numba``, ``use_order_func``, ``create_strategy_label``,
        ``initial_cash``); a key the adapter does not own is refused. Both
        start from 10,000 and compound by default.
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
    schedule : str
        ``"calendar"`` (default): the scheduled book of ``quantbox run``
        (instrument and execution calendars, deferral, ``venue.leverage``).
        ``"bars"``: every price bar is an execution bar; no deferral, no
        ``venue.leverage``.
    leverage : str | None
        ``venue.leverage`` on the calendar schedule: ``"normalize"`` (the
        default, every engine) or ``"borrow"`` (held as decided, free
        financing). Refused with ``schedule="bars"``.
    policy : dict | None
        The rebalancing policy (:mod:`quantbox.engine.policy`), e.g.
        ``{"policy": "periodic", "frequency": "monthly", "calendar": "NYSE"}``,
        ``{"policy": "tranche", "tranches": 4, "frequency": "weekly"}``,
        ``{"policy": "band", "band": 0.05}`` or
        ``{"policy": "corridor", "width": [0.02, 0.05], "bounds": {"SPY": 0.03}}``.
        Replaces ``rebalancing_freq`` + ``threshold``; passing both is refused.
    group_limits, universe : dict | None, pd.DataFrame | None
        Group limits on the decided weights (:mod:`quantbox.engine.groups`):
        ``{"by": "asset_class", "limits": {"equity": {"max": 0.6}}}``, the groups
        read from *universe* (``symbol`` + the ``by`` column). An infeasible
        limit raises ``ValueError``.

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
        timing=helper_execution(lag_bars, allow_same_bar, same_bar_reason, schedule),
        engine=engine,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        rebalancing_freq=rebalancing_freq,
        threshold=threshold,
        engine_params=params,
        trading_days=trading_days,
        leverage=leverage,
        policy=policy,
        group_limits=group_limits,
        universe=universe,
    )
