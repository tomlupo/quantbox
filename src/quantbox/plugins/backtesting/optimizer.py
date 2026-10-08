"""Parameter optimization for backtesting strategies.

Provides grid search and walk-forward optimization over strategy parameters,
wrapping the existing ``backtest()`` function (same engine seam and execution
timing, next-bar by default)::

    from quantbox.plugins.backtesting import optimize

    result = optimize(
        prices, weights_fn, {"lookback": [10, 20, 50]},
        method="walk_forward", metric="sharpe",
    )
    print(result["best_params"], result["best_metric"])
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import Any

import pandas as pd

from quantbox.engine.registry import get_engine
from quantbox.execution import helper_execution, run_record, timing_record
from quantbox.funding_guard import check_series


def _backtest_lazy():
    """Lazy import to avoid circular dependency with __init__.py."""
    from quantbox.plugins.backtesting import _backtest as _bt

    return _bt


def optimize(
    prices: pd.DataFrame,
    weights_fn: Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame],
    param_grid: dict[str, list[Any]],
    *,
    method: str = "grid",
    metric: str = "sharpe",
    engine: str = "vectorbt",
    engine_params: dict[str, Any] | None = None,
    fees: float = 0.001,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    rebalancing_freq: int | str | None = 1,
    trading_days: int = 365,
    train_size: int = 252,
    test_size: int = 63,
    lag_bars: int | None = None,
    allow_same_bar: bool = False,
    same_bar_reason: str | None = None,
    schedule: str = "calendar",
    leverage: str | None = None,
    max_leverage: float | None = None,
    funding_rates: pd.DataFrame | None = None,
    funding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Optimize strategy parameters via grid search or walk-forward.

    Args:
        prices: Wide-format price DataFrame (dates x symbols).
        weights_fn: ``fn(prices, params) -> weights_df``. Called for each
            parameter combination to produce target weights.
        param_grid: ``{param_name: [values_to_try]}``.
        method: ``"grid"`` or ``"walk_forward"``.
        metric: Key from backtest metrics dict to maximize (e.g. ``"sharpe"``).
        engine, engine_params: The engine adapter and its own parameters, as in
            ``backtest()`` (:mod:`quantbox.engine`); the search does not change with it.
        fees, fixed_fees, slippage, rebalancing_freq, trading_days:
            Forwarded to ``backtest()``.
        train_size: Training window in rows (walk-forward only).
        test_size: Test window in rows (walk-forward only).
        lag_bars: Execution lag, as in ``backtest()``: default and minimum 1
            (next-bar); ``0`` raises. Applied inside each
            window, so a walk-forward test window starts flat for
            ``lag_bars`` bars.
        allow_same_bar, same_bar_reason: The explicit same-bar override, as in
            ``backtest()`` (docs/adr/0006): only with ``lag_bars=0`` and a
            non-empty reason; the result is then RESEARCH, not a backtest.
        schedule, leverage: As in ``backtest()``: the calendar schedule (default)
            or ``"bars"``, and ``venue.leverage`` on the calendar.
        max_leverage: The gross cap, as in ``backtest()``: ``None`` = the
            default, 1 (TOM-1525); a levered search declares it.
        funding_rates, funding: The perp funding series and the ``{"ignore": True,
            "reason": ...}`` escape, as in ``backtest()``. A series on an engine
            that does not charge funding is refused once, before the search.

    Returns:
        ``{"best_params", "best_metric", "all_results", "execution"}`` for grid
        search, or ``{"best_params", "best_metric", "all_results",
        "oos_results", "execution"}`` for walk-forward. ``"execution"`` is the
        timing used (as ``run_manifest.json``).
    """
    timing = helper_execution(lag_bars, allow_same_bar, same_bar_reason, schedule)
    # Refused here, once: inside the search a failing combination is skipped, not raised.
    adapter = get_engine(engine)
    adapter.check_params(engine_params)
    check_series(adapter, funding_rates, funding)  # the seam's funding guard, before the search
    bt_kwargs = dict(
        timing=timing,
        engine=engine,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        rebalancing_freq=rebalancing_freq,
        threshold=None,
        engine_params=engine_params,
        trading_days=trading_days,
        leverage=leverage,
        max_leverage=max_leverage,
        funding_rates=funding_rates,
        funding=funding,
    )

    if method == "walk_forward":
        result = _walk_forward(prices, weights_fn, param_grid, metric, bt_kwargs, train_size, test_size)
    else:
        result = _grid_search(prices, weights_fn, param_grid, metric, bt_kwargs)
    execution = timing_record(timing)
    return {**result, "execution": execution, "run": run_record(execution)}


def _grid_search(
    prices: pd.DataFrame,
    weights_fn: Callable,
    param_grid: dict[str, list[Any]],
    metric: str,
    bt_kwargs: dict[str, Any],
) -> dict[str, Any]:
    param_names = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    rows: list[dict[str, Any]] = []

    for combo in combos:
        params = dict(zip(param_names, combo, strict=False))
        try:
            weights = weights_fn(prices, params)
            result = _backtest_lazy()(prices, weights, **bt_kwargs)
            rows.append({**params, **result["metrics"]})
        except Exception:
            continue

    if not rows:
        return {"best_params": {}, "best_metric": float("nan"), "all_results": pd.DataFrame()}

    df = pd.DataFrame(rows)
    best_idx = df[metric].idxmax()
    best_params = {n: df.loc[best_idx, n] for n in param_names}
    return {
        "best_params": best_params,
        "best_metric": df.loc[best_idx, metric],
        "all_results": df,
    }


def _walk_forward(
    prices: pd.DataFrame,
    weights_fn: Callable,
    param_grid: dict[str, list[Any]],
    metric: str,
    bt_kwargs: dict[str, Any],
    train_size: int,
    test_size: int,
) -> dict[str, Any]:
    oos_rows: list[dict[str, Any]] = []
    start = 0

    while start + train_size + test_size <= len(prices):
        train_prices = prices.iloc[start : start + train_size]
        test_prices = prices.iloc[start + train_size : start + train_size + test_size]

        # Optimize on training window
        train_result = _grid_search(train_prices, weights_fn, param_grid, metric, bt_kwargs)
        best_params = train_result["best_params"]

        if best_params:
            # Evaluate on out-of-sample window
            try:
                test_weights = weights_fn(test_prices, best_params)
                test_bt = _backtest_lazy()(test_prices, test_weights, **bt_kwargs)
                oos_rows.append(
                    {
                        "period_start": test_prices.index[0],
                        "period_end": test_prices.index[-1],
                        **best_params,
                        **{f"oos_{k}": v for k, v in test_bt["metrics"].items()},
                    }
                )
            except Exception:
                pass

        start += test_size

    oos_df = pd.DataFrame(oos_rows)

    # Aggregate: pick params that appeared most often as best
    if not oos_df.empty:
        param_names = list(param_grid.keys())
        param_cols = oos_df[param_names]
        # Most common parameter combo
        best_params = param_cols.mode().iloc[0].to_dict()
        oos_metric_col = f"oos_{metric}"
        best_metric = oos_df[oos_metric_col].mean() if oos_metric_col in oos_df.columns else float("nan")
    else:
        best_params = {}
        best_metric = float("nan")

    return {
        "best_params": best_params,
        "best_metric": best_metric,
        "all_results": oos_df,
        "oos_results": oos_df,
    }
