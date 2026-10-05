"""
VectorBT backtesting engine — Numba-accelerated portfolio simulation.

Ported from quantlab's ``vectorbt_tools/backtesting.py``.

Features
--------
- Periodic rebalancing (daily, weekly, monthly, custom dates, buy-and-hold).
- Threshold-based rebalancing bands via ``from_order_func()`` + Numba.
- Combined periodic + threshold rebalancing.
- Proportional fees, fixed fees, slippage.
- Multi-strategy grouping via MultiIndex columns.

Execution timing
----------------
An ENGINE PRIMITIVE: row ``t`` of ``weights`` fills at ``close[t]``, so the
weights must already be lagged (``quantbox.execution.apply_execution_lag``).
Research code calls ``quantbox.plugins.backtesting.backtest`` instead, which
lags them — next-bar, ``lag_bars >= 1`` (docs/adr/0005).

Examples
--------
::

    from quantbox.plugins.backtesting.vectorbt_engine import run as run_vectorbt

    weights = apply_execution_lag(decided_weights, 1)  # next-bar

    # Buy-and-hold
    pf = run_vectorbt(prices, weights, rebalancing_freq=None)

    # Monthly rebalancing
    pf = run_vectorbt(prices, weights, rebalancing_freq='1M')

    # Monthly + 5% deviation threshold
    pf = run_vectorbt(prices, weights, rebalancing_freq='1M', threshold=0.05)

    # Custom fees and slippage
    pf = run_vectorbt(prices, weights, rebalancing_freq='1M', fees=0.001, slippage=0.0005)
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantbox.exceptions import MissingExtraError

try:
    import vectorbt as vbt
    from numba import njit
    from vectorbt.portfolio.enums import Direction, SizeType
    from vectorbt.portfolio.nb import (
        get_col_elem_nb,
        get_elem_nb,
        order_nb,
        order_nothing_nb,
        sort_call_seq_out_nb,
    )
except ModuleNotFoundError as exc:  # vectorbt + numba ship in the [vectorbt] extra
    # Only the extra's OWN packages being absent means "install the extra"; a
    # module missing deeper inside an installed vectorbt is a broken install
    # and must surface as itself.
    if (exc.name or "").split(".")[0] not in ("vectorbt", "numba"):
        raise
    raise MissingExtraError("vectorbt", "the vectorbt backtest engine", exc.name) from exc

from quantbox.frequency import rebalancing_dates  # noqa: E402

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Order-function callbacks. Module level so numba compiles them, and the
# vectorbt simulator specialised on them, ONCE per process: a closure
# re-jitted on every run() call recompiled both on every backtest.
# ----------------------------------------------------------------------


def _pre_sim_func_nb(c, rebalancing_mask):
    c.segment_mask[:, :] = False
    c.segment_mask[rebalancing_mask, :] = True
    return ()


def _pre_group_func_nb(c):
    return ()


def _pre_segment_func_nb(c, size, size_type, direction, threshold, no_order, lend_col, borrow_col):
    position_values = np.empty(c.group_len, dtype=np.float64)
    for k, col in enumerate(range(c.from_col, c.to_col)):
        c.last_val_price[col] = get_col_elem_nb(c, col, c.close)
        position_values[k] = c.last_val_price[col] * c.last_position[col]

    # Portfolio value is positions + CASH. Free cash is cash net of short collateral: using it
    # (as this did until TOM-1429) misstates every weight of a book that holds shorts.
    total_value = np.sum(position_values) + c.last_cash[c.group]
    position_weights = position_values / total_value

    # A cell with no order (the `orders` mask) targets its CURRENT weight: its order value is 0,
    # so the call sequence still sorts sells before buys, and order_func_nb places nothing.
    target_weights = np.empty(c.group_len, dtype=np.float64)
    for k in range(c.group_len):
        t = size[c.i, c.from_col + k]
        target_weights[k] = position_weights[k] if (no_order[c.i, c.from_col + k] or np.isnan(t)) else t
    # The financing legs take the RESIDUAL of what is actually held after this bar's orders,
    # untouched (drifted) cells included, so the book sums to exactly 1 and no buy is cut.
    if lend_col >= 0 and not no_order[c.i, lend_col]:
        residual = 1.0
        for k in range(c.group_len):
            col = c.from_col + k
            if col != lend_col and col != borrow_col:
                residual -= target_weights[k]
        target_weights[lend_col - c.from_col] = max(residual, 0.0)
        target_weights[borrow_col - c.from_col] = min(residual, 0.0)
    deviation = np.abs(position_weights - target_weights)

    rebalancing_flag = False
    for dev in deviation:
        if dev > threshold:
            rebalancing_flag = True
            break

    if rebalancing_flag:
        order_value_out = np.empty(c.group_len, dtype=np.float64)
        for k in range(c.group_len):
            c.call_seq_now[k] = k
        sort_call_seq_out_nb(c, target_weights, size_type, direction, order_value_out, c.call_seq_now, ctx_select=False)
        return (target_weights,)
    return (None,)


def _order_func_nb(c, weights_arr, size_type, direction, fees_arr, fixed_fees_arr, slippage_arr, no_order):
    if weights_arr is None:
        return order_nothing_nb()
    if no_order[c.i, c.col]:  # a cell with no order (the `orders` mask)
        return order_nothing_nb()
    col_i = c.call_seq_now[c.call_idx]
    return order_nb(
        size=weights_arr[col_i],
        price=get_elem_nb(c, c.close),
        size_type=np.int64(get_elem_nb(c, size_type)),
        direction=np.int64(get_elem_nb(c, direction)),
        fees=np.float64(get_elem_nb(c, fees_arr)),
        fixed_fees=np.float64(get_elem_nb(c, fixed_fees_arr)),
        slippage=np.float64(get_elem_nb(c, slippage_arr)),
        log=True,
    )


def _post_order_func_nb(c, weights_arr):
    return None


_CALLBACK_NAMES = ("pre_sim", "pre_group", "pre_segment", "order", "post_order")
_PY_CALLBACKS = (_pre_sim_func_nb, _pre_group_func_nb, _pre_segment_func_nb, _order_func_nb, _post_order_func_nb)
_JIT_CALLBACKS: tuple | None = None


def _callbacks(use_numba: bool) -> tuple:
    """The five callbacks, jitted once per process when *use_numba*."""
    global _JIT_CALLBACKS
    if not use_numba:
        return _PY_CALLBACKS
    if _JIT_CALLBACKS is None:
        _JIT_CALLBACKS = tuple(njit(f) for f in _PY_CALLBACKS)
    return _JIT_CALLBACKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_labels(index: pd.MultiIndex, drop_levels: int = -1) -> pd.Index:
    """Create labels from a MultiIndex by concatenating remaining levels.

    Parameters
    ----------
    index : pd.MultiIndex
        Multi-level index to label.
    drop_levels : int | str | list
        Level(s) to drop (default: last level).

    Returns
    -------
    pd.Index
        Formatted labels like ``"x-A_y-B"``.
    """
    index_dropped = index.droplevel(drop_levels)
    labels = []
    for values in index_dropped:
        label = "_".join(
            f"{level_name}-{value}" for level_name, value in zip(index_dropped.names, values, strict=False)
        )
        labels.append(label)
    return pd.Index(labels)


def get_rebalancing_dates(
    dates: pd.DatetimeIndex,
    rebalancing_freq: int | str | list | pd.DateOffset | None,
) -> pd.DatetimeIndex:
    """Compute rebalancing dates from a frequency spec.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Full date range of the backtest.
    rebalancing_freq : None | int | str | list | pd.DateOffset
        ``None`` → buy-and-hold (first date only).
        ``int`` → every *n*-th date.
        ``str`` → pandas offset string (``"1D"``, ``"1W"``, ``"1M"``, ``"1Y"``,
            ``"1min"``, ``"30min"``). Routed through
            :func:`quantbox.frequency.parse_rebalance_offset` for strict
            validation — lowercase ``"1m"`` is rejected as ambiguous
            (was MONTHS in the legacy parser, MINUTES per pandas/ccxt).
            Use ``"1M"`` for months or ``"1min"`` for minutes explicitly.
        ``pd.DateOffset`` → used directly.
        ``list`` → explicit dates.

    Returns
    -------
    pd.DatetimeIndex
        Bars of *dates* only: a calendar date that is not a bar is snapped
        forward to the next bar (:func:`quantbox.frequency.rebalancing_dates`).
    """
    return rebalancing_dates(dates, rebalancing_freq)


def _per_column(value: float, weights_df: pd.DataFrame, fee_free: Sequence[str]) -> np.ndarray:
    """A cost as the engine's array: a scalar, or (rows x cols) with 0 on the *fee_free* tickers."""
    if not fee_free:
        return np.asarray(value)
    free = weights_df.columns.get_level_values(-1).isin(list(fee_free))
    row = np.where(free, 0.0, float(value))
    return np.ascontiguousarray(np.broadcast_to(row, weights_df.shape), dtype=np.float64)


def rebalance_fill_gaps(
    pf: vbt.Portfolio,
    target_weights: pd.DataFrame,
    rebalance_bars: pd.Index,
    orders: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """How far the book the engine HELD after each rebalance is from the target it was given.

    Returns one row per rebalance bar: ``gap`` = sum over assets of
    |held weight - target weight| at the close of the bar (held = asset value /
    portfolio value, after the bar's orders), and ``turnover`` = sum of
    |target - held weight one bar earlier|. With no costs and enough cash the
    gap is 0; a cash-constrained engine shows it as the buys it cut. With
    *orders*, only the ordered cells count. Single group (one strategy) only.
    """
    held = pf.asset_value(group_by=False)
    value = pf.value()
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError("rebalance_fill_gaps: one strategy group only")
        value = value.iloc[:, 0]
    held_w = held.div(value, axis=0)
    held_w.columns = held_w.columns.get_level_values(-1)
    target = target_weights.reindex(index=held_w.index, columns=held_w.columns).ffill().fillna(0.0)
    bars = held_w.index.intersection(pd.Index(rebalance_bars))
    gap_cells = (held_w - target).abs()
    move_cells = (target - held_w.shift(1).fillna(0.0)).abs()
    if orders is not None:  # only the cells that were ordered: an untouched cell drifts by design
        ordered = orders.reindex(index=held_w.index, columns=held_w.columns, fill_value=False).astype(bool)
        gap_cells, move_cells = gap_cells.where(ordered, 0.0), move_cells.where(ordered, 0.0)
    gap = gap_cells.loc[bars].sum(axis=1)
    turnover = move_cells.loc[bars].sum(axis=1)
    return pd.DataFrame({"gap": gap, "turnover": turnover})


def validate_prices(prices: pd.DataFrame, weights: pd.DataFrame) -> bool:
    """Check that no weight is assigned where a price is missing.

    Parameters
    ----------
    prices, weights : pd.DataFrame
        Must share the same shape/columns.

    Returns
    -------
    bool
        ``True`` if valid.

    Raises
    ------
    ValueError
        If any weight != 0 where price is NaN.
    """
    # Use the ticker-level prices aligned to weights columns
    tickers = weights.columns.get_level_values(-1)
    prices_aligned = prices[tickers]
    prices_aligned.columns = weights.columns

    bad_mask = (weights != 0) & prices_aligned.isna()
    if bad_mask.any().any():
        bad_locs = list(zip(*np.where(bad_mask), strict=False))
        for row, col in bad_locs:
            logger.warning(
                "Weight != 0 but no price: date=%s, ticker=%s",
                weights.index[row],
                weights.columns[col],
            )
        raise ValueError("Weights assigned where prices are missing")
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(
    prices: pd.DataFrame,
    weights: dict[str, pd.DataFrame] | pd.DataFrame,
    rebalancing_freq: int | str | list | None = 1,
    threshold: float | None = None,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    use_order_func: bool | None = None,
    use_numba: bool = True,
    create_strategy_label: bool = True,
    fee_free: Sequence[str] = (),
    orders: pd.DataFrame | None = None,
    residual_legs: Sequence[str] = (),
) -> vbt.Portfolio:
    """Run a vectorbt backtest.

    Parameters
    ----------
    prices : pd.DataFrame
        Asset prices (index=dates, columns=tickers).
    weights : dict[str, DataFrame] | DataFrame
        Target weights.  A dict maps strategy names → weight DataFrames.
    rebalancing_freq : None | int | str | list
        Rebalancing schedule (see :func:`get_rebalancing_dates`).
    threshold : float | None
        Deviation threshold that triggers rebalancing (requires Numba order
        function).  ``None`` disables threshold rebalancing.
    fees : float
        Proportional fee rate (e.g. 0.001 = 10 bps).
    fixed_fees : float
        Fixed fee per order.
    slippage : float
        Slippage rate per order.
    use_order_func : bool | None
        Force use of custom Numba order function.  Auto-detected when *None*.
    use_numba : bool
        Enable Numba JIT compilation (default True).
    create_strategy_label : bool
        Add a ``strategy`` level to column MultiIndex.
    fee_free : sequence of str
        Tickers traded without fees, fixed fees or slippage — the synthetic
        cash legs of ``venue.financing`` (:mod:`quantbox.financing`).
    orders : DataFrame of bool | None
        Per-cell order mask on the weights' index and tickers (or, for
        MultiIndex weights, on the weights' own columns, one mask per strategy
        slice): an order is placed only where it is True, so an instrument
        can be left untouched on a bar the rest of the book trades (it did not
        print — :mod:`quantbox.engine.schedule`). The rebalance bars are
        then the rows with any order, and *rebalancing_freq* is ignored. It
        runs the flexible (order-function) path, with *threshold* 0 when None.
    residual_legs : (lend, borrow) tickers | ()
        The financing cash legs (:mod:`quantbox.financing`). On a bar they are
        ordered, they are sized to the residual ``1 - sum`` of the weights
        actually held after the bar's orders (an untouched, drifted cell at its
        current weight), split by sign — so the engine never runs out of cash.

    Returns
    -------
    vbt.Portfolio
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a pandas DataFrame")
    if not isinstance(weights, (dict, pd.DataFrame)):
        raise ValueError("weights must be a dictionary or pandas DataFrame")

    if isinstance(weights, pd.DataFrame):
        if not weights.columns.get_level_values(-1).isin(prices.columns).all():
            raise ValueError("All tickers in weights must be present in prices")
    elif isinstance(weights, dict) and not set().union(*(v.columns for v in weights.values())).issubset(prices.columns):
        raise ValueError("All tickers in weights must be present in prices")

    # ------------------------------------------------------------------
    # Decide order-func path
    # ------------------------------------------------------------------
    if orders is not None:
        # A per-cell mask needs the flexible path: an untouched cell must still sort as "no order".
        use_order_func = True
        if threshold is None:
            threshold = 0.0
    if threshold is not None:
        if use_order_func is False:
            warnings.warn("use_order_func is False but threshold is set — overriding to True.", stacklevel=2)
        use_order_func = True
    else:
        if use_order_func is None:
            use_order_func = False
        if use_order_func:
            threshold = 0.0

    # ------------------------------------------------------------------
    # Numba JIT
    # ------------------------------------------------------------------
    os.environ["NUMBA_DISABLE_JIT"] = "0" if use_numba else "1"
    (
        pre_sim_func_nb_jit,
        pre_group_func_nb_jit,
        pre_segment_func_nb_jit,
        order_func_nb_jit,
        post_order_func_nb_jit,
    ) = _callbacks(use_numba)

    # ------------------------------------------------------------------
    # Prepare weights DataFrame & group_by
    # ------------------------------------------------------------------
    if isinstance(weights, pd.DataFrame):
        weights_df = weights.copy()
        if weights_df.columns.nlevels == 1:
            if create_strategy_label:
                cols = weights_df.columns.to_frame()
                cols.insert(0, "strategy", "strategy")
                weights_df.columns = pd.MultiIndex.from_frame(cols)
                group_by = "strategy"
            else:
                group_by = None
        else:
            if create_strategy_label and "strategy" not in weights_df.columns.names:
                labels = create_labels(weights_df.columns)
                cols = weights_df.columns.to_frame()
                cols.insert(0, "strategy", labels)
                weights_df.columns = pd.MultiIndex.from_frame(cols)
            group_by = list(weights_df.columns.names[:-1])
    else:
        weights_df = pd.concat(weights, axis=1, names=["strategy"])
        group_by = "strategy"

    # ------------------------------------------------------------------
    # Align prices and weights
    # ------------------------------------------------------------------
    index = prices.index.union(weights_df.index)
    prices = prices.reindex(index).ffill()
    weights_df = weights_df.reindex(index).ffill()
    weights_df = weights_df.fillna(0)

    if validate_prices(prices, weights_df):
        prices = prices.bfill()

    index = pd.to_datetime(index)
    prices.index = pd.to_datetime(prices.index)
    weights_df.index = pd.to_datetime(weights_df.index)

    # ------------------------------------------------------------------
    # Rebalancing dates (and the per-cell order mask)
    # ------------------------------------------------------------------
    no_order = np.zeros(weights_df.shape, dtype=np.bool_)
    if orders is not None:
        if isinstance(weights, pd.DataFrame) and orders.columns.nlevels > 1:
            # One mask per strategy slice: the columns are the weights' own, in their order.
            mask = orders.reindex(index=index, columns=weights.columns, fill_value=False)
        else:
            mask = orders.reindex(index=index, columns=weights_df.columns.get_level_values(-1), fill_value=False)
        no_order = ~mask.to_numpy(dtype=bool)
        rebalancing_dates = index[mask.to_numpy(dtype=bool).any(axis=1)]
    else:
        rebalancing_dates = get_rebalancing_dates(index, rebalancing_freq)

    # Prices with same column structure as weights
    _prices = prices[weights_df.columns.get_level_values(-1)]
    _prices.columns = weights_df.columns

    # ------------------------------------------------------------------
    # Order parameters
    # ------------------------------------------------------------------
    size_type_arr = np.asarray(SizeType.TargetPercent)
    direction_arr = np.asarray(Direction.Both)
    fees_arr = _per_column(fees, weights_df, fee_free)
    fixed_fees_arr = _per_column(fixed_fees, weights_df, fee_free)
    slippage_arr = _per_column(slippage, weights_df, fee_free)

    # ------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------
    if use_order_func:
        rebalancing_mask = index.isin(rebalancing_dates)
        size_np = weights_df.to_numpy(dtype=np.float64, copy=True)
        tickers = list(weights_df.columns.get_level_values(-1))
        legs = [tickers.index(t) if t in tickers else -1 for t in residual_legs] if len(residual_legs) == 2 else []
        lend_col, borrow_col = (legs[0], legs[1]) if legs and min(legs) >= 0 else (-1, -1)
        threshold = float(threshold)

        pf = vbt.Portfolio.from_order_func(
            _prices,
            order_func_nb_jit,
            size_type_arr,
            direction_arr,
            fees_arr,
            fixed_fees_arr,
            slippage_arr,
            no_order,
            pre_sim_func_nb=pre_sim_func_nb_jit,
            pre_sim_args=(rebalancing_mask,),
            pre_group_func_nb=pre_group_func_nb_jit,
            pre_segment_func_nb=pre_segment_func_nb_jit,
            pre_segment_args=(size_np, size_type_arr, direction_arr, threshold, no_order, lend_col, borrow_col),
            post_order_func_nb=post_order_func_nb_jit,
            group_by=group_by,
            cash_sharing=True,
            use_numba=use_numba,
        )
    else:
        size = weights_df.copy()
        size.loc[~size.index.isin(rebalancing_dates), :] = None

        pf = vbt.Portfolio.from_orders(
            close=_prices,
            size=size,
            size_type=size_type_arr,
            direction=direction_arr,
            group_by=group_by,
            cash_sharing=True,
            call_seq="auto",
            fees=fees_arr,
            fixed_fees=fixed_fees_arr,
            slippage=slippage_arr,
        )

    return pf
