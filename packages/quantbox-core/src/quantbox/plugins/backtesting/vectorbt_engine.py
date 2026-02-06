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

Examples
--------
::

    from quantbox.plugins.backtesting import run_vectorbt

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
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit
from vectorbt.portfolio.enums import Direction, NoOrder, Order, SizeType
from vectorbt.portfolio.nb import (
    get_col_elem_nb,
    get_elem_nb,
    get_group_value_ctx_nb,
    order_nb,
    order_nothing_nb,
    sort_call_seq_nb,
)

logger = logging.getLogger(__name__)


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
            f"{level_name}-{value}"
            for level_name, value in zip(index_dropped.names, values)
        )
        labels.append(label)
    return pd.Index(labels)


def get_rebalancing_dates(
    dates: pd.DatetimeIndex,
    rebalancing_freq: Optional[Union[int, str, list]],
) -> pd.DatetimeIndex:
    """Compute rebalancing dates from a frequency spec.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Full date range of the backtest.
    rebalancing_freq : None | int | str | list
        ``None`` → buy-and-hold (first date only).
        ``int`` → every *n*-th date.
        ``str`` → pandas offset string (``"1D"``, ``"1W"``, ``"1M"``, ``"1Y"``).
        ``list`` → explicit dates.

    Returns
    -------
    pd.DatetimeIndex
    """
    dates = pd.DatetimeIndex(dates)

    if rebalancing_freq is None:
        return pd.DatetimeIndex([dates[0]])

    if isinstance(rebalancing_freq, int):
        return dates[::rebalancing_freq]

    if isinstance(rebalancing_freq, str):
        if rebalancing_freq[-1].isalpha():
            unit = rebalancing_freq[-1].lower()
            value = int(rebalancing_freq[:-1]) if rebalancing_freq[:-1].isdigit() else 1
            unit_mapping = {"d": "days", "w": "weeks", "m": "months", "y": "years"}
            offset = pd.DateOffset(**{unit_mapping[unit]: value})
            return pd.date_range(start=dates[0], end=dates[-1], freq=offset)
        raise ValueError("Invalid rebalancing frequency format.")

    if isinstance(rebalancing_freq, list):
        return pd.DatetimeIndex(rebalancing_freq)

    raise ValueError("Invalid rebalancing frequency type.")


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
        bad_locs = [(i, c) for i, c in zip(*np.where(bad_mask))]
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
    weights: Union[Dict[str, pd.DataFrame], pd.DataFrame],
    rebalancing_freq: Optional[Union[int, str, list]] = 1,
    threshold: Optional[float] = None,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    use_order_func: Optional[bool] = None,
    use_numba: bool = True,
    create_strategy_label: bool = True,
) -> "vbt.Portfolio":
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
    elif isinstance(weights, dict):
        if not set().union(*(v.columns for v in weights.values())).issubset(prices.columns):
            raise ValueError("All tickers in weights must be present in prices")

    # ------------------------------------------------------------------
    # Numba callback functions (defined here so njit is applied once)
    # ------------------------------------------------------------------
    def pre_sim_func_nb(c, rebalancing_mask):
        c.segment_mask[:, :] = False
        c.segment_mask[rebalancing_mask, :] = True
        return ()

    def pre_group_func_nb(c):
        return ()

    def pre_segment_func_nb(c, size, size_type, direction, threshold):
        position_values = np.empty(c.group_len, dtype=np.float64)
        for k, col in enumerate(range(c.from_col, c.to_col)):
            c.last_val_price[col] = get_col_elem_nb(c, col, c.close)
            position_values[k] = c.last_val_price[col] * c.last_position[col]

        total_value = np.sum(position_values) + c.last_free_cash[c.group]
        position_weights = position_values / total_value

        target_weights = size[c.i, c.from_col : c.to_col]
        deviation = np.abs(position_weights - target_weights)

        rebalancing_flag = False
        for dev in deviation:
            if dev > threshold:
                rebalancing_flag = True
                break

        if rebalancing_flag:
            order_value_out = np.empty(c.group_len, dtype=np.float64)
            sort_call_seq_nb(
                c, size, size_type=size_type, direction=direction, order_value_out=order_value_out
            )
            return (target_weights,)
        return (None,)

    def order_func_nb(c, weights_arr, size_type, direction, fees_arr, fixed_fees_arr, slippage_arr):
        if weights_arr is None:
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

    def post_order_func_nb(c, weights_arr):
        return None

    # ------------------------------------------------------------------
    # Decide order-func path
    # ------------------------------------------------------------------
    if threshold is not None:
        if use_order_func is False:
            warnings.warn(
                "use_order_func is False but threshold is set — overriding to True."
            )
        use_order_func = True
    else:
        if use_order_func is None:
            use_order_func = False
        if use_order_func:
            threshold = 0.0

    # ------------------------------------------------------------------
    # Numba JIT
    # ------------------------------------------------------------------
    if use_numba:
        os.environ["NUMBA_DISABLE_JIT"] = "0"
        pre_sim_func_nb_jit = njit(pre_sim_func_nb)
        pre_group_func_nb_jit = njit(pre_group_func_nb)
        pre_segment_func_nb_jit = njit(pre_segment_func_nb)
        order_func_nb_jit = njit(order_func_nb)
        post_order_func_nb_jit = njit(post_order_func_nb)
    else:
        os.environ["NUMBA_DISABLE_JIT"] = "1"
        pre_sim_func_nb_jit = pre_sim_func_nb
        pre_group_func_nb_jit = pre_group_func_nb
        pre_segment_func_nb_jit = pre_segment_func_nb
        order_func_nb_jit = order_func_nb
        post_order_func_nb_jit = post_order_func_nb

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
            if create_strategy_label:
                if "strategy" not in weights_df.columns.names:
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
    # Rebalancing dates
    # ------------------------------------------------------------------
    rebalancing_dates = get_rebalancing_dates(index, rebalancing_freq)

    # Prices with same column structure as weights
    _prices = prices[weights_df.columns.get_level_values(-1)]
    _prices.columns = weights_df.columns

    # ------------------------------------------------------------------
    # Order parameters
    # ------------------------------------------------------------------
    size_type_arr = np.asarray(SizeType.TargetPercent)
    direction_arr = np.asarray(Direction.Both)
    fees_arr = np.asarray(fees)
    fixed_fees_arr = np.asarray(fixed_fees)
    slippage_arr = np.asarray(slippage)

    # ------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------
    if use_order_func:
        rebalancing_mask = index.isin(rebalancing_dates)
        size_np = weights_df.values
        threshold = float(threshold)

        pf = vbt.Portfolio.from_order_func(
            _prices,
            order_func_nb_jit,
            size_type_arr,
            direction_arr,
            fees_arr,
            fixed_fees_arr,
            slippage_arr,
            pre_sim_func_nb=pre_sim_func_nb_jit,
            pre_sim_args=(rebalancing_mask,),
            pre_group_func_nb=pre_group_func_nb_jit,
            pre_segment_func_nb=pre_segment_func_nb_jit,
            pre_segment_args=(size_np, size_type_arr, direction_arr, threshold),
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
