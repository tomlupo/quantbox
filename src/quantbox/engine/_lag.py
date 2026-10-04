"""THE execution lag — the one place a decided weight moves to the bar it fills on (docs/adr/0005, 0006, 0008).

Both engines are same-bar primitives: the row handed to them for bar ``t``
fills at ``close[t]``. A weight DECIDED on bar ``t`` (with data through
``close[t]``) therefore has to be moved ``lag_bars`` bars later before any
engine sees it. Every book the seam builds moves it here, through
:func:`lag_positions`:

- the scheduled book (``quantbox run``, variants) counts the lag in
  execution-calendar bars (:mod:`quantbox.engine.schedule`);
- the bar-grid book (``backtest()``, ``optimize()``, the sweep) shifts the
  weights by the same number of rows (:func:`lag_frame`), and moves a
  buy-and-hold book's one trade to the first bar a decision exists
  (:func:`lag_buy_and_hold`).

``quantbox.execution.apply_execution_lag`` delegates to :func:`lag_frame`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantbox.execution import SameBarOverride, _check_lag


def lag_positions(positions: Any, lag_bars: int, same_bar: SameBarOverride | None = None) -> np.ndarray:
    """The bar POSITION each decision fills on: ``position + lag_bars``.

    ``lag_bars`` below 1 is refused (docs/adr/0005) unless it is 0 under the
    :class:`~quantbox.execution.SameBarOverride` the resolver granted
    (docs/adr/0006). This is the only line in quantbox that applies the lag;
    deleting it turns ``tests/test_engine_seam.py`` red on every path.
    """
    _check_lag(lag_bars, same_bar)
    return np.asarray(positions, dtype=int) + int(lag_bars)


def lag_bars_of(lag_bars: int, same_bar: SameBarOverride | None = None) -> int:
    """How many bars :func:`lag_positions` moves a decision (a shift, for a frame on a bar grid)."""
    return int(lag_positions(np.zeros(1, dtype=int), lag_bars, same_bar)[0])


def lag_frame(
    weights: pd.DataFrame,
    lag_bars: int,
    *,
    same_bar: SameBarOverride | None = None,
    fill_leading: float | None = 0.0,
) -> pd.DataFrame:
    """Shift decided weights forward by the lag — row ``t`` is what the engine trades at ``close[t]``.

    The first ``lag`` rows have no decision behind them; they are set to
    ``fill_leading`` (0.0 = flat, the default) or left NaN with ``fill_leading=None``.
    """
    shift = lag_bars_of(lag_bars, same_bar)
    lagged = weights.shift(shift)
    if fill_leading is not None:
        lagged.iloc[:shift] = fill_leading
    return lagged


def lag_buy_and_hold(index: pd.Index, rebalancing_freq: Any, lag_bars: int, same_bar: SameBarOverride | None = None):
    """Move a buy-and-hold book's ONE trade to the first bar a decision exists.

    ``rebalancing_freq=None`` (buy-and-hold) trades on the engine's first bar
    only. After :func:`lag_frame` that bar is flat — no decision is behind it
    yet — so a lagged buy-and-hold would never enter. Its one trade belongs at
    the bar the bar-0 decision fills at. Every other schedule is returned
    unchanged. A window no longer than the lag has no fill bar: an empty schedule.
    """
    if rebalancing_freq is not None:
        return rebalancing_freq
    fill = lag_bars_of(lag_bars, same_bar)
    return [index[fill]] if len(index) > fill else []
