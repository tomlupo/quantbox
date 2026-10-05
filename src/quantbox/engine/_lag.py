"""THE execution lag — the one place a decided weight moves to the bar it fills on (docs/adr/0005, 0006, 0008).

Both engines are same-bar primitives: the row handed to them for bar ``t``
fills at ``close[t]``. A weight DECIDED on bar ``t`` (with data through
``close[t]``) therefore has to be moved ``lag_bars`` bars later before any
engine sees it. The one book function (:func:`quantbox.engine.simulate`)
moves it here, through :func:`lag_positions`, counted in execution-calendar
bars (:mod:`quantbox.engine.schedule`; on ``schedule: bars`` every price bar
is one). A buy-and-hold book's one decision moves with it.

:func:`lag_frame` shifts a frame by the same number of rows, for a caller
outside the seam: ``quantbox.execution.apply_execution_lag`` delegates to it
(the L0 signal helpers).
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
