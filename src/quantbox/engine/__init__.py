"""The engine seam — decided weights + prices + execution timing + costs -> a traded book (docs/adr/0008).

ONE interface to book simulation, with two adapters behind it:

- ``vectorbt`` (:class:`~quantbox.engine.vectorbt.VectorbtAdapter`, the ``[vectorbt]`` extra);
- ``rsims`` (:class:`~quantbox.engine.rsims.RsimsAdapter`, numpy, in core; perps with funding).

Every backtest door goes through it: ``quantbox run`` and its variants
(:func:`simulate_book`), ``backtest()``, ``optimize()`` and the sweep
(:func:`simulate_weights`). Pick the engine by name::

    from quantbox.engine import simulate_weights
    from quantbox.execution import resolve_execution

    book = simulate_weights(prices, weights, engine="rsims", timing=resolve_execution(None))
    book.returns, book.value, book.turnover, book.trades
    book.native  # the engine's own object: a vbt.Portfolio on vectorbt

The execution lag (docs/adr/0005, 0006) is applied in ONE place,
:func:`quantbox.engine._lag.lag_positions`, before any adapter sees a book.
Nothing outside an adapter module branches on the engine name; an adapter
declares its NaN policy, defaults and parameters (:class:`EngineAdapter`).
"""

from __future__ import annotations

from ._lag import lag_buy_and_hold, lag_frame, lag_positions
from .base import Costs, EngineAdapter, TradedBook
from .book import simulate_book, simulate_weights
from .registry import DEFAULT_ENGINE, engine_distribution, engine_names, get_engine

__all__ = [
    "DEFAULT_ENGINE",
    "Costs",
    "EngineAdapter",
    "TradedBook",
    "engine_distribution",
    "engine_names",
    "get_engine",
    "lag_buy_and_hold",
    "lag_frame",
    "lag_positions",
    "simulate_book",
    "simulate_weights",
]
