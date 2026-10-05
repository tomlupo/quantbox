"""The engine seam — decided weights + prices + execution timing + costs -> a traded book (docs/adr/0008).

ONE book function, :func:`simulate`, with two adapters behind it:

- ``vectorbt`` (:class:`~quantbox.engine.vectorbt.VectorbtAdapter`, the ``[vectorbt]`` extra);
- ``rsims`` (:class:`~quantbox.engine.rsims.RsimsAdapter`, numpy, in core; perps with funding).

Every backtest door goes through it: ``quantbox run`` and its variants,
``backtest()``, ``optimize()`` and the sweep. Pick the engine by name::

    from quantbox.engine import simulate
    from quantbox.execution import resolve_execution

    book = simulate(prices, weights, engine="rsims", timing=resolve_execution(None), rebalancing_freq="W-FRI")
    book.returns, book.value, book.turnover, book.trades, book.orders
    book.native  # the engine's own object: a vbt.Portfolio on vectorbt

The seam owns everything that decides what is traded and when, the same for
every engine: the execution lag (docs/adr/0005, 0006), applied in ONE place,
:func:`quantbox.engine._lag.lag_positions`; the rebalancing schedule and its
threshold (:mod:`quantbox.engine.schedule`) and the rebalancing policies
(:mod:`quantbox.engine.policy`: periodic, tranche, band, corridor), as a
per-cell orders mask; group limits on the decided book
(:mod:`quantbox.engine.groups`); the
NaN policy (:func:`materialise_nan`); the default ``venue.leverage``. An
adapter only executes the orders (:meth:`EngineAdapter.execute`) and declares
two real capability differences, ``charges_funding`` and ``models_margin``.
Nothing outside an adapter module branches on the engine name.
"""

from __future__ import annotations

from ._lag import lag_frame, lag_positions
from .base import Costs, EngineAdapter, TradedBook
from .book import NAN_POLICY, materialise_nan, simulate
from .groups import GroupLimits, apply_group_limits, resolve_group_limits
from .policy import POLICIES, RebalancePolicy, resolve_policy
from .registry import DEFAULT_ENGINE, engine_distribution, engine_names, get_engine

__all__ = [
    "DEFAULT_ENGINE",
    "NAN_POLICY",
    "POLICIES",
    "Costs",
    "EngineAdapter",
    "GroupLimits",
    "RebalancePolicy",
    "TradedBook",
    "engine_distribution",
    "engine_names",
    "get_engine",
    "lag_frame",
    "lag_positions",
    "apply_group_limits",
    "materialise_nan",
    "resolve_group_limits",
    "resolve_policy",
    "simulate",
]
