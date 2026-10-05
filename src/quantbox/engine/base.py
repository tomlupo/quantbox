"""The seam's types: what goes in (:class:`Costs`), what comes out (:class:`TradedBook`), and the adapter contract.

An adapter (:class:`EngineAdapter`) wraps ONE simulation library and only
EXECUTES orders (:meth:`EngineAdapter.execute`). It owns:

- its **parameters** (:meth:`EngineAdapter.plan_params`: the keys it reads,
  validated, refused when unknown);
- two **capabilities** that really differ between engines:
  :attr:`EngineAdapter.charges_funding` and :attr:`EngineAdapter.models_margin`;
- its **output normalisation** (:meth:`EngineAdapter.execute` returns a
  :class:`TradedBook`; :meth:`EngineAdapter.stats` answers a sweep's metric names).

Everything that decides WHAT is traded and WHEN is the seam's, the same for
every engine (docs/adr/0008): the lag (:mod:`quantbox.engine._lag`), the
rebalancing schedule and its threshold (:mod:`quantbox.engine.schedule`), the
NaN policy (:func:`quantbox.engine.book.materialise_nan`) and the default
``venue.leverage`` (:data:`quantbox.financing.DEFAULT_LEVERAGE`). An adapter
receives fully specified targets and a per-cell orders mask. Nothing outside
an adapter module asks which engine it is talking to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, ClassVar

import pandas as pd

from quantbox.metrics import turnover_series


@dataclass(frozen=True)
class Costs:
    """Trading costs, as fractions of traded notional (``fixed_fees`` per order, quote currency)."""

    fees: float = 0.0
    fixed_fees: float = 0.0
    slippage: float = 0.0


@dataclass
class TradedBook:
    """A simulated book — every engine's result in one shape.

    ``returns`` / ``value`` are per bar (a frame with one column per strategy
    slice when the book has several). ``weights`` is the book the engine
    held (lagged, after the schedule, deferral and leverage; MultiIndex
    columns, one slice each, when the book has several). ``orders`` is the
    per-cell orders mask the engine executed.
    ``turnover`` is ``sum |w[t] - w[t-1]|`` of those weights; ``trades`` has
    one row per fill (``date``, ``symbol``, ``size``, ``price``, ``value``,
    ``fees``). ``native`` is the engine's own object (a ``vbt.Portfolio`` on
    vectorbt, the long results frame on rsims) — the escape hatch to the wheel.
    """

    engine: str
    returns: pd.Series | pd.DataFrame
    value: pd.Series | pd.DataFrame
    weights: pd.DataFrame
    turnover: pd.Series
    #: Builds :attr:`trades` on first use (a fill table can be large; most callers never read it).
    trades_fn: Callable[[], pd.DataFrame]
    metrics: dict[str, Any]
    native: Any
    #: The name the native object travelled under before the seam (``vbt_portfolio``, ``rsims_results``).
    native_key: str
    funding_modelled: bool = False
    execution: dict[str, Any] = field(default_factory=dict)
    #: Set by the seam: the real-book prices, the orders mask, the rebalance schedule, the financing
    #: record, the calendar/timing/leverage metrics and data_validation.json (one-slice books).
    prices: pd.DataFrame | None = None
    orders: pd.DataFrame | None = None
    schedule: pd.DataFrame | None = None
    financing: dict[str, Any] | None = None
    book_metrics: dict[str, float] = field(default_factory=dict)
    data_validation: dict[str, Any] | None = None

    @cached_property
    def trades(self) -> pd.DataFrame:
        """One row per fill: ``date``, ``symbol``, ``size`` (signed units), ``price``, ``value`` (signed), ``fees``."""
        return self.trades_fn()

    @property
    def portfolio_daily(self) -> pd.DataFrame:
        """``portfolio_daily.parquet``: ``portfolio_value`` by ``date``."""
        value = self.value.iloc[:, 0] if isinstance(self.value, pd.DataFrame) else self.value
        frame = pd.DataFrame({"portfolio_value": value.to_numpy()}, index=pd.Index(value.index, name="date"))
        return frame


TRADE_COLUMNS = ("date", "symbol", "size", "price", "value", "fees")


def weight_turnover(weights: pd.DataFrame) -> pd.Series:
    """``sum |w[t] - w[t-1]|`` per bar, the first bar against a flat book (summed over strategy slices)."""
    return turnover_series(weights.select_dtypes(include="number").fillna(0.0), from_flat=True)


class EngineAdapter(ABC):
    """One simulation library behind the seam. Subclasses declare the class attributes."""

    #: The ``engine:`` value that selects this adapter.
    name: ClassVar[str]
    #: The distribution whose version IS this engine's version (run@1 ``engine.version``).
    distribution: ClassVar[str]
    #: The quantbox extra that installs it; None when it ships in core.
    extra: ClassVar[str | None] = None
    #: True: charges the funding series it is handed (perps).
    charges_funding: ClassVar[bool]
    #: True: a margin simulator — borrowing at an assumed rate of 0 needs no financing cash legs.
    models_margin: ClassVar[bool]
    #: How the native object was named in a result before the seam.
    native_key: ClassVar[str]
    #: The pipeline params this adapter reads (its own section of ``backtest.pipeline.v1``).
    PARAMS: ClassVar[Mapping[str, Any]] = {}

    @classmethod
    def installed(cls) -> bool:
        """Whether the library is importable (its extra is installed)."""
        return True

    def plan_params(self, params: Mapping[str, Any], *, where: str = "") -> dict[str, Any]:
        """This adapter's parameters out of *params* (defaults filled, types checked)."""
        return {}

    def check_params(self, engine_params: Mapping[str, Any] | None) -> dict[str, Any]:
        """Refuse a key this adapter does not own; return the params with defaults filled."""
        engine_params = dict(engine_params or {})
        unknown = sorted(set(engine_params) - set(self.PARAMS))
        if unknown:
            raise ValueError(
                f"engine {self.name!r} does not take parameter(s) {unknown}; it takes {sorted(self.PARAMS) or 'none'}"
            )
        return self.plan_params({**self.PARAMS, **engine_params})

    @abstractmethod
    def execute(
        self,
        prices: pd.DataFrame,
        targets: pd.DataFrame,
        orders: pd.DataFrame,
        costs: Costs,
        funding: pd.DataFrame | None = None,
        params: Mapping[str, Any] | None = None,
        *,
        cash_legs: Sequence[str] = (),
        trading_days: int = 365,
    ) -> TradedBook:
        """Execute the seam's orders: on a bar where ``orders`` is True for a cell, trade it to ``targets``.

        *targets* and *orders* share one index and one set of columns (MultiIndex
        columns, the ticker last, carry one strategy slice each); *prices* is
        on the same bars, one column per ticker. Row ``t`` fills at
        ``close[t]`` — the book is ALREADY lagged and scheduled. A cell with no
        order keeps its position. *targets* hold no NaN.
        """

    @abstractmethod
    def stats(self, book: TradedBook, names: Sequence[str], *, trading_days: int = 365) -> dict[tuple, dict[str, Any]]:
        """``{slice key: {metric name: value}}`` for a sweep; ``("_single_",)`` keys a one-slice book."""
