"""The vectorbt adapter (the ``[vectorbt]`` extra) — open-source vectorbt behind the engine seam.

Owns: the vectorbt primitive (:func:`quantbox.plugins.backtesting.vectorbt_engine.run`),
its NaN policy (a NaN weight HOLDS the last target), its default leverage
(``normalize``: vectorbt cannot borrow without the financing cash legs), the
fill-gap check (did the engine hold the book it was handed?) and the
normalisation of a ``vbt.Portfolio`` into a :class:`~quantbox.engine.base.TradedBook`.
The native portfolio stays reachable as ``book.native``.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .base import TRADE_COLUMNS, Costs, EngineAdapter, TradedBook, weight_turnover

logger = logging.getLogger(__name__)


class VectorbtAdapter(EngineAdapter):
    name = "vectorbt"
    distribution = "vectorbt"
    extra = "vectorbt"
    nan_policy = "hold"
    default_leverage = "normalize"
    decides_every_bar = False
    charges_funding = False
    models_margin = False
    native_key = "vbt_portfolio"
    #: Forwarded to ``vectorbt_engine.run``; the defaults are that function's.
    PARAMS: Mapping[str, Any] = {"use_numba": True, "use_order_func": None, "create_strategy_label": True}

    @classmethod
    def installed(cls) -> bool:
        try:
            return importlib.util.find_spec("vectorbt") is not None
        except ModuleNotFoundError:  # a finder that blocks the module raises instead of answering None
            return False

    def plan_params(self, params: Mapping[str, Any], *, where: str = "") -> dict[str, Any]:
        out = {
            "use_numba": params.get("use_numba", True),
            "use_order_func": params.get("use_order_func"),
            "create_strategy_label": params.get("create_strategy_label", True),
        }
        for key, value in out.items():
            if not (isinstance(value, bool) or (value is None and key == "use_order_func")):
                raise ValueError(f"{where}'{key}' must be true or false, got {value!r}")
        return out

    def run(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame | dict[str, pd.DataFrame],
        *,
        costs: Costs,
        orders: pd.DataFrame | None = None,
        rebalancing_freq: Any = 1,
        threshold: float | None = None,
        funding: pd.DataFrame | None = None,
        cash_legs: Sequence[str] = (),
        params: Mapping[str, Any] | None = None,
        trading_days: int = 365,
    ) -> TradedBook:
        from quantbox.plugins.backtesting.metrics import compute_backtest_metrics
        from quantbox.plugins.backtesting.vectorbt_engine import run as run_vectorbt

        params = self.check_params(params)
        columns = weights.columns if isinstance(weights, pd.DataFrame) else []
        legs = [c for c in cash_legs if c in columns]
        if orders is not None:
            pf = run_vectorbt(
                prices,
                weights,
                orders=orders,
                threshold=threshold,
                fees=costs.fees,
                fixed_fees=costs.fixed_fees,
                slippage=costs.slippage,
                fee_free=legs,
                residual_legs=tuple(cash_legs) if len(legs) == len(cash_legs) and legs else (),
                **params,
            )
        else:
            pf = run_vectorbt(
                prices,
                weights,
                rebalancing_freq=rebalancing_freq,
                threshold=threshold,
                fees=costs.fees,
                fixed_fees=costs.fixed_fees,
                slippage=costs.slippage,
                **params,
            )
        metrics = compute_backtest_metrics(pf, trading_days=trading_days)
        if orders is not None:
            metrics.update(self._fill_gaps(pf, prices, weights, orders, costs, threshold, cash_legs))
        return TradedBook(
            engine=self.name,
            returns=pf.returns(),
            value=pf.value(),
            weights=weights,
            turnover=weight_turnover(weights),
            trades_fn=lambda: _trades(pf),
            metrics=metrics,
            native=pf,
            native_key=self.native_key,
            funding_modelled=False,  # vectorbt charges no funding
        )

    @staticmethod
    def _fill_gaps(
        pf: Any,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        orders: pd.DataFrame,
        costs: Costs,
        threshold: float | None,
        cash_legs: Sequence[str],
    ) -> dict[str, float]:
        """Measured, not assumed: did the engine hold the book it was handed after each rebalance?

        The backstop for every way a book can ask for more cash than it has. A
        threshold run skips rebalances by design: there, only the bars the engine
        traded count. The cash legs are the engine's residual by construction;
        the real cells are the check.
        """
        from quantbox.plugins.backtesting.vectorbt_engine import rebalance_fill_gaps

        real_orders = orders.drop(columns=[c for c in cash_legs if c in orders.columns])
        bars = prices.index[orders.any(axis=1).to_numpy()]
        if threshold is not None:
            traded_rows = sorted(set(pf.orders.values["idx"].tolist()))
            bars = bars.intersection(prices.index[traded_rows])
        gaps = rebalance_fill_gaps(pf, weights, bars, real_orders)
        allowed = 1e-6 + 2.0 * (costs.fees + costs.slippage) * gaps["turnover"] + (1e-3 if costs.fixed_fees else 0.0)
        under = gaps["gap"] > allowed
        if under.any():
            logger.warning(
                "ENGINE: on %d of %d rebalance bar(s) the vectorbt book held is NOT the target "
                "(max sum|held - target| %.4f on %s) — buys were cut for lack of cash",
                int(under.sum()),
                len(gaps),
                gaps["gap"].max(),
                gaps["gap"].idxmax(),
            )
        return {
            "engine_underfilled_rebalances": float(under.sum()),
            "engine_max_fill_gap": float(gaps["gap"].max()) if len(gaps) else 0.0,
        }

    def stats(self, book: TradedBook, names: Sequence[str], *, trading_days: int = 365) -> dict[tuple, dict[str, Any]]:
        """vectorbt's own metrics by attribute name (``pf.deep_getattr``), per strategy slice.

        A multi-slice book (MultiIndex weight columns) answers a Series indexed by
        ``(label, *level values)``; the key is the level values.
        """
        pf = book.native
        # vbt's wrapper.freq accessor calls pd.Timedelta(<Day>), which raises on recent pandas.
        if pf.wrapper.index.freq is not None:
            pf.wrapper.index.freq = None
        out: dict[tuple, dict[str, Any]] = {}
        for m in names:
            try:
                v = pf.deep_getattr(m)
            except (AttributeError, KeyError) as e:
                logger.warning("vectorbt: metric %r unavailable: %s", m, e)
                continue
            if isinstance(v, pd.Series):
                if isinstance(v.index, pd.MultiIndex):
                    for idx_tuple, val in v.items():
                        key = tuple(idx_tuple[1:]) if len(idx_tuple) > 1 else (idx_tuple[0],)
                        out.setdefault(key, {})[m] = _coerce(val)
                else:
                    for slice_id, val in v.items():
                        out.setdefault((slice_id,), {})[m] = _coerce(val)
            else:
                out.setdefault(("_single_",), {})[m] = _coerce(v)
        return out


def _coerce(v: Any) -> Any:
    """A numpy scalar as a Python native, so a grid serialises cleanly."""
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, TypeError):
            pass
    return v


def _trades(pf: Any) -> pd.DataFrame:
    """``pf.orders`` as the seam's fill table."""
    rec = pf.orders.records_readable
    if rec.empty:
        return pd.DataFrame(columns=list(TRADE_COLUMNS))
    sign = rec["Side"].astype(str).str.lower().map({"buy": 1.0, "sell": -1.0}).fillna(0.0)
    size = rec["Size"].astype(float) * sign
    symbol = rec["Column"].map(lambda c: c[-1] if isinstance(c, tuple) else c)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(rec["Timestamp"]).to_numpy(),
            "symbol": symbol.astype(str).to_numpy(),
            "size": size.to_numpy(),
            "price": rec["Price"].astype(float).to_numpy(),
            "value": (size * rec["Price"].astype(float)).to_numpy(),
            "fees": rec["Fees"].astype(float).to_numpy(),
        }
    )
