"""The rsims adapter (core, numpy only) — the perps simulator behind the engine seam.

Owns: the rsims primitive
(:func:`quantbox.plugins.backtesting.rsims_engine.fixed_commission_backtest_with_funding`),
its parameters (``trade_buffer``, ``initial_cash``, ``margin``,
``capitalise_profits``, ``equity_basis``), its NaN policy (a NaN weight goes
FLAT), its defaults (``venue.leverage: borrow``: a margin book; it decides on
every execution bar; it charges the funding series it is handed) and the
normalisation of its long results frame into a
:class:`~quantbox.engine.base.TradedBook`. A book with several strategy slices
(a dict of frames, or MultiIndex columns) is simulated slice by slice.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pandas as pd

from .base import TRADE_COLUMNS, Costs, EngineAdapter, TradedBook, weight_turnover

logger = logging.getLogger(__name__)

#: A sweep's metric names (vectorbt attribute names) -> ``compute_backtest_metrics`` keys.
_STAT_NAMES = {
    "total_return": "total_return",
    "sharpe_ratio": "sharpe",
    "sortino_ratio": "sortino",
    "annualized_return": "cagr",
    "annualized_volatility": "annual_volatility",
    "max_drawdown": "max_drawdown",
    "calmar_ratio": "calmar",
}


def _number(key: str, value: Any, where: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{where}'{key}' must be a number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}'{key}' must be a number, got {value!r}") from exc


class RsimsAdapter(EngineAdapter):
    name = "rsims"
    distribution = "quantbox"  # rsims lives in quantbox
    extra = None
    nan_policy = "flat"
    default_leverage = "borrow"
    decides_every_bar = True
    charges_funding = True
    models_margin = True
    native_key = "rsims_results"
    PARAMS: Mapping[str, Any] = {
        "trade_buffer": 0.0,
        "initial_cash": 10000,
        "margin": 0.0,
        "capitalise_profits": False,
        "equity_basis": "rsims",
    }

    def plan_params(self, params: Mapping[str, Any], *, where: str = "") -> dict[str, Any]:
        equity_basis = str(params.get("equity_basis", "rsims"))
        if equity_basis not in ("rsims", "mtm"):
            raise ValueError(f"{where}'equity_basis' must be 'rsims' or 'mtm', got {equity_basis!r}")
        return {
            "trade_buffer": _number("trade_buffer", params.get("trade_buffer", 0.0), where),
            "initial_cash": _number("initial_cash", params.get("initial_cash", 10000), where),
            "margin": _number("margin", params.get("margin", 0.0), where),
            "capitalise_profits": bool(params.get("capitalise_profits", False)),
            "equity_basis": equity_basis,
        }

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
        """rsims trades every bar it is handed (``orders`` masks cells); ``rebalancing_freq`` and
        ``threshold`` are vectorbt's schedule and are not read. It charges ``costs.fees`` only."""
        from quantbox.metrics import compute_backtest_metrics

        params = self.check_params(params)
        slices = list(_slices(weights))
        results: dict[Any, pd.DataFrame] = {}
        values: dict[Any, pd.Series] = {}
        modelled = False
        for key, w in slices:
            p, w = _on_one_grid(prices, w)
            f, has_funding = _funding_on(funding, p)
            modelled = modelled or has_funding
            res = _simulate(p, w, f, orders, costs, cash_legs, params)
            results[key] = res
            values[key] = _equity(res)
        if len(slices) == 1:
            key = slices[0][0]
            value = values[key]
            returns = value.ffill().pct_change(fill_method=None).dropna()
            returns.name = None
            metrics = compute_backtest_metrics(returns, trading_days=trading_days)
            native: Any = results[key]
        else:
            value = pd.DataFrame(values)
            returns = value.ffill().pct_change(fill_method=None).iloc[1:]
            metrics = {}
            native = results
        return TradedBook(
            engine=self.name,
            returns=returns,
            value=value,
            weights=weights,
            turnover=weight_turnover(weights),
            trades_fn=lambda: _trades(results),
            metrics=metrics,
            native=native,
            native_key=self.native_key,
            funding_modelled=modelled,
        )

    def stats(self, book: TradedBook, names: Sequence[str], *, trading_days: int = 365) -> dict[tuple, dict[str, Any]]:
        """The sweep's metric names, from ``compute_backtest_metrics`` on each slice's returns."""
        from quantbox.metrics import compute_backtest_metrics

        returns = book.returns
        per_slice = (
            {("_single_",): returns}
            if isinstance(returns, pd.Series)
            else {(k if isinstance(k, tuple) else (k,)): returns[k] for k in returns.columns}
        )
        unknown = [m for m in names if m not in _STAT_NAMES]
        for m in unknown:
            logger.warning("rsims: metric %r unavailable; it answers %s", m, sorted(_STAT_NAMES))
        out: dict[tuple, dict[str, Any]] = {}
        for key, r in per_slice.items():
            computed = compute_backtest_metrics(r.dropna(), trading_days=trading_days)
            out[key] = {m: computed.get(_STAT_NAMES[m]) for m in names if m in _STAT_NAMES}
        return out


def _slices(weights: pd.DataFrame | dict[str, pd.DataFrame]) -> Iterator[tuple[Any, pd.DataFrame]]:
    """One frame per strategy slice: a dict's entries, or a MultiIndex's non-ticker levels."""
    if isinstance(weights, dict):
        yield from weights.items()
        return
    if weights.columns.nlevels == 1:
        yield None, weights
        return
    levels = list(range(weights.columns.nlevels - 1))
    for key, frame in weights.T.groupby(level=levels, sort=False):
        w = frame.T
        w.columns = w.columns.get_level_values(-1)
        yield key, w


def _on_one_grid(prices: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """rsims wants prices and weights on the same bars and tickers: the union of bars, prices carried."""
    tickers = list(weights.columns)
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"All tickers in weights must be present in prices (missing {missing})")
    if prices.index.equals(weights.index) and list(prices.columns) == tickers:
        return prices, weights
    index = prices.index.union(weights.index)
    p = prices.reindex(index=index, columns=tickers).ffill()
    return p, weights.reindex(index=index)


def _funding_on(funding: pd.DataFrame | None, prices: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """The funding series on the book's bars (0 where none); modelled when one was handed."""
    if funding is None or funding.empty:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns), False
    # The financing cash legs are not in a funding file: they get 0 here.
    return funding.reindex(index=prices.index, columns=prices.columns).fillna(0.0), True


def _simulate(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    funding: pd.DataFrame,
    orders: pd.DataFrame | None,
    costs: Costs,
    cash_legs: Sequence[str],
    params: Mapping[str, Any],
) -> pd.DataFrame:
    from quantbox.plugins.backtesting.rsims_engine import fixed_commission_backtest_with_funding

    return fixed_commission_backtest_with_funding(
        prices=prices,
        target_weights=weights,
        funding_rates=funding,
        commission_pct=costs.fees,
        fee_free=[c for c in cash_legs if c in weights.columns],
        orders=orders,
        **params,
    )


def _equity(results: pd.DataFrame) -> pd.Series:
    """Portfolio value per bar: cash plus the margin posted (as the quantlab validation script)."""
    margin_totals = results.groupby(results.index)["Margin"].sum().to_frame("TotalMargin")
    cash_balance = results[results["ticker"] == "Cash"][["Value"]].rename(columns={"Value": "Cash"})
    curve = cash_balance.join(margin_totals, how="left")
    curve["TotalMargin"] = curve["TotalMargin"].fillna(0)
    value = curve["Cash"] + curve["TotalMargin"]
    value.name = "portfolio_value"
    value.index.name = "date"
    return value


def _trades(results: Mapping[Any, pd.DataFrame]) -> pd.DataFrame:
    """The rows of each results frame that traded, as the seam's fill table."""
    frames = []
    for res in results.values():
        traded = res[(res["ticker"] != "Cash") & (res["Trades"].fillna(0.0) != 0.0)]
        frames.append(
            pd.DataFrame(
                {
                    "date": traded.index.to_numpy(),
                    "symbol": traded["ticker"].astype(str).to_numpy(),
                    "size": traded["Trades"].astype(float).to_numpy(),
                    "price": traded["Close"].astype(float).to_numpy(),
                    "value": traded["TradeValue"].astype(float).to_numpy(),
                    "fees": traded["Commission"].astype(float).to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(TRADE_COLUMNS))
