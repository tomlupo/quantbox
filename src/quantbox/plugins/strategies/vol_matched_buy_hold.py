"""Vol-matched buy-and-hold benchmark strategy.

Synthesises a constant-vol-targeted version of a single asset's buy-and-hold
return stream. Implements the Robuxio TrendCatcher v2 notebook cells 124-127:

    scale_t        = target_annual_vol / (sigma_d * sqrt(trading_days))
    weight[t, ticker] = scale_t
    weight[t, other]  = 0

With ``vol_lookback=None`` (default), ``sigma_d`` is the full-sample daily
return std — non-causal but matches the notebook's benchmark construction
(``annualized_vol_prices`` in cell 125 uses ``daily_returns.std()``). Pass
an integer ``vol_lookback`` for a causal rolling-window version.
``vol_power: 2`` squares that scale (``(target / sigma_ann) ** 2``, weight 1 where
the annualised vol equals the target): inverse variance instead of inverse vol.

``rv_window: ME`` switches to Moreira-Muir (2017), as H59's line plugin
(quantbox-lab ``research/replication-volmanaged-mm2017/strategy.py``) computes it.
Per calendar month t, at its last bar (the decision bar), with data through that bar:

    RV2_t      = sum_d x_d^2 of the month's daily returns x_d (``rv_form: centred``
                 subtracts the month's mean first)
    vol_t      = RV2_t (``vol_power: 2``) or sqrt(RV2_t) (``vol_power: 1``)
    f_t        = the month's compounded return
    weight_t   = c / vol_t, held until the next decision bar; the weights start
                 at the first decision bar (no rows before it)

``c`` is std(f_s) / std(f_s / vol_{s-1}): ``c_rule: full`` ONE value over
``c_window`` (that window's future enters the LEVEL of the weight, never its
timing), ``c_rule: expanding`` the same ratio over the pairs through t, from
``min_months`` pairs on. ``first_decision`` drops the months before it.
``target_annual_vol`` and ``trading_days`` do not enter this weight.
Refused, not computed: a month whose RV2 enters a weight or a pair and is not > 0 and finite
(a flat month of stale quotes would make the weight c / 0), and a ``c_window`` the loaded
data does not cover from its first month to its last (c would be fitted on fewer months).

The plugin never shifts a weight: the engine fills next bar (ADR-0005).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta, StrategyContext
from quantbox.params_schema import resolve_params_schema
from quantbox.strategy_runner import resolve_annualize

RV_WINDOWS = (None, "ME")
VOL_POWERS = (1, 2)
C_RULES = ("full", "expanding")
RV_FORMS = ("raw_excess", "centred")


def monthly_inputs(series: pd.Series, rv_form: str = "raw_excess", vol_power: int = 2) -> pd.DataFrame:
    """Per calendar month t: its decision bar (last bar), RV2_t, vol_t, f_t and the pair f_t / vol_{t-1}."""
    if rv_form not in RV_FORMS:
        raise ValueError(f"vol_matched_buy_hold: unknown rv_form {rv_form!r}; use one of {RV_FORMS}")
    px = series.dropna()
    idx = px.index
    # Months from the wall date: a tz-aware index groups by its local calendar, as H59 does after tz_localize(None).
    month = (idx.tz_localize(None) if idx.tz is not None else idx).to_period("M")
    x = px.pct_change(fill_method=None)
    dev = x - x.groupby(month).transform("mean") if rv_form == "centred" else x
    rv2 = (dev * dev).groupby(month).sum(min_count=1)
    out = pd.DataFrame(
        {
            "decision_bar": pd.Series(idx, index=idx).groupby(month).last(),
            "rv2": rv2,
            "vol": rv2 if vol_power == 2 else np.sqrt(rv2),
            "f": px.groupby(month).last().pct_change(fill_method=None),
        }
    )
    out["f_unscaled"] = out["f"] / out["vol"].shift(1)
    return out


def scale_constant(m: pd.DataFrame, rule: str, window: tuple[str, str] | None, min_months: int) -> pd.Series:
    """c per month: ONE value over ``window`` (full) or the expanding ratio through month t (no future data)."""
    pairs = m.loc[m["f"].notna() & m["f_unscaled"].notna(), ["f", "f_unscaled"]]
    if rule == "full":
        if window is None or len(window) != 2:
            raise ValueError("vol_matched_buy_hold: c_rule 'full' needs c_window: [first_month, last_month]")
        lo, hi = pd.Period(window[0], "M"), pd.Period(window[1], "M")
        fit = pairs.loc[(pairs.index >= lo) & (pairs.index <= hi)]
        if len(fit) < 2:
            raise ValueError(f"vol_matched_buy_hold: c_window {list(window)} holds {len(fit)} months of pairs")
        return pd.Series(fit["f"].std() / fit["f_unscaled"].std(), index=m.index)
    if rule == "expanding":
        sd = pairs.expanding(min_periods=int(min_months)).std()
        return (sd["f"] / sd["f_unscaled"]).reindex(m.index).ffill()
    raise ValueError(f"vol_matched_buy_hold: unknown c_rule {rule!r}; use one of {C_RULES}")


@dataclass
class VolMatchedBuyHoldStrategy:
    meta = PluginMeta(
        name="strategy.vol_matched_buy_hold.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.2.0",
        schema_version="v1",
        description="Buy-and-hold a single asset, scaled to a target annualized volatility.",
        tags=("benchmark", "buy-and-hold", "vol-matched"),
        params_schema={
            "type": "object",
            "properties": {
                "ticker": {"description": "Asset held."},
                "target_annual_vol": {"description": "Annualised volatility the position is scaled to."},
                "vol_lookback": {"description": "Bars of the volatility estimate; null = full history."},
                "trading_days": {"description": "Bars per year; null = the pipeline-derived value (fallback 252)."},
                "vol_power": {
                    "enum": list(VOL_POWERS),
                    "description": "1 = weight on 1/vol (default), 2 = weight on 1/variance.",
                },
                "rv_window": {
                    "enum": list(RV_WINDOWS),
                    "description": "null = daily-std scale (default); ME = monthly realised variance, "
                    "Moreira-Muir c / RV2 decided at each month's last bar.",
                },
                "c_rule": {"enum": list(C_RULES), "description": "rv_window ME: c over c_window (full) or expanding."},
                "c_window": {
                    "description": "rv_window ME, c_rule full: [first_month, last_month] the one c is fitted on, "
                    "e.g. ['1926-07', '2015-12'].",
                },
                "min_months": {"description": "rv_window ME, c_rule expanding: (f, f/vol) pairs before the first c."},
                "first_decision": {
                    "description": "rv_window ME: first month kept (YYYY-MM); null = the first in the data."
                },
                "rv_form": {
                    "enum": list(RV_FORMS),
                    "description": "rv_window ME: raw_excess = sum x^2 (default); centred = sum (x - month mean)^2.",
                },
            },
        },
    )

    ticker: str = "BTC"
    target_annual_vol: float = 0.25
    vol_lookback: int | None = None
    trading_days: int | None = None  # None = the run's StrategyContext.bars_per_year; 252 without one
    vol_power: int = 1
    rv_window: str | None = None
    c_rule: str = "full"
    c_window: list[str] | None = None
    min_months: int = 36
    first_decision: str | None = None
    rv_form: str = "raw_excess"

    @property
    def min_lookback_periods(self) -> int:
        return self.vol_lookback if self.vol_lookback is not None else 2

    def run(
        self, data: dict[str, Any], params: dict[str, Any] | None = None, context: StrategyContext | None = None
    ) -> dict[str, Any]:
        if params:
            # A key the schema does not name is a typo, not a default; `_`-keys are the pipeline's own.
            known = set(resolve_params_schema(type(self))["properties"])
            unknown = sorted(k for k in params if k not in known and not k.startswith("_"))
            if unknown:
                raise ValueError(f"vol_matched_buy_hold: unknown params {unknown}; use {sorted(known)}")
            for k, v in params.items():
                if k in known:
                    setattr(self, k, v)

        if self.vol_power not in VOL_POWERS:
            raise ValueError(f"vol_matched_buy_hold: vol_power {self.vol_power!r}; use one of {VOL_POWERS}")
        if self.rv_window not in RV_WINDOWS:
            raise ValueError(f"vol_matched_buy_hold: rv_window {self.rv_window!r}; use one of {RV_WINDOWS}")
        if self.rv_window is not None and self.vol_lookback is not None:
            raise ValueError(
                "vol_matched_buy_hold: vol_lookback is the daily-std window; with rv_window ME leave it null"
            )

        prices: pd.DataFrame = data["prices"]
        if self.ticker not in prices.columns:
            raise ValueError(
                f"vol_matched_buy_hold: ticker {self.ticker!r} not present in price universe "
                f"(have {len(prices.columns)} columns)"
            )
        if self.rv_window == "ME":
            return self._run_monthly(prices)

        # Annualisation: the strategy's explicit field wins, else the run's StrategyContext (TOM-1448).
        effective_td = resolve_annualize(
            self.trading_days, params, context, owner="VolMatchedBuyHoldStrategy.trading_days"
        )

        rets = prices[self.ticker].ffill().pct_change(fill_method=None)
        sqrt_td = float(np.sqrt(effective_td))

        if self.vol_lookback is None:
            sigma_d = float(rets.std())
            if not np.isfinite(sigma_d) or sigma_d <= 0:
                raise ValueError(f"vol_matched_buy_hold: degenerate daily-return std={sigma_d} for {self.ticker!r}")
            scale = self.target_annual_vol / (sigma_d * sqrt_td)
            scale_series = pd.Series(scale, index=prices.index, dtype=float)
        else:
            sigma_d_series = rets.rolling(int(self.vol_lookback)).std()
            scale_series = self.target_annual_vol / (sigma_d_series * sqrt_td)
            scale_series = scale_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if self.vol_power == 2:
            scale_series = scale_series**2

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        weights[self.ticker] = scale_series

        # Zero out bars where the asset has no price (pre-listing / delisted).
        weights = weights.where(prices.notna(), 0.0)

        return {
            "weights": weights,
            "details": {
                "strategy": "vol_matched_buy_hold.v1",
                "ticker": self.ticker,
                "target_annual_vol": self.target_annual_vol,
                "vol_lookback": self.vol_lookback,
                "trading_days": effective_td,
                "vol_power": self.vol_power,
                "scale_first": float(scale_series.iloc[0]) if len(scale_series) else float("nan"),
                "scale_last": float(scale_series.iloc[-1]) if len(scale_series) else float("nan"),
            },
        }

    def _run_monthly(self, prices: pd.DataFrame) -> dict[str, Any]:
        """rv_window ME: weight c / vol_t at each month's decision bar, held to the next (module docstring)."""
        m = monthly_inputs(prices[self.ticker].sort_index(), self.rv_form, self.vol_power)
        window = tuple(self.c_window) if self.c_window is not None else None
        loaded = (m.index[0], m.index[-1])
        partial = (
            window is not None
            and len(window) == 2
            and (loaded[0] > pd.Period(window[0], "M") or loaded[1] < pd.Period(window[1], "M"))
        )
        if self.c_rule == "full" and partial:
            raise ValueError(
                f"vol_matched_buy_hold: the loaded data ({loaded[0]}..{loaded[1]}) does not cover "
                f"c_window {list(window)}; load all of it (prices.lookback_days)"
            )
        first = pd.Period(self.first_decision, "M") if self.first_decision is not None else loaded[0]
        # Every vol that enters a weight (months >= first) or a pair (vol_{t-1}, so one month earlier)
        # must be > 0 and finite. Only the data's first month may have none: no return before its first bar.
        rv2 = m.loc[m.index >= first - 1, "rv2"]
        bad = rv2.index[~(np.isfinite(rv2) & (rv2 > 0)) & ~((rv2.index == loaded[0]) & rv2.isna())]
        if len(bad):
            raise ValueError(
                f"vol_matched_buy_hold: RV2 is not > 0 and finite in {[str(b) for b in bad]} for {self.ticker!r} "
                "(a month of flat or stale quotes); its weight would be c / 0"
            )
        m = m.loc[m.index >= first]
        m["c"] = scale_constant(m, self.c_rule, window, self.min_months)
        m["w"] = m["c"] / m["vol"]
        decided = m.dropna(subset=["w"])
        if decided.empty:
            raise ValueError("vol_matched_buy_hold: no decision; not enough months for the scale constant")

        w = pd.Series(decided["w"].to_numpy(), index=pd.DatetimeIndex(decided["decision_bar"]))
        # Held between decisions, a weight moves only on a decision bar. Nothing before the first decision:
        # the book starts there, as H59's does, so the engine never measures flat pre-decision bars.
        px = prices.sort_index().loc[w.index[0] :]
        held = w.reindex(px.index).ffill()
        weights = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        weights[self.ticker] = held
        weights = weights.where(px.notna(), 0.0)

        monthly = m.assign(decision_bar=m["decision_bar"].astype(str)).set_axis(m.index.astype(str))
        return {
            "weights": weights,
            "details": {
                "strategy": "vol_matched_buy_hold.v1",
                "ticker": self.ticker,
                "rv_window": self.rv_window,
                "vol_power": self.vol_power,
                "c_rule": self.c_rule,
                "c_window": list(window) if window is not None else None,
                "min_months": self.min_months,
                "first_decision": self.first_decision,
                "rv_form": self.rv_form,
                "first_decision_bar": str(w.index[0]),
                "c_last": float(decided["c"].iloc[-1]),
                "scale_last": float(held.iloc[-1]) if len(held) else float("nan"),
                "monthly": monthly,
            },
        }
