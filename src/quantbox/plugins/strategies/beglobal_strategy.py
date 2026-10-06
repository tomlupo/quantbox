"""
BeGlobal Core-Satellite Multi-Asset Strategy

Implements the BeGlobal investment methodology (https://beglobal.pl/metodologia-inwestycyjna):
- Core-Satellite approach: passive allocation (core) + active momentum (satellite)
- 14 asset classes across fixed income, equities, and alternatives
- 5 risk profiles: safe, bond_plus, mixed, profit, profit_plus
- Dual momentum, relative strength, and trend following signals
- Corridor rebalancing (configurable threshold, default 2.5%)
- Conditional volatility targeting

## Quick Start
```python
from quantbox.plugins.strategies.beglobal_strategy import BeGlobalStrategy

strategy = BeGlobalStrategy(risk_profile="mixed")
result = strategy.run(data)
print(result['simple_weights'])
```

## Data Requirements
Prices DataFrame with columns matching either ETF tickers (SPY, TLT, etc.)
or asset class names (us_stocks, us_treasury_long, etc.).
"""

from __future__ import annotations

import logging
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta, StrategyContext
from quantbox.strategy_runner import resolve_annualize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AssetClass = namedtuple("_AssetClass", ["name", "category", "etf_ticker"])

#: The money_market sleeve's default ticker (TOM-1528). It was SHV, which the pinned
#: etf-daily does not carry, and no BIL/SGOV/SHV is there either: SHY (1-3y Treasury)
#: is the shortest-duration fund it has. A dataset with a true money-market ETF sets
#: ``money_market_ticker``. SHY is also us_treasury_short's ticker: the two sleeves
#: then share one column, and money_market has no momentum series of its own.
DEFAULT_MONEY_MARKET_TICKER = "SHY"

ASSET_CLASSES: dict[str, _AssetClass] = {
    # Fixed Income
    "money_market": _AssetClass("Money Market", "fixed_income", DEFAULT_MONEY_MARKET_TICKER),
    "us_treasury_short": _AssetClass("US Treasury Short", "fixed_income", "SHY"),
    "us_treasury_medium": _AssetClass("US Treasury Medium", "fixed_income", "IEF"),
    "us_treasury_long": _AssetClass("US Treasury Long", "fixed_income", "TLT"),
    "developed_sovereigns": _AssetClass("Developed Sovereigns ex-US", "fixed_income", "BWX"),
    "corporate_ig": _AssetClass("Corporate Investment Grade", "fixed_income", "LQD"),
    "corporate_hy": _AssetClass("Corporate High Yield", "fixed_income", "HYG"),
    "em_bonds": _AssetClass("Emerging Market Bonds", "fixed_income", "EMB"),
    # Equities
    "us_stocks": _AssetClass("US Stocks", "equity", "SPY"),
    "developed_stocks": _AssetClass("Developed Markets ex-US", "equity", "EFA"),
    "em_stocks": _AssetClass("Emerging Market Stocks", "equity", "EEM"),
    # Alternatives
    "real_estate": _AssetClass("Real Estate", "alternative", "VNQ"),
    "commodities": _AssetClass("Commodities", "alternative", "DJP"),
    "gold": _AssetClass("Gold", "alternative", "GLD"),
}

RISK_PROFILE_ALLOCATIONS: dict[str, dict[str, float]] = {
    "safe": {
        "money_market": 0.60,
        "us_treasury_short": 0.30,
        "us_treasury_medium": 0.10,
    },
    "bond_plus": {
        "money_market": 0.30,
        "us_treasury_short": 0.35,
        "us_treasury_medium": 0.20,
        "corporate_ig": 0.10,
        "us_stocks": 0.05,
    },
    "mixed": {
        "money_market": 0.20,
        "us_treasury_short": 0.35,
        "us_treasury_medium": 0.15,
        "us_stocks": 0.20,
        "developed_stocks": 0.05,
        "gold": 0.05,
    },
    "profit": {
        "money_market": 0.10,
        "us_treasury_short": 0.20,
        "us_treasury_medium": 0.10,
        "us_stocks": 0.35,
        "developed_stocks": 0.10,
        "em_stocks": 0.05,
        "real_estate": 0.05,
        "gold": 0.05,
    },
    "profit_plus": {
        "money_market": 0.05,
        "us_treasury_short": 0.10,
        "us_stocks": 0.45,
        "developed_stocks": 0.15,
        "em_stocks": 0.10,
        "real_estate": 0.05,
        "commodities": 0.05,
        "gold": 0.05,
    },
}

_SAFE_ASSETS = frozenset({"money_market", "us_treasury_short"})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ticker_to_asset(money_market_ticker: str | None) -> dict[str, str]:
    """ETF ticker -> asset class key. money_market takes its ticker from the param; when
    that ticker is another class's, the other class owns the column."""
    out = {ac.etf_ticker: key for key, ac in ASSET_CLASSES.items() if key != "money_market"}
    if money_market_ticker is not None:
        out.setdefault(money_market_ticker, "money_market")
    return out


def _resolve_columns(
    prices: pd.DataFrame,
    money_market_ticker: str | None = DEFAULT_MONEY_MARKET_TICKER,
) -> dict[str, str]:
    """Build mapping from DataFrame column -> internal asset class key.

    Accepts either ETF ticker columns (SPY, TLT, ...) or asset class name
    columns (us_stocks, us_treasury_long, ...).
    """
    ticker_to_asset = _ticker_to_asset(money_market_ticker)
    col_to_asset: dict[str, str] = {}
    for col in prices.columns:
        if col in ASSET_CLASSES:
            col_to_asset[col] = col
        elif col in ticker_to_asset:
            col_to_asset[col] = ticker_to_asset[col]
    return col_to_asset


def _dual_momentum_signals(
    prices_dict: dict[str, pd.Series],
    lookback: int,
    risk_free_return: float = 0.0,
) -> dict[str, tuple]:
    """Compute dual momentum signals for each asset.

    Returns dict of asset -> (signal, score) where signal is
    "long", "neutral", or "short" and score is a float.
    """
    # Absolute momentum: return over lookback period
    mom_scores: dict[str, float] = {}
    for asset, prices in prices_dict.items():
        if len(prices) >= lookback:
            mom_scores[asset] = (prices.iloc[-1] / prices.iloc[-lookback]) - 1
        else:
            mom_scores[asset] = 0.0

    if not mom_scores:
        return {}

    # Relative momentum: percentile rank
    scores_series = pd.Series(mom_scores)
    rel_rank = scores_series.rank(ascending=False, pct=True)

    signals: dict[str, tuple] = {}
    for asset in prices_dict:
        abs_mom = mom_scores[asset]
        abs_positive = abs_mom > risk_free_return
        percentile = rel_rank[asset]

        if abs_positive and percentile >= 0.5:
            signal = "long"
            score = abs_mom
        elif not abs_positive:
            signal = "neutral"
            score = 0.0
        elif percentile < 0.25:
            signal = "short"
            score = abs_mom
        else:
            signal = "neutral"
            score = abs_mom

        signals[asset] = (signal, score)

    return signals


def _relative_strength_weights(
    prices_dict: dict[str, pd.Series],
    base_weights: dict[str, float],
    lookback_periods: list[int],
    top_n: int,
) -> dict[str, float]:
    """Adjust base weights using composite multi-period momentum ranking.

    Top N assets get +25% overweight, bottom N get -25% underweight,
    then normalize to sum to 1.
    """
    # Composite momentum: mean return across all lookback periods
    composite: dict[str, float] = {}
    for asset, prices in prices_dict.items():
        period_returns = []
        for period in lookback_periods:
            if len(prices) >= period:
                period_returns.append((prices.iloc[-1] / prices.iloc[-period]) - 1)
        composite[asset] = float(np.mean(period_returns)) if period_returns else 0.0

    ranked = sorted(composite.keys(), key=lambda a: composite[a], reverse=True)
    top_assets = set(ranked[:top_n])
    bottom_assets = set(ranked[-top_n:])

    adjusted: dict[str, float] = {}
    for asset, base_weight in base_weights.items():
        if asset in top_assets:
            adjusted[asset] = base_weight * 1.25
        elif asset in bottom_assets:
            adjusted[asset] = max(0.0, base_weight * 0.75)
        else:
            adjusted[asset] = base_weight

    total = sum(adjusted.values())
    if total > 0:
        return {k: v / total for k, v in adjusted.items()}
    return adjusted


def _trend_signals(
    prices_dict: dict[str, pd.Series],
    short_ma: int,
    long_ma: int,
) -> dict[str, str]:
    """Compute trend signals based on moving average crossover.

    Returns dict of asset -> "up" / "down" / "neutral".
    """
    signals: dict[str, str] = {}
    for asset, prices in prices_dict.items():
        if len(prices) < long_ma:
            signals[asset] = "neutral"
            continue

        sma_short = prices.rolling(short_ma).mean().iloc[-1]
        sma_long = prices.rolling(long_ma).mean().iloc[-1]
        current = prices.iloc[-1]

        if current > sma_short > sma_long:
            signals[asset] = "up"
        elif current < sma_short < sma_long:
            signals[asset] = "down"
        else:
            signals[asset] = "neutral"

    return signals


def _volatility_scalar(
    returns: pd.Series,
    target_vol: float,
    lookback: int,
    annualize: float = 252.0,
) -> float:
    """Compute volatility targeting scalar.

    Returns >1 in low-vol regimes, <1 in high-vol regimes, 1.0 otherwise.
    Caps upward scaling at 1.2x to avoid excessive leverage.

    Args:
        returns: Realized return series.
        target_vol: Target annualized volatility.
        lookback: Number of bars for vol estimation.
        annualize: Bars per year for vol annualization. Default 252 (equity
            convention). Strategy callers should derive this from the
            run's ``StrategyContext.bars_per_year`` (TOM-1448).
    """
    if len(returns) < lookback:
        return 1.0

    realized_vol = returns.tail(lookback).std() * np.sqrt(annualize)
    if realized_vol <= 0 or np.isnan(realized_vol):
        return 1.0

    vol_ratio = realized_vol / target_vol

    if vol_ratio > 1.5:
        # High vol: reduce exposure
        return target_vol / realized_vol
    elif vol_ratio < 0.5:
        # Low vol: increase slightly (capped)
        return min(1.2, target_vol / realized_vol)
    else:
        return 1.0


def _corridor_rebalance_needed(
    current: dict[str, float],
    target: dict[str, float],
    threshold: float,
) -> bool:
    """Check if any position deviates beyond the corridor threshold."""
    return any(abs(current.get(asset, 0.0) - target[asset]) > threshold for asset in target)


# ---------------------------------------------------------------------------
# Strategy plugin
# ---------------------------------------------------------------------------


@dataclass
class BeGlobalStrategy:
    """BeGlobal core-satellite multi-asset strategy with dual momentum
    and volatility targeting.
    """

    meta = PluginMeta(
        name="strategy.beglobal.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="BeGlobal core-satellite multi-asset strategy with dual momentum and volatility targeting",
        tags=("multi-asset", "etf", "core-satellite", "momentum"),
        params_schema={
            "type": "object",
            "properties": {
                "risk_profile": {
                    "enum": ["safe", "bond_plus", "mixed", "profit", "profit_plus"],
                    "description": "Base allocation profile of the core portfolio.",
                },
                "money_market_ticker": {
                    "description": (
                        "Prices column of the money_market sleeve (default SHY, the shortest Treasury fund in "
                        "etf-daily); null holds the sleeve as cash."
                    )
                },
                "core_weight": {
                    "description": "Share of the book held in the risk-profile core; the rest is the momentum satellite."
                },
                "momentum_lookback": {"description": "Bars of the dual-momentum lookback."},
                "short_momentum_lookback": {"description": "Bars of the short-term momentum lookback."},
                "strength_lookback_periods": {"description": "Lookbacks averaged into the relative-strength score."},
                "strength_top_n": {"description": "Number of strongest assets held in the satellite."},
                "trend_short_ma": {"description": "Short moving average of the trend filter."},
                "trend_long_ma": {"description": "Long moving average of the trend filter."},
                "target_volatility": {"description": "Annualised portfolio volatility target."},
                "vol_lookback": {"description": "Bars of the realised-volatility window."},
                "annualize": {
                    "description": "Bars per year for vol annualisation; null = the pipeline-derived value (fallback 252)."
                },
                "rebalance_threshold": {
                    "description": "Rebalance only when a weight drifts more than this from target."
                },
                "output_periods": {"description": "Number of most recent bars returned in the weights output."},
            },
        },
    )

    # Risk profile
    risk_profile: str = "mixed"

    # Money-market sleeve: its prices column; None = held as cash (weights then sum below 1)
    money_market_ticker: str | None = DEFAULT_MONEY_MARKET_TICKER

    # Core-satellite split
    core_weight: float = 0.70

    # Dual momentum
    momentum_lookback: int = 252
    short_momentum_lookback: int = 21

    # Relative strength
    strength_lookback_periods: list[int] = field(
        default_factory=lambda: [21, 63, 126, 252],
    )
    strength_top_n: int = 3

    # Trend following
    trend_short_ma: int = 50
    trend_long_ma: int = 200

    # Volatility targeting
    target_volatility: float = 0.10
    vol_lookback: int = 20
    annualize: float | None = None  # None = the run's StrategyContext.bars_per_year; 252.0 without one

    # Corridor rebalancing
    rebalance_threshold: float = 0.025

    # Output
    output_periods: int = 30

    def run(
        self,
        data: dict[str, Any],
        params: dict[str, Any] | None = None,
        context: StrategyContext | None = None,
    ) -> dict[str, Any]:
        """Run BeGlobal strategy.

        Args:
            data: Dict with at least ``"prices"`` (wide DataFrame).
            params: Optional parameter overrides.

        Returns:
            Dict with ``"weights"``, ``"simple_weights"``, ``"details"``.
        """
        if params:
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        # Annualisation: the strategy's explicit field wins, else the run's StrategyContext (TOM-1448).
        effective_annualize = resolve_annualize(self.annualize, params, context, owner="BeGlobalStrategy.annualize")

        prices: pd.DataFrame = data["prices"]

        # Strategy uses date-based slicing on the prices index to assemble
        # per-asset history. A non-unique index makes ``get_loc`` / ``.loc[:date]``
        # ambiguous (returns a slice instead of a scalar position) which has
        # historically masqueraded as a cryptic ``slice + int`` TypeError. Detect
        # and refuse to run with a clear message instead.
        if not prices.index.is_unique:
            raise ValueError(
                "BeGlobalStrategy requires prices with a unique index, but the "
                f"input has {len(prices.index) - prices.index.nunique()} duplicate "
                "timestamps. This typically indicates upstream data corruption "
                "(e.g. a RangeIndex coerced to datetime via "
                "pd.to_datetime(int_values))."
            )

        base_allocation = RISK_PROFILE_ALLOCATIONS.get(
            self.risk_profile,
            RISK_PROFILE_ALLOCATIONS["mixed"],
        )

        # Resolve columns to internal asset names
        mm_ticker = self.money_market_ticker
        col_to_asset = _resolve_columns(prices, mm_ticker)
        if not col_to_asset:
            raise ValueError(
                "No matching asset columns found in prices DataFrame. "
                "Expected ETF tickers (SPY, TLT, ...) or asset class names "
                "(us_stocks, us_treasury_long, ...)."
            )

        # asset_key -> column name (for output mapping); None = held as cash, not output
        asset_to_col: dict[str, str | None] = {v: k for k, v in col_to_asset.items()}
        if "money_market" not in asset_to_col:
            if mm_ticker is None:
                asset_to_col["money_market"] = None
            elif mm_ticker in prices.columns:
                # The ticker is another sleeve's column (SHY): money_market's weight lands there.
                asset_to_col["money_market"] = mm_ticker
            else:
                raise ValueError(
                    f"BeGlobalStrategy: money_market_ticker {mm_ticker!r} is not a prices column; "
                    "set money_market_ticker to a ticker the dataset carries, or null to hold the sleeve as cash."
                )

        # Build per-asset price series (keyed by internal asset name)
        asset_prices: dict[str, pd.Series] = {}
        for col, asset_key in col_to_asset.items():
            if asset_key in base_allocation or asset_key in ASSET_CLASSES:
                asset_prices[asset_key] = prices[col].dropna()

        warmup = max(self.momentum_lookback, self.trend_long_ma)
        dates = prices.index[warmup:]
        if len(dates) == 0:
            # Not enough data -- return empty weights
            empty_weights = pd.DataFrame(
                0.0,
                index=prices.index,
                columns=prices.columns,
            )
            return {
                "weights": empty_weights,
                "simple_weights": {},
                "details": {"warmup_insufficient": True},
            }

        # Pre-compute rolling portfolio returns for vol targeting
        portfolio_ret = prices[list(col_to_asset.keys())].mean(axis=1).ffill().pct_change(fill_method=None)

        satellite_weight = 1.0 - self.core_weight
        prev_weights: dict[str, float] = {}
        rows: list[dict[str, float]] = []
        row_dates: list = []

        # Iterate with explicit positional offset into ``prices`` (= ``warmup + i``)
        # rather than re-resolving via ``get_loc(date)`` per iteration. This is
        # both faster and immune to any ambiguity in the index.
        for i, date in enumerate(dates):
            pos = warmup + i

            # Slice prices up to current date. Per-asset series may be shorter
            # than ``prices`` (leading NaNs dropped for late-listing assets), so
            # we slice by date label, not by position in the full prices frame.
            prices_to_date: dict[str, pd.Series] = {}
            for asset_key, series in asset_prices.items():
                sliced = series.loc[:date]
                if len(sliced) > 0:
                    prices_to_date[asset_key] = sliced

            # -- Core weights: base_allocation * core_weight --
            core: dict[str, float] = {k: v * self.core_weight for k, v in base_allocation.items()}

            # -- Satellite weights: dual momentum -> equal weight long assets --
            mom_signals = _dual_momentum_signals(
                prices_to_date,
                self.momentum_lookback,
            )
            long_assets = [a for a, (sig, _) in mom_signals.items() if sig == "long"]

            if long_assets:
                sat_per_asset = 1.0 / len(long_assets)
                satellite: dict[str, float] = {a: sat_per_asset * satellite_weight for a in long_assets}
            else:
                satellite = {"money_market": satellite_weight}

            # -- Combine core + satellite --
            combined: dict[str, float] = {}
            all_assets = set(core.keys()) | set(satellite.keys())
            for asset in all_assets:
                combined[asset] = core.get(asset, 0.0) + satellite.get(asset, 0.0)

            # -- Volatility targeting --
            port_ret_to_date = portfolio_ret.iloc[: pos + 1].dropna()
            if len(port_ret_to_date) > self.vol_lookback:
                scalar = _volatility_scalar(
                    port_ret_to_date,
                    self.target_volatility,
                    self.vol_lookback,
                    effective_annualize,
                )
                if scalar != 1.0:
                    for asset in combined:
                        if asset not in _SAFE_ASSETS:
                            combined[asset] *= scalar
                    total = sum(combined.values())
                    if total < 1.0:
                        combined["money_market"] = combined.get("money_market", 0.0) + (1.0 - total)

            # -- Normalize --
            total = sum(combined.values())
            if total > 0:
                combined = {k: v / total for k, v in combined.items()}

            # -- Corridor rebalancing --
            if prev_weights and not _corridor_rebalance_needed(
                prev_weights,
                combined,
                self.rebalance_threshold,
            ):
                combined = prev_weights.copy()
            else:
                prev_weights = combined.copy()

            rows.append(combined)
            row_dates.append(date)

        # Build output weights DataFrame using original column names (ETF tickers
        # or asset class names, matching the input prices columns). Sleeves that
        # share a column (money_market on SHY) are summed into it; a sleeve held
        # as cash (column None) is left out, so its weight stays unallocated.
        all_asset_keys = sorted(
            {k for row in rows for k in row},
        )
        output_cols = list(dict.fromkeys(c for k in all_asset_keys if (c := asset_to_col.get(k, k)) is not None))
        col_pos = {c: j for j, c in enumerate(output_cols)}

        weights_data = np.zeros((len(rows), len(output_cols)))
        for i, row in enumerate(rows):
            for asset_key, w in row.items():
                col = asset_to_col.get(asset_key, asset_key)
                if col is not None:
                    weights_data[i, col_pos[col]] += w

        weights_df = pd.DataFrame(
            weights_data,
            index=row_dates,
            columns=output_cols,
        )

        # Latest weights as dict (non-zero only)
        latest = weights_df.iloc[-1]
        simple_weights = {k: float(v) for k, v in latest.items() if abs(v) > 1e-6}

        return {
            "weights": weights_df.tail(self.output_periods),
            "simple_weights": simple_weights,
            "details": {
                "risk_profile": self.risk_profile,
                "money_market_column": asset_to_col["money_market"],
                "base_allocation": base_allocation,
                "core_weight": self.core_weight,
                "satellite_weight": satellite_weight,
                "warmup": warmup,
                "total_dates": len(row_dates),
                "col_to_asset": col_to_asset,
            },
        }
