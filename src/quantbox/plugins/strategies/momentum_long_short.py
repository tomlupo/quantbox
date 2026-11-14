"""
Long-Short Momentum Strategy - Market-Neutral Crypto Factor

Systematic long-short momentum strategy for cryptocurrencies.
Long winners (top momentum) + Short losers (bottom momentum) = Market-neutral exposure.

## LLM Usage Guide

### Quick Start
```python
from quantbox.plugins.strategies import MomentumLongShortStrategy
from quantbox.plugins.datasources import BinanceDataFetcher

fetcher = BinanceDataFetcher()
data = fetcher.get_market_data(['BTC', 'ETH', 'SOL', 'BNB', 'XRP'], lookback_days=400)

strategy = MomentumLongShortStrategy()
result = strategy.run(data)

# Positive weights = long, negative weights = short
print(result['simple_weights'])
# {'BTC': 0.15, 'ETH': 0.10, 'SOL': -0.12, 'XRP': -0.08}
```

### Strategy Logic
1. **Momentum Signal**: 1M and 3M returns, z-scored cross-sectionally;
   ``skip_periods`` leaves the most recent bars out of every window (12-1 momentum)
2. **Long Leg**: Top N assets by momentum (positive weights)
3. **Short Leg**: Bottom N assets by momentum (negative weights)
4. **Weighting**: ``inverse_vol`` (default) or ``equal`` within each leg, or ``rank``
   across the whole cross-section (Asness-Moskowitz-Pedersen 2013)
5. **Net Exposure**: Configurable (0 = market-neutral, 0.5 = long-biased)
6. **Rebalancing**: Weekly or monthly

### 12-1 rank-weighted momentum (AMP 2013) on daily bars
```yaml
params:
  momentum_windows: [252]
  skip_periods: 21
  weighting: rank
```

### Risk Features
- Trend filter (only long if price > SMA)
- Vol targeting (scale positions to target portfolio vol)
- Drawdown control (reduce exposure during drawdowns)
- Short squeeze protection (limit short position size)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta, StrategyContext
from quantbox.frequency import pandas_offset_alias
from quantbox.strategy_runner import resolve_annualize
from quantbox.universe import DEFAULT_STABLECOINS

logger = logging.getLogger(__name__)


# ============================================================================
# Signal Generation
# ============================================================================


def compute_momentum_signal(
    prices: pd.DataFrame,
    windows: list[int] | None = None,
    weights: list[float] | None = None,
    skip_periods: int = 0,
) -> pd.DataFrame:
    """
    Compute composite momentum signal from multiple lookback windows.

    Each window is a lookback ending at the signal date t. ``skip_periods`` cuts
    the most recent bars off every window, so the return measured at t is
    ``P[t - skip] / P[t - window] - 1`` (``window - skip`` bars of return, none
    after ``t - skip``). 12-1 momentum on daily bars is ``windows=[252]``,
    ``skip_periods=21``; ``skip_periods=0`` is the plain ``window``-bar return.

    Args:
        prices: Price DataFrame (date x ticker)
        windows: Lookback windows in days [21=1M, 63=3M]
        weights: Weights for each window (default: equal weight)
        skip_periods: Most recent bars left out of every window (default 0)

    Returns:
        DataFrame of momentum signals
    """
    if windows is None:
        windows = [21, 63]
    if weights is None:
        weights = [1.0 / len(windows)] * len(windows)
    if isinstance(skip_periods, bool) or not isinstance(skip_periods, (int, np.integer)) or skip_periods < 0:
        raise ValueError(f"skip_periods must be an integer >= 0, got {skip_periods!r}")
    if skip_periods and skip_periods >= min(windows):
        raise ValueError(
            f"skip_periods ({skip_periods}) must be shorter than every momentum window "
            f"(shortest {min(windows)}): a window would measure no return"
        )

    filled = prices.ffill()
    if skip_periods:
        # Only bars up to t - skip_periods enter the signal at t (a lag, never a lead).
        filled = filled.shift(skip_periods)

    momentum_signals = []
    for window in windows:
        # Total return over window, minus the skipped most recent bars
        ret = filled.pct_change(periods=window - skip_periods, fill_method=None)
        momentum_signals.append(ret)

    # Weighted average of momentum signals
    composite = sum(w * m for w, m in zip(weights, momentum_signals, strict=False))

    return composite


def zscore_cross_sectional(
    signals: pd.DataFrame,
    clip: float = 3.0,
) -> pd.DataFrame:
    """
    Z-score signals cross-sectionally (per date).

    Removes cross-sectional mean and scales by std.
    Clips extreme values to limit outlier impact.

    Args:
        signals: Raw signal DataFrame
        clip: Maximum absolute z-score (clips outliers)

    Returns:
        Z-scored signals
    """
    # Cross-sectional z-score per date
    mean = signals.mean(axis=1)
    std = signals.std(axis=1).replace(0, np.nan)

    zscored = signals.sub(mean, axis=0).div(std, axis=0)

    # Clip extremes
    if clip > 0:
        zscored = zscored.clip(-clip, clip)

    return zscored


# ============================================================================
# Portfolio Construction
# ============================================================================


def construct_long_short_weights(
    signals: pd.DataFrame,
    n_long: int = 3,
    n_short: int = 3,
    net_exposure: float = 0.0,
    volatility: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construct long-short portfolio from momentum signals.

    Args:
        signals: Z-scored momentum signals
        n_long: Number of long positions
        n_short: Number of short positions
        net_exposure: Target net exposure (0 = market neutral)
        volatility: Optional vol DataFrame for inverse-vol weighting

    Returns:
        DataFrame of weights (positive = long, negative = short)
    """
    weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

    for date in signals.index:
        row = signals.loc[date].dropna()
        if len(row) < n_long + n_short:
            continue

        # Sort by signal
        sorted_signals = row.sort_values(ascending=False)

        # Long: top n_long
        long_tickers = sorted_signals.head(n_long).index.tolist()

        # Short: bottom n_short
        short_tickers = sorted_signals.tail(n_short).index.tolist()

        # Base weights (equal within leg)
        long_weight = (1 + net_exposure) / (2 * n_long)
        short_weight = (1 - net_exposure) / (2 * n_short)

        # Apply inverse volatility weighting if provided
        if volatility is not None:
            vol_row = volatility.loc[date]

            # Long leg
            long_vols = vol_row[long_tickers].replace(0, np.nan)
            long_inv_vol = 1 / long_vols
            long_inv_vol_norm = long_inv_vol / long_inv_vol.sum()
            for ticker in long_tickers:
                weights.loc[date, ticker] = long_inv_vol_norm.get(ticker, 0) * (1 + net_exposure) / 2

            # Short leg
            short_vols = vol_row[short_tickers].replace(0, np.nan)
            short_inv_vol = 1 / short_vols
            short_inv_vol_norm = short_inv_vol / short_inv_vol.sum()
            for ticker in short_tickers:
                weights.loc[date, ticker] = -short_inv_vol_norm.get(ticker, 0) * (1 - net_exposure) / 2
        else:
            # Equal weight within leg
            for ticker in long_tickers:
                weights.loc[date, ticker] = long_weight
            for ticker in short_tickers:
                weights.loc[date, ticker] = -short_weight

    return weights


def construct_rank_weights(
    signals: pd.DataFrame,
    net_exposure: float = 0.0,
) -> pd.DataFrame:
    """
    Rank weights across the whole cross-section (Asness, Moskowitz, Pedersen 2013).

    Per date, ``w_i = c * (rank(S_i) - mean_j rank(S_j))`` over the assets with a
    signal (ties take their average rank), with ``c`` chosen so the long leg is
    one dollar and the short leg minus one dollar. Every ranked asset holds a
    position (the one at the mean rank holds zero); ``n_long``/``n_short`` do
    not apply. ``net_exposure`` scales the legs to ``1 + net_exposure`` long and
    ``1 - net_exposure`` short, the same tilt as the equal and inverse-vol legs
    (which are half that size), so 0 is exactly AMP. A date with fewer than two
    distinct ranks holds nothing. Whole-frame, no loop over dates.

    Args:
        signals: Momentum signals (date x ticker); NaN = not ranked
        net_exposure: Target net exposure (0 = market neutral, AMP)

    Returns:
        DataFrame of weights (positive = long, negative = short)
    """
    ranks = signals.rank(axis=1)
    demeaned = ranks.sub(ranks.mean(axis=1), axis=0)
    # Rank minus its mean sums to zero, so the positive part equals the negative part.
    long_sum = demeaned.clip(lower=0).sum(axis=1).replace(0, np.nan)
    leg = demeaned.where(demeaned <= 0, demeaned * (1 + net_exposure))
    leg = leg.where(leg >= 0, leg * (1 - net_exposure))
    return leg.div(long_sum, axis=0).fillna(0.0)


def apply_trend_filter(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    sma_window: int = 100,
    filter_longs: bool = True,
    filter_shorts: bool = False,
) -> pd.DataFrame:
    """
    Apply trend filter to weights.

    Only long assets above their SMA, only short assets below their SMA.

    Args:
        weights: Raw weights
        prices: Price DataFrame
        sma_window: SMA window for trend detection
        filter_longs: Apply filter to long positions
        filter_shorts: Apply filter to short positions

    Returns:
        Filtered weights
    """
    sma = prices.rolling(sma_window, min_periods=sma_window).mean()
    above_sma = prices > sma
    below_sma = prices < sma

    filtered = weights.copy()

    if filter_longs:
        # Zero out longs that are below SMA
        long_mask = weights > 0
        filtered = filtered.where(~long_mask | above_sma, 0)

    if filter_shorts:
        # Zero out shorts that are above SMA
        short_mask = weights < 0
        filtered = filtered.where(~short_mask | below_sma, 0)

    return filtered


def apply_volatility_targeting(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float = 0.15,
    vol_lookback: int = 20,
    max_leverage: float = 2.0,
    annualize: float = 252.0,
) -> pd.DataFrame:
    """
    Scale positions to target portfolio volatility.

    Args:
        weights: Raw weights
        returns: Return DataFrame
        target_vol: Target annualized volatility
        vol_lookback: Lookback for vol estimation
        max_leverage: Maximum leverage multiplier
        annualize: Bars per year for vol annualization. Default 252 (equity
            convention). Strategy callers should derive this from the
            run's ``StrategyContext.bars_per_year`` (TOM-1448).

    Returns:
        Vol-targeted weights
    """
    # Estimate portfolio volatility
    port_returns = (weights.shift(1) * returns).sum(axis=1)
    realized_vol = port_returns.rolling(vol_lookback).std() * np.sqrt(annualize)

    # Scale factor
    scale = target_vol / realized_vol.replace(0, np.nan)
    scale = scale.clip(0.1, max_leverage)

    return weights.mul(scale, axis=0)


def apply_rebalancing(
    weights: pd.DataFrame,
    frequency: str = "W",
) -> pd.DataFrame:
    """
    Apply rebalancing frequency.

    Args:
        weights: Daily weights
        frequency: 'D' daily, 'W' weekly, 'M' monthly

    Returns:
        Rebalanced weights
    """
    if frequency == "D":
        return weights

    # Get rebalance dates ('M' -> 'ME' for pandas 3: quantbox.frequency.pandas_offset_alias)
    rebalanced = weights.resample(pandas_offset_alias(frequency)).first()

    # Forward fill to daily
    rebalanced = rebalanced.reindex(weights.index).ffill()

    return rebalanced


# ============================================================================
# Strategy Class
# ============================================================================

_WEIGHTINGS = ("equal", "inverse_vol", "rank")


@dataclass
class MomentumLongShortStrategy:
    """
    Long-Short Momentum Strategy - Market-Neutral Factor.

    Systematic strategy that goes:
    - Long on winners (highest momentum)
    - Short on losers (lowest momentum)

    Can be configured for market-neutral or long-biased exposure.

    ## Quick Start
    ```python
    strategy = MomentumLongShortStrategy()
    result = strategy.run(data)  # data from BinanceDataFetcher

    # Positive = long, negative = short
    print(result['simple_weights'])
    ```
    """

    meta = PluginMeta(
        name="strategy.momentum_long_short.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Long-short momentum strategy - market-neutral crypto factor",
        tags=("crypto", "momentum", "long-short"),
        params_schema={
            "type": "object",
            "properties": {
                "momentum_windows": {"description": "Lookbacks of the momentum score."},
                "momentum_weights": {"description": "Weights of each momentum window; null = equal."},
                "n_long": {"description": "Number of top-momentum assets held long."},
                "n_short": {"description": "Number of bottom-momentum assets held short."},
                "net_exposure": {"description": "Net exposure target: 0 = market neutral, 0.5 = long biased."},
                "vol_lookback": {"description": "Bars of the volatility estimate."},
                "trend_filter_window": {"description": "Moving average of the trend filter."},
                "enable_trend_filter": {"description": "Gate longs/shorts on the trend filter."},
                "enable_vol_targeting": {"description": "Scale the book to target_vol."},
                "target_vol": {"description": "Annualised volatility target when vol targeting is on."},
                "max_leverage": {"description": "Cap on gross leverage after vol targeting."},
                "skip_periods": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Most recent bars left out of every momentum window (12-1 on daily bars: "
                    "momentum_windows [252], skip_periods 21); 0 = none.",
                },
                "weighting": {
                    "enum": ["equal", "inverse_vol", "rank"],
                    "description": "Position weighting: inverse_vol or equal within the n_long/n_short legs "
                    "(each leg (1 +/- net_exposure)/2), or rank across every asset (AMP 2013: rank minus its "
                    "mean, $1 long and $1 short at net_exposure 0; n_long/n_short do not apply).",
                },
                "rebalance_frequency": {"enum": ["D", "W", "M"], "description": "Signal rebalance frequency."},
                "annualize": {
                    "description": "Bars per year for vol annualisation; null = the pipeline-derived value (fallback 252)."
                },
                "output_periods": {"description": "Number of most recent bars returned in the weights output."},
                "exclude_tickers": {"description": "Tickers never traded (stablecoins by default)."},
            },
        },
    )

    # Momentum parameters
    momentum_windows: list[int] = field(default_factory=lambda: [21, 63])
    momentum_weights: list[float] | None = None
    skip_periods: int = 0  # most recent bars left out of every window (21 on daily = 12-1)

    # Portfolio parameters
    n_long: int = 3
    n_short: int = 3
    net_exposure: float = 0.0  # 0 = market neutral, 0.5 = long biased
    weighting: str = "inverse_vol"  # "inverse_vol", "equal" or "rank"

    # Risk parameters
    vol_lookback: int = 20
    trend_filter_window: int = 100
    enable_trend_filter: bool = True
    enable_vol_targeting: bool = False
    target_vol: float = 0.15
    max_leverage: float = 2.0

    # Rebalancing
    rebalance_frequency: str = "W"  # 'D', 'W', 'M'

    # Annualization — None = the run's StrategyContext.bars_per_year; 252.0 without one
    annualize: float | None = None

    # Output
    output_periods: int = 30
    exclude_tickers: list[str] = field(default_factory=lambda: DEFAULT_STABLECOINS.copy())

    def describe(self) -> dict[str, Any]:
        """Describe strategy for LLM introspection."""
        return {
            "name": "MomentumLongShort",
            "type": "long_short_factor",
            "purpose": "Market-neutral momentum strategy for crypto",
            "parameters": {
                "momentum_windows": self.momentum_windows,
                "skip_periods": self.skip_periods,
                "weighting": self.weighting,
                "n_long": self.n_long,
                "n_short": self.n_short,
                "net_exposure": self.net_exposure,
            },
            "signals": "Z-scored cross-sectional momentum (1M + 3M returns)",
            "risk": {
                "trend_filter": self.enable_trend_filter,
                "vol_targeting": self.enable_vol_targeting,
                "target_vol": self.target_vol,
            },
        }

    def run(
        self,
        data: dict[str, pd.DataFrame],
        params: dict[str, Any] | None = None,
        context: StrategyContext | None = None,
    ) -> dict[str, Any]:
        """
        Run long-short momentum strategy.

        Args:
            data: Dict with 'prices', 'volume', 'market_cap'
            params: Optional parameter overrides

        Returns:
            Dict with:
            - 'weights': DataFrame with signed weights
            - 'simple_weights': Latest weights as dict
            - 'details': Intermediate calculations
            - 'exposure': Long/short/net exposure
        """
        if params:
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        # Annualisation: the strategy's explicit field wins, else the run's StrategyContext (TOM-1448).
        effective_annualize = resolve_annualize(
            self.annualize, params, context, owner="MomentumLongShortStrategy.annualize"
        )

        if self.weighting not in _WEIGHTINGS:
            raise ValueError(f"weighting must be one of {list(_WEIGHTINGS)}, got {self.weighting!r}")

        prices = data["prices"]

        # Filter out excluded tickers
        valid_tickers = [t for t in prices.columns if t not in self.exclude_tickers]
        prices = prices[valid_tickers]

        logger.info(f"Running MomentumLongShort on {len(valid_tickers)} tickers")

        # 1. Compute momentum signals
        momentum = compute_momentum_signal(
            prices,
            self.momentum_windows,
            self.momentum_weights,
            skip_periods=self.skip_periods,
        )

        # 2. Z-score cross-sectionally
        signals = zscore_cross_sectional(momentum)

        # 3. Compute volatility for weighting
        returns = prices.ffill().pct_change(fill_method=None)
        volatility = returns.rolling(self.vol_lookback).std() * np.sqrt(effective_annualize)

        # 4. Construct long-short weights
        if self.weighting == "rank":
            # Rank the raw momentum: the z-score's clip at +/-3 would tie the extremes.
            weights = construct_rank_weights(momentum, net_exposure=self.net_exposure)
        else:
            weights = construct_long_short_weights(
                signals,
                n_long=self.n_long,
                n_short=self.n_short,
                net_exposure=self.net_exposure,
                volatility=volatility if self.weighting == "inverse_vol" else None,
            )

        # 5. Apply trend filter
        if self.enable_trend_filter:
            weights = apply_trend_filter(
                weights,
                prices,
                sma_window=self.trend_filter_window,
                filter_longs=True,
                filter_shorts=False,
            )

        # 6. Apply vol targeting
        if self.enable_vol_targeting:
            weights = apply_volatility_targeting(
                weights,
                returns,
                target_vol=self.target_vol,
                vol_lookback=self.vol_lookback,
                max_leverage=self.max_leverage,
                annualize=effective_annualize,
            )

        # 7. Apply rebalancing
        weights = apply_rebalancing(weights, self.rebalance_frequency)

        # 8. Calculate exposure
        latest = weights.iloc[-1].dropna()
        long_exposure = latest[latest > 0].sum()
        short_exposure = abs(latest[latest < 0].sum())
        net_exp = long_exposure - short_exposure

        # 9. Simple weights dict
        simple = latest[abs(latest) > 0.001].to_dict()

        return {
            "weights": weights.tail(self.output_periods),
            "simple_weights": simple,
            "details": {
                "momentum": momentum,
                "signals": signals,
                "volatility": volatility,
            },
            "exposure": {
                "long": float(long_exposure),
                "short": float(short_exposure),
                "net": float(net_exp),
                "gross": float(long_exposure + short_exposure),
            },
        }

    def get_latest_weights(self, result: dict[str, Any]) -> dict[str, float]:
        """Extract latest weights as dict."""
        return result["simple_weights"]


# ============================================================================
# Standard Interface
# ============================================================================


def run(data: dict, params: dict = None, context: StrategyContext | None = None) -> dict:
    """Standard strategy interface."""
    strategy = MomentumLongShortStrategy()
    return strategy.run(data, params, context=context)
