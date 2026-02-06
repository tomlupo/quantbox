"""
Cross-Asset Momentum (XSMOM) with Core-Satellite Strategy

Systematic cross-sectional momentum strategy with volatility parity weighting
and core-satellite portfolio construction.

Based on Rockbridge Growth research.

## Algorithm
1. Multi-window momentum: compute returns over [21, 63, 126, 189, 252] days
2. Cross-sectional z-score: standardize returns across tickers per date
3. Winsorize: clip at configurable percentile tails
4. Rank & select top N: top assets by composite z-scored momentum
5. EWMA volatility: exponential weighted vol, lambda=0.94
6. Volatility parity: inverse EWMA vol weighting within selected assets
7. Trend filter: price > SMA(100) binary filter
8. Core-satellite: final = core_weight * passive + (1-core_weight) * active

## Quick Start
```python
from quantbox.plugins.strategies import CrossAssetMomentumStrategy

strategy = CrossAssetMomentumStrategy()
result = strategy.run(data)

# Weights include risk-off allocation to USDT
print(result['simple_weights'])
```
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import logging

from quantbox.contracts import PluginMeta

logger = logging.getLogger(__name__)

DEFAULT_STABLECOINS = [
    'USDT', 'USDC', 'BUSD', 'TUSD', 'DAI', 'MIM', 'USTC', 'FDUSD',
    'USDP', 'GUSD', 'FRAX', 'LUSD', 'USDD', 'PYUSD', 'USD1', 'USDJ',
    'EUR', 'EURC', 'EURT', 'EURS', 'PAXG', 'XAUT', 'WBTC', 'WETH',
    'BETH', 'ETHW', 'CBBTC', 'CBETH', 'BFUSD', 'AEUR',
]

DEFAULT_MOMENTUM_WINDOWS = [21, 63, 126, 189, 252]


# ============================================================================
# Multi-Window Momentum
# ============================================================================

def compute_multi_window_momentum(
    prices: pd.DataFrame,
    windows: List[int],
) -> pd.DataFrame:
    """
    Compute composite momentum as the mean of per-window total returns.

    Each window's returns are z-scored cross-sectionally before averaging,
    so short and long lookback windows contribute equally.

    Args:
        prices: Price DataFrame (date x ticker)
        windows: List of lookback windows in days

    Returns:
        Composite z-scored momentum DataFrame
    """
    zscored_list = []
    for w in windows:
        ret = prices.pct_change(periods=w)
        # Cross-sectional z-score per window
        mean = ret.mean(axis=1)
        std = ret.std(axis=1).replace(0, np.nan)
        z = ret.sub(mean, axis=0).div(std, axis=0)
        zscored_list.append(z)

    composite = pd.concat(zscored_list).groupby(level=0).mean()
    return composite


# ============================================================================
# Z-Score & Winsorize
# ============================================================================

def zscore_and_winsorize(
    signals: pd.DataFrame,
    pct: float = 0.05,
) -> pd.DataFrame:
    """
    Cross-sectional z-score then winsorize at given percentile tails.

    Args:
        signals: Raw signal DataFrame
        pct: Percentile for clipping (0.05 = 5th/95th)

    Returns:
        Winsorized z-scored signals
    """
    mean = signals.mean(axis=1)
    std = signals.std(axis=1).replace(0, np.nan)
    z = signals.sub(mean, axis=0).div(std, axis=0)

    # Winsorize per row
    lower = z.quantile(pct, axis=1)
    upper = z.quantile(1 - pct, axis=1)
    clipped = z.clip(lower=lower, upper=upper, axis=0)

    return clipped


# ============================================================================
# EWMA Volatility
# ============================================================================

def compute_ewma_volatility(
    prices: pd.DataFrame,
    ewma_lambda: float = 0.94,
    min_periods: int = 200,
    annualize: float = 365.0,
) -> pd.DataFrame:
    """
    Compute EWMA volatility (RiskMetrics-style).

    Args:
        prices: Price DataFrame
        ewma_lambda: Decay factor (0.94 = standard RiskMetrics)
        min_periods: Minimum observations before producing values
        annualize: Days per year for annualization

    Returns:
        Annualized EWMA volatility DataFrame
    """
    returns = prices.pct_change()
    sq_returns = returns ** 2

    # EWMA of squared returns
    span = 2.0 / (1.0 - ewma_lambda) - 1.0
    ewma_var = sq_returns.ewm(span=span, min_periods=min_periods).mean()
    ewma_vol = np.sqrt(ewma_var) * np.sqrt(annualize)

    return ewma_vol


# ============================================================================
# Trend Filter
# ============================================================================

def compute_trend_filter(
    prices: pd.DataFrame,
    sma_window: int = 100,
) -> pd.DataFrame:
    """
    Binary trend filter: 1 if price > SMA, 0 otherwise.

    Args:
        prices: Price DataFrame
        sma_window: SMA window

    Returns:
        DataFrame of 0/1 trend filter values
    """
    sma = prices.rolling(window=sma_window, min_periods=sma_window).mean()
    return (prices > sma).astype(float)


# ============================================================================
# Rank & Select
# ============================================================================

def rank_and_select_top_n(
    signals: pd.DataFrame,
    top_n: int,
    trend_filter: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Rank assets by signal and select top N, optionally filtered by trend.

    Args:
        signals: Composite momentum signals
        top_n: Number of assets to select
        trend_filter: Optional 0/1 filter (only select assets passing filter)

    Returns:
        DataFrame of 0/1 selection mask
    """
    if trend_filter is not None:
        filtered = signals * trend_filter
    else:
        filtered = signals

    # Only consider positive momentum
    filtered = filtered.where(filtered > 0, np.nan)

    # Rank descending (1 = highest)
    rank = filtered.rank(axis=1, ascending=False, method="min")
    selection = (rank <= top_n).astype(float).fillna(0)

    return selection


# ============================================================================
# Volatility Parity Weights
# ============================================================================

def compute_volatility_parity_weights(
    selection: pd.DataFrame,
    vol: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inverse volatility weights within selected assets.

    Args:
        selection: 0/1 selection mask
        vol: Volatility DataFrame

    Returns:
        Normalized inverse-vol weights (sums to 1 per row)
    """
    inv_vol = (1.0 / vol.replace(0, np.nan)) * selection
    row_sum = inv_vol.sum(axis=1).replace(0, np.nan)
    weights = inv_vol.div(row_sum, axis=0).fillna(0)
    return weights


# ============================================================================
# Core-Satellite Construction
# ============================================================================

def apply_core_satellite(
    active: pd.DataFrame,
    passive: pd.DataFrame,
    core_weight: float = 0.6,
    risk_off_ticker: str = "USDT",
) -> pd.DataFrame:
    """
    Combine active and passive portfolios with core-satellite blend.

    Unallocated active weight goes to risk-off asset.

    Args:
        active: Active (momentum) weights — sums to <= 1
        passive: Passive weights (equal weight or market cap)
        core_weight: Weight allocated to passive (core), remainder to active (satellite)
        risk_off_ticker: Ticker for unallocated weight

    Returns:
        Combined weights DataFrame
    """
    satellite_weight = 1.0 - core_weight

    # Scale each component
    passive_scaled = passive * core_weight
    active_scaled = active * satellite_weight

    combined = passive_scaled + active_scaled

    # Unallocated active weight → risk-off
    active_sum = active_scaled.sum(axis=1)
    unallocated = satellite_weight - active_sum
    unallocated = unallocated.clip(lower=0)

    if risk_off_ticker in combined.columns:
        combined[risk_off_ticker] = combined[risk_off_ticker] + unallocated
    else:
        combined[risk_off_ticker] = unallocated

    return combined


# ============================================================================
# Strategy Class
# ============================================================================

@dataclass
class CrossAssetMomentumStrategy:
    """
    Cross-Asset Momentum (XSMOM) with Core-Satellite.

    Systematic strategy combining:
    - Multi-window cross-sectional momentum
    - EWMA volatility parity weighting
    - Trend filter (SMA)
    - Core-satellite blend with passive allocation

    ## Quick Start
    ```python
    strategy = CrossAssetMomentumStrategy()
    result = strategy.run(data)
    print(result['simple_weights'])
    ```
    """

    meta = PluginMeta(
        name="strategy.cross_asset_momentum.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Cross-asset momentum (XSMOM) with core-satellite portfolio construction",
        tags=("crypto", "momentum", "xsmom", "core-satellite"),
    )

    # Momentum parameters
    momentum_windows: List[int] = field(
        default_factory=lambda: DEFAULT_MOMENTUM_WINDOWS.copy()
    )
    winsorize_pct: float = 0.05
    top_n: int = 4

    # Volatility parameters
    ewma_lambda: float = 0.94
    ewma_min_periods: int = 200

    # Trend filter
    trend_filter_window: int = 100
    enable_trend_filter: bool = True

    # Core-satellite
    core_weight: float = 0.6
    risk_off_ticker: str = "USDT"

    # Universe filtering
    exclude_tickers: List[str] = field(default_factory=lambda: DEFAULT_STABLECOINS.copy())

    # Output
    output_periods: int = 30

    # Param aliases for backward compat
    _PARAM_ALIASES: Dict[str, str] = field(default_factory=lambda: {
        "top_n_assets": "top_n",
        "last_x_days": "output_periods",
        "periods": "output_periods",
    }, repr=False)

    def describe(self) -> Dict[str, Any]:
        """Describe strategy for LLM introspection."""
        return {
            "name": "CrossAssetMomentum",
            "type": "xsmom_core_satellite",
            "purpose": "Cross-sectional momentum with core-satellite blend",
            "parameters": {
                "momentum_windows": self.momentum_windows,
                "top_n": self.top_n,
                "core_weight": self.core_weight,
                "ewma_lambda": self.ewma_lambda,
                "trend_filter_window": self.trend_filter_window,
            },
            "signals": "Multi-window z-scored momentum, EWMA vol parity, SMA trend filter",
        }

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run XSMOM strategy.

        Args:
            data: Dict with 'prices', 'volume', 'market_cap'
            params: Optional parameter overrides

        Returns:
            Dict with 'weights', 'simple_weights', 'details'
        """
        # Apply param overrides
        if params:
            for key, value in params.items():
                attr = self._PARAM_ALIASES.get(key, key)
                if hasattr(self, attr):
                    setattr(self, attr, value)

        prices = data["prices"]

        # Filter excluded tickers
        valid_tickers = [t for t in prices.columns if t not in self.exclude_tickers]
        prices_filtered = prices[valid_tickers]

        logger.info(
            f"Running CrossAssetMomentum on {len(valid_tickers)} tickers, "
            f"top_n={self.top_n}, core_weight={self.core_weight}"
        )

        # 1. Multi-window momentum
        raw_momentum = compute_multi_window_momentum(
            prices_filtered, self.momentum_windows,
        )

        # 2. Z-score & winsorize
        signals = zscore_and_winsorize(raw_momentum, self.winsorize_pct)

        # 3. Trend filter
        trend_filter = None
        if self.enable_trend_filter:
            trend_filter = compute_trend_filter(
                prices_filtered, self.trend_filter_window,
            )

        # 4. Rank & select top N
        selection = rank_and_select_top_n(signals, self.top_n, trend_filter)

        # 5. EWMA volatility
        ewma_vol = compute_ewma_volatility(
            prices_filtered, self.ewma_lambda, self.ewma_min_periods,
        )

        # 6. Volatility parity weights (active component)
        active_weights = compute_volatility_parity_weights(selection, ewma_vol)

        # 7. Passive weights (equal weight across all valid assets passing trend filter)
        if trend_filter is not None:
            n_passing = trend_filter.sum(axis=1).replace(0, np.nan)
            passive_weights = trend_filter.div(n_passing, axis=0).fillna(0)
        else:
            n_assets = len(valid_tickers)
            passive_weights = pd.DataFrame(
                1.0 / n_assets, index=prices_filtered.index,
                columns=prices_filtered.columns,
            )

        # 8. Core-satellite blend
        # Expand to include risk-off ticker if not in valid_tickers
        all_cols = list(prices.columns) if self.risk_off_ticker in prices.columns else (
            list(prices_filtered.columns) + [self.risk_off_ticker]
        )
        active_expanded = active_weights.reindex(columns=all_cols, fill_value=0)
        passive_expanded = passive_weights.reindex(columns=all_cols, fill_value=0)

        combined = apply_core_satellite(
            active_expanded, passive_expanded,
            self.core_weight, self.risk_off_ticker,
        )

        # 9. Extract simple weights
        output = combined.tail(self.output_periods)
        latest = output.iloc[-1].dropna()
        simple = latest[abs(latest) > 0.001].to_dict()

        return {
            "weights": output,
            "simple_weights": simple,
            "details": {
                "raw_momentum": raw_momentum,
                "signals": signals,
                "selection": selection,
                "ewma_vol": ewma_vol,
                "active_weights": active_weights,
                "passive_weights": passive_weights,
                "trend_filter": trend_filter,
            },
        }

    def get_latest_weights(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Extract latest weights as dict."""
        return result["simple_weights"]


# ============================================================================
# Backward-Compatible Wrappers
# ============================================================================

def cross_asset_momentum(
    prices: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    market_cap: Optional[pd.DataFrame] = None,
    momentum_windows: List[int] = None,
    vol_lookback: int = 20,
    trend_filter_window: int = 100,
    top_n_assets: int = 5,
    rebalance_frequency: str = 'W',
    atr_window: int = 14,
    stop_loss_atr_multiplier: float = 2.0,
    enable_trend_filter: bool = True,
    enable_stop_loss: bool = False,
    enable_hedging: bool = False,
    hedge_threshold: float = -0.05,
    hedge_assets: List[str] = None,
    last_x_days: int = 30,
    min_momentum_threshold: float = 0.0,
) -> dict:
    """
    Backward-compatible wrapper around CrossAssetMomentumStrategy.

    Maps old parameter names to new strategy class.
    """
    if momentum_windows is None:
        momentum_windows = [21, 63]
    if hedge_assets is None:
        hedge_assets = ['USDT', 'USDC']

    strategy = CrossAssetMomentumStrategy(
        momentum_windows=momentum_windows,
        top_n=top_n_assets,
        trend_filter_window=trend_filter_window,
        enable_trend_filter=enable_trend_filter,
        output_periods=last_x_days,
    )
    data = {"prices": prices, "volume": volume or prices, "market_cap": market_cap or prices}
    return strategy.run(data)


def run(data: dict, params: dict = None) -> dict:
    """Standard strategy interface."""
    strategy = CrossAssetMomentumStrategy()
    return strategy.run(data, params)
