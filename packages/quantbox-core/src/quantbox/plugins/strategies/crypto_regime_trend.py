"""
Crypto Regime Trend Strategy - BTC Regime-Based Long/Short Trend Following

Multi-asset trend-following strategy that conditions on BTC regime:
- Long regime (BTC > MA): go long trending assets
- Short regime (BTC < MA): go short trending assets
- Multi-window ensemble (V2): average signals across window pairs
- Universe: top N by mcap, then by volume, separate long/short pools
- Weighting: equal weight or inverse ATR

Based on Robuxio "Trend Catcher" research (V1+V2).

## Quick Start
```python
from quantbox.plugins.strategies import CryptoRegimeTrendStrategy

strategy = CryptoRegimeTrendStrategy()
result = strategy.run(data)

# Positive weights = long, negative = short
print(result['simple_weights'])
```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta

from .crypto_trend import (
    DEFAULT_STABLECOINS,
    compute_volatility_scalers,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

DEFAULT_WINDOW_PAIRS = [(10, 25), (20, 50), (40, 100), (100, 250)]


# ============================================================================
# BTC Regime Detection
# ============================================================================


def compute_btc_regime(
    btc_prices: pd.Series,
    regime_window: int,
) -> pd.DataFrame:
    """
    Compute BTC regime based on moving average crossover.

    Args:
        btc_prices: BTC price series
        regime_window: MA window for regime detection

    Returns:
        DataFrame with 'long_regime' and 'short_regime' boolean columns
    """
    ma = btc_prices.rolling(window=regime_window, min_periods=regime_window).mean()
    long_regime = (btc_prices > ma).astype(float)
    short_regime = (btc_prices < ma).astype(float)

    return pd.DataFrame(
        {
            "long_regime": long_regime,
            "short_regime": short_regime,
        },
        index=btc_prices.index,
    )


# ============================================================================
# Trend Signal Computation
# ============================================================================


def compute_trend_signals(
    prices: pd.DataFrame,
    trend_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-asset trend signals based on price vs MA.

    Args:
        prices: Price DataFrame (date x ticker)
        trend_window: MA window for trend detection

    Returns:
        (long_signal, short_signal) - each a DataFrame of 0/1
    """
    ma = prices.rolling(window=trend_window, min_periods=trend_window).mean()
    long_signal = (prices > ma).astype(float)
    short_signal = (prices < ma).astype(float)
    return long_signal, short_signal


def compute_ensemble_regime_signals(
    prices: pd.DataFrame,
    btc_prices: pd.Series,
    window_pairs: list[tuple[int, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute ensemble regime signals across multiple window pairs.

    Each pair is (regime_window, trend_window). The final signal is the
    mean across all pairs.

    Args:
        prices: Price DataFrame
        btc_prices: BTC price series
        window_pairs: List of (regime_window, trend_window) tuples

    Returns:
        (combined_long, combined_short) averaged across pairs
    """
    long_accum = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    short_accum = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for regime_w, trend_w in window_pairs:
        regime = compute_btc_regime(btc_prices, regime_w)
        long_sig, short_sig = compute_trend_signals(prices, trend_w)

        # Combined: long = long_regime & long_signal
        combined_long = long_sig.mul(regime["long_regime"], axis=0)
        combined_short = short_sig.mul(regime["short_regime"], axis=0)

        long_accum += combined_long
        short_accum += combined_short

    n = len(window_pairs)
    return long_accum / n, short_accum / n


# ============================================================================
# Universe Selection
# ============================================================================


def select_regime_universe(
    market_cap: pd.DataFrame,
    volume: pd.DataFrame,
    prices: pd.DataFrame,
    long_max: int,
    short_max: int,
    coins_to_trade: int = 30,
    exclude_tickers: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select separate long and short universes.

    1. Exclude stablecoins/wrapped tokens
    2. Top `coins_to_trade` by market cap
    3. Within those, top `long_max` / `short_max` by dollar volume

    Args:
        market_cap: Market cap DataFrame
        volume: Volume DataFrame
        prices: Price DataFrame
        long_max: Max long positions
        short_max: Max short positions
        coins_to_trade: Initial mcap filter size
        exclude_tickers: Tickers to exclude

    Returns:
        (long_universe, short_universe) DataFrames of 0/1 flags
    """
    if exclude_tickers is None:
        exclude_tickers = DEFAULT_STABLECOINS

    valid_tickers = [t for t in prices.columns if t not in exclude_tickers]
    if not valid_tickers:
        empty = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        return empty, empty.copy()

    # Market cap rank
    mc = market_cap[valid_tickers]
    mc_rank = mc.rank(axis=1, ascending=False, method="min")
    mc_mask = mc_rank <= coins_to_trade

    # Dollar volume within mcap filter
    dollar_vol = (prices[valid_tickers] * volume[valid_tickers]).where(mc_mask)
    vol_rank = dollar_vol.rank(axis=1, ascending=False, method="min")

    def _expand(mask_valid):
        full = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        full[valid_tickers] = mask_valid.astype(float)
        return full

    long_universe = _expand(vol_rank <= long_max)
    short_universe = _expand(vol_rank <= short_max)

    return long_universe, short_universe


# ============================================================================
# Weighting
# ============================================================================


def compute_atr_weights(
    prices: pd.DataFrame,
    atr_window: int = 14,
) -> pd.DataFrame:
    """
    Compute inverse-ATR normalized weights.

    Uses absolute daily returns as ATR proxy (close-only data).

    Args:
        prices: Price DataFrame
        atr_window: Rolling window for ATR computation

    Returns:
        DataFrame of inverse-ATR weights, normalized per row to sum to 1
    """
    abs_returns = prices.pct_change().abs()
    atr = abs_returns.rolling(window=atr_window, min_periods=atr_window).mean()
    inv_atr = 1.0 / atr.replace(0, np.nan)
    # Normalize per row
    row_sum = inv_atr.sum(axis=1).replace(0, np.nan)
    return inv_atr.div(row_sum, axis=0).fillna(0)


# ============================================================================
# Strategy Class
# ============================================================================


@dataclass
class CryptoRegimeTrendStrategy:
    """
    Crypto Regime Trend Catcher - BTC Regime-Based Long/Short.

    Conditions trend signals on BTC regime:
    - Bull regime: go long assets trending up
    - Bear regime: go short assets trending down
    - Optional multi-window ensemble for robustness

    Positive weights = long, negative weights = short.
    """

    meta = PluginMeta(
        name="strategy.crypto_regime_trend.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="BTC regime-based long/short trend following with multi-window ensemble",
        tags=("crypto", "trend", "regime", "long-short"),
    )

    # Regime parameters
    regime_window: int = 50
    trend_window: int = 20
    btc_ticker: str = "BTC"

    # Ensemble parameters
    use_ensemble: bool = True
    window_pairs: list[tuple[int, int]] = field(default_factory=lambda: list(DEFAULT_WINDOW_PAIRS))

    # Universe parameters
    long_max: int = 10
    short_max: int = 20
    coins_to_trade: int = 30
    exclude_tickers: list[str] = field(default_factory=lambda: DEFAULT_STABLECOINS.copy())

    # Weighting
    weighting: str = "equal"  # "equal" or "inverse_atr"
    atr_window: int = 14

    # Vol targeting (optional, reuse crypto_trend infra)
    vol_targets: list[Any] = field(default_factory=lambda: ["off"])
    tranches: list[int] = field(default_factory=lambda: [1])
    vol_lookback: int = 60

    # Output
    output_periods: int = 30
    normalize_weights: bool = True

    # Param aliases for quantlab compat
    _PARAM_ALIASES: dict[str, str] = field(
        default_factory=lambda: {
            "filtered_coins_market_cap": "coins_to_trade",
            "portfolio_coins_max_long": "long_max",
            "portfolio_coins_max_short": "short_max",
            "last_x_days": "output_periods",
            "periods": "output_periods",
        },
        repr=False,
    )

    def describe(self) -> dict[str, Any]:
        """Describe strategy for LLM introspection."""
        return {
            "name": "CryptoRegimeTrend",
            "type": "regime_trend_long_short",
            "purpose": "BTC regime-conditioned long/short trend following",
            "parameters": {
                "regime_window": self.regime_window,
                "trend_window": self.trend_window,
                "use_ensemble": self.use_ensemble,
                "long_max": self.long_max,
                "short_max": self.short_max,
                "weighting": self.weighting,
            },
            "signals": "Price vs MA conditioned on BTC regime, multi-window ensemble",
        }

    def run(
        self,
        data: dict[str, pd.DataFrame],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run regime trend strategy.

        Args:
            data: Dict with 'prices', 'volume', 'market_cap'
            params: Optional parameter overrides

        Returns:
            Dict with 'weights', 'simple_weights', 'details', 'exposure'
        """
        # Apply param overrides
        if params:
            for key, value in params.items():
                attr = self._PARAM_ALIASES.get(key, key)
                if hasattr(self, attr):
                    setattr(self, attr, value)

        prices = data["prices"]
        volume = data["volume"]
        market_cap = data["market_cap"]

        if self.btc_ticker not in prices.columns:
            raise ValueError(f"BTC ticker '{self.btc_ticker}' not found in price data")

        btc_prices = prices[self.btc_ticker]

        logger.info(f"Running CryptoRegimeTrend on {len(prices.columns)} tickers, ensemble={self.use_ensemble}")

        # 1. Compute signals
        if self.use_ensemble:
            long_sig, short_sig = compute_ensemble_regime_signals(
                prices,
                btc_prices,
                self.window_pairs,
            )
        else:
            regime = compute_btc_regime(btc_prices, self.regime_window)
            ls, ss = compute_trend_signals(prices, self.trend_window)
            long_sig = ls.mul(regime["long_regime"], axis=0)
            short_sig = ss.mul(regime["short_regime"], axis=0)

        # 2. Universe selection
        long_univ, short_univ = select_regime_universe(
            market_cap,
            volume,
            prices,
            self.long_max,
            self.short_max,
            self.coins_to_trade,
            self.exclude_tickers,
        )

        # 3. Apply universe mask
        long_weights = long_sig * long_univ
        short_weights = short_sig * short_univ

        # 4. Weighting
        if self.weighting == "inverse_atr":
            atr_w = compute_atr_weights(prices, self.atr_window)
            long_weights = long_weights * atr_w
            short_weights = short_weights * atr_w

        # 5. Normalize within each leg
        if self.normalize_weights:
            long_sum = long_weights.sum(axis=1).replace(0, np.nan)
            short_sum = short_weights.sum(axis=1).replace(0, np.nan)
            long_weights = long_weights.div(long_sum, axis=0).fillna(0)
            short_weights = short_weights.div(short_sum, axis=0).fillna(0)

        # 6. Combine: positive = long, negative = short
        combined = long_weights - short_weights

        # 7. Optional vol targeting via crypto_trend infra
        numeric_targets = [vt for vt in self.vol_targets if isinstance(vt, (int, float))]
        has_off = any(vt == "off" for vt in self.vol_targets if isinstance(vt, str))

        if numeric_targets or has_off:
            scalers = {}
            if numeric_targets:
                scalers = compute_volatility_scalers(
                    prices,
                    numeric_targets,
                    self.vol_lookback,
                )
            if has_off:
                scalers["off"] = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)

            # Build multi-index weights using construct_weights
            # We pass the combined signal (long - short) through the scaler framework
            from .crypto_trend import construct_weights as _cw

            multi_weights = _cw(
                combined,
                long_univ.clip(upper=1) + short_univ.clip(upper=1),
                scalers,
                self.tranches,
                normalize=False,
            )
            output_weights = multi_weights.tail(self.output_periods)
        else:
            output_weights = combined.tail(self.output_periods)

        # 8. Simple weights for latest day
        if isinstance(output_weights.columns, pd.MultiIndex):
            # Pick first vol target / first tranche
            vt0 = output_weights.columns.get_level_values("vol_target")[0]
            t0 = output_weights.columns.get_level_values("tranches")[0]
            simple_df = output_weights.xs((vt0, t0), axis=1, level=("vol_target", "tranches"))
        else:
            simple_df = output_weights

        latest = simple_df.iloc[-1].dropna()
        simple = latest[abs(latest) > 0.001].to_dict()

        # 9. Exposure stats
        long_exposure = float(latest[latest > 0].sum())
        short_exposure = float(abs(latest[latest < 0].sum()))

        return {
            "weights": output_weights,
            "simple_weights": simple,
            "details": {
                "long_signal": long_sig,
                "short_signal": short_sig,
                "long_universe": long_univ,
                "short_universe": short_univ,
                "combined": combined,
            },
            "exposure": {
                "long": long_exposure,
                "short": short_exposure,
                "net": long_exposure - short_exposure,
                "gross": long_exposure + short_exposure,
            },
        }

    def get_latest_weights(self, result: dict[str, Any]) -> dict[str, float]:
        """Extract latest weights as dict."""
        return result["simple_weights"]


# ============================================================================
# Standard Interface
# ============================================================================


def run(data: dict, params: dict = None) -> dict:
    """Standard strategy interface."""
    strategy = CryptoRegimeTrendStrategy()
    return strategy.run(data, params)
