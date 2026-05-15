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

from ._universe import select_universe
from .crypto_trend import (
    DEFAULT_STABLECOINS,
    compute_inv_vol_track,
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
    volume_is_dollar: bool = True,
    volume_rolling_window: int = 1,
    min_listing_days: int = 0,
    hysteresis_rank_band: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select separate long and short universes.

    Delegates to the shared :func:`_universe.select_universe` which already
    handles missing market-cap data (e.g. Hyperliquid).

    Args:
        volume_is_dollar: ``True`` when ``volume`` is already in USD/USDT notional
            (the convention for the curated Binance and Hyperliquid datasets;
            see :class:`CarverTrendFollowingStrategy` for the same default).
            Setting this to ``False`` causes :func:`select_universe` to compute
            ``prices × volume`` which double-counts price when the input is
            already dollar-denominated and corrupts the volume rank.
        volume_rolling_window: rolling-mean window for the volume rank (default
            ``1`` = legacy daily-spot behaviour). Pass ``30`` for the
            best-practice 30-day smoother.
        min_listing_days: cool-off period after a coin's first valid price
            observation (default ``0``). Pass ``60`` for the best-practice
            new-listing buffer.
        hysteresis_rank_band: sticky-membership band (default ``0``). Pass
            ``5`` for the best-practice ±5 rank band that lets coins
            drift to ``top_by_mcap + 5`` / ``top_by_volume + 5`` before
            being kicked out.

    Returns:
        (long_universe, short_universe) DataFrames of 0/1 flags
    """
    mcap = market_cap if not market_cap.empty else None
    long_universe = select_universe(
        prices,
        volume,
        market_cap=mcap,
        top_by_mcap=coins_to_trade,
        top_by_volume=long_max,
        exclude_tickers=exclude_tickers,
        volume_is_dollar=volume_is_dollar,
        volume_rolling_window=volume_rolling_window,
        min_listing_days=min_listing_days,
        hysteresis_rank_band=hysteresis_rank_band,
    )
    short_universe = select_universe(
        prices,
        volume,
        market_cap=mcap,
        top_by_mcap=coins_to_trade,
        top_by_volume=short_max,
        exclude_tickers=exclude_tickers,
        volume_is_dollar=volume_is_dollar,
        volume_rolling_window=volume_rolling_window,
        min_listing_days=min_listing_days,
        hysteresis_rank_band=hysteresis_rank_band,
    )
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
    # Volume convention. Binance/Hyperliquid curated datasets are in USD
    # notional already, so we should NOT multiply by price again (see
    # :class:`CarverTrendFollowingStrategy` for the same default).
    volume_is_dollar: bool = True
    # Best-practice universe-construction knobs (all default to legacy
    # notebook-faithful behaviour; set to recommended values for production).
    #   * ``volume_rolling_window=30``: rank by 30-day mean volume, not 1-day
    #     spot. Drops listing-day spike noise.
    #   * ``min_listing_days=60``: exclude coins for 60 days after their first
    #     valid price. Avoids trading the new-listing pump.
    #   * ``hysteresis_rank_band=5``: a coin stays in the universe at
    #     ranks up to ``top_N + 5`` once it has entered. Cuts boundary churn.
    volume_rolling_window: int = 1
    min_listing_days: int = 0
    hysteresis_rank_band: int = 0

    # Weighting
    weighting: str = "equal"  # "equal" or "inverse_atr"
    atr_window: int = 14

    # Vol targeting (optional, reuse crypto_trend infra)
    vol_targets: list[Any] = field(default_factory=lambda: ["off"])
    tranches: list[int] = field(default_factory=lambda: [1])
    vol_lookback: int = 60
    # Notebook-v2-faithful knobs (defaults preserve prior behaviour):
    #  * ``position_weight``: when set, replace per-leg row-normalisation with a
    #    fixed weight per active position (notebook cell 108:
    #    ``weights = signals * (1/portfolio_coins_max) * coins_universe``).
    #    Typical v2 value is ``1 / long_max``.
    #  * ``inv_vol_track``: when ``True``, append a 4th vol_target track keyed
    #    ``'inv_vol'`` computed from the ``off`` track via
    #    :func:`compute_inv_vol_track` (notebook cells 110-115). Requires
    #    ``"off"`` in ``vol_targets``.
    #  * ``clip_vol_scaler``: pass-through to
    #    :func:`compute_volatility_scalers`. Set to ``None`` for exact notebook
    #    math (no clipping).
    position_weight: float | None = None
    inv_vol_track: bool = False
    clip_vol_scaler: tuple[float, float] | None = (0.1, 10.0)

    # Output
    output_periods: int = 30
    normalize_weights: bool = True

    # Param aliases for quantlab compat
    _PARAM_ALIASES: dict[str, str] = field(
        default_factory=lambda: {
            "filtered_coins_market_cap": "coins_to_trade",
            "portfolio_coins_max_long": "long_max",
            "portfolio_coins_max_short": "short_max",
            "portfolio_coins_max": "long_max",
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
                "position_weight": self.position_weight,
                "inv_vol_track": self.inv_vol_track,
                "clip_vol_scaler": self.clip_vol_scaler,
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
            volume_is_dollar=self.volume_is_dollar,
            volume_rolling_window=self.volume_rolling_window,
            min_listing_days=self.min_listing_days,
            hysteresis_rank_band=self.hysteresis_rank_band,
        )

        # 3. Apply universe mask
        long_weights = long_sig * long_univ
        short_weights = short_sig * short_univ

        # 4. Weighting
        if self.weighting == "inverse_atr":
            atr_w = compute_atr_weights(prices, self.atr_window)
            long_weights = long_weights * atr_w
            short_weights = short_weights * atr_w

        # 5. Position sizing: fixed per-position (notebook v2) OR row-normalise
        #    per leg. The two paths are mutually exclusive; ``position_weight``
        #    wins when set.
        if self.position_weight is not None:
            long_weights = long_weights * self.position_weight
            short_weights = short_weights * self.position_weight
        elif self.normalize_weights:
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
                    clip_range=self.clip_vol_scaler,
                )
            if has_off:
                scalers["off"] = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)

            # Build multi-index weights using construct_weights
            # We pass the combined signal (long - short) through the scaler framework
            from .crypto_trend import construct_weights as _cw

            # When position_weight is set, ``combined`` already has the
            # universe baked in (long_sig × long_univ × position_weight), so
            # ask construct_weights to skip its post-tranche universe mask.
            # This makes tranching bleed coins out over the tranche window at
            # universe transitions, matching notebook v2 cell 115 semantics.
            multi_weights = _cw(
                combined,
                long_univ.clip(upper=1) + short_univ.clip(upper=1),
                scalers,
                self.tranches,
                normalize=False,
                signals_have_universe=self.position_weight is not None,
            )

            # 7b. Optional inv_vol track (notebook v2 cells 110-115).
            # Built from the pre-tranche ``off`` slice of the combined weights so
            # the inv_vol track row-sum matches the off row-sum (preserves the
            # ``signal × universe × position_weight`` exposure pattern).
            if self.inv_vol_track:
                if not has_off:
                    raise ValueError(
                        "inv_vol_track=True requires 'off' in vol_targets — the "
                        "inv_vol track is derived from the off-track weights."
                    )
                off_weights = combined  # off scaler is 1.0, so weights == combined
                iv_track = compute_inv_vol_track(
                    off_weights,
                    prices,
                    self.tranches,
                    vol_lookback=self.vol_lookback,
                )
                multi_weights = pd.concat([multi_weights, iv_track], axis=1).sort_index(
                    axis=1, level="ticker"
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
