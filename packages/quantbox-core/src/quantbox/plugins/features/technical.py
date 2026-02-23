"""Technical feature engineering plugin.

Computes per-symbol technical indicators from price (and optionally volume)
data, then stacks them into a single DataFrame with ``(date, symbol)``
MultiIndex. Extracted and generalized from the ``_FeatureEngineer`` helper
inside ``ml_strategy.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta

_DEFAULT_LOOKBACK_PERIODS = [5, 10, 20, 60]


@dataclass
class TechnicalFeatures:
    meta = PluginMeta(
        name="features.technical.v1",
        kind="feature",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Per-symbol technical indicators: returns, volatility, momentum, "
            "SMA ratios, RSI, Bollinger Bands, MACD, ATR proxy, volume stats, "
            "and day-of-week cyclical encoding."
        ),
        tags=("technical", "feature"),
        params_schema={
            "type": "object",
            "properties": {
                "lookback_periods": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "default": _DEFAULT_LOOKBACK_PERIODS,
                    "description": "Lookback windows (in days) for returns, volatility, momentum, SMA, and volume features.",
                },
            },
        },
    )

    def compute(self, data: dict[str, pd.DataFrame], params: dict[str, Any]) -> pd.DataFrame:
        prices = data["prices"]
        volume = data.get("volume")
        lookback_periods: list[int] = params.get("lookback_periods", _DEFAULT_LOOKBACK_PERIODS)

        frames: list[pd.DataFrame] = []
        for symbol in prices.columns:
            features = _compute_symbol_features(
                close=prices[symbol],
                volume=volume[symbol] if volume is not None else None,
                lookback_periods=lookback_periods,
            )
            features["symbol"] = symbol
            frames.append(features)

        stacked = pd.concat(frames, axis=0)
        stacked.index.name = "date"
        stacked = stacked.set_index("symbol", append=True)
        return stacked


def _compute_symbol_features(
    close: pd.Series,
    volume: pd.Series | None,
    lookback_periods: list[int],
) -> pd.DataFrame:
    """Build technical features for a single symbol's close-price series."""
    features = pd.DataFrame(index=close.index)
    daily_ret = close.pct_change()

    # Returns and volatility at each lookback period
    for p in lookback_periods:
        features[f"return_{p}d"] = close.pct_change(p)
        features[f"volatility_{p}d"] = daily_ret.rolling(p).std()

    # Momentum ratios
    for p in lookback_periods:
        features[f"momentum_{p}d"] = close / close.shift(p) - 1

    # SMA ratio and slope
    for p in lookback_periods:
        sma = close.rolling(p).mean()
        features[f"sma_ratio_{p}d"] = close / sma - 1
        features[f"sma_slope_{p}d"] = sma.pct_change(5)

    # RSI (14 and 28 period)
    for period in (14, 28):
        features[f"rsi_{period}"] = _rsi(close, period)

    # Bollinger Band position (20d)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    features["bb_position_20d"] = (close - sma20) / (2 * std20)

    # MACD (normalized by price)
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    features["macd"] = macd / close
    features["macd_signal"] = signal / close
    features["macd_histogram"] = (macd - signal) / close

    # ATR proxy (no high/low -- use abs(close.diff()) as true range)
    tr_proxy = close.diff().abs()
    for period in (14, 28):
        features[f"atr_{period}d"] = tr_proxy.rolling(period).mean() / close

    # Volume features (if available)
    if volume is not None:
        for p in lookback_periods:
            vol_sma = volume.rolling(p).mean()
            features[f"volume_ratio_{p}d"] = volume / vol_sma
            features[f"volume_trend_{p}d"] = vol_sma.pct_change(p)

    # Day-of-week cyclical encoding
    if hasattr(close.index, "dayofweek"):
        features["day_sin"] = np.sin(2 * np.pi * close.index.dayofweek / 5)
        features["day_cos"] = np.cos(2 * np.pi * close.index.dayofweek / 5)

    return features


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
