"""Technical indicators for quantitative analysis.

Provides common indicators as both static methods on ``TechnicalIndicators``
and as module-level convenience functions::

    from quantbox.indicators import sma, rsi, macd
    signal = rsi(prices['BTC'], period=14)

All functions accept a ``pd.Series`` and return a ``pd.Series``
(or tuple of Series for multi-output indicators like MACD and Bollinger Bands).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Calculate technical indicators for financial time series."""

    @staticmethod
    def sma(data: pd.Series, period: int = 20) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int = 20) -> pd.Series:
        """Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index (0-100)."""
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD: returns (macd_line, signal_line, histogram)."""
        fast_ema = data.ewm(span=fast_period, adjust=False).mean()
        slow_ema = data.ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        data: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands: returns (upper, middle, lower)."""
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    @staticmethod
    def atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Average True Range."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def returns(data: pd.Series, periods: int = 1) -> pd.Series:
        """Percentage returns."""
        return data.pct_change(periods=periods)

    @staticmethod
    def log_returns(data: pd.Series, periods: int = 1) -> pd.Series:
        """Logarithmic returns."""
        return np.log(data / data.shift(periods))

    @staticmethod
    def volatility(
        data: pd.Series,
        period: int = 20,
        annualize: bool = True,
        trading_days: int = 252,
    ) -> pd.Series:
        """Rolling volatility (annualized by default).

        Args:
            data: Price series.
            period: Lookback window.
            annualize: Scale by sqrt(trading_days).
            trading_days: Annualization factor (252 for equities, 365 for crypto).
        """
        log_ret = np.log(data / data.shift(1))
        vol = log_ret.rolling(window=period).std()
        if annualize:
            vol = vol * np.sqrt(trading_days)
        return vol

    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """Price momentum (absolute price change)."""
        return data - data.shift(period)

    @staticmethod
    def rate_of_change(data: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change (percentage)."""
        return ((data - data.shift(period)) / data.shift(period)) * 100


# Module-level convenience functions
sma = TechnicalIndicators.sma
ema = TechnicalIndicators.ema
rsi = TechnicalIndicators.rsi
macd = TechnicalIndicators.macd
bollinger_bands = TechnicalIndicators.bollinger_bands
atr = TechnicalIndicators.atr
returns = TechnicalIndicators.returns
log_returns = TechnicalIndicators.log_returns
volatility = TechnicalIndicators.volatility
momentum = TechnicalIndicators.momentum
rate_of_change = TechnicalIndicators.rate_of_change

__all__ = [
    "TechnicalIndicators",
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger_bands",
    "atr",
    "returns",
    "log_returns",
    "volatility",
    "momentum",
    "rate_of_change",
]
