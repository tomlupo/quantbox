"""Tests for quantbox.indicators module."""
import numpy as np
import pandas as pd
import pytest

from quantbox.indicators import (
    TechnicalIndicators,
    sma,
    ema,
    rsi,
    macd,
    bollinger_bands,
    atr,
    returns,
    log_returns,
    volatility,
    momentum,
    rate_of_change,
)


@pytest.fixture
def prices():
    """Simple ascending price series."""
    return pd.Series(
        [10, 11, 12, 11, 13, 14, 13, 15, 16, 15, 17, 18, 17, 19, 20],
        dtype=float,
    )


@pytest.fixture
def ohlc():
    """OHLC data for ATR tests."""
    n = 30
    rng = np.random.RandomState(42)
    close = 100 + np.cumsum(rng.randn(n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    return pd.Series(high), pd.Series(low), pd.Series(close)


class TestSMA:
    def test_basic(self, prices):
        result = sma(prices, period=3)
        assert len(result) == len(prices)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx((10 + 11 + 12) / 3)
        assert result.iloc[3] == pytest.approx((11 + 12 + 11) / 3)

    def test_period_1(self, prices):
        result = sma(prices, period=1)
        pd.testing.assert_series_equal(result, prices, check_names=False)

    def test_class_method_matches_function(self, prices):
        pd.testing.assert_series_equal(
            TechnicalIndicators.sma(prices, 5), sma(prices, 5)
        )


class TestEMA:
    def test_basic(self, prices):
        result = ema(prices, period=3)
        assert len(result) == len(prices)
        assert not np.isnan(result.iloc[0])  # EMA starts from first value

    def test_converges_to_constant(self):
        constant = pd.Series([5.0] * 20)
        result = ema(constant, period=5)
        assert result.iloc[-1] == pytest.approx(5.0)


class TestRSI:
    def test_range(self, prices):
        result = rsi(prices, period=5)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_all_up(self):
        up = pd.Series(range(1, 21), dtype=float)
        result = rsi(up, period=5)
        assert result.iloc[-1] == pytest.approx(100.0)

    def test_all_down(self):
        down = pd.Series(range(20, 0, -1), dtype=float)
        result = rsi(down, period=5)
        assert result.iloc[-1] == pytest.approx(0.0)


class TestMACD:
    def test_output_shape(self, prices):
        line, signal, hist = macd(prices, fast_period=3, slow_period=5, signal_period=3)
        assert len(line) == len(prices)
        assert len(signal) == len(prices)
        assert len(hist) == len(prices)

    def test_histogram_is_diff(self, prices):
        line, signal, hist = macd(prices, fast_period=3, slow_period=5, signal_period=3)
        pd.testing.assert_series_equal(hist, line - signal, check_names=False)


class TestBollingerBands:
    def test_output_shape(self, prices):
        upper, middle, lower = bollinger_bands(prices, period=5, std_dev=2.0)
        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)

    def test_ordering(self, prices):
        upper, middle, lower = bollinger_bands(prices, period=5)
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_middle_is_sma(self, prices):
        _, middle, _ = bollinger_bands(prices, period=5)
        expected = sma(prices, period=5)
        pd.testing.assert_series_equal(middle, expected, check_names=False)


class TestATR:
    def test_positive(self, ohlc):
        high, low, close = ohlc
        result = atr(high, low, close, period=5)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_length(self, ohlc):
        high, low, close = ohlc
        result = atr(high, low, close, period=5)
        assert len(result) == len(high)


class TestReturns:
    def test_basic(self):
        p = pd.Series([100, 110, 105, 115.5])
        r = returns(p)
        assert r.iloc[1] == pytest.approx(0.10)
        assert r.iloc[2] == pytest.approx(-5 / 110)

    def test_log_returns_basic(self):
        p = pd.Series([100, 110, 105, 115.5])
        lr = log_returns(p)
        assert lr.iloc[1] == pytest.approx(np.log(110 / 100))


class TestVolatility:
    def test_positive(self, prices):
        result = volatility(prices, period=5, trading_days=252)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_annualization(self, prices):
        vol_ann = volatility(prices, period=5, annualize=True, trading_days=252)
        vol_raw = volatility(prices, period=5, annualize=False)
        valid = ~(vol_ann.isna() | vol_raw.isna())
        ratio = vol_ann[valid] / vol_raw[valid]
        np.testing.assert_allclose(ratio.values, np.sqrt(252), rtol=1e-10)

    def test_crypto_trading_days(self, prices):
        vol_365 = volatility(prices, period=5, trading_days=365)
        vol_252 = volatility(prices, period=5, trading_days=252)
        valid = ~(vol_365.isna() | vol_252.isna())
        assert (vol_365[valid] > vol_252[valid]).all()


class TestMomentum:
    def test_basic(self):
        p = pd.Series([10, 12, 15, 13, 18], dtype=float)
        m = momentum(p, period=2)
        assert m.iloc[2] == pytest.approx(15 - 10)
        assert m.iloc[4] == pytest.approx(18 - 15)


class TestRateOfChange:
    def test_basic(self):
        p = pd.Series([100, 110, 120, 130], dtype=float)
        roc = rate_of_change(p, period=1)
        assert roc.iloc[1] == pytest.approx(10.0)
        assert roc.iloc[2] == pytest.approx(100 * (120 - 110) / 110)
