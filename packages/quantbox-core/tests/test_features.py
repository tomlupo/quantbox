"""Tests for shared feature functions."""
import numpy as np
import pandas as pd
import pytest
from quantbox.features import (
    compute_returns,
    compute_rolling_vol,
    compute_ewm_vol,
    compute_sma,
    compute_ema,
    compute_donchian,
    compute_zscore_cross_sectional,
    compute_rank_cross_sectional,
    compute_features_bundle,
)


@pytest.fixture
def prices():
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    rng = np.random.RandomState(42)
    btc = 50000 * np.cumprod(1 + rng.normal(0, 0.02, 60))
    eth = 3000 * np.cumprod(1 + rng.normal(0, 0.03, 60))
    return pd.DataFrame({"BTC": btc, "ETH": eth}, index=dates)


class TestReturns:
    def test_pct_change_matches(self, prices):
        result = compute_returns(prices, [1])
        expected = prices.pct_change(periods=1)
        pd.testing.assert_frame_equal(result["ret_1d"], expected)

    def test_multi_window(self, prices):
        result = compute_returns(prices, [1, 5, 21])
        assert set(result.keys()) == {"ret_1d", "ret_5d", "ret_21d"}
        for key, df in result.items():
            assert df.shape == prices.shape

    def test_log_returns(self, prices):
        result = compute_returns(prices, [1], method="log")
        expected = np.log(prices / prices.shift(1))
        pd.testing.assert_frame_equal(result["ret_1d"], expected)


class TestVolatility:
    def test_rolling_vol_shape(self, prices):
        result = compute_rolling_vol(prices, [21])
        assert result["vol_21d"].shape == prices.shape

    def test_rolling_vol_annualized(self, prices):
        ann = compute_rolling_vol(prices, [21], annualize=True, factor=365)
        raw = compute_rolling_vol(prices, [21], annualize=False)
        ratio = ann["vol_21d"].dropna() / raw["vol_21d"].dropna()
        np.testing.assert_allclose(ratio.values, np.sqrt(365), rtol=1e-10)

    def test_ewm_vol_shape(self, prices):
        result = compute_ewm_vol(prices, [21])
        assert result["vol_ewm_21"].shape == prices.shape

    def test_ewm_vol_keys(self, prices):
        result = compute_ewm_vol(prices, [10, 30])
        assert set(result.keys()) == {"vol_ewm_10", "vol_ewm_30"}


class TestMovingAverages:
    def test_sma_matches_rolling_mean(self, prices):
        result = compute_sma(prices, [20])
        expected = prices.rolling(window=20).mean()
        pd.testing.assert_frame_equal(result["sma_20d"], expected)

    def test_ema_matches_ewm_mean(self, prices):
        result = compute_ema(prices, [20])
        expected = prices.ewm(span=20).mean()
        pd.testing.assert_frame_equal(result["ema_20"], expected)

    def test_sma_multi_window(self, prices):
        result = compute_sma(prices, [10, 20, 50])
        assert set(result.keys()) == {"sma_10d", "sma_20d", "sma_50d"}


class TestDonchian:
    def test_donchian_channels(self, prices):
        result = compute_donchian(prices, [20])
        assert "donchian_20d_high" in result
        assert "donchian_20d_low" in result
        assert "donchian_20d_mid" in result

    def test_donchian_high_gte_low(self, prices):
        result = compute_donchian(prices, [20])
        high = result["donchian_20d_high"].dropna()
        low = result["donchian_20d_low"].dropna()
        assert (high >= low).all().all()

    def test_donchian_mid_is_average(self, prices):
        result = compute_donchian(prices, [20])
        high = result["donchian_20d_high"]
        low = result["donchian_20d_low"]
        mid = result["donchian_20d_mid"]
        expected = (high + low) / 2
        pd.testing.assert_frame_equal(mid, expected)


class TestCrossSectional:
    def test_zscore_zero_mean(self, prices):
        result = compute_zscore_cross_sectional(prices)
        row_means = result.mean(axis=1).dropna()
        np.testing.assert_allclose(row_means.values, 0, atol=1e-10)

    def test_zscore_clipping(self):
        df = pd.DataFrame({"A": [100.0], "B": [0.0], "C": [0.0], "D": [0.0]})
        result = compute_zscore_cross_sectional(df, clip=2.0)
        assert result.max().max() <= 2.0
        assert result.min().min() >= -2.0

    def test_rank_pct(self, prices):
        result = compute_rank_cross_sectional(prices, pct=True)
        # percentile ranks should be in [0, 1]
        assert result.min().min() > 0
        assert result.max().max() <= 1.0

    def test_rank_ascending(self):
        df = pd.DataFrame({"A": [1.0, 3.0], "B": [2.0, 1.0]})
        result = compute_rank_cross_sectional(df, ascending=True, pct=False)
        # row 0: A=1 (rank 1), B=2 (rank 2)
        assert result.iloc[0]["A"] == 1.0
        assert result.iloc[0]["B"] == 2.0


class TestBundle:
    def test_bundle_dispatch(self, prices):
        manifest = {
            "returns": {"windows": [1, 5]},
            "sma": {"windows": [20]},
        }
        result = compute_features_bundle(prices, manifest)
        assert "ret_1d" in result
        assert "ret_5d" in result
        assert "sma_20d" in result

    def test_bundle_unknown_type(self, prices):
        with pytest.raises(ValueError, match="Unknown feature type"):
            compute_features_bundle(prices, {"bogus": {}})

    def test_bundle_all_types(self, prices):
        manifest = {
            "returns": {"windows": [1]},
            "rolling_vol": {"windows": [21]},
            "ewm_vol": {"spans": [21]},
            "sma": {"windows": [20]},
            "ema": {"spans": [20]},
            "donchian": {"windows": [20]},
        }
        result = compute_features_bundle(prices, manifest)
        expected_keys = {
            "ret_1d", "vol_21d", "vol_ewm_21",
            "sma_20d", "ema_20", "donchian_20d_high",
            "donchian_20d_low", "donchian_20d_mid",
        }
        assert expected_keys.issubset(set(result.keys()))
