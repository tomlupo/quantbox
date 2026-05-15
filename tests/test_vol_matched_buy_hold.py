"""Tests for :class:`VolMatchedBuyHoldStrategy`.

Verifies the notebook v2 cell-125 math:

    scale = target_annual_vol / (sigma_d * sqrt(trading_days))

so that a buy-and-hold position of weight ``scale`` on the underlying asset
delivers a return series with realised annualised vol close to the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.strategies.vol_matched_buy_hold import VolMatchedBuyHoldStrategy


def _btc_only_prices(n_days: int = 730, daily_vol: float = 0.04, seed: int = 0) -> pd.DataFrame:
    """Two-asset frame with BTC + a dummy column, BTC returns iid N(0, daily_vol)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    btc_ret = rng.normal(0.0, daily_vol, n_days)
    btc_px = 100.0 * np.cumprod(1.0 + btc_ret)
    other = pd.Series(100.0, index=dates)
    return pd.DataFrame({"BTC": btc_px, "OTHER": other.values}, index=dates)


class TestVolMatchedBuyHold:
    def test_scale_matches_target_full_sample(self) -> None:
        # Daily vol ~4% → ann vol ~76%. Target 25% → scale ≈ 0.33.
        prices = _btc_only_prices(n_days=1000, daily_vol=0.04, seed=42)
        strat = VolMatchedBuyHoldStrategy(ticker="BTC", target_annual_vol=0.25, trading_days=365)
        out = strat.run({"prices": prices})

        sigma_d = prices["BTC"].pct_change().std()
        expected_scale = 0.25 / (sigma_d * np.sqrt(365))

        w = out["weights"]["BTC"]
        assert w.iloc[-1] == pytest.approx(expected_scale, rel=1e-9)
        assert w.std() == pytest.approx(0.0, abs=1e-12), "full-sample scale must be constant"

    def test_realised_vol_close_to_target(self) -> None:
        # When you scale daily returns by `scale`, the realised annualised vol of
        # the scaled return stream should equal the target (up to sample noise).
        prices = _btc_only_prices(n_days=2000, daily_vol=0.03, seed=7)
        strat = VolMatchedBuyHoldStrategy(ticker="BTC", target_annual_vol=0.30, trading_days=365)
        out = strat.run({"prices": prices})

        scale = float(out["weights"]["BTC"].iloc[-1])
        scaled_rets = prices["BTC"].pct_change().dropna() * scale
        realised = scaled_rets.std() * np.sqrt(365)
        assert realised == pytest.approx(0.30, rel=1e-9)

    def test_non_target_columns_are_zero(self) -> None:
        prices = _btc_only_prices(n_days=200, daily_vol=0.02, seed=1)
        strat = VolMatchedBuyHoldStrategy(ticker="BTC", target_annual_vol=0.20)
        out = strat.run({"prices": prices})
        assert (out["weights"]["OTHER"] == 0.0).all()

    def test_rolling_window_is_causal(self) -> None:
        prices = _btc_only_prices(n_days=500, daily_vol=0.04, seed=3)
        strat = VolMatchedBuyHoldStrategy(ticker="BTC", target_annual_vol=0.25, vol_lookback=60, trading_days=365)
        out = strat.run({"prices": prices})
        w = out["weights"]["BTC"]
        # First 60 bars have no rolling-std → scale filled to 0
        assert (w.iloc[:60] == 0.0).all()
        # Beyond that, scale varies (rolling sigma changes day to day)
        assert w.iloc[60:].std() > 0

    def test_missing_ticker_raises(self) -> None:
        prices = _btc_only_prices(n_days=100)
        prices = prices.drop(columns=["BTC"])
        strat = VolMatchedBuyHoldStrategy(ticker="BTC", target_annual_vol=0.25)
        with pytest.raises(ValueError, match="not present in price universe"):
            strat.run({"prices": prices})

    def test_zero_vol_raises(self) -> None:
        dates = pd.date_range("2024-01-01", periods=200)
        flat = pd.DataFrame({"BTC": np.full(200, 100.0)}, index=dates)
        strat = VolMatchedBuyHoldStrategy(ticker="BTC", target_annual_vol=0.25)
        with pytest.raises(ValueError, match="degenerate daily-return std"):
            strat.run({"prices": flat})

    def test_pre_listing_bars_zeroed(self) -> None:
        dates = pd.date_range("2024-01-01", periods=200)
        rng = np.random.default_rng(11)
        btc = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, 200))
        btc_series = pd.Series(btc, index=dates)
        # BTC NaN for first 30 days (simulates pre-listing)
        btc_series.iloc[:30] = np.nan
        prices = pd.DataFrame({"BTC": btc_series})
        strat = VolMatchedBuyHoldStrategy(ticker="BTC", target_annual_vol=0.20)
        out = strat.run({"prices": prices})
        assert (out["weights"]["BTC"].iloc[:30] == 0.0).all()
        assert (out["weights"]["BTC"].iloc[30:] != 0.0).all()
