"""Tests for :class:`TrendCatcherSimpleStrategy` — Robuxio PDF rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.strategies.trend_catcher_simple import TrendCatcherSimpleStrategy


def _make_data(n_days: int = 400, n_assets: int = 5, seed: int = 0) -> dict[str, pd.DataFrame]:
    """BTC plus N-1 other coins. BTC trends up; others noisy."""
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=n_days, freq="D").values)
    cols = ["BTC"] + [f"COIN{i}" for i in range(n_assets - 1)]
    # BTC: steady upward drift so it's always > MA50 from day 50
    btc = 100.0 * np.cumprod(1.0 + rng.normal(0.003, 0.02, n_days))
    # Others: random walks (no drift)
    others = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.025, (n_days, n_assets - 1)), axis=0)
    prices = pd.DataFrame(
        np.column_stack([btc, others]),
        index=dates,
        columns=cols,
    )
    volume = pd.DataFrame(rng.uniform(1e6, 1e8, prices.shape), index=dates, columns=cols)
    return {"prices": prices, "volume": volume}


class TestLongSide:
    def test_basic_signal_path(self) -> None:
        data = _make_data()
        strat = TrendCatcherSimpleStrategy(
            side="long",
            regime_filter_ma=50,
            signal_ma=20,
            universe_top_n_volume=10,
            max_positions=5,
            position_size=0.10,
        )
        out = strat.run(data)
        w = out["weights"]
        assert w.shape == data["prices"].shape
        # Only 0 or +0.10 values
        unique = np.unique(w.values[~np.isnan(w.values)])
        assert set(unique.tolist()) <= {0.0, 0.10}
        # Some positions opened
        assert (w > 0).any().any()
        # Never exceed max_positions per day
        assert (w > 0).sum(axis=1).max() <= 5

    def test_regime_filter_blocks_entries(self) -> None:
        """When BTC < MA50, long entries should be impossible."""
        data = _make_data()
        # Force BTC to crash (always < MA50 after day 50)
        data["prices"]["BTC"] = 100.0 * np.exp(-0.01 * np.arange(len(data["prices"])))
        strat = TrendCatcherSimpleStrategy(side="long", regime_filter_ma=50, signal_ma=20)
        out = strat.run(data)
        # Skip the warm-up (first 50 days)
        weights_after_warmup = out["weights"].iloc[60:]
        # No long positions should ever be opened
        assert (weights_after_warmup > 0).sum().sum() == 0

    def test_exit_on_ma_cross_below(self) -> None:
        """Position must exit when price crosses below MA20."""
        # Use short MAs so the regime filter warms up fast and the test is
        # deterministic: the coin's MA20-cross-up entry signal fires AFTER
        # BTC's MA5 regime is already valid and bullish.
        n = 200
        dates = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=n, freq="D").values)
        btc = 100.0 * np.exp(0.005 * np.arange(n))  # BTC trends up, MA5 valid from day 4
        # Target coin: flat for warmup, ramps up days 30-100, drops days 100-200
        coin = np.concatenate(
            [
                np.full(30, 100.0),
                100.0 * np.exp(0.02 * np.arange(70)),
                100.0 * np.exp(0.02 * 70) * np.exp(-0.04 * np.arange(n - 100)),
            ]
        )
        prices = pd.DataFrame({"BTC": btc, "COIN1": coin}, index=dates)
        volume = pd.DataFrame(1e7, index=dates, columns=prices.columns)
        strat = TrendCatcherSimpleStrategy(side="long", regime_filter_ma=5, signal_ma=20)
        out = strat.run({"prices": prices, "volume": volume})
        w = out["weights"]["COIN1"]
        # Position open during the rising phase (entry expected ~day 35-50)
        assert (w.iloc[50:95] > 0).any()
        # Position closed during the falling phase (after MA20 cross below)
        assert (w.iloc[140:] == 0).all()


class TestShortSide:
    def test_short_uses_negative_weights(self) -> None:
        data = _make_data()
        strat = TrendCatcherSimpleStrategy(side="short", regime_filter_ma=50, signal_ma=20)
        out = strat.run(data)
        w = out["weights"]
        # Either zero or negative
        assert (w.values[~np.isnan(w.values)] <= 0).all()


class TestUniverseAndCap:
    def test_universe_cap_excludes_low_volume(self) -> None:
        """With top_n_volume=2 and 5 assets, at most 2 columns ever have non-zero weight at entry."""
        # Strong upward trend on every coin so all are trying to enter
        n = 200
        dates = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=n, freq="D").values)
        prices = pd.DataFrame(
            {col: 100.0 * np.exp(0.005 * np.arange(n)) for col in ["BTC", "A", "B", "C", "D"]},
            index=dates,
        )
        # Volume: BTC=10, A=9, B=8, C=7, D=6 — universe top-2 should be BTC+A
        volume = pd.DataFrame(
            {"BTC": 10.0, "A": 9.0, "B": 8.0, "C": 7.0, "D": 6.0},
            index=dates,
        )
        strat = TrendCatcherSimpleStrategy(
            side="long",
            regime_filter_ma=50,
            signal_ma=20,
            universe_top_n_volume=2,
            max_positions=5,
        )
        out = strat.run({"prices": prices, "volume": volume})
        w = out["weights"].iloc[60:]
        # B/C/D never enter despite trending up — outside top-2 volume universe
        for col in ["B", "C", "D"]:
            assert (w[col] == 0).all(), f"{col} should never enter (outside volume top-2)"

    def test_position_count_capped(self) -> None:
        data = _make_data(n_assets=20, n_days=400)
        strat = TrendCatcherSimpleStrategy(
            side="long",
            universe_top_n_volume=20,
            max_positions=3,
            position_size=0.10,
        )
        out = strat.run(data)
        positions_per_day = (out["weights"] > 0).sum(axis=1)
        assert positions_per_day.max() <= 3


class TestErrors:
    def test_missing_volume_raises(self) -> None:
        prices = pd.DataFrame({"BTC": [100.0] * 100}, index=pd.date_range("2024-01-01", periods=100))
        strat = TrendCatcherSimpleStrategy()
        with pytest.raises(ValueError, match="requires data\\['volume'\\]"):
            strat.run({"prices": prices})

    def test_missing_regime_ticker_raises(self) -> None:
        prices = pd.DataFrame({"COIN": [100.0] * 100}, index=pd.date_range("2024-01-01", periods=100))
        volume = pd.DataFrame({"COIN": [1e7] * 100}, index=prices.index)
        strat = TrendCatcherSimpleStrategy()
        with pytest.raises(ValueError, match="regime ticker"):
            strat.run({"prices": prices, "volume": volume})

    def test_invalid_side_raises(self) -> None:
        data = _make_data()
        strat = TrendCatcherSimpleStrategy(side="both")
        with pytest.raises(ValueError, match="side must be 'long' or 'short'"):
            strat.run(data)
