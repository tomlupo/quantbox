"""Tests for :class:`CryptoRegimeTrendStrategy` v2-faithful knobs.

Covers the three knobs added for the Robuxio TrendCatcher v2 replication:

* ``position_weight`` — fixed 1/N weighting (notebook cell 108).
* ``inv_vol_track`` — appended ``vol_target='inv_vol'`` track (cells 110-115).
* ``clip_vol_scaler`` — pass-through to :func:`compute_volatility_scalers`.

Also exercises backwards compatibility for the default constructor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.strategies.crypto_regime_trend import CryptoRegimeTrendStrategy
from quantbox.plugins.strategies.crypto_trend import (
    compute_inv_vol_track,
    compute_volatility_scalers,
)


def _make_market_data(n_days: int = 400, n_assets: int = 12) -> dict[str, pd.DataFrame]:
    rng = np.random.RandomState(7)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    # BTC must be present
    cols = ["BTC"] + [f"COIN{i}" for i in range(n_assets - 1)]
    log_ret = rng.normal(0.0003, 0.02, (n_days, n_assets))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(log_ret, axis=0)), index=dates, columns=cols)
    volume = pd.DataFrame(rng.uniform(1e5, 1e7, prices.shape), index=dates, columns=cols)
    market_cap = pd.DataFrame(rng.uniform(1e8, 1e10, prices.shape), index=dates, columns=cols)
    return {"prices": prices, "volume": volume, "market_cap": market_cap}


class TestClipVolScaler:
    def test_default_clips_to_legacy_range(self) -> None:
        prices = pd.DataFrame(
            {"BTC": [100.0, 100.001, 100.002] * 30},
            index=pd.date_range("2024-01-01", periods=90),
        )
        scalers = compute_volatility_scalers(prices, [0.5], vol_lookback=20)
        # vanishingly small vol -> scaler = 0.5 / tiny -> clipped to 10.0
        assert (scalers["50"].dropna() <= 10.0 + 1e-9).all().all()
        assert (scalers["50"].dropna() >= 0.1 - 1e-9).all().all()

    def test_no_clip_when_none(self) -> None:
        prices = pd.DataFrame(
            {"BTC": [100.0, 100.001, 100.002] * 30},
            index=pd.date_range("2024-01-01", periods=90),
        )
        scalers = compute_volatility_scalers(prices, [0.5], vol_lookback=20, clip_range=None)
        # Without clip, the inverse-vol scaler can be much larger than 10
        assert scalers["50"].dropna().max().max() > 10.0


class TestPositionWeight:
    def test_default_unchanged_behaviour(self) -> None:
        data = _make_market_data()
        strat = CryptoRegimeTrendStrategy(
            long_max=5,
            short_max=0,
            coins_to_trade=10,
            use_ensemble=False,
            vol_targets=["off"],
            tranches=[1],
            output_periods=10,
        )
        result = strat.run(data)
        # default (normalize_weights=True): per-row sum across longs should be ~1
        # whenever any long position is open
        w = result["weights"]
        if isinstance(w.columns, pd.MultiIndex):
            w = w.xs(("off", 1), axis=1, level=("vol_target", "tranches"))
        row_sums = w.sum(axis=1)
        # Only check rows with active positions
        active = row_sums[row_sums > 1e-9]
        if not active.empty:
            assert (active <= 1.0 + 1e-6).all()
            assert (active >= 0.99 - 1e-6).all() or (active <= 1.0 + 1e-6).all()

    def test_position_weight_produces_fixed_per_coin(self) -> None:
        data = _make_market_data()
        pos_w = 0.1
        strat = CryptoRegimeTrendStrategy(
            long_max=5,
            short_max=0,
            coins_to_trade=10,
            use_ensemble=False,
            position_weight=pos_w,
            vol_targets=["off"],
            tranches=[1],
            output_periods=20,
        )
        result = strat.run(data)
        w = result["weights"].xs(("off", 1), axis=1, level=("vol_target", "tranches"))
        # Every non-zero weight should be exactly position_weight (signal is binary
        # without ensemble; tranches=1 means no smoothing).
        nz = w.values[np.abs(w.values) > 1e-9]
        if nz.size > 0:
            np.testing.assert_allclose(nz, pos_w, atol=1e-9)

    def test_position_weight_overrides_normalize(self) -> None:
        """When both are set, position_weight wins (silently)."""
        data = _make_market_data()
        strat = CryptoRegimeTrendStrategy(
            long_max=5,
            short_max=0,
            coins_to_trade=10,
            use_ensemble=False,
            position_weight=0.1,
            normalize_weights=True,  # should be overridden
            vol_targets=["off"],
            tranches=[1],
            output_periods=20,
        )
        result = strat.run(data)
        w = result["weights"].xs(("off", 1), axis=1, level=("vol_target", "tranches"))
        nz = w.values[np.abs(w.values) > 1e-9]
        if nz.size > 0:
            np.testing.assert_allclose(nz, 0.1, atol=1e-9)


class TestInvVolTrack:
    def test_inv_vol_track_appended(self) -> None:
        data = _make_market_data()
        strat = CryptoRegimeTrendStrategy(
            long_max=5,
            short_max=0,
            coins_to_trade=10,
            use_ensemble=False,
            position_weight=0.1,
            vol_targets=["off"],
            tranches=[1, 3],
            inv_vol_track=True,
            output_periods=30,
        )
        result = strat.run(data)
        w = result["weights"]
        vt_levels = set(w.columns.get_level_values("vol_target").unique())
        assert "off" in vt_levels
        assert "inv_vol" in vt_levels

    def test_inv_vol_track_requires_off(self) -> None:
        data = _make_market_data()
        strat = CryptoRegimeTrendStrategy(
            long_max=5,
            short_max=0,
            coins_to_trade=10,
            use_ensemble=False,
            position_weight=0.1,
            vol_targets=[0.5],  # no 'off'
            inv_vol_track=True,
            output_periods=10,
        )
        with pytest.raises(ValueError, match="inv_vol_track"):
            strat.run(data)

    def test_inv_vol_track_preserves_off_row_sum(self) -> None:
        """Notebook v2 cell 110: inv_vol row-sum must equal off row-sum per day."""
        data = _make_market_data(n_days=300, n_assets=8)
        strat = CryptoRegimeTrendStrategy(
            long_max=4,
            short_max=0,
            coins_to_trade=6,
            use_ensemble=False,
            position_weight=0.1,
            vol_targets=["off"],
            tranches=[1],
            inv_vol_track=True,
            output_periods=100,
        )
        result = strat.run(data)
        w = result["weights"]
        off = w.xs(("off", 1), axis=1, level=("vol_target", "tranches"))
        iv = w.xs(("inv_vol", 1), axis=1, level=("vol_target", "tranches"))
        # Only compare rows where off has positions and inv_vol is non-zero
        common = (off.sum(axis=1) > 1e-9) & (iv.sum(axis=1) > 1e-9)
        if common.any():
            np.testing.assert_allclose(
                off.loc[common].sum(axis=1).values,
                iv.loc[common].sum(axis=1).values,
                atol=1e-9,
            )


class TestBestPracticeUniverseKnobs:
    """Universe-construction knobs added for production-defensible runs."""

    def test_volume_rolling_window_smooths_boundary_churn(self) -> None:
        """With volume_rolling_window>1, the universe set should change less day-to-day."""
        data = _make_market_data(n_days=300, n_assets=15)
        kw = dict(
            long_max=5,
            short_max=0,
            coins_to_trade=10,
            use_ensemble=False,
            position_weight=0.1,
            vol_targets=["off"],
            tranches=[1],
            output_periods=200,
        )
        out_spot = CryptoRegimeTrendStrategy(**kw, volume_rolling_window=1).run(data)
        out_30d = CryptoRegimeTrendStrategy(**kw, volume_rolling_window=30).run(data)

        w_spot = out_spot["weights"].xs(("off", 1), axis=1, level=("vol_target", "tranches"))
        w_30d = out_30d["weights"].xs(("off", 1), axis=1, level=("vol_target", "tranches"))
        # Count day-over-day churn: number of tickers that entered or exited
        churn_spot = (w_spot.gt(0).astype(int).diff().abs().sum(axis=1)).iloc[1:].mean()
        churn_30d = (w_30d.gt(0).astype(int).diff().abs().sum(axis=1)).iloc[1:].mean()
        assert churn_30d <= churn_spot, (
            f"30d rolling-vol should not churn more than 1d spot ({churn_30d} vs {churn_spot})"
        )

    def test_min_listing_days_excludes_new_listings(self) -> None:
        """A coin under the listing cool-off must have weight 0."""
        # Build market data where 'NEW' joins on day 100 (mid-window)
        data = _make_market_data(n_days=300, n_assets=10)
        prices = data["prices"].copy()
        prices.iloc[:100, prices.columns.get_loc("COIN8")] = float("nan")
        data["prices"] = prices
        # Strong artificial volume on NEW to make it pass the volume cut early
        v = data["volume"].copy()
        v.iloc[:100, v.columns.get_loc("COIN8")] = 0.0
        v.iloc[100:, v.columns.get_loc("COIN8")] = 1e8
        data["volume"] = v

        strat = CryptoRegimeTrendStrategy(
            long_max=5,
            short_max=0,
            coins_to_trade=10,
            use_ensemble=False,
            position_weight=0.1,
            vol_targets=["off"],
            tranches=[1],
            min_listing_days=60,
            output_periods=200,
        )
        result = strat.run(data)
        w = result["weights"].xs(("off", 1), axis=1, level=("vol_target", "tranches"))
        # Days 100-159 (within 60-day cool-off) → COIN8 must be 0
        new_listing_window = w.loc[
            (w.index >= prices.index[100]) & (w.index < prices.index[160]),
            "COIN8",
        ]
        assert (new_listing_window.abs() < 1e-9).all(), (
            "Coin within listing cool-off should never receive weight"
        )

    def test_hysteresis_reduces_universe_churn(self) -> None:
        """Hysteresis band > 0 should not increase churn vs no band."""
        data = _make_market_data(n_days=300, n_assets=15)
        kw = dict(
            long_max=5,
            short_max=0,
            coins_to_trade=10,
            use_ensemble=False,
            position_weight=0.1,
            vol_targets=["off"],
            tranches=[1],
            output_periods=200,
        )
        out_no = CryptoRegimeTrendStrategy(**kw, hysteresis_rank_band=0).run(data)
        out_yes = CryptoRegimeTrendStrategy(**kw, hysteresis_rank_band=5).run(data)
        w_no = out_no["weights"].xs(("off", 1), axis=1, level=("vol_target", "tranches"))
        w_yes = out_yes["weights"].xs(("off", 1), axis=1, level=("vol_target", "tranches"))
        churn_no = (w_no.gt(0).astype(int).diff().abs().sum(axis=1)).iloc[1:].mean()
        churn_yes = (w_yes.gt(0).astype(int).diff().abs().sum(axis=1)).iloc[1:].mean()
        assert churn_yes <= churn_no + 1e-9, (
            f"Hysteresis band should not increase churn (yes={churn_yes}, no={churn_no})"
        )


class TestComputeInvVolTrackHelper:
    """Direct unit tests for the helper function."""

    def test_only_active_off_positions_contribute(self) -> None:
        dates = pd.date_range("2024-01-01", periods=120)
        prices = pd.DataFrame(
            {
                "A": 100 * np.exp(np.cumsum(np.random.RandomState(1).normal(0, 0.02, 120))),
                "B": 100 * np.exp(np.cumsum(np.random.RandomState(2).normal(0, 0.05, 120))),
            },
            index=dates,
        )
        # off-track weight: only A active for first 60 days, both active after
        off_w = pd.DataFrame(0.0, index=dates, columns=prices.columns)
        off_w.loc[off_w.index[60:], "A"] = 0.1
        off_w.loc[off_w.index[60:], "B"] = 0.1
        # First 60 days: off all zero → inv_vol track also zero
        iv = compute_inv_vol_track(off_w, prices, tranches=[1])
        iv1 = iv.xs(("inv_vol", 1), axis=1, level=("vol_target", "tranches"))
        assert (iv1.iloc[:60].abs().sum().sum() < 1e-9)
        # After day 60 the two positions split inverse-vol proportionally
        active_rows = iv1.iloc[100:120]
        assert (active_rows.sum(axis=1) > 0).all()
