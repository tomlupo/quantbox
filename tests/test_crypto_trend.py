"""Tests for the CryptoTrendStrategy plugin and its helper functions.

Covers:
- Donchian breakout signal generation (simple + trailing stop)
- Volatility scaling
- Universe selection (with and without market cap)
- Param alias mapping
- Full strategy run (happy path)
- Edge cases: all-NaN prices, single asset, empty universe
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbox.plugins.strategies._universe import (
    select_universe,
)
from quantbox.plugins.strategies.crypto_trend import (
    CryptoTrendStrategy,
    compute_donchian_breakout_vectorized,
    compute_donchian_simple_vectorized,
    compute_volatility_scalers,
    construct_weights,
    generate_ensemble_signals,
    get_simple_weights,
    run,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.RandomState(42)


def _make_prices(
    n_days: int = 400,
    n_assets: int = 15,
    seed: int = 42,
    prefix: str = "COIN",
) -> pd.DataFrame:
    """Synthetic random-walk prices (wide format, date index x symbol cols)."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    symbols = [f"{prefix}{i}" for i in range(n_assets)]
    log_ret = rng.normal(0.0003, 0.02, (n_days, n_assets))
    prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(prices, index=dates, columns=symbols)


def _make_market_data(
    n_days: int = 400,
    n_assets: int = 15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Return synthetic prices, volume, and market_cap."""
    rng = np.random.RandomState(seed)
    prices = _make_prices(n_days, n_assets, seed)
    dates = prices.index
    symbols = list(prices.columns)

    volume = pd.DataFrame(
        rng.uniform(1e4, 1e6, (n_days, n_assets)),
        index=dates,
        columns=symbols,
    )
    market_cap = pd.DataFrame(
        rng.uniform(1e8, 1e10, (n_days, n_assets)),
        index=dates,
        columns=symbols,
    )
    return {"prices": prices, "volume": volume, "market_cap": market_cap}


# ---------------------------------------------------------------------------
# Donchian Breakout Tests
# ---------------------------------------------------------------------------


class TestDonchianBreakout:
    """Tests for compute_donchian_breakout_vectorized and simple variant."""

    def test_simple_breakout_basic_signal(self) -> None:
        """Simple breakout should be 1 when price equals rolling high."""
        # Monotonically increasing prices -> always at rolling high after warmup
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        prices = pd.Series(np.arange(1.0, 31.0), index=dates, name="BTC")

        signal = compute_donchian_simple_vectorized(prices, window=5)

        assert signal.dtype == float
        assert len(signal) == 30
        # First 4 values should be NaN-ish (0 since < min_periods)
        assert (signal.iloc[:4] == 0.0).all()
        # After warmup, monotonically rising prices always break out
        assert (signal.iloc[4:] == 1.0).all()

    def test_simple_breakout_flat_prices(self) -> None:
        """Flat prices should always equal rolling high -> signal = 1 after warmup."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        prices = pd.Series(100.0, index=dates, name="ETH")

        signal = compute_donchian_simple_vectorized(prices, window=5)

        # Flat = price == high => signal = 1 after warmup
        assert (signal.iloc[4:] == 1.0).all()

    def test_breakout_with_trailing_stop_produces_binary(self) -> None:
        """Full Donchian with trailing stop should produce 0/1 signals."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        rng = np.random.RandomState(123)
        prices = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0, 0.02, 100))),
            index=dates,
            name="SOL",
        )

        signal = compute_donchian_breakout_vectorized(prices, window=10)

        assert set(signal.unique()).issubset({0.0, 1.0})
        assert len(signal) == 100

    def test_breakout_nan_at_start(self) -> None:
        """Signal should be 0 for the first (window-1) entries."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        prices = pd.Series(np.linspace(50, 150, 50), index=dates, name="AVAX")
        window = 15

        signal = compute_donchian_breakout_vectorized(prices, window=window)

        # Indices 0..13 should all be zero (before window kicks in)
        assert (signal.iloc[: window - 1] == 0.0).all()

    def test_trailing_stop_triggers_exit(self) -> None:
        """After a breakout, a drop below the trailing stop should exit."""
        # Construct a deliberate up-then-crash pattern
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        # Rise to trigger breakout, then crash to trigger trailing stop exit
        up = np.linspace(100, 200, 15)
        down = np.linspace(200, 80, 15)
        prices = pd.Series(np.concatenate([up, down]), index=dates, name="CRASH")

        signal = compute_donchian_breakout_vectorized(prices, window=5)

        # The rising phase should have the signal on after warmup
        assert signal.iloc[10] == 1.0
        # The crash should eventually turn signal off (price drops below mid)
        assert signal.iloc[-1] == 0.0

    def test_ensemble_signals_shape_and_range(self) -> None:
        """Ensemble signals should be between 0 and 1 (mean of binary signals)."""
        prices = _make_prices(n_days=200, n_assets=5, seed=99)
        windows = [5, 10, 20]

        signals = generate_ensemble_signals(prices, windows, use_trailing_stop=False)

        assert signals.shape == prices.shape
        assert list(signals.columns) == list(prices.columns)
        assert (signals >= 0.0).all().all()
        assert (signals <= 1.0).all().all()


# ---------------------------------------------------------------------------
# Volatility Scaling Tests
# ---------------------------------------------------------------------------


class TestVolScale:
    """Tests for compute_volatility_scalers."""

    def test_scalers_keys(self) -> None:
        """Output dict should have keys matching vol targets."""
        prices = _make_prices(n_days=200, n_assets=3, seed=7)
        scalers = compute_volatility_scalers(prices, [0.25, 0.50], vol_lookback=30)

        assert "25" in scalers
        assert "50" in scalers
        assert len(scalers) == 2

    def test_scaler_shape_matches_prices(self) -> None:
        """Each scaler DataFrame should match the shape of the input prices."""
        prices = _make_prices(n_days=150, n_assets=4, seed=8)
        scalers = compute_volatility_scalers(prices, [0.25], vol_lookback=30)

        assert scalers["25"].shape == prices.shape

    def test_scaler_clipping(self) -> None:
        """Scalers should be clipped to [0.1, 10.0] range."""
        prices = _make_prices(n_days=200, n_assets=3, seed=9)
        scalers = compute_volatility_scalers(prices, [0.25], vol_lookback=30)

        s = scalers["25"].dropna()
        assert (s >= 0.1).all().all()
        assert (s <= 10.0).all().all()

    def test_zero_vol_does_not_produce_inf(self) -> None:
        """When realized vol is zero (constant prices), scaler should be NaN then clipped."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        # Constant prices -> zero returns -> zero vol
        prices = pd.DataFrame(
            {"A": 100.0, "B": 200.0},
            index=dates,
        )
        scalers = compute_volatility_scalers(prices, [0.50], vol_lookback=30)

        s = scalers["50"]
        # Should not contain inf
        assert not np.isinf(s.values).any()


# ---------------------------------------------------------------------------
# Universe Selection Tests
# ---------------------------------------------------------------------------


class TestUniverseSelection:
    """Tests for select_universe (vectorized implementation)."""

    def test_excludes_stablecoins(self) -> None:
        """Stablecoins in the exclude list should never be selected."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        symbols = ["BTC", "ETH", "USDT", "USDC", "SOL"]
        rng = np.random.RandomState(10)

        prices = pd.DataFrame(rng.uniform(1, 1000, (50, 5)), index=dates, columns=symbols)
        volume = pd.DataFrame(rng.uniform(1e5, 1e7, (50, 5)), index=dates, columns=symbols)
        market_cap = pd.DataFrame(rng.uniform(1e9, 1e11, (50, 5)), index=dates, columns=symbols)

        universe = select_universe(prices, volume, market_cap, top_by_mcap=5, top_by_volume=5)

        assert (universe["USDT"] == 0.0).all()
        assert (universe["USDC"] == 0.0).all()

    def test_universe_shape_matches_prices(self) -> None:
        """Universe mask should have the exact same shape as prices."""
        prices = _make_prices(n_days=60, n_assets=10, seed=11)
        volume = pd.DataFrame(
            RNG.uniform(1e4, 1e6, prices.shape),
            index=prices.index,
            columns=prices.columns,
        )
        market_cap = pd.DataFrame(
            RNG.uniform(1e8, 1e10, prices.shape),
            index=prices.index,
            columns=prices.columns,
        )

        universe = select_universe(prices, volume, market_cap, top_by_mcap=5, top_by_volume=3)

        assert universe.shape == prices.shape
        # Each row should have at most top_by_volume=3 selected
        assert (universe.sum(axis=1) <= 3).all()

    def test_universe_without_market_cap(self) -> None:
        """When market_cap is None, selection should still work (volume only)."""
        prices = _make_prices(n_days=50, n_assets=8, seed=12)
        volume = pd.DataFrame(
            np.random.RandomState(12).uniform(1e4, 1e6, prices.shape),
            index=prices.index,
            columns=prices.columns,
        )

        universe = select_universe(prices, volume, market_cap=None, top_by_mcap=30, top_by_volume=4)

        assert universe.shape == prices.shape
        assert (universe.sum(axis=1) <= 4).all()
        # Should still have some selected assets
        assert universe.sum().sum() > 0

    def test_empty_after_exclusion_returns_zeros(self) -> None:
        """If all tickers are excluded, universe should be all zeros."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        symbols = ["USDT", "USDC", "BUSD"]
        prices = pd.DataFrame(1.0, index=dates, columns=symbols)
        volume = pd.DataFrame(1.0, index=dates, columns=symbols)
        market_cap = pd.DataFrame(1e9, index=dates, columns=symbols)

        universe = select_universe(prices, volume, market_cap)

        assert (universe == 0.0).all().all()


# ---------------------------------------------------------------------------
# Param Aliases Tests
# ---------------------------------------------------------------------------


class TestParamAliases:
    """Tests for _PARAM_ALIASES in CryptoTrendStrategy.run()."""

    def test_old_param_names_are_applied(self) -> None:
        """Legacy quantlab param names should map to the correct attributes."""
        strategy = CryptoTrendStrategy(
            use_duckdb=False,
            use_trailing_stop=False,
        )
        data = _make_market_data(n_days=400, n_assets=15, seed=42)

        # Use old param names
        params = {
            "tickers_to_exclude": ["COIN0"],
            "filtered_coins_market_cap": 12,
            "portfolio_coins_max": 5,
            "last_x_days": 7,
            "normalize": False,
        }

        strategy.run(data, params=params)

        # Verify the strategy internals were updated
        assert strategy.exclude_tickers == ["COIN0"]
        assert strategy.top_by_mcap == 12
        assert strategy.top_by_volume == 5
        assert strategy.output_periods == 7
        assert strategy.normalize_weights is False

    def test_new_param_names_work_directly(self) -> None:
        """New quantbox param names should work without alias mapping."""
        strategy = CryptoTrendStrategy(
            use_duckdb=False,
            use_trailing_stop=False,
        )
        data = _make_market_data(n_days=400, n_assets=15, seed=42)

        params = {
            "exclude_tickers": ["COIN1", "COIN2"],
            "top_by_mcap": 8,
            "top_by_volume": 4,
            "output_periods": 15,
        }

        strategy.run(data, params=params)

        assert strategy.exclude_tickers == ["COIN1", "COIN2"]
        assert strategy.top_by_mcap == 8
        assert strategy.top_by_volume == 4
        assert strategy.output_periods == 15

    def test_periods_alias_maps_to_output_periods(self) -> None:
        """The 'periods' alias should also map to output_periods."""
        strategy = CryptoTrendStrategy(
            use_duckdb=False,
            use_trailing_stop=False,
        )
        data = _make_market_data(n_days=400, n_assets=15, seed=42)

        strategy.run(data, params={"periods": 20})

        assert strategy.output_periods == 20


# ---------------------------------------------------------------------------
# Full Strategy Run Tests
# ---------------------------------------------------------------------------


class TestCryptoTrendStrategy:
    """Integration tests for CryptoTrendStrategy.run()."""

    def test_run_happy_path(self) -> None:
        """Strategy run should return weights, simple_weights, and details."""
        strategy = CryptoTrendStrategy(
            lookback_windows=[5, 10, 20],
            vol_targets=[0.25, 0.50],
            tranches=[1, 5],
            top_by_mcap=10,
            top_by_volume=5,
            output_periods=10,
            use_duckdb=False,
            use_trailing_stop=False,
            exclude_tickers=[],
        )
        data = _make_market_data(n_days=400, n_assets=15, seed=42)

        result = strategy.run(data)

        assert "weights" in result
        assert "simple_weights" in result
        assert "details" in result

        # Weights should have output_periods rows
        assert len(result["weights"]) == 10

        # Weights should be a DataFrame with MultiIndex columns
        assert isinstance(result["weights"].columns, pd.MultiIndex)
        assert result["weights"].columns.names == ["vol_target", "tranches", "ticker"]

        # simple_weights should be a dict with ticker -> float
        assert isinstance(result["simple_weights"], dict)

        # Details should contain signals, universe, scalers
        assert "signals" in result["details"]
        assert "universe" in result["details"]
        assert "scalers" in result["details"]

    def test_run_with_vol_off(self) -> None:
        """vol_targets=['off', 0.50] should produce unscaled weights for 'off' scaler."""
        # Note: run() internally calls get_simple_weights("50", 5) so we must
        # include 0.50 in vol_targets and 5 in tranches.
        strategy = CryptoTrendStrategy(
            lookback_windows=[5, 10],
            vol_targets=["off", 0.50],
            tranches=[1, 5],
            top_by_mcap=10,
            top_by_volume=5,
            output_periods=5,
            use_duckdb=False,
            use_trailing_stop=False,
            exclude_tickers=[],
        )
        data = _make_market_data(n_days=200, n_assets=10, seed=55)

        result = strategy.run(data)

        assert "weights" in result
        scalers = result["details"]["scalers"]
        assert "off" in scalers
        # The "off" scaler should be all 1.0
        assert (scalers["off"] == 1.0).all().all()

    def test_output_shape_varies_with_output_periods(self) -> None:
        """Changing output_periods should change the number of rows returned."""
        data = _make_market_data(n_days=400, n_assets=10, seed=42)

        for periods in [5, 15, 30]:
            strategy = CryptoTrendStrategy(
                lookback_windows=[5, 10],
                vol_targets=[0.50],
                tranches=[1, 5],
                top_by_mcap=8,
                top_by_volume=5,
                output_periods=periods,
                use_duckdb=False,
                use_trailing_stop=False,
                exclude_tickers=[],
            )
            result = strategy.run(data)
            assert len(result["weights"]) == periods

    def test_module_level_run_function(self) -> None:
        """The module-level run() function should produce valid output."""
        data = _make_market_data(n_days=400, n_assets=15, seed=42)

        result = run(
            data,
            params={
                "lookback_windows": [5, 10],
                "vol_targets": [0.50],
                "tranches": [1, 5],
                "use_duckdb": False,
                "use_trailing_stop": False,
                "exclude_tickers": [],
            },
        )

        assert "weights" in result
        assert isinstance(result["weights"], pd.DataFrame)

    def test_get_simple_weights_extracts_slice(self) -> None:
        """get_simple_weights should extract a flat DataFrame from MultiIndex."""
        strategy = CryptoTrendStrategy(
            lookback_windows=[5, 10],
            vol_targets=[0.25, 0.50],
            tranches=[1, 5],
            top_by_mcap=10,
            top_by_volume=5,
            output_periods=5,
            use_duckdb=False,
            use_trailing_stop=False,
            exclude_tickers=[],
        )
        data = _make_market_data(n_days=400, n_assets=10, seed=42)
        result = strategy.run(data)

        simple = get_simple_weights(result["weights"], vol_target="25", tranche=1)

        # Should be a plain DataFrame (not MultiIndex)
        assert not isinstance(simple.columns, pd.MultiIndex)
        assert len(simple) == 5

    def test_get_latest_weights_returns_dict(self) -> None:
        """get_latest_weights should return a dict of positive weights."""
        strategy = CryptoTrendStrategy(
            lookback_windows=[5, 10],
            vol_targets=[0.50],
            tranches=[5],
            top_by_mcap=10,
            top_by_volume=5,
            output_periods=10,
            use_duckdb=False,
            use_trailing_stop=False,
            exclude_tickers=[],
        )
        data = _make_market_data(n_days=400, n_assets=10, seed=42)
        result = strategy.run(data)

        latest = strategy.get_latest_weights(result, vol_target="50", tranche=5)

        assert isinstance(latest, dict)
        for _ticker, weight in latest.items():
            assert weight > 0.001

    def test_meta_attribute(self) -> None:
        """PluginMeta should be a class-level attribute with correct fields."""
        assert CryptoTrendStrategy.meta.name == "strategy.crypto_trend.v1"
        assert CryptoTrendStrategy.meta.kind == "strategy"
        assert "crypto" in CryptoTrendStrategy.meta.tags

    def test_describe_returns_dict(self) -> None:
        """describe() should return a structured dict for introspection."""
        strategy = CryptoTrendStrategy()
        desc = strategy.describe()

        assert desc["name"] == "CryptoTrendCatcher"
        assert desc["type"] == "trend_following"
        assert "parameters" in desc
        assert "methods" in desc


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case tests for robustness."""

    def test_all_nan_prices(self) -> None:
        """Strategy should handle all-NaN prices without crashing."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        symbols = ["A", "B", "C"]

        prices = pd.DataFrame(np.nan, index=dates, columns=symbols)
        volume = pd.DataFrame(0.0, index=dates, columns=symbols)
        market_cap = pd.DataFrame(0.0, index=dates, columns=symbols)

        strategy = CryptoTrendStrategy(
            lookback_windows=[5, 10],
            vol_targets=[0.50],
            tranches=[1, 5],
            top_by_mcap=3,
            top_by_volume=3,
            output_periods=5,
            use_duckdb=False,
            use_trailing_stop=False,
            exclude_tickers=[],
        )

        # Should not raise, even if weights are all NaN / zero
        result = strategy.run({"prices": prices, "volume": volume, "market_cap": market_cap})

        assert "weights" in result
        assert isinstance(result["weights"], pd.DataFrame)

    def test_single_asset(self) -> None:
        """Strategy should work with just one asset."""
        dates = pd.date_range("2024-01-01", periods=400, freq="D")
        rng = np.random.RandomState(77)

        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))),
            index=dates,
            columns=["SOLO"],
        )
        volume = pd.DataFrame(
            rng.uniform(1e4, 1e6, (400, 1)),
            index=dates,
            columns=["SOLO"],
        )
        market_cap = pd.DataFrame(
            rng.uniform(1e9, 1e10, (400, 1)),
            index=dates,
            columns=["SOLO"],
        )

        strategy = CryptoTrendStrategy(
            lookback_windows=[5, 10],
            vol_targets=[0.50],
            tranches=[1, 5],
            top_by_mcap=1,
            top_by_volume=1,
            output_periods=5,
            use_duckdb=False,
            use_trailing_stop=False,
            exclude_tickers=[],
        )

        result = strategy.run({"prices": prices, "volume": volume, "market_cap": market_cap})

        assert "weights" in result
        assert result["weights"].shape[0] == 5

    def test_construct_weights_with_empty_universe(self) -> None:
        """construct_weights should handle an all-zero universe gracefully."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        symbols = ["X", "Y"]

        signals = pd.DataFrame(0.5, index=dates, columns=symbols)
        universe = pd.DataFrame(0.0, index=dates, columns=symbols)
        scalers = {"50": pd.DataFrame(1.0, index=dates, columns=symbols)}

        weights = construct_weights(signals, universe, scalers, tranches=[1], normalize=True)

        # All weights should be 0 or NaN (universe is zero everywhere)
        non_nan = weights.fillna(0.0)
        assert (non_nan == 0.0).all().all()
