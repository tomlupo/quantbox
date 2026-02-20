"""Tests for the AltcoinCrashBounceStrategy plugin and its helper functions.

Covers:
- Crash signal computation (rising prices, sharp drop)
- Volume ratio computation (constant volume, spike)
- Entry signal generation (crash only, volume only, both)
- Position simulation (TP exit, SL exit, time decay, max positions, circuit breaker)
- Full strategy run (happy path, param overrides, describe, meta, module-level run)
- Edge cases: all-NaN prices, single asset, empty universe after filtering
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbox.plugins.strategies.altcoin_crash_bounce import (
    AltcoinCrashBounceStrategy,
    compute_crash_signals,
    compute_volume_ratio,
    generate_entry_signals,
    run,
    simulate_positions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hourly_prices(
    n_periods: int = 2500,
    n_assets: int = 5,
    seed: int = 42,
    prefix: str = "ALT",
) -> pd.DataFrame:
    """Synthetic random-walk prices with hourly frequency."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n_periods, freq="h")
    symbols = [f"{prefix}{i}" for i in range(n_assets)]
    log_ret = rng.normal(0.0001, 0.005, (n_periods, n_assets))
    prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(prices, index=dates, columns=symbols)


def _make_hourly_market_data(
    n_periods: int = 2500,
    n_assets: int = 5,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Return synthetic hourly prices, volume, and market_cap."""
    rng = np.random.RandomState(seed)
    prices = _make_hourly_prices(n_periods, n_assets, seed)
    dates = prices.index
    symbols = list(prices.columns)

    volume = pd.DataFrame(
        rng.uniform(1e6, 1e8, (n_periods, n_assets)),
        index=dates,
        columns=symbols,
    )
    market_cap = pd.DataFrame(
        rng.uniform(1e8, 1e10, (n_periods, n_assets)),
        index=dates,
        columns=symbols,
    )
    return {"prices": prices, "volume": volume, "market_cap": market_cap}


# ---------------------------------------------------------------------------
# Crash Signal Tests
# ---------------------------------------------------------------------------


class TestCrashSignals:
    """Tests for compute_crash_signals."""

    def test_rising_prices_no_crash(self) -> None:
        """Monotonically rising prices should never trigger crash signals."""
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        prices = pd.DataFrame(
            {"A": np.linspace(100, 200, 100)},
            index=dates,
        )

        crash_pct = compute_crash_signals(prices, lookback_periods=24)

        # After warmup, rising prices are always at rolling high -> crash_pct = 0
        valid = crash_pct.iloc[24:]
        assert (valid["A"] >= -0.01).all()  # Allow tiny float errors

    def test_sharp_drop_triggers_crash(self) -> None:
        """A 20% drop should produce crash_pct <= -15%."""
        dates = pd.date_range("2024-01-01", periods=50, freq="h")
        # Flat at 100 for 30h, then sharp drop to 80 (20% drop)
        vals = np.full(50, 100.0)
        vals[30:] = 80.0
        prices = pd.DataFrame({"A": vals}, index=dates)

        crash_pct = compute_crash_signals(prices, lookback_periods=24)

        # After the drop, crash_pct should be -20%
        assert crash_pct.loc[dates[35], "A"] < -15.0

    def test_output_shape_matches_input(self) -> None:
        """Output should have same shape as input."""
        prices = _make_hourly_prices(n_periods=100, n_assets=3)
        crash_pct = compute_crash_signals(prices, lookback_periods=10)
        assert crash_pct.shape == prices.shape

    def test_warmup_is_nan(self) -> None:
        """First (lookback-1) rows should be NaN."""
        dates = pd.date_range("2024-01-01", periods=50, freq="h")
        prices = pd.DataFrame({"A": np.linspace(100, 150, 50)}, index=dates)

        crash_pct = compute_crash_signals(prices, lookback_periods=10)

        assert crash_pct.iloc[:9]["A"].isna().all()


# ---------------------------------------------------------------------------
# Volume Ratio Tests
# ---------------------------------------------------------------------------


class TestVolumeRatio:
    """Tests for compute_volume_ratio."""

    def test_constant_volume_ratio_near_one(self) -> None:
        """Constant volume should give ratio ~1.0 after warmup."""
        dates = pd.date_range("2024-01-01", periods=600, freq="h")
        volume = pd.DataFrame({"A": 1000.0}, index=dates)

        ratio = compute_volume_ratio(volume, lookback_periods=480)

        # After warmup, ratio should be ~1.0
        valid = ratio.iloc[480:]
        assert np.allclose(valid["A"].values, 1.0, atol=0.01)

    def test_spike_produces_high_ratio(self) -> None:
        """A volume spike should produce ratio > threshold."""
        dates = pd.date_range("2024-01-01", periods=600, freq="h")
        vals = np.full(600, 1000.0)
        vals[550:] = 5000.0  # 5x spike
        volume = pd.DataFrame({"A": vals}, index=dates)

        ratio = compute_volume_ratio(volume, lookback_periods=480)

        # Spike should have ratio well above 1.3
        assert ratio.loc[dates[560], "A"] > 1.3

    def test_output_shape(self) -> None:
        """Output shape should match input."""
        volume = pd.DataFrame(
            np.random.RandomState(1).uniform(100, 1000, (500, 3)),
            index=pd.date_range("2024-01-01", periods=500, freq="h"),
            columns=["A", "B", "C"],
        )
        ratio = compute_volume_ratio(volume, lookback_periods=100)
        assert ratio.shape == volume.shape


# ---------------------------------------------------------------------------
# Entry Signal Tests
# ---------------------------------------------------------------------------


class TestEntrySignals:
    """Tests for generate_entry_signals."""

    def test_crash_only_no_signal(self) -> None:
        """Crash without volume spike should NOT generate signal."""
        dates = pd.date_range("2024-01-01", periods=600, freq="h")
        # Create a crash
        price_vals = np.full(600, 100.0)
        price_vals[500:] = 80.0  # 20% drop
        prices = pd.DataFrame({"A": price_vals}, index=dates)
        # Constant volume (no spike)
        volume = pd.DataFrame({"A": 1000.0}, index=dates)

        signals = generate_entry_signals(
            prices,
            volume,
            crash_threshold_pct=-15.0,
            volume_spike_ratio=1.3,
            lookback_periods=24,
            volume_lookback_periods=480,
        )

        # Should not fire — volume condition not met
        assert not signals.iloc[500:]["A"].any()

    def test_volume_only_no_signal(self) -> None:
        """Volume spike without crash should NOT generate signal."""
        dates = pd.date_range("2024-01-01", periods=600, freq="h")
        # Rising prices (no crash)
        prices = pd.DataFrame(
            {"A": np.linspace(100, 200, 600)},
            index=dates,
        )
        # Volume spike
        vol_vals = np.full(600, 1000.0)
        vol_vals[500:] = 5000.0
        volume = pd.DataFrame({"A": vol_vals}, index=dates)

        signals = generate_entry_signals(
            prices,
            volume,
            crash_threshold_pct=-15.0,
            volume_spike_ratio=1.3,
            lookback_periods=24,
            volume_lookback_periods=480,
        )

        assert not signals.iloc[500:]["A"].any()

    def test_both_conditions_signal_fires(self) -> None:
        """Crash + volume spike should generate entry signal."""
        dates = pd.date_range("2024-01-01", periods=600, freq="h")
        # Stable then crash
        price_vals = np.full(600, 100.0)
        price_vals[500:] = 80.0  # 20% drop
        prices = pd.DataFrame({"A": price_vals}, index=dates)
        # Volume spike at the crash
        vol_vals = np.full(600, 1000.0)
        vol_vals[500:] = 5000.0
        volume = pd.DataFrame({"A": vol_vals}, index=dates)

        signals = generate_entry_signals(
            prices,
            volume,
            crash_threshold_pct=-15.0,
            volume_spike_ratio=1.3,
            lookback_periods=24,
            volume_lookback_periods=480,
        )

        # Should fire after crash with volume
        assert signals.iloc[500:]["A"].any()

    def test_output_is_boolean(self) -> None:
        """Entry signals should be boolean."""
        data = _make_hourly_market_data(n_periods=600, n_assets=3)
        signals = generate_entry_signals(
            data["prices"],
            data["volume"],
        )
        assert signals.dtypes.apply(lambda d: d is np.dtype(bool)).all()


# ---------------------------------------------------------------------------
# Position Simulation Tests
# ---------------------------------------------------------------------------


class TestSimulatePositions:
    """Tests for simulate_positions."""

    def _make_simple_scenario(
        self,
        n_periods: int = 100,
        n_assets: int = 2,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create prices and entry_signals for controlled tests."""
        dates = pd.date_range("2024-01-01", periods=n_periods, freq="h")
        tickers = [f"T{i}" for i in range(n_assets)]
        prices = pd.DataFrame(100.0, index=dates, columns=tickers)
        signals = pd.DataFrame(False, index=dates, columns=tickers)
        return prices, signals

    def test_no_signals_no_trades(self) -> None:
        """With no entry signals, there should be no trades."""
        prices, signals = self._make_simple_scenario()

        weights, details = simulate_positions(prices, signals)

        assert details["n_trades"] == 0
        assert (weights == 0.0).all().all()

    def test_tp_exit(self) -> None:
        """Position should exit at take profit."""
        prices, signals = self._make_simple_scenario(n_periods=100, n_assets=1)
        # Entry signal at period 10
        signals.iloc[10, 0] = True
        # Price rises to trigger TP (+12% from entry with slippage)
        entry_price = 100.0 * (1 + 2.5 / 100)  # 102.5 with slippage
        tp_price = entry_price * (1 + 12.0 / 100)  # ~114.8
        prices.iloc[10:, 0] = 100.0  # entry price
        prices.iloc[20:, 0] = tp_price + 1  # TP trigger

        weights, details = simulate_positions(
            prices,
            signals,
            take_profit_pct=12.0,
            slippage_pct=2.5,
        )

        assert details["n_trades"] == 1
        assert details["trades"][0]["exit_reason"] == "take_profit"
        # Weight should be 0 after exit
        assert weights.iloc[25, 0] == 0.0

    def test_sl_exit(self) -> None:
        """Position should exit at stop loss."""
        prices, signals = self._make_simple_scenario(n_periods=100, n_assets=1)
        signals.iloc[10, 0] = True
        # Price drops to trigger SL (-12% from entry with slippage)
        entry_price = 100.0 * (1 + 2.5 / 100)  # 102.5 with slippage
        sl_price = entry_price * (1 - 12.0 / 100)  # ~90.2
        prices.iloc[10:, 0] = 100.0
        prices.iloc[20:, 0] = sl_price - 1  # SL trigger

        weights, details = simulate_positions(
            prices,
            signals,
            stop_loss_pct=12.0,
            slippage_pct=2.5,
        )

        assert details["n_trades"] == 1
        assert details["trades"][0]["exit_reason"] == "stop_loss"

    def test_time_decay_exit(self) -> None:
        """Position should exit after max hold periods."""
        prices, signals = self._make_simple_scenario(n_periods=200, n_assets=1)
        signals.iloc[10, 0] = True
        # Price stays flat (no TP/SL trigger)

        weights, details = simulate_positions(
            prices,
            signals,
            max_hold_periods=50,
            take_profit_pct=99.0,  # Won't trigger
            stop_loss_pct=99.0,  # Won't trigger
        )

        assert details["n_trades"] == 1
        assert details["trades"][0]["exit_reason"] == "time_decay"
        assert details["trades"][0]["hold_periods"] == 50

    def test_max_positions_cap(self) -> None:
        """Should not exceed max_positions."""
        dates = pd.date_range("2024-01-01", periods=200, freq="h")
        tickers = [f"T{i}" for i in range(10)]
        prices = pd.DataFrame(100.0, index=dates, columns=tickers)
        signals = pd.DataFrame(False, index=dates, columns=tickers)
        # Signal all 10 tickers at period 10
        signals.iloc[10] = True

        weights, details = simulate_positions(
            prices,
            signals,
            max_positions=3,
            take_profit_pct=99.0,
            stop_loss_pct=99.0,
            max_hold_periods=999,
        )

        # At period 15, max 3 positions should be active
        active = (weights.iloc[15] > 0).sum()
        assert active == 3

    def test_circuit_breaker(self) -> None:
        """Circuit breaker should limit entries per rolling window."""
        dates = pd.date_range("2024-01-01", periods=200, freq="h")
        tickers = [f"T{i}" for i in range(20)]
        prices = pd.DataFrame(100.0, index=dates, columns=tickers)
        signals = pd.DataFrame(False, index=dates, columns=tickers)
        # Signal all tickers at period 10
        signals.iloc[10] = True

        weights, details = simulate_positions(
            prices,
            signals,
            max_positions=20,
            circuit_breaker_entries=4,
            circuit_breaker_periods=24,
            take_profit_pct=99.0,
            stop_loss_pct=99.0,
            max_hold_periods=999,
        )

        # At period 11, at most 4 entries should have happened
        active = (weights.iloc[11] > 0).sum()
        assert active <= 4

    def test_weight_values(self) -> None:
        """Active positions should have correct weight (position_size_pct / 100)."""
        prices, signals = self._make_simple_scenario(n_periods=100, n_assets=1)
        signals.iloc[10, 0] = True

        weights, _ = simulate_positions(
            prices,
            signals,
            position_size_pct=8.0,
            take_profit_pct=99.0,
            stop_loss_pct=99.0,
            max_hold_periods=999,
        )

        # Position weight should be 0.08
        assert weights.iloc[15, 0] == 0.08

    def test_trade_log_fields(self) -> None:
        """Trade log entries should have all required fields."""
        prices, signals = self._make_simple_scenario(n_periods=200, n_assets=1)
        signals.iloc[10, 0] = True

        _, details = simulate_positions(
            prices,
            signals,
            max_hold_periods=50,
            take_profit_pct=99.0,
            stop_loss_pct=99.0,
        )

        assert details["n_trades"] == 1
        trade = details["trades"][0]
        required_fields = {
            "ticker",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "pct_return",
            "hold_periods",
            "exit_reason",
        }
        assert required_fields.issubset(trade.keys())

    def test_slippage_applied(self) -> None:
        """Entry price should include slippage."""
        prices, signals = self._make_simple_scenario(n_periods=200, n_assets=1)
        signals.iloc[10, 0] = True

        _, details = simulate_positions(
            prices,
            signals,
            slippage_pct=2.5,
            max_hold_periods=50,
            take_profit_pct=99.0,
            stop_loss_pct=99.0,
        )

        trade = details["trades"][0]
        # Entry price should be 100 * 1.025 = 102.5
        assert abs(trade["entry_price"] - 102.5) < 0.01


# ---------------------------------------------------------------------------
# Full Strategy Run Tests
# ---------------------------------------------------------------------------


class TestAltcoinCrashBounceStrategy:
    """Integration tests for AltcoinCrashBounceStrategy."""

    def test_run_happy_path(self) -> None:
        """Strategy run should return weights, simple_weights, and details."""
        strategy = AltcoinCrashBounceStrategy(
            exclude_tickers=[],
            min_daily_volume_usd=0,
            min_market_cap_usd=0,
            output_periods=100,
        )
        data = _make_hourly_market_data(n_periods=2500, n_assets=5, seed=42)

        result = strategy.run(data)

        assert "weights" in result
        assert "simple_weights" in result
        assert "details" in result
        assert isinstance(result["weights"], pd.DataFrame)
        assert isinstance(result["simple_weights"], dict)
        assert len(result["weights"]) == 100

    def test_meta_attribute(self) -> None:
        """PluginMeta should be a class-level attribute with correct fields."""
        assert AltcoinCrashBounceStrategy.meta.name == "strategy.altcoin_crash_bounce.v62"
        assert AltcoinCrashBounceStrategy.meta.kind == "strategy"
        assert "crypto" in AltcoinCrashBounceStrategy.meta.tags
        assert "mean_reversion" in AltcoinCrashBounceStrategy.meta.tags

    def test_describe(self) -> None:
        """describe() should return a structured dict for introspection."""
        strategy = AltcoinCrashBounceStrategy()
        desc = strategy.describe()

        assert desc["name"] == "AltcoinCrashBounce"
        assert desc["type"] == "mean_reversion"
        assert desc["data_frequency"] == "hourly"
        assert "parameters" in desc
        assert "risk" in desc

    def test_param_overrides(self) -> None:
        """Strategy should accept parameter overrides via run()."""
        strategy = AltcoinCrashBounceStrategy(
            exclude_tickers=[],
            min_daily_volume_usd=0,
            min_market_cap_usd=0,
        )
        data = _make_hourly_market_data(n_periods=2500, n_assets=3, seed=42)

        result = strategy.run(
            data,
            params={
                "crash_threshold_pct": -20.0,
                "volume_spike_ratio": 2.0,
                "output_periods": 50,
            },
        )

        assert strategy.crash_threshold_pct == -20.0
        assert strategy.volume_spike_ratio == 2.0
        assert len(result["weights"]) == 50

    def test_module_level_run_function(self) -> None:
        """The module-level run() function should produce valid output."""
        data = _make_hourly_market_data(n_periods=2500, n_assets=5, seed=42)

        result = run(
            data,
            params={
                "exclude_tickers": [],
                "min_daily_volume_usd": 0,
                "min_market_cap_usd": 0,
            },
        )

        assert "weights" in result
        assert isinstance(result["weights"], pd.DataFrame)

    def test_get_latest_weights(self) -> None:
        """get_latest_weights should return a dict of positive weights."""
        strategy = AltcoinCrashBounceStrategy(
            exclude_tickers=[],
            min_daily_volume_usd=0,
            min_market_cap_usd=0,
        )
        data = _make_hourly_market_data(n_periods=2500, n_assets=5, seed=42)
        result = strategy.run(data)

        latest = strategy.get_latest_weights(result)

        assert isinstance(latest, dict)
        for weight in latest.values():
            assert weight > 0.001


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case tests for robustness."""

    def test_all_nan_prices(self) -> None:
        """Strategy should handle all-NaN prices without crashing."""
        dates = pd.date_range("2024-01-01", periods=2500, freq="h")
        symbols = ["A", "B", "C"]

        prices = pd.DataFrame(np.nan, index=dates, columns=symbols)
        volume = pd.DataFrame(0.0, index=dates, columns=symbols)

        strategy = AltcoinCrashBounceStrategy(
            exclude_tickers=[],
            min_daily_volume_usd=0,
            min_market_cap_usd=0,
            output_periods=100,
        )

        result = strategy.run({"prices": prices, "volume": volume})

        assert "weights" in result
        assert isinstance(result["weights"], pd.DataFrame)

    def test_single_asset(self) -> None:
        """Strategy should work with just one asset."""
        rng = np.random.RandomState(77)
        dates = pd.date_range("2024-01-01", periods=2500, freq="h")

        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0, 0.005, 2500))),
            index=dates,
            columns=["SOLO"],
        )
        volume = pd.DataFrame(
            rng.uniform(1e6, 1e8, (2500, 1)),
            index=dates,
            columns=["SOLO"],
        )

        strategy = AltcoinCrashBounceStrategy(
            exclude_tickers=[],
            min_daily_volume_usd=0,
            min_market_cap_usd=0,
            output_periods=100,
        )

        result = strategy.run({"prices": prices, "volume": volume})

        assert "weights" in result
        assert result["weights"].shape[0] == 100

    def test_empty_universe_after_filtering(self) -> None:
        """If all tickers are excluded, should return empty weights gracefully."""
        dates = pd.date_range("2024-01-01", periods=2500, freq="h")
        # Use stablecoin names that will be filtered out
        symbols = ["USDT", "USDC", "BUSD"]

        prices = pd.DataFrame(1.0, index=dates, columns=symbols)
        volume = pd.DataFrame(1e8, index=dates, columns=symbols)

        strategy = AltcoinCrashBounceStrategy(output_periods=100)

        result = strategy.run({"prices": prices, "volume": volume})

        assert result["simple_weights"] == {}
        assert result["details"]["n_trades"] == 0

    def test_no_volume_data(self) -> None:
        """Strategy should handle missing volume data."""
        strategy = AltcoinCrashBounceStrategy(
            exclude_tickers=[],
            min_daily_volume_usd=0,
            min_market_cap_usd=0,
            output_periods=100,
        )
        prices = _make_hourly_prices(n_periods=2500, n_assets=3)
        data = {"prices": prices}  # No volume key

        result = strategy.run(data)

        assert "weights" in result
        assert isinstance(result["weights"], pd.DataFrame)
