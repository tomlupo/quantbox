"""Tests for strategy.trend_catcher.v1 — the notebook-faithful Robuxio v1 strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbox.plugins.strategies import TrendCatcherStrategy
from quantbox.registry import PluginRegistry


class TestRegistration:
    def test_trend_catcher_in_registry(self):
        reg = PluginRegistry.discover()
        assert "strategy.trend_catcher.v1" in reg.strategies, (
            "trend_catcher should be registered as a built-in strategy"
        )

    def test_strategy_meta(self):
        s = TrendCatcherStrategy()
        assert s.meta.name == "strategy.trend_catcher.v1"
        assert s.meta.kind == "strategy"
        assert "trend-following" in s.meta.tags


def _make_synthetic_data(n_days: int = 200, n_assets: int = 8, seed: int = 0):
    """Build synthetic OHLCV + eligibility data for strategy smoke tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    symbols = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOT"][:n_assets]
    rets = rng.normal(0.001, 0.02, size=(n_days, n_assets))
    closes = 100 * np.cumprod(1 + rets, axis=0)
    close_df = pd.DataFrame(closes, index=dates, columns=symbols)
    # Synthetic high/low: small range around close
    high_df = close_df * (1 + np.abs(rng.normal(0, 0.005, size=closes.shape)))
    low_df = close_df * (1 - np.abs(rng.normal(0, 0.005, size=closes.shape)))
    volume_df = pd.DataFrame(
        rng.lognormal(15, 0.5, size=closes.shape) * closes, index=dates, columns=symbols
    )
    elig = pd.DataFrame(True, index=dates, columns=symbols)
    return {
        "prices": close_df,
        "high": high_df,
        "low": low_df,
        "volume": volume_df,
        "eligibility_mask": elig,
    }


class TestEqualWeightSizing:
    def test_runs_and_returns_weights(self):
        data = _make_synthetic_data()
        s = TrendCatcherStrategy(max_positions=5)
        result = s.run(data)
        w = result["weights"]
        assert isinstance(w, pd.DataFrame)
        assert w.shape[0] == data["prices"].shape[0]
        # All weights non-negative (long-only).
        assert (w >= 0).all().all()

    def test_fixed_per_slot_sizing(self):
        """Notebook v1: each selected name gets exactly 1/max_positions."""
        data = _make_synthetic_data()
        s = TrendCatcherStrategy(max_positions=5)
        w = s.run(data)["weights"]
        # All non-zero weights should equal 1/max_positions (= 0.2). Tolerate float noise.
        non_zero = w.values[w.values > 0]
        if non_zero.size > 0:
            assert np.allclose(non_zero, 0.2, atol=1e-9), (
                f"Expected all non-zero weights == 0.2 (1/5), got {set(non_zero.round(6))}"
            )

    def test_row_sum_at_most_one(self):
        """Notebook v1: rows can sum < 1 (cash on unused slots)."""
        data = _make_synthetic_data()
        s = TrendCatcherStrategy(max_positions=5)
        row_sums = s.run(data)["weights"].sum(axis=1)
        assert (row_sums <= 1.0 + 1e-9).all(), "EW weights must never exceed 1.0 per row"


class TestRiskWeightSizing:
    def test_rw_with_ohlc_uses_true_atr(self):
        data = _make_synthetic_data()
        s = TrendCatcherStrategy(max_positions=5, sizing="rw")
        result = s.run(data)
        assert result["details"]["atr_source"] == "true_atr"
        w = result["weights"]
        # Rows that have selected names should sum > 0; tolerate early-period NaN warmup.
        assert (w >= 0).all().all()

    def test_rw_without_ohlc_falls_back_to_close_only_proxy(self):
        data = _make_synthetic_data()
        data["high"] = pd.DataFrame()  # mimic close-only dataset
        data["low"] = pd.DataFrame()
        s = TrendCatcherStrategy(max_positions=5, sizing="rw")
        result = s.run(data)
        assert result["details"]["atr_source"] == "close_only_diff"

    def test_vol_target_back_compat_switches_to_rw(self):
        """Legacy configs using vol_target=<float> should opt into RW sizing."""
        data = _make_synthetic_data()
        s = TrendCatcherStrategy(max_positions=5, vol_target=0.25)
        result = s.run(data)
        assert result["details"]["sizing"] == "rw"


class TestRegimeAndSelection:
    def test_diagnostics_emitted(self):
        data = _make_synthetic_data()
        s = TrendCatcherStrategy(max_positions=5)
        diag = s.run(data)["details"]["diagnostics"]
        assert "regime_overlay" in diag
        assert "signal_count" in diag
        assert diag["regime_overlay"]["slow_window"] == 50
        assert diag["regime_overlay"]["fast_window"] == 20

    def test_eligibility_mask_is_respected(self):
        """When elig=False for a symbol, it should never be selected."""
        data = _make_synthetic_data()
        # Block SOL entirely.
        data["eligibility_mask"]["SOL"] = False
        s = TrendCatcherStrategy(max_positions=5)
        w = s.run(data)["weights"]
        assert (w["SOL"] == 0).all(), "Ineligible ticker should never get weight"
