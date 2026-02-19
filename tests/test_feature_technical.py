"""Tests for TechnicalFeatures plugin (features.technical.v1).

Covers DataFrame shape, MultiIndex structure, expected feature columns,
volume features, and plugin metadata. Self-contained — no conftest.py required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.features.technical import TechnicalFeatures


def _make_sample_data(
    n_days: int = 100,
    symbols: tuple[str, ...] = ("BTC", "ETH"),
    include_volume: bool = False,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Build sample market data with random-walk prices."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_days, freq="B")

    prices = pd.DataFrame(index=dates)
    for sym in symbols:
        start = 100.0 if sym == "ETH" else 40000.0
        returns = rng.normal(0.001, 0.02, size=n_days)
        prices[sym] = start * np.cumprod(1 + returns)

    data: dict[str, pd.DataFrame] = {"prices": prices}

    if include_volume:
        volume = pd.DataFrame(index=dates)
        for sym in symbols:
            base = 1_000_000 if sym == "BTC" else 500_000
            volume[sym] = base * (1 + rng.normal(0, 0.3, size=n_days)).clip(0.1)
        data["volume"] = volume

    return data


class TestTechnicalFeatures:
    """Test suite for the TechnicalFeatures feature plugin."""

    @pytest.fixture()
    def plugin(self) -> TechnicalFeatures:
        return TechnicalFeatures()

    @pytest.fixture()
    def sample_data(self) -> dict[str, pd.DataFrame]:
        return _make_sample_data()

    @pytest.fixture()
    def sample_data_with_volume(self) -> dict[str, pd.DataFrame]:
        return _make_sample_data(include_volume=True)

    # ---- metadata ----

    def test_meta_name(self, plugin: TechnicalFeatures) -> None:
        assert plugin.meta.name == "features.technical.v1"

    def test_meta_kind(self, plugin: TechnicalFeatures) -> None:
        assert plugin.meta.kind == "feature"

    # ---- output structure ----

    def test_returns_dataframe_with_multiindex(
        self, plugin: TechnicalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {})
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["date", "symbol"]

    def test_multiindex_contains_all_symbols(
        self, plugin: TechnicalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {})
        symbols_in_index = result.index.get_level_values("symbol").unique().tolist()
        assert sorted(symbols_in_index) == ["BTC", "ETH"]

    def test_multiindex_dates_match_input(
        self, plugin: TechnicalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {})
        dates_in_index = result.index.get_level_values("date").unique()
        assert len(dates_in_index) == len(sample_data["prices"].index)

    # ---- expected feature columns ----

    def test_expected_core_columns_exist(
        self, plugin: TechnicalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {})
        expected = ["rsi_14", "macd", "bb_position_20d", "return_5d", "volatility_20d"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_return_columns_for_each_lookback(
        self, plugin: TechnicalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {})
        for p in [5, 10, 20, 60]:
            assert f"return_{p}d" in result.columns
            assert f"volatility_{p}d" in result.columns
            assert f"momentum_{p}d" in result.columns
            assert f"sma_ratio_{p}d" in result.columns

    def test_rsi_columns(
        self, plugin: TechnicalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {})
        assert "rsi_14" in result.columns
        assert "rsi_28" in result.columns

    def test_macd_columns(
        self, plugin: TechnicalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {})
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

    def test_day_cyclical_columns(
        self, plugin: TechnicalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {})
        assert "day_sin" in result.columns
        assert "day_cos" in result.columns

    # ---- volume features ----

    def test_volume_features_included_when_volume_provided(
        self, plugin: TechnicalFeatures, sample_data_with_volume: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data_with_volume, {})
        for p in [5, 10, 20, 60]:
            assert f"volume_ratio_{p}d" in result.columns
            assert f"volume_trend_{p}d" in result.columns

    def test_volume_features_absent_without_volume(
        self, plugin: TechnicalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {})
        volume_cols = [c for c in result.columns if c.startswith("volume_")]
        assert len(volume_cols) == 0

    # ---- custom lookback periods ----

    def test_custom_lookback_periods(self, sample_data: dict[str, pd.DataFrame]) -> None:
        plugin = TechnicalFeatures()
        result = plugin.compute(sample_data, {"lookback_periods": [3, 7]})
        assert "return_3d" in result.columns
        assert "return_7d" in result.columns
        assert "return_5d" not in result.columns

    # ---- values sanity ----

    def test_rsi_values_bounded(
        self, plugin: TechnicalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {})
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_no_infinite_values(
        self, plugin: TechnicalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {})
        assert not np.isinf(result.select_dtypes(include=[np.number]).values).any()
