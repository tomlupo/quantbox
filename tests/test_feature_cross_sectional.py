"""Tests for CrossSectionalFeatures plugin (features.cross_sectional.v1).

Covers z-score computation, percentile rank, combined methods, output
structure, and plugin metadata. Self-contained — no conftest.py required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.features.cross_sectional import CrossSectionalFeatures


def _make_sample_data(
    n_days: int = 100,
    symbols: tuple[str, ...] = ("BTC", "ETH", "SOL", "AVAX", "DOGE"),
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Build sample market data with random-walk prices for multiple symbols."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_days, freq="B")

    prices = pd.DataFrame(index=dates)
    for i, sym in enumerate(symbols):
        start = 100.0 * (i + 1)
        returns = rng.normal(0.001, 0.02, size=n_days)
        prices[sym] = start * np.cumprod(1 + returns)

    return {"prices": prices}


class TestCrossSectionalFeatures:
    """Test suite for the CrossSectionalFeatures feature plugin."""

    @pytest.fixture()
    def plugin(self) -> CrossSectionalFeatures:
        return CrossSectionalFeatures()

    @pytest.fixture()
    def sample_data(self) -> dict[str, pd.DataFrame]:
        return _make_sample_data()

    # ---- metadata ----

    def test_meta_name(self, plugin: CrossSectionalFeatures) -> None:
        assert plugin.meta.name == "features.cross_sectional.v1"

    def test_meta_kind(self, plugin: CrossSectionalFeatures) -> None:
        assert plugin.meta.kind == "feature"

    # ---- output structure ----

    def test_returns_dataframe_with_multiindex(
        self, plugin: CrossSectionalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {})
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["date", "symbol"]

    def test_multiindex_contains_all_symbols(
        self, plugin: CrossSectionalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {})
        symbols_in_index = sorted(result.index.get_level_values("symbol").unique().tolist())
        assert symbols_in_index == ["AVAX", "BTC", "DOGE", "ETH", "SOL"]

    # ---- z-score computation ----

    def test_zscore_columns_present(
        self, plugin: CrossSectionalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {"methods": ["zscore"]})
        for h in [5, 10, 20, 60]:
            assert f"return_{h}d_zscore" in result.columns

    def test_zscore_roughly_zero_mean(
        self, plugin: CrossSectionalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {"methods": ["zscore"]})
        # For each date, the cross-sectional z-scores should have mean ~0
        for h in [5, 10, 20]:
            col = f"return_{h}d_zscore"
            grouped_mean = result[col].groupby(level="date").mean().dropna()
            assert grouped_mean.abs().mean() < 0.01, f"z-score mean not near zero for {col}"

    def test_zscore_roughly_unit_std(
        self, plugin: CrossSectionalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {"methods": ["zscore"]})
        for h in [5, 10, 20]:
            col = f"return_{h}d_zscore"
            grouped_std = result[col].groupby(level="date").std().dropna()
            # std should be ~1 (ddof=1 for 5 symbols can introduce some variance)
            mean_std = grouped_std.mean()
            assert 0.5 < mean_std < 1.5, f"z-score std not near 1 for {col}: {mean_std}"

    # ---- percentile rank ----

    def test_percentile_columns_present(
        self, plugin: CrossSectionalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {"methods": ["percentile"]})
        for h in [5, 10, 20, 60]:
            assert f"return_{h}d_percentile" in result.columns

    def test_percentile_values_between_zero_and_one(
        self, plugin: CrossSectionalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {"methods": ["percentile"]})
        for h in [5, 10, 20, 60]:
            col = f"return_{h}d_percentile"
            vals = result[col].dropna()
            assert (vals >= 0).all(), f"Percentile below 0 for {col}"
            assert (vals <= 1).all(), f"Percentile above 1 for {col}"

    def test_percentile_uniform_distribution(
        self, plugin: CrossSectionalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        """Percentile ranks across a date should be uniformly distributed."""
        result = plugin.compute(sample_data, {"methods": ["percentile"]})
        col = "return_5d_percentile"
        # For 5 symbols, percentile ranks at each date should be {0.2, 0.4, 0.6, 0.8, 1.0}
        for date in result.index.get_level_values("date").unique()[:10]:
            vals = result.loc[date, col].dropna().sort_values()
            if len(vals) == 5:
                expected = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
                np.testing.assert_allclose(vals.values, expected, atol=0.01)

    # ---- both methods together ----

    def test_both_methods_default(
        self, plugin: CrossSectionalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {})
        for h in [5, 10, 20, 60]:
            assert f"return_{h}d_zscore" in result.columns
            assert f"return_{h}d_percentile" in result.columns

    def test_no_zscore_when_only_percentile(
        self, plugin: CrossSectionalFeatures, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        result = plugin.compute(sample_data, {"methods": ["percentile"]})
        zscore_cols = [c for c in result.columns if c.endswith("_zscore")]
        assert len(zscore_cols) == 0

    # ---- custom horizons ----

    def test_custom_horizons(self, sample_data: dict[str, pd.DataFrame]) -> None:
        plugin = CrossSectionalFeatures()
        result = plugin.compute(sample_data, {"horizons": [3, 7]})
        assert "return_3d_zscore" in result.columns
        assert "return_7d_zscore" in result.columns
        assert "return_5d_zscore" not in result.columns
