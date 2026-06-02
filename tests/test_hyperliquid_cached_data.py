"""Tests for HyperliquidCachedDataPlugin — hyperliquid.data.cached.v1."""

from __future__ import annotations

import pandas as pd

from quantbox.plugins.datasources.hyperliquid_cached_data_plugin import (
    HyperliquidCachedDataPlugin,
)


def _wide(symbols, dates, base=100.0):
    """Wide frame: DatetimeIndex (UTC midnight) x symbol columns."""
    idx = pd.to_datetime(dates, utc=True).normalize()
    data = {s: [base + i for i in range(len(idx))] for s in symbols}
    return pd.DataFrame(data, index=idx)


class TestHelpers:
    def test_wide_to_long_roundtrips_to_long_to_wide(self):
        dates = pd.date_range("2026-01-01", periods=3, freq="D")
        wide = _wide(["A", "B"], dates)
        long_df = HyperliquidCachedDataPlugin._wide_to_long(wide)
        assert list(long_df.columns) == ["date", "symbol", "value"]
        assert len(long_df) == 6  # 3 dates x 2 symbols
        start = pd.Timestamp("2026-01-01", tz="UTC")
        asof = pd.Timestamp("2026-01-03", tz="UTC")
        back = HyperliquidCachedDataPlugin._long_to_wide(long_df, ["A", "B"], start, asof)
        pd.testing.assert_frame_equal(back, wide, check_names=False, check_freq=False)

    def test_wide_to_long_drops_nan(self):
        dates = pd.date_range("2026-01-01", periods=2, freq="D")
        wide = _wide(["A"], dates)
        wide.loc[wide.index[0], "A"] = float("nan")
        long_df = HyperliquidCachedDataPlugin._wide_to_long(wide)
        assert len(long_df) == 1

    def test_long_to_wide_restricts_window_and_symbols(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        long_df = HyperliquidCachedDataPlugin._wide_to_long(_wide(["A", "B", "C"], dates))
        start = pd.Timestamp("2026-01-02", tz="UTC")
        asof = pd.Timestamp("2026-01-04", tz="UTC")
        wide = HyperliquidCachedDataPlugin._long_to_wide(long_df, ["A", "B"], start, asof)
        assert list(wide.columns) == ["A", "B"]
        assert wide.index.min() == start and wide.index.max() == asof

    def test_read_cache_missing_returns_empty_schema(self, tmp_path):
        plugin = HyperliquidCachedDataPlugin(cache_dir=str(tmp_path))
        df = plugin._read_cache("prices")
        assert list(df.columns) == ["date", "symbol", "value"]
        assert df.empty

    def test_write_then_read_cache_roundtrip(self, tmp_path):
        plugin = HyperliquidCachedDataPlugin(cache_dir=str(tmp_path))
        long_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
                "symbol": ["A", "A"],
                "value": [1.0, 2.0],
            }
        )
        plugin._write_cache("prices", long_df)
        assert (tmp_path / "prices.parquet").exists()
        out = plugin._read_cache("prices")
        pd.testing.assert_frame_equal(
            out.sort_values(["symbol", "date"]).reset_index(drop=True),
            long_df,
            check_dtype=False,
        )
