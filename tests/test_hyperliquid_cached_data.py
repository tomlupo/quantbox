"""Tests for HyperliquidCachedDataPlugin — hyperliquid.data.cached.v1."""

from __future__ import annotations

import pandas as pd
import pytest

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


class FakeInner:
    """Stand-in for HyperliquidDataPlugin. Records calls; serves from `history`."""

    def __init__(self, history, unavailable=()):
        self.history = history  # series -> full wide frame (date index x symbol)
        self.unavailable = unavailable
        self.calls: list[tuple[int, tuple[str, ...]]] = []

    def load_universe(self, params):
        symbols = params.get("symbols") or list(self.history["prices"].columns)
        return pd.DataFrame({"symbol": symbols})

    def load_market_data(self, universe, asof, params):
        lookback = int(params.get("lookback_days", 365))
        symbols = universe["symbol"].tolist()
        self.calls.append((lookback, tuple(symbols)))
        asof_dt = pd.Timestamp(asof, tz="UTC").normalize()
        start = asof_dt - pd.Timedelta(days=lookback)
        out: dict[str, pd.DataFrame] = {}
        for s, wide in self.history.items():
            cols = [c for c in symbols if c in wide.columns and c not in self.unavailable]
            window = wide.loc[(wide.index >= start) & (wide.index <= asof_dt), cols]
            out[s] = window
        out["market_cap"] = pd.DataFrame()
        return out

    def load_fx(self, asof, params):
        return None


@pytest.fixture
def history():
    """450 days of synthetic prices/volume/funding ending 2026-05-30."""
    dates = pd.date_range("2025-03-06", "2026-05-30", freq="D")
    out = {}
    for s_name, base in (("prices", 100.0), ("volume", 1e6), ("funding_rates", 0.0001)):
        out[s_name] = _wide(["A", "B", "C"], dates, base=base)
    return out


class FakeMcap:
    """Offline, deterministic market-cap/screen-volume provider for tests."""

    def estimate_market_cap(self, prices, volume):
        return prices * 1e6  # mcap = price × fixed supply

    def estimate_aggregate_volume(self, prices):
        agg = pd.DataFrame(index=prices.index)
        for c in prices.columns:
            agg[c] = 1e9
        return agg


def _build(tmp_path, history):
    plugin = HyperliquidCachedDataPlugin(cache_dir=str(tmp_path), overlap_days=3)
    fake = FakeInner(history)
    plugin._inner = fake
    plugin.mcap_provider = FakeMcap()  # keep tests offline + deterministic
    return plugin, fake


class TestLoadMarketData:
    def test_cold_start_full_fetch_and_persist(self, tmp_path, history):
        plugin, fake = _build(tmp_path, history)
        uni = pd.DataFrame({"symbol": ["A", "B", "C"]})
        out = plugin.load_market_data(uni, "2026-05-28", {"lookback_days": 365})

        assert fake.calls == [(365, ("A", "B", "C"))]  # one full-lookback call
        for s in ("prices", "volume", "funding_rates"):
            assert (tmp_path / f"{s}.parquet").exists()
        assert list(out["prices"].columns) == ["A", "B", "C"]
        assert out["prices"].index.max() == pd.Timestamp("2026-05-28", tz="UTC")
        # market_cap + screen_volume now sourced from the (injected) provider
        assert not out["market_cap"].empty
        assert list(out["market_cap"].columns) == ["A", "B", "C"]
        assert not out["screen_volume"].empty

    def test_warm_run_fetches_only_the_gap(self, tmp_path, history):
        plugin, fake = _build(tmp_path, history)
        uni = pd.DataFrame({"symbol": ["A", "B", "C"]})
        plugin.load_market_data(uni, "2026-05-26", {"lookback_days": 365})  # cold
        out = plugin.load_market_data(uni, "2026-05-28", {"lookback_days": 365})  # +2 days

        # gap = 2 days + overlap 3 = 5; all coins cached -> single small call
        assert fake.calls[1] == (5, ("A", "B", "C"))
        # no data loss: still spans the full 365-day window, now up to 2026-05-28
        assert out["prices"].index.min() == pd.Timestamp("2026-05-28", tz="UTC") - pd.Timedelta(days=365)
        assert out["prices"].index.max() == pd.Timestamp("2026-05-28", tz="UTC")

    def test_new_coin_mid_run_uses_separate_full_fetch_bucket(self, tmp_path, history):
        plugin, fake = _build(tmp_path, history)
        plugin.load_market_data(pd.DataFrame({"symbol": ["A", "B"]}), "2026-05-28", {"lookback_days": 365})
        plugin.load_market_data(pd.DataFrame({"symbol": ["A", "B", "C"]}), "2026-05-28", {"lookback_days": 365})

        # second run: new bucket (C, full) then cached bucket (A,B, gap=0+overlap=3)
        assert fake.calls[1] == (365, ("C",))
        assert fake.calls[2] == (3, ("A", "B"))

    def test_returns_respect_window_and_requested_symbols(self, tmp_path, history):
        plugin, fake = _build(tmp_path, history)
        out = plugin.load_market_data(pd.DataFrame({"symbol": ["A", "C"]}), "2026-05-20", {"lookback_days": 30})
        assert list(out["prices"].columns) == ["A", "C"]
        assert out["prices"].index.max() <= pd.Timestamp("2026-05-20", tz="UTC")
        assert out["prices"].index.min() >= pd.Timestamp("2026-05-20", tz="UTC") - pd.Timedelta(days=30)

    def test_empty_universe_returns_empty_frames(self, tmp_path, history):
        plugin, fake = _build(tmp_path, history)
        out = plugin.load_market_data(pd.DataFrame(columns=["symbol"]), "2026-05-20", {})
        assert all(out[s].empty for s in ("prices", "volume", "funding_rates", "market_cap"))
        assert fake.calls == []

    def test_stale_cache_survives_when_inner_returns_no_data_for_coin(self, tmp_path, history):
        plugin, fake = _build(tmp_path, history)
        uni = pd.DataFrame({"symbol": ["A", "B", "C"]})
        plugin.load_market_data(uni, "2026-05-26", {"lookback_days": 365})  # cold: all cached
        fake.unavailable = ("B",)  # B goes dark on the next (warm) fetch
        out = plugin.load_market_data(uni, "2026-05-28", {"lookback_days": 365})
        # B's prior cached history survives even though the warm fetch returned nothing for B
        assert "B" in out["prices"].columns
        assert out["prices"]["B"].notna().any()
        assert out["prices"]["B"].dropna().index.max() == pd.Timestamp("2026-05-26", tz="UTC")
        # A advances to asof (it was fetched normally)
        assert out["prices"]["A"].dropna().index.max() == pd.Timestamp("2026-05-28", tz="UTC")
