"""The OHLCV cache compares dates in UTC on every box (replay, quant review of #330).

Before: ``get_cached`` and ``get_last_date`` ran on duckdb's default connection,
whose session timezone is the box's, so ``date >= '2024-01-02'`` on a tz-aware
cache cut the window at 05:00 UTC on a New York box. ``is_fresh`` raised TypeError
on a tz-aware cache (tz-aware last date minus a naive end date). And ``store``
checked duplicates against a window that ended at 00:00 of the last day, so an
intraday re-store wrote the later bars twice; and two stores in one second wrote
the same second-resolution shard name, the second replacing the first's rows.

The "box" here is New York for duckdb: its DEFAULT connection (the one the cache
read through before) and every NEW connection start in America/New_York, as a
box set to New York gives them. So the cache's own connection is only UTC if it
sets that itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.datasources._utils import OHLCVCache

duckdb = pytest.importorskip("duckdb")


@pytest.fixture
def new_york_box(monkeypatch):
    """duckdb in New York time, default and new connections, as on a box set to New York."""
    before = duckdb.execute("SELECT current_setting('TimeZone')").fetchall()[0][0]
    duckdb.execute("SET TimeZone = 'America/New_York'")
    real_connect = duckdb.connect

    def connect(*args, **kwargs):
        con = real_connect(*args, **kwargs)
        con.execute("SET TimeZone = 'America/New_York'")
        return con

    monkeypatch.setattr(duckdb, "connect", connect)
    yield
    duckdb.execute(f"SET TimeZone = '{before}'")


def _hourly_utc(start="2024-01-01", end="2024-01-05 23:00") -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="h", tz="UTC")
    n = len(dates)
    px = np.linspace(100.0, 200.0, n)
    return pd.DataFrame({"date": dates, "open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 1.0})


def test_a_tz_aware_cache_is_cut_at_utc_midnight(tmp_path, new_york_box):
    cache = OHLCVCache(cache_dir=tmp_path)
    cache.store("BTC", _hourly_utc())
    got = cache.get_cached("BTC", "2024-01-02", "2024-01-04")
    dates = pd.DatetimeIndex(got["date"]).tz_convert("UTC")
    assert dates.min() == pd.Timestamp("2024-01-02 00:00", tz="UTC")
    assert dates.max() == pd.Timestamp("2024-01-04 00:00", tz="UTC")
    assert len(got) == 49


def test_is_fresh_answers_on_a_tz_aware_cache(tmp_path, new_york_box):
    cache = OHLCVCache(cache_dir=tmp_path, fresh_ttl_hours=4)
    cache.store("BTC", _hourly_utc())
    assert cache.is_fresh("BTC", "2024-01-06") is True  # last bar 01-05 23:00 UTC, 1 h before
    assert cache.is_fresh("BTC", "2024-01-07") is False


def test_an_intraday_restore_writes_no_duplicate(tmp_path, new_york_box):
    cache = OHLCVCache(cache_dir=tmp_path)
    cache.store("BTC", _hourly_utc())
    cache.store("BTC", _hourly_utc())
    got = cache.get_cached("BTC", "2024-01-01", "2024-01-06")
    assert len(got) == len(_hourly_utc())
    assert not pd.DatetimeIndex(got["date"]).duplicated().any()


def test_two_stores_in_one_second_keep_both(tmp_path):
    """Two different batches stored back to back: both stay (a shard name was per second)."""
    cache = OHLCVCache(cache_dir=tmp_path)
    cache.store("BTC", _hourly_utc("2024-01-01", "2024-01-02 23:00"))
    cache.store("BTC", _hourly_utc("2024-01-03", "2024-01-04 23:00"))
    got = cache.get_cached("BTC", "2024-01-01", "2024-01-05")
    assert len(got) == 96
