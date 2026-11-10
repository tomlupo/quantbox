"""A tz-aware OHLCV cache is a cache hit in every ccxt datasource (replay, follow-up of #332).

#332 made OHLCVCache compare dates in UTC; the plugins' own "complete cache hit"
check still compared the cache's last date with a naive ``pd.Timestamp(end_date)``.
On a cache that holds tz-aware dates that raised TypeError ("Cannot compare
tz-naive and tz-aware timestamps") out of ``get_ohlcv``. Red at dev 7f306ab.

The exchange is never reached: a hit returns the cache, and any fetch fails the
test. ccxt is not in the dev env, so the test injects a stand-in exchange and
marks ccxt available; nothing here calls ccxt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.datasources import binance_data, binance_futures_data, kraken_data

pytest.importorskip("duckdb")


def _daily_utc() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")
    px = np.linspace(100.0, 110.0, len(dates))
    return pd.DataFrame({"date": dates, "open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 1.0})


def _no_exchange(*args, **kwargs):
    raise AssertionError("a complete cache hit must not reach the exchange")


@pytest.mark.parametrize(
    ("module", "fetcher_name"),
    [
        (binance_data, "BinanceDataFetcher"),
        (binance_futures_data, "BinanceFuturesDataFetcher"),
        (kraken_data, "KrakenDataFetcher"),
    ],
)
def test_a_tz_aware_cache_is_a_hit(tmp_path, monkeypatch, module, fetcher_name):
    monkeypatch.setattr(module, "CCXT_AVAILABLE", True)
    fetcher = getattr(module, fetcher_name)(cache_dir=str(tmp_path), _exchange=object())
    fetcher._ohlcv_cache.store("BTC", _daily_utc())
    monkeypatch.setattr(fetcher, "_fetch_ohlcv_with_retry", _no_exchange)
    monkeypatch.setattr(fetcher, "get_valid_pairs", _no_exchange, raising=False)  # kraken has none

    got = fetcher.get_ohlcv("BTC", "2024-01-01", "2024-01-10")

    assert got is not None
    assert len(got) == 10
    assert pd.DatetimeIndex(got["date"]).max() == pd.Timestamp("2024-01-10", tz="UTC")
