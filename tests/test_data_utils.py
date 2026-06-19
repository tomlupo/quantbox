"""Tests for quantbox.plugins.datasources._utils module.

Covers:
- OHLCV validation (validate_ohlcv)
- Transient error classification (is_transient)
- DuckDB-backed OHLCV cache (OHLCVCache)
- Retry decorator (retry_transient)
"""

import time

import httpx
import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.datasources._utils import (
    MarketCapProvider,
    OHLCVCache,
    is_transient,
    retry_transient,
    validate_ohlcv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Create a valid OHLCV DataFrame with *n* rows."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.randn(n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.randn(n) * 0.5
    volume = rng.uniform(1e6, 5e6, n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


# ===========================================================================
# TestValidateOHLCV
# ===========================================================================


class TestValidateOHLCV:
    """Tests for validate_ohlcv()."""

    def test_valid_dataframe_passes(self):
        """A well-formed OHLCV DataFrame passes validation unchanged."""
        df = _make_ohlcv()
        result = validate_ohlcv(df, "BTC")
        assert len(result) == len(df)
        assert list(result.columns) >= ["date", "open", "high", "low", "close", "volume"]
        # Should be sorted ascending
        assert result["date"].is_monotonic_increasing

    def test_missing_columns_raises(self):
        """Missing required columns raise ValueError."""
        df = _make_ohlcv().drop(columns=["close", "volume"])
        with pytest.raises(ValueError, match="Missing OHLCV columns"):
            validate_ohlcv(df, "ETH")

    def test_negative_prices_dropped(self):
        """Rows with negative prices are dropped (not raised)."""
        df = _make_ohlcv(n=10)
        df = df.copy()
        # Inject two rows with negative close
        df.loc[3, "close"] = -5.0
        df.loc[7, "open"] = -1.0
        result = validate_ohlcv(df, "SOL")
        # The two bad rows should be gone
        assert len(result) == 8
        assert (result[["open", "high", "low", "close"]] > 0).all().all()

    def test_zero_prices_dropped(self):
        """Rows where any price column is zero are dropped."""
        df = _make_ohlcv(n=5)
        df = df.copy()
        df.loc[0, "low"] = 0.0
        result = validate_ohlcv(df, "DOGE")
        assert len(result) == 4

    def test_empty_after_cleaning_raises(self):
        """If all rows are invalid, a ValueError is raised."""
        df = _make_ohlcv(n=3)
        df = df.copy()
        df["close"] = -1.0  # every row has a negative price
        with pytest.raises(ValueError, match="No valid rows after cleaning"):
            validate_ohlcv(df, "SHIB")

    def test_empty_dataframe_raises(self):
        """An empty DataFrame raises ValueError."""
        df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        with pytest.raises(ValueError, match="Empty OHLCV DataFrame"):
            validate_ohlcv(df, "ADA")

    def test_duplicate_dates_deduplicated(self):
        """Duplicate dates are collapsed (keep last)."""
        df = _make_ohlcv(n=5)
        dup = pd.concat([df, df.iloc[[2]]], ignore_index=True)
        assert len(dup) == 6
        result = validate_ohlcv(dup, "BTC")
        assert len(result) == 5  # one duplicate removed

    def test_high_low_swap(self):
        """When high < low, the values are swapped."""
        df = _make_ohlcv(n=5)
        df = df.copy()
        # Force high < low on row 2
        df.loc[2, "high"] = 50.0
        df.loc[2, "low"] = 60.0
        result = validate_ohlcv(df, "XRP")
        row = result[result["date"] == df.loc[2, "date"]]
        assert row["high"].iloc[0] >= row["low"].iloc[0]


# ===========================================================================
# TestIsTransient
# ===========================================================================


class TestIsTransient:
    """Tests for is_transient() error classifier."""

    @pytest.mark.parametrize(
        "exc",
        [
            ConnectionError("connection reset"),
            TimeoutError("timed out"),
            OSError("network unreachable"),
            httpx.ConnectError("failed to connect"),
        ],
        ids=["ConnectionError", "TimeoutError", "OSError", "httpx.ConnectError"],
    )
    def test_known_transient_errors(self, exc):
        """Standard transient exception types return True."""
        assert is_transient(exc) is True

    def test_ccxt_style_exception_names(self):
        """Exceptions with ccxt-style class names are treated as transient."""
        for name in (
            "RateLimitExceeded",
            "ExchangeNotAvailable",
            "RequestTimeout",
            "NetworkError",
            "DDoSProtection",
        ):
            exc = type(name, (Exception,), {})("simulated")
            assert is_transient(exc) is True, f"Expected True for {name}"

    def test_binance_transient_codes(self):
        """Exceptions carrying Binance transient error codes are transient."""
        for code in (-1003, -1001, -1000, 503, 504):
            exc = Exception("api error")
            exc.code = code  # type: ignore[attr-defined]
            assert is_transient(exc) is True, f"Expected True for code={code}"

    def test_non_transient_errors(self):
        """Ordinary exceptions are not transient."""
        assert is_transient(ValueError("bad value")) is False
        assert is_transient(KeyError("missing")) is False
        assert is_transient(TypeError("wrong type")) is False
        assert is_transient(RuntimeError("generic")) is False


# ===========================================================================
# TestOHLCVCache
# ===========================================================================


class TestOHLCVCache:
    """Tests for the DuckDB-backed OHLCVCache."""

    def test_store_and_retrieve(self, tmp_path):
        """Stored data can be retrieved via get_cached()."""
        cache = OHLCVCache(cache_dir=tmp_path)
        df = _make_ohlcv(n=10)
        cache.store("BTC", df)

        result = cache.get_cached("BTC", "2025-01-01", "2025-01-10")
        assert result is not None
        assert len(result) == 10
        assert list(result.columns) == ["date", "open", "high", "low", "close", "volume"]

    def test_cache_miss_returns_none(self, tmp_path):
        """Querying a ticker that was never stored returns None."""
        cache = OHLCVCache(cache_dir=tmp_path)
        result = cache.get_cached("XYZ", "2025-01-01", "2025-12-31")
        assert result is None

    def test_date_range_filtering(self, tmp_path):
        """get_cached() respects the start_date / end_date window."""
        cache = OHLCVCache(cache_dir=tmp_path)
        df = _make_ohlcv(n=30)  # 2025-01-01 to 2025-01-30
        cache.store("ETH", df)

        # Request only the first 10 days
        result = cache.get_cached("ETH", "2025-01-01", "2025-01-10")
        assert result is not None
        assert len(result) == 10
        assert result["date"].min() >= pd.Timestamp("2025-01-01")
        assert result["date"].max() <= pd.Timestamp("2025-01-10")

    def test_deduplication_on_store(self, tmp_path):
        """Storing overlapping date ranges does not create duplicates."""
        cache = OHLCVCache(cache_dir=tmp_path)
        df = _make_ohlcv(n=10)  # 2025-01-01 to 2025-01-10
        cache.store("SOL", df)
        # Store again -- same dates should be deduplicated
        cache.store("SOL", df)

        result = cache.get_cached("SOL", "2025-01-01", "2025-01-10")
        assert result is not None
        assert len(result) == 10  # no duplicates

    def test_append_new_data(self, tmp_path):
        """Storing new dates after initial store appends correctly."""
        cache = OHLCVCache(cache_dir=tmp_path)

        # First batch: Jan 1-10
        df1 = _make_ohlcv(n=10)
        cache.store("AVAX", df1)

        # Sleep to ensure the second shard gets a distinct timestamp-based filename
        time.sleep(1.1)

        # Second batch: Jan 11-15 (non-overlapping)
        dates2 = pd.date_range("2025-01-11", periods=5, freq="D")
        rng = np.random.RandomState(99)
        close2 = 100 + np.cumsum(rng.randn(5))
        df2 = pd.DataFrame(
            {
                "date": dates2,
                "open": close2 + rng.randn(5) * 0.5,
                "high": close2 + rng.uniform(0.5, 2.0, 5),
                "low": close2 - rng.uniform(0.5, 2.0, 5),
                "close": close2,
                "volume": rng.uniform(1e6, 5e6, 5),
            }
        )
        cache.store("AVAX", df2)

        result = cache.get_cached("AVAX", "2025-01-01", "2025-01-15")
        assert result is not None
        assert len(result) == 15

    def test_get_last_date(self, tmp_path):
        """get_last_date() returns the most recent cached date."""
        cache = OHLCVCache(cache_dir=tmp_path)
        df = _make_ohlcv(n=10)
        cache.store("LINK", df)

        last = cache.get_last_date("LINK")
        assert last is not None
        assert last == pd.Timestamp("2025-01-10")

    def test_get_last_date_missing_ticker(self, tmp_path):
        """get_last_date() returns None for unknown tickers."""
        cache = OHLCVCache(cache_dir=tmp_path)
        assert cache.get_last_date("NOPE") is None

    def test_is_fresh(self, tmp_path):
        """is_fresh() returns True when last date is within TTL of end_date."""
        cache = OHLCVCache(cache_dir=tmp_path, fresh_ttl_hours=48)
        df = _make_ohlcv(n=10)  # last date: 2025-01-10
        cache.store("DOT", df)

        # end_date within 48 hours of last cached date
        assert cache.is_fresh("DOT", "2025-01-10") is True
        assert cache.is_fresh("DOT", "2025-01-11") is True

        # end_date far beyond TTL
        assert cache.is_fresh("DOT", "2025-03-01") is False

    def test_clear_ticker(self, tmp_path):
        """clear(ticker) removes only that ticker's cache."""
        cache = OHLCVCache(cache_dir=tmp_path)
        df = _make_ohlcv(n=5)
        cache.store("A", df)
        cache.store("B", df)

        cache.clear("A")
        assert cache.get_cached("A", "2025-01-01", "2025-01-05") is None
        assert cache.get_cached("B", "2025-01-01", "2025-01-05") is not None

    def test_clear_all(self, tmp_path):
        """clear() with no args removes all cached data."""
        cache = OHLCVCache(cache_dir=tmp_path)
        df = _make_ohlcv(n=5)
        cache.store("X", df)
        cache.store("Y", df)

        cache.clear()
        assert cache.get_cached("X", "2025-01-01", "2025-01-05") is None
        assert cache.get_cached("Y", "2025-01-01", "2025-01-05") is None


# ===========================================================================
# TestRetry
# ===========================================================================


class TestRetry:
    """Tests for the retry_transient tenacity decorator."""

    def test_retries_on_transient_then_succeeds(self):
        """Function is retried on transient failure and succeeds on later attempt."""
        call_count = 0

        @retry_transient
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "ok"

        result = flaky()
        assert result == "ok"
        assert call_count == 3  # failed twice, succeeded on third

    def test_gives_up_after_max_retries(self):
        """After 4 attempts (stop_after_attempt(4)), the exception is reraised."""
        call_count = 0

        @retry_transient
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("persistent transient error")

        with pytest.raises(ConnectionError, match="persistent transient error"):
            always_fail()

        assert call_count == 4  # 1 initial + 3 retries

    def test_non_transient_not_retried(self):
        """Non-transient exceptions are not retried and propagate immediately."""
        call_count = 0

        @retry_transient
        def bad_value():
            nonlocal call_count
            call_count += 1
            raise ValueError("not transient")

        with pytest.raises(ValueError, match="not transient"):
            bad_value()

        assert call_count == 1  # no retries for non-transient


# ---------------------------------------------------------------------------
# MarketCapProvider — CoinMarketCap source
# ---------------------------------------------------------------------------


_FAKE_CMC_PAYLOAD = {
    "data": [
        {
            "symbol": "btc",
            "cmc_rank": 1,
            "circulating_supply": 19_600_000,
            "quote": {"USD": {"market_cap": 1.2e12, "volume_24h": 4.0e10}},
        },
        {
            "symbol": "eth",
            "cmc_rank": 2,
            "circulating_supply": 120_000_000,
            "quote": {"USD": {"market_cap": 4.0e11, "volume_24h": 2.0e10}},
        },
    ]
}


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class TestMarketCapProviderSource:
    def test_default_source_is_coingecko(self):
        assert MarketCapProvider().source == "coingecko"

    def test_cmc_aliases_normalize(self):
        for alias in ("cmc", "CMC", "coinmarketcap", "CoinMarketCap", "coin_market_cap"):
            assert MarketCapProvider(source=alias).source == "coinmarketcap"

    def test_unknown_source_falls_back_to_coingecko(self):
        assert MarketCapProvider(source="nasdaq").source == "coingecko"

    def test_cache_path_namespaced_by_source(self, tmp_path):
        cg = MarketCapProvider(cache_dir=tmp_path, source="coingecko")
        cmc = MarketCapProvider(cache_dir=tmp_path, source="cmc")
        # Back-compat: CoinGecko keeps the original filename.
        assert cg._cache_path().name == "rankings.parquet"
        # CMC is namespaced so the two never overwrite each other.
        assert cmc._cache_path().name == "rankings_coinmarketcap.parquet"
        assert cg._cache_path() != cmc._cache_path()

    def test_cmc_fetch_maps_schema(self, tmp_path, monkeypatch):
        monkeypatch.setenv("API_KEY_COINMARKETCAP", "test-key")
        captured = {}

        def fake_get(url, headers=None, params=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            return _FakeResp(_FAKE_CMC_PAYLOAD)

        monkeypatch.setattr("quantbox.plugins.datasources._utils.httpx.get", fake_get)

        prov = MarketCapProvider(cache_dir=tmp_path, source="cmc")
        df = prov.fetch_rankings()

        assert "pro-api.coinmarketcap.com" in captured["url"]
        assert captured["headers"]["X-CMC_PRO_API_KEY"] == "test-key"
        assert set(df.columns) == {
            "symbol",
            "market_cap",
            "total_volume",
            "circulating_supply",
            "rank",
            "fetch_timestamp",
        }
        btc = df[df["symbol"] == "BTC"].iloc[0]
        assert btc["market_cap"] == 1.2e12
        assert btc["total_volume"] == 4.0e10
        assert btc["rank"] == 1

    def test_cmc_missing_key_returns_none_gracefully(self, tmp_path, monkeypatch):
        monkeypatch.delenv("API_KEY_COINMARKETCAP", raising=False)
        monkeypatch.delenv("CMC_API_KEY", raising=False)
        prov = MarketCapProvider(cache_dir=tmp_path, source="cmc")
        assert prov.fetch_rankings() is None

    def test_cmc_estimate_market_cap_uses_cmc(self, tmp_path, monkeypatch):
        monkeypatch.setenv("API_KEY_COINMARKETCAP", "test-key")
        monkeypatch.setattr(
            "quantbox.plugins.datasources._utils.httpx.get",
            lambda *a, **k: _FakeResp(_FAKE_CMC_PAYLOAD),
        )
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        prices = pd.DataFrame({"BTC": 60000.0, "ETH": 3000.0}, index=dates)
        mc = MarketCapProvider(cache_dir=tmp_path, source="cmc").estimate_market_cap(prices, pd.DataFrame())
        # BTC mcap > ETH mcap at every row (uses CMC circulating supply).
        assert (mc["BTC"] > mc["ETH"]).all()


class TestMarketCapNeverFabricates:
    """An uncovered ticker (no genuine mcap source) yields NaN and is dropped
    from the mcap-ranked universe screen — it is NOT given a fake ``price*1e9``
    cap (the old L4 default, which corrupted top_by_mcap selection)."""

    def _provider_with_only_btc_eth(self, tmp_path):
        # Rankings cover BTC + ETH only; ZZZ is genuinely uncovered.
        prov = MarketCapProvider(cache_dir=tmp_path, source="cmc")
        rankings = pd.DataFrame(
            {
                "symbol": ["BTC", "ETH"],
                "market_cap": [1.2e12, 4.0e11],
                "total_volume": [4.0e10, 2.0e10],
                "circulating_supply": [2.0e7, 1.2e8],
                "rank": [1, 2],
                "fetch_timestamp": pd.Timestamp.utcnow(),
            }
        )
        prov.fetch_rankings = lambda: rankings  # type: ignore[method-assign]
        return prov

    def test_uncovered_symbol_yields_nan(self, tmp_path):
        prov = self._provider_with_only_btc_eth(tmp_path)
        dates = pd.date_range("2025-01-01", periods=4, freq="D")
        # ZZZ has a HUGE unit price — under the old `price * 1e9` default it
        # would have been fabricated into a fake top-rank mega-cap.
        prices = pd.DataFrame({"BTC": 60000.0, "ETH": 3000.0, "ZZZ": 250000.0}, index=dates)
        mc = prov.estimate_market_cap(prices, pd.DataFrame())

        # Covered coins carry a finite cap; the uncovered coin is all-NaN.
        assert mc["BTC"].notna().all()
        assert mc["ETH"].notna().all()
        assert mc["ZZZ"].isna().all()
        # And specifically NOT the old fabricated price*1e9 value.
        assert not (mc["ZZZ"] == prices["ZZZ"] * 1e9).any()

    def test_uncovered_symbol_excluded_from_ranking(self, tmp_path):
        from quantbox.plugins.strategies._universe import select_universe

        prov = self._provider_with_only_btc_eth(tmp_path)
        dates = pd.date_range("2025-01-01", periods=4, freq="D")
        prices = pd.DataFrame({"BTC": 60000.0, "ETH": 3000.0, "ZZZ": 250000.0}, index=dates)
        # ZZZ has the highest dollar volume — if it leaked into the mcap tier
        # (via the old fabricated cap) it would dominate the screen.
        volume = pd.DataFrame({"BTC": 1000.0, "ETH": 1000.0, "ZZZ": 1_000_000.0}, index=dates)
        mc = prov.estimate_market_cap(prices, pd.DataFrame())

        mask = select_universe(
            prices,
            volume,
            mc,
            top_by_mcap=2,
            top_by_volume=2,
        )
        # ZZZ (NaN mcap) must never be selected; BTC/ETH are.
        assert not mask["ZZZ"].any()
        assert mask["BTC"].any()
        assert mask["ETH"].any()
