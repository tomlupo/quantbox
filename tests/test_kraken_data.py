"""Tests for the Kraken data fetcher + plugin.

Mirrors ``test_binance_data_universe.py``. ccxt is mocked via an injected fake
exchange — these tests NEVER hit live Kraken.
"""

from __future__ import annotations

import pandas as pd

from quantbox.plugins.datasources.kraken_data import (
    KRAKEN_OHLC_MAX_CANDLES,
    KrakenDataFetcher,
    normalize_kraken_asset,
)
from quantbox.plugins.datasources.kraken_data_plugin import KrakenDataPlugin

# ---------------------------------------------------------------------------
# Asset-code normalisation
# ---------------------------------------------------------------------------


def test_normalize_legacy_codes():
    assert normalize_kraken_asset("XXBT") == "BTC"
    assert normalize_kraken_asset("ZUSD") == "USD"
    assert normalize_kraken_asset("XETH") == "ETH"
    assert normalize_kraken_asset("XXDG") == "DOGE"
    # Already-canonical codes are unchanged.
    assert normalize_kraken_asset("BTC") == "BTC"
    assert normalize_kraken_asset("USDC") == "USDC"
    assert normalize_kraken_asset("SOL") == "SOL"


def test_normalize_strips_earn_suffixes():
    assert normalize_kraken_asset("DOT.S") == "DOT"
    assert normalize_kraken_asset("ETH.F") == "ETH"
    assert normalize_kraken_asset("XXBT.S") == "BTC"


# ---------------------------------------------------------------------------
# Fake ccxt exchange
# ---------------------------------------------------------------------------


class _FakeExchange:
    def __init__(self, tickers):
        self._tickers = tickers

    def fetch_tickers(self):
        return self._tickers


def _tickers() -> dict:
    """A high-volume but alphabetically-late symbol and a wrong-quote pair."""
    return {
        "BTC/USD": {"quoteVolume": 9_000_000_000, "last": 60000},
        "MID/USD": {"quoteVolume": 100_000_000, "last": 10},
        "AAA/USD": {"quoteVolume": 5_000_000, "last": 1},
        "LOW/USD": {"quoteVolume": 500_000, "last": 1},  # below min volume
        "ETH/EUR": {"quoteVolume": 8_000_000_000, "last": 3000},  # wrong quote
        "USDC/USD": {"quoteVolume": 7_000_000_000, "last": 1},  # stablecoin base
        # quoteVolume omitted -> derived from baseVolume * last
        "SOL/USD": {"baseVolume": 1_000_000, "last": 150},
    }


def test_tradable_tickers_ranked_by_volume_desc(monkeypatch):
    # get_tradable_tickers short-circuits on the module-level CCXT_AVAILABLE
    # guard; force it True so the injected fake exchange is used even when the
    # ccxt extra isn't installed (the CI no-extra env).
    monkeypatch.setattr("quantbox.plugins.datasources.kraken_data.CCXT_AVAILABLE", True)
    # Inject the fake exchange at construction so __post_init__ doesn't try to
    # build a real ccxt.kraken (ccxt is absent in the CI no-extra env).
    fetcher = KrakenDataFetcher(quote_asset="USD", stablecoins=["USDC"], _exchange=_FakeExchange(_tickers()))

    tickers = fetcher.get_tradable_tickers(min_volume_usd=1_000_000)

    # BTC (9e9) first; SOL derived = 1e6*150 = 1.5e8 ranks above MID (1e8).
    assert tickers[0] == "BTC"
    assert tickers == ["BTC", "SOL", "MID", "AAA"]
    assert "LOW" not in tickers  # below min volume
    assert "ETH" not in tickers  # wrong quote (EUR)
    assert "USDC" not in tickers  # stablecoin base


def test_quote_asset_filter_usdc(monkeypatch):
    """quote_asset=USDC only keeps */USDC books."""
    monkeypatch.setattr("quantbox.plugins.datasources.kraken_data.CCXT_AVAILABLE", True)
    fetcher = KrakenDataFetcher(
        quote_asset="USDC",
        stablecoins=["USDC", "USDT"],
        _exchange=_FakeExchange(
            {
                "BTC/USDC": {"quoteVolume": 5_000_000, "last": 60000},
                "ETH/USD": {"quoteVolume": 9_000_000_000, "last": 3000},
            }
        ),
    )
    assert fetcher.get_tradable_tickers(min_volume_usd=1_000_000) == ["BTC"]


# ---------------------------------------------------------------------------
# CMC-mcap-ranked candidate path (quantlab parity, mirrors binance test)
# ---------------------------------------------------------------------------


def _cmc_rankings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "BTC", "market_cap": 1.3e12, "rank": 1},
            {"symbol": "ETH", "market_cap": 4.0e11, "rank": 2},
            {"symbol": "BNB", "market_cap": 9.0e10, "rank": 3},
            {"symbol": "SOL", "market_cap": 8.0e10, "rank": 4},
            {"symbol": "XRP", "market_cap": 7.0e10, "rank": 5},
            {"symbol": "USDT", "market_cap": 1.2e11, "rank": 6},
        ]
    )


def test_cmc_path_excludes_high_volume_low_mcap_new_coin(monkeypatch):
    plugin = KrakenDataPlugin(quote_asset="USD", mcap_source="coinmarketcap")
    fetcher = plugin._fetcher
    fetcher.stablecoins = ["USDT", "USDC"]
    monkeypatch.setattr(fetcher._cmc_provider, "fetch_rankings", lambda: _cmc_rankings())
    monkeypatch.setattr(
        fetcher,
        "get_tradable_tickers",
        lambda min_volume_usd=1e6: ["PUMP", "BTC", "ETH", "BNB", "SOL", "XRP"],
    )
    uni = plugin.load_universe({"top_n": 5})
    syms = uni["symbol"].tolist()
    assert "PUMP" not in syms
    assert syms == ["BTC", "ETH", "BNB", "SOL", "XRP"]


def test_cmc_path_not_tradable_removed_before_top_n_cut(monkeypatch):
    plugin = KrakenDataPlugin(quote_asset="USD", mcap_source="coinmarketcap")
    fetcher = plugin._fetcher
    fetcher.stablecoins = ["USDT", "USDC"]
    monkeypatch.setattr(fetcher._cmc_provider, "fetch_rankings", lambda: _cmc_rankings())
    monkeypatch.setattr(
        fetcher,
        "get_tradable_tickers",
        lambda min_volume_usd=1e6: ["BTC", "ETH", "BNB", "SOL", "XRP"],
    )
    uni = plugin.load_universe({"top_n": 4, "not_tradable": ["BNB"]})
    assert uni["symbol"].tolist() == ["BTC", "ETH", "SOL", "XRP"]


def test_cmc_path_falls_back_to_volume_when_ranking_unavailable(monkeypatch):
    plugin = KrakenDataPlugin(quote_asset="USD", mcap_source="coinmarketcap")
    fetcher = plugin._fetcher
    fetcher.stablecoins = ["USDT", "USDC"]
    monkeypatch.setattr(fetcher._cmc_provider, "fetch_rankings", lambda: None)
    monkeypatch.setattr(
        fetcher,
        "get_tradable_tickers",
        lambda min_volume_usd=1e6: ["PUMP", "BTC", "ETH"],
    )
    uni = plugin.load_universe({"top_n": 2})
    assert uni["symbol"].tolist() == ["PUMP", "BTC"]


def test_coingecko_path_keeps_volume_ordering(monkeypatch):
    plugin = KrakenDataPlugin(quote_asset="USD", mcap_source="coingecko")
    fetcher = plugin._fetcher
    fetcher.stablecoins = ["USDT", "USDC"]

    def _boom():
        raise AssertionError("coingecko path must not call fetch_rankings")

    monkeypatch.setattr(fetcher._cmc_provider, "fetch_rankings", _boom)
    monkeypatch.setattr(
        fetcher,
        "get_tradable_tickers",
        lambda min_volume_usd=1e6: ["ZZZ", "MID", "AAA"],
    )
    uni = plugin.load_universe({"top_n": 2})
    assert uni["symbol"].tolist() == ["ZZZ", "MID"]


# ---------------------------------------------------------------------------
# load_universe explicit symbols + empty market data
# ---------------------------------------------------------------------------


def test_load_universe_explicit_symbols():
    plugin = KrakenDataPlugin(quote_asset="USD")
    uni = plugin.load_universe({"symbols": ["BTC", "ETH"]})
    assert uni["symbol"].tolist() == ["BTC", "ETH"]


def test_load_market_data_empty_universe_returns_empty_frames():
    plugin = KrakenDataPlugin(quote_asset="USD")
    out = plugin.load_market_data(pd.DataFrame(columns=["symbol"]), "2026-06-01", {})
    assert set(out) == {"prices", "volume", "market_cap", "screen_volume"}
    assert all(df.empty for df in out.values())


# ---------------------------------------------------------------------------
# OHLC 720-candle cap warning
# ---------------------------------------------------------------------------


class _OHLCExchange:
    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        # one daily candle so the fetch loop terminates immediately
        return [[since, 1.0, 2.0, 0.5, 1.5, 100.0]]


def test_ohlc_cap_warning_logged(caplog):
    fetcher = KrakenDataFetcher(quote_asset="USD")
    fetcher._exchange = _OHLCExchange()
    # request 3 years daily -> well over the 720-candle cap
    with caplog.at_level("WARNING"):
        fetcher._fetch_ohlcv_with_retry("BTC/USD", "2023-01-01", "2026-06-01", "1d")
    assert any(str(KRAKEN_OHLC_MAX_CANDLES) in r.message for r in caplog.records)
