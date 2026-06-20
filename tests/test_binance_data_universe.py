"""Tests for BinanceDataFetcher.get_tradable_tickers universe ranking.

Regression for the bug where ``get_tradable_tickers`` returned tickers sorted
ALPHABETICALLY. Combined with ``load_universe``'s ``tickers[:top_n]`` truncation
this silently dropped every symbol past the alphabetical cutoff (e.g. SOL, XRP,
XLM, NEAR) from the candidate universe regardless of liquidity. The fix ranks by
24h quote-volume DESCENDING so the most-liquid pairs survive a top_n cut.
"""

from __future__ import annotations

from quantbox.plugins.datasources.binance_data import BinanceDataFetcher


def _fake_24hr_stats() -> list[dict]:
    """24hr stats where a high-volume symbol is alphabetically LATE and a
    low-volume one is alphabetically EARLY."""
    return [
        {"symbol": "AAAUSDT", "quoteVolume": "5000000"},  # early, low vol
        {"symbol": "ZZZUSDT", "quoteVolume": "9000000000"},  # late, huge vol
        {"symbol": "MIDUSDT", "quoteVolume": "100000000"},  # middle
        {"symbol": "LOWUSDT", "quoteVolume": "500000"},  # below min_volume
        {"symbol": "ETHBTC", "quoteVolume": "8000000000"},  # wrong quote asset
        {"symbol": "USDCUSDT", "quoteVolume": "7000000000"},  # stablecoin base
    ]


def test_tradable_tickers_ranked_by_volume_desc(monkeypatch):
    # Set stablecoins explicitly: DEFAULT_STABLECOINS is empty unless the
    # quantbox-datasets catalog is installed, so don't rely on it here.
    fetcher = BinanceDataFetcher(quote_asset="USDT", stablecoins=["USDC"])

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return _fake_24hr_stats()

    monkeypatch.setattr(
        "quantbox.plugins.datasources.binance_data.httpx.get",
        lambda *a, **k: _Resp(),
    )

    tickers = fetcher.get_tradable_tickers(min_volume_usd=1_000_000)

    # High-volume alphabetically-late symbol must come FIRST.
    assert tickers[0] == "ZZZ"
    # Volume-descending order overall (AAA filtered: 5M >= 1M, so kept last).
    assert tickers == ["ZZZ", "MID", "AAA"]
    # Below-min-volume and wrong-quote and stablecoin filtered out.
    assert "LOW" not in tickers
    assert "ETH" not in tickers
    assert "USDC" not in tickers


def test_top_n_truncation_keeps_high_volume_late_symbol(monkeypatch):
    """The end-to-end failure mode: a top_n cut must KEEP the high-volume
    symbol even though it is alphabetically last."""
    fetcher = BinanceDataFetcher(quote_asset="USDT", stablecoins=["USDC"])

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return _fake_24hr_stats()

    monkeypatch.setattr(
        "quantbox.plugins.datasources.binance_data.httpx.get",
        lambda *a, **k: _Resp(),
    )

    tickers = fetcher.get_tradable_tickers(min_volume_usd=1_000_000)
    top1 = tickers[:1]
    assert top1 == ["ZZZ"], "top_n truncation must keep the most-liquid pair"


# ---------------------------------------------------------------------------
# FIX B: CMC-mcap-ranked candidate universe (quantlab match).
# A high-Binance-VOLUME but low-CMC-mcap new listing must NOT enter the
# candidate set when mcap_source=coinmarketcap — exactly like quantlab, whose
# candidate set is CMC-top-N by genuine market cap (mature coins only).
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402

from quantbox.plugins.datasources.binance_data_plugin import (  # noqa: E402
    BinanceDataPlugin,
)


def _cmc_rankings() -> pd.DataFrame:
    """A genuine CMC market-cap ranking: mature majors only. The new
    high-Binance-volume junk listing PUMP is deliberately ABSENT (low real
    mcap -> not in CMC top-N)."""
    return pd.DataFrame(
        [
            {"symbol": "BTC", "market_cap": 1.3e12, "rank": 1},
            {"symbol": "ETH", "market_cap": 4.0e11, "rank": 2},
            {"symbol": "BNB", "market_cap": 9.0e10, "rank": 3},
            {"symbol": "SOL", "market_cap": 8.0e10, "rank": 4},
            {"symbol": "XRP", "market_cap": 7.0e10, "rank": 5},
            {"symbol": "USDT", "market_cap": 1.2e11, "rank": 6},  # stablecoin
        ]
    )


def test_cmc_path_excludes_high_volume_low_mcap_new_coin(monkeypatch):
    """mcap_source=coinmarketcap: candidate set is CMC-rank-ordered and the
    high-Binance-VOLUME new coin PUMP is NOT in it (quantlab parity)."""
    plugin = BinanceDataPlugin(quote_asset="USDT", mcap_source="coinmarketcap")
    fetcher = plugin._fetcher
    fetcher.stablecoins = ["USDT", "USDC"]

    monkeypatch.setattr(fetcher._cmc_provider, "fetch_rankings", lambda: _cmc_rankings())
    # PUMP IS Binance-tradable with huge volume, so the OLD volume-ranked path
    # would have included it. CMC ranking must drop it.
    monkeypatch.setattr(
        fetcher,
        "get_tradable_tickers",
        lambda min_volume_usd=1e6: ["PUMP", "BTC", "ETH", "BNB", "SOL", "XRP"],
    )

    uni = plugin.load_universe({"top_n": 5})
    syms = uni["symbol"].tolist()

    assert "PUMP" not in syms, "CMC ranking must exclude the low-mcap new listing"
    assert syms == ["BTC", "ETH", "BNB", "SOL", "XRP"]
    assert "USDT" not in syms


def test_cmc_path_not_tradable_removed_before_top_n_cut(monkeypatch):
    """quantlab parity: ``not_tradable`` symbols are removed from the CMC ranking
    BEFORE the iloc[:top_n] cut, so a not_tradable coin frees a slot for the next
    tradable coin. A Binance-USDT-tradable opt-out (e.g. ZEC/XMR analogue, here
    BNB) must NOT leak into the candidate set even though it IS tradable, and the
    coin that was ranked just past top_n (XRP) takes the freed slot."""
    plugin = BinanceDataPlugin(quote_asset="USDT", mcap_source="coinmarketcap")
    fetcher = plugin._fetcher
    fetcher.stablecoins = ["USDT", "USDC"]

    monkeypatch.setattr(fetcher._cmc_provider, "fetch_rankings", lambda: _cmc_rankings())
    # All of BTC/ETH/BNB/SOL/XRP are Binance-tradable: the tradable intersection
    # alone would NOT drop BNB. Only the not_tradable opt-out removes it.
    monkeypatch.setattr(
        fetcher,
        "get_tradable_tickers",
        lambda min_volume_usd=1e6: ["BTC", "ETH", "BNB", "SOL", "XRP"],
    )

    # top_n=4 with BNB opted out: ranking minus stablecoin USDT is
    # [BTC, ETH, BNB, SOL, XRP]; drop BNB BEFORE the cut -> [BTC, ETH, SOL, XRP];
    # first 4 -> all four. If not_tradable were applied AFTER the cut, XRP would
    # be missing (BNB would have consumed slot 3).
    uni = plugin.load_universe({"top_n": 4, "not_tradable": ["BNB"]})
    syms = uni["symbol"].tolist()

    assert "BNB" not in syms, "not_tradable opt-out must be excluded even if Binance-tradable"
    assert syms == ["BTC", "ETH", "SOL", "XRP"], "removal must precede the top_n cut (frees a slot)"


def test_cmc_path_not_tradable_case_insensitive(monkeypatch):
    """not_tradable matching is case-insensitive against the upper-cased ranking."""
    plugin = BinanceDataPlugin(quote_asset="USDT", mcap_source="coinmarketcap")
    fetcher = plugin._fetcher
    fetcher.stablecoins = ["USDT", "USDC"]
    monkeypatch.setattr(fetcher._cmc_provider, "fetch_rankings", lambda: _cmc_rankings())
    monkeypatch.setattr(
        fetcher,
        "get_tradable_tickers",
        lambda min_volume_usd=1e6: ["BTC", "ETH", "BNB", "SOL", "XRP"],
    )
    uni = plugin.load_universe({"top_n": 5, "not_tradable": ["bnb", "Xrp"]})
    assert uni["symbol"].tolist() == ["BTC", "ETH", "SOL"]


def test_cmc_path_falls_back_to_volume_when_ranking_unavailable(monkeypatch):
    """No genuine CMC ranking (no key / API fail) -> fall back to the
    Binance-volume candidate set rather than produce an empty/wrong universe."""
    plugin = BinanceDataPlugin(quote_asset="USDT", mcap_source="coinmarketcap")
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
    """coingecko (shadow) path is unchanged: Binance-volume ordering, and
    fetch_rankings is NOT consulted for candidate selection."""
    plugin = BinanceDataPlugin(quote_asset="USDC", mcap_source="coingecko")
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
