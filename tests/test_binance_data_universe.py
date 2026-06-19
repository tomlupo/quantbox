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
