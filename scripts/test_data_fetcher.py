#!/usr/bin/env python3
"""
Test script for BinanceDataFetcher - validates live data fetching.

Usage:
    cd ~/workspace/quantbox
    PYTHONPATH="packages/quantbox-core/src" .venv/bin/python scripts/test_data_fetcher.py
"""

import sys
from pathlib import Path

# Add quantbox to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "quantbox-core" / "src"))

from quantbox.plugins.datasources import BinanceDataFetcher


def test_current_prices():
    """Test getting current prices."""
    print("=" * 60)
    print("Test: Current Prices")
    print("=" * 60)
    
    fetcher = BinanceDataFetcher()
    tickers = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']
    
    prices = fetcher.get_current_prices(tickers)
    
    print(f"\nCurrent prices for {tickers}:")
    for ticker, price in prices.items():
        print(f"  {ticker}: ${price:,.2f}")
    
    assert len(prices) == len(tickers), f"Expected {len(tickers)} prices, got {len(prices)}"
    assert all(p > 0 for p in prices.values()), "All prices should be positive"
    print("\n✅ Current prices test passed!")
    
    return prices


def test_market_snapshot():
    """Test getting market snapshot."""
    print("\n" + "=" * 60)
    print("Test: Market Snapshot")
    print("=" * 60)
    
    fetcher = BinanceDataFetcher()
    tickers = ['BTC', 'ETH', 'SOL', 'DOGE', 'ADA']
    
    snapshot = fetcher.get_snapshot(tickers)
    
    print(f"\nMarket snapshot at {snapshot.timestamp}:")
    print(snapshot.data.to_string(index=False))
    
    assert len(snapshot.data) == len(tickers), f"Expected {len(tickers)} rows"
    assert 'price' in snapshot.data.columns
    assert 'volume_24h' in snapshot.data.columns
    print("\n✅ Market snapshot test passed!")
    
    return snapshot


def test_valid_pairs():
    """Test getting valid trading pairs."""
    print("\n" + "=" * 60)
    print("Test: Valid Trading Pairs")
    print("=" * 60)
    
    fetcher = BinanceDataFetcher()
    tickers = ['BTC', 'ETH', 'INVALID_TICKER', 'SOL']
    
    pairs = fetcher.get_valid_pairs(tickers)
    
    print(f"\nTrading pairs:")
    for ticker, pair in pairs.items():
        status = "✓" if pair else "✗"
        print(f"  {ticker} → {pair or 'NOT FOUND'} {status}")
    
    assert pairs['BTC'] == 'BTCUSDT'
    assert pairs['ETH'] == 'ETHUSDT'
    assert pairs['INVALID_TICKER'] is None
    print("\n✅ Valid pairs test passed!")


def test_ohlcv_fetch():
    """Test OHLCV data fetching."""
    print("\n" + "=" * 60)
    print("Test: OHLCV Data Fetch")
    print("=" * 60)
    
    fetcher = BinanceDataFetcher()
    
    # Fetch 7 days of BTC data
    df = fetcher.get_ohlcv('BTC', '2025-01-01', '2025-01-07')
    
    if df is not None:
        print(f"\nBTC OHLCV data (7 days):")
        print(df.head(7).to_string(index=False))
        
        assert len(df) >= 5, f"Expected at least 5 days, got {len(df)}"
        assert 'close' in df.columns
        assert 'volume' in df.columns
        print("\n✅ OHLCV fetch test passed!")
    else:
        print("\n⚠️ OHLCV fetch returned None (ccxt might not be installed)")
    
    return df


def test_market_data():
    """Test full market data fetch for strategies."""
    print("\n" + "=" * 60)
    print("Test: Full Market Data (for strategies)")
    print("=" * 60)
    
    fetcher = BinanceDataFetcher()
    tickers = ['BTC', 'ETH', 'SOL']
    
    # Fetch 30 days of data (quick test)
    data = fetcher.get_market_data(tickers, lookback_days=30)
    
    print(f"\nMarket data shape:")
    print(f"  prices: {data['prices'].shape}")
    print(f"  volume: {data['volume'].shape}")
    print(f"  market_cap: {data['market_cap'].shape}")
    
    print(f"\nLatest prices:")
    print(data['prices'].tail(3))
    
    if not data['prices'].empty:
        assert len(data['prices'].columns) == len(tickers)
        print("\n✅ Market data test passed!")
    else:
        print("\n⚠️ Market data returned empty")
    
    return data


def test_tradable_tickers():
    """Test getting tradable tickers by volume."""
    print("\n" + "=" * 60)
    print("Test: Tradable Tickers (by volume)")
    print("=" * 60)
    
    fetcher = BinanceDataFetcher()
    
    # Get tickers with >$10M daily volume
    tickers = fetcher.get_tradable_tickers(min_volume_usd=10e6)
    
    print(f"\nTickers with >$10M daily volume: {len(tickers)}")
    print(f"Top 20: {tickers[:20]}")
    
    assert 'BTC' in tickers, "BTC should be in tradable tickers"
    assert 'ETH' in tickers, "ETH should be in tradable tickers"
    print("\n✅ Tradable tickers test passed!")
    
    return tickers


def test_describe():
    """Test LLM-friendly describe method."""
    print("\n" + "=" * 60)
    print("Test: Describe (LLM introspection)")
    print("=" * 60)
    
    fetcher = BinanceDataFetcher()
    desc = fetcher.describe()
    
    print(f"\nPurpose: {desc['purpose']}")
    print(f"API key required: {desc['api_key_required']}")
    print(f"CCXT available: {desc['ccxt_available']}")
    print(f"\nCapabilities:")
    for cap, detail in desc['capabilities'].items():
        print(f"  - {cap}: {detail}")
    
    print("\n✅ Describe test passed!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BinanceDataFetcher Test Suite")
    print("=" * 60)
    
    # Run tests
    test_describe()
    test_valid_pairs()
    prices = test_current_prices()
    snapshot = test_market_snapshot()
    test_ohlcv_fetch()
    test_market_data()
    test_tradable_tickers()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
