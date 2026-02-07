#!/usr/bin/env python3
"""
Test script for UniverseSelector - validates universe selection logic.

Usage:
    cd ~/workspace/quantbox
    PYTHONPATH="packages/quantbox-core/src" .venv/bin/python scripts/test_universe_selection.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quantbox.plugins.broker import UniverseSelector, UniverseConfig


def generate_mock_market_data(
    tickers: list[str],
    days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate mock market data for testing universe selection.
    
    Creates realistic-ish price, volume, and market cap data where:
    - BTC has highest market cap
    - Stablecoins have low volatility but high volume
    - Smaller altcoins have higher volatility
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Base prices (approximate real values)
    base_prices = {
        'BTC': 95000, 'ETH': 3200, 'BNB': 650, 'SOL': 190, 'XRP': 2.5,
        'DOGE': 0.35, 'ADA': 1.0, 'AVAX': 35, 'LINK': 22, 'DOT': 7,
        'MATIC': 0.5, 'SHIB': 0.00002, 'LTC': 120, 'TRX': 0.25, 'ATOM': 8,
        'UNI': 14, 'APT': 9, 'NEAR': 5, 'INJ': 25, 'FIL': 5,
        # Stablecoins (flat price)
        'USDT': 1.0, 'USDC': 1.0, 'BUSD': 1.0, 'DAI': 1.0, 'FDUSD': 1.0,
        # Low-cap coins
        'ETHW': 3.5, 'BETH': 3200,
    }
    
    # Market cap tiers (in billions USD)
    market_cap_billions = {
        'BTC': 1900, 'ETH': 385, 'USDT': 140, 'BNB': 95, 'SOL': 90,
        'USDC': 50, 'XRP': 130, 'DOGE': 50, 'ADA': 35, 'AVAX': 14,
        'LINK': 14, 'DOT': 10, 'MATIC': 5, 'SHIB': 15, 'LTC': 9,
        'TRX': 22, 'ATOM': 3, 'UNI': 10, 'APT': 5, 'NEAR': 6,
        'INJ': 2.5, 'FIL': 3, 'BUSD': 1, 'DAI': 5, 'FDUSD': 3,
        'ETHW': 0.5, 'BETH': 0.1,
    }
    
    # Volume tiers (daily, in billions USD)
    volume_billions = {
        'BTC': 35, 'ETH': 18, 'USDT': 80, 'BNB': 2, 'SOL': 5,
        'USDC': 10, 'XRP': 3, 'DOGE': 2, 'ADA': 0.8, 'AVAX': 0.5,
        'LINK': 0.6, 'DOT': 0.3, 'MATIC': 0.4, 'SHIB': 0.5, 'LTC': 0.4,
        'TRX': 0.5, 'ATOM': 0.2, 'UNI': 0.3, 'APT': 0.2, 'NEAR': 0.3,
        'INJ': 0.15, 'FIL': 0.2, 'BUSD': 0.5, 'DAI': 0.2, 'FDUSD': 1,
        'ETHW': 0.01, 'BETH': 0.001,
    }
    
    prices = {}
    volumes = {}
    market_caps = {}
    
    for ticker in tickers:
        base = base_prices.get(ticker, 10.0)
        mc_base = market_cap_billions.get(ticker, 1) * 1e9
        vol_base = volume_billions.get(ticker, 0.1) * 1e9
        
        # Volatility based on ticker type
        if ticker in ['USDT', 'USDC', 'BUSD', 'DAI', 'FDUSD']:
            volatility = 0.0001  # Stablecoins
        elif ticker in ['BTC', 'ETH']:
            volatility = 0.02   # Large caps
        else:
            volatility = 0.04   # Altcoins
        
        # Generate random walk for price
        returns = np.random.normal(0, volatility, days)
        price_path = base * np.cumprod(1 + returns)
        prices[ticker] = price_path
        
        # Market cap follows price with some noise
        mc_noise = np.random.normal(1, 0.05, days)
        market_caps[ticker] = mc_base * (price_path / base) * mc_noise
        
        # Volume is noisier
        vol_noise = np.random.lognormal(0, 0.3, days)
        volumes[ticker] = vol_base / base * vol_noise  # Volume in units
    
    prices_df = pd.DataFrame(prices, index=dates)
    volume_df = pd.DataFrame(volumes, index=dates)
    market_cap_df = pd.DataFrame(market_caps, index=dates)
    
    return prices_df, volume_df, market_cap_df


def test_universe_selector():
    """Test UniverseSelector with mock data."""
    print("=" * 60)
    print("UniverseSelector Test")
    print("=" * 60)
    
    # Define test tickers
    tickers = [
        'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'DOT',
        'MATIC', 'SHIB', 'LTC', 'TRX', 'ATOM', 'UNI', 'APT', 'NEAR', 'INJ', 'FIL',
        'USDT', 'USDC', 'BUSD', 'DAI', 'FDUSD',  # Stablecoins
        'ETHW', 'BETH',  # Exclusions
    ]
    
    # Generate mock data
    print("\n1. Generating mock market data...")
    prices, volume, market_cap = generate_mock_market_data(tickers, days=90)
    print(f"   - {len(tickers)} tickers")
    print(f"   - {len(prices)} days of data")
    print(f"   - Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # Create selector with default config
    print("\n2. Creating UniverseSelector...")
    config = UniverseConfig(
        market_cap_top_n=30,
        portfolio_max_coins=10,
        exclude_stablecoins=True,
        additional_exclusions=['ETHW', 'BETH'],
    )
    selector = UniverseSelector(config=config)
    
    # Test describe()
    print("\n3. Selector description:")
    desc = selector.describe()
    print(f"   Purpose: {desc['purpose']}")
    print(f"   Filters:")
    for key, val in desc['filters'].items():
        print(f"     - {key}: {val}")
    
    # Test select()
    print("\n4. Selecting universe...")
    universe = selector.select(prices, volume, market_cap)
    print(f"   Universe shape: {universe.shape}")
    print(f"   Average coins in universe: {universe.sum(axis=1).mean():.1f}")
    
    # Test get_current_universe()
    print("\n5. Current universe (latest day):")
    current = selector.get_current_universe(prices, volume, market_cap)
    print(f"   Coins ({len(current)}): {current}")
    
    # Test get_universe_stats()
    print("\n6. Universe statistics:")
    stats = selector.get_universe_stats(prices, volume, market_cap)
    print(f"   Total tickers: {stats['total_tickers']}")
    print(f"   Valid (after exclusions): {stats['valid_tickers']}")
    print(f"   Excluded: {stats['excluded_tickers']}")
    print(f"   Current universe: {stats['current_universe']}")
    
    # Test apply_mask()
    print("\n7. Testing weight masking...")
    # Create dummy weights (equal weight for all)
    raw_weights = pd.DataFrame(
        {t: 1.0 / len(tickers) for t in tickers},
        index=[prices.index[-1]]
    )
    masked = selector.apply_mask(raw_weights, universe.iloc[[-1]], normalize=True)
    print(f"   Raw weights sum: {raw_weights.sum(axis=1).iloc[0]:.4f}")
    print(f"   Masked weights sum: {masked.sum(axis=1).iloc[0]:.4f}")
    print(f"   Non-zero weights: {(masked.iloc[0] > 0).sum()}")
    
    # Verify stablecoins excluded
    print("\n8. Verifying exclusions...")
    stablecoins_in_universe = [t for t in ['USDT', 'USDC', 'BUSD', 'DAI', 'FDUSD'] 
                               if t in current]
    print(f"   Stablecoins in universe: {stablecoins_in_universe if stablecoins_in_universe else 'None ✓'}")
    
    excluded_in_universe = [t for t in ['ETHW', 'BETH'] if t in current]
    print(f"   Excluded tokens in universe: {excluded_in_universe if excluded_in_universe else 'None ✓'}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    
    return selector, universe, current


def test_custom_config():
    """Test with custom configuration."""
    print("\n" + "=" * 60)
    print("Custom Config Test")
    print("=" * 60)
    
    # Smaller universe for concentrated portfolio
    config = UniverseConfig(
        market_cap_top_n=15,      # Only top 15 by market cap
        portfolio_max_coins=5,     # Only trade top 5 by volume
        exclude_stablecoins=True,
        min_market_cap_usd=10e9,  # Minimum $10B market cap
    )
    
    selector = UniverseSelector(config=config)
    print(f"\nConfig: {config.to_dict()}")
    
    tickers = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 
               'USDT', 'USDC', 'ETHW']
    prices, volume, market_cap = generate_mock_market_data(tickers, days=30)
    
    current = selector.get_current_universe(prices, volume, market_cap)
    print(f"Current universe ({len(current)}): {current}")
    
    stats = selector.get_universe_stats(prices, volume, market_cap)
    print(f"Universe stats: {stats}")


if __name__ == "__main__":
    # Run tests
    selector, universe, current = test_universe_selector()
    test_custom_config()
