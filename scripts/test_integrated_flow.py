#!/usr/bin/env python3
"""
Test integrated flow: Data → Universe → Broker

This tests the full quantbox pipeline with live Binance data.

Usage:
    cd ~/workspace/quantbox
    PYTHONPATH="packages/quantbox-core/src" .venv/bin/python scripts/test_integrated_flow.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "quantbox-core" / "src"))

from quantbox.plugins.datasources import BinanceDataFetcher
from quantbox.plugins.broker import BinanceLiveBroker, UniverseSelector, UniverseConfig


def main():
    print("=" * 60)
    print("QUANTBOX INTEGRATED FLOW TEST")
    print("=" * 60)
    
    # 1. Fetch live data
    print("\n1. Fetching live market data...")
    fetcher = BinanceDataFetcher()
    
    # Get top tickers by volume
    all_tickers = fetcher.get_tradable_tickers(min_volume_usd=20e6)
    print(f"   Found {len(all_tickers)} tradable tickers")
    
    # Use top 15 for the test
    tickers = all_tickers[:15]
    print(f"   Using: {tickers}")
    
    # Fetch 90 days of data
    data = fetcher.get_market_data(tickers, lookback_days=90)
    print(f"   Data shape: prices={data['prices'].shape}, volume={data['volume'].shape}")
    
    # 2. Universe selection
    print("\n2. Selecting tradable universe...")
    config = UniverseConfig(
        market_cap_top_n=20,
        portfolio_max_coins=8,
        exclude_stablecoins=True,
    )
    selector = UniverseSelector(config=config)
    
    current_universe = selector.get_current_universe(
        data['prices'], 
        data['volume'], 
        data['market_cap']
    )
    print(f"   Current universe ({len(current_universe)}): {current_universe}")
    
    # 3. Generate sample weights (equal weight for universe)
    print("\n3. Generating target weights...")
    n = len(current_universe)
    target_weights = {ticker: 0.85 / n for ticker in current_universe}  # 85% invested, 15% cash
    
    print("   Target weights:")
    for ticker, weight in sorted(target_weights.items(), key=lambda x: -x[1]):
        print(f"     {ticker}: {weight:.2%}")
    
    # 4. Initialize broker with live prices
    print("\n4. Initializing broker...")
    broker = BinanceLiveBroker(
        paper_trading=True,
        account_name="integrated_test",
        stable_coin="USDC",
    )
    
    # Load live prices
    prices = fetcher.get_current_prices(list(target_weights.keys()))
    for ticker, price in prices.items():
        broker._price_cache.set(ticker, price)
    print(f"   Loaded {len(prices)} live prices")
    
    # 5. Generate rebalancing analysis
    print("\n5. Rebalancing analysis...")
    state = broker.describe()
    print(f"   Portfolio value: {state['portfolio_value']:,.2f} {state['stable_coin']}")
    
    analysis = broker.generate_rebalancing(target_weights)
    
    if not analysis.empty:
        print("\n   Proposed trades:")
        display_cols = ['Asset', 'Current_Weight', 'Target_Weight', 'Action', 'Delta_Qty', 'Notional']
        cols_present = [c for c in display_cols if c in analysis.columns]
        print(analysis[cols_present].to_string(index=False))
        
        buys = analysis[analysis['Action'] == 'Buy']
        sells = analysis[analysis['Action'] == 'Sell']
        print(f"\n   Summary: {len(buys)} buys, {len(sells)} sells")
    
    # 6. Execute in paper mode
    print("\n6. Executing paper trades...")
    result = broker.execute_rebalancing(target_weights)
    
    print(f"   Executed: {result['summary']['total_executed']} orders")
    print(f"   Failed: {result['summary']['total_failed']} orders")
    print(f"   Total value traded: ${result['summary']['total_value']:,.2f}")
    
    # 7. Check final state
    print("\n7. Final portfolio state...")
    final_state = broker.describe()
    print(f"   Portfolio value: {final_state['portfolio_value']:,.2f} {final_state['stable_coin']}")
    print(f"   Positions: {final_state['positions']}")
    print(f"   Cash: {final_state['cash']}")
    
    print("\n" + "=" * 60)
    print("✅ INTEGRATED FLOW TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
