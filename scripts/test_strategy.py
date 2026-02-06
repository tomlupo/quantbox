#!/usr/bin/env python3
"""
Test ported CryptoTrendStrategy with live data.

Usage:
    cd ~/workspace/quantbox
    PYTHONPATH="packages/quantbox-core/src" .venv/bin/python scripts/test_strategy.py
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "quantbox-core" / "src"))

from quantbox.plugins.datasources import BinanceDataFetcher
from quantbox.plugins.strategies import CryptoTrendStrategy
from quantbox.plugins.broker import BinanceLiveBroker, UniverseSelector, UniverseConfig


def main():
    print("=" * 60)
    print("QUANTBOX STRATEGY TEST - Live Data")
    print("=" * 60)
    
    # 1. Fetch live data
    print("\n1. Fetching live market data...")
    t0 = time.time()
    fetcher = BinanceDataFetcher()
    
    # Get tradable tickers
    all_tickers = fetcher.get_tradable_tickers(min_volume_usd=50e6)[:20]
    print(f"   Tickers: {all_tickers}")
    
    data = fetcher.get_market_data(all_tickers, lookback_days=400)
    print(f"   Fetched: {data['prices'].shape} in {time.time()-t0:.1f}s")
    
    # 2. Run strategy
    print("\n2. Running CryptoTrendStrategy...")
    t0 = time.time()
    strategy = CryptoTrendStrategy(
        top_by_mcap=20,
        top_by_volume=8,
        use_duckdb=False,  # Use vectorized pandas
        use_trailing_stop=True,
    )
    
    result = strategy.run(data)
    print(f"   Strategy completed in {time.time()-t0:.1f}s")
    
    # 3. Show results
    print("\n3. Strategy Results:")
    print(f"   Weights shape: {result['weights'].shape}")
    
    simple_weights = result['simple_weights']
    print(f"\n   Latest weights (vol=50%, tranche=5):")
    for ticker, weight in sorted(simple_weights.items(), key=lambda x: -x[1]):
        print(f"     {ticker}: {weight:.2%}")
    
    # 4. Describe strategy
    print("\n4. Strategy Description:")
    desc = strategy.describe()
    print(f"   Name: {desc['name']}")
    print(f"   Type: {desc['type']}")
    print(f"   Signals: {desc['signals']}")
    print(f"   Features: DuckDB={desc['features']['duckdb']}, QuantStats={desc['features']['quantstats']}")
    
    # 5. Try backtest if quantstats available
    print("\n5. Backtest (if quantstats available)...")
    try:
        bt = strategy.backtest(data)
        if bt['metrics']:
            print(f"   Total Return: {bt['metrics'].get('total_return', 0):.2%}")
            print(f"   CAGR: {bt['metrics'].get('cagr', 0):.2%}")
            print(f"   Sharpe: {bt['metrics'].get('sharpe', 0):.2f}")
            print(f"   Max Drawdown: {bt['metrics'].get('max_drawdown', 0):.2%}")
        else:
            print("   (quantstats not installed - metrics not available)")
    except Exception as e:
        print(f"   Backtest error: {e}")
    
    # 6. Integrate with broker
    print("\n6. Broker Integration...")
    broker = BinanceLiveBroker(
        paper_trading=True,
        account_name="strategy_test",
    )
    
    # Load live prices
    prices = fetcher.get_current_prices(list(simple_weights.keys()))
    for ticker, price in prices.items():
        broker._price_cache.set(ticker, price)
    
    # Scale weights (keep some cash)
    target = {k: v * 0.85 for k, v in simple_weights.items()}
    
    analysis = broker.generate_rebalancing(target)
    if not analysis.empty:
        print(f"\n   Rebalancing for {len(target)} positions:")
        display_cols = ['Asset', 'Target_Weight', 'Action', 'Delta_Qty']
        print(analysis[display_cols].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("✅ STRATEGY TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
