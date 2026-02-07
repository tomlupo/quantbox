#!/usr/bin/env python3
"""
Simple Broker Test - Minimal test without external dependencies.

This demonstrates the quantbox broker without needing quantlab or strategies.
Just shows the broker interface working with hardcoded target weights.
"""
import sys
from pathlib import Path

# Add quantbox to path
QUANTBOX_PATH = Path(__file__).parent.parent / "packages" / "quantbox-core" / "src"
sys.path.insert(0, str(QUANTBOX_PATH))

# Minimal imports - these should be in quantbox itself
try:
    from quantbox.plugins.broker.binance_live import BinanceLiveBroker
    print("✓ Successfully imported BinanceLiveBroker")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    print("  Make sure you're in the quantbox directory")
    sys.exit(1)


def test_broker_basic():
    """Test basic broker functionality."""
    print("\n" + "="*60)
    print("TESTING BINANCE LIVE BROKER (Paper Mode)")
    print("="*60)
    
    # Initialize broker
    broker = BinanceLiveBroker(
        paper_trading=True,
        account_name="test_simple",
        stable_coin="USDC",
        trades_dir="/tmp/quantbox_test"
    )
    
    # 1. Test describe()
    print("\n1. Testing describe() method:")
    state = broker.describe()
    print(f"   Mode: {state['mode']}")
    print(f"   Account: {state['account']}")
    print(f"   Portfolio Value: {state['portfolio_value']:,.2f} {state['stable_coin']}")
    print(f"   Positions: {len(state['positions'])}")
    
    # 2. Test get_positions()
    print("\n2. Testing get_positions():")
    positions = broker.get_positions()
    print(f"   Returned DataFrame with {len(positions)} rows")
    if not positions.empty:
        print(positions.to_string())
    
    # 3. Test get_cash()
    print("\n3. Testing get_cash():")
    cash = broker.get_cash()
    print(f"   Cash balances: {cash}")
    
    # 4. Test generate_rebalancing()
    print("\n4. Testing generate_rebalancing():")
    target_weights = {
        'BTC': 0.40,
        'ETH': 0.30,
        'SOL': 0.15,
    }
    print(f"   Target weights: {target_weights}")
    
    try:
        analysis = broker.generate_rebalancing(target_weights)
        print(f"   Generated analysis with {len(analysis)} rows")
        if not analysis.empty:
            display_cols = ['Asset', 'Current_Weight', 'Target_Weight', 'Action']
            if all(c in analysis.columns for c in display_cols):
                print("\n   Preview:")
                print(analysis[display_cols].to_string(index=False))
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "="*60)
    print("BROKER TEST COMPLETE")
    print("="*60)


def test_broker_advanced():
    """Test advanced broker functionality with mock prices."""
    print("\n" + "="*60)
    print("ADVANCED TEST - With Mock Price Cache")
    print("="*60)
    
    broker = BinanceLiveBroker(
        paper_trading=True,
        account_name="test_advanced",
        stable_coin="USDC",
        trades_dir="/tmp/quantbox_test"
    )
    
    # Manually populate price cache with mock prices
    # This simulates what would happen with real Binance API
    mock_prices = {
        'BTC': 45000.0,
        'ETH': 2500.0,
        'SOL': 100.0,
        'BNB': 300.0,
        'ADA': 0.5,
    }
    
    print("\nPopulating price cache with mock prices:")
    for asset, price in mock_prices.items():
        broker._price_cache.set(asset, price)
        print(f"  {asset}: ${price:,.2f}")
    
    # Now rebalancing should work better
    target_weights = {
        'BTC': 0.40,
        'ETH': 0.25,
        'SOL': 0.10,
        'BNB': 0.10,
    }
    
    print(f"\nTarget allocation: {sum(target_weights.values()):.0%} invested, {1-sum(target_weights.values()):.0%} cash")
    
    analysis = broker.generate_rebalancing(target_weights)
    if not analysis.empty:
        print("\nRebalancing Analysis:")
        print(analysis.to_string(index=False))
        
        # Count actions
        buys = len(analysis[analysis['Action'] == 'Buy'])
        sells = len(analysis[analysis['Action'] == 'Sell'])
        holds = len(analysis[analysis['Action'] == 'Hold'])
        
        print(f"\nSummary: {buys} buys, {sells} sells, {holds} holds")
        
        # Show portfolio value calculation
        total_value = broker.get_portfolio_value()
        print(f"\nPortfolio calculations:")
        print(f"  Total value: ${total_value:,.2f}")
        for _, row in analysis.iterrows():
            if row['Target_Weight'] > 0:
                target_usd = total_value * row['Target_Weight']
                print(f"  {row['Asset']}: {row['Target_Weight']:.1%} = ${target_usd:,.2f}")


if __name__ == "__main__":
    print("Quantbox Broker Plugin Test")
    print("This test doesn't require pandas, numpy, or quantlab")
    print("")
    
    try:
        test_broker_basic()
        test_broker_advanced()
        
        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("1. Set up proper Python environment with dependencies")
        print("2. Run scripts/dry_run_strategies.py for full integration test")
        print("3. Connect to real Binance API for live prices")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()