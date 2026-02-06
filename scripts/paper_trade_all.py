#!/usr/bin/env python3
"""
Paper Trade All Strategies - Run all 3 strategies in parallel paper trading.

Usage:
    cd ~/workspace/quantbox
    PYTHONPATH="packages/quantbox-core/src" .venv/bin/python scripts/paper_trade_all.py
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "quantbox-core" / "src"))

from quantbox.plugins.datasources import BinanceDataFetcher
from quantbox.plugins.strategies import (
    CryptoTrendStrategy,
    MomentumLongShortStrategy, 
    CarverTrendStrategy,
)
from quantbox.plugins.broker import BinanceLiveBroker


def run_paper_trading():
    print("=" * 70)
    print("QUANTBOX PAPER TRADING - All Strategies")
    print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)
    
    # 1. Fetch live data
    print("\n📊 Fetching live market data...")
    fetcher = BinanceDataFetcher()
    
    tickers = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'DOT', 'ATOM', 'LTC']
    data = fetcher.get_market_data(tickers, lookback_days=400)
    print(f"   Data: {len(data['prices'].columns)} tickers × {len(data['prices'])} days")
    
    # Get current prices for broker
    current_prices = fetcher.get_current_prices(tickers)
    
    # 2. Define strategies with separate accounts
    strategies = {
        'trend': {
            'name': 'CryptoTrend (Donchian)',
            'strategy': CryptoTrendStrategy(
                top_by_mcap=15,
                top_by_volume=8,
            ),
            'account': 'paper_trend',
            'capital': 10000,
        },
        'momentum_ls': {
            'name': 'Momentum L/S (Market Neutral)',
            'strategy': MomentumLongShortStrategy(
                n_long=4,
                n_short=4,
                net_exposure=0.0,
                enable_trend_filter=False,
            ),
            'account': 'paper_momentum_ls',
            'capital': 10000,
        },
        'carver': {
            'name': 'Carver Trend (Vol Target 25%)',
            'strategy': CarverTrendStrategy(
                target_vol=0.25,
                allow_shorts=True,
                max_gross=1.5,
            ),
            'account': 'paper_carver',
            'capital': 10000,
        },
    }
    
    results = {}
    
    # 3. Run each strategy
    for key, config in strategies.items():
        print(f"\n{'='*70}")
        print(f"📈 {config['name']}")
        print('='*70)
        
        # Run strategy
        result = config['strategy'].run(data)
        weights = result.get('simple_weights', {})
        
        print(f"\nSignals:")
        if weights:
            for t, w in sorted(weights.items(), key=lambda x: -x[1])[:8]:
                direction = '📈 LONG' if w > 0 else '📉 SHORT'
                print(f"  {t}: {w:+.1%} {direction}")
        else:
            print("  (No positions - all signals flat)")
        
        # Initialize broker for this strategy
        broker = BinanceLiveBroker(
            paper_trading=True,
            account_name=config['account'],
            stable_coin="USDC",
            trades_dir="/tmp/quantbox_paper",
        )
        
        # Set initial capital (reset paper portfolio)
        broker._save_paper_holdings({'USDC': config['capital']})
        
        # Load current prices
        for ticker, price in current_prices.items():
            broker._price_cache.set(ticker, price)
        
        # Execute if we have positions
        if weights:
            # Scale weights (keep some cash buffer for long-short)
            if 'exposure' in result:
                gross = result['exposure'].get('gross', 1.0)
                if gross > 0.95:
                    # Already near full exposure
                    target_weights = weights
                else:
                    target_weights = weights
            else:
                # Scale to 85% for long-only
                total = sum(abs(w) for w in weights.values())
                if total > 0:
                    scale = min(0.85 / total, 1.0)
                    target_weights = {k: v * scale for k, v in weights.items()}
                else:
                    target_weights = weights
            
            print(f"\nExecuting paper trades...")
            exec_result = broker.execute_rebalancing(target_weights)
            
            print(f"  Executed: {exec_result['summary']['total_executed']} orders")
            print(f"  Value traded: ${exec_result['summary']['total_value']:,.2f}")
        else:
            print(f"\nNo trades (flat signal)")
            exec_result = None
        
        # Get final state
        state = broker.describe()
        
        print(f"\nPortfolio State:")
        print(f"  Value: ${state['portfolio_value']:,.2f}")
        print(f"  Cash: ${state['cash'].get('USDC', 0):,.2f}")
        
        if state['positions']:
            print(f"  Positions:")
            for pos in sorted(state['positions'], key=lambda x: -abs(x['value']))[:6]:
                print(f"    {pos['symbol']}: ${pos['value']:,.2f} ({pos['weight']:.1%})")
        
        results[key] = {
            'name': config['name'],
            'weights': weights,
            'state': state,
            'exposure': result.get('exposure', {}),
        }
    
    # 4. Summary
    print(f"\n{'='*70}")
    print("📊 SUMMARY - All Strategies")
    print('='*70)
    
    total_value = 0
    total_capital = 30000  # 3 × $10,000
    
    for key, r in results.items():
        value = r['state']['portfolio_value']
        total_value += value
        exp = r.get('exposure', {})
        
        if exp:
            exp_str = f"L:{exp.get('long',0):.0%} S:{exp.get('short',0):.0%} Net:{exp.get('net',0):+.0%}"
        else:
            exp_str = "Long-only"
        
        n_pos = len([w for w in r['weights'].values() if abs(w) > 0.001]) if r['weights'] else 0
        print(f"\n{r['name']}:")
        print(f"  Value: ${value:,.2f} | Positions: {n_pos} | {exp_str}")
    
    print(f"\n{'='*70}")
    print(f"TOTAL VALUE: ${total_value:,.2f} (from ${total_capital:,.2f} capital)")
    print('='*70)
    
    return results


if __name__ == "__main__":
    results = run_paper_trading()
