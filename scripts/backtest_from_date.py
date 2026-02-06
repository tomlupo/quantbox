#!/usr/bin/env python3
"""
Backtest All Strategies from a Start Date

Simulates paper trading as if we started on a specific date.
Runs strategies day-by-day and tracks P&L.

Usage:
    cd ~/workspace/quantbox
    PYTHONPATH="packages/quantbox-core/src" .venv/bin/python scripts/backtest_from_date.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "quantbox-core" / "src"))

from quantbox.plugins.datasources import BinanceDataFetcher
from quantbox.plugins.strategies import (
    CryptoTrendStrategy,
    MomentumLongShortStrategy,
    CarverTrendStrategy,
)


def simulate_portfolio(
    weights_history: pd.DataFrame,
    prices: pd.DataFrame,
    initial_capital: float = 10000,
    rebalance_freq: str = 'W',  # Weekly rebalancing
) -> pd.DataFrame:
    """
    Simulate portfolio P&L from weight history.
    
    Args:
        weights_history: DataFrame of weights (date × ticker)
        prices: DataFrame of prices (date × ticker)
        initial_capital: Starting capital
        rebalance_freq: Rebalancing frequency ('D', 'W', 'M')
        
    Returns:
        DataFrame with daily portfolio value
    """
    # Align dates
    common_dates = weights_history.index.intersection(prices.index)
    weights = weights_history.loc[common_dates]
    prices = prices.loc[common_dates]
    
    # Get rebalance dates
    if rebalance_freq == 'D':
        rebalance_dates = set(common_dates)
    else:
        rebalance_dates = set(weights.resample(rebalance_freq).first().index)
    
    # Initialize
    portfolio_value = [initial_capital]
    cash = initial_capital
    positions = {}  # ticker -> {'qty': float, 'entry_price': float}
    
    daily_data = []
    
    for i, date in enumerate(common_dates):
        if i == 0:
            daily_data.append({
                'date': date,
                'value': initial_capital,
                'cash': initial_capital,
                'long_value': 0,
                'short_value': 0,
                'short_pnl': 0,
            })
            continue
        
        prev_date = common_dates[i-1]
        
        # Calculate current value of positions
        long_value = 0
        short_pnl = 0
        
        for ticker, pos in positions.items():
            if ticker not in prices.columns:
                continue
            current_price = prices.loc[date, ticker]
            prev_price = prices.loc[prev_date, ticker]
            
            if pos['qty'] > 0:
                # Long position
                long_value += pos['qty'] * current_price
            else:
                # Short position - P&L from price change
                qty = abs(pos['qty'])
                # Short P&L = (entry_price - current_price) * qty
                pnl = (pos['entry_price'] - current_price) * qty
                short_pnl += pnl
        
        total_value = cash + long_value + short_pnl
        
        # Rebalance if needed
        if date in rebalance_dates:
            target_weights = weights.loc[date].dropna()
            target_weights = target_weights[abs(target_weights) > 0.001]
            
            # Close all positions first (simplified)
            for ticker, pos in list(positions.items()):
                if ticker not in prices.columns:
                    continue
                current_price = prices.loc[date, ticker]
                
                if pos['qty'] > 0:
                    # Sell long
                    cash += pos['qty'] * current_price
                else:
                    # Close short - realize P&L
                    qty = abs(pos['qty'])
                    pnl = (pos['entry_price'] - current_price) * qty
                    cash += pnl  # Add P&L (can be negative)
            
            positions = {}
            long_value = 0
            short_pnl = 0
            
            # Open new positions
            total_value = cash  # After closing all
            
            for ticker, weight in target_weights.items():
                if ticker not in prices.columns:
                    continue
                current_price = prices.loc[date, ticker]
                
                position_value = total_value * abs(weight)
                qty = position_value / current_price
                
                if weight > 0:
                    # Long
                    positions[ticker] = {'qty': qty, 'entry_price': current_price}
                    cash -= qty * current_price
                    long_value += qty * current_price
                else:
                    # Short
                    positions[ticker] = {'qty': -qty, 'entry_price': current_price}
                    # For shorts, we receive cash but have liability
                    # Don't add to cash yet - track P&L separately
        
        # Recalculate after rebalance
        long_value = 0
        short_pnl = 0
        for ticker, pos in positions.items():
            if ticker not in prices.columns:
                continue
            current_price = prices.loc[date, ticker]
            if pos['qty'] > 0:
                long_value += pos['qty'] * current_price
            else:
                qty = abs(pos['qty'])
                pnl = (pos['entry_price'] - current_price) * qty
                short_pnl += pnl
        
        total_value = cash + long_value + short_pnl
        
        daily_data.append({
            'date': date,
            'value': total_value,
            'cash': cash,
            'long_value': long_value,
            'short_value': sum(abs(p['qty']) * prices.loc[date, t] 
                              for t, p in positions.items() 
                              if p['qty'] < 0 and t in prices.columns),
            'short_pnl': short_pnl,
        })
    
    return pd.DataFrame(daily_data).set_index('date')


def run_backtest(start_date: str = '2025-12-31'):
    print("=" * 70)
    print(f"QUANTBOX BACKTEST - Starting from {start_date}")
    print("=" * 70)
    
    # 1. Fetch data
    print("\n📊 Fetching market data...")
    fetcher = BinanceDataFetcher()
    
    tickers = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'DOT', 'ATOM', 'LTC']
    data = fetcher.get_market_data(tickers, lookback_days=500)
    
    prices = data['prices']
    print(f"   Data: {prices.shape[1]} tickers × {len(prices)} days")
    print(f"   Range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # Filter to backtest period
    start = pd.Timestamp(start_date)
    prices_bt = prices[prices.index >= start]
    data_bt = {
        'prices': prices[prices.index <= start],  # Data available at start
        'volume': data['volume'][data['volume'].index <= start],
        'market_cap': data['market_cap'][data['market_cap'].index <= start],
    }
    
    print(f"   Backtest period: {start.date()} to {prices.index[-1].date()} ({len(prices_bt)} days)")
    
    # 2. Define strategies
    strategies = {
        'CryptoTrend': CryptoTrendStrategy(
            top_by_mcap=15,
            top_by_volume=8,
            output_periods=500,
        ),
        'Momentum_LS': MomentumLongShortStrategy(
            n_long=4,
            n_short=4,
            net_exposure=0.0,
            enable_trend_filter=False,
            output_periods=500,
        ),
        'Carver': CarverTrendStrategy(
            target_vol=0.25,
            allow_shorts=True,
            max_gross=1.5,
            output_periods=500,
        ),
    }
    
    results = {}
    
    # 3. Run each strategy
    for name, strategy in strategies.items():
        print(f"\n{'='*70}")
        print(f"📈 {name}")
        print('='*70)
        
        # Run strategy on full data to get weight history
        result = strategy.run(data)
        
        # Get weights DataFrame
        weights = result['weights']
        
        # For multi-index weights, flatten
        if isinstance(weights.columns, pd.MultiIndex):
            # Use default params (vol=50, tranche=5 for trend, etc.)
            if 'CryptoTrend' in name:
                try:
                    weights = weights.xs(('50', 5), axis=1, level=('vol_target', 'tranches'))
                except:
                    weights = weights.iloc[:, :len(tickers)]
            else:
                weights = weights
        
        # Filter to backtest period
        weights_bt = weights[weights.index >= start]
        
        if weights_bt.empty:
            print(f"   No weights for backtest period")
            continue
        
        # Simulate portfolio
        portfolio = simulate_portfolio(
            weights_bt,
            prices_bt,
            initial_capital=10000,
            rebalance_freq='W',
        )
        
        if portfolio.empty:
            print(f"   Portfolio simulation failed")
            continue
        
        # Calculate metrics
        start_value = portfolio['value'].iloc[0]
        end_value = portfolio['value'].iloc[-1]
        total_return = (end_value / start_value - 1) * 100
        
        # Drawdown
        cummax = portfolio['value'].cummax()
        drawdown = (portfolio['value'] - cummax) / cummax
        max_dd = drawdown.min() * 100
        
        # Volatility (annualized)
        returns = portfolio['value'].pct_change().dropna()
        vol = returns.std() * np.sqrt(365) * 100
        
        # Sharpe (assuming 0 risk-free rate)
        sharpe = (returns.mean() * 365) / (returns.std() * np.sqrt(365)) if returns.std() > 0 else 0
        
        print(f"\n   Performance ({start_date} → today):")
        print(f"   Start:       ${start_value:,.2f}")
        print(f"   End:         ${end_value:,.2f}")
        print(f"   Return:      {total_return:+.1f}%")
        print(f"   Max DD:      {max_dd:.1f}%")
        print(f"   Volatility:  {vol:.1f}% (ann.)")
        print(f"   Sharpe:      {sharpe:.2f}")
        
        # Show last positions
        last_weights = weights_bt.iloc[-1].dropna()
        last_weights = last_weights[abs(last_weights) > 0.001]
        
        if not last_weights.empty:
            print(f"\n   Current positions:")
            for t, w in sorted(last_weights.items(), key=lambda x: -x[1])[:6]:
                direction = '📈' if w > 0 else '📉'
                print(f"     {t}: {w:+.1%} {direction}")
        else:
            print(f"\n   Current: FLAT (no positions)")
        
        results[name] = {
            'portfolio': portfolio,
            'return': total_return,
            'max_dd': max_dd,
            'sharpe': sharpe,
            'final_value': end_value,
        }
    
    # 4. Summary
    print(f"\n{'='*70}")
    print("📊 BACKTEST SUMMARY")
    print(f"Period: {start_date} → {prices.index[-1].date()}")
    print('='*70)
    
    print(f"\n{'Strategy':<20} {'Return':>10} {'Max DD':>10} {'Sharpe':>10} {'Final $':>12}")
    print("-" * 62)
    
    for name, r in results.items():
        print(f"{name:<20} {r['return']:>+9.1f}% {r['max_dd']:>9.1f}% {r['sharpe']:>10.2f} ${r['final_value']:>10,.0f}")
    
    # Combined portfolio (equal weight)
    if len(results) == 3:
        combined_value = sum(r['final_value'] for r in results.values())
        combined_return = (combined_value / 30000 - 1) * 100
        print("-" * 62)
        print(f"{'COMBINED (equal)':<20} {combined_return:>+9.1f}%{' ':>10}{' ':>10} ${combined_value:>10,.0f}")
    
    print('='*70)
    
    return results


if __name__ == "__main__":
    import sys
    start_date = sys.argv[1] if len(sys.argv) > 1 else '2025-12-31'
    results = run_backtest(start_date)
