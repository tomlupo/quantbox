#!/usr/bin/env python3
"""
Generate Daily Trading Report

Creates daily artifacts:
- positions.json: Current positions for each strategy
- signals.json: Current signals/forecasts
- summary.md: Human-readable summary
- portfolio_history.parquet: Historical P&L data

Usage:
    cd ~/workspace/quantbox
    PYTHONPATH="packages/quantbox-core/src" .venv/bin/python scripts/generate_report.py
"""

import sys
import os
import json
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

# Report output directory
REPORTS_DIR = Path.home() / "workspace" / "trading-reports"


def run_strategies(data: dict) -> dict:
    """Run all strategies and return results."""
    strategies = {
        'CryptoTrend': CryptoTrendStrategy(
            top_by_mcap=15,
            top_by_volume=8,
        ),
        'Momentum_LS': MomentumLongShortStrategy(
            n_long=4,
            n_short=4,
            net_exposure=0.0,
            enable_trend_filter=False,
        ),
        'Carver': CarverTrendStrategy(
            target_vol=0.25,
            allow_shorts=True,
            max_gross=1.5,
        ),
    }
    
    results = {}
    for name, strategy in strategies.items():
        result = strategy.run(data)
        results[name] = {
            'weights': result.get('simple_weights', {}),
            'exposure': result.get('exposure', {}),
            'forecasts': result.get('forecasts', pd.DataFrame()).iloc[-1].to_dict() if 'forecasts' in result and not result['forecasts'].empty else {},
        }
    
    return results


def generate_positions_json(results: dict, prices: dict) -> dict:
    """Generate positions JSON artifact."""
    positions = {}
    
    for strategy, data in results.items():
        weights = data.get('weights', {})
        strategy_positions = []
        
        for ticker, weight in weights.items():
            price = prices.get(ticker, 0)
            notional = abs(weight) * 10000  # Assuming $10k per strategy
            
            strategy_positions.append({
                'ticker': ticker,
                'weight': weight,
                'direction': 'LONG' if weight > 0 else 'SHORT',
                'price': price,
                'notional_usd': round(notional, 2),
            })
        
        exposure = data.get('exposure', {})
        positions[strategy] = {
            'positions': sorted(strategy_positions, key=lambda x: -x['weight']),
            'exposure': {
                'long': exposure.get('long', sum(w for w in weights.values() if w > 0)),
                'short': exposure.get('short', abs(sum(w for w in weights.values() if w < 0))),
                'net': exposure.get('net', sum(weights.values())),
                'gross': exposure.get('gross', sum(abs(w) for w in weights.values())),
            },
            'n_positions': len(weights),
        }
    
    return positions


def generate_signals_json(results: dict) -> dict:
    """Generate signals JSON artifact."""
    signals = {}
    
    for strategy, data in results.items():
        weights = data.get('weights', {})
        forecasts = data.get('forecasts', {})
        
        strategy_signals = []
        for ticker in set(list(weights.keys()) + list(forecasts.keys())):
            strategy_signals.append({
                'ticker': ticker,
                'weight': weights.get(ticker, 0),
                'forecast': forecasts.get(ticker, None),
                'signal': 'LONG' if weights.get(ticker, 0) > 0.01 else 'SHORT' if weights.get(ticker, 0) < -0.01 else 'FLAT',
            })
        
        signals[strategy] = sorted(strategy_signals, key=lambda x: -(x.get('forecast') or x['weight'] or 0))
    
    return signals


def generate_summary_md(results: dict, prices: dict, date: str) -> str:
    """Generate human-readable summary."""
    lines = [
        f"# Trading Report - {date}",
        f"",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"",
        f"## Market Overview",
        f"",
    ]
    
    # Market prices
    lines.append("| Asset | Price | 24h |")
    lines.append("|-------|------:|----:|")
    for ticker in ['BTC', 'ETH', 'SOL', 'BNB']:
        price = prices.get(ticker, 0)
        lines.append(f"| {ticker} | ${price:,.2f} | - |")
    
    lines.append("")
    lines.append("## Strategy Positions")
    lines.append("")
    
    for strategy, data in results.items():
        weights = data.get('weights', {})
        exposure = data.get('exposure', {})
        
        lines.append(f"### {strategy}")
        lines.append("")
        
        if not weights:
            lines.append("**FLAT** - No positions")
            lines.append("")
            continue
        
        # Exposure
        long_exp = exposure.get('long', sum(w for w in weights.values() if w > 0))
        short_exp = exposure.get('short', abs(sum(w for w in weights.values() if w < 0)))
        net_exp = exposure.get('net', sum(weights.values()))
        
        lines.append(f"**Exposure:** Long {long_exp:.0%} | Short {short_exp:.0%} | Net {net_exp:+.0%}")
        lines.append("")
        
        # Positions table
        lines.append("| Asset | Weight | Direction | Notional |")
        lines.append("|-------|-------:|-----------|----------|")
        
        for ticker, weight in sorted(weights.items(), key=lambda x: -x[1]):
            direction = "📈 LONG" if weight > 0 else "📉 SHORT"
            notional = abs(weight) * 10000
            lines.append(f"| {ticker} | {weight:+.1%} | {direction} | ${notional:,.0f} |")
        
        lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Strategy | Positions | Net Exposure |")
    lines.append("|----------|----------:|-------------:|")
    
    for strategy, data in results.items():
        weights = data.get('weights', {})
        n_pos = len(weights)
        net = data.get('exposure', {}).get('net', sum(weights.values()))
        lines.append(f"| {strategy} | {n_pos} | {net:+.0%} |")
    
    return "\n".join(lines)


def save_daily_report(date: str, positions: dict, signals: dict, summary: str):
    """Save daily report artifacts."""
    daily_dir = REPORTS_DIR / "daily" / date
    daily_dir.mkdir(parents=True, exist_ok=True)
    
    # Save positions
    with open(daily_dir / "positions.json", 'w') as f:
        json.dump(positions, f, indent=2, default=str)
    
    # Save signals
    with open(daily_dir / "signals.json", 'w') as f:
        json.dump(signals, f, indent=2, default=str)
    
    # Save summary
    with open(daily_dir / "summary.md", 'w') as f:
        f.write(summary)
    
    # Update latest.md symlink/copy
    with open(REPORTS_DIR / "latest.md", 'w') as f:
        f.write(summary)
    
    print(f"   Saved to {daily_dir}")


def update_portfolio_history(date: str, results: dict):
    """Append to portfolio history parquet."""
    history_file = REPORTS_DIR / "data" / "portfolio_history.parquet"
    
    # Create row for each strategy
    rows = []
    for strategy, data in results.items():
        weights = data.get('weights', {})
        exposure = data.get('exposure', {})
        
        rows.append({
            'date': date,
            'strategy': strategy,
            'n_positions': len(weights),
            'long_exposure': exposure.get('long', 0),
            'short_exposure': exposure.get('short', 0),
            'net_exposure': exposure.get('net', 0),
            'gross_exposure': exposure.get('gross', 0),
            'positions_json': json.dumps(weights),
        })
    
    new_df = pd.DataFrame(rows)
    
    # Append to existing or create new
    if history_file.exists():
        existing = pd.read_parquet(history_file)
        # Remove existing rows for this date
        existing = existing[existing['date'] != date]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    
    combined.to_parquet(history_file, index=False)
    print(f"   Updated portfolio history ({len(combined)} records)")


def generate_weekly_summary(week_start: str) -> str:
    """Generate weekly summary from daily reports."""
    history_file = REPORTS_DIR / "data" / "portfolio_history.parquet"
    
    if not history_file.exists():
        return "No historical data available."
    
    df = pd.read_parquet(history_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter to this week
    start = pd.to_datetime(week_start)
    end = start + timedelta(days=7)
    week_df = df[(df['date'] >= start) & (df['date'] < end)]
    
    if week_df.empty:
        return f"No data for week starting {week_start}"
    
    lines = [
        f"# Weekly Summary - Week of {week_start}",
        f"",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"",
        f"## Strategy Overview",
        f"",
    ]
    
    for strategy in week_df['strategy'].unique():
        strat_df = week_df[week_df['strategy'] == strategy]
        
        lines.append(f"### {strategy}")
        lines.append("")
        lines.append(f"- Days active: {len(strat_df)}")
        lines.append(f"- Avg positions: {strat_df['n_positions'].mean():.1f}")
        lines.append(f"- Avg net exposure: {strat_df['net_exposure'].mean():.1%}")
        lines.append("")
    
    return "\n".join(lines)


def format_telegram_message(results: dict, prices: dict, date: str) -> str:
    """Format mobile-friendly Telegram notification."""
    lines = [f"📊 {date}"]
    
    for strategy, data in results.items():
        weights = data.get('weights', {})
        exposure = data.get('exposure', {})
        
        # Short strategy name
        short_name = strategy.replace('_', ' ')
        
        if not weights:
            lines.append(f"{short_name}: FLAT")
            continue
        
        # Count positions
        longs = [(t, w) for t, w in weights.items() if w > 0.01]
        shorts = [(t, w) for t, w in weights.items() if w < -0.01]
        
        net = exposure.get('net', sum(weights.values()))
        
        lines.append(f"{short_name} {net:+.0%}")
        
        # Show top 3 positions per side, stacked vertically
        if longs:
            top_longs = sorted(longs, key=lambda x: -x[1])[:3]
            lines.append("📈 " + " ".join([f"{t}" for t, w in top_longs]))
        
        if shorts:
            top_shorts = sorted(shorts, key=lambda x: x[1])[:3]
            lines.append("📉 " + " ".join([f"{t}" for t, w in top_shorts]))
    
    # Market prices compact
    btc = prices.get('BTC', 0)
    eth = prices.get('ETH', 0)
    lines.append(f"BTC {btc/1000:.1f}k ETH {eth:.0f}")
    
    return "\n".join(lines)


def send_telegram_notification(message: str):
    """Send notification via OpenClaw message tool."""
    # This will be called by the agent, not directly
    # Return the message for the agent to send
    return message


def main():
    print("=" * 60)
    print("TRADING REPORT GENERATOR")
    print("=" * 60)
    
    today = datetime.utcnow().strftime('%Y-%m-%d')
    print(f"\n📅 Date: {today}")
    
    # 1. Fetch data
    print("\n📊 Fetching market data...")
    fetcher = BinanceDataFetcher()
    
    tickers = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'DOT', 'ATOM', 'LTC']
    data = fetcher.get_market_data(tickers, lookback_days=400)
    prices = fetcher.get_current_prices(tickers)
    
    print(f"   Loaded {len(data['prices'].columns)} tickers")
    
    # 2. Run strategies
    print("\n🎯 Running strategies...")
    results = run_strategies(data)
    
    for name, res in results.items():
        n_pos = len(res.get('weights', {}))
        net = res.get('exposure', {}).get('net', 0)
        print(f"   {name}: {n_pos} positions, net {net:+.0%}")
    
    # 3. Generate artifacts
    print("\n📝 Generating artifacts...")
    
    positions = generate_positions_json(results, prices)
    signals = generate_signals_json(results)
    summary = generate_summary_md(results, prices, today)
    
    # 4. Save
    print("\n💾 Saving reports...")
    save_daily_report(today, positions, signals, summary)
    update_portfolio_history(today, results)
    
    # 5. Check if we need weekly summary (if today is Sunday or Monday)
    weekday = datetime.utcnow().weekday()
    if weekday == 0:  # Monday - generate last week's summary
        week_start = (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
        weekly = generate_weekly_summary(week_start)
        
        week_num = datetime.utcnow().isocalendar()[1] - 1
        weekly_file = REPORTS_DIR / "weekly" / f"{datetime.utcnow().year}-W{week_num:02d}.md"
        weekly_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(weekly_file, 'w') as f:
            f.write(weekly)
        print(f"   Generated weekly summary: {weekly_file.name}")
    
    print("\n" + "=" * 60)
    print("✅ Report generation complete!")
    print(f"   Daily: {REPORTS_DIR}/daily/{today}/")
    print(f"   Latest: {REPORTS_DIR}/latest.md")
    print("=" * 60)
    
    # Generate Telegram message
    telegram_msg = format_telegram_message(results, prices, today)
    
    # Save telegram message for agent to send
    telegram_file = REPORTS_DIR / "latest_telegram.txt"
    with open(telegram_file, 'w') as f:
        f.write(telegram_msg)
    
    print("\n📱 Telegram notification:")
    print(telegram_msg)
    
    # Return message for programmatic use
    return telegram_msg


if __name__ == "__main__":
    msg = main()
    
    # If run with --send flag, output just the message for piping
    if len(sys.argv) > 1 and sys.argv[1] == '--send':
        print("\n---TELEGRAM_MESSAGE---")
        print(msg)
        print("---END_MESSAGE---")
