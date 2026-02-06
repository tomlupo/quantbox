#!/usr/bin/env python3
"""
Carver Trend Following - Live Trading Script

Runs the Carver strategy on Binance Futures with:
- 50% target volatility
- Built-in risk limits
- Telegram notifications
- Dry-run mode for safety

Usage:
    # Dry run (see what would happen)
    python run_carver_live.py --dry-run
    
    # Live trading
    python run_carver_live.py --live
    
    # With custom budget (default: uses full account)
    python run_carver_live.py --live --budget 1000

Requirements:
    - API_KEY_BINANCE and API_SECRET_BINANCE in environment
    - TELEGRAM_TOKEN and TELEGRAM_CHAT_ID for notifications
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add quantbox to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages/quantbox-core/src"))

from quantbox.plugins.broker.binance_futures import BinanceFuturesBroker, RiskConfig, send_telegram
from quantbox.plugins.strategies.carver_trend import CarverTrendStrategy
from quantbox.plugins.datasources.binance_data import BinanceDataFetcher

# ============================================================================
# Configuration
# ============================================================================

# Strategy config
TARGET_VOL = 0.50  # 50% target volatility
MAX_POSITION = 0.50  # Max 50% single position
MAX_GROSS = 2.0  # Max 200% gross exposure
LOOKBACK_DAYS = 400  # Data lookback for signals

# Universe - top liquid futures
UNIVERSE = [
    'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX',
    'DOT', 'LINK', 'MATIC', 'UNI', 'LTC', 'ATOM', 'ETC',
]

# Risk limits (match Carver strategy)
RISK_CONFIG = RiskConfig(
    max_position_pct=0.50,
    max_gross_exposure=2.0,
    max_daily_loss_pct=0.10,  # Stop at 10% daily loss
    min_order_notional=5.0,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================================
# Main Functions
# ============================================================================

def load_env():
    """Load environment variables from quantlab .env if needed."""
    env_file = Path(__file__).parent.parent.parent / "quantlab" / ".env"
    if env_file.exists():
        logger.info(f"Loading env from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key not in os.environ:
                        os.environ[key] = value


def fetch_data(universe: list) -> dict:
    """Fetch market data for strategy."""
    logger.info(f"Fetching data for {len(universe)} symbols...")
    
    fetcher = BinanceDataFetcher()
    data = fetcher.get_market_data(
        tickers=universe,
        lookback_days=LOOKBACK_DAYS,
    )
    
    logger.info(f"Data shape: {data['prices'].shape}")
    return data


def run_strategy(data: dict) -> dict:
    """Run Carver strategy and get target weights."""
    logger.info(f"Running Carver strategy (target_vol={TARGET_VOL:.0%})...")
    
    strategy = CarverTrendStrategy(
        target_vol=TARGET_VOL,
        max_position=MAX_POSITION,
        max_gross=MAX_GROSS,
        allow_shorts=True,
    )
    
    result = strategy.run(data)
    
    # Get latest weights
    weights = result['simple_weights']
    exposure = result['exposure']
    
    logger.info(f"Positions: {len(weights)}")
    logger.info(f"Exposure - Long: {exposure['long']:.1%}, Short: {exposure['short']:.1%}")
    
    return result


def format_weights_summary(weights: dict, exposure: dict) -> str:
    """Format weights for display."""
    lines = [
        f"📊 <b>Carver Signals</b> (vol={TARGET_VOL:.0%})",
        f"",
        f"Exposure: L={exposure['long']:.0%} / S={exposure['short']:.0%}",
        f"Net: {exposure['net']:+.0%} | Gross: {exposure['gross']:.0%}",
        f"",
    ]
    
    # Sort by weight
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    for symbol, weight in sorted_weights:
        if weight > 0:
            lines.append(f"📈 {symbol}: {weight:+.1%}")
        else:
            lines.append(f"📉 {symbol}: {weight:+.1%}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Carver Trend - Live Trading")
    parser.add_argument('--live', action='store_true', help="Execute live trades")
    parser.add_argument('--dry-run', action='store_true', help="Show what would happen")
    parser.add_argument('--budget', type=float, help="Budget limit (default: full account)")
    parser.add_argument('--notify', action='store_true', help="Send Telegram notification")
    args = parser.parse_args()
    
    if not args.live and not args.dry_run:
        print("Specify --live or --dry-run")
        print("Use --dry-run first to see what would happen!")
        sys.exit(1)
    
    # Load environment
    load_env()
    
    # Check credentials
    if not os.environ.get('API_KEY_BINANCE'):
        print("ERROR: API_KEY_BINANCE not set")
        sys.exit(1)
    
    # Fetch data and run strategy
    data = fetch_data(UNIVERSE)
    result = run_strategy(data)
    
    weights = result['simple_weights']
    exposure = result['exposure']
    
    # Print summary
    print("\n" + "="*60)
    print(format_weights_summary(weights, exposure).replace('<b>', '').replace('</b>', ''))
    print("="*60 + "\n")
    
    if args.dry_run:
        # Dry run - show what would happen
        logger.info("DRY RUN - Connecting to Binance...")
        
        broker = BinanceFuturesBroker(risk=RISK_CONFIG)
        result = broker.rebalance_to_weights(weights, dry_run=True)
        
        print("\n📋 Planned Trades:")
        print(f"Portfolio Value: ${result['portfolio_value']:,.2f}")
        print()
        
        for trade in result.get('trades', []):
            emoji = "🟢" if trade['side'] == 'buy' else "🔴"
            print(f"  {emoji} {trade['side'].upper():4s} {trade['quantity']:12.6f} {trade['symbol']:6s} "
                  f"(${trade['notional']:8.2f}) | {trade['current_weight']:+.1%} → {trade['target_weight']:+.1%}")
        
        if not result.get('trades'):
            print("  No trades needed - positions already aligned")
        
        print("\nRun with --live to execute these trades.")
        
        # Notify if requested
        if args.notify:
            msg = format_weights_summary(weights, exposure)
            msg += "\n\n<i>Dry run - no trades executed</i>"
            send_telegram(
                os.environ.get('TELEGRAM_TOKEN', ''),
                os.environ.get('TELEGRAM_CHAT_ID', ''),
                msg,
            )
    
    else:
        # LIVE TRADING
        print("⚠️  LIVE TRADING MODE ⚠️")
        print()
        
        # Confirm
        confirm = input("Type 'EXECUTE' to proceed with live trades: ")
        if confirm != 'EXECUTE':
            print("Aborted.")
            sys.exit(0)
        
        logger.info("Connecting to Binance Futures...")
        broker = BinanceFuturesBroker(risk=RISK_CONFIG)
        
        # Show account status
        balance = broker.get_balance()
        print(f"\nAccount Balance: ${balance['total']:,.2f}")
        
        if args.budget and args.budget < balance['total']:
            # Scale weights to budget
            scale = args.budget / balance['total']
            weights = {k: v * scale for k, v in weights.items()}
            print(f"Scaled to budget: ${args.budget:,.2f}")
        
        # Execute rebalance
        logger.info("Executing rebalance...")
        result = broker.rebalance_to_weights(weights, dry_run=False)
        
        print(f"\n✅ Rebalance complete")
        print(f"   Trades planned: {result['trades_planned']}")
        print(f"   Trades executed: {result['trades_executed']}")
        
        # Show final positions
        print("\n" + broker.format_status().replace('<b>', '').replace('</b>', ''))


if __name__ == "__main__":
    main()
