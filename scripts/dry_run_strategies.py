#!/usr/bin/env python3
"""
Dry Run Strategies - Test quantlab strategies with quantbox broker.

This script:
1. Fetches market data using quantlab's data fetcher
2. Runs 2-3 strategies to generate target weights
3. Uses quantbox broker (paper mode) to preview trades

Usage:
    python scripts/dry_run_strategies.py
    python scripts/dry_run_strategies.py --strategies crypto_trend_catcher cross_asset_momentum
"""
import sys
import os
import importlib
import argparse
from pathlib import Path

# Add quantlab to path
QUANTLAB_PATH = Path(__file__).parent.parent.parent / "quantlab"
sys.path.insert(0, str(QUANTLAB_PATH / "src"))
sys.path.insert(0, str(QUANTLAB_PATH))

# Add quantbox to path
QUANTBOX_PATH = Path(__file__).parent.parent / "packages" / "quantbox-core" / "src"
sys.path.insert(0, str(QUANTBOX_PATH))

import pandas as pd
import numpy as np

# Setup basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_mock_market_data() -> dict:
    """
    Generate mock market data for dry run testing.
    In production, this would use quantlab's CryptoDataFetcher.
    
    Returns:
        dict with 'prices', 'volume', 'market_cap' DataFrames
    """
    logger.info("Generating mock market data for dry run...")
    
    # Create date range (last 400 days)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=400, freq='D')
    
    # Expanded list of crypto assets (including stablecoins for universe filtering demo)
    base_prices = {
        # Major cryptos
        'BTC': 95000, 'ETH': 3200, 'SOL': 190, 'BNB': 650, 'XRP': 2.5,
        'ADA': 1.0, 'AVAX': 35, 'DOT': 7, 'LINK': 22, 'MATIC': 0.5,
        'DOGE': 0.35, 'SHIB': 0.00002, 'LTC': 120, 'TRX': 0.25, 'ATOM': 8,
        'UNI': 14, 'APT': 9, 'NEAR': 5, 'INJ': 25, 'FIL': 5,
        # Stablecoins (should be filtered out by universe selection)
        'USDT': 1.0, 'USDC': 1.0, 'BUSD': 1.0, 'DAI': 1.0, 'FDUSD': 1.0,
        # Low-cap / problematic tokens (should be filtered)
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
    
    symbols = list(base_prices.keys())
    np.random.seed(42)
    
    prices_data = {}
    volume_data = {}
    mcap_data = {}
    
    for sym in symbols:
        base = base_prices[sym]
        mc_base = market_cap_billions.get(sym, 1) * 1e9
        vol_base = volume_billions.get(sym, 0.1) * 1e9
        
        # Volatility based on ticker type
        if sym in ['USDT', 'USDC', 'BUSD', 'DAI', 'FDUSD']:
            volatility = 0.0001  # Stablecoins
        elif sym in ['BTC', 'ETH']:
            volatility = 0.02   # Large caps
        else:
            volatility = 0.04   # Altcoins
        
        # Generate random walk for price
        returns = np.random.normal(0.001, volatility, len(dates))
        price_path = base * np.cumprod(1 + returns)
        prices_data[sym] = price_path
        
        # Market cap follows price with some noise
        mc_noise = np.random.normal(1, 0.05, len(dates))
        mcap_data[sym] = mc_base * (price_path / base) * mc_noise
        
        # Volume is noisier
        vol_noise = np.random.lognormal(0, 0.3, len(dates))
        volume_data[sym] = vol_base / base * vol_noise  # Volume in units
    
    prices = pd.DataFrame(prices_data, index=dates)
    volume = pd.DataFrame(volume_data, index=dates)
    market_cap = pd.DataFrame(mcap_data, index=dates)
    
    logger.info(f"Generated data: {len(symbols)} assets, {len(dates)} days")
    
    return {
        'prices': prices,
        'volume': volume,
        'market_cap': market_cap
    }


def run_strategy(strategy_name: str, data: dict, params: dict = None) -> dict:
    """
    Run a strategy module and return weights.
    
    Args:
        strategy_name: Name of strategy (e.g., 'crypto_trend_catcher')
        data: Market data dict
        params: Strategy parameters (optional)
    
    Returns:
        dict with 'weights' DataFrame and 'details'
    """
    logger.info(f"Running strategy: {strategy_name}")
    
    # Load strategy module directly to avoid quantlab dependency chains
    strategy_path = QUANTLAB_PATH / "workflow" / "strategies" / f"{strategy_name}.py"
    if not strategy_path.exists():
        logger.error(f"Strategy file not found: {strategy_path}")
        return None
    
    try:
        spec = importlib.util.spec_from_file_location(strategy_name, strategy_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        logger.error(f"Could not load strategy {strategy_name}: {e}")
        return None
    
    # Use default params if not provided
    if params is None:
        params = getattr(module, 'DEFAULT_PARAMS', {})
    
    result = module.run(data=data, params=params)
    
    # Get latest weights
    weights_df = result.get('weights')
    if weights_df is not None and not weights_df.empty:
        # Handle multi-level columns if present
        if weights_df.columns.nlevels > 1:
            weights_df = weights_df.T.groupby('ticker').sum().T
        
        # Get the most recent weights
        latest_weights = weights_df.iloc[-1].to_dict()
        # Filter out zero/nan weights
        latest_weights = {k: v for k, v in latest_weights.items() 
                         if pd.notna(v) and v > 0.001}
        
        logger.info(f"Strategy {strategy_name} generated weights for {len(latest_weights)} assets")
        return {
            'strategy': strategy_name,
            'weights': latest_weights,
            'weights_df': weights_df,
            'details': result.get('details', {})
        }
    
    logger.warning(f"Strategy {strategy_name} returned no weights")
    return None


def combine_strategy_weights(strategy_results: list, strategy_weights: dict = None) -> dict:
    """
    Combine weights from multiple strategies.
    
    Args:
        strategy_results: List of strategy result dicts
        strategy_weights: Dict of strategy_name -> allocation weight (default: equal)
    
    Returns:
        Combined weights dict
    """
    if not strategy_results:
        return {}
    
    # Default: equal weight
    if strategy_weights is None:
        n = len(strategy_results)
        strategy_weights = {r['strategy']: 1.0/n for r in strategy_results}
    
    combined = {}
    for result in strategy_results:
        strat_name = result['strategy']
        strat_weight = strategy_weights.get(strat_name, 1.0 / len(strategy_results))
        
        for asset, weight in result['weights'].items():
            if asset not in combined:
                combined[asset] = 0.0
            combined[asset] += weight * strat_weight
    
    # Normalize to sum to 0.85 (keep 15% cash buffer)
    total = sum(combined.values())
    if total > 0:
        combined = {k: v / total * 0.85 for k, v in combined.items()}
    
    return combined


def filter_weights_by_universe(
    target_weights: dict,
    data: dict,
    universe_config: dict = None
) -> dict:
    """
    Filter target weights by universe selection.
    
    Args:
        target_weights: Dict of asset -> target weight
        data: Market data dict with 'prices', 'volume', 'market_cap'
        universe_config: Optional UniverseConfig params
    
    Returns:
        Filtered weights dict (only assets in current universe)
    """
    try:
        from quantbox.plugins.broker import UniverseSelector, UniverseConfig
    except ImportError:
        logger.warning("Could not import UniverseSelector - returning unfiltered weights")
        return target_weights
    
    # Create selector
    config_params = universe_config or {}
    config = UniverseConfig(
        market_cap_top_n=config_params.get('market_cap_top_n', 30),
        portfolio_max_coins=config_params.get('portfolio_max_coins', 10),
        exclude_stablecoins=config_params.get('exclude_stablecoins', True),
    )
    selector = UniverseSelector(config=config)
    
    # Get current universe
    current_universe = selector.get_current_universe(
        data['prices'],
        data['volume'],
        data['market_cap']
    )
    
    logger.info(f"Universe filter: {len(current_universe)} assets in universe")
    logger.info(f"Current universe: {current_universe}")
    
    # Filter weights
    filtered = {k: v for k, v in target_weights.items() if k in current_universe}
    
    # Re-normalize if we filtered anything
    if filtered and sum(filtered.values()) > 0:
        total = sum(filtered.values())
        filtered = {k: v / total * sum(target_weights.values()) for k, v in filtered.items()}
    
    removed = set(target_weights.keys()) - set(filtered.keys())
    if removed:
        logger.info(f"Filtered out (not in universe): {removed}")
    
    return filtered


def dry_run_with_broker(target_weights: dict, data: dict = None, use_universe_filter: bool = True):
    """
    Use quantbox broker to preview trades.
    
    Args:
        target_weights: Dict of asset -> target weight
        data: Market data dict (for universe filtering)
        use_universe_filter: Apply universe selection filter
    """
    logger.info("Initializing quantbox broker (paper mode)...")
    
    try:
        from quantbox.plugins.broker import BinanceLiveBroker
    except ImportError:
        logger.warning("Could not import BinanceLiveBroker - showing weights only")
        print("\n" + "="*60)
        print("TARGET WEIGHTS (would be passed to broker)")
        print("="*60)
        for asset, weight in sorted(target_weights.items(), key=lambda x: -x[1]):
            print(f"  {asset:8s}: {weight:6.2%}")
        return
    
    # Apply universe filter if requested and data available
    if use_universe_filter and data is not None:
        print("\n" + "="*60)
        print("UNIVERSE SELECTION")
        print("="*60)
        original_count = len(target_weights)
        target_weights = filter_weights_by_universe(target_weights, data)
        print(f"Filtered: {original_count} → {len(target_weights)} assets")
    
    # Initialize broker in paper mode (no real trades)
    broker = BinanceLiveBroker(
        paper_trading=True,
        account_name="dry_run_test",
        stable_coin="USDC",
        trades_dir="/tmp/quantbox_dry_run"
    )
    
    # Get live prices for the broker
    try:
        from quantbox.plugins.datasources import BinanceDataFetcher
        fetcher = BinanceDataFetcher()
        tickers_to_price = list(target_weights.keys())
        live_prices = fetcher.get_current_prices(tickers_to_price)
        for asset, price in live_prices.items():
            broker._price_cache.set(asset, price)
        print(f"Loaded {len(live_prices)} live prices from Binance")
    except Exception as e:
        logger.warning(f"Could not fetch live prices: {e} - using fallback")
        mock_prices = {
            'BTC': 95000.0, 'ETH': 3200.0, 'SOL': 190.0, 'BNB': 650.0,
            'XRP': 2.5, 'ADA': 1.0, 'AVAX': 35.0, 'DOT': 7.0, 'LINK': 22.0, 'MATIC': 0.5,
            'DOGE': 0.35, 'SHIB': 0.00002, 'LTC': 120.0, 'TRX': 0.25,
        }
        for asset, price in mock_prices.items():
            broker._price_cache.set(asset, price)
    
    print("\n" + "="*60)
    print("BROKER STATE")
    print("="*60)
    state = broker.describe()
    print(f"Mode: {state['mode']}")
    print(f"Portfolio Value: {state['portfolio_value']:,.2f} {state['stable_coin']}")
    print(f"Cash: {state['cash']}")
    
    print("\n" + "="*60)
    print("TARGET WEIGHTS")
    print("="*60)
    for asset, weight in sorted(target_weights.items(), key=lambda x: -x[1]):
        print(f"  {asset:8s}: {weight:6.2%}")
    
    print("\n" + "="*60)
    print("REBALANCING ANALYSIS (DRY RUN)")
    print("="*60)
    
    analysis = broker.generate_rebalancing(target_weights)
    if not analysis.empty:
        # Show key columns
        display_cols = ['Asset', 'Current_Weight', 'Target_Weight', 'Weight_Delta', 'Action']
        print(analysis[display_cols].to_string(index=False))
        
        # Summary
        buys = analysis[analysis['Action'] == 'Buy']
        sells = analysis[analysis['Action'] == 'Sell']
        print(f"\nTrades: {len(buys)} buys, {len(sells)} sells")


def get_live_market_data(tickers: list[str] = None, lookback_days: int = 400) -> dict:
    """
    Fetch live market data from Binance.
    
    Args:
        tickers: List of tickers (default: top volume tickers)
        lookback_days: Days of history to fetch
        
    Returns:
        dict with 'prices', 'volume', 'market_cap' DataFrames
    """
    logger.info("Fetching live market data from Binance...")
    
    try:
        from quantbox.plugins.datasources import BinanceDataFetcher
    except ImportError:
        logger.error("BinanceDataFetcher not available - using mock data")
        return get_mock_market_data()
    
    fetcher = BinanceDataFetcher()
    
    # Get default tickers if not provided
    if tickers is None:
        tickers = fetcher.get_tradable_tickers(min_volume_usd=50e6)[:20]
        logger.info(f"Using top {len(tickers)} tickers by volume")
    
    data = fetcher.get_market_data(tickers, lookback_days=lookback_days)
    
    if data['prices'].empty:
        logger.error("No live data fetched - falling back to mock")
        return get_mock_market_data()
    
    logger.info(f"Fetched live data: {len(data['prices'].columns)} tickers, {len(data['prices'])} days")
    return data


def main():
    parser = argparse.ArgumentParser(description='Dry run quantlab strategies with quantbox broker')
    parser.add_argument('--strategies', nargs='+', 
                       default=['crypto_trend_catcher', 'cross_asset_momentum'],
                       help='Strategies to run')
    parser.add_argument('--use-mock-data', action='store_true', default=False,
                       help='Use mock data instead of live fetch')
    parser.add_argument('--lookback-days', type=int, default=400,
                       help='Days of history to fetch')
    parser.add_argument('--tickers', nargs='+', default=None,
                       help='Specific tickers to use (default: auto-select top volume)')
    args = parser.parse_args()
    
    print("="*60)
    print("QUANTBOX DRY RUN - Strategy Testing")
    print("="*60)
    
    # 1. Get market data
    if args.use_mock_data:
        data = get_mock_market_data()
    else:
        data = get_live_market_data(args.tickers, args.lookback_days)
    
    # 2. Run strategies
    results = []
    for strat_name in args.strategies:
        result = run_strategy(strat_name, data)
        if result:
            results.append(result)
            print(f"\n{strat_name} weights:")
            for asset, weight in sorted(result['weights'].items(), key=lambda x: -x[1])[:5]:
                print(f"  {asset}: {weight:.2%}")
    
    if not results:
        logger.error("No strategies produced weights")
        return
    
    # 3. Combine weights
    print("\n" + "="*60)
    print(f"COMBINING {len(results)} STRATEGIES")
    print("="*60)
    
    combined_weights = combine_strategy_weights(results)
    
    # If strategies returned empty (mock data issue), use fallback weights
    if not combined_weights:
        print("\nStrategies returned empty weights (mock data limitation)")
        print("Using fallback demonstration weights...")
        combined_weights = {
            'BTC': 0.35,
            'ETH': 0.25,
            'SOL': 0.15,
            'BNB': 0.10,
        }
    
    # 4. Dry run with broker (with universe filtering)
    dry_run_with_broker(combined_weights, data=data, use_universe_filter=True)
    
    print("\n" + "="*60)
    print("DRY RUN COMPLETE")
    print("="*60)
    print("Next steps:")
    print("1. Review the rebalancing analysis above")
    print("2. Adjust strategy params if needed")
    print("3. Run with real data: --use-mock-data=false")
    print("4. Execute: broker.execute_rebalancing(target_weights)")


if __name__ == "__main__":
    main()
