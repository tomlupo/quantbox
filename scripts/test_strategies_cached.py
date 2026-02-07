#!/usr/bin/env python3
"""
Test strategies using cached data from quantlab.

This avoids the data fetcher dependencies by loading pre-cached parquet files.
"""
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta

# Add paths
sys.path.insert(0, str(Path('/home/node/workspace/quantlab')))
sys.path.insert(0, str(Path('/home/node/workspace/quantlab/src')))

import pandas as pd
import numpy as np
import duckdb
import importlib
import yaml

QUANTLAB_ROOT = Path('/home/node/workspace/quantlab')
CACHE_DIR = QUANTLAB_ROOT / 'data' / 'cache'


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_cached_ohlcv() -> dict:
    """Load OHLCV data from parquet cache using DuckDB."""
    ohlcv_dir = CACHE_DIR / 'binance_ohlcv'
    
    if not ohlcv_dir.exists():
        raise FileNotFoundError(f"No cached OHLCV at {ohlcv_dir}")
    
    # Use DuckDB to query parquet files
    con = duckdb.connect()
    
    # Get list of tickers
    tickers = [d.name.replace('ticker=', '') for d in ohlcv_dir.iterdir() if d.is_dir() and d.name.startswith('ticker=')]
    
    ohlcv = {}
    for ticker in tickers:
        ticker_dir = ohlcv_dir / f'ticker={ticker}'
        try:
            df = con.execute(f"""
                SELECT * FROM read_parquet('{ticker_dir}/**/*.parquet')
                ORDER BY date
            """).df()
            if not df.empty:
                ohlcv[ticker] = df
        except Exception as e:
            print(f"  Skipping {ticker}: {e}")
    
    con.close()
    print(f"Loaded {len(ohlcv)} tickers from cache")
    return ohlcv


def load_cmc_rankings() -> pd.DataFrame:
    """Load CMC rankings from cache."""
    rankings_dir = CACHE_DIR / 'cmc_rankings'
    
    # Find most recent rankings file
    rankings_files = sorted(rankings_dir.glob('*.parquet'), reverse=True)
    if not rankings_files:
        raise FileNotFoundError("No CMC rankings cache found")
    
    df = pd.read_parquet(rankings_files[0])
    print(f"Loaded rankings: {len(df)} coins")
    return df


def preprocess_data(ohlcv: dict, rankings: pd.DataFrame) -> dict:
    """Preprocess into format expected by strategies."""
    # Build prices DataFrame
    prices_dict = {}
    volume_dict = {}
    
    for ticker, df in ohlcv.items():
        if 'date' not in df.columns or 'close' not in df.columns:
            continue
        df_clean = df.drop_duplicates(subset=['date'], keep='last').set_index('date')
        prices_dict[ticker] = df_clean['close']
        if 'volume' in df_clean.columns:
            volume_dict[ticker] = df_clean['volume']
    
    prices = pd.DataFrame(prices_dict)
    volume = pd.DataFrame(volume_dict)
    
    # Market cap from rankings
    if 'market_cap' in rankings.columns and 'symbol' in rankings.columns:
        market_cap = rankings.set_index('symbol')['market_cap']
    else:
        market_cap = pd.Series()
    
    # Find intersection: tickers that have BOTH price data AND market cap
    price_tickers = set(prices.columns)
    mc_tickers = set(market_cap.index)
    valid_tickers = list(price_tickers & mc_tickers)
    
    print(f"Raw - Prices: {len(price_tickers)}, MarketCap: {len(mc_tickers)}, Intersection: {len(valid_tickers)}")
    
    # Filter everything to intersection
    prices = prices[valid_tickers]
    volume = volume[[t for t in valid_tickers if t in volume.columns]]
    market_cap = market_cap[valid_tickers]
    rankings_final = rankings[rankings['symbol'].isin(valid_tickers)].copy()
    
    print(f"Final - Prices: {prices.shape}, Volume: {volume.shape}, MarketCap: {len(market_cap)}")
    
    return {
        'prices': prices,
        'volume': volume,
        'market_cap': market_cap,
        'coins_ranking': rankings_final,
        'ohlcv': {k: v for k, v in ohlcv.items() if k in valid_tickers},
        'tickers': valid_tickers
    }


def run_strategies(data: dict, account_config: dict, strategy_configs: dict) -> dict:
    """Run strategy modules."""
    results = {}
    
    for strat in account_config.get('strategies', []):
        name = strat['name']
        weight = strat.get('weight', 1.0)
        
        strat_config = strategy_configs.get(name, {})
        params = {**strat_config, **account_config, **strat.get('params', {})}
        
        try:
            # Import strategy module directly (bypass workflow/__init__ which triggers data_queries)
            import importlib.util
            strategy_path = QUANTLAB_ROOT / 'workflow' / 'strategies' / f'{name}.py'
            spec = importlib.util.spec_from_file_location(f'strategy_{name}', strategy_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            result = module.run(data=data, params=params)
            
            # Handle multi-level columns
            if result['weights'].columns.nlevels > 1:
                result['weights'] = result['weights'].T.groupby('ticker').sum().T
            
            results[name] = {'result': result, 'weight': weight}
            print(f"✅ Strategy {name}: {result['weights'].shape}")
        except Exception as e:
            print(f"❌ Strategy {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def aggregate_weights(strategy_results: dict, account_config: dict) -> pd.Series:
    """Aggregate strategy weights."""
    strategies_names = list(strategy_results.keys())
    
    strategies_asset_weights = pd.concat(
        [strategy_results[key]['result']['weights'] for key in strategies_names],
        axis=1,
        keys=strategies_names,
        names=['strategy']
    )
    
    account_strategies_weights = pd.Series(
        [strategy_results[key]['weight'] for key in strategies_names],
        index=pd.Index(strategies_names, name='strategy')
    )
    
    account_asset_weights = strategies_asset_weights.mul(
        account_strategies_weights, level='strategy'
    ).droplevel(0, axis=1)
    
    tranches = account_config.get('tranches', 1)
    max_leverage = account_config.get('max_leverage', 1)
    
    final_weights = account_asset_weights.rolling(window=tranches).mean().iloc[-1]
    
    if final_weights.sum() > max_leverage:
        final_weights = final_weights / final_weights.sum() * max_leverage
    
    if not account_config.get('allow_negative_weights', False):
        final_weights = final_weights.clip(lower=0)
    
    final_weights = final_weights.sort_values(ascending=False)
    final_weights = final_weights.loc[final_weights != 0]
    
    return final_weights


def load_quantlab_positions() -> dict:
    """Load today's positions from quantlab."""
    today = datetime.now().strftime('%Y-%m-%d')
    reports_path = Path('/home/node/workspace/trading-reports/daily') / today / 'positions.json'
    
    if not reports_path.exists():
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        reports_path = Path('/home/node/workspace/trading-reports/daily') / yesterday / 'positions.json'
    
    with open(reports_path) as f:
        data = json.load(f)
    
    # Sum weights across all strategies
    all_weights = {}
    for strategy_name, strategy_data in data.items():
        for pos in strategy_data.get('positions', []):
            ticker = pos['ticker']
            weight = pos['weight']
            if ticker not in all_weights:
                all_weights[ticker] = 0
            all_weights[ticker] += weight
    
    return all_weights


def compare(quantlab: dict, quantbox: dict):
    """Compare weights."""
    print("\n" + "="*60)
    print("COMPARISON: QUANTLAB (cached) vs QUANTBOX (fresh)")
    print("="*60)
    
    all_assets = sorted(set(quantlab.keys()) | set(quantbox.keys()))
    
    print(f"\n{'Asset':<8} {'QL Weight':>12} {'QB Weight':>12} {'Diff':>10} {'Match':>6}")
    print("-"*50)
    
    matches = 0
    total = 0
    
    for asset in all_assets:
        ql = quantlab.get(asset, 0)
        qb = quantbox.get(asset, 0)
        diff = ql - qb
        
        ql_dir = 'L' if ql > 0.001 else ('S' if ql < -0.001 else '-')
        qb_dir = 'L' if qb > 0.001 else ('S' if qb < -0.001 else '-')
        
        match = '✅' if ql_dir == qb_dir else '❌'
        if ql_dir == qb_dir:
            matches += 1
        total += 1
        
        print(f"{asset:<8} {ql:>+12.4%} {qb:>+12.4%} {diff:>+10.4%} {match:>6}")
    
    print("-"*50)
    print(f"Direction match: {matches}/{total} ({matches/total*100:.1f}%)")


def main():
    account_name = sys.argv[1] if len(sys.argv) > 1 else 'binance'
    print(f"Testing strategies for account: {account_name}")
    
    # 1. Load config
    account_config = load_yaml(QUANTLAB_ROOT / 'config' / 'accounts' / f'{account_name}.yaml')
    strategy_configs = {}
    for strat in account_config.get('strategies', []):
        name = strat['name']
        cfg_path = QUANTLAB_ROOT / 'config' / 'strategies' / f'{name}.yaml'
        if cfg_path.exists():
            strategy_configs[name] = load_yaml(cfg_path)
    
    # 2. Load cached data
    print("\n1. Loading cached data...")
    ohlcv = load_cached_ohlcv()
    rankings = load_cmc_rankings()
    data = preprocess_data(ohlcv, rankings)
    
    # 3. Run strategies
    print("\n2. Running strategies...")
    results = run_strategies(data, account_config, strategy_configs)
    
    if not results:
        print("No strategy results!")
        return
    
    # 4. Aggregate weights
    print("\n3. Aggregating weights...")
    final_weights = aggregate_weights(results, account_config)
    quantbox_weights = dict(final_weights)
    
    print(f"\nQuantbox final weights ({len(quantbox_weights)} assets):")
    for asset, weight in list(quantbox_weights.items())[:10]:
        print(f"  {asset}: {weight:+.4%}")
    
    # 5. Load quantlab positions
    print("\n4. Loading quantlab positions...")
    quantlab_weights = load_quantlab_positions()
    print(f"Quantlab weights ({len(quantlab_weights)} assets)")
    
    # 6. Compare
    compare(quantlab_weights, quantbox_weights)


if __name__ == '__main__':
    main()
