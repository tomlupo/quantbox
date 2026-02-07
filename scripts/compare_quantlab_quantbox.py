#!/usr/bin/env python3
"""
Compare Quantlab vs Quantbox Strategy Outputs

Runs the same account config through both systems and compares:
1. Final asset weights
2. Position directions (LONG/SHORT)
3. Weight magnitudes

Usage:
    python scripts/compare_quantlab_quantbox.py binance
    python scripts/compare_quantlab_quantbox.py paper1
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# IMPORTANT: Order matters! quantbox-core MUST come before quantlab
# because quantlab has an old quantbox module that conflicts
QUANTBOX_PATH = Path(__file__).parent.parent / "packages" / "quantbox-core" / "src"
sys.path.insert(0, str(QUANTBOX_PATH))

# Quantlab paths come AFTER quantbox-core
QUANTLAB_PATH = Path('/home/node/workspace/quantlab')
# NOTE: Don't add quantlab/src to path - it has a conflicting 'quantbox' module
# The runner will handle adding paths as needed

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_quantlab_weights(account_name: str) -> dict:
    """Load today's weights from quantlab trading-reports."""
    today = datetime.now().strftime('%Y-%m-%d')
    reports_path = Path('/home/node/workspace/trading-reports/daily') / today / 'positions.json'
    
    if not reports_path.exists():
        logger.warning(f"No quantlab report for {today}, trying yesterday")
        yesterday = (datetime.now() - __import__('datetime').timedelta(days=1)).strftime('%Y-%m-%d')
        reports_path = Path('/home/node/workspace/trading-reports/daily') / yesterday / 'positions.json'
    
    if not reports_path.exists():
        raise FileNotFoundError(f"No quantlab positions found at {reports_path}")
    
    with open(reports_path) as f:
        data = json.load(f)
    
    # Extract weights from all strategies
    all_weights = {}
    for strategy_name, strategy_data in data.items():
        for pos in strategy_data.get('positions', []):
            ticker = pos['ticker']
            weight = pos['weight']
            if ticker not in all_weights:
                all_weights[ticker] = 0
            all_weights[ticker] += weight
    
    return all_weights


def run_quantbox_strategies(account_name: str) -> dict:
    """Run strategies through quantbox and get weights."""
    from quantbox.plugins.strategies.quantlab_runner import run
    
    result = run(
        account_name=account_name,
        quantlab_root=str(QUANTLAB_PATH)
    )
    
    return result['final_asset_weights']


def compare_weights(quantlab_weights: dict, quantbox_weights: dict) -> dict:
    """
    Compare weights from both systems.
    
    Returns comparison dict with:
        - matching: assets with same direction
        - mismatched: assets with different directions
        - only_quantlab: assets only in quantlab
        - only_quantbox: assets only in quantbox
    """
    all_assets = set(quantlab_weights.keys()) | set(quantbox_weights.keys())
    
    comparison = {
        'matching': [],
        'mismatched': [],
        'only_quantlab': [],
        'only_quantbox': [],
        'weight_diff': {}
    }
    
    for asset in sorted(all_assets):
        ql_weight = quantlab_weights.get(asset, 0)
        qb_weight = quantbox_weights.get(asset, 0)
        
        ql_dir = 'LONG' if ql_weight > 0 else ('SHORT' if ql_weight < 0 else 'FLAT')
        qb_dir = 'LONG' if qb_weight > 0 else ('SHORT' if qb_weight < 0 else 'FLAT')
        
        if asset in quantlab_weights and asset not in quantbox_weights:
            comparison['only_quantlab'].append({
                'asset': asset,
                'weight': ql_weight,
                'direction': ql_dir
            })
        elif asset not in quantlab_weights and asset in quantbox_weights:
            comparison['only_quantbox'].append({
                'asset': asset,
                'weight': qb_weight,
                'direction': qb_dir
            })
        elif ql_dir == qb_dir:
            comparison['matching'].append({
                'asset': asset,
                'quantlab_weight': ql_weight,
                'quantbox_weight': qb_weight,
                'direction': ql_dir,
                'diff': abs(ql_weight - qb_weight)
            })
        else:
            comparison['mismatched'].append({
                'asset': asset,
                'quantlab': {'weight': ql_weight, 'direction': ql_dir},
                'quantbox': {'weight': qb_weight, 'direction': qb_dir}
            })
        
        comparison['weight_diff'][asset] = {
            'quantlab': ql_weight,
            'quantbox': qb_weight,
            'diff': ql_weight - qb_weight
        }
    
    return comparison


def print_comparison(comparison: dict):
    """Print formatted comparison."""
    print("\n" + "="*60)
    print("QUANTLAB vs QUANTBOX COMPARISON")
    print("="*60)
    
    print(f"\n✅ MATCHING ({len(comparison['matching'])} assets):")
    for item in comparison['matching']:
        diff_pct = item['diff'] * 100
        print(f"   {item['asset']:6} {item['direction']:5} | QL: {item['quantlab_weight']:+.4f} | QB: {item['quantbox_weight']:+.4f} | Δ: {diff_pct:+.2f}%")
    
    if comparison['mismatched']:
        print(f"\n❌ MISMATCHED ({len(comparison['mismatched'])} assets):")
        for item in comparison['mismatched']:
            print(f"   {item['asset']:6} | QL: {item['quantlab']['direction']} {item['quantlab']['weight']:+.4f} | QB: {item['quantbox']['direction']} {item['quantbox']['weight']:+.4f}")
    
    if comparison['only_quantlab']:
        print(f"\n⚠️  ONLY IN QUANTLAB ({len(comparison['only_quantlab'])} assets):")
        for item in comparison['only_quantlab']:
            print(f"   {item['asset']:6} {item['direction']:5} {item['weight']:+.4f}")
    
    if comparison['only_quantbox']:
        print(f"\n⚠️  ONLY IN QUANTBOX ({len(comparison['only_quantbox'])} assets):")
        for item in comparison['only_quantbox']:
            print(f"   {item['asset']:6} {item['direction']:5} {item['weight']:+.4f}")
    
    # Summary
    total = len(comparison['matching']) + len(comparison['mismatched'])
    match_rate = len(comparison['matching']) / total * 100 if total > 0 else 0
    print(f"\n📊 SUMMARY: {match_rate:.1f}% direction match ({len(comparison['matching'])}/{total})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_quantlab_quantbox.py <account_name>")
        print("  e.g.: python compare_quantlab_quantbox.py binance")
        sys.exit(1)
    
    account_name = sys.argv[1]
    
    print(f"Comparing Quantlab vs Quantbox for account: {account_name}")
    
    # 1. Load quantlab weights from reports
    print("\n1. Loading Quantlab weights from trading-reports...")
    try:
        quantlab_weights = load_quantlab_weights(account_name)
        print(f"   Found {len(quantlab_weights)} assets")
    except FileNotFoundError as e:
        print(f"   Error: {e}")
        quantlab_weights = {}
    
    # 2. Run quantbox strategies
    print("\n2. Running Quantbox strategies...")
    try:
        quantbox_weights = run_quantbox_strategies(account_name)
        print(f"   Generated {len(quantbox_weights)} assets")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        quantbox_weights = {}
    
    # 3. Compare
    if quantlab_weights and quantbox_weights:
        comparison = compare_weights(quantlab_weights, quantbox_weights)
        print_comparison(comparison)
    else:
        print("\nCannot compare - missing data from one or both systems")


if __name__ == '__main__':
    main()
