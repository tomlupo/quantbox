#!/usr/bin/env python3
"""
Validate Quantbox port against Quantlab.
Runs identical inputs through both pipelines and compares outputs.
"""
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Setup paths
QUANTLAB_ROOT = '/home/node/workspace/quantlab'
QUANTBOX_ROOT = '/home/node/workspace/quantbox'

# Add paths for both codebases
sys.path.insert(0, os.path.join(QUANTLAB_ROOT, 'src'))
sys.path.insert(0, os.path.join(QUANTLAB_ROOT))
sys.path.insert(0, os.path.join(QUANTBOX_ROOT, 'packages/quantbox-core/src'))

# Shared cache - use quantlab's cache for both
SHARED_CACHE_DIR = os.path.join(QUANTLAB_ROOT, 'data/cache')

def compare_dataframes(df1, df2, name, tolerance=1e-6):
    """Compare two DataFrames and report differences."""
    issues = []
    
    if df1 is None and df2 is None:
        return []
    if df1 is None or df2 is None:
        return [f"{name}: One is None (df1={df1 is not None}, df2={df2 is not None})"]
    
    # Shape comparison
    if df1.shape != df2.shape:
        issues.append(f"{name}: Shape mismatch - {df1.shape} vs {df2.shape}")
    
    # Column comparison
    cols1 = set(df1.columns) if hasattr(df1, 'columns') else set()
    cols2 = set(df2.columns) if hasattr(df2, 'columns') else set()
    if cols1 != cols2:
        missing_in_2 = cols1 - cols2
        missing_in_1 = cols2 - cols1
        if missing_in_2:
            issues.append(f"{name}: Columns missing in quantbox: {missing_in_2}")
        if missing_in_1:
            issues.append(f"{name}: Extra columns in quantbox: {missing_in_1}")
    
    # Value comparison for common columns
    common_cols = cols1 & cols2
    if common_cols and df1.shape[0] == df2.shape[0]:
        for col in common_cols:
            try:
                if df1[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    max_diff = (df1[col] - df2[col]).abs().max()
                    if max_diff > tolerance:
                        issues.append(f"{name}.{col}: Max diff = {max_diff}")
            except:
                pass
    
    return issues

def validate_step_by_step():
    """Run validation comparing each pipeline step."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'status': 'running',
        'steps': {},
        'issues': [],
        'summary': {}
    }
    
    print("=" * 60)
    print("QUANTLAB → QUANTBOX PORT VALIDATION")
    print("=" * 60)
    
    # ==== STEP 1: Load Config ====
    print("\n[1/6] Comparing load_config()...")
    try:
        # Quantlab
        from workflow.trading import load_config as ql_load_config
        ql_config, ql_strat_configs = ql_load_config('binance')
        
        # Quantbox - force it to use quantlab config
        os.environ['QUANTBOX_CONFIG_DIR'] = '/nonexistent'  # Force fallback
        from quantbox.plugins.trading.quantlab.trading import load_config as qb_load_config
        qb_config, qb_strat_configs = qb_load_config('binance', config_root=QUANTLAB_ROOT)
        
        # Compare
        config_match = ql_config == qb_config
        strat_match = ql_strat_configs == qb_strat_configs
        
        report['steps']['load_config'] = {
            'status': 'PASS' if config_match and strat_match else 'FAIL',
            'config_match': config_match,
            'strat_configs_match': strat_match
        }
        print(f"   Config match: {config_match}")
        print(f"   Strategy configs match: {strat_match}")
        
    except Exception as e:
        report['steps']['load_config'] = {'status': 'ERROR', 'error': str(e)}
        report['issues'].append(f"load_config: {e}")
        print(f"   ERROR: {e}")
    
    # ==== STEP 2: Fetch Market Data ====
    print("\n[2/6] Comparing fetch_market_data()...")
    try:
        # Both should use quantlab's cache
        from workflow.data_fetcher import CryptoDataFetcher as QL_Fetcher
        from quantbox.plugins.trading.quantlab.data_fetcher import CryptoDataFetcher as QB_Fetcher
        
        ql_fetcher = QL_Fetcher({'cache_dir': SHARED_CACHE_DIR})
        qb_fetcher = QB_Fetcher({'cache_dir': SHARED_CACHE_DIR})
        
        # Fetch rankings
        ql_rankings = ql_fetcher.fetch_cmc_rankings(limit=30)
        qb_rankings = qb_fetcher.fetch_cmc_rankings(limit=30)
        
        # Compare rankings
        rankings_issues = compare_dataframes(ql_rankings, qb_rankings, 'rankings')
        
        # Fetch OHLCV for same tickers
        test_tickers = ['BTC', 'ETH', 'BNB']
        ql_ohlcv = ql_fetcher.fetch_ohlcv(test_tickers, lookback_days=30)
        qb_ohlcv = qb_fetcher.fetch_ohlcv(test_tickers, lookback_days=30)
        
        ohlcv_issues = []
        for ticker in test_tickers:
            if ticker in ql_ohlcv and ticker in qb_ohlcv:
                issues = compare_dataframes(ql_ohlcv[ticker], qb_ohlcv[ticker], f'ohlcv.{ticker}')
                ohlcv_issues.extend(issues)
            elif ticker in ql_ohlcv:
                ohlcv_issues.append(f"ohlcv.{ticker}: Missing in quantbox")
            elif ticker in qb_ohlcv:
                ohlcv_issues.append(f"ohlcv.{ticker}: Missing in quantlab")
        
        all_issues = rankings_issues + ohlcv_issues
        report['steps']['fetch_market_data'] = {
            'status': 'PASS' if not all_issues else 'FAIL',
            'issues': all_issues,
            'ql_rankings_shape': ql_rankings.shape if ql_rankings is not None else None,
            'qb_rankings_shape': qb_rankings.shape if qb_rankings is not None else None,
        }
        print(f"   Rankings: QL={ql_rankings.shape}, QB={qb_rankings.shape}")
        print(f"   OHLCV tickers: QL={len(ql_ohlcv)}, QB={len(qb_ohlcv)}")
        if all_issues:
            for issue in all_issues[:5]:
                print(f"   ⚠️ {issue}")
        else:
            print("   ✅ Data matches")
            
    except Exception as e:
        import traceback
        report['steps']['fetch_market_data'] = {'status': 'ERROR', 'error': str(e)}
        report['issues'].append(f"fetch_market_data: {e}")
        print(f"   ERROR: {e}")
        traceback.print_exc()
    
    # ==== STEP 3: Preprocess Data ====
    print("\n[3/6] Comparing preprocess_data()...")
    try:
        from workflow.trading import preprocess_data as ql_preprocess
        from quantbox.plugins.trading.quantlab.trading import preprocess_data as qb_preprocess
        
        # Create market data dict (simplified)
        market_data = {
            'ohlcv': ql_ohlcv if 'ql_ohlcv' in dir() else {},
            'coins_ranking': ql_rankings if 'ql_rankings' in dir() else pd.DataFrame()
        }
        
        ql_processed = ql_preprocess(market_data)
        qb_processed = qb_preprocess(market_data)
        
        preprocess_issues = []
        for key in ql_processed.keys():
            if key in qb_processed:
                if isinstance(ql_processed[key], pd.DataFrame):
                    issues = compare_dataframes(ql_processed[key], qb_processed[key], f'processed.{key}')
                    preprocess_issues.extend(issues)
            else:
                preprocess_issues.append(f"processed.{key}: Missing in quantbox")
        
        report['steps']['preprocess_data'] = {
            'status': 'PASS' if not preprocess_issues else 'FAIL',
            'issues': preprocess_issues
        }
        if preprocess_issues:
            for issue in preprocess_issues[:5]:
                print(f"   ⚠️ {issue}")
        else:
            print("   ✅ Preprocessing matches")
            
    except Exception as e:
        report['steps']['preprocess_data'] = {'status': 'ERROR', 'error': str(e)}
        report['issues'].append(f"preprocess_data: {e}")
        print(f"   ERROR: {e}")
    
    # ==== STEP 4: Run Strategies ====
    print("\n[4/6] Comparing run_strategies()...")
    try:
        from workflow.trading import run_strategies as ql_run_strategies
        from quantbox.plugins.trading.quantlab.trading import run_strategies as qb_run_strategies
        
        # Use processed data from quantlab
        ql_results = ql_run_strategies(ql_processed, ql_config, ql_strat_configs)
        qb_results = qb_run_strategies(qb_processed, qb_config, qb_strat_configs)
        
        strategy_issues = []
        for strat in ql_results.keys():
            if strat in qb_results:
                ql_weights = ql_results[strat]['result']['weights']
                qb_weights = qb_results[strat]['result']['weights']
                issues = compare_dataframes(ql_weights, qb_weights, f'strategy.{strat}.weights')
                strategy_issues.extend(issues)
            else:
                strategy_issues.append(f"strategy.{strat}: Missing in quantbox")
        
        report['steps']['run_strategies'] = {
            'status': 'PASS' if not strategy_issues else 'FAIL',
            'issues': strategy_issues,
            'ql_strategies': list(ql_results.keys()),
            'qb_strategies': list(qb_results.keys())
        }
        print(f"   Strategies: QL={list(ql_results.keys())}, QB={list(qb_results.keys())}")
        if strategy_issues:
            for issue in strategy_issues[:5]:
                print(f"   ⚠️ {issue}")
        else:
            print("   ✅ Strategy results match")
            
    except Exception as e:
        import traceback
        report['steps']['run_strategies'] = {'status': 'ERROR', 'error': str(e)}
        report['issues'].append(f"run_strategies: {e}")
        print(f"   ERROR: {e}")
        traceback.print_exc()
    
    # ==== Summary ====
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for s in report['steps'].values() if s.get('status') == 'PASS')
    failed = sum(1 for s in report['steps'].values() if s.get('status') == 'FAIL')
    errors = sum(1 for s in report['steps'].values() if s.get('status') == 'ERROR')
    
    report['summary'] = {
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'total': len(report['steps'])
    }
    report['status'] = 'PASS' if failed == 0 and errors == 0 else 'FAIL'
    
    print(f"Passed: {passed}/{len(report['steps'])}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"\nOverall: {'✅ PASS' if report['status'] == 'PASS' else '❌ FAIL'}")
    
    # Save report
    report_path = os.path.join(QUANTBOX_ROOT, 'validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")
    
    return report

if __name__ == '__main__':
    validate_step_by_step()
