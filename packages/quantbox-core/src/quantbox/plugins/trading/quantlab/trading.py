# --- Imports ---
#%%
import argparse
import importlib
import os
import sys
import yaml
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Relative imports for quantbox plugin structure
from .data_fetcher import CryptoDataFetcher
from .token_policy import TokenPolicy
from . import utils
from . import trading_bot

# --- Constants ---
#%%
# Constants moved to trading_bot modules

# --- Logging Setup ---
#%%
logger = utils.get_logger()

# --- YAML Loader ---
#%%
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# --- Stage: Load Config ---
#%%
# Default config root - points to original quantlab repo
QUANTLAB_ROOT = os.environ.get('QUANTLAB_ROOT', '/home/node/workspace/quantlab')

# Quantbox config directory
QUANTBOX_CONFIG_DIR = os.environ.get('QUANTBOX_CONFIG_DIR', '/home/node/workspace/quantbox/config')

def load_config(account_name, config_root=None):
    # Check quantbox config first, then fall back to quantlab
    quantbox_config_path = os.path.join(QUANTBOX_CONFIG_DIR, f'accounts/{account_name}.yaml')
    if os.path.exists(quantbox_config_path):
        config_root = QUANTBOX_CONFIG_DIR
        account_config_path = quantbox_config_path
        logger.info(f"Using quantbox config: {account_config_path}")
    else:
        if config_root is None:
            config_root = QUANTLAB_ROOT
        account_config_path = os.path.join(config_root, f'config/accounts/{account_name}.yaml')
    
    if not os.path.exists(account_config_path):
        logger.error(f"Account config not found: {account_config_path}")
        sys.exit(1)
    account_config = load_yaml(account_config_path)
    strategy_configs = {}
    for strat in account_config.get('strategies', []):
        name = strat['name']
        strat_config_path = os.path.join(config_root, f'config/strategies/{name}.yaml')
        if os.path.exists(strat_config_path):
            strategy_configs[name] = load_yaml(strat_config_path)
        else:
            strategy_configs[name] = {}
    logger.info(f"Loaded config for account: {account_name}")
    return account_config, strategy_configs

# --- Stage: Fetch Market Data ---
#%%
# Quantbox data directory (writable)
QUANTBOX_CACHE_DIR = os.environ.get('QUANTBOX_CACHE_DIR', '/home/node/workspace/quantbox/data/cache')

def fetch_market_data(account_config, token_policy: TokenPolicy = None):
    """
    Fetch market data with token policy filtering.
    
    Args:
        account_config: Account configuration dict
        token_policy: TokenPolicy instance for filtering (optional, uses config if None)
    """
    # Use quantbox's own cache directory (writable), fall back to quantlab
    config = account_config.copy()
    if 'cache_dir' not in config:
        if os.path.exists(QUANTBOX_CACHE_DIR) and os.access(QUANTBOX_CACHE_DIR, os.W_OK):
            config['cache_dir'] = QUANTBOX_CACHE_DIR
        else:
            config['cache_dir'] = os.path.join(QUANTLAB_ROOT, 'data/cache')
    
    fetcher = CryptoDataFetcher(config)
    top_coins = account_config.get('top_coins', 100)
    lookback_days = account_config.get('lookback_days', 730)

    coins_ranking = fetcher.fetch_cmc_rankings()
    
    # Use TokenPolicy if provided or config has token_policy
    if token_policy is None and 'token_policy' in account_config:
        token_policy = TokenPolicy.from_dict(account_config)
    
    if token_policy is not None:
        # NEW: Use TokenPolicy for filtering (allowlist mode)
        logger.info(f"Using TokenPolicy: {token_policy}")
        
        # Check for new tokens and alert
        new_tokens = token_policy.detect_new_tokens(coins_ranking, top_n=top_coins)
        if new_tokens:
            alert_msg = token_policy.format_new_token_alert(new_tokens)
            logger.warning(f"NEW TOKENS DETECTED:\n{alert_msg}")
            # Store alert for later delivery (e.g., Telegram)
            config['_new_token_alert'] = alert_msg
            config['_new_tokens'] = new_tokens
        
        # Filter to allowed tokens only
        all_symbols = coins_ranking['symbol'].tolist()
        allowed_symbols = set(token_policy.filter_allowed(all_symbols))
        
        # Log denied tokens with reasons
        denied = token_policy.filter_denied(all_symbols[:top_coins])
        if denied:
            logger.info(f"Denied tokens in top {top_coins}:")
            for symbol, reason in denied[:10]:  # Log first 10
                logger.info(f"  ❌ {symbol}: {reason}")
            if len(denied) > 10:
                logger.info(f"  ... and {len(denied) - 10} more")
        
        # Filter coins ranking to allowed only
        tradable_coins_ranking = coins_ranking[coins_ranking['symbol'].isin(allowed_symbols)].copy()
        tradable_coins_ranking = tradable_coins_ranking.reset_index(drop=True)
        
        logger.info(f"TokenPolicy: {len(all_symbols)} total → {len(allowed_symbols)} allowed → {len(tradable_coins_ranking)} tradable")
    else:
        # LEGACY: Use not_tradable_on_binance from config (quantlab compatibility)
        not_tradable_config = account_config.get('not_tradable_on_binance', [])
        not_tradable_symbols = {item['symbol'] for item in not_tradable_config}
        
        # Filter out non-tradable symbols from coins ranking
        tradable_coins_ranking = coins_ranking[~coins_ranking['symbol'].isin(not_tradable_symbols)].copy()
        tradable_coins_ranking = tradable_coins_ranking.reset_index(drop=True)
        
        logger.info(f"Legacy mode: {len(not_tradable_symbols)} non-tradable symbols filtered")
    
    # Use filtered ranking for ticker selection
    tickers = tradable_coins_ranking.iloc[:top_coins]['symbol'].tolist()
    ohlcv = fetcher.fetch_ohlcv(tickers, lookback_days)

    # missing tickers from ohlcv with their ranking (now using filtered ranking)
    missing_tickers = set(tradable_coins_ranking.iloc[:top_coins]['symbol'].tolist()) - set(ohlcv.keys())
    missing_tickers_ranking = tradable_coins_ranking[tradable_coins_ranking['symbol'].isin(missing_tickers)]
    
    # Log counts
    logger.info(f"Original top {top_coins} coins: {len(coins_ranking.iloc[:top_coins])}")
    logger.info(f"Tradable coins after filtering: {len(tradable_coins_ranking.iloc[:top_coins])}")
    logger.info(f"Missing tickers from OHLCV: {len(missing_tickers)}")
    
    if len(missing_tickers) > 0:
        logger.info(f"Missing tickers: {missing_tickers_ranking[['rank', 'symbol']]}")
    
    # Add missing tickers information to filtered coins ranking
    tradable_coins_ranking['is_missing'] = tradable_coins_ranking['symbol'].isin(missing_tickers)
    tradable_coins_ranking['is_unexpected_missing'] = tradable_coins_ranking['symbol'].isin(missing_tickers)

    data = {
        'ohlcv': ohlcv, 
        'tickers': tickers, 
        'coins_ranking': tradable_coins_ranking,
        'token_policy': token_policy,
        '_new_token_alert': config.get('_new_token_alert'),
        '_new_tokens': config.get('_new_tokens', [])
    }

    return data

# --- Stage: Preprocess Data ---
#%%
def preprocess_data(market_data):
    logger.info("Preprocessing market data...")
    
    # Extract prices and volume from OHLCV data
    ohlcv = market_data.get('ohlcv', {})
    coins_ranking = market_data.get('coins_ranking', pd.DataFrame())
    prices = pd.DataFrame([v.drop_duplicates(subset=['date'], keep='last').set_index('date')['close'].rename(k) for k,v in ohlcv.items()]).T
    volume = pd.DataFrame([v.drop_duplicates(subset=['date'], keep='last').set_index('date')['volume'].rename(k) for k,v in ohlcv.items()]).T
    market_cap = market_data['coins_ranking'].set_index('symbol')['market_cap']
    
    # Validate data
    if prices.empty:
        logger.error("No price data available")
        sys.exit(1)
    if volume.empty:
        logger.error("No volume data available")
        sys.exit(1)
    if market_cap.empty:
        logger.error("No market cap data available")
        sys.exit(1)

    end_date = pd.Timestamp.now().normalize() - timedelta(days=1)
    if end_date not in prices.index:
        logger.error(f"No price data for {end_date}")
        sys.exit(1)
    if end_date not in volume.index:
        logger.error(f"No volume data for {end_date}")
        sys.exit(1)


    logger.info(f"Price data shape: {prices.shape}")
    logger.info(f"Volume data shape: {volume.shape}")
    logger.info(f"Market cap data shape: {market_cap.shape}")
    logger.info(f"Coins ranking data shape: {coins_ranking.shape}")


    processed_data = market_data.copy()
    processed_data['prices'] = prices
    processed_data['volume'] = volume
    processed_data['market_cap'] = market_cap
    logger.info(f"Preprocessing complete. Final data shape: {prices.shape}")
    return processed_data

# --- Stage: Run Strategies ---
#%%
def run_strategies(market_data, account_config, strategy_configs, use_strategy_configs=True):
    results = {}
    for strat in account_config.get('strategies', []):
        name = strat['name']
        weight = strat.get('weight', 1.0)
        if use_strategy_configs:
            strat_config = strategy_configs.get(name, {})
            params = {**strat_config, **account_config, **strat.get('params', {})}
        else:
            params = {}
        try:
            # Import from local strategies directory (copied from quantlab)
            module = importlib.import_module(f'.strategies.{name}', package=__package__)
        except ImportError as e:
            logger.error(f"Could not import strategy module .strategies.{name}: {e}")
            sys.exit(1)
        result = module.run(data=market_data, params=params)
        # check weights columns
        if result['weights'].columns.nlevels > 1:
            if result['weights'].droplevel('ticker',axis=1).columns.unique().shape[0] > 1:
                logger.warning(f"Strategy {name} has multiple weights columns: {result['weights'].droplevel('ticker',axis=1).columns.unique()}")
            result['weights'] = result['weights'].T.groupby('ticker').sum().T
        results[name] = {'result': result, 'weight': weight}
        logger.info(f"Ran strategy: {name}")
    return results

# --- Stage: Generate Enhanced Report ---
#%%
def generate_report(execution_report, portfolio_orders):
    logger.info("Generating comprehensive trading report...")
    #TODO: to be implemented
    return {}

# --- Artifact Payload Builder ---
#%%
def _build_artifact_payload(portfolio_orders: dict, execution_report: dict, final_asset_weights, account_config: dict) -> dict:
    """Build the artifact payload from trading pipeline outputs."""
    total_value = portfolio_orders.get('total_value', 0)
    paper_trading = account_config.get('paper_trading', True)
    trading_enabled = account_config.get('trading_enabled', False)

    # Strategy weights
    strategy_weights = {}
    if final_asset_weights is not None:
        try:
            strategy_weights = {k: round(float(v), 6) for k, v in final_asset_weights.items()}
        except Exception:
            pass

    # Rebalancing table
    rebalancing_table = []
    rebalancing_df = portfolio_orders.get('rebalancing')
    if rebalancing_df is not None and not rebalancing_df.empty:
        col_map = {
            'Asset': 'asset', 'Current Quantity': 'current_qty',
            'Current Value': 'current_value', 'Current Weight': 'current_weight',
            'Target Quantity': 'target_qty', 'Target Value': 'target_value',
            'Target Weight': 'target_weight', 'Weight Delta': 'weight_delta',
            'Delta Quantity': 'delta_qty', 'Price': 'price', 'Trade Action': 'trade_action',
        }
        for _, row in rebalancing_df.iterrows():
            entry = {}
            for src, dst in col_map.items():
                val = row.get(src)
                if val is not None:
                    entry[dst] = round(float(val), 6) if isinstance(val, (int, float, np.floating)) else str(val)
            rebalancing_table.append(entry)

    # Orders table
    orders_table = []
    orders_df = portfolio_orders.get('orders')
    if orders_df is not None and not orders_df.empty:
        col_map = {
            'Asset': 'asset', 'Symbol': 'symbol', 'Action': 'action',
            'Raw Quantity': 'raw_qty', 'Adjusted Quantity': 'adjusted_qty',
            'Price': 'price', 'Notional Value': 'notional_value',
            'Min Notional': 'min_notional', 'Min Qty': 'min_qty',
            'Step Size': 'step_size', 'Scaling Factor': 'scaling_factor',
            'Order Status': 'order_status', 'Reason': 'reason',
        }
        for _, row in orders_df.iterrows():
            entry = {}
            for src, dst in col_map.items():
                val = row.get(src)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    entry[dst] = None
                elif isinstance(val, (int, float, np.floating)):
                    entry[dst] = round(float(val), 6)
                else:
                    entry[dst] = str(val)
            orders_table.append(entry)

    # Execution summary
    summary = execution_report.get('summary', {})
    executed_orders = []
    failed_orders = []
    for detail in execution_report.get('orders_details', []):
        if detail.get('status') == 'FILLED':
            executed_orders.append({
                'symbol': str(detail.get('Symbol', '')),
                'action': str(detail.get('Action', '')),
                'quantity': detail.get('executed_quantity'),
                'executed_price': detail.get('executed_price'),
                'reference_price': detail.get('reference_price'),
                'commission_paid': detail.get('commission_paid'),
                'commission_asset': detail.get('commission_asset'),
                'spread_bps': round(float(detail.get('spread_pct', 0) or 0) * 10000, 1),
                'total_cost_pct': detail.get('total_cost_pct'),
                'status': 'FILLED',
            })
        elif detail.get('status') == 'FAILED':
            failed_orders.append({
                'symbol': str(detail.get('Symbol', '')),
                'action': str(detail.get('Action', '')),
                'error': str(detail.get('error', '')),
                'status': 'FAILED',
            })

    execution_summary = {
        'total_executed': summary.get('total_executed', 0),
        'total_failed': summary.get('total_failed', 0),
        'total_value_traded': round(float(summary.get('total_value', 0)), 2),
        'total_cost': round(float(summary.get('total_cost', 0)), 4),
        'executed_orders': executed_orders,
        'failed_orders': failed_orders,
    }

    # Price diagnostics (from orders generation — count prices)
    all_assets_in_rebalancing = set()
    ok_prices = 0
    fail_prices = 0
    failed_assets = []
    held_with_failed = []
    if rebalancing_df is not None and not rebalancing_df.empty:
        for _, row in rebalancing_df.iterrows():
            asset = row.get('Asset')
            if asset:
                all_assets_in_rebalancing.add(asset)
                price = row.get('Price')
                if price is not None and price > 0:
                    ok_prices += 1
                else:
                    fail_prices += 1
                    failed_assets.append(asset)
                    if row.get('Current Quantity', 0) > 0:
                        held_with_failed.append(asset)

    price_diagnostics = {
        'total_assets': len(all_assets_in_rebalancing),
        'fetched_ok': ok_prices,
        'fetched_fail': fail_prices,
        'failed_assets': failed_assets,
        'held_assets_with_failed_price': held_with_failed,
    }

    return {
        'portfolio_value': round(float(total_value), 2),
        'paper_trading': paper_trading,
        'trading_enabled': trading_enabled,
        'strategy_weights': strategy_weights,
        'rebalancing_table': rebalancing_table,
        'orders_table': orders_table,
        'execution_summary': execution_summary,
        'price_diagnostics': price_diagnostics,
    }


# --- Main Entry Point ---
#%%
def run(account_name, config, use_strategy_configs=True,*args,**kwargs):
    logger.info(f"Starting enhanced trading pipeline for account: {account_name}")
    
    # Load configuration
    account_config, strategy_configs = load_config(account_name)
    
    # Validate API credentials
    api_key = os.environ.get('API_KEY_BINANCE')
    api_secret = os.environ.get('API_SECRET_BINANCE')
    
    if not api_key or not api_secret:
        logger.warning("Binance API credentials not found, using paper trading mode")
        account_config['paper_trading'] = True


    # Get Binance client
    client = trading_bot.get_binance_client(api_key, api_secret)

    # Initialize paper trading if enabled
    if account_config.get('paper_trading', True):
        logger.info("Paper trading mode enabled")
        balance_info = trading_bot.initialize_paper_trading(account_name, account_config, client)
        logger.info(f"Paper trading initialized: ${balance_info['current_balance']:.2f} balance")
    
    # Fetch and preprocess market data
    market_data = fetch_market_data(account_config)
    processed_data = preprocess_data(market_data)
    
    # Run strategies - mandatory, no fallback
    if not account_config.get('strategies'):
        error_msg = f"No strategies configured for account: {account_name}"
        logger.error(error_msg)
        trading_bot.send_telegram_notifications({'error': error_msg}, {}, {}, account_config)
        sys.exit(1)
    
    strategy_results = run_strategies(processed_data, account_config, strategy_configs, use_strategy_configs)
    if not strategy_results:
        error_msg = f"Strategy execution failed for account: {account_name}"
        logger.error(error_msg)
        trading_bot.send_telegram_notifications({'error': error_msg}, {}, {}, account_config)
        sys.exit(1)
    
    # Aggregate strategy weights with account-level weights
    try:
        strategies_names = [k for k in strategy_results.keys()]
        strategies_asset_weights =  pd.concat([strategy_results[key]['result']['weights'] for key in strategy_results.keys()],axis=1,keys=strategies_names,names=['strategy'])
        account_strategies_weights = pd.Series([strategy_results[key]['weight'] for key in strategy_results.keys()],index=pd.Index(strategies_names,name='strategy'))
        account_asset_weights = strategies_asset_weights.mul(account_strategies_weights,level='strategy').droplevel(0,axis=1)
    except:
        error_msg = f"Error aggregating strategy weights for account: {account_name}"
        logger.error(error_msg)
        trading_bot.send_telegram_notifications({'error': error_msg}, {}, {}, account_config)
        sys.exit(1)
    
    # Apply risk management
    #TODO: migrate to src/trading_bot/risk_management.py
    tranches = account_config.get('tranches', 1)
    max_leverage = account_config.get('max_leverage', 1)
    final_asset_weights = account_asset_weights.rolling(window=tranches).mean().iloc[-1]
    if final_asset_weights.sum() > max_leverage:
        logger.warning(f"Max leverage exceeded for account: {account_name}. Adjusting weights to {max_leverage}")
        final_asset_weights = final_asset_weights / final_asset_weights.sum() * max_leverage
    # Clamp negatives to zero for spot trading
    if not account_config.get('allow_negative_weights', False):
        final_asset_weights = final_asset_weights.clip(lower=0)
    final_asset_weights = final_asset_weights.sort_values(ascending=False)
    final_asset_weights = final_asset_weights.loc[final_asset_weights!=0]
    logger.info(f"Final asset weights after applying risk management: {final_asset_weights}")

    # Generate portfolio orders with enhanced analysis using modular system
    logger.info("Generating portfolio orders...")
    portfolio_orders = trading_bot.generate_portfolio_orders(account_config, final_asset_weights, client)
    
    # Execute orders with sell-before-buy logic
    logger.info("Executing orders...")
    execution_report = trading_bot.execute_orders(portfolio_orders['orders'], account_name, account_config, client)
    
    # Reconcile account balance with portfolio orders
    #TODO: add reconciliation of account balance
    #account_balance = trading_bot.reconcile_account_balance(execution_report, portfolio_orders, account_config, client)

    # Generate comprehensive report
    #report = generate_report(execution_report, portfolio_orders)
    
    # Send enhanced Telegram notifications
    trading_bot.send_telegram_notifications(portfolio_orders, execution_report, account_config)
    
    logger.info("Enhanced trading pipeline completed.")

    # --- Build artifact payload ---
    artifact_payload = _build_artifact_payload(
        portfolio_orders, execution_report, final_asset_weights, account_config
    )

    return {
        'orders': portfolio_orders,
        'execution_report': execution_report,
        #'report': report,
        'market_data': market_data,
        'processed_data': processed_data,
        'strategy_results': strategy_results,
        'final_asset_weights': final_asset_weights,
        'portfolio_orders': portfolio_orders,
        'account_config': account_config,
        'strategy_configs': strategy_configs,
        'artifact_payload': artifact_payload,
    }