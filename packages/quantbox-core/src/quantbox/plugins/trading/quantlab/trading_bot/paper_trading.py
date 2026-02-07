# --- Imports ---
import os
import json
import time
import random
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from .. import utils

logger = utils.get_logger()

# --- Paper Trading Constants ---
PAPER_TRADING_SLIPPAGE = 0.0005  # 0.05% slippage for market orders
PAPER_TRADING_SPREAD = 0.001     # 0.1% bid-ask spread
PAPER_TRADING_COMMISSION = 0.001 # 0.1% commission
PAPER_TRADING_MIN_FILL_DELAY = 0.1  # Minimum seconds for order fill
PAPER_TRADING_MAX_FILL_DELAY = 2.0  # Maximum seconds for order fill

# --- Paper Trading State Management ---
def save_paper_trading_state(account_name: str, balance_info: dict) -> None:
    """
    Save paper trading state to file for persistence.
    
    Args:
        account_name: Name of the trading account
        balance_info: Balance and position information
    """
    state_file = f"data/paper_trading/paper_trading_{account_name}.json"
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    
    state_data = {
        'account_name': account_name,
        'last_updated': datetime.now().isoformat(),
        'balance_info': balance_info
    }
    
    try:
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        logger.info(f"Paper trading state saved to {state_file}")
    except Exception as e:
        logger.error(f"Failed to save paper trading state: {e}")

def load_paper_trading_state(account_name: str) -> dict:
    """
    Load paper trading state from file.
    
    Args:
        account_name: Name of the trading account
        
    Returns:
        dict: Loaded balance information or default values
    """
    state_file = f"data/paper_trading/paper_trading_{account_name}.json"
    
    if not os.path.exists(state_file):
        logger.info(f"No existing paper trading state found for {account_name}, using defaults")
        return {
            'initial_balance': 10000,
            'current_balance': 10000,
            'total_pnl': 0,
            'positions': {}
        }
    
    try:
        with open(state_file, 'r') as f:
            state_data = json.load(f)
        
        logger.info(f"Loaded paper trading state from {state_file}")
        return state_data.get('balance_info', {})
    except Exception as e:
        logger.error(f"Failed to load paper trading state: {e}")
        return {
            'initial_balance': 10000,
            'current_balance': 10000,
            'total_pnl': 0,
            'positions': {}
        }

def reset_paper_trading_state(account_name: str, initial_balance: float = 10000) -> dict:
    """
    Reset paper trading state to initial values.
    
    Args:
        account_name: Name of the trading account
        initial_balance: Initial balance to set
        
    Returns:
        dict: Reset balance information
    """
    reset_state = {
        'initial_balance': initial_balance,
        'current_balance': initial_balance,
        'current_portfolio_value': initial_balance,
        'total_pnl': 0,
        'positions': {}
    }
    
    save_paper_trading_state(account_name, reset_state)
    logger.info(f"Paper trading state reset for {account_name} with balance ${initial_balance:.2f}")
    return reset_state

def check_and_fix_corrupted_state(account_name: str, account_config: dict, client=None) -> dict:
    """
    Check for and fix corrupted paper trading state.
    
    Args:
        account_name: Name of the trading account
        account_config: Account configuration
        
    Returns:
        dict: Fixed balance information
    """
    state = load_paper_trading_state(account_name)
    
    # Check for corruption indicators
    current_balance = state.get('current_balance', 0)
    initial_balance = state.get('initial_balance', account_config.get('paper_trading_balance', 10000))
    positions = state.get('positions', {})
    
    # Calculate actual portfolio value
    portfolio_value = calculate_portfolio_value(current_balance, positions, account_config, client)
    
    # If balance is severely negative or portfolio value calculation fails
    if current_balance < -initial_balance * 2 or portfolio_value <= 0:
        logger.warning(f"Corrupted paper trading state detected for {account_name}")
        logger.warning(f"Current balance: ${current_balance:.2f}, Portfolio value: ${portfolio_value:.2f}")
        logger.info("Resetting to initial state...")
        return reset_paper_trading_state(account_name, initial_balance)
    
    return state

def initialize_paper_trading(account_name: str, account_config: dict, client=None) -> dict:
    """
    Initialize paper trading state for the account with corruption checking.
    
    Args:
        account_name: Name of the trading account
        account_config: Account configuration
        
    Returns:
        dict: Initialized balance information
    """
    initial_balance = account_config.get('paper_trading_balance', 10000)
    
    # Check if state exists, if not create it
    state_file = f"data/paper_trading/paper_trading_{account_name}.json"
    if not os.path.exists(state_file):
        logger.info(f"Initializing paper trading for {account_name} with balance ${initial_balance:.2f}")
        return reset_paper_trading_state(account_name, initial_balance)
    else:
        # Load and check for corruption
        return check_and_fix_corrupted_state(account_name, account_config, client)

# --- Market Simulation ---
def simulate_market_execution(order: dict, base_price: float, account_config: dict) -> dict:
    """
    Simulate realistic market execution for paper trading.
    
    Args:
        order: Order dictionary with Symbol, Action, Adjusted Quantity, etc.
        client: Binance client for price data
        account_config: Account configuration
        
    Returns:
        dict: Simulated execution result
    """
    # Import here to avoid circular imports
    from .binance import get_price
    
    # Get current market price
    stable_coin = account_config.get('stable_coin_symbol', 'USDC')
    
    if base_price is None:
        raise ValueError(f"Cannot get price for {order['Asset']}")
    
    # Simulate bid-ask spread
    spread = PAPER_TRADING_SPREAD
    if order['Action'].lower() == 'buy':
        # Buy at ask price (higher)
        execution_price = base_price * (1 + spread/2)
    else:
        # Sell at bid price (lower)
        execution_price = base_price * (1 - spread/2)
    
    # Simulate slippage based on order size
    quantity = order['Adjusted Quantity']
    price_impact = min(quantity * 0.0001, 0.002)  # Max 0.2% price impact
    
    if order['Action'].lower() == 'buy':
        execution_price *= (1 + price_impact)
    else:
        execution_price *= (1 - price_impact)
    
    # Add random slippage
    slippage = random.uniform(-PAPER_TRADING_SLIPPAGE, PAPER_TRADING_SLIPPAGE)
    execution_price *= (1 + slippage)
    
    # Simulate execution delay
    delay = random.uniform(PAPER_TRADING_MIN_FILL_DELAY, PAPER_TRADING_MAX_FILL_DELAY)
    time.sleep(delay)
    
    # Calculate commission
    commission_rate = account_config.get('transaction_costs', PAPER_TRADING_COMMISSION)
    commission = abs(quantity * execution_price * commission_rate)
    
    # Calculate net quantity (after commission)
    if order['Action'].lower() == 'buy':
        net_quantity = quantity
        net_value = quantity * execution_price + commission
    else:
        net_quantity = quantity
        net_value = quantity * execution_price - commission
    
    return {
        'execution_price': execution_price,
        'executed_quantity': net_quantity,
        'commission': commission,
        'execution_delay': delay,
        'slippage': slippage,
        'price_impact': price_impact,
        'net_value': net_value
    }

# --- Balance Management ---
def update_paper_trading_balance(account_name: str, account_config: dict, execution_results: list, client=None) -> dict:
    """
    Update paper trading balance based on executed orders with proper validation.
    
    Args:
        account_name: Name of the trading account
        account_config: Account configuration
        execution_results: List of execution results
        
    Returns:
        dict: Updated balance information
    """
    # Load existing state or use defaults
    existing_state = load_paper_trading_state(account_name)
    initial_balance = existing_state.get('initial_balance', account_config.get('paper_trading_balance', 10000))
    current_balance = existing_state.get('current_balance', initial_balance)
    positions = existing_state.get('positions', {})
    
    # Get leverage limits
    max_leverage = account_config.get('max_leverage', 1.0)
    
    for result in execution_results:
        order = result['order']
        execution = result['execution']
        asset = order['Asset']
        action = order['Action'].lower()
        
        if action == 'buy':
            cost = execution['net_value']
            
            # Check if we have enough buying power (including leverage)
            current_portfolio_value = calculate_portfolio_value(current_balance, positions, account_config, client)
            max_buying_power = current_portfolio_value * max_leverage
            
            if cost > current_balance and (current_balance + cost) > max_buying_power:
                logger.warning(f"Order would exceed maximum leverage ({max_leverage}x). Skipping trade.")
                continue
                
            # Deduct from balance
            current_balance -= cost
            
            # Add to positions
            if asset not in positions:
                positions[asset] = {'quantity': 0, 'total_cost': 0}
            positions[asset]['quantity'] += execution['executed_quantity']
            positions[asset]['total_cost'] += cost
            
        elif action == 'sell':
            # Add to balance
            proceeds = execution['net_value']
            current_balance += proceeds
            
            # Reduce positions
            if asset not in positions:
                positions[asset] = {'quantity': 0, 'total_cost': 0}
            
            # Calculate proportional cost reduction
            if positions[asset]['quantity'] > 0:
                cost_reduction_ratio = execution['executed_quantity'] / positions[asset]['quantity']
                positions[asset]['total_cost'] *= (1 - cost_reduction_ratio)
            
            positions[asset]['quantity'] -= execution['executed_quantity']
            if positions[asset]['quantity'] <= 0:
                positions[asset]['quantity'] = 0
                positions[asset]['total_cost'] = 0
    
    # Calculate proper PnL including position values
    current_portfolio_value = calculate_portfolio_value(current_balance, positions, account_config, client)
    total_pnl = current_portfolio_value - initial_balance
    
    return {
        'initial_balance': initial_balance,
        'current_balance': current_balance,
        'current_portfolio_value': current_portfolio_value,
        'total_pnl': total_pnl,
        'positions': positions
    }

# --- Paper Trading Execution ---
def paper_order_executor(order: dict, base_price: float, account_config: dict, account_name: str, client=None) -> dict:
    """
    Simulate order execution and return a Binance-like response dictionary, including updated balance info.

    Args:
        order: Order dict with keys: Symbol, Action, Adjusted Quantity, Asset
        base_price: Base price of the asset
        account_config: Account config dict
        account_name: Name of the trading account

    Returns:
        dict: Simulated response formatted like Binance API, with updated balance info
    """
    import uuid

    # Simulate market execution
    execution = simulate_market_execution(order, base_price, account_config)

    # Build response to mimic Binance's `order_market_*` structure
    fills = [{
        'price': f"{execution['execution_price']:.8f}",
        'qty': f"{order['Adjusted Quantity']:.8f}",
        'commission': f"{execution['commission']:.8f}",
        'commissionAsset': account_config.get('stable_coin_symbol', 'USDC'),
        'tradeId': int(str(uuid.uuid4().int)[-9:])  # random trade ID
    }]

    # Update balance info
    execution_result = {'order': order, 'execution': execution}
    updated_balance_info = update_paper_trading_balance(account_name, account_config, [execution_result], client)
    save_paper_trading_state(account_name, updated_balance_info)

    response = {
        'symbol': order['Symbol'],
        'orderId': int(str(uuid.uuid4().int)[-9:]),
        'clientOrderId': f"paper_{order['Symbol']}_{uuid.uuid4().hex[:6]}",
        'transactTime': int(pd.Timestamp.now().timestamp() * 1000),
        'fills': fills,
        'updated_balance_info': updated_balance_info
    }

    return response

def calculate_portfolio_value(cash_balance: float, positions: dict, account_config: dict, client=None) -> float:
    """
    Calculate total portfolio value including cash and position values.
    
    Args:
        cash_balance: Current cash balance
        positions: Position dictionary
        account_config: Account configuration
        client: Binance client (optional, if None uses fallback pricing)
        
    Returns:
        float: Total portfolio value
    """
    from .binance import get_price
    
    stable_coin = account_config.get('stable_coin_symbol', 'USDC')
    total_value = max(0, cash_balance)  # Don't count negative cash as value
    
    for asset, pos in positions.items():
        if isinstance(pos, dict) and pos.get('quantity', 0) > 0:
            try:
                if client is not None:
                    price = get_price(client, asset, stable_coin)
                else:
                    # Fallback: try to get price without client (for paper trading compatibility)
                    price = None
                    logger.warning(f"No client provided for price lookup of {asset}, using cost basis")
                
                if price is not None:
                    total_value += pos['quantity'] * price
                else:
                    logger.warning(f"Could not get price for {asset}, using cost basis")
                    total_value += pos.get('total_cost', 0)
            except Exception as e:
                logger.warning(f"Error getting price for {asset}: {e}")
                total_value += pos.get('total_cost', 0)
    
    return total_value

def get_paper_trading_holdings(account_name: str, stable_coin_symbol: str = 'USDC') -> Dict[str, float]:
    """
    Fetch current paper trading holdings as a dict of asset: amount.
    Uses the stable_coin_symbol for cash.
    """
    state = load_paper_trading_state(account_name)
    balance_info = state.get('balance_info', state)  # Handle both old and new formats
    positions = balance_info.get('positions', {})
    holdings = {asset: pos['quantity'] if isinstance(pos, dict) and 'quantity' in pos else float(pos)
                for asset, pos in positions.items() if pos and (isinstance(pos, dict) and pos.get('quantity', 0) != 0 or isinstance(pos, (int, float)) and pos != 0)}
    
    # Get cash balance from the correct location
    cash = balance_info.get('current_balance', 0.0)
    if cash > 0:  # Only add positive cash to holdings
        holdings[stable_coin_symbol] = cash
    
    logger.info(f"Paper trading holdings for {account_name}: {holdings}")
    return holdings
