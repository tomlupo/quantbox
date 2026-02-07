# --- Imports ---
import os
import json
import time
from datetime import datetime, timezone
from decimal import Decimal, getcontext, ROUND_DOWN
from typing import Tuple, Dict, Any, List, Callable, Optional
import pandas as pd
from .. import utils

logger = utils.get_logger()

# --- Constants ---
DEFAULT_CAPITAL_AT_RISK = 1.0
DEFAULT_MIN_NOTIONAL = 1.0
DEFUALT_MIN_TRADE_SIZE = 0.01
DEFAULT_STABLE_COIN_SYMBOL = 'USDC'

# --- Order Utilities ---
def adjust_quantity(qty: float, step_size: float) -> float:
    """Adjust the quantity to conform to the step size requirement, ensuring correct decimal places."""
    step_size_str = f"{step_size:.8f}"
    decimal_places = step_size_str.rstrip('0').split('.')[-1]
    precision = len(decimal_places)
    getcontext().rounding = ROUND_DOWN
    adjusted_qty = Decimal(qty).quantize(Decimal('1.' + '0' * precision))
    return float(adjusted_qty)

def get_lot_size_and_min_notional(symbol_info: dict) -> Tuple[float, float, float]:
    """Extract min_qty, step_size, min_notional from symbol info."""
    min_qty, step_size, min_notional = 0, 0, 0
    if symbol_info:
        for f in symbol_info.get('filters', []):
            if f['filterType'] == 'LOT_SIZE':
                min_qty = float(f['minQty'])
                step_size = float(f['stepSize'])
            if f['filterType'] == 'NOTIONAL':
                min_notional = float(f['minNotional'])
    return min_qty, step_size, min_notional

# --- Portfolio-based Order Generation ---
def generate_portfolio_orders(account_config: dict, strategy_weights: dict, client) -> dict:
    """
    Generate portfolio orders based on strategy weights and current holdings.
    
    Args:
        account_config: Account configuration
        strategy_weights: Strategy weights dictionary
        client: Binance client
        
    Returns:
        dict: Portfolio orders with analysis
    """
    # Import here to avoid circular imports
    from .portfolio import get_current_holdings, get_portfolio_value, get_target_positions, get_cash_available

    stable_coin_symbol = account_config.get('stable_coin_symbol', DEFAULT_STABLE_COIN_SYMBOL)
    capital_at_risk = account_config.get('capital_at_risk', DEFAULT_CAPITAL_AT_RISK)
    min_notional = account_config.get('min_notional', DEFAULT_MIN_NOTIONAL)
    min_trade_size = account_config.get('min_trade_size', DEFUALT_MIN_TRADE_SIZE)

    exclusions = ['ETHW', 'BETH', stable_coin_symbol]
    # Add symbols from config's not-tradable list if present
    try:
        not_tradable_config = account_config.get('not_tradable_on_binance', []) or []
        exclusions += [item['symbol'] for item in not_tradable_config if isinstance(item, dict) and item.get('symbol')]
    except Exception:
        pass
    exclusions = list(set(exclusions))

    # --- TTL-based price cache to avoid redundant API calls ---
    PRICE_CACHE_TTL = 30  # seconds
    _price_cache: Dict[str, tuple] = {}

    def cached_get_price(asset: str) -> Optional[float]:
        now = time.time()
        if asset in _price_cache:
            cached_price, cached_at = _price_cache[asset]
            if now - cached_at < PRICE_CACHE_TTL:
                return cached_price
        price = get_price(client, asset, stable_coin_symbol)
        _price_cache[asset] = (price, now)
        return price

    # 1. Fetch current holdings
    current_holdings = get_current_holdings(client=client, account_config=account_config)

    # 2. Compute total portfolio value with enhanced error handling
    total_value = get_portfolio_value(current_holdings, cached_get_price, stable_coin_symbol, exclusions)
    cash_available = get_cash_available(current_holdings, stable_coin_symbol)

    if total_value <= 0:
        logger.error("Portfolio value is zero or negative, cannot proceed with trading")
        return {
            'orders': pd.DataFrame(),
            'rebalancing': pd.DataFrame(),
            'total_value': total_value,
        }

    # 3. Apply capital at risk to strategy weights
    adjusted_weights = {asset: weight * capital_at_risk for asset, weight in strategy_weights.items()}

    # 4. Compute target allocations
    target_positions = get_target_positions(adjusted_weights, total_value, cached_get_price, stable_coin_symbol)

    # 5. Build rebalancing DataFrame with enhanced analysis
    rebalancing_df = generate_rebalancing_dataframe(current_holdings, target_positions, cached_get_price, total_value, adjusted_weights, exclusions)

    # 6. Price completeness diagnostics
    all_assets = sorted(set(current_holdings.keys()) | set(adjusted_weights.keys()))
    all_assets = [a for a in all_assets if a not in exclusions]
    fetched_ok = [a for a in all_assets if a in _price_cache and _price_cache[a][0] is not None]
    fetched_fail = [a for a in all_assets if a in _price_cache and _price_cache[a][0] is None]
    not_fetched = [a for a in all_assets if a not in _price_cache]
    logger.info(f"Price fetch summary: {len(fetched_ok)}/{len(all_assets)} succeeded, "
                f"{len(fetched_fail)} failed, {len(not_fetched)} not requested")
    if fetched_fail:
        held_fails = [a for a in fetched_fail if current_holdings.get(a, 0) > 0]
        if held_fails:
            logger.warning(f"HELD assets with failed price fetch: {held_fails} — portfolio value may be understated")
        else:
            logger.info(f"Assets with failed price fetch (not currently held): {fetched_fail}")

    # 7. Generate orders
    orders_df = generate_orders_from_rebalancing(rebalancing_df, client, stable_coin_symbol, min_notional, min_trade_size, cash_available)
    return {
        'orders': orders_df,
        'rebalancing': rebalancing_df,
        'total_value': total_value,
    }

# --- Rebalancing DataFrame ---
def generate_rebalancing_dataframe(current_holdings: Dict[str, float], target_allocations: Dict[str, float], get_price: Callable, total_value: float, strategy_weights: Dict[str, float], exclusions: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Build a DataFrame with current, target, delta, and value columns for each asset.
    Columns:
        - Asset
        - Current Quantity
        - Current Value
        - Current Weight
        - Target Quantity
        - Target Value
        - Target Weight
        - Weight Delta
        - Delta Quantity
        - Price
        - Trade Action
    """
    assets = sorted(set(current_holdings.keys()) | set(target_allocations.keys()))
    if exclusions:
        assets = [a for a in assets if a not in exclusions]
    data = []
    for asset in assets:
        current_qty = current_holdings.get(asset, 0.0)
        price = get_price(asset)
        current_value = current_qty * price if price else 0
        target_qty = target_allocations.get(asset, 0.0)
        target_value = target_qty * price if price else 0
        current_weight = current_value / total_value if total_value > 0 else 0
        target_weight = target_value / total_value if total_value > 0 else 0
        weight_delta = strategy_weights.get(asset, 0) - current_weight
        delta_qty = target_qty - current_qty
        if delta_qty > 0:
            trade_action = "Buy"
        elif delta_qty < 0:
            trade_action = "Sell"
        else:
            trade_action = "Hold"
        data.append({
            'Asset': asset,
            'Current Quantity': current_qty,
            'Current Value': current_value,
            'Current Weight': current_weight,
            'Target Quantity': target_qty,
            'Target Value': target_value,
            'Target Weight': target_weight,
            'Weight Delta': weight_delta,
            'Delta Quantity': delta_qty,
            'Price': price,
            'Trade Action': trade_action
        })
    df = pd.DataFrame(data)
    logger.info(f"Generated rebalancing DataFrame shape: {df.shape}")
    return df

# --- Order Generation from Rebalancing ---
def generate_orders_from_rebalancing(
    df: pd.DataFrame,
    client,
    stable_coin_symbol: str = 'USDC',
    min_notional_default: float = 1.0,
    min_trade_size: float = 0.01,
    cash_available: float = 0.0,
    scalling_factor_min: float = 0.9
) -> pd.DataFrame:
    """
    Given the rebalancing DataFrame, generate buy/sell orders, adjusting for lot size and min notional.
    Tracks all considered orders, including those dropped for threshold, min notional, min qty, etc.
    Adds columns: Adjusted Quantity (0 for dropped), Order Status, and Reason.
    Returns a DataFrame with all considered orders and their status.
    Only actionable orders (Adjusted Quantity > 0 and status 'To be placed') should be used for execution.
    """
    order_records = []
    # --- Step 0: Consider all rows in df ---
    for _, row in df.iterrows():
        asset = row['Asset']
        symbol = f"{asset}{stable_coin_symbol}"
        action = row['Trade Action'].lower()
        delta_qty = row['Delta Quantity']
        price = row['Price']
        status = None
        reason = None
        adjusted_qty = 0.0
        notional_value = 0.0
        min_qty = None
        step_size = None
        min_notional = None
        scaling_factor = None
        # Only consider buy/sell, not hold
        if action == 'hold' or delta_qty == 0:
            status = 'Zero delta'
            reason = 'No trade needed'
        else:
            # Allow zero-target sells to bypass the weight-delta threshold
            zero_target_and_sell = (action == 'sell') and (row.get('Target Weight', 0) == 0) and (row.get('Current Quantity', 0) > 0)
            if (abs(row['Weight Delta']) < min_trade_size) and (not zero_target_and_sell):
                status = 'Below threshold'
                reason = f'abs(weight delta) < {min_trade_size}'
            else:
                # Use centralized function from binance module
                from .binance import get_symbol_info
                symbol_info = get_symbol_info(client, symbol)
                min_qty, step_size, min_notional = get_lot_size_and_min_notional(symbol_info)
                if min_notional == 0:
                    min_notional = min_notional_default
                if action == 'sell':
                    adjusted_qty = adjust_quantity(abs(delta_qty), step_size) if step_size else 0.0
                    notional_value = adjusted_qty * price if price else 0.0
                    if price is None or price == 0:
                        status = 'Zero price'
                        reason = 'No price available'
                        adjusted_qty = 0.0
                    elif notional_value < min_notional:
                        status = 'Below min notional'
                        reason = f'Notional {notional_value:.4f} < min_notional {min_notional:.4f}'
                        adjusted_qty = 0.0
                    elif adjusted_qty < min_qty:
                        status = 'Below min qty'
                        reason = f'Qty {adjusted_qty:.8f} < min_qty {min_qty:.8f}'
                        adjusted_qty = 0.0
                    else:
                        status = 'To be placed'
                        reason = ''
                elif action == 'buy':
                    # For buys, scaling is applied after all sells are processed, so we mark for now
                    # We'll fill in scaling_factor and adjusted_qty after all buys are collected
                    status = 'Pending scaling'
                    reason = ''
        order_records.append({
            'Asset': asset,
            'Symbol': symbol,
            'Action': action.capitalize(),
            'Raw Quantity': abs(delta_qty),
            'Adjusted Quantity': adjusted_qty,
            'Notional Value': notional_value,
            'Price': price,
            'Min Notional': min_notional,
            'Min Qty': min_qty,
            'Step Size': step_size,
            'Order Status': status,
            'Reason': reason,
            'Scaling Factor': scaling_factor
        })
    # --- Step 1: Process buys for scaling ---
    # Find all buys with status 'Pending scaling'
    buy_indices = [i for i, rec in enumerate(order_records) if rec['Action'] == 'Buy' and rec['Order Status'] == 'Pending scaling']
    total_buy_value = sum(abs(order_records[i]['Raw Quantity']) * order_records[i]['Price'] for i in buy_indices if order_records[i]['Price'])
    # Calculate available cash from sells
    cash_from_sells = sum(rec['Notional Value'] for rec in order_records if rec['Action'] == 'Sell' and rec['Order Status'] == 'To be placed')
    cash_available = cash_available + cash_from_sells

    scaling_factor = min(1.0, cash_available / total_buy_value) if total_buy_value > 0 else 0.0
    if total_buy_value == 0:
        logger.info("No buy orders to scale")
    elif scaling_factor < scalling_factor_min:
        logger.warning(f"Buy orders scaling factor {scaling_factor} is less than the minimum {scalling_factor_min}.")
    else:
        logger.info(f"Buy orders scaling factor {scaling_factor} is greater than the minimum {scalling_factor_min}.")
        for i in buy_indices:
            rec = order_records[i]
            price = rec['Price']
            step_size = rec['Step Size']
            min_qty = rec['Min Qty']
            min_notional = rec['Min Notional']
            raw_qty = rec['Raw Quantity']
            scaled_qty = adjust_quantity(raw_qty * scaling_factor, step_size) if step_size and price else 0.0
            scaled_notional = scaled_qty * price if price else 0.0
            rec['Scaling Factor'] = scaling_factor
            rec['Adjusted Quantity'] = scaled_qty
            rec['Notional Value'] = scaled_notional
            # Now check constraints
            if price is None or price == 0:
                rec['Order Status'] = 'Zero price'
                rec['Reason'] = 'No price available'
                rec['Adjusted Quantity'] = 0.0
            elif scaled_notional < min_notional:
                rec['Order Status'] = 'Below min notional'
                rec['Reason'] = f'Notional {scaled_notional:.4f} < min_notional {min_notional:.4f}'
                rec['Adjusted Quantity'] = 0.0
            elif scaled_qty < min_qty:
                rec['Order Status'] = 'Below min qty'
                rec['Reason'] = f'Qty {scaled_qty:.8f} < min_qty {min_qty:.8f}'
                rec['Adjusted Quantity'] = 0.0
            elif scaled_qty == 0:
                rec['Order Status'] = 'Zero quantity'
                rec['Reason'] = 'Scaled quantity is zero'
                rec['Adjusted Quantity'] = 0.0
            else:
                rec['Order Status'] = 'To be placed'
                rec['Reason'] = ''
    # --- Step 2: Return as DataFrame ---
    columns = [
    'Asset', 'Symbol', 'Action', 'Raw Quantity', 'Adjusted Quantity', 'Price',
    'Notional Value', 'Min Notional', 'Min Qty', 'Step Size', 'Scaling Factor',
    'Order Status', 'Reason']
    order_df = pd.DataFrame(order_records, columns=columns)
    order_df['Executable'] = (order_df['Adjusted Quantity'] > 0) & (order_df['Order Status'] == 'To be placed')
    logger.info(f"Generated order DataFrame shape: {order_df.shape}")
    return order_df

# --- Order Execution Orchestration ---
def execute_orders(orders: pd.DataFrame, account_name: str, account_config: dict, client) -> dict:
    from .binance import get_price
    from binance.exceptions import BinanceAPIException
    from .paper_trading import paper_order_executor

    paper_trading = account_config.get('paper_trading', True)
    trading_enabled = account_config.get('trading_enabled', False)
    stable_coin_symbol = account_config.get('stable_coin_symbol', 'USDC')

    execution_report = {
        'executed_orders': [],
        'failed_orders': [],
        'summary': {
            'total_executed': 0,
            'total_failed': 0,
            'total_value': 0
        },
        'paper_trading': paper_trading,
        'trading_enabled': trading_enabled,
        'orders_details': []
    }

    if orders is None or not trading_enabled or orders.empty:
        logger.warning("No orders to execute or trading is disabled")
        return execution_report

    orders_df = orders.query('Executable')
    if orders_df.empty:
        logger.info("No executable orders to execute")
        return execution_report

    orders_df = orders_df.sort_values(by='Action', ascending=False)
    logger.info(f"Order execution sequence: {orders_df[['Symbol', 'Action', 'Adjusted Quantity']].to_dict('records')}")

    for index, order in orders_df.iterrows():
        result = {
            'Symbol': order.get('Symbol'),
            'Action': order.get('Action'),
            'Adjusted Quantity': order.get('Adjusted Quantity'),
            'Asset': order.get('Asset'),
            'placed_at': pd.Timestamp.now()
        }

        try:
            symbol = result['Symbol']
            side = result['Action'].upper()
            qty = result['Adjusted Quantity']
            asset = result['Asset']
            # Format quantity to avoid scientific notation and ensure proper decimal format for Binance API
            # Binance requires format: ^([0-9]{1,20})(\.[0-9]{1,20})?$ (no scientific notation)
            if qty == 0:
                qty_str = "0"
            else:
                # Use f-string with fixed precision to avoid scientific notation
                qty_str = f"{qty:.8f}".rstrip('0').rstrip('.')
                # Ensure we don't end up with empty string for very small numbers
                if not qty_str or qty_str == '.':
                    qty_str = "0"

            price = get_price(client, asset, stable_coin_symbol)
            if price is None:
                raise ValueError(f"Cannot fetch market price for asset: {asset}")
            result['reference_price'] = price

            if paper_trading:
                response = paper_order_executor(order, price, account_config, account_name, client)
            else:
                # Use centralized functions from binance module
                from .binance import place_market_order
                response = place_market_order(client, symbol, side.lower(), qty_str)
                response['_execution_mode'] = 'REAL'

            result['executed_at'] = pd.Timestamp.now()
            result['status'] = 'FILLED'
            result['execution_mode'] = response.get('_execution_mode')
            result['exchange_response'] = response

            fills = response.get('fills', [])
            if fills:
                fill = fills[0]
                fill_price = float(fill.get('price', 0))
                fill_qty = float(fill.get('qty', 0))
                fill_commission = float(fill.get('commission', 0))
                commission_asset = fill.get('commissionAsset')

                result.update({
                    'executed_price': fill_price,
                    'executed_quantity': fill_qty,
                    'commission_paid': fill_commission,
                    'commission_asset': commission_asset,
                    'spread': abs(fill_price - price),
                    'spread_pct': abs(fill_price - price) / price if price else None,
                    'commission_pct': (fill_commission / (fill_price * fill_qty)) if fill_price and fill_qty else None
                })

                result['total_cost_pct'] = (
                    (result.get('spread_pct') or 0) + (result.get('commission_pct') or 0)
                )

                logger.info(
                    f"Trade executed: {symbol} {side} {qty_str} @ {fill_price:.2f} "
                    f"(spread: {result['spread_pct']:.4%}, commission: {result['commission_pct']:.4%})"
                )
            else:
                logger.warning(f"No fills returned for {symbol}, cannot compute analytics.")

            execution_report['summary']['total_executed'] += 1

        except (BinanceAPIException, Exception) as e:
            result['status'] = 'FAILED'
            result['error'] = str(e)
            result['failed_at'] = pd.Timestamp.now()
            logger.error(f"Order execution failed for {order.get('Symbol', '')}: {e}")
            execution_report['summary']['total_failed'] += 1

        execution_report['orders_details'].append(result)

    execution_report['summary']['total_value'] = sum(
        r.get('executed_quantity', 0) * r.get('executed_price', 1)
        for r in execution_report['orders_details']
        if r.get('status') == 'FILLED'
    )

    # Add total_cost: sum of commission_paid + abs(executed_price - reference_price) * executed_quantity for all executed orders
    execution_report['summary']['total_cost'] = sum(
        (float(r.get('commission_paid', 0)) if r.get('commission_paid') is not None else 0)
        + (abs(float(r.get('executed_price', 0)) - float(r.get('reference_price', 0))) * float(r.get('executed_quantity', 0))
           if r.get('executed_price') is not None and r.get('reference_price') is not None and r.get('executed_quantity') is not None else 0)
        for r in execution_report['orders_details']
        if r.get('status') == 'FILLED'
    )

    logger.info(f"Execution complete: {execution_report['summary']['total_executed']} executed, "
                f"{execution_report['summary']['total_failed']} failed")

    save_trade_history(execution_report, account_name)
    return execution_report


def save_trade_history(execution_report: dict, account_name: str) -> Optional[str]:
    """
    Persist trade execution details to data/{account_name}/trades/.
    Only saves when there are executed orders.
    Returns the file path if saved, None otherwise.
    """
    orders_details = execution_report.get('orders_details', [])
    executed = [o for o in orders_details if o.get('status') == 'FILLED']

    if not executed:
        logger.info("No executed trades to save")
        return None

    # Fields to persist per trade
    persist_fields = [
        'Symbol', 'Action', 'executed_price', 'executed_quantity',
        'commission_paid', 'commission_asset', 'placed_at', 'executed_at',
        'status', 'execution_mode', 'spread_pct', 'total_cost_pct',
    ]

    trades = []
    for order in executed:
        trade = {}
        for field in persist_fields:
            val = order.get(field)
            # Convert Timestamps to ISO strings for JSON serialization
            if isinstance(val, pd.Timestamp):
                val = val.isoformat()
            trade[field] = val
        trades.append(trade)

    # Build file path: data/{account_name}/trades/{date}--{HHMMSS}.json
    now_utc = datetime.now(timezone.utc)
    date_str = now_utc.strftime('%Y-%m-%d')
    time_str = now_utc.strftime('%H%M%S')

    trades_dir = os.path.join('data', account_name.lower(), 'trades')
    os.makedirs(trades_dir, exist_ok=True)

    filename = f"{date_str}--{time_str}.json"
    file_path = os.path.join(trades_dir, filename)

    with open(file_path, 'w') as f:
        json.dump({
            'date': date_str,
            'timestamp': now_utc.isoformat(),
            'paper_trading': execution_report.get('paper_trading', False),
            'summary': execution_report.get('summary', {}),
            'trades': trades,
        }, f, indent=2, default=str)

    logger.info(f"Saved {len(trades)} trade(s) to {file_path}")
    return file_path



# --- Helper function for price retrieval ---
def get_price(client, asset: str, stable_coin: str) -> float:
    """
    Get current price for an asset.
    
    Args:
        client: Binance client
        asset: Asset symbol
        stable_coin: Stable coin symbol
        
    Returns:
        float: Current price or None if not available
    """
    try:
        # Import here to avoid circular imports
        from .binance import get_price as binance_get_price
        return binance_get_price(client, asset, stable_coin)
    except Exception as e:
        logger.error(f"Failed to get price for {asset}: {e}")
        return None 