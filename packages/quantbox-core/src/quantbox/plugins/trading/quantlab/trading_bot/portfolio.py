# --- Imports ---
from typing import Dict, Callable, Optional
from .. import utils

logger = utils.get_logger()

# --- Portfolio Utilities ---
def get_current_holdings(client=None, account_config: Optional[dict] = None) -> Dict[str, float]:
    """
    Fetch current holdings as a dict of asset: amount.
    Handles both live and paper trading modes.
    Args:
        client: Binance client (required for live trading)
        account_config: Account configuration dict (must include 'paper_trading' key)
    Returns:
        Dict[str, float]: Holdings as asset: amount
    """
    if account_config and account_config.get('paper_trading', False):
        account_name = account_config.get('account_name')
        if not account_name:
            raise ValueError("account_name must be provided for paper trading mode.")
        from .paper_trading import get_paper_trading_holdings
        stable_coin_symbol = account_config.get('stable_coin_symbol', 'USDC')
        return get_paper_trading_holdings(account_name, stable_coin_symbol)
    else:
        if client is None:
            raise ValueError("client must be provided for live trading mode.")
        holdings = {}
        # Use centralized function from binance module
        from .binance import get_account_info
        account_info = get_account_info(client)
        for item in account_info.get('balances', []):
            amount = float(item['free'])
            if amount != 0:
                holdings[item['asset']] = amount
        logger.info(f"Fetched current holdings: {holdings}")
        return holdings

def get_cash_available(holdings: Dict[str, float], stable_coin_symbol: str = 'USDC') -> float:
    """
    Calculate cash available in stablecoin (e.g., USDC).
    """
    return holdings.get(stable_coin_symbol, 0.0)

def format_holdings_for_report(holdings: Dict[str, float]) -> str:
    """Format holdings for reporting or messaging."""
    lines = [f"{asset}: {amount:.8f}" for asset, amount in holdings.items()]
    return '\n'.join(lines)

# --- New: Portfolio Value Calculation ---
def get_portfolio_value(holdings: Dict[str, float], get_price: Callable[[str], float], stable_coin_symbol: str = 'USDC', exclusions=None) -> float:
    """
    Calculate total portfolio value in stablecoin (e.g., USDC).
    Excludes specified assets (e.g., dust, non-tradables).
    """
    if exclusions is None:
        exclusions = []
    total = 0.0
    
    for asset, amount in holdings.items():
        if asset == stable_coin_symbol:
            # Add stablecoin balance directly, but only if positive
            total += max(0, amount)
        elif asset in exclusions:
            continue
        else:
            price = get_price(asset)
            if price is not None and amount > 0:
                total += amount * price
    
    logger.info(f"Calculated portfolio value: {total:.2f}")
    return total

# --- New: Target Allocation Calculation ---
def get_target_positions(strategy_weights: Dict[str, float], total_value: float, get_price: Callable[[str], float], stable_coin_symbol: str = 'USDC') -> Dict[str, float]:
    """
    Given strategy weights and total portfolio value, compute target quantity for each asset.
    """
    targets = {}
    for asset, weight in strategy_weights.items():
        price = get_price(asset)
        if price is not None and price > 0:
            targets[asset] = (total_value * weight) / price
    logger.info(f"Calculated target allocations: {targets}")
    return targets 