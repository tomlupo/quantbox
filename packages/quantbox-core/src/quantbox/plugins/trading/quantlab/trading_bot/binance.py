# --- Imports ---
from binance.client import Client
from binance.exceptions import BinanceAPIException
from typing import Dict, Any, Optional, Union
import time
import ccxt
from .. import utils

logger = utils.get_logger()

# --- Binance Client Initialization ---
def get_binance_client(api_key: str, api_secret: str) -> Client:
    """Initialize and return a Binance client instance."""
    return Client(api_key, api_secret)

# --- Order Placement ---
def place_market_order(client: Client, symbol: str, side: str, quantity: Union[float, str]) -> Dict[str, Any]:
    """Place a market order (buy/sell) and return the response."""
    try:
        # Convert quantity to string to preserve precision and avoid scientific notation
        quantity_str = str(quantity) if isinstance(quantity, (int, float)) else quantity
        
        if side.upper() == 'BUY':
            return client.order_market_buy(symbol=symbol, quantity=quantity_str)
        elif side.upper() == 'SELL':
            return client.order_market_sell(symbol=symbol, quantity=quantity_str)
        else:
            raise ValueError(f"Invalid order side: {side}")
    except BinanceAPIException as e:
        logger.error(f"Binance API error placing order: {e}")
        return {'error': str(e)}
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return {'error': str(e)}

# --- Account Information ---
def get_account_info(client: Client, recv_window: int = 60000) -> Dict[str, Any]:
    """Fetch account information from Binance."""
    try:
        return client.get_account(recvWindow=recv_window)
    except Exception as e:
        logger.error(f"Error fetching account info: {e}")
        return {}

# --- Symbol Info ---
def get_symbol_info(client: Client, symbol: str) -> Optional[Dict[str, Any]]:
    """Get symbol info (lot size, min notional, etc)."""
    try:
        return client.get_symbol_info(symbol)
    except Exception as e:
        logger.error(f"Error fetching symbol info for {symbol}: {e}")
        return None

# --- Price Fetching ---
_TRANSIENT_API_CODES = {0, -1003}  # 0 = invalid JSON (HTML 503 page), -1003 = rate limit

def _is_transient_error(e: Exception) -> bool:
    """Check if an exception is a transient error worth retrying."""
    if isinstance(e, BinanceAPIException):
        return e.code in _TRANSIENT_API_CODES
    # Connection errors, timeouts, etc.
    return isinstance(e, (ConnectionError, TimeoutError, OSError))

def get_price(client: Client, asset: str, stable_coin_symbol: str = 'USDC', max_retries: int = 3) -> Optional[float]:
    """
    Get the price of the asset against the stable coin symbol (e.g., USDC).
    If not available, try to get price via BTC as an intermediate.
    Retries on transient errors (503/rate limit) with exponential backoff.
    Returns None if price cannot be determined.
    """
    for attempt in range(1, max_retries + 1):
        try:
            symbol = f'{asset}{stable_coin_symbol}'
            response = client.get_symbol_ticker(symbol=symbol)
            if response:
                return float(response['price'])
            return None
        except BinanceAPIException as e:
            if e.code == -1121:
                # Permanent error: no such symbol pair — try BTC route (no retry)
                logger.info(f"No {stable_coin_symbol} pair for {asset}, trying BTC pair...")
                try:
                    btc_price = float(client.get_symbol_ticker(symbol=f'BTC{stable_coin_symbol}')['price'])
                    asset_btc_price = float(client.get_symbol_ticker(symbol=f'{asset}BTC')['price'])
                    return asset_btc_price * btc_price
                except BinanceAPIException:
                    logger.warning(f"No valid market for {asset}, skipping...")
                    return None
            elif _is_transient_error(e) and attempt < max_retries:
                backoff = 2 ** (attempt - 1)  # 1s, 2s, 4s
                logger.warning(f"Transient error fetching price for {asset} (code={e.code}), "
                               f"retrying {attempt}/{max_retries} in {backoff}s...")
                time.sleep(backoff)
                continue
            else:
                logger.error(f"Binance API error for {asset}: {e}")
                return None
        except Exception as e:
            if _is_transient_error(e) and attempt < max_retries:
                backoff = 2 ** (attempt - 1)
                logger.warning(f"Transient error fetching price for {asset} ({type(e).__name__}), "
                               f"retrying {attempt}/{max_retries} in {backoff}s...")
                time.sleep(backoff)
                continue
            logger.error(f"Error fetching price for {asset}: {e}")
            return None
    logger.error(f"Failed to fetch price for {asset} after {max_retries} attempts")
    return None

 