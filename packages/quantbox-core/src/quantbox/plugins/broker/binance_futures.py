"""
Binance Futures (USDM) Broker Plugin

Live trading on Binance USDⓈ-M Futures with:
- Real position management via FAPI
- Funding rate awareness
- Proper margin/leverage handling
- Position size validation
- Risk limits enforcement

## LLM Usage Guide

### Quick Start
```python
from quantbox.plugins.broker import BinanceFuturesBroker

broker = BinanceFuturesBroker(
    api_key="...",
    api_secret="...",
    target_leverage=1,  # Start conservative
)

# Check account
print(broker.get_balance())
print(broker.get_positions())

# Execute target weights
broker.rebalance_to_weights({"BTC": 0.5, "ETH": -0.3})
```

### Key Features
- Uses ccxt.binanceusdm for Futures API
- Handles LONG/SHORT positions natively
- Tracks funding rates
- Validates against Binance filters (lot size, min notional)
- Telegram notifications on trades

### Risk Controls (Built-in)
- max_position_pct: Max single position (default: 50%)
- max_gross_exposure: Max total exposure (default: 2.0x)
- max_daily_loss_pct: Stop trading if exceeded (default: 10%)
- min_order_notional: Binance minimum (auto-detected)
"""
from __future__ import annotations
import os
import json
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, getcontext
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd

from quantbox.contracts import PluginMeta

try:
    import ccxt
except ImportError:  # pragma: no cover
    ccxt = None  # type: ignore

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

QUOTE_CURRENCY = "USDT"
DEFAULT_LEVERAGE = 1
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


# ============================================================================
# Risk Configuration
# ============================================================================

@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_pct: float = 0.50  # Max 50% in single position
    max_gross_exposure: float = 2.0  # Max 200% gross
    max_daily_loss_pct: float = 0.10  # Stop at 10% daily loss
    min_order_notional: float = 5.0  # Binance minimum
    max_slippage_pct: float = 0.005  # 0.5% max slippage
    
    def describe(self) -> Dict[str, Any]:
        return {
            "max_position_pct": f"{self.max_position_pct:.0%}",
            "max_gross_exposure": f"{self.max_gross_exposure:.1f}x",
            "max_daily_loss_pct": f"{self.max_daily_loss_pct:.0%}",
            "min_order_notional": f"${self.min_order_notional}",
        }


# ============================================================================
# Telegram Notifications
# ============================================================================

def send_telegram(token: str, chat_id: str, message: str) -> bool:
    """Send a Telegram message."""
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
        }, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")
        return False


# ============================================================================
# Utility Functions
# ============================================================================

def adjust_quantity(qty: float, step_size: float, precision: int = 8) -> float:
    """Adjust quantity to Binance step size."""
    if step_size <= 0:
        return round(qty, precision)
    getcontext().rounding = ROUND_DOWN
    step_str = f"{step_size:.{precision}f}".rstrip('0')
    decimal_places = len(step_str.split('.')[-1]) if '.' in step_str else 0
    adjusted = Decimal(str(qty)).quantize(Decimal(f"1e-{decimal_places}"))
    return float(adjusted)


def adjust_price(price: float, tick_size: float) -> float:
    """Adjust price to Binance tick size."""
    if tick_size <= 0:
        return price
    return round(price / tick_size) * tick_size


# ============================================================================
# Broker Class
# ============================================================================

@dataclass
class BinanceFuturesBroker:
    """
    Binance USDⓈ-M Futures Broker.

    Handles real trading on Binance Futures with proper risk management.

    ## Quick Start
    ```python
    broker = BinanceFuturesBroker(api_key="...", api_secret="...")
    broker.rebalance_to_weights({"BTC": 0.5, "ETH": -0.3})
    ```
    """

    meta = PluginMeta(
        name="binance.futures.v1",
        kind="broker",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Binance USDM Futures broker (live) with leverage and short support",
        tags=("live", "futures", "binance"),
        capabilities=("live", "futures", "shorts", "leverage"),
        schema_version="v1",
    )

    # Credentials
    api_key: str = ""
    api_secret: str = ""
    
    # Trading config
    target_leverage: int = DEFAULT_LEVERAGE
    quote_currency: str = QUOTE_CURRENCY
    
    # Risk config
    risk: RiskConfig = field(default_factory=RiskConfig)
    
    # Notifications
    telegram_token: str = ""
    telegram_chat_id: str = ""
    
    # State
    _exchange: Optional[ccxt.binanceusdm] = field(default=None, repr=False)
    _markets: Dict = field(default_factory=dict, repr=False)
    _daily_pnl: float = 0.0
    _starting_balance: float = 0.0
    
    def __post_init__(self):
        """Initialize exchange connection."""
        # Load from env if not provided
        if not self.api_key:
            self.api_key = os.environ.get("API_KEY_BINANCE", "")
        if not self.api_secret:
            self.api_secret = os.environ.get("API_SECRET_BINANCE", "")
        if not self.telegram_token:
            self.telegram_token = os.environ.get("TELEGRAM_TOKEN", "")
        if not self.telegram_chat_id:
            self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API credentials required")
        
        self._init_exchange()
    
    def _init_exchange(self) -> None:
        """Initialize ccxt exchange."""
        self._exchange = ccxt.binanceusdm({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            }
        })
        
        # Load markets
        self._markets = self._exchange.load_markets()
        logger.info(f"Connected to Binance Futures, {len(self._markets)} markets loaded")
        
        # Record starting balance for daily PnL tracking
        balance = self.get_balance()
        self._starting_balance = balance.get('total', 0)
    
    def describe(self) -> Dict[str, Any]:
        """Describe broker for LLM introspection."""
        return {
            "name": "BinanceFuturesBroker",
            "exchange": "Binance USDⓈ-M Futures",
            "leverage": self.target_leverage,
            "risk_config": self.risk.describe(),
            "notifications": bool(self.telegram_token),
        }
    
    # ========================================================================
    # Account Methods
    # ========================================================================
    
    def get_balance(self) -> Dict[str, float]:
        """
        Get account balance.
        
        Returns:
            Dict with 'total', 'free', 'used' in USDT
        """
        try:
            balance = self._exchange.fetch_balance()
            usdt = balance.get(self.quote_currency, {})
            return {
                'total': float(usdt.get('total', 0)),
                'free': float(usdt.get('free', 0)),
                'used': float(usdt.get('used', 0)),
            }
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {'total': 0, 'free': 0, 'used': 0}
    
    def _get_positions_dict(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions as a dict (internal)."""
        try:
            positions = self._exchange.fetch_positions()
            result = {}

            for pos in positions:
                contracts = float(pos.get('contracts', 0))
                if abs(contracts) < 1e-8:
                    continue

                symbol = pos['symbol'].replace(f'/{self.quote_currency}:USDT', '')
                side = pos.get('side', 'long')
                notional = float(pos.get('notional', 0))
                entry_price = float(pos.get('entryPrice', 0))
                unrealized_pnl = float(pos.get('unrealizedPnl', 0))

                result[symbol] = {
                    'side': side,
                    'size': contracts if side == 'long' else -contracts,
                    'notional': abs(notional),
                    'entry_price': entry_price,
                    'unrealized_pnl': unrealized_pnl,
                }

            return result
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {}

    # ========================================================================
    # BrokerPlugin protocol methods
    # ========================================================================

    def get_positions(self) -> pd.DataFrame:
        """BrokerPlugin-compliant positions as DataFrame."""
        pos_dict = self._get_positions_dict()
        rows = []
        for symbol, p in pos_dict.items():
            rows.append({
                "symbol": symbol,
                "qty": p["size"],  # signed
                "notional": p["notional"],
                "entry_price": p["entry_price"],
                "unrealized_pnl": p["unrealized_pnl"],
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["symbol", "qty", "notional", "entry_price", "unrealized_pnl"]
        )

    def get_cash(self) -> Dict[str, float]:
        """BrokerPlugin-compliant cash."""
        bal = self.get_balance()
        return {self.quote_currency: bal.get("free", 0.0)}

    def get_market_snapshot(self, symbols: List[str]) -> pd.DataFrame:
        """BrokerPlugin-compliant market snapshot."""
        prices = self.get_prices(symbols)
        rows = []
        for s in symbols:
            info = self._get_symbol_info(s)
            rows.append({
                "symbol": s,
                "mid": prices.get(s),
                "min_qty": info["min_qty"] if info else 0.0,
                "step_size": info["step_size"] if info else 0.0,
                "min_notional": info["min_notional"] if info else 5.0,
            })
        return pd.DataFrame(rows)

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        """BrokerPlugin-compliant order execution."""
        fills: List[Dict[str, Any]] = []
        for _, o in orders.iterrows():
            sym = str(o["symbol"])
            side = str(o["side"]).lower()
            qty = float(o["qty"])
            result = self.place_order(sym, side, qty)
            if result:
                fill_price = float(result.get("average", result.get("price", 0)) or 0)
                filled_qty = float(result.get("filled", qty) or qty)
                fills.append({
                    "symbol": sym,
                    "side": side,
                    "qty": filled_qty,
                    "price": fill_price,
                    "order_id": result.get("id"),
                })
        return pd.DataFrame(fills) if fills else pd.DataFrame(
            columns=["symbol", "side", "qty", "price", "order_id"]
        )

    def fetch_fills(self, since: str) -> pd.DataFrame:
        """BrokerPlugin-compliant fill history."""
        try:
            since_ts = int(pd.Timestamp(since).timestamp() * 1000)
            all_trades: List[Dict[str, Any]] = []
            pos = self._get_positions_dict()
            for symbol in pos:
                market_symbol = f"{symbol}/{self.quote_currency}:USDT"
                trades = self._exchange.fetch_my_trades(market_symbol, since=since_ts)
                for t in trades:
                    all_trades.append({
                        "symbol": symbol,
                        "side": t.get("side", ""),
                        "qty": float(t.get("amount", 0)),
                        "price": float(t.get("price", 0)),
                        "timestamp": t.get("datetime", ""),
                    })
            return pd.DataFrame(all_trades) if all_trades else pd.DataFrame(
                columns=["symbol", "side", "qty", "price", "timestamp"]
            )
        except Exception as e:
            logger.error(f"Error fetching fills: {e}")
            return pd.DataFrame(
                columns=["symbol", "side", "qty", "price", "timestamp"]
            )
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        try:
            ticker = self._exchange.fetch_ticker(f"{symbol}/{self.quote_currency}:USDT")
            return float(ticker['last'])
        except Exception as e:
            logger.warning(f"Error fetching price for {symbol}: {e}")
            return None
    
    def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get prices for multiple symbols."""
        prices = {}
        for symbol in symbols:
            price = self.get_price(symbol)
            if price:
                prices[symbol] = price
        return prices
    
    # ========================================================================
    # Trading Methods
    # ========================================================================
    
    def _get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol trading info (lot size, tick size, min notional)."""
        market_symbol = f"{symbol}/{self.quote_currency}:USDT"
        market = self._markets.get(market_symbol)
        if not market:
            return None
        
        return {
            'symbol': market_symbol,
            'step_size': market['precision']['amount'],
            'tick_size': market['precision']['price'],
            'min_notional': market.get('limits', {}).get('cost', {}).get('min', 5),
            'min_qty': market.get('limits', {}).get('amount', {}).get('min', 0),
        }
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        try:
            market_symbol = f"{symbol}/{self.quote_currency}:USDT"
            self._exchange.set_leverage(leverage, market_symbol)
            logger.info(f"Set leverage for {symbol} to {leverage}x")
            return True
        except Exception as e:
            logger.warning(f"Failed to set leverage for {symbol}: {e}")
            return False
    
    def place_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        quantity: float,
        order_type: str = 'market',
    ) -> Optional[Dict]:
        """
        Place an order.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: 'market' or 'limit'
            
        Returns:
            Order result or None if failed
        """
        market_symbol = f"{symbol}/{self.quote_currency}:USDT"
        
        # Get symbol info
        info = self._get_symbol_info(symbol)
        if not info:
            logger.error(f"Unknown symbol: {symbol}")
            return None
        
        # Adjust quantity
        adj_qty = adjust_quantity(quantity, info['step_size'])
        if adj_qty <= 0:
            logger.warning(f"Quantity too small for {symbol}: {quantity}")
            return None
        
        # Check minimum notional
        price = self.get_price(symbol)
        if not price:
            return None
        
        notional = adj_qty * price
        if notional < info['min_notional']:
            logger.warning(f"Order notional ${notional:.2f} below minimum ${info['min_notional']}")
            return None
        
        # Set leverage
        self.set_leverage(symbol, self.target_leverage)
        
        # Place order with retries
        for attempt in range(MAX_RETRIES):
            try:
                order = self._exchange.create_order(
                    symbol=market_symbol,
                    type=order_type,
                    side=side,
                    amount=adj_qty,
                )
                
                fill_price = float(order.get('average', order.get('price', price)))
                filled_qty = float(order.get('filled', adj_qty))
                
                logger.info(f"Order filled: {side.upper()} {filled_qty} {symbol} @ ${fill_price:.4f}")
                
                # Send Telegram notification
                emoji = "🟢" if side == 'buy' else "🔴"
                msg = f"{emoji} <b>{side.upper()}</b> {filled_qty:.4f} {symbol}\n"
                msg += f"Price: ${fill_price:,.2f}\n"
                msg += f"Notional: ${filled_qty * fill_price:,.2f}"
                send_telegram(self.telegram_token, self.telegram_chat_id, msg)
                
                return order
                
            except ccxt.InsufficientFunds as e:
                logger.error(f"Insufficient funds for {symbol}: {e}")
                return None
            except ccxt.InvalidOrder as e:
                logger.error(f"Invalid order for {symbol}: {e}")
                return None
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"Order failed, retrying ({attempt+1}/{MAX_RETRIES}): {e}")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    logger.error(f"Order failed after {MAX_RETRIES} attempts: {e}")
                    return None
        
        return None
    
    def close_position(self, symbol: str) -> Optional[Dict]:
        """Close an existing position."""
        positions = self._get_positions_dict()
        pos = positions.get(symbol)
        
        if not pos:
            logger.info(f"No position to close for {symbol}")
            return None
        
        # Opposite side to close
        side = 'sell' if pos['side'] == 'long' else 'buy'
        quantity = abs(pos['size'])
        
        return self.place_order(symbol, side, quantity)
    
    # ========================================================================
    # Rebalancing
    # ========================================================================
    
    def _check_risk_limits(self, target_weights: Dict[str, float]) -> bool:
        """Check if target weights pass risk limits."""
        # Check single position limit
        for symbol, weight in target_weights.items():
            if abs(weight) > self.risk.max_position_pct:
                logger.error(f"Position {symbol} weight {weight:.1%} exceeds limit {self.risk.max_position_pct:.0%}")
                return False
        
        # Check gross exposure
        gross = sum(abs(w) for w in target_weights.values())
        if gross > self.risk.max_gross_exposure:
            logger.error(f"Gross exposure {gross:.1%} exceeds limit {self.risk.max_gross_exposure:.1%}")
            return False
        
        # Check daily loss limit
        balance = self.get_balance()
        if self._starting_balance > 0:
            daily_return = (balance['total'] - self._starting_balance) / self._starting_balance
            if daily_return < -self.risk.max_daily_loss_pct:
                logger.error(f"Daily loss {daily_return:.1%} exceeds limit, trading halted")
                send_telegram(
                    self.telegram_token, 
                    self.telegram_chat_id,
                    f"⚠️ <b>TRADING HALTED</b>\nDaily loss limit exceeded: {daily_return:.1%}"
                )
                return False
        
        return True
    
    def rebalance_to_weights(
        self,
        target_weights: Dict[str, float],
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Rebalance portfolio to target weights.
        
        Args:
            target_weights: Dict[symbol, weight] where weight is fraction of portfolio
                           Positive = long, Negative = short
            dry_run: If True, calculate orders but don't execute
            
        Returns:
            Dict with orders executed and summary
        """
        # Check risk limits
        if not self._check_risk_limits(target_weights):
            return {'error': 'Risk limits exceeded', 'orders': []}
        
        # Get current state
        balance = self.get_balance()
        portfolio_value = balance['total']
        current_positions = self._get_positions_dict()

        # Get prices
        all_symbols = set(target_weights.keys()) | set(current_positions.keys())
        prices = self.get_prices(list(all_symbols))

        # Calculate current weights
        current_weights = {}
        for symbol, pos in current_positions.items():
            if symbol in prices:
                weight = pos['notional'] / portfolio_value
                if pos['side'] == 'short':
                    weight = -weight
                current_weights[symbol] = weight
        
        # Calculate required trades
        trades = []
        for symbol in all_symbols:
            target = target_weights.get(symbol, 0)
            current = current_weights.get(symbol, 0)
            diff = target - current
            
            if abs(diff) < 0.01:  # Skip small rebalances
                continue
            
            price = prices.get(symbol)
            if not price:
                logger.warning(f"No price for {symbol}, skipping")
                continue
            
            # Calculate quantity
            target_notional = abs(diff) * portfolio_value
            quantity = target_notional / price
            
            # Determine side
            if diff > 0:
                side = 'buy'
            else:
                side = 'sell'
            
            trades.append({
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'notional': target_notional,
                'current_weight': current,
                'target_weight': target,
                'diff': diff,
            })
        
        # Sort: close positions first, then open
        trades.sort(key=lambda t: (0 if t['diff'] * current_weights.get(t['symbol'], 0) < 0 else 1))
        
        if dry_run:
            return {
                'dry_run': True,
                'portfolio_value': portfolio_value,
                'trades': trades,
            }
        
        # Execute trades
        executed = []
        for trade in trades:
            order = self.place_order(
                trade['symbol'],
                trade['side'],
                trade['quantity'],
            )
            if order:
                executed.append({
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'quantity': trade['quantity'],
                    'order_id': order.get('id'),
                })
        
        # Summary notification
        if executed:
            msg = f"📊 <b>Rebalance Complete</b>\n"
            msg += f"Portfolio: ${portfolio_value:,.2f}\n\n"
            for ex in executed:
                emoji = "📈" if ex['side'] == 'buy' else "📉"
                msg += f"{emoji} {ex['side'].upper()} {ex['quantity']:.4f} {ex['symbol']}\n"
            send_telegram(self.telegram_token, self.telegram_chat_id, msg)
        
        return {
            'portfolio_value': portfolio_value,
            'trades_planned': len(trades),
            'trades_executed': len(executed),
            'orders': executed,
        }
    
    # ========================================================================
    # Status Methods
    # ========================================================================
    
    def get_funding_rates(self, symbols: Optional[List[str]] = None) -> Dict[str, float]:
        """Get current funding rates for symbols."""
        if symbols is None:
            positions = self._get_positions_dict()
            symbols = list(positions.keys())
        
        rates = {}
        for symbol in symbols:
            try:
                market_symbol = f"{symbol}/{self.quote_currency}:USDT"
                funding = self._exchange.fetch_funding_rate(market_symbol)
                rates[symbol] = float(funding.get('fundingRate', 0))
            except Exception as e:
                logger.warning(f"Error fetching funding for {symbol}: {e}")
        
        return rates
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get full account summary."""
        balance = self.get_balance()
        positions = self._get_positions_dict()
        
        # Calculate exposure
        long_notional = sum(p['notional'] for p in positions.values() if p['side'] == 'long')
        short_notional = sum(p['notional'] for p in positions.values() if p['side'] == 'short')
        
        # Unrealized PnL
        unrealized_pnl = sum(p['unrealized_pnl'] for p in positions.values())
        
        return {
            'balance': balance,
            'positions': positions,
            'exposure': {
                'long': long_notional,
                'short': short_notional,
                'net': long_notional - short_notional,
                'gross': long_notional + short_notional,
            },
            'unrealized_pnl': unrealized_pnl,
            'position_count': len(positions),
        }
    
    def format_status(self) -> str:
        """Format account status for display/notification."""
        summary = self.get_account_summary()
        balance = summary['balance']
        exposure = summary['exposure']
        positions = summary['positions']
        
        lines = [
            f"💰 <b>Account Status</b>",
            f"Balance: ${balance['total']:,.2f}",
            f"Unrealized PnL: ${summary['unrealized_pnl']:+,.2f}",
            f"",
            f"📊 Exposure:",
            f"  Long: ${exposure['long']:,.2f}",
            f"  Short: ${exposure['short']:,.2f}",
            f"  Net: ${exposure['net']:+,.2f}",
            f"",
            f"📈 Positions ({len(positions)}):",
        ]
        
        for symbol, pos in sorted(positions.items()):
            emoji = "📈" if pos['side'] == 'long' else "📉"
            lines.append(f"  {emoji} {symbol}: ${pos['notional']:,.0f} ({pos['unrealized_pnl']:+.2f})")
        
        return "\n".join(lines)


# ============================================================================
# Factory Function
# ============================================================================

def create_broker(
    api_key: str = None,
    api_secret: str = None,
    leverage: int = 1,
    **kwargs,
) -> BinanceFuturesBroker:
    """
    Factory function to create broker.
    
    Args:
        api_key: Binance API key (or use env var)
        api_secret: Binance API secret (or use env var)
        leverage: Target leverage (default: 1)
        
    Returns:
        Configured BinanceFuturesBroker
    """
    return BinanceFuturesBroker(
        api_key=api_key or "",
        api_secret=api_secret or "",
        target_leverage=leverage,
        **kwargs,
    )
