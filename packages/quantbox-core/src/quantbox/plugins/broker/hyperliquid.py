"""
Hyperliquid Perpetuals Broker Plugin

Live trading on Hyperliquid DEX with:
- Decentralized perpetuals (no geo-restrictions)
- Native USDC settlement
- EVM wallet-based authentication
- No KYC required

## LLM Usage Guide

### Quick Start
```python
from quantbox.plugins.broker import HyperliquidBroker

# Using private key (API wallet recommended)
broker = HyperliquidBroker(
    wallet_address="0x...",
    private_key="0x...",
)

# Check account
print(broker.get_balance())
print(broker.get_positions())

# Execute target weights
broker.rebalance_to_weights({"BTC": 0.5, "ETH": -0.3})
```

### Authentication
Hyperliquid uses EVM-style wallet signing:
1. Go to https://app.hyperliquid.xyz/API
2. Generate an API Wallet (separate from main wallet)
3. Set `wallet_address` = main wallet, `private_key` = API wallet private key

### Key Features
- Uses ccxt.hyperliquid
- Handles LONG/SHORT positions natively
- Very low fees (0.02% taker)
- Deep liquidity on majors
- No geo-restrictions (decentralized)

### Risk Controls (Built-in)
- max_position_pct: Max single position (default: 50%)
- max_gross_exposure: Max total exposure (default: 2.0x)
- max_daily_loss_pct: Stop trading if exceeded (default: 10%)
"""
from __future__ import annotations
import os
import json
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

QUOTE_CURRENCY = "USDC"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

# Hyperliquid API URLs
MAINNET_API = "https://api.hyperliquid.xyz"
TESTNET_API = "https://api.hyperliquid-testnet.xyz"


# ============================================================================
# Risk Configuration
# ============================================================================

@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_pct: float = 0.50  # Max 50% in single position
    max_gross_exposure: float = 2.0  # Max 200% gross
    max_daily_loss_pct: float = 0.10  # Stop at 10% daily loss
    min_order_notional: float = 10.0  # Hyperliquid minimum ~$10
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
# Broker Class
# ============================================================================

@dataclass
class HyperliquidBroker:
    """
    Hyperliquid DEX Perpetuals Broker.

    Decentralized perpetuals trading with no geo-restrictions.
    Uses EVM wallet for authentication.

    ## Quick Start
    ```python
    broker = HyperliquidBroker(
        wallet_address="0x...",
        private_key="0x...",
    )
    broker.rebalance_to_weights({"BTC": 0.5, "ETH": -0.3})
    ```

    ## Authentication
    1. Go to https://app.hyperliquid.xyz/API
    2. Generate an API Wallet
    3. Use main wallet address + API wallet private key
    """

    meta = PluginMeta(
        name="hyperliquid.perps.v1",
        kind="broker",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Hyperliquid DEX perpetuals broker (live) with short support",
        tags=("live", "futures", "hyperliquid", "decentralized"),
        capabilities=("live", "futures", "shorts", "leverage"),
        schema_version="v1",
    )

    # Credentials
    wallet_address: str = ""
    private_key: str = ""
    
    # Network
    testnet: bool = False
    
    # Risk config
    risk: RiskConfig = field(default_factory=RiskConfig)
    
    # Notifications
    telegram_token: str = ""
    telegram_chat_id: str = ""
    
    # State
    _exchange: Optional[ccxt.hyperliquid] = field(default=None, repr=False)
    _markets: Dict = field(default_factory=dict, repr=False)
    _daily_pnl: float = 0.0
    _starting_balance: float = 0.0
    
    def __post_init__(self):
        """Initialize exchange connection."""
        # Load from env if not provided
        if not self.wallet_address:
            self.wallet_address = os.environ.get("HYPERLIQUID_WALLET", "")
        if not self.private_key:
            self.private_key = os.environ.get("HYPERLIQUID_PRIVATE_KEY", "")
        if not self.telegram_token:
            self.telegram_token = os.environ.get("TELEGRAM_TOKEN", "")
        if not self.telegram_chat_id:
            self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        
        if not self.wallet_address or not self.private_key:
            raise ValueError(
                "Hyperliquid credentials required. Set:\n"
                "  HYPERLIQUID_WALLET=0x... (your main wallet)\n"
                "  HYPERLIQUID_PRIVATE_KEY=0x... (API wallet private key)\n"
                "Generate API wallet at https://app.hyperliquid.xyz/API"
            )
        
        self._init_exchange()
    
    def _init_exchange(self) -> None:
        """Initialize ccxt exchange."""
        self._exchange = ccxt.hyperliquid({
            'walletAddress': self.wallet_address,
            'privateKey': self.private_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # perpetual swaps
                'defaultSlippage': 0.05,  # 5% max slippage for market orders
            }
        })
        
        if self.testnet:
            self._exchange.set_sandbox_mode(True)
        
        # Load markets
        self._markets = self._exchange.load_markets()
        logger.info(f"Connected to Hyperliquid {'testnet' if self.testnet else 'mainnet'}, "
                    f"{len(self._markets)} markets loaded")
        
        # Record starting balance for daily PnL tracking
        balance = self.get_balance()
        self._starting_balance = balance.get('total', 0)
    
    def describe(self) -> Dict[str, Any]:
        """Describe broker for LLM introspection."""
        return {
            "name": "HyperliquidBroker",
            "exchange": "Hyperliquid DEX",
            "type": "decentralized",
            "network": "testnet" if self.testnet else "mainnet",
            "wallet": self.wallet_address[:10] + "..." if self.wallet_address else None,
            "risk_config": self.risk.describe(),
            "notifications": bool(self.telegram_token),
            "features": [
                "No geo-restrictions",
                "No KYC",
                "USDC settlement",
                "0.02% taker fee",
            ],
        }
    
    # ========================================================================
    # Account Methods
    # ========================================================================
    
    def get_balance(self) -> Dict[str, float]:
        """
        Get account balance.

        Checks both swap (perps) and spot balances to support
        Hyperliquid unified accounts where USDC may sit on
        the spot side.

        Returns:
            Dict with 'total', 'free', 'used' in USDC
        """
        try:
            balance = self._exchange.fetch_balance()
            usdc = balance.get(QUOTE_CURRENCY, balance.get('USDC', {}))
            total = float(usdc.get('total', 0) or 0)
            free = float(usdc.get('free', 0) or 0)
            used = float(usdc.get('used', 0) or 0)

            # Unified accounts: USDC may be on the spot side
            if total == 0:
                spot_balance = self._exchange.fetch_balance({'type': 'spot'})
                spot_usdc = spot_balance.get(QUOTE_CURRENCY, spot_balance.get('USDC', {}))
                total = float(spot_usdc.get('total', 0) or 0)
                free = float(spot_usdc.get('free', 0) or 0)
                used = float(spot_usdc.get('used', 0) or 0)

            return {'total': total, 'free': free, 'used': used}
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {'total': 0, 'free': 0, 'used': 0}
    
    def _get_positions_dict(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions as a dict (internal)."""
        try:
            positions = self._exchange.fetch_positions()
            result = {}

            for pos in positions:
                contracts = float(pos.get('contracts', 0) or 0)
                if abs(contracts) < 1e-8:
                    continue

                # Extract base symbol (e.g., "BTC" from "BTC/USDC:USDC")
                symbol = pos['symbol'].split('/')[0]
                side = pos.get('side', 'long')
                notional = abs(float(pos.get('notional', 0) or 0))
                entry_price = float(pos.get('entryPrice', 0) or 0)
                unrealized_pnl = float(pos.get('unrealizedPnl', 0) or 0)

                result[symbol] = {
                    'side': side,
                    'size': contracts if side == 'long' else -contracts,
                    'notional': notional,
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
        return {QUOTE_CURRENCY: bal.get("free", 0.0)}

    def get_market_snapshot(self, symbols: List[str]) -> pd.DataFrame:
        """BrokerPlugin-compliant market snapshot."""
        prices = self.get_prices(symbols)
        rows = []
        for s in symbols:
            market_symbol = self._get_market_symbol(s)
            market = self._markets.get(market_symbol, {}) if market_symbol else {}
            precision = market.get("precision", {})
            limits = market.get("limits", {})
            rows.append({
                "symbol": s,
                "mid": prices.get(s),
                "min_qty": limits.get("amount", {}).get("min", 0.0) or 0.0,
                "step_size": precision.get("amount", 0.0) or 0.0,
                "min_notional": limits.get("cost", {}).get("min", 10.0) or 10.0,
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
                market_symbol = self._get_market_symbol(symbol)
                if not market_symbol:
                    continue
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
        for attempt in range(MAX_RETRIES):
            try:
                for market_symbol in [
                    f"{symbol}/USDC:USDC",
                    f"{symbol}/USD:USDC",
                    f"{symbol}USDC",
                ]:
                    if market_symbol in self._markets:
                        ticker = self._exchange.fetch_ticker(market_symbol)
                        return float(ticker['last'])

                logger.warning(f"Market not found for {symbol}")
                return None
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    backoff = RETRY_DELAY_SECONDS * (2 ** attempt)
                    logger.warning(
                        "Rate limited fetching %s, retry %d/%d in %ds: %s",
                        symbol, attempt + 1, MAX_RETRIES, backoff, e,
                    )
                    time.sleep(backoff)
                else:
                    logger.warning(f"Error fetching price for {symbol}: {e}")
                    return None
        return None

    def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get prices for multiple symbols."""
        prices = {}
        for i, symbol in enumerate(symbols):
            if i > 0:
                time.sleep(0.1)  # 100ms between API calls
            price = self.get_price(symbol)
            if price:
                prices[symbol] = price
        return prices
    
    def _get_market_symbol(self, symbol: str) -> Optional[str]:
        """Get full market symbol for trading."""
        for market_symbol in [
            f"{symbol}/USDC:USDC",
            f"{symbol}/USD:USDC",
        ]:
            if market_symbol in self._markets:
                return market_symbol
        return None
    
    # ========================================================================
    # Trading Methods
    # ========================================================================
    
    def place_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        quantity: float,
        order_type: str = 'market',
        reduce_only: bool = False,
    ) -> Optional[Dict]:
        """
        Place an order.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            side: 'buy' or 'sell'
            quantity: Order quantity in base asset
            order_type: 'market' or 'limit'
            reduce_only: If True, only reduces position
            
        Returns:
            Order result or None if failed
        """
        market_symbol = self._get_market_symbol(symbol)
        if not market_symbol:
            logger.error(f"Unknown symbol: {symbol}")
            return None
        
        market = self._markets[market_symbol]
        
        # Get price for notional check
        price = self.get_price(symbol)
        if not price:
            return None
        
        # Check minimum notional
        notional = quantity * price
        if notional < self.risk.min_order_notional:
            logger.warning(f"Order notional ${notional:.2f} below minimum ${self.risk.min_order_notional}")
            return None
        
        # Adjust quantity to precision
        # ccxt may return precision as a float step size (e.g. 0.001)
        # or as an int number of decimals (e.g. 3)
        precision = market.get('precision', {}).get('amount', 8)
        if isinstance(precision, float) and precision < 1:
            import math
            precision = max(0, -int(math.floor(math.log10(precision))))
        quantity = round(quantity, int(precision))
        
        if quantity <= 0:
            logger.warning(f"Quantity too small for {symbol}")
            return None
        
        # Place order with retries
        for attempt in range(MAX_RETRIES):
            try:
                params = {}
                if reduce_only:
                    params['reduceOnly'] = True
                
                order = self._exchange.create_order(
                    symbol=market_symbol,
                    type=order_type,
                    side=side,
                    amount=quantity,
                    params=params,
                )
                
                fill_price = float(order.get('average', order.get('price', price)) or price)
                filled_qty = float(order.get('filled', quantity) or quantity)
                
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
        
        return self.place_order(symbol, side, quantity, reduce_only=True)
    
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
        
        if portfolio_value <= 0:
            return {'error': 'No balance', 'portfolio_value': 0, 'orders': []}
        
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
                market_symbol = self._get_market_symbol(symbol)
                if market_symbol:
                    funding = self._exchange.fetch_funding_rate(market_symbol)
                    rates[symbol] = float(funding.get('fundingRate', 0) or 0)
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
            f"💰 <b>Hyperliquid Account</b>",
            f"Balance: ${balance['total']:,.2f} USDC",
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
        
        if not positions:
            lines.append("  (none)")
        
        return "\n".join(lines)


# ============================================================================
# Factory Function
# ============================================================================

def create_broker(
    wallet_address: str = None,
    private_key: str = None,
    testnet: bool = False,
    **kwargs,
) -> HyperliquidBroker:
    """
    Factory function to create broker.
    
    Args:
        wallet_address: Main wallet address (or use env var)
        private_key: API wallet private key (or use env var)
        testnet: Use testnet instead of mainnet
        
    Returns:
        Configured HyperliquidBroker
    """
    return HyperliquidBroker(
        wallet_address=wallet_address or "",
        private_key=private_key or "",
        testnet=testnet,
        **kwargs,
    )
