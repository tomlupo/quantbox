"""
Binance Live Broker Plugin - Production-grade adapter for AI-powered trading.

This module provides the BinanceLiveBroker class - a full-featured Binance trading
adapter designed for use by LLM agents in quantitative research and trading workflows.

## LLM Usage Guide

### Quick Start (Paper Trading)
```python
from quantbox.plugins.broker import BinanceLiveBroker

# Initialize in paper mode (safe for testing)
broker = BinanceLiveBroker(paper_trading=True, account_name="research")

# Check current portfolio
positions = broker.get_positions()  # DataFrame: symbol, qty
cash = broker.get_cash()            # Dict: {'USDC': 10000.0}
value = broker.get_portfolio_value() # Float: total value in stable coin

# Get prices for assets
snapshot = broker.get_market_snapshot(['BTCUSDC', 'ETHUSDC'])
```

### Rebalancing Workflow
```python
# Define target weights (must sum to <= 1.0)
target_weights = {
    'BTC': 0.40,   # 40% Bitcoin
    'ETH': 0.30,   # 30% Ethereum  
    'SOL': 0.15,   # 15% Solana
    # Remaining 15% stays in USDC cash
}

# Analyze what trades are needed (no execution)
analysis = broker.generate_rebalancing(target_weights)
# Returns DataFrame with: Asset, Current_Weight, Target_Weight, Delta_Qty, Action

# Execute the rebalancing (with all safety checks)
result = broker.execute_rebalancing(target_weights)
# Returns: {'orders': DataFrame, 'summary': {...}, 'total_value': float}
```

### Key Methods for LLM Agents
- `get_positions()` → Current holdings as DataFrame
- `get_cash()` → Cash balances as dict
- `get_portfolio_value()` → Total value as float
- `get_market_snapshot(symbols)` → Current prices as DataFrame
- `generate_rebalancing(weights)` → Analysis without execution
- `execute_rebalancing(weights)` → Full execution with report
- `fetch_fills(since)` → Trade history as DataFrame

### Safety Features
- Paper trading mode (default) - no real money
- Sell-before-buy sequencing (frees up capital)
- Buy order scaling (only buys what you can afford)
- Min notional / lot size validation (Binance rules)
- Price caching (avoids rate limits)
- Retry logic (handles transient errors)

### Configuration
All parameters have sensible defaults. Key ones:
- `paper_trading: bool = False` - Set True for testing
- `stable_coin: str = 'USDC'` - Base currency
- `capital_at_risk: float = 1.0` - Fraction of portfolio to trade
- `min_trade_size: float = 0.01` - Min weight change to trigger trade

Features ported from quantlab production system.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal, getcontext, ROUND_DOWN
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Callable
import os
import json
import time
import pandas as pd
import logging

from quantbox.contracts import PluginMeta

logger = logging.getLogger(__name__)

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    Client = None  # type: ignore
    BinanceAPIException = Exception  # type: ignore

# ============================================================================
# Constants
# ============================================================================
DEFAULT_STABLE_COIN = "USDC"
DEFAULT_MIN_NOTIONAL = 1.0
DEFAULT_MIN_TRADE_SIZE = 0.01
DEFAULT_CAPITAL_AT_RISK = 1.0
DEFAULT_SCALING_FACTOR_MIN = 0.9
DEFAULT_PRICE_CACHE_TTL = 30  # seconds
DEFAULT_MAX_RETRIES = 3

# Transient Binance API error codes
_TRANSIENT_API_CODES = {0, -1003}  # 0 = invalid JSON (HTML 503), -1003 = rate limit

# ============================================================================
# Market Simulation Constants (from quantlab)
# ============================================================================
DEFAULT_SLIPPAGE_BPS = 5.0        # 0.05% base slippage
DEFAULT_SPREAD_BPS = 10.0         # 0.1% bid-ask spread
DEFAULT_MAX_PRICE_IMPACT_BPS = 20.0  # 0.2% max price impact
DEFAULT_COMMISSION_BPS = 10.0     # 0.1% commission (Binance typical)
DEFAULT_MIN_FILL_DELAY_MS = 100   # Minimum fill delay
DEFAULT_MAX_FILL_DELAY_MS = 2000  # Maximum fill delay
DEFAULT_MAX_LEVERAGE = 1.0        # No leverage by default


# ============================================================================
# Market Simulation Config
# ============================================================================

@dataclass
class MarketSimConfig:
    """
    Configuration for realistic market execution simulation.
    
    All costs in basis points (bps). 1 bp = 0.01% = 0.0001.
    
    Example:
        >>> config = MarketSimConfig(slippage_bps=5, spread_bps=10, commission_bps=10)
        >>> # Total expected cost: ~25 bps (0.25%) per round trip
    """
    # Execution costs
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    spread_bps: float = DEFAULT_SPREAD_BPS
    max_price_impact_bps: float = DEFAULT_MAX_PRICE_IMPACT_BPS
    commission_bps: float = DEFAULT_COMMISSION_BPS
    
    # Timing simulation
    min_fill_delay_ms: int = DEFAULT_MIN_FILL_DELAY_MS
    max_fill_delay_ms: int = DEFAULT_MAX_FILL_DELAY_MS
    simulate_delay: bool = False  # Set True for realistic timing
    
    # Risk limits
    max_leverage: float = DEFAULT_MAX_LEVERAGE
    
    # Randomization
    random_slippage: bool = True  # Add random component to slippage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            'slippage_bps': self.slippage_bps,
            'spread_bps': self.spread_bps,
            'max_price_impact_bps': self.max_price_impact_bps,
            'commission_bps': self.commission_bps,
            'min_fill_delay_ms': self.min_fill_delay_ms,
            'max_fill_delay_ms': self.max_fill_delay_ms,
            'simulate_delay': self.simulate_delay,
            'max_leverage': self.max_leverage,
        }


def simulate_execution_price(
    base_price: float,
    side: str,
    quantity: float,
    config: MarketSimConfig,
) -> Tuple[float, Dict[str, float]]:
    """
    Simulate realistic execution price with spread, slippage, and price impact.
    
    Args:
        base_price: Mid-market price
        side: 'BUY' or 'SELL'
        quantity: Order quantity
        config: Market simulation config
        
    Returns:
        Tuple of (execution_price, breakdown_dict)
    
    Example:
        >>> price, breakdown = simulate_execution_price(100.0, 'BUY', 1.0, MarketSimConfig())
        >>> print(f"Execution: ${price:.4f}, costs: {breakdown}")
    """
    import random
    
    is_buy = side.upper() == 'BUY'
    
    # 1. Bid-ask spread: buy at ask (higher), sell at bid (lower)
    spread_factor = config.spread_bps / 10000 / 2  # Half spread each side
    if is_buy:
        price = base_price * (1 + spread_factor)
    else:
        price = base_price * (1 - spread_factor)
    spread_cost = abs(price - base_price)
    
    # 2. Price impact based on order size (linear, capped)
    # Assumes larger orders have more market impact
    notional = quantity * base_price
    # Scale: $10k notional = ~0.01% impact, $100k = ~0.1%, capped at max
    impact_bps = min(notional / 1000000 * 10, config.max_price_impact_bps)
    impact_factor = impact_bps / 10000
    if is_buy:
        price *= (1 + impact_factor)
    else:
        price *= (1 - impact_factor)
    impact_cost = abs(price - base_price - spread_cost)
    
    # 3. Random slippage
    if config.random_slippage:
        slippage_factor = config.slippage_bps / 10000
        random_slip = random.uniform(-slippage_factor, slippage_factor)
        price *= (1 + random_slip)
        slippage_cost = abs(random_slip * base_price)
    else:
        slippage_cost = 0.0
    
    # 4. Simulate fill delay if enabled
    if config.simulate_delay:
        delay_ms = random.uniform(config.min_fill_delay_ms, config.max_fill_delay_ms)
        time.sleep(delay_ms / 1000)
    else:
        delay_ms = 0
    
    breakdown = {
        'base_price': base_price,
        'spread_cost': spread_cost,
        'impact_cost': impact_cost,
        'slippage_cost': slippage_cost,
        'total_cost_bps': (abs(price - base_price) / base_price) * 10000,
        'delay_ms': delay_ms,
    }
    
    return price, breakdown


def calculate_commission(
    quantity: float,
    execution_price: float,
    config: MarketSimConfig,
) -> float:
    """
    Calculate trading commission.
    
    Args:
        quantity: Order quantity
        execution_price: Execution price
        config: Market simulation config
        
    Returns:
        Commission amount in quote currency
    """
    notional = quantity * execution_price
    commission = notional * (config.commission_bps / 10000)
    return commission


# ============================================================================
# Utility Functions (ported from quantlab)
# ============================================================================

def adjust_quantity(qty: float, step_size: float) -> float:
    """
    Adjust quantity to conform to Binance step size requirement.
    Uses Decimal for proper precision handling.
    
    Args:
        qty: Raw quantity
        step_size: Binance LOT_SIZE stepSize filter
        
    Returns:
        Adjusted quantity rounded down to valid step
    """
    if step_size <= 0:
        return qty
    step_size_str = f"{step_size:.8f}"
    decimal_places = step_size_str.rstrip('0').split('.')[-1]
    precision = len(decimal_places)
    getcontext().rounding = ROUND_DOWN
    adjusted_qty = Decimal(str(qty)).quantize(Decimal('1.' + '0' * precision))
    return float(adjusted_qty)


def get_lot_size_and_min_notional(symbol_info: Optional[dict]) -> Tuple[float, float, float]:
    """
    Extract LOT_SIZE and NOTIONAL filters from Binance symbol info.
    
    Returns:
        Tuple of (min_qty, step_size, min_notional)
    """
    min_qty, step_size, min_notional = 0.0, 0.0, 0.0
    if symbol_info:
        for f in symbol_info.get('filters', []):
            if f['filterType'] == 'LOT_SIZE':
                min_qty = float(f['minQty'])
                step_size = float(f['stepSize'])
            if f['filterType'] == 'NOTIONAL':
                min_notional = float(f['minNotional'])
    return min_qty, step_size, min_notional


def _is_transient_error(e: Exception) -> bool:
    """Check if exception is a transient error worth retrying."""
    if isinstance(e, BinanceAPIException):
        return e.code in _TRANSIENT_API_CODES
    return isinstance(e, (ConnectionError, TimeoutError, OSError))


def format_quantity_for_binance(qty: float) -> str:
    r"""
    Format quantity for Binance API (no scientific notation).
    Binance requires: ^([0-9]{1,20})(\.[0-9]{1,20})?$
    """
    if qty == 0:
        return "0"
    qty_str = f"{qty:.8f}".rstrip('0').rstrip('.')
    if not qty_str or qty_str == '.':
        return "0"
    return qty_str


# ============================================================================
# Price Cache
# ============================================================================

class PriceCache:
    """TTL-based price cache to avoid redundant API calls."""
    
    def __init__(self, ttl_seconds: int = DEFAULT_PRICE_CACHE_TTL):
        self.ttl = ttl_seconds
        self._cache: Dict[str, Tuple[Optional[float], float]] = {}
    
    def get(self, asset: str) -> Tuple[Optional[float], bool]:
        """Get cached price if valid. Returns (price, is_cached)."""
        if asset in self._cache:
            price, cached_at = self._cache[asset]
            if time.time() - cached_at < self.ttl:
                return price, True
        return None, False
    
    def set(self, asset: str, price: Optional[float]) -> None:
        """Cache a price."""
        self._cache[asset] = (price, time.time())
    
    def clear(self) -> None:
        """Clear all cached prices."""
        self._cache.clear()


# ============================================================================
# Binance Live Broker
# ============================================================================

@dataclass
class BinanceLiveBroker:
    """
    Production Binance broker with full quantlab features.
    
    Implements BrokerPlugin protocol with extended capabilities:
    - Portfolio rebalancing with constraint validation
    - Sell-before-buy order sequencing
    - Buy order scaling to available cash
    - Paper trading mode
    - Trade history persistence
    """
    
    # Connection
    api_key_env: str = "BINANCE_API_KEY"
    api_secret_env: str = "BINANCE_API_SECRET"
    testnet: bool = False
    
    # Trading config
    stable_coin: str = DEFAULT_STABLE_COIN
    paper_trading: bool = False
    account_name: str = "default"
    readonly: bool = False
    
    # Thresholds
    min_notional: float = DEFAULT_MIN_NOTIONAL
    min_trade_size: float = DEFAULT_MIN_TRADE_SIZE
    capital_at_risk: float = DEFAULT_CAPITAL_AT_RISK
    scaling_factor_min: float = DEFAULT_SCALING_FACTOR_MIN
    
    # Features
    retry_transient: bool = True
    max_retries: int = DEFAULT_MAX_RETRIES
    price_cache_ttl: int = DEFAULT_PRICE_CACHE_TTL
    persist_trades: bool = True
    trades_dir: str = "data"
    
    # Market simulation (for paper trading)
    market_sim: Optional[MarketSimConfig] = None
    
    # Exclusions (assets to ignore)
    exclusions: List[str] = field(default_factory=lambda: ['ETHW', 'BETH'])
    
    # Internal
    _client: Any = field(default=None, repr=False)
    _price_cache: PriceCache = field(default=None, repr=False)
    
    meta = PluginMeta(
        name="binance.live.v1",
        kind="broker",
        version="1.0.0",
        core_compat=">=0.1,<1.0",
        description="Production Binance broker with rebalancing, paper trading, and trade persistence",
        tags=("binance", "broker", "crypto", "production"),
        capabilities=("paper", "live", "crypto", "rebalancing"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "api_key_env": {"type": "string", "default": "BINANCE_API_KEY"},
                "api_secret_env": {"type": "string", "default": "BINANCE_API_SECRET"},
                "testnet": {"type": "boolean", "default": False},
                "stable_coin": {"type": "string", "default": "USDC"},
                "paper_trading": {"type": "boolean", "default": False},
                "account_name": {"type": "string", "default": "default"},
                "readonly": {"type": "boolean", "default": False},
                "min_notional": {"type": "number", "default": 1.0},
                "min_trade_size": {"type": "number", "default": 0.01},
                "capital_at_risk": {"type": "number", "default": 1.0},
                "persist_trades": {"type": "boolean", "default": True},
            }
        },
        inputs=("target_weights",),
        outputs=("execution_report", "trade_history"),
        examples=(
            "plugins:\n  broker:\n    name: binance.live.v1\n    params_init:\n      stable_coin: USDC\n      paper_trading: true",
        )
    )
    
    def __post_init__(self):
        self._price_cache = PriceCache(self.price_cache_ttl)
        
        # Initialize market simulation config for paper trading
        if self.market_sim is None and self.paper_trading:
            self.market_sim = MarketSimConfig()
        
        # Add stable coin to exclusions
        if self.stable_coin not in self.exclusions:
            self.exclusions = list(self.exclusions) + [self.stable_coin]
        
        # Try to connect to Binance for price data (even in paper mode)
        api_key = os.environ.get(self.api_key_env)
        api_secret = os.environ.get(self.api_secret_env)
        
        if api_key and api_secret and Client is not None:
            try:
                self._client = Client(api_key, api_secret, testnet=self.testnet)
                logger.info(f"Connected to Binance API for price data")
            except Exception as e:
                logger.warning(f"Could not connect to Binance: {e}")
                self._client = None
        else:
            self._client = None
            if not self.paper_trading:
                raise EnvironmentError(f"Live mode requires: {self.api_key_env} / {self.api_secret_env}")
        
        mode = "paper" if self.paper_trading else "live"
        prices = "real" if self._client else "mock"
        sim = "with market sim" if self.market_sim else "instant fills"
        logger.info(f"Initialized {mode} broker ({prices} prices, {sim}): {self.account_name}")
    
    # ========================================================================
    # BrokerPlugin Protocol Implementation
    # ========================================================================
    
    def get_positions(self) -> pd.DataFrame:
        """
        Get current portfolio positions (holdings).
        
        Returns:
            DataFrame with columns:
            - symbol: Asset symbol (e.g., 'BTC', 'ETH', 'USDC')
            - qty: Quantity held (float)
        
        Example:
            >>> broker.get_positions()
               symbol        qty
            0     BTC   0.523400
            1     ETH   4.120000
            2    USDC  1523.45
        
        LLM Note: Use this to understand current portfolio state before rebalancing.
        """
        if self.paper_trading:
            holdings = self._get_paper_holdings()
        else:
            holdings = self._get_live_holdings()
        
        rows = [{"symbol": k, "qty": v} for k, v in holdings.items() if v != 0]
        return pd.DataFrame(rows)
    
    def get_cash(self) -> Dict[str, float]:
        """
        Get cash balances (stablecoins).
        
        Returns:
            Dict of stablecoin balances
        """
        pos = self.get_positions()
        cash = {}
        stables = ("USDT", "USD", "BUSD", "USDC", self.stable_coin)
        for _, r in pos.iterrows():
            sym = str(r["symbol"])
            if sym in stables:
                cash[sym] = float(r["qty"])
        return cash
    
    def get_market_snapshot(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get current prices for symbols.
        
        Args:
            symbols: List of trading pairs (e.g., ['BTCUSDC', 'ETHUSDC'])
            
        Returns:
            DataFrame with columns: symbol, mid
        """
        rows = []
        for sym in symbols:
            # Extract base asset from pair
            base = sym.replace(self.stable_coin, "").replace("USDT", "").replace("BUSD", "")
            price = self._get_price_cached(base)
            rows.append({"symbol": sym, "mid": price})
        return pd.DataFrame(rows)
    
    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        """
        Execute orders with sell-before-buy sequencing.
        
        Expected columns: symbol, side, qty, [price]
        
        Returns:
            DataFrame of fills
        """
        if self.readonly:
            raise PermissionError("readonly broker: order placement disabled")
        
        if orders.empty:
            return pd.DataFrame()
        
        # Sort: sells first, then buys
        orders = orders.copy()
        orders['_sort'] = orders['side'].apply(lambda x: 0 if x.lower() == 'sell' else 1)
        orders = orders.sort_values('_sort').drop(columns=['_sort'])
        
        fills = []
        for _, o in orders.iterrows():
            sym = str(o["symbol"])
            side = str(o["side"]).upper()
            qty = float(o["qty"])
            price = o.get("price", None)
            
            fill = self._execute_single_order(sym, side, qty, price)
            if fill:
                fills.append(fill)
        
        return pd.DataFrame(fills)
    
    def fetch_fills(self, since: str) -> pd.DataFrame:
        """
        Fetch trade history from persisted files.
        
        Args:
            since: ISO date string (YYYY-MM-DD)
            
        Returns:
            DataFrame of historical fills
        """
        trades_path = os.path.join(self.trades_dir, self.account_name.lower(), "trades")
        if not os.path.exists(trades_path):
            return pd.DataFrame()
        
        all_trades = []
        for filename in os.listdir(trades_path):
            if filename.endswith('.json') and filename >= since:
                filepath = os.path.join(trades_path, filename)
                with open(filepath) as f:
                    data = json.load(f)
                    all_trades.extend(data.get('trades', []))
        
        return pd.DataFrame(all_trades)
    
    # ========================================================================
    # Extended API (quantlab features)
    # ========================================================================
    
    def describe(self) -> Dict[str, Any]:
        """
        Get a complete snapshot of broker state - designed for LLM consumption.
        
        Returns:
            Dict with:
            - mode: 'paper' or 'live'
            - account: account name
            - stable_coin: base currency
            - portfolio_value: total value
            - positions: list of {symbol, qty, value, weight}
            - cash: cash balances
            - config: key configuration values
        
        Example:
            >>> broker.describe()
            {
                'mode': 'paper',
                'account': 'research',
                'stable_coin': 'USDC',
                'portfolio_value': 15234.56,
                'positions': [
                    {'symbol': 'BTC', 'qty': 0.52, 'value': 5200.0, 'weight': 0.34},
                    {'symbol': 'ETH', 'qty': 4.1, 'value': 3100.0, 'weight': 0.20},
                ],
                'cash': {'USDC': 6934.56},
                'config': {'capital_at_risk': 1.0, 'min_trade_size': 0.01}
            }
        
        LLM Note: Call this first to understand the current state before making decisions.
        """
        holdings = self._get_holdings_dict()
        total_value = self.get_portfolio_value()
        
        positions = []
        for asset, qty in holdings.items():
            if asset == self.stable_coin or asset in self.exclusions:
                continue
            price = self._get_price_cached(asset)
            value = qty * price if price else 0
            weight = value / total_value if total_value > 0 else 0
            positions.append({
                'symbol': asset,
                'qty': round(qty, 8),
                'value': round(value, 2),
                'weight': round(weight, 4),
                'price': round(price, 2) if price else None,
            })
        
        # Sort by value descending
        positions.sort(key=lambda x: x['value'], reverse=True)
        
        result = {
            'mode': 'paper' if self.paper_trading else 'live',
            'account': self.account_name,
            'stable_coin': self.stable_coin,
            'portfolio_value': round(total_value, 2),
            'positions': positions,
            'cash': self.get_cash(),
            'config': {
                'capital_at_risk': self.capital_at_risk,
                'min_trade_size': self.min_trade_size,
                'min_notional': self.min_notional,
                'readonly': self.readonly,
            }
        }
        
        # Add market simulation config if enabled
        if self.market_sim:
            result['market_sim'] = self.market_sim.to_dict()
        
        return result

    def get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value in stable coin.
        
        Handles both long and short positions with proper mark-to-market:
        - Long positions: qty × current_price
        - Short positions: entry_value + unrealized_pnl
          where unrealized_pnl = (entry_price - current_price) × qty
        
        Returns:
            Float: Total value (e.g., 15234.56 USDC)
        
        Example:
            >>> broker.get_portfolio_value()
            15234.56
        
        LLM Note: Use this to calculate position sizes. 
        If you want 10% in BTC: target_value = 0.10 * get_portfolio_value()
        """
        holdings = self._get_holdings_dict()
        total = 0.0
        long_value = 0.0
        short_unrealized_pnl = 0.0
        
        # Get short entries for P&L calculation
        short_entries = {}
        if self.paper_trading:
            short_entries = self._get_paper_short_entries()
        
        for asset, amount in holdings.items():
            if asset == self.stable_coin:
                total += amount  # Cash balance
            elif asset in self.exclusions:
                continue
            else:
                price = self._get_price_cached(asset)
                if price is not None:
                    if amount > 0:
                        # Long position
                        long_value += amount * price
                    elif amount < 0:
                        # Short position - calculate unrealized P&L
                        short_qty = abs(amount)
                        entry = short_entries.get(asset, {})
                        entry_price = entry.get('price', price)
                        # Unrealized P&L = (entry - current) × qty
                        unrealized = (entry_price - price) * short_qty
                        short_unrealized_pnl += unrealized
        
        # Total = cash + long_value + short_unrealized_pnl
        total += long_value + short_unrealized_pnl
        
        logger.info(f"Portfolio value: {total:.2f} {self.stable_coin} (longs: {long_value:.2f}, short P&L: {short_unrealized_pnl:.2f})")
        return max(0, total)
    
    def generate_rebalancing(self, target_weights: Dict[str, float]) -> pd.DataFrame:
        """
        Analyze what trades are needed to reach target weights (NO EXECUTION).
        
        This is a safe, read-only operation. Use it to preview trades before executing.
        
        Args:
            target_weights: Dict mapping asset symbols to target weights.
                           Weights should be between 0.0 and 1.0.
                           Weights don't need to sum to 1.0 - remainder stays in cash.
        
        Returns:
            DataFrame with columns:
            - Asset: symbol (BTC, ETH, etc.)
            - Symbol: trading pair (BTCUSDC)
            - Current_Qty, Current_Value, Current_Weight
            - Target_Qty, Target_Value, Target_Weight
            - Weight_Delta: target - current weight
            - Delta_Qty: how much to buy/sell
            - Price: current price
            - Action: 'Buy', 'Sell', or 'Hold'
        
        Example:
            >>> weights = {'BTC': 0.40, 'ETH': 0.30, 'SOL': 0.15}
            >>> analysis = broker.generate_rebalancing(weights)
            >>> print(analysis[['Asset', 'Current_Weight', 'Target_Weight', 'Action']])
              Asset  Current_Weight  Target_Weight Action
            0   BTC            0.35           0.40    Buy
            1   ETH            0.25           0.30    Buy
            2   SOL            0.20           0.15   Sell
        
        LLM Note: 
        - Call this BEFORE execute_rebalancing() to preview trades
        - Check the 'Action' column to see what will happen
        - If Delta_Qty is very small, the trade may be skipped (below threshold)
        """
        holdings = self._get_holdings_dict()
        total_value = self.get_portfolio_value()
        
        if total_value <= 0:
            logger.error("Portfolio value is zero or negative")
            return pd.DataFrame()
        
        # Apply capital at risk
        adjusted_weights = {
            asset: weight * self.capital_at_risk 
            for asset, weight in target_weights.items()
        }
        
        # Build analysis
        assets = sorted(set(holdings.keys()) | set(adjusted_weights.keys()))
        assets = [a for a in assets if a not in self.exclusions]
        
        data = []
        for asset in assets:
            current_qty = holdings.get(asset, 0.0)
            price = self._get_price_cached(asset)
            current_value = current_qty * price if price else 0
            current_weight = current_value / total_value if total_value > 0 else 0
            
            target_weight = adjusted_weights.get(asset, 0.0)
            target_value = total_value * target_weight
            target_qty = target_value / price if price and price > 0 else 0
            
            delta_qty = target_qty - current_qty
            weight_delta = target_weight - current_weight
            
            if delta_qty > 0.0001:
                action = "Buy"
            elif delta_qty < -0.0001:
                action = "Sell"
            else:
                action = "Hold"
            
            data.append({
                'Asset': asset,
                'Symbol': f"{asset}{self.stable_coin}",
                'Current_Qty': current_qty,
                'Current_Value': current_value,
                'Current_Weight': current_weight,
                'Target_Qty': target_qty,
                'Target_Value': target_value,
                'Target_Weight': target_weight,
                'Weight_Delta': weight_delta,
                'Delta_Qty': delta_qty,
                'Price': price,
                'Action': action,
            })
        
        return pd.DataFrame(data)
    
    def execute_rebalancing(self, target_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Execute full portfolio rebalancing - THE MAIN TRADING METHOD.
        
        This method:
        1. Calculates required trades from target weights
        2. Validates all orders (lot size, min notional, etc.)
        3. Executes sells first (to free up capital)
        4. Scales buy orders to available cash
        5. Persists trade history
        
        Args:
            target_weights: Dict mapping asset symbols to target weights (0.0-1.0).
                           Example: {'BTC': 0.40, 'ETH': 0.30, 'SOL': 0.15}
        
        Returns:
            Dict containing:
            - orders: DataFrame of all considered orders with status
            - rebalancing: Analysis DataFrame (same as generate_rebalancing)
            - total_value: Portfolio value after execution
            - executed_orders: List of successfully executed trades
            - failed_orders: List of failed trades
            - summary: {total_executed, total_failed, total_value, total_cost}
            - paper_trading: bool indicating mode
        
        Example:
            >>> weights = {'BTC': 0.40, 'ETH': 0.30}
            >>> result = broker.execute_rebalancing(weights)
            >>> print(f"Executed {result['summary']['total_executed']} trades")
            >>> print(f"Portfolio now worth {result['total_value']:.2f}")
        
        LLM Note:
        - In paper_trading mode, this is safe to call - no real money moves
        - In live mode, this EXECUTES REAL TRADES - use with caution
        - Always call generate_rebalancing() first to preview
        - Check result['summary'] for execution stats
        - Check result['orders']['Order_Status'] for why orders were skipped
        """
        # Generate rebalancing plan
        rebal_df = self.generate_rebalancing(target_weights)
        if rebal_df.empty:
            return {'orders': pd.DataFrame(), 'rebalancing': rebal_df, 'total_value': 0}
        
        # Generate orders with constraints
        orders_df = self._generate_orders_from_rebalancing(rebal_df)
        
        # Execute
        if not self.readonly:
            execution_report = self._execute_orders(orders_df)
        else:
            execution_report = {
                'executed_orders': [],
                'failed_orders': [],
                'summary': {'total_executed': 0, 'total_failed': 0},
                'readonly': True
            }
        
        execution_report['orders'] = orders_df
        execution_report['rebalancing'] = rebal_df
        execution_report['total_value'] = self.get_portfolio_value()
        
        return execution_report
    
    # ========================================================================
    # Internal Methods
    # ========================================================================
    
    def _get_holdings_dict(self) -> Dict[str, float]:
        """Get holdings as dict (works for both paper and live)."""
        if self.paper_trading:
            return self._get_paper_holdings()
        return self._get_live_holdings()
    
    def _get_live_holdings(self) -> Dict[str, float]:
        """Fetch live holdings from Binance."""
        if self._client is None:
            logger.warning("No Binance client - returning empty holdings")
            return {}
        
        holdings = {}
        try:
            account_info = self._client.get_account(recvWindow=60000)
            for item in account_info.get('balances', []):
                amount = float(item.get('free', 0)) + float(item.get('locked', 0))
                if amount != 0:
                    holdings[item['asset']] = amount
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
        return holdings
    
    def _get_paper_holdings(self) -> Dict[str, float]:
        """Get paper trading holdings from file."""
        paper_file = os.path.join(
            self.trades_dir, self.account_name.lower(), "paper_portfolio.json"
        )
        if os.path.exists(paper_file):
            with open(paper_file) as f:
                return json.load(f)
        # Default: start with stable coin balance
        return {self.stable_coin: 10000.0}
    
    def _get_paper_short_entries(self) -> Dict[str, Dict]:
        """Get short position entry prices for P&L calculation."""
        short_file = os.path.join(
            self.trades_dir, self.account_name.lower(), "short_entries.json"
        )
        if os.path.exists(short_file):
            with open(short_file) as f:
                return json.load(f)
        return {}
    
    def _save_paper_short_entries(self, entries: Dict[str, Dict]) -> None:
        """Save short position entry prices."""
        paper_dir = os.path.join(self.trades_dir, self.account_name.lower())
        os.makedirs(paper_dir, exist_ok=True)
        short_file = os.path.join(paper_dir, "short_entries.json")
        with open(short_file, 'w') as f:
            json.dump(entries, f, indent=2)
    
    def _save_paper_holdings(self, holdings: Dict[str, float]) -> None:
        """Save paper trading holdings to file."""
        paper_dir = os.path.join(self.trades_dir, self.account_name.lower())
        os.makedirs(paper_dir, exist_ok=True)
        paper_file = os.path.join(paper_dir, "paper_portfolio.json")
        with open(paper_file, 'w') as f:
            json.dump(holdings, f, indent=2)
    
    def _get_price_cached(self, asset: str) -> Optional[float]:
        """Get price with caching."""
        # Check cache
        cached_price, is_cached = self._price_cache.get(asset)
        if is_cached:
            return cached_price
        
        # Fetch with retry
        price = self._fetch_price_with_retry(asset)
        self._price_cache.set(asset, price)
        return price
    
    def _fetch_price_with_retry(self, asset: str) -> Optional[float]:
        """Fetch price from Binance with retry logic and BTC fallback."""
        # Paper mode without client - return None (use cached/mock prices)
        if self._client is None:
            logger.debug(f"No client - cannot fetch price for {asset}")
            return None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                symbol = f"{asset}{self.stable_coin}"
                response = self._client.get_symbol_ticker(symbol=symbol)
                if response:
                    return float(response['price'])
                return None
            except BinanceAPIException as e:
                if e.code == -1121:  # Invalid symbol
                    # Try BTC route
                    return self._get_price_via_btc(asset)
                elif _is_transient_error(e) and attempt < self.max_retries:
                    backoff = 2 ** (attempt - 1)
                    logger.warning(f"Transient error for {asset}, retry {attempt}/{self.max_retries} in {backoff}s")
                    time.sleep(backoff)
                    continue
                else:
                    logger.error(f"Binance API error for {asset}: {e}")
                    return None
            except Exception as e:
                if _is_transient_error(e) and attempt < self.max_retries:
                    backoff = 2 ** (attempt - 1)
                    time.sleep(backoff)
                    continue
                logger.error(f"Error fetching price for {asset}: {e}")
                return None
        return None
    
    def _get_price_via_btc(self, asset: str) -> Optional[float]:
        """Get price via BTC as intermediate pair."""
        if self._client is None:
            return None
        try:
            btc_price = float(self._client.get_symbol_ticker(symbol=f"BTC{self.stable_coin}")['price'])
            asset_btc = float(self._client.get_symbol_ticker(symbol=f"{asset}BTC")['price'])
            return asset_btc * btc_price
        except Exception:
            logger.warning(f"No valid market for {asset}")
            return None
    
    def _get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Get symbol info from Binance."""
        if self._client is None:
            # Return mock symbol info for paper trading
            return {
                'filters': [
                    {'filterType': 'LOT_SIZE', 'minQty': '0.00001', 'stepSize': '0.00001'},
                    {'filterType': 'NOTIONAL', 'minNotional': '5.0'}
                ]
            }
        try:
            return self._client.get_symbol_info(symbol)
        except Exception as e:
            logger.error(f"Error fetching symbol info for {symbol}: {e}")
            return None
    
    def _generate_orders_from_rebalancing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate validated orders from rebalancing DataFrame."""
        cash_available = self._get_holdings_dict().get(self.stable_coin, 0.0)
        
        order_records = []
        
        # First pass: process sells
        for _, row in df.iterrows():
            if row['Action'] != 'Sell':
                continue
            
            order = self._prepare_order(row, is_buy=False)
            order_records.append(order)
            
            if order['Order_Status'] == 'To_be_placed':
                cash_available += order['Notional_Value']
        
        # Second pass: process buys with scaling
        buy_rows = df[df['Action'] == 'Buy']
        total_buy_value = sum(
            abs(row['Delta_Qty']) * row['Price'] 
            for _, row in buy_rows.iterrows() 
            if row['Price']
        )
        
        scaling_factor = min(1.0, cash_available / total_buy_value) if total_buy_value > 0 else 0.0
        
        if scaling_factor < self.scaling_factor_min:
            logger.warning(f"Scaling factor {scaling_factor:.2f} < min {self.scaling_factor_min}")
            # Don't place any buy orders if scaling is too aggressive
            for _, row in buy_rows.iterrows():
                order = {
                    'Asset': row['Asset'],
                    'Symbol': row['Symbol'],
                    'Action': 'Buy',
                    'Raw_Qty': abs(row['Delta_Qty']),
                    'Adjusted_Qty': 0.0,
                    'Price': row['Price'],
                    'Notional_Value': 0.0,
                    'Order_Status': 'Scaling_rejected',
                    'Reason': f'Scaling {scaling_factor:.2f} < min {self.scaling_factor_min}',
                    'Scaling_Factor': scaling_factor,
                }
                order_records.append(order)
        else:
            for _, row in buy_rows.iterrows():
                order = self._prepare_order(row, is_buy=True, scaling_factor=scaling_factor)
                order_records.append(order)
        
        orders_df = pd.DataFrame(order_records)
        if not orders_df.empty:
            orders_df['Executable'] = (
                (orders_df['Adjusted_Qty'] > 0) & 
                (orders_df['Order_Status'] == 'To_be_placed')
            )
        return orders_df
    
    def _prepare_order(
        self, 
        row: pd.Series, 
        is_buy: bool, 
        scaling_factor: float = 1.0
    ) -> Dict[str, Any]:
        """Prepare a single order with validations."""
        asset = row['Asset']
        symbol = row['Symbol']
        price = row['Price']
        raw_qty = abs(row['Delta_Qty'])
        
        # Get symbol constraints
        symbol_info = self._get_symbol_info(symbol)
        min_qty, step_size, min_notional = get_lot_size_and_min_notional(symbol_info)
        if min_notional == 0:
            min_notional = self.min_notional
        
        # Check weight delta threshold (skip small rebalances)
        zero_target_sell = (
            not is_buy and 
            row.get('Target_Weight', 0) == 0 and 
            row.get('Current_Qty', 0) > 0
        )
        
        if abs(row['Weight_Delta']) < self.min_trade_size and not zero_target_sell:
            return {
                'Asset': asset,
                'Symbol': symbol,
                'Action': 'Buy' if is_buy else 'Sell',
                'Raw_Qty': raw_qty,
                'Adjusted_Qty': 0.0,
                'Price': price,
                'Notional_Value': 0.0,
                'Min_Notional': min_notional,
                'Min_Qty': min_qty,
                'Step_Size': step_size,
                'Order_Status': 'Below_threshold',
                'Reason': f'Weight delta < {self.min_trade_size}',
                'Scaling_Factor': scaling_factor if is_buy else None,
            }
        
        # Apply scaling for buys
        qty = raw_qty * scaling_factor if is_buy else raw_qty
        
        # Adjust to step size
        adjusted_qty = adjust_quantity(qty, step_size) if step_size else qty
        notional_value = adjusted_qty * price if price else 0.0
        
        # Validate
        status = 'To_be_placed'
        reason = ''
        
        if price is None or price == 0:
            status = 'Zero_price'
            reason = 'No price available'
            adjusted_qty = 0.0
        elif notional_value < min_notional:
            status = 'Below_min_notional'
            reason = f'Notional {notional_value:.4f} < {min_notional:.4f}'
            adjusted_qty = 0.0
        elif adjusted_qty < min_qty:
            status = 'Below_min_qty'
            reason = f'Qty {adjusted_qty:.8f} < {min_qty:.8f}'
            adjusted_qty = 0.0
        
        return {
            'Asset': asset,
            'Symbol': symbol,
            'Action': 'Buy' if is_buy else 'Sell',
            'Raw_Qty': raw_qty,
            'Adjusted_Qty': adjusted_qty,
            'Price': price,
            'Notional_Value': notional_value,
            'Min_Notional': min_notional,
            'Min_Qty': min_qty,
            'Step_Size': step_size,
            'Order_Status': status,
            'Reason': reason,
            'Scaling_Factor': scaling_factor if is_buy else None,
        }
    
    def _execute_single_order(
        self, 
        symbol: str, 
        side: str, 
        qty: float, 
        price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute a single order."""
        qty_str = format_quantity_for_binance(qty)
        
        try:
            if self.paper_trading:
                return self._execute_paper_order(symbol, side, qty, price)
            
            if price is None or price == 0:
                res = self._client.create_order(
                    symbol=symbol, 
                    side=side, 
                    type="MARKET", 
                    quantity=qty_str
                )
            else:
                res = self._client.create_order(
                    symbol=symbol, 
                    side=side, 
                    type="LIMIT", 
                    timeInForce="GTC", 
                    quantity=qty_str, 
                    price=str(price)
                )
            
            # Parse fills
            for f in res.get("fills", []) or []:
                return {
                    "symbol": symbol,
                    "side": side.lower(),
                    "qty": float(f.get("qty", 0.0)),
                    "price": float(f.get("price", 0.0)),
                    "commission": float(f.get("commission", 0.0)),
                    "commission_asset": f.get("commissionAsset"),
                    "execution_mode": "LIVE",
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Order execution failed for {symbol}: {e}")
            return None
    
    def _execute_paper_order(
        self, 
        symbol: str, 
        side: str, 
        qty: float, 
        price: Optional[float]
    ) -> Dict[str, Any]:
        """
        Execute a paper trading order with realistic market simulation.
        
        Simulates:
        - Bid-ask spread (buy at ask, sell at bid)
        - Price impact based on order size
        - Random slippage
        - Trading commission
        - Optional fill delay
        
        Returns:
            Fill dict with execution details and cost breakdown
        """
        import random
        
        # Extract base asset
        base = symbol.replace(self.stable_coin, "")
        
        # Get current mid-market price
        base_price = price if price is not None else self._get_price_cached(base)
        
        if base_price is None:
            raise ValueError(f"Cannot get price for {base}")
        
        # Apply market simulation if configured
        if self.market_sim:
            exec_price, sim_breakdown = simulate_execution_price(
                base_price=base_price,
                side=side,
                quantity=qty,
                config=self.market_sim,
            )
            commission = calculate_commission(qty, exec_price, self.market_sim)
        else:
            # Instant fill at mid price (no costs)
            exec_price = base_price
            commission = 0.0
            sim_breakdown = {'base_price': base_price, 'total_cost_bps': 0}
        
        # Update holdings
        holdings = self._get_paper_holdings()
        short_entries = self._get_paper_short_entries()
        
        current_qty = holdings.get(base, 0)
        
        if side.upper() == "BUY":
            # Buying: either opening long or closing short
            cost = qty * exec_price + commission
            holdings[self.stable_coin] = holdings.get(self.stable_coin, 0) - cost
            
            if current_qty < 0:
                # Closing short position - realize P&L
                short_qty_to_close = min(qty, abs(current_qty))
                if base in short_entries:
                    entry_price = short_entries[base].get('price', exec_price)
                    # Short P&L = (entry_price - exit_price) * qty
                    pnl = (entry_price - exec_price) * short_qty_to_close
                    holdings[self.stable_coin] = holdings.get(self.stable_coin, 0) + pnl
                    
                    # Update or remove short entry
                    remaining_short = abs(current_qty) - short_qty_to_close
                    if remaining_short <= 1e-10:
                        short_entries.pop(base, None)
                    else:
                        short_entries[base]['qty'] = remaining_short
            
            holdings[base] = current_qty + qty
            
        else:  # SELL
            # Selling: either closing long or opening short
            proceeds = qty * exec_price - commission
            holdings[self.stable_coin] = holdings.get(self.stable_coin, 0) + proceeds
            
            new_qty = current_qty - qty
            
            if new_qty < 0 and current_qty >= 0:
                # Opening new short position
                short_qty = abs(new_qty)
                short_entries[base] = {
                    'qty': short_qty,
                    'price': exec_price,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                }
            elif new_qty < 0 and current_qty < 0:
                # Adding to existing short
                existing = short_entries.get(base, {})
                existing_qty = existing.get('qty', 0)
                existing_price = existing.get('price', exec_price)
                # Average entry price
                total_qty = existing_qty + qty
                avg_price = (existing_qty * existing_price + qty * exec_price) / total_qty
                short_entries[base] = {
                    'qty': total_qty,
                    'price': avg_price,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                }
            
            holdings[base] = new_qty
            
            # Clean up zero holdings
            if abs(holdings.get(base, 0)) <= 1e-10:
                holdings.pop(base, None)
                short_entries.pop(base, None)
        
        self._save_paper_holdings(holdings)
        self._save_paper_short_entries(short_entries)
        
        # Return detailed fill info
        return {
            "symbol": symbol,
            "side": side.lower(),
            "qty": qty,
            "price": exec_price,
            "base_price": base_price,
            "commission": commission,
            "commission_asset": self.stable_coin,
            "execution_mode": "PAPER",
            "simulation": {
                "spread_cost": sim_breakdown.get('spread_cost', 0),
                "impact_cost": sim_breakdown.get('impact_cost', 0),
                "slippage_cost": sim_breakdown.get('slippage_cost', 0),
                "total_cost_bps": sim_breakdown.get('total_cost_bps', 0),
                "delay_ms": sim_breakdown.get('delay_ms', 0),
            }
        }
    
    def _execute_orders(self, orders_df: pd.DataFrame) -> Dict[str, Any]:
        """Execute all orders from orders DataFrame."""
        report = {
            'executed_orders': [],
            'failed_orders': [],
            'summary': {
                'total_executed': 0,
                'total_failed': 0,
                'total_value': 0.0,
                'total_cost': 0.0,
            },
            'paper_trading': self.paper_trading,
        }
        
        if orders_df.empty:
            return report
        
        executable = orders_df[orders_df.get('Executable', False)]
        if executable.empty:
            return report
        
        # Sort: sells first
        executable = executable.sort_values(
            by='Action', 
            ascending=False  # Sell before Buy
        )
        
        for _, order in executable.iterrows():
            symbol = order['Symbol']
            side = order['Action'].upper()
            qty = order['Adjusted_Qty']
            
            fill = self._execute_single_order(symbol, side, qty)
            
            if fill:
                report['executed_orders'].append(fill)
                report['summary']['total_executed'] += 1
                report['summary']['total_value'] += fill.get('qty', 0) * fill.get('price', 0)
            else:
                report['failed_orders'].append({
                    'symbol': symbol,
                    'side': side,
                    'qty': qty,
                })
                report['summary']['total_failed'] += 1
        
        # Save trade history
        if self.persist_trades and report['executed_orders']:
            self._save_trade_history(report)
        
        return report
    
    def _save_trade_history(self, report: Dict[str, Any]) -> Optional[str]:
        """Persist trade history to JSON file."""
        trades = report.get('executed_orders', [])
        if not trades:
            return None
        
        now_utc = datetime.now(timezone.utc)
        date_str = now_utc.strftime('%Y-%m-%d')
        time_str = now_utc.strftime('%H%M%S')
        
        trades_dir = os.path.join(self.trades_dir, self.account_name.lower(), "trades")
        os.makedirs(trades_dir, exist_ok=True)
        
        filename = f"{date_str}--{time_str}.json"
        filepath = os.path.join(trades_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                'date': date_str,
                'timestamp': now_utc.isoformat(),
                'paper_trading': self.paper_trading,
                'summary': report.get('summary', {}),
                'trades': trades,
            }, f, indent=2, default=str)
        
        logger.info(f"Saved {len(trades)} trade(s) to {filepath}")
        return filepath


# ============================================================================
# Universe Selection (ported from quantlab crypto_trend_catcher)
# ============================================================================

# Default stablecoins to exclude from universe
DEFAULT_STABLECOIN_EXCLUSIONS = [
    'USDT', 'USDC', 'BUSD', 'TUSD', 'DAI', 'MIM', 'USTC', 'FDUSD',
    'USDP', 'GUSD', 'FRAX', 'LUSD', 'USDD', 'PYUSD', 'EURC', 'EURT',
]


@dataclass
class UniverseConfig:
    """
    Configuration for universe selection.
    
    Universe selection filters tradable assets based on:
    1. Market cap rank (top N by market cap)
    2. Volume rank (top M by dollar volume within market cap filter)
    3. Exclusions (stablecoins, wrapped tokens, etc.)
    
    Example:
        >>> config = UniverseConfig(
        ...     market_cap_top_n=30,      # Consider top 30 by market cap
        ...     portfolio_max_coins=10,    # Trade top 10 by volume
        ...     exclude_stablecoins=True,  # Exclude USDT, USDC, etc.
        ... )
    """
    # Market cap filter: include top N coins by market cap
    market_cap_top_n: int = 30
    
    # Volume filter: from market_cap_filtered coins, select top M by volume
    portfolio_max_coins: int = 10
    
    # Stablecoin handling
    exclude_stablecoins: bool = True
    stablecoin_list: List[str] = field(default_factory=lambda: DEFAULT_STABLECOIN_EXCLUSIONS.copy())
    
    # Additional exclusions (e.g., wrapped tokens, delisted coins)
    additional_exclusions: List[str] = field(default_factory=lambda: ['ETHW', 'BETH', 'WBTC', 'WETH'])
    
    # Minimum thresholds (optional, filter before ranking)
    min_market_cap_usd: float = 0.0        # Minimum market cap (0 = no filter)
    min_daily_volume_usd: float = 0.0       # Minimum daily volume (0 = no filter)
    
    def get_all_exclusions(self) -> List[str]:
        """Get combined list of all excluded tickers."""
        exclusions = list(self.additional_exclusions)
        if self.exclude_stablecoins:
            exclusions.extend(self.stablecoin_list)
        return list(set(exclusions))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            'market_cap_top_n': self.market_cap_top_n,
            'portfolio_max_coins': self.portfolio_max_coins,
            'exclude_stablecoins': self.exclude_stablecoins,
            'stablecoin_list': self.stablecoin_list,
            'additional_exclusions': self.additional_exclusions,
            'min_market_cap_usd': self.min_market_cap_usd,
            'min_daily_volume_usd': self.min_daily_volume_usd,
        }


@dataclass
class UniverseSelector:
    """
    Selects tradable asset universe based on market cap and volume filters.
    
    Ported from quantlab's crypto_trend_catcher strategy. Filters assets through:
    1. Exclusion filter (remove stablecoins, wrapped tokens, etc.)
    2. Market cap filter (top N by market cap)
    3. Volume filter (top M by dollar volume from market-cap-filtered set)
    
    ## LLM Usage Guide
    
    ### Quick Start
    ```python
    from quantbox.plugins.broker import UniverseSelector, UniverseConfig
    
    # Create selector with default config
    selector = UniverseSelector()
    
    # Or with custom config
    selector = UniverseSelector(config=UniverseConfig(
        market_cap_top_n=50,      # Wider initial filter
        portfolio_max_coins=15,    # More coins in portfolio
    ))
    ```
    
    ### Selecting Universe
    ```python
    # Input: DataFrames with DatetimeIndex and asset columns
    # prices: daily close prices
    # volume: daily volume (in asset units)
    # market_cap: daily market cap (in USD)
    
    universe = selector.select(
        prices=prices_df,
        volume=volume_df,
        market_cap=market_cap_df,
    )
    # Returns: DataFrame of 0/1 flags (1 = asset is in universe that day)
    ```
    
    ### Filtering Weights
    ```python
    # Apply universe mask to strategy weights
    filtered_weights = selector.apply_mask(
        weights=raw_weights_df,
        universe=universe_df,
        normalize=True,  # Re-normalize to sum to 1
    )
    ```
    
    ### Get Current Universe (Single Day)
    ```python
    # Get today's tradable assets as a list
    current_assets = selector.get_current_universe(
        prices=prices_df,
        volume=volume_df,
        market_cap=market_cap_df,
    )
    # Returns: ['BTC', 'ETH', 'SOL', 'BNB', ...] (list of tickers)
    ```
    """
    
    config: UniverseConfig = field(default_factory=UniverseConfig)
    
    def describe(self) -> Dict[str, Any]:
        """
        Describe universe selector for LLM introspection.
        
        Returns dict with:
        - purpose: What this selector does
        - config: Current configuration
        - methods: Available methods with signatures
        - example: Usage example
        """
        return {
            "purpose": "Selects tradable asset universe based on market cap and volume filters",
            "config": self.config.to_dict(),
            "filters": {
                "1_exclusions": f"Remove {len(self.config.get_all_exclusions())} excluded tickers (stablecoins, wrapped tokens)",
                "2_market_cap": f"Rank by market cap, keep top {self.config.market_cap_top_n}",
                "3_volume": f"From market-cap-filtered, rank by dollar volume, keep top {self.config.portfolio_max_coins}",
            },
            "methods": {
                "select(prices, volume, market_cap)": "Returns DataFrame of 0/1 universe flags",
                "get_current_universe(prices, volume, market_cap)": "Returns list of current tradable tickers",
                "apply_mask(weights, universe, normalize)": "Filter weights by universe, optionally normalize",
            },
            "example": """
# Select universe and filter weights
universe = selector.select(prices, volume, market_cap)
filtered_weights = selector.apply_mask(strategy_weights, universe, normalize=True)
            """,
        }
    
    def select(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        market_cap: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Select tradable universe for each day in the data.
        
        Algorithm (per day):
        1. Exclude stablecoins and other excluded tickers
        2. Rank remaining assets by market cap (descending)
        3. Filter to top `market_cap_top_n` by market cap
        4. Compute dollar volume (price * volume)
        5. Within market-cap-filtered set, rank by dollar volume
        6. Select top `portfolio_max_coins` by dollar volume
        
        Args:
            prices: DataFrame with DatetimeIndex, columns = asset tickers, values = prices
            volume: DataFrame with same shape, values = daily volume (asset units)
            market_cap: DataFrame with same shape, values = market cap (USD)
            
        Returns:
            DataFrame with same shape, values = 0.0 or 1.0 (1 = in universe)
            
        Example:
            >>> universe = selector.select(prices, volume, market_cap)
            >>> universe.iloc[-1]  # Today's universe
            BTC     1.0
            ETH     1.0
            SOL     1.0
            USDT    0.0  # Excluded (stablecoin)
            ...
        """
        # Get valid tickers (exclude stablecoins + additional exclusions)
        exclusions = self.config.get_all_exclusions()
        valid_tickers = [t for t in prices.columns if t not in exclusions]
        
        if not valid_tickers:
            logger.warning("No valid tickers after exclusions")
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        
        # Subset to valid tickers
        prices_valid = prices[valid_tickers]
        volume_valid = volume[valid_tickers]
        mc_valid = market_cap[valid_tickers]
        
        # Apply minimum thresholds if configured
        if self.config.min_market_cap_usd > 0:
            mc_valid = mc_valid.where(mc_valid >= self.config.min_market_cap_usd)
        
        dollar_volume = prices_valid * volume_valid
        if self.config.min_daily_volume_usd > 0:
            dollar_volume = dollar_volume.where(dollar_volume >= self.config.min_daily_volume_usd)
        
        # Step 1: Rank by market cap (descending), lower rank = larger market cap
        mc_rank = mc_valid.rank(axis=1, ascending=False, method='min')
        
        # Step 2: Market cap filter - keep top N
        mc_flag = mc_rank <= self.config.market_cap_top_n
        
        # Step 3: Within MC-filtered, compute dollar volume rank
        # Only rank assets that pass market cap filter
        vol_for_ranking = dollar_volume.where(mc_flag)
        vol_rank = vol_for_ranking.rank(axis=1, ascending=False, method='min')
        
        # Step 4: Volume filter - keep top M by volume (within MC-filtered)
        universe_valid = (vol_rank <= self.config.portfolio_max_coins).astype(float)
        
        # Create full universe DataFrame (include excluded tickers as 0)
        universe = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        universe[valid_tickers] = universe_valid
        
        return universe
    
    def get_current_universe(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        market_cap: pd.DataFrame,
    ) -> List[str]:
        """
        Get list of currently tradable tickers (based on latest data).
        
        Convenience method that returns the universe for the most recent date
        as a simple list of ticker strings.
        
        Args:
            prices: Price DataFrame (DatetimeIndex)
            volume: Volume DataFrame
            market_cap: Market cap DataFrame
            
        Returns:
            List of ticker strings in the current universe
            
        Example:
            >>> current = selector.get_current_universe(prices, volume, market_cap)
            >>> print(current)
            ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'DOT']
        """
        universe = self.select(prices, volume, market_cap)
        latest = universe.iloc[-1]
        return latest[latest == 1.0].index.tolist()
    
    def apply_mask(
        self,
        weights: pd.DataFrame,
        universe: pd.DataFrame,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Apply universe mask to strategy weights.
        
        Zeroes out weights for assets not in the universe. Optionally re-normalizes
        remaining weights to sum to 1 (or the original sum).
        
        Args:
            weights: DataFrame of strategy weights (DatetimeIndex, asset columns)
            universe: DataFrame of 0/1 flags from select()
            normalize: If True, re-normalize weights to sum to 1 per row
            
        Returns:
            DataFrame of filtered weights
            
        Example:
            >>> # Raw weights from strategy
            >>> raw = pd.DataFrame({'BTC': [0.5], 'ETH': [0.3], 'USDT': [0.2]})
            >>> # Universe excludes USDT
            >>> universe = pd.DataFrame({'BTC': [1.0], 'ETH': [1.0], 'USDT': [0.0]})
            >>> filtered = selector.apply_mask(raw, universe, normalize=True)
            >>> # BTC: 0.625, ETH: 0.375 (renormalized to sum to 1)
        """
        # Align columns
        common_cols = weights.columns.intersection(universe.columns)
        
        # Apply mask
        masked = weights[common_cols].mul(universe[common_cols])
        
        if normalize:
            # Re-normalize: divide by row sum (number of assets in universe)
            row_sums = masked.sum(axis=1).replace(0, float('nan'))
            masked = masked.div(row_sums, axis=0)
        
        return masked
    
    def get_universe_stats(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        market_cap: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Get statistics about the universe selection.
        
        Useful for debugging and understanding what assets are selected.
        
        Returns:
            Dict with selection stats and breakdown
        """
        exclusions = self.config.get_all_exclusions()
        valid_tickers = [t for t in prices.columns if t not in exclusions]
        excluded_tickers = [t for t in prices.columns if t in exclusions]
        
        universe = self.select(prices, volume, market_cap)
        current_universe = self.get_current_universe(prices, volume, market_cap)
        
        # Get latest market caps for context
        latest_mc = market_cap.iloc[-1]
        latest_prices = prices.iloc[-1]
        
        return {
            "config": self.config.to_dict(),
            "total_tickers": len(prices.columns),
            "valid_tickers": len(valid_tickers),
            "excluded_tickers": excluded_tickers,
            "current_universe": current_universe,
            "current_universe_size": len(current_universe),
            "universe_market_caps": {
                t: latest_mc.get(t, 0) for t in current_universe
            },
            "universe_prices": {
                t: latest_prices.get(t, 0) for t in current_universe
            },
            "avg_universe_size": universe.sum(axis=1).mean(),
            "date_range": {
                "start": str(prices.index[0]),
                "end": str(prices.index[-1]),
                "days": len(prices.index),
            },
        }
