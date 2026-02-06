"""
Binance Market Data Plugin - Live market data for AI-powered trading.

This module provides the BinanceDataFetcher class - a production-grade data fetcher
for Binance market data, designed for use by LLM agents and quantitative strategies.

## LLM Usage Guide

### Quick Start (No API Key Needed)
```python
from quantbox.plugins.datasources import BinanceDataFetcher

# Initialize fetcher
fetcher = BinanceDataFetcher()

# Get current prices
prices = fetcher.get_current_prices(['BTC', 'ETH', 'SOL'])
# {'BTC': 95000.0, 'ETH': 3200.0, 'SOL': 190.0}

# Get historical OHLCV data
data = fetcher.get_market_data(
    tickers=['BTC', 'ETH', 'SOL'],
    lookback_days=90
)
# Returns dict with 'prices', 'volume', 'market_cap' DataFrames
```

### Strategy Integration
```python
# Get data for strategy
data = fetcher.get_market_data(tickers=['BTC', 'ETH', 'SOL'], lookback_days=400)

# Run strategy
result = strategy.run(data=data, params=params)

# Get current snapshot for trading
snapshot = fetcher.get_snapshot(['BTC', 'ETH', 'SOL'])
# DataFrame with: ticker, price, volume_24h, change_24h
```

### Key Methods
- `get_current_prices(tickers)` → Dict of current prices
- `get_market_data(tickers, lookback_days)` → Full market data dict
- `get_snapshot(tickers)` → Current market snapshot DataFrame
- `get_valid_pairs(tickers)` → Check which pairs trade on Binance
- `describe()` → LLM-friendly capability description

### No API Key Required
Public Binance endpoints (prices, OHLCV) don't require authentication.
Only private endpoints (account, orders) need API keys.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import time
import requests

logger = logging.getLogger(__name__)

# Try to import ccxt for OHLCV data
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    ccxt = None
    CCXT_AVAILABLE = False
    logger.warning("ccxt not installed - OHLCV fetching will be limited")


# ============================================================================
# Constants
# ============================================================================

# Default stablecoins to exclude
DEFAULT_STABLECOINS = [
    'USDT', 'USDC', 'BUSD', 'TUSD', 'DAI', 'MIM', 'USTC', 'FDUSD',
    'USDP', 'GUSD', 'FRAX', 'LUSD', 'USDD', 'PYUSD', 'EURC', 'EURT',
]

# Binance API endpoints
BINANCE_API_BASE = "https://api.binance.com"
BINANCE_EXCHANGE_INFO = f"{BINANCE_API_BASE}/api/v3/exchangeInfo"
BINANCE_TICKER_PRICE = f"{BINANCE_API_BASE}/api/v3/ticker/price"
BINANCE_TICKER_24H = f"{BINANCE_API_BASE}/api/v3/ticker/24hr"

# Rate limiting
DEFAULT_REQUEST_DELAY_MS = 100  # 100ms between requests
DEFAULT_MAX_RETRIES = 3


# ============================================================================
# Market Data Snapshot
# ============================================================================

@dataclass
class MarketDataSnapshot:
    """
    Point-in-time market data snapshot.
    
    Captures current prices, volumes, and changes for a set of assets.
    Useful for trading decisions and portfolio analysis.
    """
    timestamp: datetime
    data: pd.DataFrame  # ticker, price, volume_24h, change_24h, market_cap
    quote_asset: str = "USDT"
    
    def get_price(self, ticker: str) -> Optional[float]:
        """Get price for a single ticker."""
        if ticker in self.data['ticker'].values:
            return self.data.loc[self.data['ticker'] == ticker, 'price'].iloc[0]
        return None
    
    def get_prices_dict(self) -> Dict[str, float]:
        """Get all prices as a dict."""
        return dict(zip(self.data['ticker'], self.data['price']))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'quote_asset': self.quote_asset,
            'data': self.data.to_dict('records'),
        }


# ============================================================================
# Binance Data Fetcher
# ============================================================================

@dataclass
class BinanceDataFetcher:
    """
    Production Binance market data fetcher.
    
    Fetches live and historical market data from Binance public APIs.
    No API key required for public data (prices, OHLCV).
    
    ## LLM Usage Guide
    
    ### Get Current Prices
    ```python
    fetcher = BinanceDataFetcher()
    prices = fetcher.get_current_prices(['BTC', 'ETH', 'SOL'])
    # {'BTC': 95000.0, 'ETH': 3200.0, 'SOL': 190.0}
    ```
    
    ### Get Historical Data for Strategy
    ```python
    data = fetcher.get_market_data(
        tickers=['BTC', 'ETH', 'SOL', 'BNB', 'XRP'],
        lookback_days=400,
    )
    # Returns:
    # {
    #   'prices': DataFrame (date x ticker),
    #   'volume': DataFrame (date x ticker),
    #   'market_cap': DataFrame (date x ticker),  # estimated
    # }
    ```
    """
    
    # Configuration
    quote_asset: str = "USDT"
    fallback_quotes: List[str] = field(default_factory=lambda: ["USDC", "BUSD", "BTC"])
    stablecoins: List[str] = field(default_factory=lambda: DEFAULT_STABLECOINS.copy())
    
    # Rate limiting
    request_delay_ms: int = DEFAULT_REQUEST_DELAY_MS
    max_retries: int = DEFAULT_MAX_RETRIES
    
    # Caching
    cache_ttl_seconds: int = 60
    _price_cache: Dict[str, Tuple[float, float]] = field(default_factory=dict, repr=False)
    _exchange_info_cache: Optional[Dict] = field(default=None, repr=False)
    _exchange_info_time: float = field(default=0.0, repr=False)
    
    # CCXT exchange instance
    _exchange: Any = field(default=None, repr=False)
    
    def __post_init__(self):
        """Initialize CCXT exchange if available."""
        if CCXT_AVAILABLE and self._exchange is None:
            self._exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
            })
    
    def describe(self) -> Dict[str, Any]:
        """
        Describe data fetcher capabilities for LLM introspection.
        
        Returns dict with:
        - purpose: What this fetcher does
        - capabilities: Available data types
        - methods: Key methods with signatures
        - example: Usage example
        """
        return {
            "purpose": "Fetch live and historical market data from Binance",
            "api_key_required": False,
            "capabilities": {
                "current_prices": "Real-time prices for any Binance pair",
                "historical_ohlcv": "Daily OHLCV data (up to 2 years)",
                "24h_stats": "24-hour volume, price change, high/low",
                "market_cap": "Estimated market cap (price * circulating supply)",
            },
            "methods": {
                "get_current_prices(tickers)": "Returns Dict[ticker, price]",
                "get_market_data(tickers, lookback_days)": "Returns {'prices', 'volume', 'market_cap'} DataFrames",
                "get_snapshot(tickers)": "Returns MarketDataSnapshot with current state",
                "get_valid_pairs(tickers)": "Returns Dict[ticker, binance_pair]",
            },
            "example": """
fetcher = BinanceDataFetcher()
prices = fetcher.get_current_prices(['BTC', 'ETH', 'SOL'])
data = fetcher.get_market_data(['BTC', 'ETH'], lookback_days=90)
            """,
            "ccxt_available": CCXT_AVAILABLE,
        }
    
    def _rate_limit(self):
        """Apply rate limiting delay."""
        time.sleep(self.request_delay_ms / 1000)
    
    def _get_exchange_info(self, force_refresh: bool = False) -> Dict:
        """Get Binance exchange info (cached)."""
        now = time.time()
        if (not force_refresh and 
            self._exchange_info_cache is not None and 
            now - self._exchange_info_time < 3600):  # 1 hour cache
            return self._exchange_info_cache
        
        try:
            resp = requests.get(BINANCE_EXCHANGE_INFO, timeout=10)
            resp.raise_for_status()
            self._exchange_info_cache = resp.json()
            self._exchange_info_time = now
            return self._exchange_info_cache
        except Exception as e:
            logger.error(f"Failed to fetch exchange info: {e}")
            return self._exchange_info_cache or {}
    
    def get_valid_pairs(self, tickers: List[str]) -> Dict[str, Optional[str]]:
        """
        Get valid Binance trading pairs for given tickers.
        
        Tries primary quote asset first, then fallbacks.
        
        Args:
            tickers: List of base asset symbols (e.g., ['BTC', 'ETH'])
            
        Returns:
            Dict mapping ticker to Binance pair (or None if not found)
            
        Example:
            >>> fetcher.get_valid_pairs(['BTC', 'ETH', 'INVALID'])
            {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'INVALID': None}
        """
        exchange_info = self._get_exchange_info()
        if not exchange_info:
            return {t: None for t in tickers}
        
        # Build set of active trading pairs
        active_pairs = {
            s['symbol'] for s in exchange_info.get('symbols', [])
            if s.get('status') == 'TRADING'
        }
        
        result = {}
        quote_order = [self.quote_asset] + self.fallback_quotes
        
        for ticker in tickers:
            ticker_upper = ticker.upper()
            found = None
            
            for quote in quote_order:
                pair = f"{ticker_upper}{quote}"
                if pair in active_pairs:
                    found = pair
                    break
            
            result[ticker] = found
        
        return result
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get current prices for given tickers.
        
        Uses cached values if available and fresh.
        
        Args:
            tickers: List of base asset symbols (e.g., ['BTC', 'ETH'])
            
        Returns:
            Dict mapping ticker to current price
            
        Example:
            >>> fetcher.get_current_prices(['BTC', 'ETH', 'SOL'])
            {'BTC': 95000.0, 'ETH': 3200.0, 'SOL': 190.0}
        """
        # Check cache first
        now = time.time()
        prices = {}
        tickers_to_fetch = []
        
        for ticker in tickers:
            if ticker in self._price_cache:
                cached_price, cached_time = self._price_cache[ticker]
                if now - cached_time < self.cache_ttl_seconds:
                    prices[ticker] = cached_price
                    continue
            tickers_to_fetch.append(ticker)
        
        if not tickers_to_fetch:
            return prices
        
        # Get valid pairs
        pairs = self.get_valid_pairs(tickers_to_fetch)
        
        # Fetch all prices at once
        try:
            resp = requests.get(BINANCE_TICKER_PRICE, timeout=10)
            resp.raise_for_status()
            all_prices = {item['symbol']: float(item['price']) for item in resp.json()}
        except Exception as e:
            logger.error(f"Failed to fetch prices: {e}")
            return prices
        
        # Map back to tickers
        for ticker in tickers_to_fetch:
            pair = pairs.get(ticker)
            if pair and pair in all_prices:
                price = all_prices[pair]
                prices[ticker] = price
                self._price_cache[ticker] = (price, now)
            else:
                logger.warning(f"No price found for {ticker}")
        
        return prices
    
    def get_snapshot(self, tickers: List[str]) -> MarketDataSnapshot:
        """
        Get current market snapshot for given tickers.
        
        Includes price, 24h volume, 24h change.
        
        Args:
            tickers: List of base asset symbols
            
        Returns:
            MarketDataSnapshot with current market state
            
        Example:
            >>> snapshot = fetcher.get_snapshot(['BTC', 'ETH'])
            >>> snapshot.data
               ticker     price    volume_24h  change_24h
            0     BTC  95000.00  35000000000        2.50
            1     ETH   3200.00  18000000000        1.80
        """
        pairs = self.get_valid_pairs(tickers)
        valid_pairs = {t: p for t, p in pairs.items() if p is not None}
        
        if not valid_pairs:
            return MarketDataSnapshot(
                timestamp=datetime.now(),
                data=pd.DataFrame(columns=['ticker', 'price', 'volume_24h', 'change_24h']),
                quote_asset=self.quote_asset,
            )
        
        # Fetch 24h ticker stats
        try:
            resp = requests.get(BINANCE_TICKER_24H, timeout=15)
            resp.raise_for_status()
            all_stats = {item['symbol']: item for item in resp.json()}
        except Exception as e:
            logger.error(f"Failed to fetch 24h stats: {e}")
            # Fallback to just prices
            prices = self.get_current_prices(tickers)
            data = pd.DataFrame([
                {'ticker': t, 'price': p, 'volume_24h': 0, 'change_24h': 0}
                for t, p in prices.items()
            ])
            return MarketDataSnapshot(
                timestamp=datetime.now(),
                data=data,
                quote_asset=self.quote_asset,
            )
        
        # Build snapshot data
        rows = []
        for ticker, pair in valid_pairs.items():
            if pair in all_stats:
                stats = all_stats[pair]
                rows.append({
                    'ticker': ticker,
                    'price': float(stats['lastPrice']),
                    'volume_24h': float(stats['quoteVolume']),  # Volume in quote asset
                    'change_24h': float(stats['priceChangePercent']),
                    'high_24h': float(stats['highPrice']),
                    'low_24h': float(stats['lowPrice']),
                })
        
        data = pd.DataFrame(rows)
        
        return MarketDataSnapshot(
            timestamp=datetime.now(),
            data=data,
            quote_asset=self.quote_asset,
        )
    
    def get_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = '1d',
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single ticker.
        
        Args:
            ticker: Base asset symbol (e.g., 'BTC')
            start_date: Start date as 'YYYY-MM-DD'
            end_date: End date as 'YYYY-MM-DD'
            interval: Candle interval ('1d', '1h', '4h', etc.)
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
            
        Example:
            >>> df = fetcher.get_ohlcv('BTC', '2024-01-01', '2024-03-01')
            >>> df.head()
                      date     open     high      low    close      volume
            0   2024-01-01  42000.0  42500.0  41500.0  42200.0  25000000.0
        """
        if not CCXT_AVAILABLE:
            logger.error("ccxt not installed - cannot fetch OHLCV data")
            return None
        
        # Get valid pair
        pairs = self.get_valid_pairs([ticker])
        pair = pairs.get(ticker)
        if not pair:
            logger.warning(f"No valid Binance pair for {ticker}")
            return None
        
        # Format pair for ccxt (BTC/USDT not BTCUSDT)
        ccxt_symbol = f"{ticker}/{self.quote_asset}"
        
        try:
            since = int(pd.Timestamp(start_date).timestamp() * 1000)
            end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
            
            all_ohlcv = []
            limit = 1000
            
            while since < end_ts:
                self._rate_limit()
                ohlcv = self._exchange.fetch_ohlcv(
                    ccxt_symbol, 
                    timeframe=interval, 
                    since=since, 
                    limit=limit
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                last_ts = ohlcv[-1][0]
                
                if last_ts == since:
                    break
                since = last_ts + 1
                
                if len(ohlcv) < limit:
                    break
            
            if not all_ohlcv:
                return None
            
            df = pd.DataFrame(
                all_ohlcv,
                columns=['date', 'open', 'high', 'low', 'close', 'volume']
            )
            df['date'] = pd.to_datetime(df['date'], unit='ms')
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            df['ticker'] = ticker
            
            return df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {ticker}: {e}")
            return None
    
    def get_market_data(
        self,
        tickers: List[str],
        lookback_days: int = 400,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get full market data for strategies (prices, volume, market_cap).
        
        This is the main method for feeding data to trading strategies.
        Returns DataFrames in the format expected by quantlab strategies.
        
        Args:
            tickers: List of base asset symbols
            lookback_days: Number of days of history to fetch
            end_date: End date (default: yesterday)
            
        Returns:
            Dict with:
            - 'prices': DataFrame (date index, ticker columns)
            - 'volume': DataFrame (date index, ticker columns)
            - 'market_cap': DataFrame (date index, ticker columns)
            
        Example:
            >>> data = fetcher.get_market_data(['BTC', 'ETH', 'SOL'], lookback_days=90)
            >>> data['prices'].tail()
                          BTC      ETH     SOL
            date                              
            2024-02-01  95000   3200.0   190.0
            2024-02-02  94500   3180.0   188.0
        """
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        start_date = (pd.to_datetime(end_date) - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching market data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Fetch OHLCV for each ticker
        prices_data = {}
        volume_data = {}
        
        for ticker in tickers:
            # Skip stablecoins
            if ticker.upper() in self.stablecoins:
                logger.debug(f"Skipping stablecoin: {ticker}")
                continue
            
            df = self.get_ohlcv(ticker, start_date, end_date)
            if df is not None and not df.empty:
                df = df.set_index('date')
                prices_data[ticker] = df['close']
                volume_data[ticker] = df['volume']
                logger.debug(f"Fetched {len(df)} days for {ticker}")
            else:
                logger.warning(f"No data for {ticker}")
        
        if not prices_data:
            logger.error("No data fetched for any ticker")
            return {
                'prices': pd.DataFrame(),
                'volume': pd.DataFrame(),
                'market_cap': pd.DataFrame(),
            }
        
        # Combine into DataFrames
        prices = pd.DataFrame(prices_data)
        volume = pd.DataFrame(volume_data)
        
        # Estimate market cap (price * volume as proxy)
        # Real market cap would come from CoinGecko/CoinMarketCap
        market_cap = self._estimate_market_cap(prices, volume)
        
        logger.info(f"Fetched data for {len(prices.columns)} tickers, {len(prices)} days")
        
        return {
            'prices': prices,
            'volume': volume,
            'market_cap': market_cap,
        }
    
    def _estimate_market_cap(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Estimate market cap from price and volume.
        
        This is a rough estimate - for accurate market cap, use CoinGecko/CMC APIs.
        Uses circulating supply estimates for major coins.
        """
        # Approximate circulating supplies (as of 2024)
        circulating_supply = {
            'BTC': 19.6e6,
            'ETH': 120e6,
            'SOL': 440e6,
            'BNB': 150e6,
            'XRP': 55e9,
            'DOGE': 143e9,
            'ADA': 35e9,
            'AVAX': 390e6,
            'LINK': 600e6,
            'DOT': 1.4e9,
            'MATIC': 10e9,
            'SHIB': 589e12,
            'LTC': 74e6,
            'TRX': 89e9,
            'ATOM': 390e6,
            'UNI': 600e6,
            'APT': 470e6,
            'NEAR': 1.1e9,
            'INJ': 93e6,
            'FIL': 530e6,
        }
        
        market_cap = pd.DataFrame(index=prices.index)
        
        for ticker in prices.columns:
            supply = circulating_supply.get(ticker.upper(), 1e9)  # Default 1B supply
            market_cap[ticker] = prices[ticker] * supply
        
        return market_cap
    
    def get_tradable_tickers(self, min_volume_usd: float = 1e6) -> List[str]:
        """
        Get list of tradable tickers with sufficient volume.
        
        Filters Binance pairs by 24h volume.
        
        Args:
            min_volume_usd: Minimum 24h volume in USD
            
        Returns:
            List of ticker symbols
        """
        try:
            resp = requests.get(BINANCE_TICKER_24H, timeout=15)
            resp.raise_for_status()
            all_stats = resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch 24h stats: {e}")
            return []
        
        tickers = []
        for item in all_stats:
            symbol = item['symbol']
            quote_vol = float(item.get('quoteVolume', 0))
            
            # Check if pair ends with our quote asset
            if not symbol.endswith(self.quote_asset):
                continue
            
            if quote_vol >= min_volume_usd:
                base = symbol[:-len(self.quote_asset)]
                if base not in self.stablecoins:
                    tickers.append(base)
        
        return sorted(tickers)
