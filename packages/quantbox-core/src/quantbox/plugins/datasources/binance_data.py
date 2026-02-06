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
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import time
import httpx

from quantbox.plugins.datasources._utils import (
    validate_ohlcv,
    retry_transient,
    OHLCVCache,
    MarketCapProvider,
)

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
    cache_dir: Optional[str] = None  # Path to DuckDB Parquet cache directory
    cache_fresh_ttl_hours: float = 4.0
    _price_cache: Dict[str, Tuple[float, float]] = field(default_factory=dict, repr=False)
    _exchange_info_cache: Optional[Dict] = field(default=None, repr=False)
    _exchange_info_time: float = field(default=0.0, repr=False)
    _ohlcv_cache: Optional[OHLCVCache] = field(default=None, repr=False)

    # CoinMarketCap integration
    cmc_api_key: Optional[str] = None  # Falls back to CMC_API_KEY env var
    _cmc_provider: Optional[MarketCapProvider] = field(default=None, repr=False)

    # CCXT exchange instance
    _exchange: Any = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize CCXT exchange, OHLCV cache, and CMC provider."""
        if CCXT_AVAILABLE and self._exchange is None:
            self._exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
            })

        # Initialize DuckDB Parquet cache if cache_dir is provided
        if self.cache_dir and self._ohlcv_cache is None:
            self._ohlcv_cache = OHLCVCache(
                cache_dir=self.cache_dir,
                fresh_ttl_hours=self.cache_fresh_ttl_hours,
            )

        # Initialize CMC provider
        if self._cmc_provider is None:
            self._cmc_provider = MarketCapProvider(
                api_key=self.cmc_api_key,
                cache_dir=self.cache_dir,
                fresh_ttl_hours=self.cache_fresh_ttl_hours,
            )
    
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

    @retry_transient
    def _fetch_all_prices(self) -> Dict[str, float]:
        """Fetch all ticker prices from Binance with retry on transient errors."""
        resp = httpx.get(BINANCE_TICKER_PRICE, timeout=10)
        resp.raise_for_status()
        return {item['symbol']: float(item['price']) for item in resp.json()}
    
    def _get_exchange_info(self, force_refresh: bool = False) -> Dict:
        """Get Binance exchange info (cached)."""
        now = time.time()
        if (not force_refresh and 
            self._exchange_info_cache is not None and 
            now - self._exchange_info_time < 3600):  # 1 hour cache
            return self._exchange_info_cache
        
        try:
            resp = httpx.get(BINANCE_EXCHANGE_INFO, timeout=10)
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
        
        # Fetch all prices at once (with retry)
        try:
            all_prices = self._fetch_all_prices()
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
            resp = httpx.get(BINANCE_TICKER_24H, timeout=15)
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
        Fetch OHLCV data for a single ticker with caching and validation.

        Uses DuckDB Parquet cache (if ``cache_dir`` is set) for incremental
        fetching: only requests candles not already cached.  Validates the
        resulting DataFrame for data quality.

        Args:
            ticker: Base asset symbol (e.g., 'BTC')
            start_date: Start date as 'YYYY-MM-DD'
            end_date: End date as 'YYYY-MM-DD'
            interval: Candle interval ('1d', '1h', '4h', etc.)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, ticker
        """
        if not CCXT_AVAILABLE:
            logger.error("ccxt not installed - cannot fetch OHLCV data")
            return None

        # --- Check cache for a complete hit ---
        if self._ohlcv_cache is not None and interval == "1d":
            cached = self._ohlcv_cache.get_cached(ticker, start_date, end_date)
            if cached is not None and not cached.empty:
                last_cached = cached["date"].max()
                target_end = pd.Timestamp(end_date)
                if last_cached >= target_end - timedelta(hours=self.cache_fresh_ttl_hours * 24):
                    # Cache is fresh enough
                    cached["ticker"] = ticker
                    try:
                        cached = validate_ohlcv(cached, ticker)
                    except ValueError as exc:
                        logger.warning("Cached OHLCV invalid for %s: %s", ticker, exc)
                    else:
                        cached["ticker"] = ticker
                        return cached.reset_index(drop=True)

        # Get valid pair
        pairs = self.get_valid_pairs([ticker])
        pair = pairs.get(ticker)
        if not pair:
            logger.warning(f"No valid Binance pair for {ticker}")
            return None

        # Determine fetch start: use cache to do incremental fetching
        fetch_start = start_date
        if self._ohlcv_cache is not None and interval == "1d":
            last_cached_date = self._ohlcv_cache.get_last_date(ticker)
            if last_cached_date is not None:
                # Fetch from the day after last cached date
                next_day = (last_cached_date + timedelta(days=1)).strftime("%Y-%m-%d")
                if next_day > start_date:
                    fetch_start = next_day

        # Format pair for ccxt (BTC/USDT not BTCUSDT)
        ccxt_symbol = f"{ticker}/{self.quote_asset}"

        try:
            df = self._fetch_ohlcv_with_retry(ccxt_symbol, fetch_start, end_date, interval)

            if df is not None and not df.empty:
                # Store new data in cache
                if self._ohlcv_cache is not None and interval == "1d":
                    self._ohlcv_cache.store(ticker, df)

            # Merge with cached data for the full range
            if self._ohlcv_cache is not None and interval == "1d":
                full = self._ohlcv_cache.get_cached(ticker, start_date, end_date)
                if full is not None and not full.empty:
                    df = full

            if df is None or df.empty:
                return None

            # Validate
            try:
                df = validate_ohlcv(df, ticker)
            except ValueError as exc:
                logger.error("OHLCV validation failed for %s: %s", ticker, exc)
                return None

            df["ticker"] = ticker
            return df.reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {ticker}: {e}")
            # Try to serve from cache even on failure
            if self._ohlcv_cache is not None and interval == "1d":
                cached = self._ohlcv_cache.get_cached(ticker, start_date, end_date)
                if cached is not None and not cached.empty:
                    logger.info("Serving stale cache for %s after fetch failure", ticker)
                    cached["ticker"] = ticker
                    return cached.reset_index(drop=True)
            return None

    @retry_transient
    def _fetch_ohlcv_with_retry(
        self,
        ccxt_symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV from the exchange with automatic retry on transient errors."""
        since = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

        if since >= end_ts:
            return None

        all_ohlcv = []
        limit = 1000

        while since < end_ts:
            self._rate_limit()
            ohlcv = self._exchange.fetch_ohlcv(
                ccxt_symbol,
                timeframe=interval,
                since=since,
                limit=limit,
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
            columns=['date', 'open', 'high', 'low', 'close', 'volume'],
        )
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        return df.reset_index(drop=True)
    
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
        """Estimate market cap using CoinMarketCap data with hardcoded fallback.

        If ``CMC_API_KEY`` is set (or passed via ``cmc_api_key``), fetches live
        rankings from CoinMarketCap and caches them as Parquet.  Falls back to
        hardcoded circulating-supply estimates for coins not covered by CMC.
        """
        return self._cmc_provider.estimate_market_cap(prices, volume)
    
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
            resp = httpx.get(BINANCE_TICKER_24H, timeout=15)
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
