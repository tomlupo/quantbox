"""
Simplified Data Fetcher for Crypto Trading Pipeline

This module provides simplified data fetching for crypto trading strategies:
1. Downloads current coin snapshots from CoinCodex
2. Fetches historical OHLCV data from Binance
3. Returns single DataFrame with all data
4. Fast Parquet-based caching with DuckDB query support
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import os
import logging
import pickle
from tqdm import tqdm
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import time
import shutil
import requests
from pathlib import Path
from .utils import get_logger

# Import data query modules - wrapped to handle missing psycopg2
try:
    import data_queries as dq
except ImportError:
    dq = None  # data_queries not available (needs psycopg2)

# Direct ccxt import for standalone operation
import ccxt

from .cache import FastParquetCache

# --- Consolidated Configuration ---
CONFIG = {
    'cache_dir': 'data/cache',
    'top_coins': 100,
    'lookback_days': 730,  # 2 years
    'stablecoins': ['USDT', 'USDC', 'BUSD', 'TUSD', 'DAI', 'MIM', 'USTC', 'FDUSD', 'USD1'],
    'cache_ttl_hours': 24,
    'min_price': 0.000001,
    'required_columns': ['date', 'open', 'high', 'low', 'close', 'volume'],
    # New CMC-specific settings
    'cmc_fallback_max_age_hours': 48,  # How old cached data can be for fallback
    'cmc_api_timeout_seconds': 30,     # API timeout
    'cmc_max_retries': 3              # Number of retry attempts
}

def basic_validate_ohlcv(df: pd.DataFrame, ticker: str) -> bool:
    """
    Basic validation for OHLCV data - essential checks only.
    
    Args:
        df: DataFrame with OHLCV data
        ticker: Ticker symbol for logging
        
    Returns:
        True if data is valid, False otherwise
    """
    if df.empty:
        logging.warning(f"Empty dataframe for {ticker}")
        return False
    
    # Check required columns
    missing_cols = [col for col in CONFIG['required_columns'] if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required columns for {ticker}: {missing_cols}")
        return False
    
    # Check for negative prices
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if (df[col] < CONFIG['min_price']).any():
            logging.warning(f"Found negative prices in {col} for {ticker}")
            return False
    
    # Basic price consistency
    if (df['high'] < df['low']).any():
        logging.warning(f"Found high < low for {ticker}")
        return False
    
    return True

class CryptoDataFetcher:
    """
    Simplified data fetcher for crypto trading strategies.

    Features:
    - Downloads current coin snapshots from CoinCodex
    - Fetches historical OHLCV data from Binance
    - Returns single DataFrame with all data
    - Fast Parquet-based caching with DuckDB query support
    """

    def __init__(self, config: Dict = None):
        self.config = CONFIG.copy()
        if config:
            self.config.update(config)

        self.logger = logging.getLogger(__name__)
        os.makedirs(self.config['cache_dir'], exist_ok=True)
        self.cache = FastParquetCache(self.config['cache_dir'])

    def fetch_cmc_rankings(self, limit: int = 100, max_cache_age_hours: int = 4, max_fallback_age_hours: int = 28) -> pd.DataFrame:
        cache_key = "cmc_rankings"
        
        # Always check cache first (for quantbox we may not have dq)
        fresh_data = self._get_fallback_cmc_rankings(cache_key, max_cache_age_hours)
        if fresh_data is not None and not fresh_data.empty:
            self.logger.info(f"Using fresh cached data from {fresh_data['fetch_timestamp'].iloc[0]}")
            return fresh_data

        # If data_queries is not available, try CoinGecko API
        if dq is None:
            self.logger.info("data_queries not available, trying CoinGecko API...")
            try:
                fresh_data = self._fetch_rankings_coingecko(limit)
                if fresh_data is not None and not fresh_data.empty:
                    fresh_data['fetch_timestamp'] = datetime.now()
                    self.cache.set(cache_key, fresh_data, partition_cols=None)
                    self.logger.info(f"Fetched {len(fresh_data)} rankings from CoinGecko")
                    return fresh_data
            except Exception as e:
                self.logger.warning(f"CoinGecko fetch failed: {e}")
            
            # Fall back to older cache
            fallback_data = self._get_fallback_cmc_rankings(cache_key, max_fallback_age_hours * 24)
            if fallback_data is not None and not fallback_data.empty:
                return fallback_data
            self.logger.error("No rankings data available")
            return pd.DataFrame()

        try:
            self.logger.info(f"Fetching fresh CMC rankings (top {limit}) from API...")
            fresh_data = dq.coinmarketcap.fetch_cmc_rankings(limit=limit)
            fresh_data['fetch_timestamp'] = datetime.now()
            self.cache.set(cache_key, fresh_data, partition_cols=None)
            self.logger.info(f"Successfully fetched and cached {len(fresh_data)} fresh rankings")
            return fresh_data
        except Exception as e:
            self.logger.error(f"API fetch failed: {e}")
            fallback_data = self._get_fallback_cmc_rankings(cache_key, max_fallback_age_hours)
            if fallback_data is not None and not fallback_data.empty:
                self.logger.warning(f"Using fallback data from {fallback_data['fetch_timestamp'].iloc[0]}")
                return fallback_data
            self.logger.error("No CMC rankings data available (API failed + no valid cache)")
            return pd.DataFrame()

    def _fetch_rankings_coingecko(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch coin rankings from CoinGecko API (free, no API key needed).
        
        Returns DataFrame with columns matching CMC format:
            rank, name, symbol, price, market_cap, volume_24h, circulating_supply
        """
        try:
            url = f"https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': 'false',
                'price_change_percentage': '1h,24h,7d'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame matching CMC format
            records = []
            for i, coin in enumerate(data):
                records.append({
                    'rank': i + 1,
                    'name': coin.get('name', ''),
                    'symbol': coin.get('symbol', '').upper(),
                    'price': coin.get('current_price', 0),
                    '1h%': coin.get('price_change_percentage_1h_in_currency', 0),
                    '24h%': coin.get('price_change_percentage_24h', 0),
                    '7d%': coin.get('price_change_percentage_7d_in_currency', 0),
                    'market_cap': coin.get('market_cap', 0),
                    'volume_24h': coin.get('total_volume', 0),
                    'circulating_supply': coin.get('circulating_supply', 0),
                })
            
            df = pd.DataFrame(records)
            self.logger.info(f"Fetched {len(df)} coins from CoinGecko")
            return df
            
        except Exception as e:
            self.logger.error(f"CoinGecko API error: {e}")
            return pd.DataFrame()

    def _get_fallback_cmc_rankings(self, cache_key: str, max_age_hours: int) -> pd.DataFrame:
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
            fallback_data = self.cache.get_latest(
                cache_key,
                order_col='fetch_timestamp',
                sql_where=f"fetch_timestamp >= '{cutoff_str}'",
                return_type='record'
            )
            if not fallback_data.empty:
                self.logger.info(f"Found fallback data: {len(fallback_data)} records from {fallback_data['fetch_timestamp'].iloc[0]}")
                return fallback_data
            else:
                self.logger.warning(f"No cached data found within {max_age_hours} hours")
                return None
        except Exception as e:
            self.logger.error(f"Error retrieving fallback data: {e}")
            return None

    def _format_ticker_for_binance(self, ticker: str) -> str:
        for stablecoin in self.config['stablecoins']:
            if ticker.endswith(stablecoin):
                return ticker
        return f"{ticker}USDT"

    def get_valid_binance_pairs(self, base_assets: List[str], quote_assets: List[str] = None) -> Dict[str, str]:
        if quote_assets is None:
            quote_assets = ['USDT', 'BUSD', 'BTC']
        try:
            resp = requests.get("https://api.binance.com/api/v3/exchangeInfo")
            all_symbols = {s['symbol'] for s in resp.json()['symbols'] if s['status'] == 'TRADING'}
        except Exception as e:
            self.logger.error(f"Failed to load Binance exchange info: {e}")
            return {base: None for base in base_assets}
        result = {}
        for base in base_assets:
            for quote in quote_assets:
                pair = f"{base.upper()}{quote}"
                if pair in all_symbols:
                    result[base] = pair
                    break
            else:
                result[base] = None
        return result

    def _fetch_ohlcv_ccxt(self, symbol: str, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data directly from Binance using ccxt.
        
        Args:
            symbol: Binance symbol (e.g., 'BTCUSDT')
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        try:
            # Initialize ccxt Binance
            exchange = ccxt.binance({'enableRateLimit': True})
            
            # Convert symbol format (BTCUSDT -> BTC/USDT)
            if symbol.endswith('USDT'):
                ccxt_symbol = symbol[:-4] + '/USDT'
            elif symbol.endswith('USDC'):
                ccxt_symbol = symbol[:-4] + '/USDC'
            elif symbol.endswith('BUSD'):
                ccxt_symbol = symbol[:-4] + '/BUSD'
            else:
                ccxt_symbol = symbol[:3] + '/' + symbol[3:]
            
            # Calculate since timestamp
            since = exchange.parse8601(pd.Timestamp(start_date).isoformat())
            
            # Fetch data in chunks (ccxt has limits)
            all_data = []
            current_since = since
            end_ts = exchange.parse8601(pd.Timestamp(end_date).isoformat())
            
            while current_since < end_ts:
                ohlcv = exchange.fetch_ohlcv(ccxt_symbol, '1d', since=current_since, limit=500)
                if not ohlcv:
                    break
                all_data.extend(ohlcv)
                # Move to next batch
                current_since = ohlcv[-1][0] + 86400000  # +1 day in ms
                if len(ohlcv) < 500:
                    break
            
            if not all_data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.normalize()
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df = df.drop_duplicates(subset=['date'], keep='last')
            
            self.logger.debug(f"Fetched {len(df)} days for {symbol} via ccxt")
            return df
            
        except Exception as e:
            self.logger.error(f"ccxt fetch failed for {symbol}: {e}")
            return None

    def fetch_ohlcv(self, tickers: List[str], lookback_days: int = None,
                   start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        if end_date is None:
            end_date = pd.Timestamp.now().normalize() - timedelta(days=1)
        else:
            end_date = pd.to_datetime(end_date).normalize()

        if start_date is None:
            if lookback_days is None:
                lookback_days = self.config['lookback_days']
            start_date = end_date - timedelta(days=lookback_days)
        else:
            start_date = pd.to_datetime(start_date).normalize()

        ohlcv_data = {}
        cache_key = "binance_ohlcv"
        self.logger.info(f"Fetching OHLCV data for {len(tickers)} tickers from {start_date} to {end_date}")

        valid_pairs = self.get_valid_binance_pairs(tickers)

        for ticker in tqdm(tickers, desc="Fetching OHLCV data"):
            binance_ticker = valid_pairs.get(ticker)
            if not binance_ticker:
                self.logger.warning(f"Skipping {ticker}: no valid trading pair found on Binance")
                continue
            try:
                last_date = self.cache.get_latest(cache_key, order_col='date', return_type='date', ticker=ticker)
                fetch_start = start_date if last_date is None or pd.isna(last_date) else last_date + timedelta(days=1)
                if fetch_start == end_date: fetch_end = end_date + timedelta(days=1)
                else: fetch_end = end_date
                
                # Fetch fresh data if needed
                fresh_df = None
                if fetch_start < fetch_end:
                    df = self._fetch_ohlcv_ccxt(binance_ticker, fetch_start, fetch_end)
                    if df is not None and not df.empty and basic_validate_ohlcv(df, ticker):
                        df = df.query(f"date >= '{fetch_start}' and date <= '{end_date}'")
                        df['ticker'] = ticker
                        df['year'] = pd.to_datetime(df['date']).dt.year
                        fresh_df = df.copy()
                        # Try to cache but ignore permission errors (read-only cache)
                        try:
                            self.cache.set(cache_key, df, partition_cols=['ticker', 'year'], primary_keys=['ticker', 'date'])
                        except PermissionError:
                            self.logger.debug(f"Cache write skipped (read-only): {ticker}")
                        except Exception as cache_err:
                            if 'Permission denied' in str(cache_err):
                                self.logger.debug(f"Cache write skipped (read-only): {ticker}")
                            else:
                                raise

                # Read from cache
                sql_where = f"ticker='{ticker}' AND date >= '{start_date}' AND date <= '{end_date}'"
                full_df = self.cache.get(cache_key, sql_where=sql_where)
                
                # Combine cached + fresh data
                if fresh_df is not None:
                    if full_df is not None and not full_df.empty:
                        full_df = pd.concat([full_df, fresh_df]).drop_duplicates(subset=['date'], keep='last')
                    else:
                        full_df = fresh_df
                
                if full_df is not None and not full_df.empty:
                    ohlcv_data[ticker] = full_df.sort_values('date').reset_index(drop=True)
                else:
                    self.logger.warning(f"No valid data for {ticker} (tried {binance_ticker})")
                    continue
                    
                if pd.Timestamp(end_date) not in pd.to_datetime(full_df['date']).values:
                    self.logger.warning(f"No data for {ticker} on {end_date}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker} (tried {binance_ticker}): {e}")
        self.logger.info(f"Successfully fetched OHLCV data for {len(ohlcv_data)} tickers")
        return ohlcv_data

    def get_ohlcv(self, tickers: List[str], start_date: str = None, end_date: str = None,
                  lookback_days: int = None) -> Dict[str, pd.DataFrame]:
        """
        Retrieve OHLCV data from cache without fetching from external sources.
        
        Args:
            tickers: List of ticker symbols to retrieve data for
            start_date: Start date for data range (optional)
            end_date: End date for data range (optional)
            lookback_days: Number of days to look back from end_date (optional)
            
        Returns:
            Dictionary mapping ticker symbols to their OHLCV DataFrames
        """
        if end_date is None:
            end_date = pd.Timestamp.now().normalize() - timedelta(days=1)
        else:
            end_date = pd.to_datetime(end_date).normalize()

        if start_date is None:
            if lookback_days is None:
                lookback_days = self.config['lookback_days']
            start_date = end_date - timedelta(days=lookback_days)
        else:
            start_date = pd.to_datetime(start_date).normalize()

        ohlcv_data = {}
        cache_key = "binance_ohlcv"
        self.logger.info(f"Retrieving cached OHLCV data for {len(tickers)} tickers from {start_date} to {end_date}")

        for ticker in tickers:
            try:
                sql_where = f"ticker='{ticker}' AND date >= '{start_date}' AND date <= '{end_date}'"
                cached_df = self.cache.get(cache_key, sql_where=sql_where)
                
                if cached_df is not None and not cached_df.empty:
                    # Sort by date and reset index for consistency
                    cached_df = cached_df.sort_values('date').reset_index(drop=True)
                    ohlcv_data[ticker] = cached_df
                    self.logger.debug(f"Retrieved {len(cached_df)} records for {ticker}")
                else:
                    self.logger.warning(f"No cached data found for {ticker} in date range {start_date} to {end_date}")
                    
            except Exception as e:
                self.logger.error(f"Error retrieving cached data for {ticker}: {e}")

        self.logger.info(f"Successfully retrieved cached OHLCV data for {len(ohlcv_data)} tickers")
        return ohlcv_data

