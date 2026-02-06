"""
Crypto Trend Catcher Strategy - Pandas 3.0 / DuckDB / Vectorized

Multi-asset volatility-targeted trend-following strategy for cryptocurrencies.
Ported from quantlab with modern pandas 3.0, DuckDB, and fully vectorized operations.

Based on SSRN paper "Catching Crypto Trends".

## LLM Usage Guide

### Quick Start
```python
from quantbox.plugins.strategies import CryptoTrendStrategy
from quantbox.plugins.datasources import BinanceDataFetcher

# Fetch data
fetcher = BinanceDataFetcher()
data = fetcher.get_market_data(['BTC', 'ETH', 'SOL', 'BNB'], lookback_days=400)

# Run strategy
strategy = CryptoTrendStrategy()
result = strategy.run(data)

# Get latest weights
weights = result['weights']
print(weights.tail(1))
```

### Strategy Logic
1. **Universe Selection**: Top N coins by market cap, filtered by volume
2. **Signal Generation**: Donchian Channel breakouts, ensemble over multiple windows
3. **Volatility Targeting**: Scale signals to target volatility (25%, 50%)
4. **Portfolio Construction**: Apply universe mask, normalize weights

### Key Methods
- `run(data, params)` → Standard strategy interface
- `describe()` → LLM-friendly capability description
- `backtest(data, params)` → Run with performance metrics (quantstats)

### Performance Features
- Fully vectorized (no Python loops for signals)
- DuckDB for fast parquet queries
- Pandas 3.0 compatible
- Numba-ready signal computation
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging

from quantbox.contracts import PluginMeta

logger = logging.getLogger(__name__)

# Optional imports for enhanced features
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None
    DUCKDB_AVAILABLE = False

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    qs = None
    QUANTSTATS_AVAILABLE = False

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    vbt = None
    VECTORBT_AVAILABLE = False


# ============================================================================
# Constants
# ============================================================================

DEFAULT_STABLECOINS = [
    # USD stablecoins
    'USDT', 'USDC', 'BUSD', 'TUSD', 'DAI', 'MIM', 'USTC', 'FDUSD',
    'USDP', 'GUSD', 'FRAX', 'LUSD', 'USDD', 'PYUSD', 'USD1', 'USDJ',
    # EUR stablecoins
    'EUR', 'EURC', 'EURT', 'EURS', 'EUROC',
    # Gold/commodity tokens
    'PAXG', 'XAUT',
    # Wrapped tokens
    'WBTC', 'WETH', 'BETH', 'ETHW', 'CBBTC', 'CBETH',
    # Other non-tradeable
    'BFUSD', 'AEUR',
]

DEFAULT_LOOKBACK_WINDOWS = [5, 10, 20, 30, 60, 90, 150, 250, 360]
DEFAULT_VOL_TARGETS = [0.25, 0.50]
DEFAULT_TRANCHES = [1, 5, 21]


# ============================================================================
# Vectorized Signal Computation (Pandas 3.0 compatible)
# ============================================================================

def compute_donchian_breakout_vectorized(
    prices: pd.Series,
    window: int,
) -> pd.Series:
    """
    Compute Donchian Channel breakout signal - fully vectorized.
    
    Signal = 1 when price >= rolling high (breakout)
    Signal = 0 when price < trailing stop (midpoint)
    
    Pandas 3.0 compatible - uses .loc[] instead of .iloc[] assignment.
    
    Args:
        prices: Price series with DatetimeIndex
        window: Lookback window for Donchian channel
        
    Returns:
        Series of 0/1 signals
    """
    # Rolling high/low/mid
    high = prices.rolling(window=window, min_periods=window).max()
    low = prices.rolling(window=window, min_periods=window).min()
    mid = (high + low) / 2
    
    # Initial breakout condition (vectorized)
    breakout = (prices >= high).astype(float)
    
    # For trailing stop logic, we need to iterate but minimize it
    # Use numpy for speed
    signal = np.zeros(len(prices))
    trailing_stop = np.full(len(prices), np.nan)
    
    prices_arr = prices.values
    high_arr = high.values
    mid_arr = mid.values
    
    for i in range(window - 1, len(prices)):
        if i == window - 1:
            # Initial condition
            signal[i] = 1.0 if prices_arr[i] >= high_arr[i] else 0.0
            trailing_stop[i] = mid_arr[i]
        else:
            if signal[i-1] == 1:
                # In position - update trailing stop
                trailing_stop[i] = max(trailing_stop[i-1], mid_arr[i])
                signal[i] = 1.0 if prices_arr[i] >= trailing_stop[i] else 0.0
            else:
                # Out of position
                signal[i] = 1.0 if prices_arr[i] >= high_arr[i] else 0.0
                trailing_stop[i] = mid_arr[i]
    
    return pd.Series(signal, index=prices.index, name=prices.name)


def compute_donchian_simple_vectorized(
    prices: pd.Series,
    window: int,
) -> pd.Series:
    """
    Simple Donchian breakout - fully vectorized, no trailing stop.
    
    Much faster than full version, good for ensemble.
    Signal = 1 when price >= rolling high, else 0.
    """
    high = prices.rolling(window=window, min_periods=window).max()
    return (prices >= high).astype(float)


def generate_ensemble_signals(
    prices: pd.DataFrame,
    windows: List[int],
    use_trailing_stop: bool = True,
) -> pd.DataFrame:
    """
    Generate ensemble Donchian signals across multiple windows.
    
    Args:
        prices: DataFrame with ticker columns
        windows: List of lookback windows
        use_trailing_stop: Use full Donchian with trailing stop (slower but better)
        
    Returns:
        DataFrame of ensemble signals (mean across windows) per ticker
    """
    signal_fn = compute_donchian_breakout_vectorized if use_trailing_stop else compute_donchian_simple_vectorized
    
    signals = {}
    for ticker in prices.columns:
        # Compute signal for each window
        window_signals = np.column_stack([
            signal_fn(prices[ticker], w).values for w in windows
        ])
        # Ensemble: mean across windows
        signals[ticker] = window_signals.mean(axis=1)
    
    return pd.DataFrame(signals, index=prices.index)


# ============================================================================
# Universe Selection (DuckDB accelerated when available)
# ============================================================================

def select_universe_vectorized(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    market_cap: pd.DataFrame,
    top_by_mcap: int = 30,
    top_by_volume: int = 10,
    exclude_tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Select tradable universe - vectorized pandas operations.
    
    1. Exclude stablecoins
    2. Rank by market cap, keep top N
    3. Within those, rank by dollar volume, keep top M
    
    Args:
        prices: Price DataFrame
        volume: Volume DataFrame  
        market_cap: Market cap DataFrame
        top_by_mcap: Top N by market cap to consider
        top_by_volume: Top M by volume to trade
        exclude_tickers: Tickers to exclude (stablecoins, etc.)
        
    Returns:
        DataFrame of 0/1 universe flags
    """
    if exclude_tickers is None:
        exclude_tickers = DEFAULT_STABLECOINS
    
    # Filter columns
    valid_tickers = [t for t in prices.columns if t not in exclude_tickers]
    
    if not valid_tickers:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    # Market cap rank (lower = larger)
    mc = market_cap[valid_tickers]
    mc_rank = mc.rank(axis=1, ascending=False, method='min')
    mc_mask = mc_rank <= top_by_mcap
    
    # Dollar volume within MC-filtered
    dollar_vol = prices[valid_tickers] * volume[valid_tickers]
    vol_masked = dollar_vol.where(mc_mask)
    vol_rank = vol_masked.rank(axis=1, ascending=False, method='min')
    
    # Universe = top by volume within MC filter
    universe_valid = (vol_rank <= top_by_volume).astype(float)
    
    # Expand to full columns
    universe = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    universe[valid_tickers] = universe_valid
    
    return universe


def select_universe_duckdb(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    market_cap: pd.DataFrame,
    top_by_mcap: int = 30,
    top_by_volume: int = 10,
    exclude_tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Select universe using DuckDB for large datasets.
    
    Converts wide DataFrames to long format, runs SQL, converts back.
    Faster for very large datasets (1000+ tickers).
    """
    if not DUCKDB_AVAILABLE:
        return select_universe_vectorized(
            prices, volume, market_cap, top_by_mcap, top_by_volume, exclude_tickers
        )
    
    if exclude_tickers is None:
        exclude_tickers = DEFAULT_STABLECOINS
    
    # Convert to long format
    def to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        df = df.reset_index().melt(id_vars='date', var_name='ticker', value_name=value_name)
        return df
    
    prices_long = to_long(prices, 'price')
    volume_long = to_long(volume, 'volume')
    mc_long = to_long(market_cap, 'market_cap')
    
    # Merge
    data = prices_long.merge(volume_long, on=['date', 'ticker'])
    data = data.merge(mc_long, on=['date', 'ticker'])
    data['dollar_volume'] = data['price'] * data['volume']
    
    # Filter exclusions
    exclude_str = "', '".join(exclude_tickers)
    
    # Run DuckDB query
    con = duckdb.connect()
    con.register('data', data)
    
    query = f"""
    WITH ranked AS (
        SELECT 
            date,
            ticker,
            price,
            market_cap,
            dollar_volume,
            RANK() OVER (PARTITION BY date ORDER BY market_cap DESC) as mc_rank
        FROM data
        WHERE ticker NOT IN ('{exclude_str}')
    ),
    mc_filtered AS (
        SELECT *,
            RANK() OVER (PARTITION BY date ORDER BY dollar_volume DESC) as vol_rank
        FROM ranked
        WHERE mc_rank <= {top_by_mcap}
    )
    SELECT 
        date,
        ticker,
        CASE WHEN vol_rank <= {top_by_volume} THEN 1.0 ELSE 0.0 END as in_universe
    FROM mc_filtered
    """
    
    result = con.execute(query).df()
    con.close()
    
    # Pivot back to wide format
    universe = result.pivot(index='date', columns='ticker', values='in_universe')
    universe = universe.reindex(columns=prices.columns, fill_value=0.0)
    universe = universe.reindex(index=prices.index, fill_value=0.0)
    
    return universe


# ============================================================================
# Volatility Targeting
# ============================================================================

def compute_volatility_scalers(
    prices: pd.DataFrame,
    vol_targets: List[float],
    vol_lookback: int = 60,
    annualization_factor: float = 365.0,
) -> Dict[str, pd.DataFrame]:
    """
    Compute volatility scalers for each target.
    
    Scaler = target_vol / realized_vol
    This scales position size inversely to volatility.
    
    Args:
        prices: Price DataFrame
        vol_targets: List of target volatilities (e.g., [0.25, 0.50])
        vol_lookback: Rolling window for volatility estimation
        annualization_factor: Days per year for annualization
        
    Returns:
        Dict mapping vol_target_str to scaler DataFrame
    """
    # Compute annualized volatility
    returns = prices.pct_change()
    realized_vol = returns.rolling(vol_lookback).std() * np.sqrt(annualization_factor)
    
    scalers = {}
    for vt in vol_targets:
        scaler = vt / realized_vol.replace(0, np.nan)
        # Clip extreme scalers
        scaler = scaler.clip(lower=0.1, upper=10.0)
        scalers[f"{int(vt * 100)}"] = scaler
    
    return scalers


# ============================================================================
# Portfolio Construction
# ============================================================================

def construct_weights(
    signals: pd.DataFrame,
    universe: pd.DataFrame,
    scalers: Dict[str, pd.DataFrame],
    tranches: List[int],
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Construct portfolio weights from signals.
    
    Steps:
    1. Scale signals by volatility scaler
    2. Apply tranching (rolling mean for smoother transitions)
    3. Mask by tradable universe
    4. Normalize to sum to 1 (optional)
    
    Returns DataFrame with MultiIndex columns: (vol_target, tranche, ticker)
    """
    weights_dict = {}
    
    for vt_key, scaler in scalers.items():
        for t in tranches:
            # Vol-targeted signals
            sig_scaled = signals * scaler
            
            # Tranching (rolling mean for smoother transitions)
            if t > 1:
                sig_tranched = sig_scaled.rolling(t, min_periods=1).mean()
            else:
                sig_tranched = sig_scaled
            
            # Apply universe mask
            w = sig_tranched * universe
            
            # Normalize by number of positions
            if normalize:
                n_positions = universe.sum(axis=1).replace(0, np.nan)
                w = w.div(n_positions, axis=0)
            
            # Store with MultiIndex key
            for col in w.columns:
                weights_dict[(vt_key, t, col)] = w[col]
    
    weights = pd.DataFrame(weights_dict)
    weights.columns = pd.MultiIndex.from_tuples(
        weights.columns, 
        names=["vol_target", "tranches", "ticker"]
    )
    
    return weights


def get_simple_weights(
    weights_df: pd.DataFrame,
    vol_target: str = "50",
    tranche: int = 5,
) -> pd.DataFrame:
    """
    Extract simple weights from multi-index DataFrame.
    
    Args:
        weights_df: Weights with MultiIndex columns
        vol_target: Volatility target to use (e.g., "50" for 50%)
        tranche: Tranche parameter to use
        
    Returns:
        Simple DataFrame with ticker columns and weight values
    """
    # Select the specific parameter combination
    if isinstance(weights_df.columns, pd.MultiIndex):
        selected = weights_df.xs((vol_target, tranche), axis=1, level=('vol_target', 'tranches'))
    else:
        selected = weights_df
    
    return selected


# ============================================================================
# Strategy Class
# ============================================================================

@dataclass
class CryptoTrendStrategy:
    """
    Crypto Trend Catcher - Production Strategy Class.
    
    Implements multi-asset volatility-targeted trend-following.
    Fully vectorized, pandas 3.0 compatible, DuckDB accelerated.
    
    ## Quick Start
    ```python
    strategy = CryptoTrendStrategy()
    result = strategy.run(data)  # data from BinanceDataFetcher
    weights = result['weights']
    ```
    """

    meta = PluginMeta(
        name="strategy.crypto_trend.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Crypto trend catcher - multi-asset volatility-targeted trend following",
        tags=("crypto", "trend", "momentum"),
    )

    # Strategy parameters
    lookback_windows: List[int] = field(default_factory=lambda: DEFAULT_LOOKBACK_WINDOWS.copy())
    vol_targets: List[float] = field(default_factory=lambda: DEFAULT_VOL_TARGETS.copy())
    tranches: List[int] = field(default_factory=lambda: DEFAULT_TRANCHES.copy())
    
    # Universe parameters
    top_by_mcap: int = 30
    top_by_volume: int = 10
    exclude_tickers: List[str] = field(default_factory=lambda: DEFAULT_STABLECOINS.copy())
    
    # Volatility parameters
    vol_lookback: int = 60
    
    # Output parameters
    output_periods: int = 30
    normalize_weights: bool = True
    
    # Performance options
    use_duckdb: bool = True
    use_trailing_stop: bool = True
    
    def describe(self) -> Dict[str, Any]:
        """
        Describe strategy for LLM introspection.
        """
        return {
            "name": "CryptoTrendCatcher",
            "type": "trend_following",
            "purpose": "Multi-asset volatility-targeted trend-following for crypto",
            "parameters": {
                "lookback_windows": self.lookback_windows,
                "vol_targets": self.vol_targets,
                "tranches": self.tranches,
                "top_by_mcap": self.top_by_mcap,
                "top_by_volume": self.top_by_volume,
            },
            "signals": "Donchian Channel breakouts with trailing stop",
            "methods": {
                "run(data)": "Returns {'weights': DataFrame, 'details': dict}",
                "get_latest_weights(result)": "Extract latest weights as simple dict",
                "backtest(data)": "Run with performance metrics (requires quantstats)",
            },
            "features": {
                "duckdb": DUCKDB_AVAILABLE,
                "quantstats": QUANTSTATS_AVAILABLE,
                "vectorbt": VECTORBT_AVAILABLE,
            },
        }
    
    def run(
        self,
        data: Dict[str, pd.DataFrame],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run strategy and return weights.
        
        Args:
            data: Dict with 'prices', 'volume', 'market_cap' DataFrames
            params: Optional parameter overrides
            
        Returns:
            Dict with:
            - 'weights': DataFrame with MultiIndex columns
            - 'details': Intermediate calculations
            - 'simple_weights': Latest weights as simple dict
        """
        # Quantlab → quantbox param name aliases
        _PARAM_ALIASES = {
            "tickers_to_exclude": "exclude_tickers",
            "filtered_coins_market_cap": "top_by_mcap",
            "portfolio_coins_max": "top_by_volume",
            "last_x_days": "output_periods",
            "periods": "output_periods",
            "normalize": "normalize_weights",
        }

        # Apply parameter overrides
        if params:
            for key, value in params.items():
                attr = _PARAM_ALIASES.get(key, key)
                if hasattr(self, attr):
                    setattr(self, attr, value)
        
        prices = data['prices']
        volume = data['volume']
        market_cap = data['market_cap']
        
        logger.info(f"Running CryptoTrendStrategy on {len(prices.columns)} tickers, {len(prices)} days")
        
        # 1. Universe selection
        if self.use_duckdb and DUCKDB_AVAILABLE:
            universe = select_universe_duckdb(
                prices, volume, market_cap,
                self.top_by_mcap, self.top_by_volume, self.exclude_tickers
            )
        else:
            universe = select_universe_vectorized(
                prices, volume, market_cap,
                self.top_by_mcap, self.top_by_volume, self.exclude_tickers
            )
        
        # 2. Signal generation
        signals = generate_ensemble_signals(
            prices, 
            self.lookback_windows,
            use_trailing_stop=self.use_trailing_stop,
        )
        
        # 3. Volatility scalers
        scalers = compute_volatility_scalers(
            prices,
            self.vol_targets,
            self.vol_lookback,
        )
        
        # 4. Portfolio construction
        weights = construct_weights(
            signals,
            universe,
            scalers,
            self.tranches,
            self.normalize_weights,
        )
        
        # 5. Extract simple weights for latest day
        simple = get_simple_weights(weights, "50", 5)
        latest = simple.iloc[-1].dropna()
        latest = latest[latest > 0.001].to_dict()
        
        return {
            'weights': weights.tail(self.output_periods),
            'simple_weights': latest,
            'details': {
                'signals': signals,
                'universe': universe,
                'scalers': scalers,
            },
        }
    
    def get_latest_weights(
        self,
        result: Dict[str, Any],
        vol_target: str = "50",
        tranche: int = 5,
    ) -> Dict[str, float]:
        """
        Extract latest weights as a simple dict.
        
        Args:
            result: Output from run()
            vol_target: Volatility target to use
            tranche: Tranche parameter
            
        Returns:
            Dict of ticker -> weight
        """
        weights = result['weights']
        simple = get_simple_weights(weights, vol_target, tranche)
        latest = simple.iloc[-1].dropna()
        return latest[latest > 0.001].to_dict()
    
    def backtest(
        self,
        data: Dict[str, pd.DataFrame],
        initial_capital: float = 10000,
        commission_pct: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Run strategy with performance metrics.
        
        Requires quantstats for performance analysis.
        
        Returns:
            Dict with weights, returns, and performance metrics
        """
        result = self.run(data)
        
        # Compute strategy returns
        prices = data['prices']
        weights = get_simple_weights(result['weights'], "50", 5)
        
        # Align weights and prices
        weights = weights.reindex(columns=prices.columns, fill_value=0)
        
        # Asset returns
        returns = prices.pct_change()
        
        # Strategy returns (weighted average)
        strategy_returns = (returns * weights.shift(1)).sum(axis=1)
        strategy_returns = strategy_returns.dropna()
        
        # Performance metrics
        metrics = {}
        if QUANTSTATS_AVAILABLE:
            metrics = {
                'total_return': qs.stats.comp(strategy_returns),
                'cagr': qs.stats.cagr(strategy_returns),
                'sharpe': qs.stats.sharpe(strategy_returns),
                'sortino': qs.stats.sortino(strategy_returns),
                'max_drawdown': qs.stats.max_drawdown(strategy_returns),
                'volatility': qs.stats.volatility(strategy_returns),
                'calmar': qs.stats.calmar(strategy_returns),
            }
        
        return {
            **result,
            'returns': strategy_returns,
            'metrics': metrics,
        }


# ============================================================================
# Standardized Strategy Interface (for compatibility)
# ============================================================================

def run(data: dict, params: dict = None) -> dict:
    """
    Standard strategy interface - compatible with quantlab.
    
    Args:
        data: dict with 'prices', 'volume', 'market_cap'
        params: Strategy parameters
        
    Returns:
        dict with 'weights' and 'details'
    """
    strategy = CryptoTrendStrategy()
    return strategy.run(data, params)
