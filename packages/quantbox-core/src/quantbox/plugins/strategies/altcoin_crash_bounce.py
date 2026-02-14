"""
Altcoin Crash Bounce Strategy - Mean Reversion Entry on Capitulation

Long-only mean-reversion strategy that enters altcoin positions after significant
price crashes with volume spikes (capitulation signals), with TP/SL exits and time decay.

## LLM Usage Guide

### Quick Start
```python
from quantbox.plugins.strategies import AltcoinCrashBounceStrategy
from quantbox.plugins.datasources import BinanceDataFetcher

fetcher = BinanceDataFetcher()
data = fetcher.get_market_data(universe='top100', lookback_days=180, interval='1h')

strategy = AltcoinCrashBounceStrategy()
result = strategy.run(data)

# Weights represent active positions (8% each when in position)
print(result['simple_weights'])
# {'SOL': 0.08, 'AVAX': 0.08, 'LINK': 0.08}
```

### Strategy Logic
1. **Crash Detection**: Price drops 15%+ from 24h rolling high
2. **Volume Spike**: Current volume >= 1.3x the 20-day (480h) average volume
3. **Entry**: Long when both conditions met (capitulation signal)
4. **Exit Conditions** (first triggered wins):
   - Take Profit: +12% from entry
   - Stop Loss: -12% from entry  
   - Time Decay: 72 hours max hold period
5. **Position Sizing**: 8% per position, max 15 concurrent positions
6. **Circuit Breaker**: Max 6 entries per rolling 24h period

### Risk Features
- Fixed fractional sizing (8% per position)
- Symmetric TP/SL (12%)
- Time-based exit prevents holding losers forever
- Circuit breaker limits cascade entries during market-wide crashes
- 2.5% slippage assumption for crash conditions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta

logger = logging.getLogger(__name__)


# ============================================================================
# Default parameters
# ============================================================================

DEFAULT_STABLECOINS = [
    "USDT", "USDC", "BUSD", "TUSD", "DAI", "MIM", "USTC", "FDUSD",
    "USDP", "GUSD", "FRAX", "LUSD", "USDD", "PYUSD", "USD1", "USDJ",
    "EUR", "EURC", "EURT", "EURS", "PAXG", "XAUT", "WBTC", "WETH",
    "BETH", "ETHW", "CBBTC", "CBETH", "BFUSD", "AEUR",
]


# ============================================================================
# Signal Generation
# ============================================================================


def compute_crash_signals(
    prices: pd.DataFrame,
    lookback_periods: int = 24,
    crash_threshold_pct: float = -15.0,
) -> pd.DataFrame:
    """
    Compute crash percentage from rolling high.
    
    Args:
        prices: Price DataFrame (datetime index x ticker columns)
        lookback_periods: Periods for rolling high (24 for 24h with hourly data)
        crash_threshold_pct: Threshold for crash detection (e.g., -15 means 15% drop)
    
    Returns:
        DataFrame of crash percentages (negative values = price below high)
    """
    rolling_high = prices.rolling(window=lookback_periods, min_periods=lookback_periods).max()
    crash_pct = (prices - rolling_high) / rolling_high * 100
    return crash_pct


def compute_volume_ratio(
    volume: pd.DataFrame,
    lookback_periods: int = 480,
) -> pd.DataFrame:
    """
    Compute volume ratio vs rolling average.
    
    Args:
        volume: Volume DataFrame
        lookback_periods: Periods for SMA (480h = 20 days with hourly data)
    
    Returns:
        DataFrame of volume ratios (>1 means above average volume)
    """
    vol_sma = volume.rolling(window=lookback_periods, min_periods=lookback_periods // 2).mean()
    volume_ratio = volume / vol_sma.replace(0, np.nan)
    return volume_ratio


def generate_entry_signals(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    crash_threshold_pct: float = -15.0,
    volume_spike_ratio: float = 1.3,
    lookback_periods: int = 24,
    volume_lookback_periods: int = 480,
) -> pd.DataFrame:
    """
    Generate entry signals: crash + volume spike = capitulation entry.
    
    Returns:
        DataFrame of boolean entry signals
    """
    crash_pct = compute_crash_signals(prices, lookback_periods, crash_threshold_pct)
    vol_ratio = compute_volume_ratio(volume, volume_lookback_periods)
    
    # Entry when both conditions met
    crash_condition = crash_pct <= crash_threshold_pct
    volume_condition = vol_ratio >= volume_spike_ratio
    
    entry_signals = crash_condition & volume_condition
    return entry_signals.fillna(False)


# ============================================================================
# Position Tracking with TP/SL/Time Exits
# ============================================================================


def simulate_positions(
    prices: pd.DataFrame,
    entry_signals: pd.DataFrame,
    take_profit_pct: float = 12.0,
    stop_loss_pct: float = 12.0,
    max_hold_periods: int = 72,
    max_positions: int = 15,
    position_size_pct: float = 8.0,
    circuit_breaker_entries: int = 6,
    circuit_breaker_periods: int = 24,
    slippage_pct: float = 2.5,
    commission_pct: float = 0.1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Simulate position entry/exit with TP/SL/time decay.
    
    This is the core backtesting logic that tracks position state.
    
    Args:
        prices: Price DataFrame
        entry_signals: Boolean DataFrame of entry opportunities
        take_profit_pct: Take profit percentage (12 = +12%)
        stop_loss_pct: Stop loss percentage (12 = -12%)
        max_hold_periods: Max periods to hold before time exit
        max_positions: Maximum concurrent positions
        position_size_pct: Position size as % of portfolio
        circuit_breaker_entries: Max entries per rolling period
        circuit_breaker_periods: Rolling window for circuit breaker
        slippage_pct: Expected slippage for fills
        commission_pct: Commission per trade
    
    Returns:
        Tuple of (weights DataFrame, details dict)
    """
    tickers = prices.columns.tolist()
    dates = prices.index
    n_dates = len(dates)
    n_tickers = len(tickers)
    
    # Output: weights at each timestamp
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)
    
    # Position tracking: dict of ticker -> {entry_idx, entry_price}
    positions: dict[str, dict[str, Any]] = {}
    
    # Entry tracking for circuit breaker
    recent_entries: list[int] = []  # List of entry indices
    
    # Trade log
    trades: list[dict] = []
    
    # Iterate through time
    for i in range(n_dates):
        current_prices = prices.iloc[i]
        
        # --- Check exits for existing positions ---
        tickers_to_close = []
        for ticker, pos in positions.items():
            entry_price = pos['entry_price']
            entry_idx = pos['entry_idx']
            current_price = current_prices.get(ticker, np.nan)
            
            if pd.isna(current_price):
                continue
            
            # Calculate return since entry
            pct_change = (current_price - entry_price) / entry_price * 100
            hold_periods = i - entry_idx
            
            # Check exit conditions
            exit_reason = None
            if pct_change >= take_profit_pct:
                exit_reason = 'take_profit'
            elif pct_change <= -stop_loss_pct:
                exit_reason = 'stop_loss'
            elif hold_periods >= max_hold_periods:
                exit_reason = 'time_decay'
            
            if exit_reason:
                tickers_to_close.append(ticker)
                trades.append({
                    'ticker': ticker,
                    'entry_date': dates[entry_idx],
                    'exit_date': dates[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pct_return': pct_change,
                    'hold_periods': hold_periods,
                    'exit_reason': exit_reason,
                })
        
        # Close positions
        for ticker in tickers_to_close:
            del positions[ticker]
        
        # --- Check circuit breaker ---
        # Remove old entries from tracking
        recent_entries = [idx for idx in recent_entries if i - idx < circuit_breaker_periods]
        can_enter = len(recent_entries) < circuit_breaker_entries
        
        # --- Check new entries ---
        if can_enter and len(positions) < max_positions:
            entry_sigs = entry_signals.iloc[i]
            
            # Find tickers with entry signals not already in positions
            for ticker in tickers:
                if len(positions) >= max_positions:
                    break
                if ticker in positions:
                    continue
                if not entry_sigs.get(ticker, False):
                    continue
                
                entry_price = current_prices.get(ticker, np.nan)
                if pd.isna(entry_price) or entry_price <= 0:
                    continue
                
                # Enter position
                positions[ticker] = {
                    'entry_idx': i,
                    'entry_price': entry_price * (1 + slippage_pct / 100),  # Slippage on entry
                }
                recent_entries.append(i)
                
                # Check circuit breaker again
                if len(recent_entries) >= circuit_breaker_entries:
                    break
        
        # --- Update weights ---
        for ticker in tickers:
            if ticker in positions:
                weights.loc[dates[i], ticker] = position_size_pct / 100
            else:
                weights.loc[dates[i], ticker] = 0.0
    
    details = {
        'trades': trades,
        'n_trades': len(trades),
        'entry_signals': entry_signals,
    }
    
    return weights, details


# ============================================================================
# Strategy Class
# ============================================================================


@dataclass
class AltcoinCrashBounceStrategy:
    """
    Altcoin Crash Bounce - Mean Reversion Entry on Capitulation.
    
    Long-only strategy that enters altcoin positions after significant
    price crashes with volume confirmation (capitulation signals).
    
    ## Quick Start
    ```python
    strategy = AltcoinCrashBounceStrategy()
    result = strategy.run(data)  # data from BinanceDataFetcher
    weights = result['weights']
    ```
    """
    
    meta = PluginMeta(
        name="strategy.altcoin_crash_bounce.v62",
        kind="strategy",
        version="0.6.2",
        core_compat=">=0.1,<0.2",
        description="Altcoin crash bounce - mean reversion entry on capitulation",
        tags=("crypto", "mean_reversion", "altcoins", "crash", "bounce"),
    )
    
    # Signal parameters
    crash_threshold_pct: float = -15.0
    volume_spike_ratio: float = 1.3
    lookback_periods: int = 24  # 24h for hourly data
    volume_lookback_periods: int = 480  # 480h = 20 days
    
    # Exit parameters
    take_profit_pct: float = 12.0
    stop_loss_pct: float = 12.0
    max_hold_hours: int = 72
    
    # Position sizing
    position_size_pct: float = 8.0
    max_positions: int = 15
    
    # Risk management
    circuit_breaker_entries: int = 6
    circuit_breaker_hours: int = 24
    slippage_pct: float = 2.5
    commission_pct: float = 0.1
    
    # Universe filtering
    min_market_cap_usd: float = 10_000_000
    min_daily_volume_usd: float = 500_000
    exclude_tickers: list[str] = field(default_factory=lambda: DEFAULT_STABLECOINS.copy())
    
    # Output
    output_periods: int = 168  # 1 week of hourly data
    
    def describe(self) -> dict[str, Any]:
        """Describe strategy for LLM introspection."""
        return {
            "name": "AltcoinCrashBounce",
            "type": "mean_reversion",
            "purpose": "Long altcoins after crash + volume spike (capitulation entry)",
            "data_frequency": "hourly",
            "parameters": {
                "crash_threshold_pct": self.crash_threshold_pct,
                "volume_spike_ratio": self.volume_spike_ratio,
                "take_profit_pct": self.take_profit_pct,
                "stop_loss_pct": self.stop_loss_pct,
                "max_hold_hours": self.max_hold_hours,
                "position_size_pct": self.position_size_pct,
                "max_positions": self.max_positions,
            },
            "entry": "crash_pct <= -15% AND volume_ratio >= 1.3",
            "exit": "TP: +12%, SL: -12%, Time: 72h",
            "risk": {
                "max_exposure": f"{self.max_positions * self.position_size_pct}%",
                "circuit_breaker": f"{self.circuit_breaker_entries} entries per {self.circuit_breaker_hours}h",
            },
        }
    
    def filter_universe(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        market_cap: pd.DataFrame | None = None,
    ) -> list[str]:
        """
        Filter to valid altcoin universe.
        
        Args:
            prices: Price DataFrame
            volume: Volume DataFrame
            market_cap: Market cap DataFrame (optional)
        
        Returns:
            List of valid tickers
        """
        valid_tickers = []
        
        for ticker in prices.columns:
            # Skip excluded tickers (stablecoins, wrapped assets)
            if ticker in self.exclude_tickers:
                continue
            
            # Check minimum daily volume (approximate from hourly)
            if volume is not None and ticker in volume.columns:
                avg_daily_vol = volume[ticker].tail(168).sum() / 7  # Last week avg
                if avg_daily_vol < self.min_daily_volume_usd:
                    continue
            
            # Check minimum market cap
            if market_cap is not None and ticker in market_cap.columns:
                latest_mcap = market_cap[ticker].iloc[-1]
                if pd.notna(latest_mcap) and latest_mcap < self.min_market_cap_usd:
                    continue
            
            valid_tickers.append(ticker)
        
        return valid_tickers
    
    def run(
        self,
        data: dict[str, pd.DataFrame],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run strategy and return weights.
        
        Args:
            data: Dict with 'prices', 'volume', optionally 'market_cap'
            params: Optional parameter overrides
        
        Returns:
            Dict with:
            - 'weights': DataFrame of position weights over time
            - 'simple_weights': Latest weights as dict
            - 'details': Trade log and intermediate calculations
        """
        # Apply parameter overrides
        if params:
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        prices = data["prices"]
        volume = data.get("volume", pd.DataFrame(index=prices.index))
        market_cap = data.get("market_cap")
        
        logger.info(f"Running AltcoinCrashBounceStrategy on {len(prices.columns)} tickers, {len(prices)} periods")
        
        # 1. Filter universe
        valid_tickers = self.filter_universe(prices, volume, market_cap)
        logger.info(f"Universe filtered to {len(valid_tickers)} tickers")
        
        if not valid_tickers:
            logger.warning("No valid tickers in universe")
            return {
                "weights": pd.DataFrame(index=prices.index),
                "simple_weights": {},
                "details": {"trades": [], "n_trades": 0},
            }
        
        prices_filtered = prices[valid_tickers]
        volume_filtered = volume[[t for t in valid_tickers if t in volume.columns]]
        
        # 2. Generate entry signals
        entry_signals = generate_entry_signals(
            prices_filtered,
            volume_filtered,
            crash_threshold_pct=self.crash_threshold_pct,
            volume_spike_ratio=self.volume_spike_ratio,
            lookback_periods=self.lookback_periods,
            volume_lookback_periods=self.volume_lookback_periods,
        )
        
        # 3. Simulate positions with TP/SL/time exits
        weights, details = simulate_positions(
            prices_filtered,
            entry_signals,
            take_profit_pct=self.take_profit_pct,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_periods=self.max_hold_hours,  # Assuming hourly data
            max_positions=self.max_positions,
            position_size_pct=self.position_size_pct,
            circuit_breaker_entries=self.circuit_breaker_entries,
            circuit_breaker_periods=self.circuit_breaker_hours,
            slippage_pct=self.slippage_pct,
            commission_pct=self.commission_pct,
        )
        
        # 4. Extract simple weights for latest period
        latest = weights.iloc[-1].dropna()
        latest = latest[latest > 0.001].to_dict()
        
        # 5. Compute trade statistics
        trades = details['trades']
        if trades:
            returns = [t['pct_return'] for t in trades]
            exit_reasons = [t['exit_reason'] for t in trades]
            
            details['stats'] = {
                'n_trades': len(trades),
                'avg_return': np.mean(returns),
                'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                'avg_hold_periods': np.mean([t['hold_periods'] for t in trades]),
                'exits_by_reason': {
                    'take_profit': exit_reasons.count('take_profit'),
                    'stop_loss': exit_reasons.count('stop_loss'),
                    'time_decay': exit_reasons.count('time_decay'),
                },
            }
            logger.info(f"Strategy generated {len(trades)} trades, avg return: {details['stats']['avg_return']:.2f}%")
        
        return {
            "weights": weights.tail(self.output_periods),
            "simple_weights": latest,
            "details": details,
        }
    
    def get_latest_weights(self, result: dict[str, Any]) -> dict[str, float]:
        """Extract latest weights as a simple dict."""
        weights = result["weights"]
        latest = weights.iloc[-1].dropna()
        return latest[latest > 0.001].to_dict()


# ============================================================================
# Standardized Strategy Interface
# ============================================================================


def run(data: dict, params: dict = None) -> dict:
    """Standard strategy interface."""
    strategy = AltcoinCrashBounceStrategy()
    return strategy.run(data, params)
