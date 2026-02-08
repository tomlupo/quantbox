"""
Carver-Style Trend Following Strategy

Systematic trend-following inspired by Rob Carver's approach from
"Systematic Trading" and "Leveraged Trading".

Key concepts:
- **Forecast**: Scaled signal from -20 to +20 (or -1 to +1)
- **Volatility targeting**: Scale positions to target portfolio vol
- **Forecast combining**: Multiple rules combined into one forecast
- **Position sizing**: forecast × vol_scalar × capital / instrument_risk

NOT market-neutral - goes with the trend. Long when bullish, flat/short when bearish.

## LLM Usage Guide

### Quick Start
```python
from quantbox.plugins.strategies import CarverTrendStrategy
from quantbox.plugins.datasources import BinanceDataFetcher

fetcher = BinanceDataFetcher()
data = fetcher.get_market_data(['BTC', 'ETH', 'SOL'], lookback_days=400)

strategy = CarverTrendStrategy(target_vol=0.25)
result = strategy.run(data)

# Positions sized by forecast and vol
print(result['simple_weights'])
# {'BTC': 0.35, 'ETH': 0.22, 'SOL': -0.08}  # Can be long, short, or flat
```

### Rob Carver Principles
1. **Trend rules**: EWMAC (exponentially weighted moving average crossover)
2. **Breakout rules**: Donchian-style channel breakouts
3. **Carry rules**: (not applicable to crypto spot)
4. **Forecast scaling**: Raw signal → scaled to avg abs value of 10
5. **Forecast capping**: Clip to [-20, +20] (or [-1, +1] normalized)
6. **Vol targeting**: Scale total exposure to target portfolio vol
7. **IDM**: Instrument Diversification Multiplier (more instruments → higher IDM)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import logging

from quantbox.contracts import PluginMeta

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

DEFAULT_STABLECOINS = [
    'USDT', 'USDC', 'BUSD', 'TUSD', 'DAI', 'MIM', 'USTC', 'FDUSD',
    'USDP', 'GUSD', 'FRAX', 'LUSD', 'USDD', 'PYUSD', 'USD1', 'USDJ',
    'EUR', 'EURC', 'EURT', 'EURS', 'PAXG', 'XAUT', 'WBTC', 'WETH',
    'BETH', 'ETHW', 'CBBTC', 'CBETH', 'BFUSD', 'AEUR',
]

# Carver's typical EWMAC spans
EWMAC_SPANS = [
    (8, 32),    # Fast trend
    (16, 64),   # Medium trend
    (32, 128),  # Slow trend
    (64, 256),  # Very slow trend
]

# Breakout windows
BREAKOUT_WINDOWS = [20, 40, 80, 160]


# ============================================================================
# Forecast Generation (Carver Style)
# ============================================================================

def ewmac_forecast(
    prices: pd.Series,
    fast_span: int,
    slow_span: int,
) -> pd.Series:
    """
    EWMAC (Exponentially Weighted Moving Average Crossover) forecast.
    
    Carver's primary trend rule. Signal = fast_ewma - slow_ewma,
    scaled by price volatility.
    
    Args:
        prices: Price series
        fast_span: Fast EMA span
        slow_span: Slow EMA span
        
    Returns:
        Raw forecast series
    """
    fast_ewma = prices.ewm(span=fast_span, min_periods=fast_span).mean()
    slow_ewma = prices.ewm(span=slow_span, min_periods=slow_span).mean()
    
    # Raw crossover
    raw_forecast = fast_ewma - slow_ewma
    
    # Scale by volatility (Carver uses price vol to normalize)
    vol = prices.diff().ewm(span=36, min_periods=10).std()
    
    # Normalized forecast
    forecast = raw_forecast / vol.replace(0, np.nan)
    
    return forecast


def breakout_forecast(
    prices: pd.Series,
    window: int,
) -> pd.Series:
    """
    Breakout forecast - position relative to recent range.
    
    Forecast = (price - midpoint) / (high - low)
    Ranges from -1 (at low) to +1 (at high).
    
    Args:
        prices: Price series
        window: Lookback window
        
    Returns:
        Forecast series [-1, +1]
    """
    rolling_high = prices.rolling(window, min_periods=window).max()
    rolling_low = prices.rolling(window, min_periods=window).min()
    midpoint = (rolling_high + rolling_low) / 2
    range_size = (rolling_high - rolling_low).replace(0, np.nan)
    
    # Position in range, scaled to [-1, +1]
    forecast = (prices - midpoint) / (range_size / 2)
    
    return forecast.clip(-1, 1)


def scale_forecast(
    forecast: pd.Series,
    target_abs_avg: float = 10.0,
    lookback: int = 252,
) -> pd.Series:
    """
    Scale forecast to target absolute average.
    
    Carver scales forecasts so average absolute value ≈ 10.
    This normalizes different rules to comparable scales.
    
    Args:
        forecast: Raw forecast
        target_abs_avg: Target average absolute forecast (Carver uses 10)
        lookback: Lookback for scaling estimation
        
    Returns:
        Scaled forecast
    """
    # Rolling mean absolute forecast
    abs_avg = forecast.abs().rolling(lookback, min_periods=20).mean()
    
    # Scale factor
    scale = target_abs_avg / abs_avg.replace(0, np.nan)
    
    # Apply scaling
    scaled = forecast * scale
    
    return scaled


def cap_forecast(
    forecast: pd.Series,
    cap: float = 20.0,
) -> pd.Series:
    """
    Cap forecast to maximum absolute value.
    
    Carver caps at ±20 (or ±2 if using -1 to +1 scale).
    
    Args:
        forecast: Scaled forecast
        cap: Maximum absolute value
        
    Returns:
        Capped forecast
    """
    return forecast.clip(-cap, cap)


def combine_forecasts(
    forecasts: List[pd.Series],
    weights: Optional[List[float]] = None,
) -> pd.Series:
    """
    Combine multiple forecasts into one.
    
    Args:
        forecasts: List of forecast series
        weights: Weights for each forecast (default: equal)
        
    Returns:
        Combined forecast
    """
    if weights is None:
        weights = [1.0 / len(forecasts)] * len(forecasts)
    
    combined = sum(w * f for w, f in zip(weights, forecasts))
    
    return combined


def generate_carver_forecast(
    prices: pd.Series,
    ewmac_spans: List[Tuple[int, int]] = None,
    breakout_windows: List[int] = None,
    ewmac_weight: float = 0.6,
    breakout_weight: float = 0.4,
) -> pd.Series:
    """
    Generate combined Carver-style forecast for one instrument.
    
    Combines multiple EWMAC and breakout rules.
    
    Args:
        prices: Price series
        ewmac_spans: List of (fast, slow) spans for EWMAC
        breakout_windows: List of windows for breakout
        ewmac_weight: Weight for EWMAC rules
        breakout_weight: Weight for breakout rules
        
    Returns:
        Combined, scaled, capped forecast
    """
    if ewmac_spans is None:
        ewmac_spans = EWMAC_SPANS
    if breakout_windows is None:
        breakout_windows = BREAKOUT_WINDOWS
    
    all_forecasts = []
    all_weights = []
    
    # EWMAC forecasts
    n_ewmac = len(ewmac_spans)
    for fast, slow in ewmac_spans:
        f = ewmac_forecast(prices, fast, slow)
        f = scale_forecast(f, target_abs_avg=10)
        f = cap_forecast(f, cap=20)
        all_forecasts.append(f)
        all_weights.append(ewmac_weight / n_ewmac)
    
    # Breakout forecasts
    n_breakout = len(breakout_windows)
    for window in breakout_windows:
        f = breakout_forecast(prices, window)
        f = f * 20  # Scale to ±20 range
        f = cap_forecast(f, cap=20)
        all_forecasts.append(f)
        all_weights.append(breakout_weight / n_breakout)
    
    # Combine
    combined = combine_forecasts(all_forecasts, all_weights)
    
    # Final cap
    combined = cap_forecast(combined, cap=20)
    
    return combined


# ============================================================================
# Position Sizing (Carver Style)
# ============================================================================

def calculate_instrument_risk(
    prices: pd.DataFrame,
    vol_lookback: int = 36,
) -> pd.DataFrame:
    """
    Calculate instrument risk (annualized volatility).
    
    Args:
        prices: Price DataFrame
        vol_lookback: EMA span for volatility
        
    Returns:
        Volatility DataFrame (annualized)
    """
    returns = prices.pct_change()
    vol = returns.ewm(span=vol_lookback, min_periods=10).std() * np.sqrt(365)
    return vol


def calculate_position_sizes(
    forecasts: pd.DataFrame,
    volatilities: pd.DataFrame,
    target_vol: float = 0.25,
    forecast_cap: float = 20.0,
    idm: Optional[float] = None,
) -> pd.DataFrame:
    """
    Calculate position sizes using Carver's formula.
    
    Position = (forecast / forecast_cap) × (target_vol / instrument_vol) × IDM / N
    
    Args:
        forecasts: Forecast DataFrame (each column = instrument)
        volatilities: Volatility DataFrame
        target_vol: Target portfolio volatility
        forecast_cap: Maximum forecast (for scaling)
        idm: Instrument Diversification Multiplier (default: auto-calculate)
        
    Returns:
        Position sizes as portfolio weights
    """
    n_instruments = len(forecasts.columns)
    
    # Auto-calculate IDM if not provided
    # Carver's rule of thumb: IDM ≈ sqrt(N) for uncorrelated, ~1.5 for correlated
    if idm is None:
        idm = min(np.sqrt(n_instruments), 2.5)
    
    # Normalize forecast to [-1, +1] scale
    normalized_forecast = forecasts / forecast_cap
    
    # Volatility scalar per instrument
    vol_scalar = target_vol / volatilities.replace(0, np.nan)
    
    # Position = forecast × vol_scalar × IDM / N
    positions = normalized_forecast * vol_scalar * idm / n_instruments
    
    return positions


def apply_position_limits(
    positions: pd.DataFrame,
    max_position: float = 1.0,
    max_gross: float = 2.0,
) -> pd.DataFrame:
    """
    Apply position limits.
    
    Args:
        positions: Raw position sizes
        max_position: Maximum position per instrument
        max_gross: Maximum gross exposure
        
    Returns:
        Limited positions
    """
    # Per-instrument cap
    limited = positions.clip(-max_position, max_position)
    
    # Gross exposure cap
    gross = limited.abs().sum(axis=1)
    scale = (max_gross / gross).clip(upper=1.0)
    limited = limited.mul(scale, axis=0)
    
    return limited


# ============================================================================
# Strategy Class
# ============================================================================

@dataclass
class CarverTrendStrategy:
    """
    Carver-Style Trend Following Strategy.
    
    Systematic trend-following with:
    - Multiple EWMAC (moving average crossover) rules
    - Breakout rules
    - Volatility targeting
    - Forecast combining and scaling
    
    NOT market-neutral - goes with the trend.
    
    ## Quick Start
    ```python
    strategy = CarverTrendStrategy(target_vol=0.25)
    result = strategy.run(data)
    ```
    """

    meta = PluginMeta(
        name="strategy.carver_trend.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Carver-style trend following with EWMAC and breakout rules",
        tags=("crypto", "trend", "carver"),
    )

    # Forecast parameters
    ewmac_spans: List[Tuple[int, int]] = field(default_factory=lambda: EWMAC_SPANS.copy())
    breakout_windows: List[int] = field(default_factory=lambda: BREAKOUT_WINDOWS.copy())
    ewmac_weight: float = 0.6
    breakout_weight: float = 0.4
    
    # Position sizing
    target_vol: float = 0.25
    vol_lookback: int = 36
    idm: Optional[float] = None  # Auto-calculate if None
    
    # Risk limits
    max_position: float = 1.0
    max_gross: float = 2.0
    allow_shorts: bool = True  # Set False for long-only

    # Universe selection (set use_universe_selection=True to enable)
    use_universe_selection: bool = False
    top_by_mcap: int = 30
    top_by_volume: int = 10

    # Output
    output_periods: int = 30
    exclude_tickers: List[str] = field(default_factory=lambda: DEFAULT_STABLECOINS.copy())
    
    def describe(self) -> Dict[str, Any]:
        """Describe strategy for LLM introspection."""
        return {
            "name": "CarverTrendFollowing",
            "type": "trend_following",
            "style": "Rob Carver systematic trading",
            "purpose": "Trend-following with vol targeting and forecast combining",
            "parameters": {
                "ewmac_spans": self.ewmac_spans,
                "breakout_windows": self.breakout_windows,
                "target_vol": self.target_vol,
            },
            "rules": {
                "ewmac": f"{len(self.ewmac_spans)} EWMAC rules (weight: {self.ewmac_weight})",
                "breakout": f"{len(self.breakout_windows)} breakout rules (weight: {self.breakout_weight})",
            },
            "features": [
                "Forecast scaling (avg abs = 10)",
                "Forecast capping (±20)",
                "Vol targeting",
                "IDM (Instrument Diversification Multiplier)",
            ],
        }
    
    def run(
        self,
        data: Dict[str, pd.DataFrame],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run Carver trend strategy.
        
        Args:
            data: Dict with 'prices', 'volume', 'market_cap'
            params: Optional parameter overrides
            
        Returns:
            Dict with weights, forecasts, and details
        """
        if params:
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        prices = data['prices']
        
        # Filter tickers
        valid_tickers = [t for t in prices.columns if t not in self.exclude_tickers]
        prices = prices[valid_tickers]
        
        logger.info(f"Running CarverTrend on {len(valid_tickers)} instruments")
        
        # 1. Generate forecasts for each instrument
        forecasts = {}
        for ticker in prices.columns:
            forecast = generate_carver_forecast(
                prices[ticker],
                self.ewmac_spans,
                self.breakout_windows,
                self.ewmac_weight,
                self.breakout_weight,
            )
            forecasts[ticker] = forecast
        
        forecasts_df = pd.DataFrame(forecasts)
        
        # 2. Calculate instrument volatilities
        volatilities = calculate_instrument_risk(prices, self.vol_lookback)
        
        # 3. Calculate position sizes
        positions = calculate_position_sizes(
            forecasts_df,
            volatilities,
            target_vol=self.target_vol,
            idm=self.idm,
        )
        
        # 4. Apply shorts restriction if needed
        if not self.allow_shorts:
            positions = positions.clip(lower=0)
        
        # 5. Apply position limits
        positions = apply_position_limits(
            positions,
            max_position=self.max_position,
            max_gross=self.max_gross,
        )

        # 5b. Universe selection (optional)
        if self.use_universe_selection:
            from quantbox.plugins.strategies._universe import select_universe

            volume = data.get("volume", pd.DataFrame())
            market_cap = data.get("market_cap", pd.DataFrame())
            universe_mask = select_universe(
                prices,
                volume.reindex(index=prices.index, columns=prices.columns).fillna(0.0),
                market_cap if not market_cap.empty else None,
                self.top_by_mcap,
                self.top_by_volume,
                self.exclude_tickers,
            )
            positions = positions * universe_mask.reindex(
                index=positions.index, columns=positions.columns,
            ).fillna(0.0)
            n_in = int(universe_mask.iloc[-1].sum()) if not universe_mask.empty else 0
            logger.info(
                "Universe selection: top %d by volume from %d available",
                n_in, len(prices.columns),
            )

        # 6. Calculate exposure
        latest = positions.iloc[-1].dropna()
        long_exp = latest[latest > 0].sum()
        short_exp = abs(latest[latest < 0].sum())
        
        # 7. Simple weights
        simple = latest[abs(latest) > 0.001].to_dict()
        
        return {
            'weights': positions.tail(self.output_periods),
            'simple_weights': simple,
            'forecasts': forecasts_df.tail(self.output_periods),
            'details': {
                'volatilities': volatilities,
                'raw_forecasts': forecasts_df,
            },
            'exposure': {
                'long': float(long_exp),
                'short': float(short_exp),
                'net': float(long_exp - short_exp),
                'gross': float(long_exp + short_exp),
            },
        }
    
    def get_latest_weights(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Extract latest weights."""
        return result['simple_weights']
    
    def get_forecast_summary(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Get latest forecasts with direction."""
        forecasts = result['forecasts'].iloc[-1]
        weights = result['weights'].iloc[-1]
        
        summary = pd.DataFrame({
            'forecast': forecasts,
            'weight': weights,
            'direction': ['LONG' if f > 0 else 'SHORT' if f < 0 else 'FLAT' for f in forecasts],
        })
        
        return summary.sort_values('forecast', ascending=False)


# ============================================================================
# Standard Interface
# ============================================================================

def run(data: dict, params: dict = None) -> dict:
    """Standard strategy interface."""
    strategy = CarverTrendStrategy()
    return strategy.run(data, params)
