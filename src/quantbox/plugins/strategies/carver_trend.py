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

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta
from quantbox.plugins.strategies._universe import DEFAULT_STABLECOINS

logger = logging.getLogger(__name__)

# Carver's typical EWMAC spans
EWMAC_SPANS = [
    (8, 32),  # Fast trend
    (16, 64),  # Medium trend
    (32, 128),  # Slow trend
    (64, 256),  # Very slow trend
]

# Breakout windows
BREAKOUT_WINDOWS = [20, 40, 80, 160]

# Bollinger windows — Strategy v2 (2026-04-25). See docs/methodology/carver-trend-v2.md.
BOLLINGER_WINDOWS = [20, 40, 80]


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


def bollinger_forecast(
    prices: pd.Series,
    window: int,
    n_std: float = 2.0,
) -> pd.Series:
    """
    Bollinger-band forecast — position relative to MA bands.

    Inherently vol-scaled (uses standard deviation), which is why it adds
    diversification with a similar risk profile to EWMAC. Per Scott Phillips
    research: "significantly more predictive on crypto and tradfi futures"
    than EWMAC alone.

    Forecast = (price - MA) / (n_std × std)
    Roughly in [-1, +1] when within bands; can exceed on strong moves.

    Args:
        prices: Price series
        window: Lookback window for MA + std
        n_std: Bollinger band width in standard deviations (default 2.0)

    Returns:
        Forecast series approximately in [-1, +1] range
    """
    ma = prices.rolling(window, min_periods=window).mean()
    std = prices.rolling(window, min_periods=window).std()
    forecast = (prices - ma) / (n_std * std.replace(0, np.nan))
    return forecast


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
    forecasts: list[pd.Series],
    weights: list[float] | None = None,
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

    combined = sum(w * f for w, f in zip(weights, forecasts, strict=False))

    return combined


def generate_carver_forecast(
    prices: pd.Series,
    ewmac_spans: list[tuple[int, int]] | None = None,
    breakout_windows: list[int] | None = None,
    bollinger_windows: list[int] | None = None,
    ewmac_weight: float = 0.6,
    breakout_weight: float = 0.4,
    bollinger_weight: float = 0.0,
    bollinger_n_std: float = 2.0,
) -> pd.Series:
    """
    Generate combined Carver-style forecast for one instrument.

    Combines multiple EWMAC, breakout, and (optionally) Bollinger rules.
    Bollinger contributes only when ``bollinger_weight > 0``; default 0.0
    keeps the original two-family combination for backward compatibility.

    Weights should sum to 1.0; the function does not auto-normalize.

    Args:
        prices: Price series
        ewmac_spans: List of (fast, slow) spans for EWMAC
        breakout_windows: List of windows for breakout
        bollinger_windows: List of windows for Bollinger (default BOLLINGER_WINDOWS)
        ewmac_weight: Weight for EWMAC family
        breakout_weight: Weight for breakout family
        bollinger_weight: Weight for Bollinger family (default 0.0 = disabled)
        bollinger_n_std: Bollinger band width in standard deviations

    Returns:
        Combined, scaled, capped forecast in approximately ±20 range
    """
    if ewmac_spans is None:
        ewmac_spans = EWMAC_SPANS
    if breakout_windows is None:
        breakout_windows = BREAKOUT_WINDOWS
    if bollinger_windows is None:
        bollinger_windows = BOLLINGER_WINDOWS

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

    # Bollinger forecasts (additive — only contributes when bollinger_weight > 0)
    if bollinger_weight > 0:
        n_bollinger = len(bollinger_windows)
        for window in bollinger_windows:
            f = bollinger_forecast(prices, window, n_std=bollinger_n_std)
            f = f * 20  # naturally in [-1, +1], scale to ±20 like breakout
            f = cap_forecast(f, cap=20)
            all_forecasts.append(f)
            all_weights.append(bollinger_weight / n_bollinger)

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
    annualize: float = 252.0,
) -> pd.DataFrame:
    """
    Calculate instrument risk (annualized volatility).

    Args:
        prices: Price DataFrame
        vol_lookback: EMA span for volatility
        annualize: Bars per year for vol annualization. Default 252 (equity
            convention). Strategy callers should derive this from the
            pipeline-injected ``_pipeline_annualize`` per issue #20 / #23.

    Returns:
        Volatility DataFrame (annualized)
    """
    returns = prices.ffill().pct_change(fill_method=None)
    vol = returns.ewm(span=vol_lookback, min_periods=10).std() * np.sqrt(annualize)
    return vol


def calculate_position_sizes(
    forecasts: pd.DataFrame,
    volatilities: pd.DataFrame,
    target_vol: float = 0.25,
    forecast_cap: float = 20.0,
    idm: float | None = None,
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
        version="0.2.0",
        core_compat=">=0.1,<0.2",
        status="research",  # v0.2 methodology in DRAFT; flip via /promote-lock once validated
        description="Carver-style trend following with EWMAC, breakout, and optional Bollinger rules.",
        tags=("crypto", "trend", "carver"),
        capabilities=("backtest", "paper", "live"),
        outputs=("strategy_weights",),
        params_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_vol": {"type": "number", "minimum": 0.01, "maximum": 1.0, "default": 0.25},
                "vol_lookback": {"type": "integer", "minimum": 10, "maximum": 252, "default": 36},
                "idm": {"type": ["number", "null"], "minimum": 0.5, "maximum": 5.0, "default": None},
                "max_position": {"type": "number", "minimum": 0.0, "maximum": 2.0, "default": 1.0},
                "max_gross": {"type": "number", "minimum": 0.0, "maximum": 5.0, "default": 2.0},
                "allow_shorts": {"type": "boolean", "default": True},
                "ewmac_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.6},
                "breakout_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.4},
                "use_bollinger_feature": {
                    "type": "boolean",
                    "default": False,
                    "description": "When True, applies Strategy v2 weights (ewmac=0.4, breakout=0.3, bollinger=0.3) and overrides ewmac_weight/breakout_weight.",
                },
                "bollinger_n_std": {"type": "number", "minimum": 0.5, "maximum": 4.0, "default": 2.0},
                "use_universe_selection": {"type": "boolean", "default": False},
                "top_by_mcap": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 30},
                "top_by_volume": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 10},
                "volume_is_dollar": {"type": "boolean", "default": True},
                "output_periods": {"type": "integer", "minimum": 1, "maximum": 365, "default": 30},
            },
        },
        examples=(
            (
                "plugins:\n"
                "  strategies:\n"
                "    - name: strategy.carver_trend.v1\n"
                "      params:\n"
                "        target_vol: 0.30\n"
                "        use_universe_selection: true\n"
                "        top_by_volume: 8\n"
                "        use_bollinger_feature: true"
            ),
        ),
    )

    # Forecast parameters
    ewmac_spans: list[tuple[int, int]] = field(default_factory=lambda: EWMAC_SPANS.copy())
    breakout_windows: list[int] = field(default_factory=lambda: BREAKOUT_WINDOWS.copy())
    ewmac_weight: float = 0.6
    breakout_weight: float = 0.4

    # Bollinger feature (Strategy v2, 2026-04-25 — additive, default off).
    # When use_bollinger_feature=True, weights become (0.4, 0.3, 0.3)
    # for (ewmac, breakout, bollinger). See docs/methodology/carver-trend-v2.md
    # and notebook/projects/quantlab/scott-phillips-tweet-research.md.
    use_bollinger_feature: bool = False
    bollinger_windows: list[int] = field(default_factory=lambda: BOLLINGER_WINDOWS.copy())
    bollinger_n_std: float = 2.0

    # Position sizing
    target_vol: float = 0.25
    vol_lookback: int = 36
    idm: float | None = None  # Auto-calculate if None
    annualize: float | None = None  # None = pipeline-injected via _pipeline_annualize; falls back to 252.0

    # Risk limits
    max_position: float = 1.0
    max_gross: float = 2.0
    allow_shorts: bool = True  # Set False for long-only

    # Universe selection (set use_universe_selection=True to enable)
    use_universe_selection: bool = False
    top_by_mcap: int = 30
    top_by_volume: int = 10
    volume_is_dollar: bool = True  # All current sources (Binance, Hyperliquid) provide USD notional

    # Output
    output_periods: int = 30
    exclude_tickers: list[str] = field(default_factory=lambda: DEFAULT_STABLECOINS.copy())

    def describe(self) -> dict[str, Any]:
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
        data: dict[str, pd.DataFrame],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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

        # Resolve annualize: explicit (self/params) wins, else pipeline-injected, else 252.0.
        pipeline_annualize = (params or {}).get("_pipeline_annualize")
        if self.annualize is None:
            effective_annualize = float(pipeline_annualize) if pipeline_annualize is not None else 252.0
        else:
            effective_annualize = float(self.annualize)
            if pipeline_annualize is not None and abs(effective_annualize - pipeline_annualize) > 1:
                logger.warning(
                    "CarverTrendStrategy.annualize=%s overrides pipeline-derived %.1f. "
                    "If intentional, ignore; otherwise drop the explicit value.",
                    effective_annualize,
                    pipeline_annualize,
                )

        prices = data["prices"]

        # Filter tickers
        valid_tickers = [t for t in prices.columns if t not in self.exclude_tickers]
        prices = prices[valid_tickers]

        # 0. Universe selection — time-varying, no look-ahead bias.
        #    Computes a 0/1 mask for every date: top-N by trailing dollar-volume
        #    at that date.  Signals are generated for ALL tickers; the mask is
        #    applied to positions AFTER sizing so a coin only has non-zero weight
        #    when it is actually in the top-N at that point in time.
        universe_mask_ts: pd.DataFrame | None = None
        if self.use_universe_selection:
            from quantbox.plugins.strategies._universe import select_universe

            volume = data.get("volume", pd.DataFrame())
            market_cap = data.get("market_cap", pd.DataFrame())
            universe_mask_ts = select_universe(
                prices,
                volume.reindex(index=prices.index, columns=prices.columns).fillna(0.0),
                market_cap if not market_cap.empty else None,
                self.top_by_mcap,
                self.top_by_volume,
                self.exclude_tickers,
                volume_is_dollar=self.volume_is_dollar,
            )
            n_avg = universe_mask_ts.sum(axis=1).mean()
            logger.info(
                "Universe selection: avg %.1f active instruments from %d available (top-%d by volume)",
                n_avg,
                len(valid_tickers),
                self.top_by_volume,
            )

        logger.info(f"Running CarverTrend on {len(prices.columns)} instruments")

        # 1. Generate forecasts for each instrument.
        # Resolve effective forecast weights — when use_bollinger_feature is on,
        # apply Strategy v2 spec defaults (0.4 / 0.3 / 0.3); otherwise the
        # original two-family split (0.6 / 0.4 / 0.0) preserves backward compatibility.
        if self.use_bollinger_feature:
            ew_w, br_w, bo_w = 0.4, 0.3, 0.3
        else:
            ew_w, br_w, bo_w = self.ewmac_weight, self.breakout_weight, 0.0

        forecasts = {}
        for ticker in prices.columns:
            forecast = generate_carver_forecast(
                prices[ticker],
                ewmac_spans=self.ewmac_spans,
                breakout_windows=self.breakout_windows,
                bollinger_windows=self.bollinger_windows,
                ewmac_weight=ew_w,
                breakout_weight=br_w,
                bollinger_weight=bo_w,
                bollinger_n_std=self.bollinger_n_std,
            )
            forecasts[ticker] = forecast

        forecasts_df = pd.DataFrame(forecasts)

        # 2. Calculate instrument volatilities
        volatilities = calculate_instrument_risk(prices, self.vol_lookback, effective_annualize)

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

        # 5b. Apply time-varying universe mask: zero positions for coins that
        #     are not in the top-N at each date (no look-ahead bias).
        if universe_mask_ts is not None:
            mask = universe_mask_ts.reindex(index=positions.index, columns=positions.columns).fillna(0.0)
            positions = positions * mask

        # 6. Calculate exposure
        latest = positions.iloc[-1].dropna()
        long_exp = latest[latest > 0].sum()
        short_exp = abs(latest[latest < 0].sum())

        # 7. Simple weights
        simple = latest[abs(latest) > 0.001].to_dict()

        return {
            "weights": positions.tail(self.output_periods),
            "simple_weights": simple,
            "forecasts": forecasts_df.tail(self.output_periods),
            "details": {
                "volatilities": volatilities,
                "raw_forecasts": forecasts_df,
            },
            "exposure": {
                "long": float(long_exp),
                "short": float(short_exp),
                "net": float(long_exp - short_exp),
                "gross": float(long_exp + short_exp),
            },
        }

    def get_latest_weights(self, result: dict[str, Any]) -> dict[str, float]:
        """Extract latest weights."""
        return result["simple_weights"]

    def get_forecast_summary(self, result: dict[str, Any]) -> pd.DataFrame:
        """Get latest forecasts with direction."""
        forecasts = result["forecasts"].iloc[-1]
        weights = result["weights"].iloc[-1]

        summary = pd.DataFrame(
            {
                "forecast": forecasts,
                "weight": weights,
                "direction": ["LONG" if f > 0 else "SHORT" if f < 0 else "FLAT" for f in forecasts],
            }
        )

        return summary.sort_values("forecast", ascending=False)


# ============================================================================
# Standard Interface
# ============================================================================


def run(data: dict, params: dict = None) -> dict:
    """Standard strategy interface."""
    strategy = CarverTrendStrategy()
    return strategy.run(data, params)
