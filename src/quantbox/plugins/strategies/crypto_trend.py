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

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

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

from quantbox.plugins.strategies._universe import (
    DEFAULT_STABLECOINS,
    select_universe as select_universe_vectorized,
    select_universe_duckdb,
)


# ============================================================================
# Constants
# ============================================================================

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
    _breakout = (prices >= high).astype(float)  # noqa: F841 kept for reference

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
            if signal[i - 1] == 1:
                # In position - update trailing stop
                trailing_stop[i] = max(trailing_stop[i - 1], mid_arr[i])
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
    windows: list[int],
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
        window_signals = np.column_stack([signal_fn(prices[ticker], w).values for w in windows])
        # Ensemble: mean across windows
        signals[ticker] = window_signals.mean(axis=1)

    return pd.DataFrame(signals, index=prices.index)


# ============================================================================
# Volatility Targeting
# ============================================================================


def compute_volatility_scalers(
    prices: pd.DataFrame,
    vol_targets: list[float],
    vol_lookback: int = 60,
    annualization_factor: float = 365.0,
    clip_range: tuple[float, float] | None = (0.1, 10.0),
) -> dict[str, pd.DataFrame]:
    """
    Compute volatility scalers for each target.

    Scaler = target_vol / realized_vol
    This scales position size inversely to volatility.

    Args:
        prices: Price DataFrame
        vol_targets: List of target volatilities (e.g., [0.25, 0.50])
        vol_lookback: Rolling window for volatility estimation
        annualization_factor: Days per year for annualization
        clip_range: (lower, upper) bounds for the scaler. Default ``(0.1, 10.0)``
            preserves prior behaviour. Pass ``None`` to skip clipping (notebook-
            faithful for the Robuxio TrendCatcher v2 replication).

    Returns:
        Dict mapping vol_target_str to scaler DataFrame
    """
    # Compute annualized volatility
    returns = prices.ffill().pct_change(fill_method=None)
    realized_vol = returns.rolling(vol_lookback).std() * np.sqrt(annualization_factor)

    scalers = {}
    for vt in vol_targets:
        scaler = vt / realized_vol.replace(0, np.nan)
        if clip_range is not None:
            scaler = scaler.clip(lower=clip_range[0], upper=clip_range[1])
        scalers[f"{int(vt * 100)}"] = scaler

    return scalers


def compute_inv_vol_track(
    off_weights: pd.DataFrame,
    prices: pd.DataFrame,
    tranches: list[int],
    vol_lookback: int = 60,
    annualization_factor: float = 365.0,
) -> pd.DataFrame:
    """
    Build the ``vol_target='inv_vol'`` track from the ``off`` track weights.

    Replicates notebook v2 cells 110-115:

    - ``inv_vol = 1 / realized_vol``
    - ``iv_raw = inv_vol.where(off != 0) * off_weights``  (inherit signal pattern)
    - Per-row normalisation so the inv_vol track sums to the same row total as
      the ``off`` track (preserves total exposure)
    - Tranching applied AFTER normalisation, per ``tranches`` entry

    Args:
        off_weights: per-(date, ticker) ``off`` track weights (signal x universe x position_weight).
            Rows where this is all-zero (e.g., regime-off, no longs) produce an
            all-zero inv_vol row.
        prices: Price DataFrame used to compute realised volatility.
        tranches: Tranching periods (rolling-mean windows).
        vol_lookback: Rolling window for the volatility estimate.
        annualization_factor: Days per year (365 for crypto, 252 for equities).

    Returns:
        DataFrame with MultiIndex columns ``(vol_target, tranches, ticker)`` where
        ``vol_target`` is always the literal string ``"inv_vol"``.
    """
    returns = prices.ffill().pct_change(fill_method=None)
    realized_vol = returns.rolling(vol_lookback).std() * np.sqrt(annualization_factor)
    inv_vol = 1.0 / realized_vol.replace(0, np.nan)

    iv_raw = inv_vol.where(off_weights != 0).mul(off_weights)
    iv_rowsum = iv_raw.sum(axis=1).replace(0, np.nan)
    off_rowsum = off_weights.sum(axis=1)
    iv_normalised = iv_raw.div(iv_rowsum, axis=0).mul(off_rowsum, axis=0).fillna(0)

    out: dict[tuple[str, int, str], pd.Series] = {}
    for t in tranches:
        if t > 1:
            iv_t = iv_normalised.rolling(t, min_periods=1).mean()
        else:
            iv_t = iv_normalised
        for col in iv_t.columns:
            out[("inv_vol", t, col)] = iv_t[col]

    df = pd.DataFrame(out)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["vol_target", "tranches", "ticker"])
    return df


# ============================================================================
# Portfolio Construction
# ============================================================================


def construct_weights(
    signals: pd.DataFrame,
    universe: pd.DataFrame,
    scalers: dict[str, pd.DataFrame],
    tranches: list[int],
    normalize: bool = True,
    signals_have_universe: bool = False,
) -> pd.DataFrame:
    """
    Construct portfolio weights from signals.

    Steps:
    1. Scale signals by volatility scaler
    2. Apply tranching (rolling mean for smoother transitions)
    3. Mask by tradable universe  (skipped when ``signals_have_universe=True``)
    4. Normalize to sum to 1 (optional)

    Args:
        signals_have_universe: When ``True``, the caller has already multiplied
            ``signals`` by the universe mask, so we skip the post-tranche
            ``× universe`` step. This matters at universe-transition days:
            when a coin leaves the universe, post-tranche masking zeros its
            weight immediately, whereas notebook-style pre-tranche masking
            lets the rolling mean bleed it out over ``tranches`` days. Set
            this when replicating notebook semantics where
            ``weights = signals × universe × position_weight``  is computed
            BEFORE tranching.

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

            # Apply universe mask (skip if signals already have it baked in).
            # CRITICAL: a NON-selected asset must get weight 0, never NaN. Plain
            # ``sig_tranched * universe`` yields ``NaN * 0 = NaN`` whenever the
            # signal is NaN (e.g. a newly-listed coin with insufficient history)
            # AND the asset is outside the selected universe. That NaN then leaks
            # into ``aggregated_weights`` and FREEZES the rebalancer ("Invalid
            # (NaN)") even though the asset is never held. Use ``where`` so the
            # mask hard-zeros non-selected columns regardless of the signal value,
            # then fill any residual NaN (selected asset with NaN signal due to
            # short history) with 0 so no NaN can ever reach the aggregator.
            if signals_have_universe:
                w = sig_tranched.fillna(0.0)
            else:
                # Guard the mask itself: `universe != 0` is True for a NaN mask
                # cell (NaN != 0 == True), which would RETAIN a non-selected
                # asset. Treat any NaN/absent mask value as not-selected via
                # ``fillna(0) > 0`` so a malformed mask can never leak a coin
                # into the book regardless of which universe path produced it.
                w = sig_tranched.where(universe.fillna(0.0) > 0, 0.0).fillna(0.0)

            # Normalize by number of positions
            if normalize:
                n_positions = universe.sum(axis=1).replace(0, np.nan)
                w = w.div(n_positions, axis=0).fillna(0.0)

            # Store with MultiIndex key
            for col in w.columns:
                weights_dict[(vt_key, t, col)] = w[col]

    weights = pd.DataFrame(weights_dict)
    weights.columns = pd.MultiIndex.from_tuples(weights.columns, names=["vol_target", "tranches", "ticker"])

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
        selected = weights_df.xs((vol_target, tranche), axis=1, level=("vol_target", "tranches"))
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
    lookback_windows: list[int] = field(default_factory=lambda: DEFAULT_LOOKBACK_WINDOWS.copy())
    vol_targets: list[Any] = field(default_factory=lambda: DEFAULT_VOL_TARGETS.copy())
    tranches: list[int] = field(default_factory=lambda: DEFAULT_TRANCHES.copy())

    # Universe parameters
    top_by_mcap: int = 30
    top_by_volume: int = 10
    exclude_tickers: list[str] = field(default_factory=lambda: DEFAULT_STABLECOINS.copy())

    # Volatility parameters
    vol_lookback: int = 60

    # Output parameters
    output_periods: int = 30
    normalize_weights: bool = True

    # Performance options
    use_duckdb: bool = True
    use_trailing_stop: bool = True

    # Notebook-parity knobs (SSRN "Catching Crypto Trends" replication)
    #  * ``inv_vol_track``: when True, derive an additional ``vol_target='inv_vol'``
    #    track from the ``off`` track via :func:`compute_inv_vol_track` (notebook
    #    cells 678–686). Requires ``"off"`` in ``vol_targets``.
    #  * ``clip_vol_scaler``: pass-through to :func:`compute_volatility_scalers`.
    #    Default ``(0.1, 10.0)`` preserves the prior behaviour; pass ``None`` to
    #    skip clipping (notebook-faithful — no clamp applied).
    #  * ``regime_ticker``: ticker used for the ``donchian_overlay`` diagnostic
    #    (notebook charts 3, 6, 22). Default ``"BTC"``. Set to ``None`` to disable
    #    the regime/donchian-overlay diagnostic entirely (e.g. for a literal port
    #    comparison vs quantlab's crypto_trend_catcher, which has no regime overlay).
    #    The overlay is DIAGNOSTIC-ONLY — it never affects the output weights.
    #  * ``output_track``: (vol_target, tranches) slice returned in ``weights``
    #    for the pipeline. The full multi-index panel is preserved under
    #    ``details["weights_panel"]`` for research. Default ``("50", 5)`` —
    #    the notebook-chosen paper trade variant.
    inv_vol_track: bool = False
    clip_vol_scaler: tuple[float, float] | None = (0.1, 10.0)
    regime_ticker: str | None = "BTC"
    output_track: tuple[str, int] = ("50", 5)
    # Volume convention for universe selection. Curated Binance/Hyperliquid
    # datasets store quote-volume (USD notional) directly, so DO NOT multiply
    # by price again. Default ``True`` matches the convention of every
    # currently-shipped crypto data source (binance.live_data.v1,
    # binance.futures_data.v1, dataset.curated.v1) and aligns with the
    # defaults of strategy.crypto_regime_trend.v1 and strategy.carver_trend.v1.
    # Pass ``False`` only when supplying a custom data source that stores
    # volume in base-coin units.
    volume_is_dollar: bool = True
    volume_rolling_window: int = 1
    min_listing_days: int = 0
    hysteresis_rank_band: int = 0

    def describe(self) -> dict[str, Any]:
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
        data: dict[str, pd.DataFrame],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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

        # YAML / JSON only carry lists, not tuples. Normalise tuple-typed knobs.
        if isinstance(self.output_track, (list, tuple)) and len(self.output_track) == 2:
            self.output_track = (str(self.output_track[0]), int(self.output_track[1]))
        if isinstance(self.clip_vol_scaler, list) and len(self.clip_vol_scaler) == 2:
            self.clip_vol_scaler = (float(self.clip_vol_scaler[0]), float(self.clip_vol_scaler[1]))

        prices = data["prices"]
        volume = data["volume"]
        market_cap = data["market_cap"]
        screen_volume = data.get("screen_volume", pd.DataFrame())
        screen_volume = screen_volume if not screen_volume.empty else None

        logger.info(f"Running CryptoTrendStrategy on {len(prices.columns)} tickers, {len(prices)} days")

        # 1. Universe selection. The DuckDB path does not yet support the
        # best-practice knobs (volume_is_dollar, rolling-window, listing
        # cool-off, hysteresis) or the market-wide screen_volume override, so
        # fall back to the vectorized impl whenever any of them is in effect.
        needs_vectorized = (
            self.volume_is_dollar
            or self.volume_rolling_window > 1
            or self.min_listing_days > 0
            or self.hysteresis_rank_band > 0
            or screen_volume is not None
        )
        if self.use_duckdb and DUCKDB_AVAILABLE and not needs_vectorized:
            universe = select_universe_duckdb(
                prices, volume, market_cap, self.top_by_mcap, self.top_by_volume, self.exclude_tickers
            )
        else:
            universe = select_universe_vectorized(
                prices,
                volume,
                market_cap,
                top_by_mcap=self.top_by_mcap,
                top_by_volume=self.top_by_volume,
                exclude_tickers=self.exclude_tickers,
                volume_is_dollar=self.volume_is_dollar,
                volume_rolling_window=self.volume_rolling_window,
                min_listing_days=self.min_listing_days,
                hysteresis_rank_band=self.hysteresis_rank_band,
                screen_volume=screen_volume,
            )

        # 2. Signal generation
        signals = generate_ensemble_signals(
            prices,
            self.lookback_windows,
            use_trailing_stop=self.use_trailing_stop,
        )

        # 3. Volatility scalers
        numeric_targets = [vt for vt in self.vol_targets if isinstance(vt, (int, float))]
        has_off = any(vt == "off" for vt in self.vol_targets if isinstance(vt, str))

        scalers: dict[str, pd.DataFrame] = {}
        if numeric_targets:
            scalers = compute_volatility_scalers(
                prices,
                numeric_targets,
                self.vol_lookback,
                clip_range=self.clip_vol_scaler,
            )
        if has_off:
            scalers["off"] = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)

        # 4. Portfolio construction
        weights = construct_weights(
            signals,
            universe,
            scalers,
            self.tranches,
            self.normalize_weights,
        )

        # 4b. Optional inverse-vol track (notebook cells 678–686).
        if self.inv_vol_track:
            if not has_off:
                raise ValueError(
                    "inv_vol_track=True requires 'off' in vol_targets — the "
                    "inv_vol track is derived from the off-track weights."
                )
            # Build the inv_vol track from the pre-tranche off-weights so the
            # inverse-vol normalisation matches the notebook (cells 678–686):
            # inv_vol is applied to the un-tranched off-weights, then tranched.
            off_untranched = (signals * scalers["off"]) * universe
            if self.normalize_weights:
                n_positions = universe.sum(axis=1).replace(0, np.nan)
                off_untranched = off_untranched.div(n_positions, axis=0).fillna(0.0)
            iv_track = compute_inv_vol_track(
                off_untranched,
                prices,
                self.tranches,
                vol_lookback=self.vol_lookback,
            )
            weights = pd.concat([weights, iv_track], axis=1).sort_index(axis=1, level="ticker")

        # 5. Pick the output track for the pipeline. The strategy's full
        # multi-index panel (every vol_target × tranches × ticker) is preserved
        # under details["weights_panel"]; the pipeline only consumes one slice
        # so we don't accidentally stack tracks via sum-across-levels.
        vt_key, t_key = self.output_track
        try:
            slice_df = get_simple_weights(weights, str(vt_key), int(t_key))
        except KeyError as exc:
            raise ValueError(
                f"output_track={self.output_track!r} not present in weights panel — "
                f"available vol_targets={sorted(set(weights.columns.get_level_values('vol_target')))}, "
                f"tranches={sorted(set(weights.columns.get_level_values('tranches')))}"
            ) from exc

        latest = slice_df.iloc[-1].dropna()
        latest = latest[latest > 0.001].to_dict()

        # 6. Diagnostics emit — used by reporter to render notebook charts.
        diagnostics = self._build_diagnostics(
            prices=prices,
            signals=signals,
            universe=universe,
            scalers=scalers,
        )

        # Pipeline-facing weights = single-level DataFrame at the chosen track.
        # Full multi-index panel is preserved for research consumers via details.
        output = slice_df.tail(self.output_periods)
        return {
            "weights": output,
            "simple_weights": latest,
            "details": {
                "signals": signals,
                "universe": universe,
                "scalers": scalers,
                "diagnostics": diagnostics,
                "weights_panel": weights,
                "output_track": tuple(self.output_track),
            },
        }

    def _build_diagnostics(
        self,
        *,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        universe: pd.DataFrame,
        scalers: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        """Build reporter-facing diagnostics payload.

        Emits the contract documented in ``docs/plans/crypto-trend-ssrn-finalize.md``:

        - ``donchian_overlay``: price + bands + trailing stop for ``regime_ticker``
          across the headline windows.
        - ``signal_count``: per-day active-signal count across the top-N universe.
        - ``per_window_signal``: per-window binary signal for the regime ticker
          (feeds the per-window stats heatmap in the reporter).
        - ``vol_overview``: realized vol time series + scalers (used for charts 12,
          13, 18, 19).
        - ``regime_overlay``: stub-compatible with the existing reporter builder so
          the report renders a price chart even when no diagnostics-specific
          overlay is desired.
        """
        diagnostics: dict[str, Any] = {}

        # Resolve regime ticker (may be quoted as BTCUSDT etc.). ``None`` disables
        # the regime/donchian-overlay diagnostic entirely (diagnostic-only — no
        # weight impact; used for literal 1:1 ports vs quantlab's catcher).
        regime_col = (
            None
            if self.regime_ticker is None
            else next(
                (c for c in prices.columns if c in (self.regime_ticker, self.regime_ticker + "USDT")),
                None,
            )
        )

        # Active-signal count across the top-N universe.
        active = (signals * universe > 0).astype(int)
        diagnostics["signal_count"] = {
            "series": active.sum(axis=1).astype(int),
            "cap": int(self.top_by_volume),
            "label": "Active Donchian signals (top-N universe)",
        }

        # Vol overview.
        returns = prices.ffill().pct_change(fill_method=None)
        realized_vol = returns.rolling(self.vol_lookback).std() * np.sqrt(365.0)
        diagnostics["vol_overview"] = {
            "realized_vol": realized_vol,
            "scalers": scalers,
            "vol_lookback": int(self.vol_lookback),
        }

        if regime_col is None:
            return diagnostics

        # Donchian overlay (headline window) for the regime ticker.
        headline_w = self._select_headline_window()
        ref_prices = prices[regime_col]
        high = ref_prices.rolling(headline_w, min_periods=headline_w).max()
        low = ref_prices.rolling(headline_w, min_periods=headline_w).min()
        mid = 0.5 * (high + low)

        # Recompute trailing stop for the headline window for visualization.
        signal_arr = np.zeros(len(ref_prices))
        trailing_stop = np.full(len(ref_prices), np.nan)
        prices_arr = ref_prices.values
        high_arr = high.values
        mid_arr = mid.values
        for i in range(headline_w - 1, len(ref_prices)):
            if i == headline_w - 1:
                signal_arr[i] = 1.0 if prices_arr[i] >= high_arr[i] else 0.0
                trailing_stop[i] = mid_arr[i]
            else:
                if signal_arr[i - 1] == 1:
                    trailing_stop[i] = max(trailing_stop[i - 1], mid_arr[i])
                    signal_arr[i] = 1.0 if prices_arr[i] >= trailing_stop[i] else 0.0
                else:
                    signal_arr[i] = 1.0 if prices_arr[i] >= high_arr[i] else 0.0
                    trailing_stop[i] = mid_arr[i]
        ts_series = pd.Series(trailing_stop, index=ref_prices.index, name="trailing_stop")
        sig_series = pd.Series(signal_arr, index=ref_prices.index, name="signal")

        diagnostics["donchian_overlay"] = {
            "ref_ticker": str(self.regime_ticker),
            "window": int(headline_w),
            "price": ref_prices,
            "high": high,
            "low": low,
            "mid": mid,
            "trailing_stop": ts_series,
            "signal": sig_series,
        }

        # Per-window signal map for the regime ticker (drives per-window heatmaps).
        per_window: dict[int, pd.Series] = {}
        for w in self.lookback_windows:
            per_window[int(w)] = compute_donchian_breakout_vectorized(ref_prices, int(w))
        diagnostics["per_window_signal"] = {
            "ref_ticker": str(self.regime_ticker),
            "signals": per_window,
        }

        # Stub regime_overlay (reuses existing reporter builder shape).
        diagnostics["regime_overlay"] = {
            "ref_ticker": str(self.regime_ticker),
            "fast_window": int(headline_w),
            "slow_window": int(headline_w),
            "price": ref_prices,
        }

        return diagnostics

    def _select_headline_window(self) -> int:
        """Pick the headline Donchian window for the donchian_overlay diagnostic.

        Convention: prefer ``60`` (notebook default for single-asset analysis),
        fall back to the median configured window.
        """
        if not self.lookback_windows:
            return 60
        if 60 in self.lookback_windows:
            return 60
        return int(sorted(self.lookback_windows)[len(self.lookback_windows) // 2])

    def get_latest_weights(
        self,
        result: dict[str, Any],
        vol_target: str = "50",
        tranche: int = 5,
    ) -> dict[str, float]:
        """
        Extract latest weights as a simple dict.

        Args:
            result: Output from run()
            vol_target: Volatility target to use
            tranche: Tranche parameter

        Returns:
            Dict of ticker -> weight
        """
        weights = result["weights"]
        simple = get_simple_weights(weights, vol_target, tranche)
        latest = simple.iloc[-1].dropna()
        return latest[latest > 0.001].to_dict()

    def backtest(
        self,
        data: dict[str, pd.DataFrame],
        initial_capital: float = 10000,
        commission_pct: float = 0.001,
    ) -> dict[str, Any]:
        """
        Run strategy with performance metrics.

        Requires quantstats for performance analysis.

        Returns:
            Dict with weights, returns, and performance metrics
        """
        result = self.run(data)

        # Compute strategy returns
        prices = data["prices"]
        weights = get_simple_weights(result["weights"], "50", 5)

        # Align weights and prices
        weights = weights.reindex(columns=prices.columns, fill_value=0)

        # Asset returns
        returns = prices.ffill().pct_change(fill_method=None)

        # Strategy returns (weighted average)
        strategy_returns = (returns * weights.shift(1)).sum(axis=1)
        strategy_returns = strategy_returns.dropna()

        # Performance metrics
        metrics = {}
        if QUANTSTATS_AVAILABLE:
            metrics = {
                "total_return": qs.stats.comp(strategy_returns),
                "cagr": qs.stats.cagr(strategy_returns),
                "sharpe": qs.stats.sharpe(strategy_returns),
                "sortino": qs.stats.sortino(strategy_returns),
                "max_drawdown": qs.stats.max_drawdown(strategy_returns),
                "volatility": qs.stats.volatility(strategy_returns),
                "calmar": qs.stats.calmar(strategy_returns),
            }

        return {
            **result,
            "returns": strategy_returns,
            "metrics": metrics,
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
