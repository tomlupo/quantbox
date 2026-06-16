"""strategy.carver_trend_proper.v1 — pysystemtrade-faithful Carver trend for crypto perps.

This variant closes the fidelity gaps between ``strategy.carver_trend.v1`` and the
canonical Rob Carver framework as implemented in pysystemtrade
(``/home/tom/workspace/projects/pysystemtrade``).  Each addition cites the
canonical source it reproduces:

  * **Full EWMAC family** — canonical six speeds 2/8 … 64/256
    (``systems/provided/rules/ewmac.py`` + ``futures_chapter15`` config).  The
    crypto default drops the 2/8 pair (speed-limit / cost reasoning, see below).
  * **Causal forecast scaling to avg-abs 10** — expanding mean-abs, not the
    rolling-252 used by v1 (``sysquant/estimators/forecast_scalar.py``,
    ``target_abs_forecast = 10``).
  * **Forecast Diversification Multiplier (FDM)** — ABSENT in v1.  Estimated from
    the pooled correlation of the rule forecasts and the forecast weights
    (``systems/forecast_combine.py`` ``get_forecast_diversification_multiplier``;
    ``sysquant.estimators.diversification_multipliers``).
  * **forecast / 10 position normalization** — v1 divides the forecast by the ±20
    *cap*, which halves every position.  Canonical divides by the average-absolute
    forecast (10): ``systems/positionsizing.py``
    ``subsystem_position = vol_scalar * forecast / avg_abs_forecast``.
  * **Carver mixed volatility estimator** — short EWMA blended with a long-run
    average plus a floor, annualised by 365 for 24/7 crypto
    (``sysquant/estimators/vol.py`` ``mixed_vol_calc``: days=35,
    proportion_of_slow_vol=0.3).
  * **Correlation-based IDM** — v1 uses ``min(sqrt(N), 2.5)`` which over-states the
    multiplier for highly-correlated crypto.  Canonical estimates it from the
    instrument correlation matrix (``systems/portfolio.py``).
  * **Forecast-method buffering / trade-to-edge** — v1 has no no-trade band (only a
    downstream min-notional).  Canonical buffers around the optimal position
    (``systems/buffering.py``: buffer = average_position * buffer_size,
    buffer_size = 0.10, trade to edge).

The output contract matches v1 (a ``weights`` time-series DataFrame plus
``simple_weights`` / ``forecasts`` / diagnostics) so it plugs into the same
``run_vectorbt`` backtest path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta
from quantbox.plugins.strategies.carver_trend import (
    breakout_forecast,
    cap_forecast,
    ewmac_forecast,
)

logger = logging.getLogger(__name__)

# Canonical Carver EWMAC speeds (Lfast, Lslow), futures_chapter15.
CANONICAL_EWMAC_SPANS: list[tuple[int, int]] = [
    (2, 8),
    (4, 16),
    (8, 32),
    (16, 64),
    (32, 128),
    (64, 256),
]
# Crypto default: drop the 2/8 pair.  Carver's "speed limit" rule says a rule may
# only be traded if its pre-cost SR benefit exceeds ~2x its SR trading cost; the
# fastest crossover is the highest-turnover and first to breach that ceiling, and
# crypto perp round-trip costs (taker + funding) are materially higher than the
# liquid futures Carver calibrated on.
CRYPTO_EWMAC_SPANS: list[tuple[int, int]] = [
    (4, 16),
    (8, 32),
    (16, 64),
    (32, 128),
    (64, 256),
]


# ---------------------------------------------------------------------------
# Forecast scaling (causal, expanding) — canonical target avg-abs = 10
# ---------------------------------------------------------------------------
def scale_forecast_expanding(
    forecast: pd.Series,
    target_abs_avg: float = 10.0,
    min_periods: int = 100,
) -> pd.Series:
    """Scale a raw forecast so its average absolute value targets ``target_abs_avg``.

    Canonical pysystemtrade uses ``window=250000`` (effectively all history) with
    ``backfill=True`` in ``forecast_scalar``.  We use a strictly-causal expanding
    mean-abs (no backfill) so there is no look-ahead — the only cost is a slightly
    noisier scalar during the warm-up, which the ``min_periods`` guard absorbs.
    v1 by contrast uses a rolling-252 window, which makes the scalar drift with
    recent forecast magnitude rather than converging to a stable constant.
    """
    abs_avg = forecast.abs().expanding(min_periods=min_periods).mean()
    scale = target_abs_avg / abs_avg.replace(0, np.nan)
    return forecast * scale


# ---------------------------------------------------------------------------
# Carver mixed volatility estimator (sysquant/estimators/vol.py: mixed_vol_calc)
# ---------------------------------------------------------------------------
def mixed_volatility(
    prices: pd.DataFrame,
    fast_span: int = 32,
    slow_years: float = 1.0,
    proportion_slow: float = 0.3,
    annualize: float = 365.0,
    floor_window: int = 365,
    floor_quantile: float = 0.05,
    floor_min_periods: int = 90,
) -> pd.DataFrame:
    """Annualised volatility, faithful to Carver's ``mixed_vol_calc``.

    vol = (1 - p) * EWMA(fast) + p * long_run_average(EWMA(fast)), then floored.

    Canonical defaults: days=35, slow_vol_years=10, proportion_of_slow_vol=0.3,
    with a vol floor (``apply_vol_floor``: 5th percentile over a rolling window).
    Crypto adaptations vs the canonical futures defaults:
      * ``annualize=365`` (24/7 markets) rather than 256 business days.
      * ``slow_years=1.0`` rather than 10 — crypto has < a decade of history and
        vol regimes turn over far faster, so the long-run anchor uses a 1-year
        blend instead of a (data-exceeding) 10-year one.
    """
    returns = prices.ffill().pct_change(fill_method=None)
    fast_vol = returns.ewm(span=fast_span, min_periods=10).std()
    slow_span = max(int(slow_years * annualize), fast_span)
    long_run = fast_vol.ewm(span=slow_span, min_periods=10).mean()
    vol = (1.0 - proportion_slow) * fast_vol + proportion_slow * long_run
    # Vol floor: never let an instrument's vol estimate collapse below its own
    # recent 5th-percentile (canonical apply_vol_floor).
    floor = vol.rolling(floor_window, min_periods=floor_min_periods).quantile(floor_quantile)
    vol = vol.where(vol >= floor, floor)
    return vol * np.sqrt(annualize)


# ---------------------------------------------------------------------------
# Diversification multipliers (FDM and IDM share the same maths)
# ---------------------------------------------------------------------------
def _diversification_multiplier(corr: np.ndarray, weights: np.ndarray, cap: float) -> float:
    """DM = 1 / sqrt(w' C w), clipped to [1.0, cap].

    Canonical: ``sysquant.estimators.diversification_multipliers``.  A DM of 1.0
    means perfectly correlated (no diversification benefit); higher means more
    independent signals/instruments to lever up into the vol target.
    """
    w = np.asarray(weights, dtype=float)
    if w.sum() != 0:
        w = w / w.sum()
    c = np.asarray(corr, dtype=float)
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 1.0)
    variance = float(w @ c @ w)
    if variance <= 0:
        return 1.0
    dm = 1.0 / np.sqrt(variance)
    return float(np.clip(dm, 1.0, cap))


def forecast_diversification_multiplier(
    rule_forecasts: list[pd.Series],
    weights: list[float],
    cap: float = 2.5,
) -> float:
    """FDM from the pooled correlation of the (scaled) rule forecasts.

    Pooling across instruments (stack every instrument's rule forecasts) gives a
    single robust FDM, which is how Carver pools estimation for a homogeneous
    universe.  v1 has no FDM, so its combined forecast — a weighted sum of
    positively-correlated rules — has an average-absolute value well below 10,
    under-sizing every position.
    """
    df = pd.concat(rule_forecasts, axis=1).dropna()
    if df.shape[0] < 50 or df.shape[1] < 2:
        return 1.0
    corr = df.corr().to_numpy()
    return _diversification_multiplier(corr, np.asarray(weights), cap)


def instrument_diversification_multiplier(
    returns: pd.DataFrame,
    cap: float = 2.5,
) -> float:
    """IDM from the instrument return correlation matrix, equal weights.

    Canonical ``systems/portfolio.py`` estimates the IDM from the instrument
    correlation matrix and instrument weights.  For highly-correlated crypto the
    estimate lands ~1.1–1.5 — far below v1's ``min(sqrt(N), 2.5)`` (e.g. 2.24 for
    N=5), which assumes near-independence the crypto cross-section does not have.
    """
    clean = returns.dropna(how="all", axis=1).dropna()
    n = clean.shape[1]
    if n < 2 or clean.shape[0] < 50:
        return 1.0
    corr = clean.corr().to_numpy()
    weights = np.full(n, 1.0 / n)
    return _diversification_multiplier(corr, weights, cap)


# ---------------------------------------------------------------------------
# Forecast generation: scaled + capped rules → FDM → combined, capped
# ---------------------------------------------------------------------------
def generate_proper_forecast(
    prices: pd.Series,
    ewmac_spans: list[tuple[int, int]],
    breakout_windows: list[int] | None,
    ewmac_weight: float,
    breakout_weight: float,
    cap: float = 20.0,
) -> tuple[pd.Series, dict[str, pd.Series], list[float]]:
    """Return (combined_uncapped_pre_fdm, {rule_name: scaled_capped_forecast}, weights).

    The combined forecast is intentionally returned WITHOUT the FDM applied so the
    caller can estimate one pooled FDM across instruments, apply it, then cap.
    """
    rule_forecasts: dict[str, pd.Series] = {}
    weights: list[float] = []

    n_ewmac = len(ewmac_spans)
    for fast, slow in ewmac_spans:
        f = ewmac_forecast(prices, fast, slow)
        f = scale_forecast_expanding(f, target_abs_avg=10.0)
        f = cap_forecast(f, cap=cap)
        rule_forecasts[f"ewmac_{fast}_{slow}"] = f
        weights.append(ewmac_weight / n_ewmac)

    if breakout_windows and breakout_weight > 0:
        n_bo = len(breakout_windows)
        for window in breakout_windows:
            f = breakout_forecast(prices, window) * 20.0
            f = cap_forecast(f, cap=cap)
            rule_forecasts[f"breakout_{window}"] = f
            weights.append(breakout_weight / n_bo)

    # combine: weighted sum (FDM applied by the caller, pooled across instruments)
    combined: pd.Series | None = None
    for w, f in zip(weights, rule_forecasts.values(), strict=False):
        combined = (w * f) if combined is None else (combined + w * f)
    return combined, rule_forecasts, weights


# ---------------------------------------------------------------------------
# Forecast-method buffering (systems/buffering.py)
# ---------------------------------------------------------------------------
def apply_carver_buffer(
    optimal: pd.DataFrame,
    avg_position: pd.DataFrame,
    buffer_size: float = 0.10,
) -> pd.DataFrame:
    """Trade-to-edge-of-buffer on a weights time-series.

    buffer = |average_position| * buffer_size; the held position only moves when
    the optimal position leaves [held - buffer, held + buffer], and then only to
    the near edge (``apply_buffers_to_position`` + ``apply_buffer`` in
    pysystemtrade).  This is the canonical no-trade band that cuts turnover; v1
    has nothing equivalent at the position level.
    """
    opt = optimal.fillna(0.0).to_numpy()
    buf = (avg_position.abs() * buffer_size).reindex_like(optimal).fillna(0.0).to_numpy()
    held = np.zeros_like(opt)
    prev = np.zeros(opt.shape[1])
    for t in range(opt.shape[0]):
        top = opt[t] + buf[t]
        bot = opt[t] - buf[t]
        cur = prev.copy()
        # below band → move up to bottom edge; above band → move down to top edge
        cur = np.where(prev < bot, bot, cur)
        cur = np.where(prev > top, top, cur)
        held[t] = cur
        prev = cur
    return pd.DataFrame(held, index=optimal.index, columns=optimal.columns)


class CarverTrendProperStrategy:
    """pysystemtrade-faithful Carver trend, adapted for crypto perps."""

    meta = PluginMeta(
        name="strategy.carver_trend_proper.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        status="research",
        description=(
            "pysystemtrade-faithful Carver trend: full EWMAC family, FDM, "
            "forecast/10 sizing, mixed vol estimator, correlation IDM, buffering."
        ),
        tags=("crypto", "trend", "carver", "pysystemtrade"),
        capabilities=("backtest", "paper"),
        outputs=("strategy_weights",),
        params_schema={
            "type": "object",
            "properties": {
                "target_vol": {"type": "number", "minimum": 0.01, "maximum": 2.0, "default": 0.25},
                "ewmac_spans": {"type": ["array", "null"], "default": None},
                "breakout_windows": {"type": ["array", "null"], "default": None},
                "ewmac_weight": {"type": "number", "default": 1.0},
                "breakout_weight": {"type": "number", "default": 0.0},
                "vol_fast_span": {"type": "integer", "default": 32},
                "vol_slow_years": {"type": "number", "default": 1.0},
                "annualize": {"type": "number", "default": 365.0},
                "fdm": {"type": ["number", "string", "null"], "default": "auto"},
                "idm": {"type": ["number", "string", "null"], "default": "auto"},
                "fdm_cap": {"type": "number", "default": 2.5},
                "idm_cap": {"type": "number", "default": 2.5},
                "cross_sectional": {"type": "boolean", "default": False},
                "buffer_size": {"type": "number", "minimum": 0.0, "default": 0.10},
                "max_position": {"type": "number", "default": 1.0},
                "max_gross": {"type": "number", "default": 4.0},
                "allow_shorts": {"type": "boolean", "default": True},
                "forecast_cap": {"type": "number", "default": 20.0},
                "use_universe_selection": {"type": "boolean", "default": False},
                "top_by_mcap": {"type": "integer", "default": 30},
                "top_by_volume": {"type": "integer", "default": 10},
                "volume_is_dollar": {"type": "boolean", "default": True},
                "output_periods": {"type": "integer", "default": 30},
                "fine_lot_guard": {"type": "boolean", "default": False},
                "fine_lot_min_notional": {"type": "number", "default": 10.0},
                "fine_lot_max_lot_fraction": {"type": "number", "default": 0.2},
                "sz_decimals": {"type": ["object", "string", "null"], "default": None},
            },
        },
    )

    # defaults (overridable via params)
    target_vol: float = 0.25
    ewmac_spans: list[tuple[int, int]] | None = None
    breakout_windows: list[int] | None = None
    ewmac_weight: float = 1.0
    breakout_weight: float = 0.0
    vol_fast_span: int = 32
    vol_slow_years: float = 1.0
    annualize: float = 365.0
    fdm: float | str | None = "auto"
    idm: float | str | None = "auto"
    fdm_cap: float = 2.5
    idm_cap: float = 2.5
    cross_sectional: bool = False
    buffer_size: float = 0.10
    max_position: float = 1.0
    max_gross: float = 4.0
    allow_shorts: bool = True
    forecast_cap: float = 20.0
    use_universe_selection: bool = False
    top_by_mcap: int = 30
    top_by_volume: int = 10
    volume_is_dollar: bool = True
    output_periods: int = 30
    exclude_tickers: tuple[str, ...] = ()
    # Small-book fine-lot guardrail (see _universe.select_universe). Off by
    # default; the dynamic-universe config turns it on so the $1,500 book never
    # selects a coin that is not fine-lot tradeable at that size.
    fine_lot_guard: bool = False
    fine_lot_min_notional: float = 10.0
    fine_lot_max_lot_fraction: float = 0.2
    sz_decimals: dict[str, int] | str | None = None

    def _resolve_sz_decimals(self) -> dict[str, int] | None:
        """Return the szDecimals (lot precision) map for the fine-lot guard.

        ``sz_decimals`` may be an inline ``{coin: szDecimals}`` dict or a path to
        a JSON file holding that mapping (e.g. a snapshot of the Hyperliquid
        ``metaAndAssetCtxs`` universe). Returns ``None`` when unset, which makes
        the guard fail closed (every coin excluded) — surfacing a mis-wired
        guard loudly rather than silently trading coarse-lot coins.
        """
        src = self.sz_decimals
        if src is None:
            logger.warning("fine_lot_guard is on but sz_decimals is unset; the guard will exclude all coins")
            return None
        if isinstance(src, str):
            data = json.loads(Path(src).read_text())
        else:
            data = dict(src)
        return {str(k): int(v) for k, v in data.items()}

    def run(self, data: dict[str, pd.DataFrame], params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        spans = self.ewmac_spans or CRYPTO_EWMAC_SPANS
        spans = [tuple(s) for s in spans]
        prices = data["prices"]
        valid = [t for t in prices.columns if t not in self.exclude_tickers]
        prices = prices[valid]

        # 0. Time-varying universe mask (no look-ahead) — same machinery as v1.
        universe_mask_ts: pd.DataFrame | None = None
        if self.use_universe_selection:
            from quantbox.plugins.strategies._universe import select_universe

            volume = data.get("volume", pd.DataFrame())
            market_cap = data.get("market_cap", pd.DataFrame())
            sz_map = self._resolve_sz_decimals() if self.fine_lot_guard else None
            universe_mask_ts = select_universe(
                prices,
                volume.reindex(index=prices.index, columns=prices.columns).fillna(0.0),
                market_cap if not market_cap.empty else None,
                self.top_by_mcap,
                self.top_by_volume,
                list(self.exclude_tickers),
                volume_is_dollar=self.volume_is_dollar,
                fine_lot_sz_decimals=sz_map,
                fine_lot_min_notional=(self.fine_lot_min_notional if self.fine_lot_guard else 0.0),
                fine_lot_max_lot_fraction=(self.fine_lot_max_lot_fraction if self.fine_lot_guard else 0.0),
            )

        # 1. Per-instrument scaled+capped rule forecasts (pre-FDM combine).
        combined_pre: dict[str, pd.Series] = {}
        pooled_rule_forecasts: dict[str, list[pd.Series]] = {}
        weights_ref: list[float] = []
        for ticker in prices.columns:
            comb, rule_fc, weights_ref = generate_proper_forecast(
                prices[ticker],
                ewmac_spans=spans,
                breakout_windows=self.breakout_windows,
                ewmac_weight=self.ewmac_weight,
                breakout_weight=self.breakout_weight,
                cap=self.forecast_cap,
            )
            combined_pre[ticker] = comb
            for name, fc in rule_fc.items():
                pooled_rule_forecasts.setdefault(name, []).append(fc)

        # 2. FDM from pooled rule-forecast correlation, then apply + cap.
        if isinstance(self.fdm, str) and self.fdm == "auto":
            pooled = [pd.concat(v, axis=0) for v in pooled_rule_forecasts.values()]
            fdm_val = forecast_diversification_multiplier(pooled, weights_ref, cap=self.fdm_cap)
        else:
            fdm_val = float(self.fdm) if self.fdm is not None else 1.0

        forecasts_df = pd.DataFrame(combined_pre)
        forecasts_df = (forecasts_df * fdm_val).clip(-self.forecast_cap, self.forecast_cap)

        # 2b. Optional cross-sectional demeaning (relative / market-neutral trend).
        #     Subtract the per-date cross-sectional mean forecast to strip the common
        #     "everything trends with BTC together" factor — the dominant Sharpe drag
        #     in a one-beta universe.  This turns the book into relative-strength trend
        #     (roughly dollar-neutral).  Carver-adjacent (relative momentum); off by
        #     default so the canonical directional system is the baseline.
        if self.cross_sectional:
            xs_mean = forecasts_df.mean(axis=1)
            forecasts_df = forecasts_df.sub(xs_mean, axis=0).clip(-self.forecast_cap, self.forecast_cap)

        # 3. Volatility (Carver mixed estimator, 365d annualisation).
        vol = mixed_volatility(
            prices,
            fast_span=self.vol_fast_span,
            slow_years=self.vol_slow_years,
            annualize=self.annualize,
        )

        # 4. IDM from instrument return correlation.
        if isinstance(self.idm, str) and self.idm == "auto":
            rets = prices.pct_change(fill_method=None)
            idm_val = instrument_diversification_multiplier(rets, cap=self.idm_cap)
        else:
            idm_val = float(self.idm) if self.idm is not None else 1.0

        # 5. Position sizing — Carver: weight = (forecast/10) * (target_vol/vol)
        #    * (1/N) * IDM.   N is the active universe count per date.
        if universe_mask_ts is not None:
            mask = universe_mask_ts.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
            n_active = mask.sum(axis=1).replace(0, np.nan)
        else:
            mask = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
            n_active = pd.Series(float(len(prices.columns)), index=prices.index)

        vol_scalar = self.target_vol / vol.replace(0, np.nan)
        # average position (|forecast|=10 → forecast/10 = 1) used for sizing & buffer width
        avg_position = vol_scalar.mul(1.0 / n_active, axis=0) * idm_val
        optimal = (forecasts_df / 10.0) * avg_position
        optimal = optimal * mask

        if not self.allow_shorts:
            optimal = optimal.clip(lower=0)
        optimal = optimal.clip(-self.max_position, self.max_position)

        # gross cap (scale the whole row down if over budget)
        gross = optimal.abs().sum(axis=1)
        scale = (self.max_gross / gross).clip(upper=1.0).fillna(1.0)
        optimal = optimal.mul(scale, axis=0)

        # 6. Buffering / no-trade band → held positions.
        if self.buffer_size and self.buffer_size > 0:
            held = apply_carver_buffer(optimal, (avg_position * mask), buffer_size=self.buffer_size)
        else:
            held = optimal

        latest = held.iloc[-1].dropna()
        simple = latest[abs(latest) > 1e-4].to_dict()
        long_exp = float(latest[latest > 0].sum())
        short_exp = float(abs(latest[latest < 0].sum()))

        return {
            "weights": held.tail(self.output_periods),
            "weights_optimal": optimal.tail(self.output_periods),
            "simple_weights": simple,
            "forecasts": forecasts_df.tail(self.output_periods),
            "details": {
                "fdm": fdm_val,
                "idm": idm_val,
                "volatilities": vol,
                "avg_position": avg_position,
                "full_weights": held,
                "full_optimal": optimal,
                "full_forecasts": forecasts_df,
            },
            "exposure": {
                "long": long_exp,
                "short": short_exp,
                "net": long_exp - short_exp,
                "gross": long_exp + short_exp,
            },
        }


def run(data: dict, params: dict | None = None) -> dict:
    """Standard strategy interface."""
    return CarverTrendProperStrategy().run(data, params)
