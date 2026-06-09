"""Conditional volatility targeting strategy.

Scales asset exposure based on rolling volatility quantiles: increases
exposure when realized vol is low (below 20th percentile), decreases
when high (above 80th percentile), holds neutral otherwise.

Generic implementation — works for any asset class by configuring the
lookback windows and vol quantile thresholds.

Config example::

    strategy = VolTargetingStrategy()
    result = strategy.run(
        data={"prices": prices_df},
        params={
            "risk_off_ticker": "t-bills",
            "exposure": {"min": 0.0, "max": 1.2, "neutral": 1.0},
            "lookbacks": [252],
            "quantile_low": 0.2,
            "quantile_high": 0.8,
        },
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantbox.contracts import PluginMeta
from quantbox.features.volatility import compute_rolling_vol


@dataclass
class VolTargetingStrategy:
    """Volatility-regime conditional exposure scaling.

    For each ticker, computes rolling volatility, then checks whether
    the current vol is in an extreme quantile (high or low). In extreme
    regimes, scales exposure by the ratio of lagged vol to current vol.
    In normal regimes, holds neutral exposure.
    """

    meta: PluginMeta = PluginMeta(
        name="strategy.vol_targeting.v1",
        kind="strategy",
        version="1.0.0",
        core_compat=">=0.1,<1.0",
        description=(
            "Conditional volatility targeting: scales exposure based on "
            "rolling volatility quantiles. Increases in low-vol, decreases "
            "in high-vol regimes."
        ),
        tags=("volatility", "targeting", "tactical"),
        capabilities=("backtest", "live"),
        inputs=("prices",),
        outputs=("weights",),
    )

    lookbacks: list[int] = field(default_factory=lambda: [252])
    exposure_min: float = 0.0
    exposure_max: float = 1.2
    exposure_neutral: float = 1.0
    quantile_low: float = 0.2
    quantile_high: float = 0.8
    vol_lag: int = 21

    def run(
        self,
        data: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute vol-targeting weights.

        Args:
            data: Must contain ``"prices"`` (wide DataFrame).
            params:
                - ``risk_off_ticker``: str or dict — risk-off per ticker
                - ``exposure``: dict with ``min``, ``max``, ``neutral``
                - ``lookbacks``: list[int] — vol lookback windows
                - ``quantile_low``: float — low vol threshold (default 0.2)
                - ``quantile_high``: float — high vol threshold (default 0.8)
                - ``vol_lag``: int — lag for vol ratio (default 21)
                - ``trading_days``: int — annualization factor (default 252)

        Returns:
            Dict with ``"weights"`` (DataFrame: date x ticker).
        """
        params = params or {}
        prices = data["prices"]

        risk_off = params.get("risk_off_ticker")
        exposure = params.get("exposure", {})
        exp_min = exposure.get("min", self.exposure_min)
        exp_max = exposure.get("max", self.exposure_max)
        exp_neutral = exposure.get("neutral", self.exposure_neutral)
        lookbacks = params.get("lookbacks", self.lookbacks)
        q_low = params.get("quantile_low", self.quantile_low)
        q_high = params.get("quantile_high", self.quantile_high)
        vol_lag = params.get("vol_lag", self.vol_lag)
        trading_days = params.get("trading_days", 252)

        # Normalize risk_off_ticker
        tickers = [c for c in prices.columns if c != risk_off]
        if risk_off is None:
            risk_off_map = {t: None for t in tickers}
        elif isinstance(risk_off, str):
            risk_off_map = {t: risk_off for t in tickers}
        elif isinstance(risk_off, dict):
            default = risk_off.get("default")
            risk_off_map = {t: risk_off.get(t, default) for t in tickers}
        else:
            risk_off_map = {t: None for t in tickers}

        # Compute rolling vol for each lookback, then average
        all_vol = compute_rolling_vol(
            prices[tickers],
            lookbacks,
            annualize=True,
            factor=float(trading_days),
        )
        # Average across lookback windows
        vol = pd.concat(all_vol.values(), axis=1).T.groupby(level=0).mean().T
        if vol.columns.nlevels > 1:
            vol.columns = vol.columns.droplevel(0)

        weights_parts: dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            if ticker not in vol.columns:
                continue
            vol_t = vol[[ticker]]

            raw_parts = []
            for lookback in lookbacks:
                vol_q_low = vol_t.rolling(window=lookback).quantile(q_low)
                vol_q_high = vol_t.rolling(window=lookback).quantile(q_high)
                vol_prev = vol_t.shift(vol_lag)
                vol_ratio = (vol_prev / vol_t).dropna()

                # Extreme flag: vol outside [q_low, q_high] band
                extreme = ~vol_t.apply(
                    lambda x: x.between(
                        vol_q_low[x.name],
                        vol_q_high[x.name],
                    )
                ).reindex(vol_ratio.index)

                raw_parts.append(vol_ratio.where(extreme).fillna(exp_neutral))

            raw = pd.concat(raw_parts, axis=1)
            w = raw.mean(axis=1).rename(ticker).to_frame().clip(exp_min, exp_max)

            ro = risk_off_map.get(ticker)
            if ro:
                w[ro] = exp_max - w[ticker]

            weights_parts[ticker] = w

        # Combine
        if weights_parts:
            weights = pd.concat(weights_parts.values(), axis=1)
            if weights.columns.duplicated().any():
                weights = weights.T.groupby(level=0).mean().T
            weights = weights.dropna(how="all")
        else:
            weights = pd.DataFrame()

        return {"weights": weights}
