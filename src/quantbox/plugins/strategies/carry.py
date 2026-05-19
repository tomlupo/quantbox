"""Funding-rate carry strategy — Mode A (funding_momentum).

Trades directionally with the funding-rate signal:
  - Long top-N assets with the highest (most positive) 3-day EMA annualized funding
  - Short top-N assets with the lowest (most negative) 3-day EMA annualized funding

Spec: /srv/obsidian/notebook/projects/quantlab/strategy-carry-v1-spec.md

Phase 2 ships Mode A only.  Mode B (basis_carry / delta-neutral) is deferred
to Phase 2.5 pending spot-venue infrastructure.
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


@dataclass
class CarryStrategy:
    """Funding-rate carry for Hyperliquid perpetual futures (Mode A).

    Ranks the universe by 3-day EMA of annualized daily funding rate, goes
    long top-N and short bottom-N at equal weight (0.5/N per leg, capped at
    max_concentration).  Positions with signal below min_signal_annualized
    (5 % annualized) are skipped.  The equal-weighted book is then scaled by
    a vol-targeting multiplier so realized portfolio vol ≈ target_vol.

    ## Quick Start
    ```python
    from quantbox.plugins.strategies import CarryStrategy

    strategy = CarryStrategy()
    result = strategy.run(data)   # data must contain 'funding_rates' and 'prices'
    print(result['simple_weights'])
    # e.g. {'BTC': 0.20, 'ETH': 0.20, 'SOL': -0.20, 'AVAX': -0.20}
    ```
    """

    meta = PluginMeta(
        name="strategy.carry.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        status="research",
        description=(
            "Funding-rate carry for Hyperliquid perps (Mode A: funding_momentum). "
            "Long top-N by highest 3-day EMA annualized funding, "
            "short top-N by lowest.  Vol-targeted at 20 % by default."
        ),
        tags=("crypto", "carry", "funding", "hyperliquid"),
        capabilities=("backtest", "paper", "live"),
        outputs=("strategy_weights",),
        params_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["funding_momentum"],
                    "default": "funding_momentum",
                    "description": "Strategy variant. Only Mode A ships in Phase 2.",
                },
                "signal_span_days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "default": 3,
                    "description": (
                        "EWM span (days) for smoothing daily funding rates. "
                        "At daily granularity, span=3 ≈ 9 observations at 8h cadence."
                    ),
                },
                "top_n_long": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                },
                "top_n_short": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                },
                "min_signal_annualized": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.05,
                    "description": (
                        "Minimum absolute annualized funding to enter a position. "
                        "Filters noise and covers transaction costs."
                    ),
                },
                "target_vol": {
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 1.0,
                    "default": 0.20,
                    "description": "Target annualized portfolio volatility.",
                },
                "vol_lookback": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 252,
                    "default": 20,
                    "description": "Rolling window (days) for realized vol estimation.",
                },
                "max_concentration": {
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 1.0,
                    "default": 0.20,
                    "description": "Maximum weight per single position (concentration cap).",
                },
                "output_periods": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 2000,
                    "default": 365,
                    "description": "Number of trailing days to include in returned weights DataFrame.",
                },
            },
        },
        examples=(
            (
                "plugins:\n"
                "  strategies:\n"
                "    - name: strategy.carry.v1\n"
                "      weight: 0.30\n"
                "      params:\n"
                "        mode: funding_momentum\n"
                "        signal_span_days: 3\n"
                "        top_n_long: 3\n"
                "        top_n_short: 3\n"
                "        min_signal_annualized: 0.05\n"
                "        target_vol: 0.20\n"
                "        max_concentration: 0.20"
            ),
        ),
    )

    # Signal
    mode: str = "funding_momentum"
    signal_span_days: int = 3

    # Portfolio construction
    top_n_long: int = 3
    top_n_short: int = 3
    min_signal_annualized: float = 0.05
    max_concentration: float = 0.20

    # Vol targeting
    target_vol: float = 0.20
    vol_lookback: int = 20
    annualize: float | None = None  # None = pipeline-injected via _pipeline_annualize; falls back to 252.0

    # Output
    output_periods: int = 365
    exclude_tickers: list[str] = field(default_factory=lambda: DEFAULT_STABLECOINS.copy())

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _annualized_signal(self, funding: pd.DataFrame) -> pd.DataFrame:
        """3-day EMA of daily funding rate, annualized.

        ``funding`` is daily *summed* 8h rates from HyperliquidDataPlugin.
        daily_sum × 365 ≈ 8h_rate × 3 × 365 (the spec formula).
        """
        return funding.ewm(span=self.signal_span_days, adjust=False).mean() * 365.0

    def _build_equal_weights(self, annualized: pd.DataFrame) -> pd.DataFrame:
        """Rank-and-select equal-weight matrix from annualized signal."""
        weights = pd.DataFrame(0.0, index=annualized.index, columns=annualized.columns)

        for date in annualized.index:
            row = annualized.loc[date].dropna()
            if len(row) < max(self.top_n_long, self.top_n_short):
                continue

            sorted_row = row.sort_values(ascending=False)

            long_cands = sorted_row.head(self.top_n_long)
            long_cands = long_cands[long_cands >= self.min_signal_annualized]

            short_cands = sorted_row.tail(self.top_n_short)
            short_cands = short_cands[short_cands <= -self.min_signal_annualized]

            n_long = len(long_cands)
            n_short = len(short_cands)

            if n_long > 0:
                # 0.5 gross per leg so max gross ≈ 100 %
                w = min(0.5 / n_long, self.max_concentration)
                weights.loc[date, long_cands.index] = w

            if n_short > 0:
                w = min(0.5 / n_short, self.max_concentration)
                weights.loc[date, short_cands.index] = -w

        return weights

    def _apply_vol_target(
        self,
        weights: pd.DataFrame,
        prices: pd.DataFrame,
        annualize: float = 252.0,
    ) -> pd.DataFrame:
        """Scale weights so realized portfolio vol ≈ target_vol.

        Uses lagged weights (shift(1)) to keep the estimator causal.
        Fills the warm-up window with scale=1 (no scaling before enough data).

        Args:
            annualize: Bars per year for vol annualization. Default 252 (equity).
                Strategy callers should derive this from pipeline-injected
                ``_pipeline_annualize`` per issue #20 / #23.
        """
        returns = prices.ffill().pct_change(fill_method=None)
        port_rets = (weights.shift(1) * returns).sum(axis=1)
        realized_vol = port_rets.ewm(span=self.vol_lookback, min_periods=5).std() * np.sqrt(annualize)
        scale = (self.target_vol / realized_vol.replace(0.0, np.nan)).clip(0.1, 3.0).fillna(1.0)
        scaled = weights.mul(scale.reindex(weights.index).fillna(1.0), axis=0)
        # Re-apply concentration cap: vol-targeting can push individual weights above max_concentration
        return scaled.clip(-self.max_concentration, self.max_concentration)

    # ------------------------------------------------------------------ #
    # StrategyPlugin interface
    # ------------------------------------------------------------------ #

    def run(
        self,
        data: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the carry strategy.

        Args:
            data: Dict with ``'prices'`` (required) and ``'funding_rates'``
                  (wide DataFrames, date × symbol).  ``funding_rates`` comes
                  from HyperliquidDataPlugin as daily summed 8h rates.
            params: Optional runtime overrides for any dataclass field.

        Returns:
            Dict with ``'weights'`` (DataFrame), ``'simple_weights'`` (dict),
            and ``'details'`` (annualized signal + exposure).
        """
        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        # Resolve annualize: explicit (self/params) wins, else pipeline-injected, else 252.0.
        pipeline_annualize = (params or {}).get("_pipeline_annualize")
        if self.annualize is None:
            effective_annualize = float(pipeline_annualize) if pipeline_annualize is not None else 252.0
        else:
            effective_annualize = float(self.annualize)
            if pipeline_annualize is not None and abs(effective_annualize - pipeline_annualize) > 1:
                logger.warning(
                    "CarryStrategy.annualize=%s overrides pipeline-derived %.1f. "
                    "If intentional, ignore; otherwise drop the explicit value.",
                    effective_annualize,
                    pipeline_annualize,
                )

        prices: pd.DataFrame = data["prices"]
        funding_raw: pd.DataFrame = data.get("funding_rates", pd.DataFrame())

        valid = [t for t in prices.columns if t not in self.exclude_tickers]
        prices = prices[valid]

        if funding_raw.empty:
            logger.warning("carry.v1: no funding_rates in data — returning zero weights")
            empty = pd.DataFrame(0.0, index=prices.index[-self.output_periods :], columns=prices.columns)
            return {"weights": empty, "simple_weights": {}, "details": {}}

        funding = funding_raw.reindex(columns=valid).fillna(0.0)

        annualized = self._annualized_signal(funding)
        weights = self._build_equal_weights(annualized)
        weights = self._apply_vol_target(weights, prices, effective_annualize)

        out = weights.tail(self.output_periods)
        latest = out.iloc[-1].dropna()
        simple = latest[abs(latest) > 0.001].to_dict()

        long_exp = float(latest[latest > 0].sum())
        short_exp = float(abs(latest[latest < 0].sum()))

        return {
            "weights": out,
            "simple_weights": simple,
            "details": {
                "annualized_funding": annualized.tail(self.output_periods),
                "exposure": {
                    "long": long_exp,
                    "short": short_exp,
                    "net": long_exp - short_exp,
                    "gross": long_exp + short_exp,
                },
            },
        }


def run(data: dict, params: dict = None) -> dict:
    """Standard module-level strategy interface."""
    return CarryStrategy().run(data, params)
