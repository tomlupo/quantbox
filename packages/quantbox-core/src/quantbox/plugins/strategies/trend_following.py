"""Trend-following strategy plugin (signal engine).

Converts TSMOM composite signals into portfolio weights. This is the
generic signal-to-weight engine — it handles per-ticker binary trend
decisions and risk-off allocation. Application-specific logic (e.g.
waterfall rules, basket cascades) lives in the consumer codebase.

Works for any asset class: equities, fixed income, commodities, crypto.

Config example::

    strategy = TrendFollowingStrategy()
    result = strategy.run(
        data={"prices": prices_df},
        params={
            "risk_off_ticker": "t-bills",
            "signal_columns": ["TF_fast", "TF_slow", "TF"],
        },
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantbox.contracts import PluginMeta
from quantbox.features.momentum import compute_tsmom


@dataclass
class TrendFollowingStrategy:
    """TSMOM-based trend-following strategy.

    Computes TSMOM indicator suite, then for each selected composite
    signal, produces per-ticker binary trend weights. When a ticker's
    signal is below the threshold, that allocation goes to the
    risk-off asset.
    """

    meta: PluginMeta = PluginMeta(
        name="strategy.trend_following.v1",
        kind="strategy",
        version="1.0.0",
        core_compat=">=0.1,<1.0",
        description=(
            "Time-series momentum trend-following strategy with "
            "fast/slow signal classification and configurable risk-off."
        ),
        tags=("momentum", "tsmom", "trend"),
        capabilities=("backtest", "live"),
        inputs=("prices",),
        outputs=("weights",),
    )

    signal_columns: list[str] = field(default_factory=lambda: ["TF_fast", "TF_slow", "TF"])

    def run(
        self,
        data: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute trend-following weights.

        Args:
            data: Must contain ``"prices"`` (wide DataFrame). Optionally
                ``"tsmom"`` (pre-computed TSMOM results to skip recomputation).
            params:
                - ``risk_off_ticker``: str or dict — per-ticker risk-off
                  mapping, or a single ticker applied to all.
                  Example: ``"t-bills"`` or ``{"default": "t-bills", "equity europe": "eur tbonds"}``
                - ``signal_columns``: list[str] — which composite signals to use
                - ``tsmom_kwargs``: dict — extra kwargs for ``compute_tsmom``
                - ``threshold``: float — signal threshold (default 0.5)

        Returns:
            Dict with ``"weights"`` (DataFrame per signal) and ``"tsmom"``
            (full TSMOM indicator results).
        """
        params = params or {}
        prices = data["prices"]

        risk_off = params.get("risk_off_ticker")
        signal_cols = params.get("signal_columns", self.signal_columns)
        threshold = params.get("threshold", 0.5)
        tsmom_kwargs = params.get("tsmom_kwargs", {})

        tickers = [c for c in prices.columns if c != risk_off]

        # Compute or reuse TSMOM
        tsmom = data.get("tsmom")
        if tsmom is None:
            tsmom = compute_tsmom(prices[tickers], **tsmom_kwargs)

        composite = tsmom["composite"]

        # Normalize risk_off_ticker to per-ticker dict
        if risk_off is None:
            risk_off_map = {t: None for t in tickers}
        elif isinstance(risk_off, str):
            risk_off_map = {t: risk_off for t in tickers}
        elif isinstance(risk_off, dict):
            default = risk_off.get("default")
            risk_off_map = {t: risk_off.get(t, default) for t in tickers}
        else:
            risk_off_map = {t: None for t in tickers}

        # Build weights per signal column
        weights_all: dict[str, pd.DataFrame] = {}

        for sig_name in signal_cols:
            if sig_name not in composite.columns:
                continue
            signal = composite[sig_name]

            # Per-ticker: weight = signal value (continuous [0, 1])
            # or binary: 1 if signal > threshold, 0 otherwise
            w_parts = {}
            for ticker in tickers:
                w_ticker = signal.rename(ticker).to_frame()
                ro = risk_off_map.get(ticker)
                if ro:
                    w_ticker[ro] = 1 - w_ticker[ticker]
                w_parts[ticker] = w_ticker

            # Combine into single DataFrame (average across tickers)
            if w_parts:
                combined = pd.concat(w_parts.values(), axis=1)
                # Sum duplicate columns (e.g. same risk-off ticker)
                if combined.columns.duplicated().any():
                    combined = combined.T.groupby(level=0).mean().T
                weights_all[sig_name] = combined.dropna(how="all")

        # Default output: use "TF" (all signals average) if available
        primary_key = "TF" if "TF" in weights_all else next(iter(weights_all), None)
        primary_weights = weights_all.get(primary_key, pd.DataFrame())

        return {
            "weights": primary_weights,
            "weights_by_signal": weights_all,
            "tsmom": tsmom,
        }
