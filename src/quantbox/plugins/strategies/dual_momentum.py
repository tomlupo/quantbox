"""Dual Momentum strategy.

Implements Gary Antonacci's dual momentum framework: combines absolute
momentum (vs risk-free/safe asset) with relative momentum (asset A vs
asset B) to switch between asset classes.

Fully configurable — works for any pair of competing assets plus a safe
haven, in any asset class (equities/bonds, crypto, commodities).

Config example::

    strategy = DualMomentumStrategy()
    result = strategy.run(
        data={"prices": prices_df},
        params={
            "asset_a": "equity world",
            "asset_b": "global bonds",
            "safe_asset": "t-bills",
            "windows": [21, 63, 126, 189, 252],
        },
    )
    result["weights"]  # DataFrame: date x [asset_a, asset_b, safe_asset]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantbox.contracts import PluginMeta
from quantbox.features.momentum import compute_total_returns


@dataclass
class DualMomentumStrategy:
    """Absolute + relative momentum switching strategy.

    Computes total returns over multiple windows for each asset and a
    safe asset. For each window, checks whether the excess return
    (asset minus safe) is positive. The allocation to each asset is
    the fraction of windows with positive excess return, with the
    remainder going to the safe asset.
    """

    meta: PluginMeta = PluginMeta(
        name="strategy.dual_momentum.v1",
        kind="strategy",
        version="1.0.0",
        core_compat=">=0.1,<1.0",
        description=(
            "Dual momentum: combines absolute momentum (vs safe asset) "
            "with relative momentum to switch between asset classes."
        ),
        tags=("momentum", "dual", "tactical"),
        capabilities=("backtest", "live"),
        inputs=("prices",),
        outputs=("weights",),
    )

    # Defaults (overridable via params)
    windows: list[int] = field(default_factory=lambda: [21, 63, 126, 189, 252])

    def run(
        self,
        data: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute dual-momentum weights.

        Args:
            data: Must contain ``"prices"`` (wide DataFrame with at least
                the three tickers specified in params).
            params:
                - ``asset_a``: str — primary asset ticker (e.g. "equity world")
                - ``asset_b``: str — competing asset ticker (e.g. "global bonds")
                - ``safe_asset``: str — risk-off ticker (e.g. "t-bills")
                - ``windows``: list[int] — lookback windows (optional)

        Returns:
            Dict with ``"weights"`` (DataFrame: date x ticker) and
            ``"details"`` (diagnostic DataFrames).
        """
        params = params or {}
        prices = data["prices"]

        asset_a = params["asset_a"]
        asset_b = params["asset_b"]
        safe_asset = params["safe_asset"]
        windows = params.get("windows", self.windows)

        # Total returns per window (wide format: date x ticker)
        tr_dict = compute_total_returns(prices, windows)

        # For each window, compute excess return vs safe asset
        a_signals = []
        b_signals = []
        for tr_df in tr_dict.values():
            safe_ret = tr_df[safe_asset]
            a_signals.append(tr_df[asset_a] - safe_ret > 0)
            b_signals.append(tr_df[asset_b] - safe_ret > 0)

        # Fraction of windows with positive excess return
        a_frac = pd.concat(a_signals, axis=1).mean(axis=1)
        b_frac = pd.concat(b_signals, axis=1).mean(axis=1)

        # asset_a gets its fraction; asset_b gets its fraction scaled
        # by the remainder after asset_a
        weights = pd.DataFrame(index=a_frac.index)
        weights[asset_a] = a_frac
        weights[asset_b] = b_frac * (1 - a_frac)
        weights[safe_asset] = 1 - weights[asset_a] - weights[asset_b]
        weights = weights.dropna(how="any")

        return {
            "weights": weights,
            "details": {
                "asset_a_signal_frac": a_frac,
                "asset_b_signal_frac": b_frac,
            },
        }
