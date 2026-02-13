"""Dispatch feature computations from a manifest dict."""

from __future__ import annotations

from typing import Any

import pandas as pd

from quantbox.features.channels import compute_donchian
from quantbox.features.moving_averages import compute_ema, compute_sma
from quantbox.features.returns import compute_returns
from quantbox.features.volatility import compute_ewm_vol, compute_rolling_vol

_DISPATCHERS = {
    "returns": lambda prices, cfg: compute_returns(
        prices, cfg.get("windows", [1]), method=cfg.get("method", "pct_change")
    ),
    "rolling_vol": lambda prices, cfg: compute_rolling_vol(
        prices,
        cfg.get("windows", [21]),
        annualize=cfg.get("annualize", True),
        factor=cfg.get("factor", 365.0),
    ),
    "ewm_vol": lambda prices, cfg: compute_ewm_vol(
        prices,
        cfg.get("spans", [21]),
        annualize=cfg.get("annualize", True),
        factor=cfg.get("factor", 365.0),
    ),
    "sma": lambda prices, cfg: compute_sma(prices, cfg.get("windows", [20])),
    "ema": lambda prices, cfg: compute_ema(prices, cfg.get("spans", [20])),
    "donchian": lambda prices, cfg: compute_donchian(prices, cfg.get("windows", [20])),
}


def compute_features_bundle(
    prices: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Compute multiple feature sets from a manifest.

    Args:
        prices: Wide-format DataFrame (DatetimeIndex x symbol columns).
        manifest: Dict mapping feature type to config.
            Example::

                {
                    "returns": {"windows": [1, 5, 21]},
                    "sma": {"windows": [20, 50]},
                    "rolling_vol": {"windows": [21], "annualize": True},
                }

    Returns:
        Merged dict of all computed features, keyed by feature name.
    """
    result: dict[str, pd.DataFrame] = {}
    for feature_type, config in manifest.items():
        dispatcher = _DISPATCHERS.get(feature_type)
        if dispatcher is None:
            raise ValueError(f"Unknown feature type: '{feature_type}'")
        result.update(dispatcher(prices, config or {}))
    return result
