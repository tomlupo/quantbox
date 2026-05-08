from __future__ import annotations

import pandas as pd


def compute_donchian(
    prices: pd.DataFrame,
    windows: list[int],
) -> dict[str, pd.DataFrame]:
    """Compute Donchian channels (rolling high/low/mid).

    Args:
        prices: Wide-format DataFrame (DatetimeIndex x symbol columns).
        windows: Rolling window sizes.

    Returns:
        Dict with keys ``"donchian_{w}d_high"``, ``"donchian_{w}d_low"``,
        ``"donchian_{w}d_mid"`` for each window.
    """
    result: dict[str, pd.DataFrame] = {}
    for w in windows:
        high = prices.rolling(window=w).max()
        low = prices.rolling(window=w).min()
        mid = (high + low) / 2
        result[f"donchian_{w}d_high"] = high
        result[f"donchian_{w}d_low"] = low
        result[f"donchian_{w}d_mid"] = mid
    return result
