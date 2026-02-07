from __future__ import annotations
from typing import Dict, List
import pandas as pd


def compute_returns(
    prices: pd.DataFrame,
    windows: List[int],
    *,
    method: str = "pct_change",
) -> Dict[str, pd.DataFrame]:
    """Compute period returns for multiple windows.

    Args:
        prices: Wide-format DataFrame (DatetimeIndex x symbol columns).
        windows: List of lookback periods (e.g. [1, 5, 21]).
        method: "pct_change" (default) or "log".

    Returns:
        Dict keyed ``"ret_{w}d"`` -> DataFrame of returns.
    """
    result: Dict[str, pd.DataFrame] = {}
    for w in windows:
        if method == "log":
            import numpy as np
            ret = np.log(prices / prices.shift(w))
        else:
            ret = prices.pct_change(periods=w)
        result[f"ret_{w}d"] = ret
    return result
