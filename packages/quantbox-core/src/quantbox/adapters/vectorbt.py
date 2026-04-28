"""Adapter for vectorbt — re-exports + thin helpers.

The pass-through is the contract:

    from quantbox.adapters.vectorbt import vbt
    pf = vbt.Portfolio.from_signals(prices, entries, exits)

Convenience helpers (e.g. ``from_signals_with_costs``) are bonus — they exist
when an idiom recurs across ≥2 consumers. Otherwise call ``vbt`` directly.

For the L1 backtest convenience layer, see ``quantbox.bt``.
"""

from __future__ import annotations

import vectorbt as vbt

__all__ = ["vbt", "from_signals_with_costs"]


def from_signals_with_costs(
    prices,
    signals,
    *,
    fees: float = 0.001,
    slippage: float = 0.0005,
    freq: str = "1D",
):
    """Convenience wrapper around ``vbt.Portfolio.from_signals`` with cost defaults.

    Treats ``signals > 0`` as entries and ``signals <= 0`` as exits — long-only
    by construction. For more complex setups, call ``vbt.Portfolio.from_signals``
    directly with explicit ``entries`` / ``exits`` / ``short_entries`` / ``short_exits``.

    Args:
        prices: Wide-format close prices (date index × symbol columns).
        signals: Same shape as prices; positive = enter, non-positive = exit.
        fees: Per-trade fee fraction (default 0.001 = 10 bps).
        slippage: Per-trade slippage fraction (default 0.0005 = 5 bps).
        freq: Frequency string for vbt (default ``"1D"``).

    Returns:
        ``vbt.Portfolio`` instance.
    """
    return vbt.Portfolio.from_signals(
        close=prices,
        entries=signals > 0,
        exits=signals <= 0,
        fees=fees,
        slippage=slippage,
        freq=freq,
    )
