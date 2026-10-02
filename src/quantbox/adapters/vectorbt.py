"""Adapter for vectorbt — re-exports + thin helpers.

The pass-through is the contract:

    from quantbox.adapters.vectorbt import vbt
    # vbt fills the bar it is handed: lag the signals one bar first
    # (next-bar is mandatory, docs/adr/0005-next-bar-is-mandatory.md).
    entries, exits = entries.shift(1, fill_value=False), exits.shift(1, fill_value=False)
    pf = vbt.Portfolio.from_signals(prices, entries, exits)

Convenience helpers (e.g. ``from_signals_with_costs``) are bonus — they exist
when an idiom recurs across ≥2 consumers. Otherwise call ``vbt`` directly.

For the L1 backtest convenience layer, see ``quantbox.bt``.
"""

from __future__ import annotations

from quantbox.exceptions import MissingExtraError
from quantbox.execution import DEFAULT_LAG_BARS, apply_execution_lag, resolve_lag_bars

try:
    import vectorbt as vbt
except ModuleNotFoundError as exc:  # vectorbt ships in the [vectorbt] extra
    # Only the extra's OWN packages being absent means "install the extra"; a
    # module missing deeper inside an installed vectorbt is a broken install
    # and must surface as itself.
    if (exc.name or "").split(".")[0] not in ("vectorbt",):
        raise
    raise MissingExtraError("vectorbt", "quantbox.adapters.vectorbt", exc.name) from exc

__all__ = ["vbt", "from_signals_with_costs"]


def from_signals_with_costs(
    prices,
    signals,
    *,
    lag_bars: int = DEFAULT_LAG_BARS,
    fees: float = 0.001,
    slippage: float = 0.0005,
    freq: str = "1D",
):
    """Convenience wrapper around ``vbt.Portfolio.from_signals`` with cost defaults.

    Treats ``signals > 0`` as entries and ``signals <= 0`` as exits — long-only
    by construction. For more complex setups, call ``vbt.Portfolio.from_signals``
    directly with explicit ``entries`` / ``exits`` / ``short_entries`` / ``short_exits``.

    A signal computed with data through close ``t`` trades at close
    ``t + lag_bars`` — default and minimum 1, ``0`` raises (docs/adr/0005) —
    the same convention as ``quantbox run -c`` and ``backtest()``.

    Args:
        prices: Wide-format close prices (date index × symbol columns).
        signals: Same shape as prices; positive = enter, non-positive = exit.
        lag_bars: Execution lag in bars (default 1, next-bar).
        fees: Per-trade fee fraction (default 0.001 = 10 bps).
        slippage: Per-trade slippage fraction (default 0.0005 = 5 bps).
        freq: Frequency string for vbt (default ``"1D"``).

    Returns:
        ``vbt.Portfolio`` instance.
    """
    signals = apply_execution_lag(signals.astype(float), resolve_lag_bars({"lag_bars": lag_bars}))
    return vbt.Portfolio.from_signals(
        close=prices,
        entries=signals > 0,
        exits=signals <= 0,
        fees=fees,
        slippage=slippage,
        freq=freq,
    )
