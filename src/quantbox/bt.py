"""L1 convenience helpers for backtesting.

The fast path. One function call covers the most common backtest idiom — no
plugin registry, no YAML config, no run manifest. Reach for higher layers
(L3 plugin instances, L4 ``run_from_config``, L5 CLI ``--strict``) only when
the task demands them.

    >>> import quantbox.bt as qbt
    >>> result = qbt.run(prices, signals, fees=0.001)
    >>> print(result.metrics)

For full re-exports of the underlying library, see ``quantbox.adapters.vectorbt``.
For the layered API doctrine, see ``docs/architecture/api-layers.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .adapters.vectorbt import vbt

__all__ = ["run", "BacktestResult"]


@dataclass
class BacktestResult:
    """Result of an L1 backtest.

    Carries the underlying ``vbt.Portfolio`` (for users who want to drop
    down to the wheel) plus a robust core-metrics dict (for quick inspection).

    Attributes:
        portfolio: Underlying ``vbt.Portfolio`` instance.
    """

    portfolio: Any  # vbt.Portfolio — typed loosely to keep import cheap

    @property
    def metrics(self) -> dict[str, float]:
        """Core metrics — total return, Sharpe, max drawdown.

        Computed metric-by-metric so a single failing metric doesn't take
        the whole call down. For the full vbt stats (including trade-level
        breakdowns), call ``result.stats()`` or ``result.portfolio.stats()``
        directly.
        """
        pf = self.portfolio

        def _safe(fn) -> float:
            try:
                v = fn()
                if hasattr(v, "iloc") and len(v):
                    v = v.iloc[0]
                return float(v)
            except Exception:
                return float("nan")

        return {
            "total_return": _safe(pf.total_return),
            "sharpe_ratio": _safe(pf.sharpe_ratio),
            "max_drawdown": _safe(pf.max_drawdown),
        }

    def stats(self) -> Any:
        """Full ``vbt.Portfolio.stats()`` — may fail on edge cases per vbt internals."""
        return self.portfolio.stats()

    @property
    def returns(self) -> pd.Series:
        """Strategy returns series."""
        return self.portfolio.returns()

    @property
    def value(self) -> pd.Series:
        """Portfolio value over time."""
        return self.portfolio.value()


def run(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    fees: float = 0.001,
    slippage: float = 0.0005,
    freq: str = "1D",
) -> BacktestResult:
    """Quick vectorbt backtest with sensible defaults — the L1 fast path.

    Treats ``signals > 0`` as entries and ``signals <= 0`` as exits. For
    short-side trading, position-sizing rules, or rebalancing strategies,
    use ``quantbox.adapters.vectorbt.vbt`` directly or escalate to L3/L4.

    Args:
        prices: Wide-format close prices (date index × symbol columns).
        signals: Same shape as prices; positive = long, non-positive = flat.
        fees: Per-trade fee fraction (default 0.001 = 10 bps).
        slippage: Per-trade slippage fraction (default 0.0005 = 5 bps).
        freq: Frequency string for vbt (default ``"1D"``).

    Returns:
        ``BacktestResult`` with ``portfolio``, ``metrics``, ``returns``, ``value``.

    Example:
        >>> import pandas as pd, quantbox.bt as qbt
        >>> prices = pd.read_parquet("prices.parquet")
        >>> signals = (prices > prices.rolling(60).mean()).astype(int)
        >>> result = qbt.run(prices, signals)
        >>> print(result.metrics)
    """
    portfolio = vbt.Portfolio.from_signals(
        close=prices,
        entries=signals > 0,
        exits=signals <= 0,
        fees=fees,
        slippage=slippage,
        freq=freq,
    )
    return BacktestResult(portfolio=portfolio)
