"""Re-export of :mod:`quantbox.metrics`, the one metrics module (TOM-1448).

The metrics lived here until TOM-1448 moved them to core. This path stays
importable (robo-lab imports it) and hands back the SAME objects, so a caller
of either path computes with one implementation.
"""

from quantbox.metrics import (  # noqa: F401
    TRADING_DAYS_PER_YEAR,
    _extract_returns,
    _max_drawdown_duration,
    annual_turnover,
    compute_backtest_metrics,
    compute_cvar,
    compute_drawdown_series,
    compute_portfolio_cvar,
    compute_portfolio_var,
    compute_rolling_sharpe,
    compute_var,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    top_drawdowns,
    tracking_error,
    turnover_series,
)
