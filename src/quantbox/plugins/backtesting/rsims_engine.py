"""
rsims backtesting engine — production-grade daily portfolio simulator.

Ported from quantlab's ``backtester_rsims_py.py``.

Features
--------
- Daily mark-to-market P&L from price changes.
- Per-asset funding rates (perpetual futures).
- Linear % commissions on traded notional.
- Margin requirements with optional maintenance buffer.
- No-trade buffer to reduce turnover.
- Optional compounding control (``capitalise_profits``).
- Two equity modes: ``"rsims"`` (legacy) and ``"mtm"`` (mark-to-market).
- Max gross leverage cap.
- Forced pro-rata liquidation on margin calls.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def positions_from_no_trade_buffer(
    current_positions: np.ndarray,
    current_prices: np.ndarray,
    current_target_weights: np.ndarray,
    cap_equity: float,
    trade_buffer: float,
) -> np.ndarray:
    """Calculate target positions using a no-trade buffer around target weights.

    Positions are only rebalanced when they deviate from the target by more
    than *trade_buffer*.  When rebalancing *is* triggered for an asset, the
    new position targets the edge of the band (heuristic optimal for linear
    transaction costs).

    Parameters
    ----------
    current_positions : np.ndarray
        Current contracts/shares per asset.
    current_prices : np.ndarray
        Current prices per asset.
    current_target_weights : np.ndarray
        Desired portfolio weights (need not sum to 1).
    cap_equity : float
        Capital base for converting weights → notional.
    trade_buffer : float
        Half-width of the no-trade band around each target weight.

    Returns
    -------
    np.ndarray
        Target positions after applying the no-trade buffer.
    """
    num_assets = len(current_positions)
    with np.errstate(divide="ignore", invalid="ignore"):
        current_weights = np.where(
            cap_equity != 0,
            (current_positions * current_prices) / cap_equity,
            0.0,
        )

    target_positions = current_positions.copy()

    for j in range(num_assets):
        tw = current_target_weights[j]
        if np.isnan(tw) or tw == 0:
            target_positions[j] = 0
        elif current_weights[j] < tw - trade_buffer:
            target_positions[j] = (tw - trade_buffer) * cap_equity / current_prices[j]
        elif current_weights[j] > tw + trade_buffer:
            target_positions[j] = (tw + trade_buffer) * cap_equity / current_prices[j]
        # else: keep current position (within band)

    return target_positions


def fixed_commission_backtest_with_funding(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    funding_rates: pd.DataFrame,
    trade_buffer: float = 0.0,
    initial_cash: float = 10_000,
    margin: float = 0.05,
    commission_pct: float = 0.0,
    capitalise_profits: bool = False,
    *,
    equity_basis: str = "rsims",
    maintenance_buffer: float = 0.0,
    max_gross_leverage: float | None = None,
) -> pd.DataFrame:
    """Daily fixed-commission backtest with funding rates and margin.

    Parameters
    ----------
    prices : pd.DataFrame
        Trade prices (index=dates, columns=tickers).
    target_weights : pd.DataFrame
        Desired portfolio weights (same shape as *prices*).
    funding_rates : pd.DataFrame
        Per-period funding rates (same shape as *prices*).
    trade_buffer : float
        No-trade band half-width around target weights.
    initial_cash : float
        Starting cash balance.
    margin : float
        Maintenance margin rate applied to gross exposure.
    commission_pct : float
        Linear commission as fraction of traded notional.
    capitalise_profits : bool
        If True, size off current equity (compound). Otherwise cap at
        ``min(initial_cash, equity)``.
    equity_basis : ``"rsims"`` | ``"mtm"``
        ``"rsims"``: equity = cash + maintenance_margin (legacy).
        ``"mtm"``: equity = cash + sum(position_value).
    maintenance_buffer : float
        Require equity >= (1 + buffer) * maintenance_margin.
    max_gross_leverage : float | None
        Cap gross exposure / equity.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame (date x ticker + Cash row per date) with
        columns: Close, Position, Value, Margin, Funding, PeriodPnL,
        Trades, TradeValue, Commission, MarginCall, ReducedTargetPos.
    """
    if trade_buffer < 0:
        raise ValueError("trade_buffer must be >= 0")

    # Ensure DatetimeIndex
    for df in (prices, target_weights, funding_rates):
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    # Alignment checks
    if not prices.index.equals(target_weights.index):
        raise ValueError("Prices and target weights must have the same date index")
    if not prices.index.equals(funding_rates.index):
        raise ValueError("Prices and funding rates must have the same date index")
    if prices.shape != target_weights.shape:
        raise ValueError("Prices and weights must have same shape")
    if prices.shape != funding_rates.shape:
        raise ValueError("Prices and funding must have same shape")

    # Replace NAs
    if target_weights.isna().any().any():
        logger.warning("NA in target weights — replacing with zeros")
        target_weights = target_weights.fillna(0)
    if funding_rates.isna().any().any():
        logger.warning("NA in funding rates — replacing with zeros")
        funding_rates = funding_rates.fillna(0)

    tickers = prices.columns.tolist()
    dates = prices.index
    num_assets = len(tickers)

    current_positions = np.zeros(num_assets)
    previous_prices = np.full(num_assets, np.nan)
    cash = float(initial_cash)
    maint_margin = 0.0

    results: list[dict] = []

    for i in range(len(dates)):
        current_date = dates[i]
        current_prices = prices.iloc[i].values.astype(float)
        current_target_weights = target_weights.iloc[i].values.astype(float)
        current_funding_rates = funding_rates.iloc[i].values.astype(float)

        # --- Funding on current positions ---
        funding = current_positions * current_prices * current_funding_rates
        funding = np.where(np.isnan(funding), 0.0, funding)

        # --- Mark-to-market PnL ---
        period_pnl = current_positions * (current_prices - previous_prices)
        period_pnl = np.where(np.isnan(period_pnl), 0.0, period_pnl) + funding

        # --- Update cash ---
        cash = (
            cash + np.nansum(period_pnl) + maint_margin - margin * np.nansum(np.abs(current_positions * current_prices))
        )

        position_value = current_positions * current_prices
        position_value = np.where(np.isnan(position_value), 0.0, position_value)
        maint_margin = margin * np.nansum(np.abs(position_value))

        # --- Margin call check ---
        equity_mtm = cash + np.nansum(position_value)
        equity_required = (1.0 + maintenance_buffer) * maint_margin

        margin_call = False
        liq_contracts = np.zeros(num_assets)
        liq_commissions = np.zeros(num_assets)
        liq_trade_value = np.zeros(num_assets)

        if equity_mtm < equity_required:
            margin_call = True

            target_maint_margin = equity_mtm / max(1e-12, 1.0 + maintenance_buffer)
            if maint_margin > 0:
                liquidate_factor = np.clip(1.0 - target_maint_margin / maint_margin, 0.0, 1.0)
            else:
                liquidate_factor = 0.0

            liq_contracts = liquidate_factor * current_positions
            liq_trade_value = liq_contracts * current_prices
            liq_commissions = np.abs(liq_trade_value) * commission_pct

            current_positions = current_positions - liq_contracts
            position_value = current_positions * current_prices
            position_value = np.where(np.isnan(position_value), 0.0, position_value)

            maint_margin = margin * np.nansum(np.abs(position_value))
            cash -= np.nansum(liq_commissions)
            equity_mtm = cash + np.nansum(position_value)

        # --- Equity for sizing ---
        if equity_basis.lower() == "mtm":
            equity = equity_mtm
        elif equity_basis.lower() == "rsims":
            equity = cash + maint_margin
        else:
            raise ValueError("equity_basis must be 'rsims' or 'mtm'")

        cap_equity = equity if capitalise_profits else min(initial_cash, equity)

        # --- Target positions via no-trade buffer ---
        target_positions = positions_from_no_trade_buffer(
            current_positions, current_prices, current_target_weights, cap_equity, trade_buffer
        )

        # --- Leverage cap ---
        target_position_value = target_positions * current_prices
        gross_exposure = np.nansum(np.abs(target_position_value))

        if max_gross_leverage is not None and equity > 0:
            max_exposure = max_gross_leverage * equity
            if gross_exposure > max_exposure and gross_exposure > 0:
                scale = max_exposure / gross_exposure
                target_positions = target_positions * scale
                target_position_value = target_positions * current_prices
                gross_exposure = np.nansum(np.abs(target_position_value))

        # --- Trades & commissions ---
        trades = target_positions - current_positions
        trade_value = trades * current_prices
        commissions = np.abs(trade_value) * commission_pct

        required_margin_target = margin * np.nansum(np.abs(target_position_value))
        post_trade_cash = cash + maint_margin - required_margin_target - np.nansum(commissions)

        reduced_target_pos = False

        if post_trade_cash < (1.0 + maintenance_buffer) * required_margin_target:
            reduced_target_pos = True

            denom = margin * (1.0 + maintenance_buffer)
            if denom <= 0:
                max_post_trade_contracts_value = 0.0
            else:
                max_post_trade_contracts_value = 0.95 * max(0.0, cash + maint_margin - np.nansum(commissions)) / denom

            if gross_exposure > 0:
                reduce_by = np.clip(max_post_trade_contracts_value / gross_exposure, 0.0, 1.0)
                target_positions = np.sign(target_positions) * reduce_by * np.abs(target_positions)

                trades = target_positions - current_positions
                trade_value = trades * current_prices
                commissions = np.abs(trade_value) * commission_pct

                current_positions = target_positions
                position_value = current_positions * current_prices
                position_value = np.where(np.isnan(position_value), 0.0, position_value)

                required_margin_target = margin * np.nansum(np.abs(position_value))
                post_trade_cash = cash + maint_margin - required_margin_target - np.nansum(commissions)
        else:
            current_positions = target_positions
            position_value = current_positions * current_prices
            position_value = np.where(np.isnan(position_value), 0.0, position_value)

        # --- Finalize ---
        cash = post_trade_cash
        maint_margin = margin * np.nansum(np.abs(position_value))

        for j, ticker in enumerate(tickers):
            results.append(
                {
                    "date": current_date,
                    "ticker": ticker,
                    "Close": current_prices[j],
                    "Position": current_positions[j],
                    "Value": position_value[j],
                    "Margin": margin * abs(position_value[j]),
                    "Funding": funding[j],
                    "PeriodPnL": period_pnl[j],
                    "Trades": trades[j] - liq_contracts[j],
                    "TradeValue": trade_value[j] - liq_trade_value[j],
                    "Commission": commissions[j] + liq_commissions[j],
                    "MarginCall": margin_call,
                    "ReducedTargetPos": reduced_target_pos,
                }
            )

        results.append(
            {
                "date": current_date,
                "ticker": "Cash",
                "Close": 0.0,
                "Position": cash,
                "Value": cash,
                "Margin": 0.0,
                "Funding": 0.0,
                "PeriodPnL": 0.0,
                "Trades": 0.0,
                "TradeValue": 0.0,
                "Commission": 0.0,
                "MarginCall": margin_call,
                "ReducedTargetPos": reduced_target_pos,
            }
        )

        previous_prices = current_prices.copy()

    results_df = pd.DataFrame(results)
    results_df["date"] = pd.to_datetime(results_df["date"])
    results_df = results_df.set_index("date")
    return results_df
