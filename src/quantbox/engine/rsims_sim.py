"""
rsims backtesting engine — production-grade daily portfolio simulator.

Ported from quantlab's ``backtester_rsims_py.py``.

Features
--------
- Daily mark-to-market P&L from price changes.
- Per-asset funding rates (perpetual futures).
- Linear % commissions on traded notional, a fixed fee per order and
  proportional slippage on the fill price (the three ``quantbox.engine.Costs``).
- Margin requirements with optional maintenance buffer.
- No-trade buffer to reduce turnover.
- Compounding control (``capitalise_profits``; on by default, as vectorbt).
- Two equity modes: ``"rsims"`` (legacy) and ``"mtm"`` (mark-to-market).
- Max gross leverage cap.
- Forced pro-rata liquidation on margin calls.

Core code (numpy and pandas only), behind the engine seam's rsims adapter
(:mod:`quantbox.engine.rsims`). It lived at
``quantbox.plugins.backtesting.rsims_engine`` until TOM-1451; since 0.13.0 that
path raises ``ImportError`` naming this one (TOM-1457).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["fixed_commission_backtest_with_funding", "positions_from_no_trade_buffer"]

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
    margin: float = 0.0,
    commission_pct: float = 0.0,
    capitalise_profits: bool = True,
    *,
    slippage: float = 0.0,
    fixed_fees: float = 0.0,
    equity_basis: str = "rsims",
    maintenance_buffer: float = 0.0,
    max_gross_leverage: float | None = None,
    fee_free: Sequence[str] = (),
    orders: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Daily fixed-commission backtest with funding rates and margin.

    Parameters
    ----------
    prices : pd.DataFrame
        Trade prices (index=dates, columns=tickers).
    target_weights : pd.DataFrame
        Desired portfolio weights (same shape as *prices*), ALREADY lagged:
        row ``t`` fills at ``close[t]`` (``apply_execution_lag``;
        docs/adr/0005-next-bar-is-mandatory.md).
    funding_rates : pd.DataFrame
        Per-period funding rates (same shape as *prices*).
    trade_buffer : float
        No-trade band half-width around target weights.
    initial_cash : float
        Starting cash balance.
    margin : float
        Maintenance margin rate applied to gross exposure.
    commission_pct : float
        Linear commission as fraction of traded notional (at the fill price).
    capitalise_profits : bool
        If True (the default, as vectorbt), size off current equity
        (compound). Otherwise cap at ``min(initial_cash, equity)``.
    slippage : float
        Proportional slippage: a buy fills at ``close * (1 + slippage)``, a sell
        at ``close * (1 - slippage)``; the position is still marked at the close,
        so the slippage is a cost paid from cash on the trade bar (vectorbt's
        convention).
    fixed_fees : float
        A fixed fee per order, in quote currency: charged once per asset on a
        bar where that asset's position changes by more than dust (vectorbt's
        ``is_close`` rule, 1e-9 relative). A dust change is not traded at all,
        and a sell whose proceeds do not cover its fees is not placed
        (vectorbt rejects it the same way).
    equity_basis : ``"rsims"`` | ``"mtm"``
        ``"rsims"``: equity = cash + maintenance_margin (legacy).
        ``"mtm"``: equity = cash + sum(position_value).
    maintenance_buffer : float
        Require equity >= (1 + buffer) * maintenance_margin.
    max_gross_leverage : float | None
        Cap gross exposure / equity.
    fee_free : sequence of str
        Tickers traded without commission, fixed fee or slippage — the synthetic cash legs of
        ``venue.financing`` (:mod:`quantbox.financing`).
    orders : DataFrame of bool | None
        Per-cell order mask (same index and tickers as *prices*): where it is
        False the position is left as it is — no trade on a bar the instrument
        did not print, or off the execution calendar
        (:mod:`quantbox.engine.schedule`). None trades every cell, every bar.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame (date x ticker + Cash row per date) with
        columns: Close, Position, Value, Margin, Funding, PeriodPnL,
        Trades, TradeValue (signed, at the fill price), Commission (the
        proportional and fixed fees), Slippage (its cost), MarginCall,
        ReducedTargetPos.
    """
    if trade_buffer < 0:
        raise ValueError("trade_buffer must be >= 0")
    if slippage < 0 or fixed_fees < 0:
        raise ValueError(f"slippage and fixed_fees must be >= 0, got slippage={slippage}, fixed_fees={fixed_fees}")

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
    # Per-asset costs: the financing cash legs trade free.
    free = np.isin(tickers, list(fee_free))
    commission_pct = np.where(free, 0.0, float(commission_pct))
    slippage_pct = np.where(free, 0.0, float(slippage))
    fixed_fee = np.where(free, 0.0, float(fixed_fees))

    def _costs(trades: np.ndarray, px: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per asset: the signed traded value at the fill price, the fees (proportional + fixed), the slippage."""
        fill = px * (1.0 + slippage_pct * np.sign(trades))
        value = trades * fill
        slip = np.abs(trades * px) * slippage_pct
        fees = np.abs(value) * commission_pct + np.where(np.abs(trades) > 0, fixed_fee, 0.0)
        return value, fees, slip

    order_mask = (
        np.ones(prices.shape, dtype=bool)
        if orders is None
        else orders.reindex(index=dates, columns=tickers, fill_value=False).to_numpy(dtype=bool)
    )

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
        liq_slippage = np.zeros(num_assets)
        liq_trade_value = np.zeros(num_assets)

        if equity_mtm < equity_required:
            margin_call = True

            target_maint_margin = equity_mtm / max(1e-12, 1.0 + maintenance_buffer)
            if maint_margin > 0:
                liquidate_factor = np.clip(1.0 - target_maint_margin / maint_margin, 0.0, 1.0)
            else:
                liquidate_factor = 0.0

            liq_contracts = liquidate_factor * current_positions
            sold_value, liq_commissions, liq_slippage = _costs(-liq_contracts, current_prices)
            liq_trade_value = -sold_value

            current_positions = current_positions - liq_contracts
            position_value = current_positions * current_prices
            position_value = np.where(np.isnan(position_value), 0.0, position_value)

            maint_margin = margin * np.nansum(np.abs(position_value))
            cash -= np.nansum(liq_commissions) + np.nansum(liq_slippage)
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
        target_positions = np.where(order_mask[i], target_positions, current_positions)
        # A dust change is no order (vectorbt's is_close: 1e-9 relative, 1e-12 absolute).
        dust = np.abs(target_positions - current_positions) <= np.maximum(
            1e-9 * np.maximum(np.abs(target_positions), np.abs(current_positions)), 1e-12
        )
        target_positions = np.where(dust, current_positions, target_positions)
        # A sell whose proceeds do not cover its fees is not placed (vectorbt: CantCoverFees).
        sold = current_positions - target_positions
        proceeds = sold * current_prices * (1.0 - slippage_pct) * (1.0 - commission_pct)
        uneconomic = (sold > 0) & (proceeds < fixed_fee)
        target_positions = np.where(uneconomic, current_positions, target_positions)

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

        # --- Trades & costs ---
        trades = target_positions - current_positions
        trade_value, commissions, slippage_cost = _costs(trades, current_prices)
        costs = np.nansum(commissions) + np.nansum(slippage_cost)

        required_margin_target = margin * np.nansum(np.abs(target_position_value))
        post_trade_cash = cash + maint_margin - required_margin_target - costs

        reduced_target_pos = False

        if post_trade_cash < (1.0 + maintenance_buffer) * required_margin_target:
            reduced_target_pos = True

            denom = margin * (1.0 + maintenance_buffer)
            if denom <= 0:
                max_post_trade_contracts_value = 0.0
            else:
                max_post_trade_contracts_value = 0.95 * max(0.0, cash + maint_margin - costs) / denom

            if gross_exposure > 0:
                reduce_by = np.clip(max_post_trade_contracts_value / gross_exposure, 0.0, 1.0)
                target_positions = np.sign(target_positions) * reduce_by * np.abs(target_positions)

                trades = target_positions - current_positions
                trade_value, commissions, slippage_cost = _costs(trades, current_prices)
                costs = np.nansum(commissions) + np.nansum(slippage_cost)

                current_positions = target_positions
                position_value = current_positions * current_prices
                position_value = np.where(np.isnan(position_value), 0.0, position_value)

                required_margin_target = margin * np.nansum(np.abs(position_value))
                post_trade_cash = cash + maint_margin - required_margin_target - costs
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
                    "Slippage": slippage_cost[j] + liq_slippage[j],
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
                "Slippage": 0.0,
                "MarginCall": margin_call,
                "ReducedTargetPos": reduced_target_pos,
            }
        )

        previous_prices = current_prices.copy()

    results_df = pd.DataFrame(results)
    results_df["date"] = pd.to_datetime(results_df["date"])
    results_df = results_df.set_index("date")
    return results_df
