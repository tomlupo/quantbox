"""Futures rebalancing plugin.

Handles order generation for perpetual-futures accounts where:
- Positions are signed (negative = short)
- Portfolio value equals margin balance (not cash + position value)
- Leverage cap applies to gross exposure (sum of abs weights)
- No short clamping: negative weights are respected
- Symbols use base asset names (broker handles exchange-specific formatting)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import BrokerPlugin, PluginMeta

logger = logging.getLogger(__name__)

DEFAULT_CAPITAL_AT_RISK = 1.0
DEFAULT_STABLE_COIN = "USDT"


@dataclass
class FuturesRebalancer:
    """Futures rebalancer: margin-based with signed positions.

    ``generate_orders()`` params keys:
      - tranches, max_leverage  (risk transforms — no allow_short, always allowed)
      - min_trade_size, capital_at_risk, stable_coin_symbol
      - strategy_results  (for tranching time-series)
    """

    meta = PluginMeta(
        name="rebalancing.futures.v1",
        kind="rebalancing",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Futures rebalancer: margin-based with signed position support",
        tags=("rebalancing", "futures", "trading"),
    )

    def generate_orders(
        self,
        *,
        weights: dict[str, float],
        broker: BrokerPlugin,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        risk_adjusted = self._apply_risk_transforms(weights, params)

        stable_coin = str(params.get("stable_coin_symbol", DEFAULT_STABLE_COIN))
        capital_at_risk = float(params.get("capital_at_risk", DEFAULT_CAPITAL_AT_RISK))

        order_result = self._generate_orders(
            broker=broker,
            weights=risk_adjusted,
            capital_at_risk=capital_at_risk,
            stable_coin=stable_coin,
            params=params,
        )
        order_result["weights"] = risk_adjusted
        return order_result

    # ==================================================================
    # Risk transforms
    # ==================================================================
    def _apply_risk_transforms(
        self,
        weights: dict[str, float],
        params: dict[str, Any],
    ) -> dict[str, float]:
        """Apply tranching and leverage cap. No short clamping."""
        tranches = int(params.get("tranches", 1))
        max_leverage = float(params.get("max_leverage", 1))

        s = pd.Series(weights, dtype=float)

        # Tranching: rolling mean over N days (requires historical weights)
        if tranches > 1:
            strategy_results = params.get("strategy_results", {})
            if strategy_results:
                try:
                    names = list(strategy_results.keys())
                    weight_overrides = params.get("strategy_weights", {})
                    weight_dfs = []
                    account_weights = []
                    for sname in names:
                        sinfo = strategy_results[sname]
                        w_df = sinfo["result"].get("weights", pd.DataFrame())
                        if w_df is not None and not w_df.empty:
                            weight_dfs.append(w_df)
                            account_weights.append(float(weight_overrides.get(sname, sinfo["weight"])))

                    if len(weight_dfs) == 1:
                        full_ts = weight_dfs[0] * account_weights[0]
                    else:
                        combined = pd.concat(
                            weight_dfs,
                            axis=1,
                            keys=names[: len(weight_dfs)],
                            names=["strategy"],
                        )
                        acct_w = pd.Series(
                            account_weights,
                            index=pd.Index(names[: len(weight_dfs)], name="strategy"),
                        )
                        weighted = combined.mul(acct_w, level="strategy")
                        full_ts = weighted.droplevel(0, axis=1)
                        if isinstance(full_ts.columns, pd.MultiIndex) or full_ts.columns.duplicated().any():
                            full_ts = full_ts.T.groupby(level=0).sum().T

                    smoothed = full_ts.rolling(window=tranches).mean().iloc[-1]
                    s = smoothed
                except Exception:
                    logger.warning("Tranching failed, using un-smoothed weights")

        # Leverage cap on gross exposure
        gross = s.abs().sum()
        if gross > max_leverage:
            logger.warning(
                "Leverage %.4f exceeds max_leverage %.1f, scaling down",
                gross,
                max_leverage,
            )
            s = s / gross * max_leverage

        # NO short clamping — negative weights are valid for futures

        # Drop zeros and sort by absolute value descending
        s = s[s != 0].reindex(s[s != 0].abs().sort_values(ascending=False).index)

        return {str(k): float(v) for k, v in s.items()}

    # ==================================================================
    # Order generation
    # ==================================================================
    def _generate_orders(
        self,
        broker: BrokerPlugin,
        weights: dict[str, float],
        capital_at_risk: float,
        stable_coin: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate rebalancing DataFrame and executable orders."""
        min_trade_size = float(params.get("min_trade_size", 0.01))
        exclusions = list(params.get("exclusions", [])) + [stable_coin]
        exclusions = list(set(exclusions))

        positions_df = broker.get_positions()
        cash = broker.get_cash() or {}

        # For futures, portfolio value = margin balance (cash), not cash + positions
        quote = stable_coin
        cash_available = float(cash.get(quote, cash.get("USD", cash.get("USDC", cash.get("USDT", 0.0)))))
        total_value = max(0, cash_available)

        # Current holdings: signed qty
        current_holdings: dict[str, float] = {}
        if positions_df is not None and not positions_df.empty:
            for _, row in positions_df.iterrows():
                sym = str(row.get("symbol", ""))
                qty = float(row.get("qty", 0))
                if qty != 0:
                    current_holdings[sym] = qty

        all_symbols = sorted(set(list(weights.keys()) + list(current_holdings.keys())))
        all_symbols = [s for s in all_symbols if s not in exclusions]

        # Fetch prices
        price_map: dict[str, float | None] = {}
        if all_symbols:
            try:
                snap = broker.get_market_snapshot(all_symbols)
                if snap is not None and not snap.empty:
                    for _, row in snap.iterrows():
                        sym = str(row.get("symbol", ""))
                        mid = row.get("mid")
                        if mid is not None and not (isinstance(mid, float) and np.isnan(mid)):
                            price_map[sym] = float(mid)
            except Exception:
                pass

        def get_price(asset: str) -> float | None:
            return price_map.get(asset)

        if total_value <= 0:
            logger.error("Portfolio value (margin) is zero or negative")
            return {
                "orders": pd.DataFrame(),
                "rebalancing": pd.DataFrame(),
                "total_value": 0.0,
            }

        adjusted_weights = {a: w * capital_at_risk for a, w in weights.items()}

        # Target positions: signed qty
        target_positions: dict[str, float] = {}
        for asset, weight in adjusted_weights.items():
            p = get_price(asset)
            if p is not None and p > 0:
                target_positions[asset] = (total_value * weight) / p

        rebalancing_df = self._build_rebalancing(
            current_holdings=current_holdings,
            target_positions=target_positions,
            get_price=get_price,
            total_value=total_value,
            strategy_weights=adjusted_weights,
            exclusions=exclusions,
        )

        orders_df = self._create_executable_orders(
            rebalancing_df=rebalancing_df,
            min_trade_size=min_trade_size,
        )

        return {
            "orders": orders_df,
            "rebalancing": rebalancing_df,
            "total_value": total_value,
        }

    # ------------------------------------------------------------------

    def _build_rebalancing(
        self,
        current_holdings: dict[str, float],
        target_positions: dict[str, float],
        get_price: Callable[[str], float | None],
        total_value: float,
        strategy_weights: dict[str, float],
        exclusions: list[str],
    ) -> pd.DataFrame:
        """Build the rebalancing analysis DataFrame with signed positions."""
        assets = sorted(set(current_holdings.keys()) | set(target_positions.keys()))
        assets = [a for a in assets if a not in exclusions]

        rows: list[dict[str, Any]] = []
        for asset in assets:
            current_qty = current_holdings.get(asset, 0.0)
            price = get_price(asset)
            # Signed current value
            current_value = current_qty * price if price else 0
            target_qty = target_positions.get(asset, 0.0)
            target_value = target_qty * price if price else 0
            current_weight = current_value / total_value if total_value > 0 else 0
            target_weight = target_value / total_value if total_value > 0 else 0
            weight_delta = strategy_weights.get(asset, 0) - current_weight
            delta_qty = target_qty - current_qty

            if delta_qty > 0:
                trade_action = "Buy"
            elif delta_qty < 0:
                trade_action = "Sell"
            else:
                trade_action = "Hold"

            rows.append(
                {
                    "Asset": asset,
                    "Current Quantity": current_qty,
                    "Current Value": current_value,
                    "Current Weight": current_weight,
                    "Target Quantity": target_qty,
                    "Target Value": target_value,
                    "Target Weight": target_weight,
                    "Weight Delta": weight_delta,
                    "Delta Quantity": delta_qty,
                    "Price": price,
                    "Trade Action": trade_action,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------

    def _create_executable_orders(
        self,
        rebalancing_df: pd.DataFrame,
        min_trade_size: float,
    ) -> pd.DataFrame:
        """Convert rebalancing to executable orders.

        For futures, no lot-size or step-size adjustments — the broker handles
        exchange-specific formatting. Symbols are base asset names.
        """
        if rebalancing_df.empty:
            return pd.DataFrame(
                columns=[
                    "Asset",
                    "Symbol",
                    "Action",
                    "Raw Quantity",
                    "Adjusted Quantity",
                    "Price",
                    "Notional Value",
                    "Order Status",
                    "Reason",
                    "Executable",
                ]
            )

        order_records: list[dict[str, Any]] = []

        for _, row in rebalancing_df.iterrows():
            asset = row["Asset"]
            symbol = asset  # base asset name, broker handles formatting
            action = row["Trade Action"].lower()
            delta_qty = row["Delta Quantity"]
            price = row["Price"]
            status = None
            reason = None
            adjusted_qty = abs(delta_qty)
            notional_value = adjusted_qty * price if price else 0.0

            if action == "hold" or delta_qty == 0:
                status = "Zero delta"
                reason = "No trade needed"
                adjusted_qty = 0.0
            elif abs(row["Weight Delta"]) < min_trade_size:
                zero_target = row.get("Target Weight", 0) == 0 and row.get("Current Quantity", 0) != 0
                if not zero_target:
                    status = "Below threshold"
                    reason = f"abs(weight delta) < {min_trade_size}"
                    adjusted_qty = 0.0
                else:
                    status = "To be placed"
                    reason = ""
            elif price is None or price == 0:
                status = "Zero price"
                reason = "No price available"
                adjusted_qty = 0.0
            else:
                status = "To be placed"
                reason = ""

            order_records.append(
                {
                    "Asset": asset,
                    "Symbol": symbol,
                    "Action": action.capitalize(),
                    "Raw Quantity": abs(delta_qty),
                    "Adjusted Quantity": adjusted_qty,
                    "Price": price,
                    "Notional Value": notional_value,
                    "Order Status": status,
                    "Reason": reason,
                }
            )

        order_df = pd.DataFrame(order_records)
        order_df["Executable"] = (order_df["Adjusted Quantity"] > 0) & (order_df["Order Status"] == "To be placed")
        return order_df
