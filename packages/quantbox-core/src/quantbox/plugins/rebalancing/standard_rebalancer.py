"""Standard rebalancing plugin.

Handles risk transforms (tranching, leverage cap, short clamping) and
order generation (rebalancing analysis, lot/step/notional validation,
buy scaling).

Ported from ``TradingPipeline._apply_risk_transforms()``,
``_generate_orders()``, ``_build_rebalancing()``, and
``_create_executable_orders()``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal, getcontext
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quantbox.contracts import BrokerPlugin, PluginMeta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CAPITAL_AT_RISK = 1.0
DEFAULT_MIN_NOTIONAL = 1.0
DEFAULT_MIN_TRADE_SIZE = 0.01
DEFAULT_STABLE_COIN = "USDC"


# ---------------------------------------------------------------------------
# Low-level helpers (from quantlab orders.py / portfolio.py)
# ---------------------------------------------------------------------------

def _adjust_quantity(qty: float, step_size: float) -> float:
    """Round *qty* down to the nearest *step_size* (Binance-style)."""
    if step_size <= 0:
        return qty
    step_str = f"{step_size:.8f}"
    decimal_places = step_str.rstrip("0").split(".")[-1]
    precision = len(decimal_places)
    getcontext().rounding = ROUND_DOWN
    return float(Decimal(qty).quantize(Decimal("1." + "0" * precision)))


def _get_lot_size_and_min_notional(
    symbol_info: Optional[Dict],
) -> Tuple[float, float, float]:
    """Extract (min_qty, step_size, min_notional) from Binance symbol info."""
    min_qty, step_size, min_notional = 0.0, 0.0, 0.0
    if not symbol_info:
        return min_qty, step_size, min_notional
    for f in symbol_info.get("filters", []):
        if f.get("filterType") == "LOT_SIZE":
            min_qty = float(f.get("minQty", 0))
            step_size = float(f.get("stepSize", 0))
        if f.get("filterType") == "NOTIONAL":
            min_notional = float(f.get("minNotional", 0))
    return min_qty, step_size, min_notional


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

@dataclass
class StandardRebalancer:
    """Standard rebalancing: risk transforms + order generation.

    ``generate_orders()`` params keys:
      - tranches, max_leverage, allow_short  (risk transforms)
      - min_trade_size, min_notional, scaling_factor_min  (order gen)
      - capital_at_risk, stable_coin_symbol, exclusions
      - strategy_results  (for tranching time-series)
    """

    meta = PluginMeta(
        name="rebalancing.standard.v1",
        kind="rebalancing",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Standard rebalancer: risk transforms + order generation",
        tags=("rebalancing", "trading"),
    )

    def generate_orders(
        self,
        *,
        weights: Dict[str, float],
        broker: BrokerPlugin,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        # 1. Risk transforms
        risk_adjusted = self._apply_risk_transforms(weights, params)

        # 2. Order generation
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
        weights: Dict[str, float],
        params: Dict[str, Any],
    ) -> Dict[str, float]:
        """Apply tranching, leverage cap, and negative-weight clamping."""
        tranches = int(params.get("tranches", 1))
        max_leverage = float(params.get("max_leverage", 1))
        allow_short = bool(params.get("allow_short", False))

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
                            account_weights.append(
                                float(weight_overrides.get(sname, sinfo["weight"]))
                            )

                    if len(weight_dfs) == 1:
                        full_ts = weight_dfs[0] * account_weights[0]
                    else:
                        combined = pd.concat(
                            weight_dfs,
                            axis=1,
                            keys=names[:len(weight_dfs)],
                            names=["strategy"],
                        )
                        acct_w = pd.Series(
                            account_weights,
                            index=pd.Index(names[:len(weight_dfs)], name="strategy"),
                        )
                        weighted = combined.mul(acct_w, level="strategy")
                        full_ts = weighted.droplevel(0, axis=1)
                        if isinstance(full_ts.columns, pd.MultiIndex) or full_ts.columns.duplicated().any():
                            full_ts = full_ts.T.groupby(level=0).sum().T

                    smoothed = full_ts.rolling(window=tranches).mean().iloc[-1]
                    s = smoothed
                except Exception:
                    logger.warning("Tranching failed, using un-smoothed weights")

        # Max leverage
        gross = s.abs().sum()
        if gross > max_leverage:
            logger.warning(
                "Leverage %.4f exceeds max_leverage %.1f, scaling down",
                gross, max_leverage,
            )
            s = s / gross * max_leverage

        # Clamp negatives
        if not allow_short:
            s = s.clip(lower=0)

        # Drop zeros and sort
        s = s[s != 0].sort_values(ascending=False)

        return {str(k): float(v) for k, v in s.items()}

    # ==================================================================
    # Order generation
    # ==================================================================
    def _generate_orders(
        self,
        broker: BrokerPlugin,
        weights: Dict[str, float],
        capital_at_risk: float,
        stable_coin: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate rebalancing DataFrame and executable orders."""
        min_notional_cfg = float(params.get("min_notional", DEFAULT_MIN_NOTIONAL))
        min_trade_size = float(params.get("min_trade_size", DEFAULT_MIN_TRADE_SIZE))
        scaling_factor_min = float(params.get("scaling_factor_min", 0.9))
        exclusions = list(params.get("exclusions", [])) + [stable_coin]
        exclusions = list(set(exclusions))

        positions_df = broker.get_positions()
        cash = broker.get_cash() or {}
        cash_available = float(cash.get(stable_coin, cash.get("USD", 0.0)))

        current_holdings: Dict[str, float] = {}
        if positions_df is not None and not positions_df.empty:
            for _, row in positions_df.iterrows():
                sym = str(row.get("symbol", ""))
                qty = float(row.get("qty", 0))
                if qty != 0:
                    current_holdings[sym] = qty

        all_symbols = sorted(set(list(weights.keys()) + list(current_holdings.keys())))
        all_symbols = [s for s in all_symbols if s not in exclusions]

        price_map: Dict[str, Optional[float]] = {}
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

        def get_price(asset: str) -> Optional[float]:
            return price_map.get(asset)

        total_value = max(0, cash_available)
        for asset, qty in current_holdings.items():
            if asset == stable_coin:
                total_value += max(0, qty)
            elif asset not in exclusions:
                p = get_price(asset)
                if p is not None and qty > 0:
                    total_value += qty * p

        if total_value <= 0:
            logger.error("Portfolio value is zero or negative")
            return {
                "orders": pd.DataFrame(),
                "rebalancing": pd.DataFrame(),
                "total_value": 0.0,
            }

        adjusted_weights = {a: w * capital_at_risk for a, w in weights.items()}

        target_positions: Dict[str, float] = {}
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
            broker=broker,
            stable_coin=stable_coin,
            min_notional_default=min_notional_cfg,
            min_trade_size=min_trade_size,
            cash_available=cash_available,
            scaling_factor_min=scaling_factor_min,
        )

        return {
            "orders": orders_df,
            "rebalancing": rebalancing_df,
            "total_value": total_value,
        }

    # ------------------------------------------------------------------

    def _build_rebalancing(
        self,
        current_holdings: Dict[str, float],
        target_positions: Dict[str, float],
        get_price: Callable[[str], Optional[float]],
        total_value: float,
        strategy_weights: Dict[str, float],
        exclusions: List[str],
    ) -> pd.DataFrame:
        """Build the rebalancing analysis DataFrame."""
        assets = sorted(set(current_holdings.keys()) | set(target_positions.keys()))
        assets = [a for a in assets if a not in exclusions]

        rows: List[Dict[str, Any]] = []
        for asset in assets:
            current_qty = current_holdings.get(asset, 0.0)
            price = get_price(asset)
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

            rows.append({
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
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------

    def _create_executable_orders(
        self,
        rebalancing_df: pd.DataFrame,
        broker: BrokerPlugin,
        stable_coin: str,
        min_notional_default: float,
        min_trade_size: float,
        cash_available: float,
        scaling_factor_min: float,
    ) -> pd.DataFrame:
        """Convert rebalancing to executable orders with lot/step/notional validation."""
        if rebalancing_df.empty:
            return pd.DataFrame(columns=[
                "Asset", "Symbol", "Action", "Raw Quantity", "Adjusted Quantity",
                "Price", "Notional Value", "Min Notional", "Min Qty", "Step Size",
                "Scaling Factor", "Order Status", "Reason", "Executable",
            ])

        order_records: List[Dict[str, Any]] = []

        for _, row in rebalancing_df.iterrows():
            asset = row["Asset"]
            symbol = f"{asset}{stable_coin}"
            action = row["Trade Action"].lower()
            delta_qty = row["Delta Quantity"]
            price = row["Price"]
            status = None
            reason = None
            adjusted_qty = 0.0
            notional_value = 0.0
            min_qty = 0.0
            step_size = 0.0
            min_notional = min_notional_default
            scaling_factor = None

            if action == "hold" or delta_qty == 0:
                status = "Zero delta"
                reason = "No trade needed"
            else:
                zero_target_sell = (
                    action == "sell"
                    and row.get("Target Weight", 0) == 0
                    and row.get("Current Quantity", 0) > 0
                )
                if abs(row["Weight Delta"]) < min_trade_size and not zero_target_sell:
                    status = "Below threshold"
                    reason = f"abs(weight delta) < {min_trade_size}"
                else:
                    try:
                        snap = broker.get_market_snapshot([asset])
                        if snap is not None and not snap.empty:
                            info_row = snap.iloc[0]
                            min_qty = float(info_row.get("min_qty", 0) or 0)
                            step_size = float(info_row.get("step_size", 0) or 0)
                            mn = info_row.get("min_notional")
                            if mn is not None and float(mn) > 0:
                                min_notional = float(mn)
                    except Exception:
                        pass

                    if action == "sell":
                        adjusted_qty = _adjust_quantity(abs(delta_qty), step_size) if step_size > 0 else abs(delta_qty)
                        notional_value = adjusted_qty * price if price else 0.0

                        if price is None or price == 0:
                            status = "Zero price"
                            reason = "No price available"
                            adjusted_qty = 0.0
                        elif notional_value < min_notional:
                            status = "Below min notional"
                            reason = f"Notional {notional_value:.4f} < min_notional {min_notional:.4f}"
                            adjusted_qty = 0.0
                        elif min_qty > 0 and adjusted_qty < min_qty:
                            status = "Below min qty"
                            reason = f"Qty {adjusted_qty:.8f} < min_qty {min_qty:.8f}"
                            adjusted_qty = 0.0
                        else:
                            status = "To be placed"
                            reason = ""
                    elif action == "buy":
                        status = "Pending scaling"
                        reason = ""

            order_records.append({
                "Asset": asset,
                "Symbol": symbol,
                "Action": action.capitalize(),
                "Raw Quantity": abs(delta_qty),
                "Adjusted Quantity": adjusted_qty,
                "Notional Value": notional_value,
                "Price": price,
                "Min Notional": min_notional,
                "Min Qty": min_qty,
                "Step Size": step_size,
                "Order Status": status,
                "Reason": reason,
                "Scaling Factor": scaling_factor,
            })

        # --- Buy scaling ---
        buy_indices = [
            i for i, rec in enumerate(order_records)
            if rec["Action"] == "Buy" and rec["Order Status"] == "Pending scaling"
        ]
        total_buy_value = sum(
            order_records[i]["Raw Quantity"] * (order_records[i]["Price"] or 0)
            for i in buy_indices
        )
        cash_from_sells = sum(
            rec["Notional Value"]
            for rec in order_records
            if rec["Action"] == "Sell" and rec["Order Status"] == "To be placed"
        )
        available_cash = cash_available + cash_from_sells
        scaling_factor = min(1.0, available_cash / total_buy_value) if total_buy_value > 0 else 0.0

        if total_buy_value > 0 and scaling_factor >= scaling_factor_min:
            for i in buy_indices:
                rec = order_records[i]
                price = rec["Price"]
                step_size = rec["Step Size"]
                min_qty_val = rec["Min Qty"]
                mn = rec["Min Notional"]
                raw_qty = rec["Raw Quantity"]

                scaled_qty = (
                    _adjust_quantity(raw_qty * scaling_factor, step_size)
                    if step_size and price
                    else raw_qty * scaling_factor
                )
                scaled_notional = scaled_qty * price if price else 0.0
                rec["Scaling Factor"] = scaling_factor
                rec["Adjusted Quantity"] = scaled_qty
                rec["Notional Value"] = scaled_notional

                if price is None or price == 0:
                    rec["Order Status"] = "Zero price"
                    rec["Reason"] = "No price available"
                    rec["Adjusted Quantity"] = 0.0
                elif scaled_notional < mn:
                    rec["Order Status"] = "Below min notional"
                    rec["Reason"] = f"Notional {scaled_notional:.4f} < min_notional {mn:.4f}"
                    rec["Adjusted Quantity"] = 0.0
                elif min_qty_val > 0 and scaled_qty < min_qty_val:
                    rec["Order Status"] = "Below min qty"
                    rec["Reason"] = f"Qty {scaled_qty:.8f} < min_qty {min_qty_val:.8f}"
                    rec["Adjusted Quantity"] = 0.0
                elif scaled_qty == 0:
                    rec["Order Status"] = "Zero quantity"
                    rec["Reason"] = "Scaled quantity is zero"
                else:
                    rec["Order Status"] = "To be placed"
                    rec["Reason"] = ""
        elif total_buy_value > 0:
            logger.warning(
                "Buy scaling factor %.4f < minimum %.2f, skipping buys",
                scaling_factor, scaling_factor_min,
            )

        columns = [
            "Asset", "Symbol", "Action", "Raw Quantity", "Adjusted Quantity",
            "Price", "Notional Value", "Min Notional", "Min Qty", "Step Size",
            "Scaling Factor", "Order Status", "Reason",
        ]
        order_df = pd.DataFrame(order_records, columns=columns)
        order_df["Executable"] = (order_df["Adjusted Quantity"] > 0) & (order_df["Order Status"] == "To be placed")
        return order_df
