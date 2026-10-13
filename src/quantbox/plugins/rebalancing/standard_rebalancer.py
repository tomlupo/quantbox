"""Standard rebalancing plugin.

Handles risk transforms (leverage cap, short clamping) and order generation
(rebalancing analysis, lot/step/notional validation, buy scaling): target
weights in, orders out. WHEN to rebalance and to WHICH targets (the tranche
cadence, the trigger, ``min_trade``) is the rebalancing policy's, decided by
the backtest's own code (``quantbox.engine.policy``, TOM-1518).

Since TOM-1458 it is the trading pipeline's order generator for a spot book:
the pipeline's private copy is deleted, and a paper or live config names
``plugins.rebalancing``. Step rounding is :mod:`quantbox.trade.lots`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from quantbox.contracts import BrokerPlugin, PluginMeta
from quantbox.decision import gross_cap, risk_caps_row
from quantbox.engine.policy import TRANCHES_MOVED
from quantbox.plugins.rebalancing.futures_rebalancer import declares_open
from quantbox.trade.book import (
    BASIS_MARK,
    DEFAULT_RECONCILIATION_TOLERANCE,
    MarkedBook,
    nan_if_unknown,
    read_book,
)
from quantbox.trade.lots import floor_to_step

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CAPITAL_AT_RISK = 1.0
DEFAULT_MIN_NOTIONAL = 1.0
DEFAULT_MIN_TRADE_SIZE = 0.01
DEFAULT_STABLE_COIN = "USDC"


def _opens_exposure(row: pd.Series) -> bool:
    """The row's declared intent (:func:`declares_open`); an unknown quantity is no open."""
    cur, tgt = row.get("Current Quantity", 0.0), row.get("Target Quantity", 0.0)
    if cur is None or tgt is None or pd.isna(cur) or pd.isna(tgt):
        return False
    return declares_open(float(cur), float(tgt))


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


@dataclass
class StandardRebalancer:
    """Standard rebalancing: risk transforms + order generation.

    ``generate_orders()`` params keys:
      - max_leverage, allow_short  (risk transforms)
      - min_trade_size, min_notional, scaling_factor_min  (order gen)
      - capital_at_risk, stable_coin_symbol, exclusions
      - tranches: deprecated, N > 1 refused (the policy's tranche cadence)
    """

    meta = PluginMeta(
        name="rebalancing.standard.v1",
        kind="rebalancing",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Standard rebalancer: risk transforms + order generation",
        tags=("rebalancing", "trading"),
        params_schema={
            "type": "object",
            "properties": {
                "capital_at_risk": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Fraction of portfolio value the targets are sized against (pipeline value if unset).",
                },
                "tranches": {
                    "type": "integer",
                    "default": 1,
                    "description": (
                        "DEPRECATED (TOM-1518): trade.full_pipeline.v1 reads N > 1 as rebalancing_policy "
                        "{cadence: tranche, tranches: N}; this plugin refuses N > 1 itself (no tranching here)."
                    ),
                },
                "max_leverage": {
                    "type": "number",
                    "default": 1,
                    "description": "Cap on gross leverage of the targets.",
                },
                "min_trade_size": {
                    "type": "number",
                    "default": 0.01,
                    "description": "Minimum abs(weight delta) to trade.",
                },
                "exclusions": {
                    "type": "array",
                    "default": [],
                    "description": "Assets never traded (pipeline value if unset).",
                },
                "equity_reconciliation_tolerance": {
                    "type": "number",
                    "default": 0.005,
                    "description": "Relative tolerance of the live pre-trade reconciliation against broker equity.",
                },
                "require_equity_reconciliation": {
                    "type": "boolean",
                    "default": True,
                    "description": "Require the live pre-trade reconciliation against broker equity.",
                },
                "stable_coin_symbol": {
                    "type": "string",
                    "default": "USDC",
                    "description": "Cash asset of the book (pipeline value if unset).",
                },
                "allow_short": {
                    "type": "boolean",
                    "default": False,
                    "description": "False clips negative targets to zero.",
                },
                "min_notional": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Orders below this notional are dropped.",
                },
                "scaling_factor_min": {
                    "type": "number",
                    "default": 0.9,
                    "description": "Lowest scale-down applied to buys when cash is short.",
                },
            },
        },
    )

    def generate_orders(
        self,
        *,
        weights: dict[str, float],
        broker: BrokerPlugin,
        params: dict[str, Any],
    ) -> dict[str, Any]:
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
        weights: dict[str, float],
        params: dict[str, Any],
    ) -> dict[str, float]:
        """The short clip, then the gross cap: :func:`quantbox.decision.risk_caps_row` with :meth:`risk_rules`.

        The decision's own steps 1-2, in the decision's order (TOM-1520: the
        clip used to run after the cap here, the other way round from the
        backtest). trade.full_pipeline.v1 already applied them to the targets
        it hands in, so this changes nothing there; a direct caller gets the
        same caps. No tranching: the rebalancing policy owns WHEN and to WHICH
        targets (``quantbox.engine.policy``, TOM-1518). This plugin turns
        target weights into orders.
        """
        caps = self.risk_rules(params)
        s = pd.Series(risk_caps_row(weights, **caps), dtype=float)
        s = s[s != 0].sort_values(ascending=False)
        return {str(k): float(v) for k, v in s.items()}

    def risk_rules(self, params: dict[str, Any]) -> dict[str, Any]:
        """This order generator's short clip and gross cap, as the decision reads them (TOM-1520)."""
        if int(params.get("tranches", 1) or 1) > 1:
            raise ValueError(TRANCHES_MOVED)
        return {
            "allow_short": bool(params.get("allow_short", False)),
            "max_leverage": gross_cap(params),
        }

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
        min_notional_cfg = float(params.get("min_notional", DEFAULT_MIN_NOTIONAL))
        min_trade_size = float(params.get("min_trade_size", DEFAULT_MIN_TRADE_SIZE))
        scaling_factor_min = float(params.get("scaling_factor_min", 0.9))
        exclusions = list(params.get("exclusions", [])) + [stable_coin]
        exclusions = list(set(exclusions))

        # The marked book: positions, quote cash and one snapshot of marks for
        # the targets and the held names. A failed snapshot leaves the held
        # names UNMARKED, and the valuation below refuses on them (live) rather
        # than collapsing the portfolio value to cash.
        book = read_book(broker, wanted=weights.keys(), exclude=exclusions, cash=(stable_coin, "USD"))
        cash_available = float(book.cash or 0.0)

        # The run mode drives the pre-trade valuation gate. The production
        # caller (`trade.full_pipeline.v1`) always threads it through. A caller
        # that omits it is GATED, not treated as paper: only a run that
        # positively declares itself a simulation is exempt, so a forgotten
        # `mode` fails closed rather than quietly ungating a live book. See
        # quantbox.trade.book.UNGATED_MODES.
        mode = str(params.get("mode") or "")

        valuation = book.valuation(
            broker=broker,
            mode=mode,
            # This rebalancer is the spot/cash path, so an undeclared broker in
            # a SIMULATION keeps being marked the way this call site always
            # marked it. On live an undeclared broker refuses instead.
            fallback_basis=BASIS_MARK,
            stable_coin=stable_coin,
            exclusions=exclusions,
            tolerance=float(params.get("equity_reconciliation_tolerance", DEFAULT_RECONCILIATION_TOLERANCE)),
            require_reconciliation=bool(params.get("require_equity_reconciliation", True)),
        )
        total_value = valuation.value

        if total_value <= 0:
            logger.error(
                "Portfolio value is zero or negative (cash=%.2f, %d holding(s), state=%s)",
                valuation.cash,
                valuation.n_holdings,
                valuation.state,
            )
            return {
                "orders": pd.DataFrame(),
                "rebalancing": pd.DataFrame(),
                "total_value": 0.0,
                "valuation": valuation,
            }

        adjusted_weights = {a: w * capital_at_risk for a, w in weights.items()}

        target_positions: dict[str, float] = {}
        for asset, weight in adjusted_weights.items():
            p = book.mark(asset)
            if p is not None:
                target_positions[asset] = (total_value * weight) / p

        rebalancing_df = self._build_rebalancing(
            book=book,
            target_positions=target_positions,
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
            # Carried on the SUCCESS exit too, not only on the zero-value one
            # above. `trading_pipeline` reads `order_result.get("valuation")`,
            # so dropping it here made every HEALTHY run record
            # `portfolio_valuation_state='unknown'` and seven None metrics --
            # this repo's "could not be MEASURED" signal -- while a broken run
            # recorded a state. That inverts the property this change exists to
            # establish. `FuturesRebalancer` already carries it on both.
            "valuation": valuation,
        }

    # ------------------------------------------------------------------

    def _build_rebalancing(
        self,
        book: MarkedBook,
        target_positions: dict[str, float],
        total_value: float,
        strategy_weights: dict[str, float],
        exclusions: list[str],
    ) -> pd.DataFrame:
        """Build the rebalancing analysis DataFrame.

        Values come from the marked book. A held name with no mark has an
        UNKNOWN current value and weight (NaN), never 0 (TOM-1455).
        """
        current_holdings = book.held
        assets = sorted(set(current_holdings.keys()) | set(target_positions.keys()))
        assets = [a for a in assets if a not in exclusions]

        rows: list[dict[str, Any]] = []
        for asset in assets:
            current_qty = current_holdings.get(asset, 0.0)
            price = book.mark(asset)
            current_value = nan_if_unknown(book.value_of(asset))
            target_qty = target_positions.get(asset, 0.0)
            target_value = nan_if_unknown(book.value_of(asset, target_qty))
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
        broker: BrokerPlugin,
        stable_coin: str,
        min_notional_default: float,
        min_trade_size: float,
        cash_available: float,
        scaling_factor_min: float,
    ) -> pd.DataFrame:
        """Convert rebalancing to executable orders with lot/step/notional validation."""
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
                    "Min Notional",
                    "Min Qty",
                    "Step Size",
                    "Scaling Factor",
                    "Order Status",
                    "Reason",
                    "Executable",
                ]
            )

        order_records: list[dict[str, Any]] = []

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
            elif price is None or pd.isna(price) or price == 0:
                # No mark: the held name's value and weight are UNKNOWN (NaN, the
                # marked book), so no threshold, notional or scaling can judge
                # it. Before TOM-1455 its weight read as 0 -- a short slipped into
                # buy scaling with a NaN price, and a long close was "To be
                # placed" at a NaN price.
                status = "Zero price"
                reason = "No price available"
            else:
                zero_target_sell = (
                    action == "sell" and row.get("Target Weight", 0) == 0 and row.get("Current Quantity", 0) > 0
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
                        adjusted_qty = floor_to_step(abs(delta_qty), step_size)
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

            order_records.append(
                {
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
                    # The order's declared intent (TOM-901): the execution stage
                    # reads it instead of guessing intent from the side.
                    "opens_exposure": _opens_exposure(row),
                }
            )

        # --- Buy scaling ---
        buy_indices = [
            i
            for i, rec in enumerate(order_records)
            if rec["Action"] == "Buy" and rec["Order Status"] == "Pending scaling"
        ]
        total_buy_value = sum(order_records[i]["Raw Quantity"] * (order_records[i]["Price"] or 0) for i in buy_indices)
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

                scaled_qty = floor_to_step(raw_qty * scaling_factor, step_size) if price else raw_qty * scaling_factor
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
                scaling_factor,
                scaling_factor_min,
            )

        columns = [
            "Asset",
            "Symbol",
            "Action",
            "Raw Quantity",
            "Adjusted Quantity",
            "Price",
            "Notional Value",
            "Min Notional",
            "Min Qty",
            "Step Size",
            "Scaling Factor",
            "Order Status",
            "Reason",
            "opens_exposure",
        ]
        order_df = pd.DataFrame(order_records, columns=columns)
        order_df["Executable"] = (order_df["Adjusted Quantity"] > 0) & (order_df["Order Status"] == "To be placed")
        return order_df
