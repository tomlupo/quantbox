"""Futures rebalancing plugin.

Handles order generation for perpetual-futures accounts where:
- Positions are signed (negative = short)
- Portfolio value equals margin balance (not cash + position value)
- Leverage cap applies to gross exposure (sum of abs weights)
- No short clamping: negative weights are respected
- Symbols use base asset names (broker handles exchange-specific formatting)

**The margin-balance valuation is no longer this plugin's to assume.** It used
to be, as a bare ``total_value = max(0, cash_available)`` justified by the file's
own first paragraph — and that is how the live ``crypto-trend-kraken`` book came
to size every target off cash: its config named ``rebalancing.futures.v1`` while
its broker was ``kraken.spot.v1``, a SPOT venue, and the code applied the futures
rule faithfully to a book where it is wrong by the whole value of the positions
held. Whether margin balance is equity is a fact about the VENUE, so it is asked
of the broker (``valuation_basis``) rather than assumed from this plugin's own
identity. Against a perps broker the answer is the same number as before; against
a spot broker it is now cash + holdings, and a live run against a broker that
does not say refuses rather than guessing. See :mod:`quantbox.trade.book`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import BrokerPlugin, PluginMeta
from quantbox.decision import gross_cap, risk_caps_row
from quantbox.engine.policy import TRANCHES_MOVED
from quantbox.trade.book import (
    BASIS_MARGIN,
    DEFAULT_RECONCILIATION_TOLERANCE,
    read_book,
)
from quantbox.trade.lots import SKIP_MIN_QTY, LotRules, lot_venue

logger = logging.getLogger(__name__)


def declares_open(current_qty: float, target_qty: float) -> bool:
    """The order's declared intent: True for an open from flat or an add on the same side.

    The one definition every builtin rebalancer writes as ``opens_exposure`` on
    an order row (TOM-1778, TOM-901). A flip through zero is NOT an open: it
    exits the held side first. The execution stage (:mod:`quantbox.trade.executor`)
    reads the flag instead of guessing intent from the order's side.
    """
    if target_qty == 0:
        return False
    if current_qty == 0:
        return True
    same_sign = (target_qty > 0) == (current_qty > 0)
    return bool(same_sign and abs(target_qty) > abs(current_qty))


#: ``Order Status`` of an exit the venue rejects at this size, so it is not sent:
#: a partial reduce below the venue minimum (TOM-1778), or a full close the
#: venue can never accept (TOM-2212, :meth:`quantbox.trade.lots.LotRules.never_accepts_close`).
#: A known residual, not a freeze. The ``Reason`` says which.
RESIDUAL_STATUS = "Residual below min notional"

DEFAULT_CAPITAL_AT_RISK = 1.0
DEFAULT_STABLE_COIN = "USDT"


def _is_nan(x: Any) -> bool:
    """True only for a float NaN. ``None`` (no price yet) is NOT NaN — it has
    its own ('Zero price') handling — so guard the type before ``isnan``."""
    return isinstance(x, float) and np.isnan(x)


@dataclass
class FuturesRebalancer:
    """Futures rebalancer: margin-based with signed positions.

    ``generate_orders()`` params keys:
      - max_leverage  (risk transforms — no allow_short, always allowed)
      - min_trade_size, capital_at_risk, stable_coin_symbol
      - tranches: deprecated, N > 1 refused (the policy's tranche cadence, TOM-1518)
    """

    meta = PluginMeta(
        name="rebalancing.futures.v1",
        kind="rebalancing",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Futures rebalancer: margin-based with signed position support",
        tags=("rebalancing", "futures", "trading"),
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
                    "default": "USDT",
                    "description": "Margin asset of the book (pipeline value if unset).",
                },
                "min_notional": {
                    "type": ["number", "null"],
                    "description": "Operator order floor (never bypassed); unset = the venue's per-pair minimum, else 10.",
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
        """The gross cap (:func:`quantbox.decision.risk_caps_row` with :meth:`risk_rules`). No short clamping.

        The decision's own step 2 (TOM-1520): trade.full_pipeline.v1 already
        applied it to the targets it hands in, so this changes nothing there.
        No tranching: the rebalancing policy owns WHEN and to WHICH targets
        (``quantbox.engine.policy``, TOM-1518). This plugin turns target
        weights into orders.
        """
        s = pd.Series(risk_caps_row(weights, **self.risk_rules(params)), dtype=float)
        # Drop zeros and sort by absolute value descending
        s = s[s != 0].reindex(s[s != 0].abs().sort_values(ascending=False).index)
        return {str(k): float(v) for k, v in s.items()}

    def risk_rules(self, params: dict[str, Any]) -> dict[str, Any]:
        """This order generator's caps, as the decision reads them (TOM-1520): no short clip, a gross cap."""
        if int(params.get("tranches", 1) or 1) > 1:
            raise ValueError(TRANCHES_MOVED)
        # NO short clamping — negative weights are valid for futures
        return {"allow_short": True, "max_leverage": gross_cap(params)}

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
        # Distinguish an EXPLICIT operator floor from the unset default. When the
        # operator set `min_notional` it is a deliberate churn/risk floor that must
        # never be silently bypassed (even below an exchange minimum); when unset,
        # the venue's per-pair floor governs. `min_notional` (numeric) is the
        # default-backstop used only when no per-pair snapshot is available.
        _mn_cfg = params.get("min_notional")
        min_notional_configured = float(_mn_cfg) if _mn_cfg is not None else None
        min_notional = float(_mn_cfg) if _mn_cfg is not None else 10.0
        exclusions = list(params.get("exclusions", [])) + [stable_coin]
        exclusions = list(set(exclusions))

        # The marked book: signed positions, the margin/cash balance, and one
        # snapshot of marks for the targets and the held names. Whether the
        # balance IS the portfolio value is the venue's answer, not this
        # plugin's -- resolved below by the book's valuation.
        quote = stable_coin
        book = read_book(
            broker,
            wanted=weights.keys(),
            exclude=exclusions,
            cash=(quote, "USD", "USDC", "USDT"),
        )
        cash_available = float(book.cash or 0.0)

        # Per-pair exchange minimums from the same snapshot. It carries the
        # venue's REAL per-symbol floors (Kraken: costmin via limits.cost.min,
        # base-unit ordermin via limits.amount.min). Capturing them here is what
        # lets a small book (e.g. the $278 Kraken-USD book) trade $4-8 deltas that
        # a flat $10 min_notional would freeze — while still respecting each
        # pair's true ordermin so ADA/DOGE-style sub-base-unit orders stay gated.
        min_notional_map: dict[str, float] = {}
        min_qty_map: dict[str, float] = {}
        # The venue's lot rules per symbol, from the same snapshot row and the
        # venue the broker declares: the rules the broker sends with (TOM-2207).
        venue = lot_venue(broker)
        lot_rules: dict[str, LotRules] = {}
        snap = book.snapshot
        if snap is not None and not snap.empty:
            for _, row in snap.iterrows():
                sym = str(row.get("symbol", ""))
                mn = row.get("min_notional")
                if mn is not None and not _is_nan(float(mn)) and float(mn) > 0:
                    min_notional_map[sym] = float(mn)
                mq = row.get("min_qty")
                if mq is not None and not _is_nan(float(mq)) and float(mq) > 0:
                    min_qty_map[sym] = float(mq)
                lot_rules[sym] = LotRules.from_snapshot(row.to_dict(), venue=venue)

        # Portfolio value, per the VENUE's rule rather than this plugin's name.
        # `mode` is threaded by `trade.full_pipeline.v1`; a caller that omits it
        # is gated, so a forgotten mode fails closed (book.is_gated).
        mode = str(params.get("mode") or "")
        valuation = book.valuation(
            broker=broker,
            mode=mode,
            # Before venue declarations existed this call site always used the
            # margin balance. That stays the answer for a SIMULATION against an
            # undeclared broker; on live an undeclared broker refuses.
            fallback_basis=BASIS_MARGIN,
            stable_coin=stable_coin,
            exclusions=exclusions,
            tolerance=float(params.get("equity_reconciliation_tolerance", DEFAULT_RECONCILIATION_TOLERANCE)),
            require_reconciliation=bool(params.get("require_equity_reconciliation", True)),
        )
        total_value = valuation.value

        if total_value <= 0:
            raise RuntimeError(
                f"Portfolio value is zero or negative (cash={cash_available!r}, "
                f"{valuation.n_holdings} holding(s), basis={valuation.basis}, state={valuation.state}). "
                "Check broker connectivity and account balance."
            )

        adjusted_weights = {a: w * capital_at_risk for a, w in weights.items()}

        # Target positions: signed qty
        target_positions: dict[str, float] = {}
        for asset, weight in adjusted_weights.items():
            p = book.mark(asset)
            if p is not None:
                target_positions[asset] = (total_value * weight) / p

        # Signed quantities and values: the book's one frame builder (TOM-1859).
        rebalancing_df = book.rebalancing_frame(
            target_positions=target_positions,
            total_value=total_value,
            strategy_weights=adjusted_weights,
            exclusions=exclusions,
        )

        orders_df = self._create_executable_orders(
            rebalancing_df=rebalancing_df,
            min_trade_size=min_trade_size,
            min_notional=min_notional,
            min_notional_configured=min_notional_configured,
            min_notional_map=min_notional_map,
            min_qty_map=min_qty_map,
            lot_rules=lot_rules,
            default_lot_rules=LotRules.from_snapshot({}, venue=venue),
        )

        return {
            "orders": orders_df,
            "rebalancing": rebalancing_df,
            "total_value": total_value,
            # Carried so the run record can say WHICH rule valued this book and
            # what it could not mark, rather than only the resulting number.
            "valuation": valuation,
        }

    # ------------------------------------------------------------------

    def _create_executable_orders(
        self,
        rebalancing_df: pd.DataFrame,
        min_trade_size: float,
        min_notional: float = 10.0,
        min_notional_configured: float | None = None,
        min_notional_map: dict[str, float] | None = None,
        min_qty_map: dict[str, float] | None = None,
        lot_rules: dict[str, LotRules] | None = None,
        default_lot_rules: LotRules | None = None,
    ) -> pd.DataFrame:
        """Convert rebalancing to executable orders.

        Symbols are base asset names; the broker formats them for the venue.

        The quantity is the one the venue gets (TOM-2207). ``lot_rules`` holds
        each symbol's :class:`quantbox.trade.lots.LotRules`, built from the
        broker's snapshot. A symbol without a row takes ``default_lot_rules``;
        without those, the quantity is not rounded. The lot rules round the
        quantity toward zero to the step BEFORE every floor below. So the floors
        judge the rounded quantity, and ``Adjusted Quantity`` is the quantity the
        broker sends. A row the lot rules refuse is skipped here with its reason.
        A full close the venue can never accept (it rounds to zero, or it is
        below the minimum quantity on a venue without a reduce-only exemption)
        is ``RESIDUAL_STATUS`` (TOM-2212). Any other quantity that rounds to zero
        is ``Zero quantity``. A close below the minimum quantity on a broker
        that declares no venue is ``Below min qty``. The broker keeps its own
        lot-rules check. Before TOM-2207 this
        method planned the unrounded ``abs(delta)`` and the broker rounded it.
        So the order frame showed orders the broker shrank or dropped.

        Per-pair minimums (``min_notional_map`` / ``min_qty_map``, from the broker
        snapshot) give the venue's true binding floor ``max(costmin, ordermin*price)``:

          * ``costmin`` = the venue's quote-notional minimum (Kraken limits.cost.min,
            ~$0.50 for USD pairs),
          * ``ordermin`` = the venue's base-unit minimum (Kraken limits.amount.min);
            ``ordermin * price`` is its notional equivalent.

        How that combines with the config ``min_notional`` (additive, never a silent
        bypass — codex review of #106):

          * ``min_notional_configured is not None`` — the operator EXPLICITLY set a
            churn/risk floor. It is honored as an ADDITIONAL floor:
            ``effective = max(configured, per_pair_min)``. A deliberate $50 floor is
            never bypassed just because Kraken's own minimum is lower.
          * ``min_notional_configured is None`` (unset) — the venue's per-pair floor
            GOVERNS. This is what unfreezes a small book whose $4-8 deltas clear the
            real per-pair floors but were suppressed by the flat $10 *default*.
          * No per-pair floor available (empty snapshot / offline) AND no explicit
            config — fall back to the flat ``min_notional`` default backstop.
        """
        min_notional_map = min_notional_map or {}
        min_qty_map = min_qty_map or {}
        lot_rules = lot_rules or {}
        default_lot_rules = default_lot_rules or LotRules()
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
            # NOTE: `adjusted_qty` / `notional_value` are derived AFTER the
            # reduce classification below, because a partial reduce is clamped to
            # the position size first. Every floor check and every reported column
            # must see the same quantity — the one actually sent.

            # Resolve the binding per-pair minimums. Prefer the venue's real
            # floors from the snapshot; fall back to the flat config min_notional
            # only when no per-pair floor is known for this symbol.
            costmin = float(min_notional_map.get(asset, 0.0) or 0.0)
            ordermin = float(min_qty_map.get(asset, 0.0) or 0.0)
            ordermin_notional = ordermin * price if (price and ordermin > 0) else 0.0
            per_pair_min = max(costmin, ordermin_notional)
            if min_notional_configured is not None:
                # Explicit operator floor — an ADDITIONAL floor, never bypassed.
                effective_min_notional = max(min_notional_configured, per_pair_min)
            elif per_pair_min > 0:
                # No explicit floor: the venue's real per-pair minimum governs.
                effective_min_notional = per_pair_min
            else:
                # No explicit floor and no per-pair snapshot: default backstop.
                effective_min_notional = min_notional

            # A position-flattening order: the target is flat (weight 0) but we
            # still hold the asset. On a futures/perps venue a closing order is
            # sent REDUCE-ONLY, which the venue exempts from its min-notional
            # floor and which can never flip the position through zero. So a close
            # is ALWAYS sendable — exempt from our churn bands AND the venue floor.
            #
            # This corrects the quantbox#87 story: carver-HL's sub-$10 ETH/SOL/ARB
            # closes were NOT rejected by Hyperliquid. They were rejected by our
            # own client-side min-notional guard before submission, and the
            # "placement failed" error string made it look like the venue's doing.
            # Verified live 2026-07-22: HL accepts a full reduce-only close below
            # its $10 minimum (SOL 0.03, $2.33, filled). The exit was available
            # the whole time — the earlier "trapped residual" suppression was built
            # on that false premise and is removed.
            # TOM-402 adds a SECOND, deliberately narrower category: a PARTIAL
            # reduce (same sign, |target| < |current|, e.g. -0.40 -> -0.10).
            #
            # These two are NOT interchangeable, and the difference is why they get
            # different exemptions. A flat-target close is a one-shot, terminal act
            # of intent. A partial reduce is mostly PRICE-DRIVEN: target quantity is
            # `total_value * weight / price` (line 233), so for an unchanged target
            # weight any asset that appreciates relative to the book gets a smaller
            # same-sign target. On a diversified book that describes the winning
            # names on essentially every run — not a rare event.
            #
            # So a partial reduce earns exactly ONE exemption, the `reduce_only`
            # flag itself (it can never flip the position through zero), and NOT
            # the other three:
            #   * the min-notional floor — TOM-402 exempted it on the premise that
            #     the venue waives the floor for reduceOnly. Only a FULL close was
            #     ever verified (SOL, 2026-07-22). Hyperliquid REJECTS a partial
            #     reduce below $10: "Order must have minimum value of $10" on ETH
            #     $6.33 and NEAR $2.63, every carver-HL run since 2026-10-09
            #     (TOM-1778). Such a reduce is now RESIDUAL_STATUS: not sent, and
            #     a known residual the dead-man freeze does not count (Tom,
            #     2026-10-12: "Skip as residual", "Residual is not a freeze").
            #   * the min_trade_size churn band — exempting a recurring, price-
            #     driven trim while adds stay banded would trim winners at any size
            #     and never re-add, drifting the book to under-exposure and paying
            #     the fees the band exists to prevent.
            #   * the `ordermin` base-unit floor — venues do NOT waive lot size or
            #     step size for reduceOnly (binance_futures.py:487 adjust_quantity,
            #     hyperliquid.py place_order precision floor, neither with a reduce_only
            #     escape). Exempting it would mark the row Executable, then have the
            #     broker silently return None — the exact send-and-vanish failure
            #     the floor exists to prevent.
            cur_qty = row.get("Current Quantity", 0)
            tgt_qty = row.get("Target Quantity")
            if _is_nan(cur_qty) or tgt_qty is None or _is_nan(tgt_qty):
                # Fail closed: a missing or NaN target is never read as intent to
                # reduce. NaN is surfaced loudly below.
                is_flat_target = row.get("Target Weight", 0) == 0 and cur_qty != 0 and not _is_nan(cur_qty)
                is_partial_reduce = False
                opens_exposure = False
            else:
                is_flat_target = row.get("Target Weight", 0) == 0 and cur_qty != 0
                # Sign comparison rather than a product: no multiply, and it reads
                # as the question being asked.
                same_sign = (tgt_qty > 0) == (cur_qty > 0)
                is_partial_reduce = cur_qty != 0 and same_sign and abs(tgt_qty) < abs(cur_qty)
                # An open from flat or an add on the same side: the only rows a
                # residual day may leave unsent (TOM-1778), and the declared intent
                # the trapped-residual counter reads (TOM-901). A flip through zero
                # is NOT one — it exits the held side.
                opens_exposure = declares_open(cur_qty, tgt_qty)

            # Sent reduce-only.
            is_exposure_reducing = bool(is_flat_target or is_partial_reduce)
            # Exempt from the min-notional floor, the churn band and the base-unit
            # floor — closes only.
            is_closing = bool(is_flat_target)

            adjusted_qty = abs(delta_qty)
            if is_partial_reduce:
                # Enforce, do not merely assert, the no-zero-crossing invariant,
                # and do it HERE — above the gate chain — so the churn band, the
                # min-notional floor, the base-unit floor and the reported
                # `Notional Value` all judge the quantity we actually send. The
                # invariant holds arithmetically today because delta is
                # `target - current` from the same two columns (line 285), but
                # this method accepts those columns independently and nothing else
                # checks them for mutual consistency. Mirrors the gate-path clamp
                # in quantbox.trade.reconciliation.cycle.filter_reduce_only.
                clamped_qty = min(adjusted_qty, abs(cur_qty))
                if clamped_qty != adjusted_qty:
                    logger.warning(
                        "Clamping partial reduce for %s: delta %.8f exceeds position %.8f — "
                        "inconsistent rebalancing frame, check the target/current columns.",
                        asset,
                        adjusted_qty,
                        abs(cur_qty),
                    )
                    adjusted_qty = clamped_qty
            # The venue's lot rules round the clamped quantity toward zero to the
            # step (TOM-2207). This also runs above the gate chain. So every floor
            # below and every reported column judge the quantity the broker sends.
            # No price: the notional floor stays this method's own, below.
            rules = lot_rules.get(asset, default_lot_rules)
            unrounded_qty = adjusted_qty
            lot = rules.accept(unrounded_qty, None, reduce_only=is_exposure_reducing)
            adjusted_qty = abs(lot.rounded)
            notional_value = adjusted_qty * price if price else 0.0

            # Guard against NaN / missing-data targets. A NaN price or quantity
            # (e.g. a Hyperliquid missing-candle glitch) otherwise slips through
            # as a silent no-op: `NaN < min_notional` is False and `NaN > 0` is
            # False, so the order vanishes with no signal. Surface it loudly.
            nan_inputs = _is_nan(delta_qty) or _is_nan(price) or _is_nan(notional_value)

            if nan_inputs:
                status = "Invalid (NaN)"
                reason = "NaN price/quantity from data feed (missing candle?)"
                adjusted_qty = 0.0
                logger.error(
                    "Rebalancer produced NaN target for %s (delta=%r price=%r) — "
                    "skipping and flagging; check data feed for missing candles.",
                    asset,
                    delta_qty,
                    price,
                )
            elif action == "hold" or delta_qty == 0:
                status = "Zero delta"
                reason = "No trade needed"
                adjusted_qty = 0.0
            elif abs(row["Weight Delta"]) < min_trade_size and not is_closing:
                status = "Below threshold"
                reason = f"abs(weight delta) < {min_trade_size}"
                adjusted_qty = 0.0
            elif price is None or price == 0:
                status = "Zero price"
                reason = "No price available"
                adjusted_qty = 0.0
            elif notional_value < effective_min_notional and not is_exposure_reducing:
                status = "Below min notional"
                reason = f"Notional {notional_value:.2f} < {effective_min_notional:.2f}"
                adjusted_qty = 0.0
                logger.debug(
                    "Skipping %s %s: notional $%.2f < $%.2f (per-pair min)",
                    action,
                    asset,
                    notional_value,
                    effective_min_notional,
                )
            elif notional_value < effective_min_notional and not is_closing:
                # A partial reduce below the floor: the venue rejects it (TOM-1778),
                # so it is not sent. It stays reduce-only in the record and is a
                # known residual, which the dead-man freeze does not count.
                status = RESIDUAL_STATUS
                if ordermin > 0 and adjusted_qty < ordermin:
                    # The base-unit floor binds: name it, not a derived notional.
                    shortfall = f"qty {adjusted_qty:.8f} < min_qty {ordermin:.8f}"
                else:
                    shortfall = f"{notional_value:.2f} < {effective_min_notional:.2f}"
                if min_notional_configured is not None and notional_value >= per_pair_min:
                    blocker = "the configured min_notional blocks it"
                else:
                    blocker = "the venue rejects it"
                reason = f"Partial reduce {shortfall}: {blocker}, residual retained"
                adjusted_qty = 0.0
                logger.info(
                    "Residual %s %s: partial reduce $%.2f < $%.2f venue minimum — not sent, residual retained.",
                    action,
                    asset,
                    notional_value,
                    effective_min_notional,
                )
            elif ordermin > 0 and adjusted_qty < ordermin and not is_closing:
                # Base-unit floor: even if the notional clears, an order below the
                # venue's ordermin (e.g. ADA/DOGE sub-base-unit size) is rejected
                # by the exchange. Suppress it cleanly rather than send-and-reject.
                status = "Below min qty"
                reason = f"Qty {adjusted_qty:.8f} < min_qty {ordermin:.8f}"
                logger.debug("Skipping %s %s: qty %.8f < ordermin %.8f", action, asset, adjusted_qty, ordermin)
                adjusted_qty = 0.0
            elif lot.skip:
                # The venue's lot rules refuse what is left: the broker would
                # drop it (TOM-2207). Its reason goes on the row, before the broker.
                if is_closing and rules.never_accepts_close(lot):
                    # A full close the venue can never accept: dust below the
                    # venue's minimum quantity on a venue with no reduce-only
                    # exemption (Kraken: ADA 0.0088 vs min 20), or a quantity
                    # that rounds to zero at the venue's step. No order can exit
                    # this position, today or on a later day. It is a known
                    # residual, not a freeze (Tom, 2026-10-23, TOM-2212, which
                    # extends "Residual is not a freeze", TOM-1778).
                    status = RESIDUAL_STATUS
                    if lot.skip == SKIP_MIN_QTY:
                        # The held quantity, not the rounded one: the reader
                        # matches it against the balance.
                        shortfall = f"qty {unrounded_qty:.8f} < min_qty {rules.min_qty:.8f}, no reduce-only exemption"
                    else:
                        shortfall = f"qty {unrounded_qty:.8f} {lot.skip} (step {rules.step_size:g})"
                    reason = f"Close {shortfall}: the venue rejects it, residual retained"
                elif lot.skip == SKIP_MIN_QTY:
                    # Only a close reaches here: the base-unit floor above judges
                    # every other row. And only on a broker that declares no
                    # venue: without the venue's facts, the missing reduce-only
                    # exemption is not proven, so the close stays suppressed and
                    # the dead-man still counts it (TOM-2212).
                    status = "Below min qty"
                    reason = (
                        f"Close qty {adjusted_qty:.8f} < min_qty {rules.min_qty:.8f}: no reduce-only exemption here"
                    )
                else:
                    status = "Zero quantity"
                    reason = f"Qty {unrounded_qty:.8f} {lot.skip} (step {rules.step_size:g})"
                logger.info("Skipping %s %s: %s", action, asset, reason)
                adjusted_qty = 0.0
            else:
                status = "To be placed"
                if is_closing and notional_value < effective_min_notional:
                    reason = "Reducing position (min-notional exempt)"
                else:
                    reason = ""
                # (The partial-reduce clamp runs above the gate chain, so
                # `adjusted_qty` and `notional_value` are already final here.)

            order_records.append(
                {
                    "Asset": asset,
                    "Symbol": symbol,
                    "Action": action.capitalize(),
                    "Raw Quantity": abs(delta_qty),
                    "Adjusted Quantity": adjusted_qty,
                    "Price": price,
                    "Notional Value": notional_value,
                    # The floor this row was judged against, and the venue's
                    # base-unit floor: the same columns standard_rebalancer
                    # writes. Without them the pipeline's suppressed-order
                    # records (residuals, TOM-1778; frozen legs) carry
                    # min_notional_usd 0 and no shortfall, so no reader can say
                    # how far a leg is from tradable (TOM-1820). A full close is
                    # exempt from the floor (`is_closing`, above), so its row
                    # says 0.0: a failed small close must not read as a leg that
                    # was short of a minimum it never had to meet.
                    "Min Notional": 0.0 if is_closing else effective_min_notional,
                    "Min Qty": ordermin,
                    # The step `Adjusted Quantity` was rounded to (TOM-2207); 0.0
                    # when no step is known.
                    "Step Size": rules.step_size,
                    "Order Status": status,
                    "Reason": reason,
                    # An exposure-reducing order (full close or partial reduce) is
                    # sent reduce-only: it can't flip through zero. Note this is a
                    # WIDER set than `is_closing`, which additionally waives the
                    # min-notional floor, the churn band and the base-unit floor and
                    # so stays closes-only.
                    "reduce_only": is_exposure_reducing,
                    # True only for an open from flat or a same-side add (see above);
                    # the dead-man reads it to tell a residual day from a trapped book.
                    "opens_exposure": opens_exposure,
                    # True for a full close (flat target, position held). This
                    # rebalancer exempts every full close from the churn band, so a
                    # band row that declares False is never a held-back close: the
                    # dead-man reads it to clear band rows as untrapped (TOM-2212).
                    "full_close": is_closing,
                }
            )

        order_df = pd.DataFrame(order_records)
        order_df["Executable"] = (order_df["Adjusted Quantity"] > 0) & (order_df["Order Status"] == "To be placed")
        return order_df
