from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quantbox.contracts import ArtifactStore, BrokerPlugin, DataPlugin, Mode, PluginMeta, RiskPlugin, RunResult
from quantbox.parquet_io import read_parquet
from quantbox.portfolio_value import (
    BASIS_MARK,
    DEFAULT_RECONCILIATION_TOLERANCE,
    PortfolioValuation,
    resolve_portfolio_value,
    value_holdings,
)
from quantbox.run_history import resolve_latest_artifact

logger = logging.getLogger(__name__)

# --------------------------
# Helpers
# --------------------------


def _adv_usd(
    volume: pd.DataFrame | None,
    latest_px: pd.DataFrame,
    lookback: int,
    volume_is_dollar: bool,
) -> pd.Series:
    """Average daily volume in USD per symbol from a wide volume frame.

    ``volume`` is wide (date index, symbol columns). When *volume_is_dollar* the
    values are already USD notional (e.g. Binance ``quoteVolume``); otherwise
    they are base-asset quantity and are multiplied by the latest price. Used as
    a robust, always-available proxy for the execution venue's book depth.
    """
    if volume is None or volume.empty:
        return pd.Series(dtype=float)
    adv = volume.tail(max(1, lookback)).mean(axis=0)
    if not volume_is_dollar:
        if latest_px.empty:
            return pd.Series(dtype=float)
        px = latest_px.set_index("symbol")["price"]
        adv = adv.mul(px.reindex(adv.index))
    return adv.dropna().astype(float)


def _latest_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Extract latest price per symbol from wide-format DataFrame."""
    if prices.empty:
        return pd.DataFrame(columns=["symbol", "price"])
    latest = prices.iloc[-1]
    return pd.DataFrame({"symbol": latest.index, "price": latest.values})


def _read_instrument_map(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(
            columns=[
                "symbol",
                "asset_type",
                "currency",
                "multiplier",
                "lot_size",
                "min_qty",
                "qty_step",
                "min_notional",
            ]
        )
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"instrument_map not found: {path}")

    if p.suffix.lower() in (".yml", ".yaml"):
        obj = yaml.safe_load(p.read_text(encoding="utf-8"))
        rows = obj.get("instruments", obj) if isinstance(obj, dict) else obj
        df = pd.DataFrame(rows)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError("instrument_map must be .yaml/.yml or .csv")

    if "symbol" not in df.columns:
        raise ValueError("instrument_map must include 'symbol' column/field")

    # Defaults
    if "multiplier" not in df.columns:
        df["multiplier"] = 1.0
    if "currency" not in df.columns:
        df["currency"] = "USD"
    if "lot_size" not in df.columns:
        df["lot_size"] = 1.0
    for c in ("min_qty", "qty_step", "min_notional"):
        if c not in df.columns:
            df[c] = 0.0
    if "asset_type" not in df.columns:
        df["asset_type"] = "spot"

    df["symbol"] = df["symbol"].astype(str)
    return df


def _usd_marks(pos: pd.DataFrame) -> dict[str, float]:
    """Per-unit USD mark per symbol, NaN where the price is unknown.

    A duplicated symbol keeps the LAST row's mark, which is correct: a per-unit
    price does not accumulate. Quantities do — see :func:`_summed_holdings`.
    """
    return dict(
        zip(
            pos["symbol"],
            (pos["price"] * pos["multiplier"] * pos["fx_to_usd"]).astype(float),
            strict=False,
        )
    )


def _summed_holdings(pos: pd.DataFrame) -> dict[str, float]:
    """Total quantity per symbol, ADDING duplicate rows.

    A plain ``dict(zip(...))`` keeps only the last row, where the ``value_usd``
    sum this replaced added them — so a position reported across two rows (two
    accounts, two lots) silently lost all but one, understating the book. That
    is the same defect class this module exists to close, so it is summed here.
    """
    grouped = pos.groupby("symbol", as_index=False)["qty"].sum()
    return dict(zip(grouped["symbol"], grouped["qty"].astype(float), strict=False))


def _fx_rate_to_usd(fx: pd.DataFrame | None, ccy: str) -> float:
    if ccy.upper() == "USD":
        return 1.0
    if fx is None or len(fx) == 0:
        return 1.0

    ccy = ccy.upper()
    fx2 = fx.copy()
    fx2["pair"] = fx2["pair"].astype(str).str.upper()
    fx2 = fx2.sort_values("date").groupby("pair", as_index=False).tail(1)

    direct = f"{ccy}USD"
    inv = f"USD{ccy}"
    pairs = set(fx2["pair"])
    if direct in pairs:
        return float(fx2.loc[fx2["pair"] == direct, "rate"].iloc[0])
    if inv in pairs:
        r = float(fx2.loc[fx2["pair"] == inv, "rate"].iloc[0])
        return 1.0 / r if r else 1.0
    return 1.0


def _round_down_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.floor(x / step) * step


def _apply_qty_rules(qty: float, lot_size: float, min_qty: float, qty_step: float) -> float:
    q = qty
    if lot_size and lot_size > 0:
        q = round(q / lot_size) * lot_size
    if qty_step and qty_step > 0:
        q = _round_down_to_step(q, qty_step)
    if min_qty and min_qty > 0 and abs(q) < min_qty:
        return 0.0
    return float(q)


# --------------------------
# Pipeline
# --------------------------


@dataclass
class AllocationsToOrdersPipeline:
    meta = PluginMeta(
        name="trade.allocations_to_orders.v1",
        kind="pipeline",
        version="0.3.0",
        core_compat=">=0.1,<0.2",
        description="Bridge allocations -> targets/orders (+ execute). Supports multipliers, lot/step rules, FX (USD base).",
        tags=("trading", "bridge"),
        capabilities=("paper", "live", "etf", "stocks", "futures", "crypto"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "allocations_path": {"type": ["string", "null"], "description": "Explicit path to allocations.parquet"},
                "allocations_ref": {
                    "type": ["string", "null"],
                    "description": "Auto-resolve: latest:<pipeline_id> (e.g. latest:fund_selection.simple.v1)",
                },
                "allocations_artifact": {"type": "string", "default": "allocations.parquet"},
                "approval_required": {"type": "boolean", "default": False},
                "approval_path": {
                    "type": ["string", "null"],
                    "description": "Path to approval JSON. If null, defaults to ./approvals/<orders_digest>.json",
                },
                "instrument_map": {"type": ["string", "null"], "description": "YAML/CSV with symbol metadata"},
                "prices": {
                    "type": "object",
                    "properties": {"lookback_days": {"type": "integer", "minimum": 1, "default": 5}},
                },
                "base_currency": {"type": "string", "default": "USD"},
                "min_abs_qty": {"type": "number", "minimum": 0, "default": 0.0},
                "allow_short": {"type": "boolean", "default": False},
                "cash_fallback_usd": {"type": "number", "default": 100000.0},
                "equity_reconciliation_tolerance": {
                    "type": "number",
                    "default": DEFAULT_RECONCILIATION_TOLERANCE,
                    "description": (
                        "Relative tolerance for the LIVE pre-trade reconciliation between this "
                        "pipeline's valuation and the broker's get_equity(). Beyond it, a live "
                        "run refuses to trade. Default 0.005 (0.5%)."
                    ),
                },
                "require_equity_reconciliation": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Whether a long-only spot book must reconcile against broker equity "
                        "before trading. Does NOT disable the completeness gate."
                    ),
                },
                "max_adv_participation": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "default": None,
                    "description": "Venue-depth sizing cap: max position value as a multiple of the "
                    "execution venue's average daily volume (USD). null/0 disables.",
                },
                "adv_lookback_days": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 20,
                    "description": "Window for the ADV used by max_adv_participation.",
                },
                "adv_volume_is_dollar": {
                    "type": "boolean",
                    "default": True,
                    "description": "True if the data plugin's volume is USD notional "
                    "(e.g. Binance quoteVolume); False multiplies by price (e.g. Hyperliquid).",
                },
            },
            "required": ["allocations_path"],
        },
        inputs=("allocations",),
        outputs=("targets", "orders", "fills", "portfolio_daily"),
        examples=(
            "plugins:\n  pipeline:\n    name: trade.allocations_to_orders.v1\n    params:\n      allocations_path: ./artifacts/<run_id>/allocations.parquet\n      instrument_map: ./configs/instruments.yaml\n      prices:\n        lookback_days: 5",
        ),
    )
    kind = "trading"

    def run(
        self,
        *,
        mode: Mode,
        asof: str,
        params: dict[str, Any],
        data: DataPlugin,
        store: ArtifactStore,
        broker: BrokerPlugin | None,
        risk: list[RiskPlugin],
        **kwargs,
    ) -> RunResult:
        if mode in ("paper", "live") and broker is None:
            raise ValueError("broker_required_for_paper_or_live")

        base_ccy = str(params.get("base_currency", "USD")).upper()
        if base_ccy != "USD":
            # keep artifact contracts stable (portfolio_daily is USD-based)
            base_ccy = "USD"

        alloc_path = params.get("allocations_path")
        ref = params.get("allocations_ref")
        artifact_file = str(params.get("allocations_artifact", "allocations.parquet"))
        if (not alloc_path) and ref and str(ref).startswith("latest:"):
            pipe = str(ref).split("latest:", 1)[1].strip()
            artifacts_root = Path(store.root).parent
            alloc_path = str(resolve_latest_artifact(artifacts_root, pipe, artifact_file))
        if not alloc_path:
            raise ValueError("Must provide allocations_path or allocations_ref=latest:<pipeline>")
        alloc = read_parquet(alloc_path)
        if "symbol" not in alloc.columns or "weight" not in alloc.columns:
            raise ValueError("allocations_path must contain columns: symbol, weight")

        alloc = alloc.copy()
        alloc["symbol"] = alloc["symbol"].astype(str)
        alloc["weight"] = alloc["weight"].astype(float)
        alloc["asof"] = asof

        inst = _read_instrument_map(params.get("instrument_map"))
        alloc = alloc.merge(inst, on="symbol", how="left")

        # defaults
        alloc["multiplier"] = alloc["multiplier"].fillna(1.0).astype(float)
        alloc["currency"] = alloc["currency"].fillna("USD").astype(str)
        alloc["lot_size"] = alloc["lot_size"].fillna(1.0).astype(float)
        alloc["min_qty"] = alloc["min_qty"].fillna(0.0).astype(float)
        alloc["qty_step"] = alloc["qty_step"].fillna(0.0).astype(float)
        alloc["min_notional"] = alloc["min_notional"].fillna(0.0).astype(float)
        alloc["asset_type"] = alloc["asset_type"].fillna("spot").astype(str)

        universe = pd.DataFrame({"symbol": alloc["symbol"].tolist()})
        store.put_parquet("universe", universe)

        market_data = data.load_market_data(
            universe, asof, {**params.get("prices", {"lookback_days": 5}), "mode": mode}
        )
        prices_wide = market_data["prices"]
        latest = _latest_prices(prices_wide)
        store.put_parquet(
            "prices", prices_wide.reset_index() if isinstance(prices_wide.index, pd.DatetimeIndex) else prices_wide
        )

        alloc = alloc.merge(latest, on="symbol", how="left")
        if alloc["price"].isna().any():
            missing = alloc.loc[alloc["price"].isna(), "symbol"].tolist()
            raise ValueError(f"missing_prices_for: {missing}")

        fx = data.load_fx(asof, params.get("fx", {}))

        # cash
        if broker is None:
            cash = {"USD": float(params.get("cash_fallback_usd", 100000.0))}
            pos = pd.DataFrame(columns=["symbol", "qty"])
        else:
            cash = broker.get_cash() or {}
            pos = broker.get_positions()

        cash_usd = 0.0
        for ccy, amt in cash.items():
            cash_usd += float(amt) * _fx_rate_to_usd(fx, str(ccy))

        # positions
        if pos is None or len(pos) == 0:
            pos = pd.DataFrame({"symbol": [], "qty": []})
        pos = pos.copy()
        pos["symbol"] = pos["symbol"].astype(str)
        pos["qty"] = pos["qty"].astype(float)

        pos = pos.merge(alloc[["symbol", "price", "multiplier", "currency"]], on="symbol", how="left")
        # A held symbol that is not in today's allocations has NO price here.
        # `fillna(0.0)` used to turn that into a position worth nothing, which
        # is why an unpriceable book and an empty one produced the same
        # portfolio value. The NaN is kept so the valuation can tell them apart;
        # it reaches nothing else (see the note at the orders merge below).
        pos["price"] = pos["price"].astype(float)
        pos["multiplier"] = pos["multiplier"].fillna(1.0).astype(float)
        pos["currency"] = pos["currency"].fillna("USD").astype(str)
        pos["fx_to_usd"] = pos["currency"].apply(lambda c: _fx_rate_to_usd(fx, c)).astype(float)

        pos_price_usd = _usd_marks(pos)
        holdings = _summed_holdings(pos)

        def _held_price(symbol: str) -> float | None:
            px = pos_price_usd.get(symbol)
            return None if px is None or not np.isfinite(px) else float(px)

        valuation = resolve_portfolio_value(
            broker=broker,
            mode=mode,
            cash=cash_usd,
            holdings=holdings,
            get_price=_held_price,
            # Allocations carry a per-unit USD mark (price * multiplier * fx), so
            # this path is mark-to-market. That stays the answer for a SIMULATION
            # against an undeclared broker; on live an undeclared broker refuses.
            fallback_basis=BASIS_MARK,
            # No `stable_coin=` / `exclusions=` here, unlike the three crypto
            # call sites: this pipeline has neither concept. Cash is already
            # multi-currency and converted to USD above, so there is no quote
            # token sitting in the positions table, and it exposes no exclusions
            # parameter. Consequence worth knowing: a held symbol absent from
            # TODAY'S allocations has no mark, so it is `unpriced` and refuses a
            # live run — correct (the NAV really is unknown), but the remedy is
            # to carry that symbol in the allocations file with a price, since
            # targets are built from `alloc` alone and the book cannot otherwise
            # sell it.
            tolerance=float(params.get("equity_reconciliation_tolerance", DEFAULT_RECONCILIATION_TOLERANCE)),
            require_reconciliation=bool(params.get("require_equity_reconciliation", True)),
        )
        portfolio_value_usd_pre = float(valuation.value)

        # `current_value_usd` used to be computed here and added to cash. It is
        # gone: the marked value now comes from the valuation above, which knows
        # which positions it FAILED to mark. Nothing downstream reads pos'
        # price or value columns (only symbol/qty, at the orders merge below).

        # targets
        alloc["fx_to_usd"] = alloc["currency"].apply(lambda c: _fx_rate_to_usd(fx, c))
        denom = alloc["price"].astype(float) * alloc["multiplier"].astype(float) * alloc["fx_to_usd"].astype(float)
        alloc["target_value_usd"] = alloc["weight"] * portfolio_value_usd_pre

        # Venue-depth participation cap (layer-b sizing): never target a position
        # larger than `max_adv_participation` x the execution venue's average
        # daily volume. ADV is a robust, always-available proxy for book depth
        # (full L2 depth could refine this later). Opt-in: disabled when unset.
        max_part = params.get("max_adv_participation")
        if max_part is not None and float(max_part) > 0:
            adv_usd = _adv_usd(
                market_data.get("volume", pd.DataFrame()),
                latest,
                int(params.get("adv_lookback_days", 20)),
                bool(params.get("adv_volume_is_dollar", True)),
            )
            alloc["adv_usd"] = alloc["symbol"].map(adv_usd)
            alloc["cap_value_usd"] = float(max_part) * alloc["adv_usd"]
            capped = alloc["cap_value_usd"].notna() & (alloc["target_value_usd"].abs() > alloc["cap_value_usd"])
            if capped.any():
                alloc.loc[capped, "target_value_usd"] = (
                    np.sign(alloc.loc[capped, "target_value_usd"]) * alloc.loc[capped, "cap_value_usd"]
                )
                logger.info(
                    "Venue-depth cap: clamped %d/%d positions to %.3gx ADV (%s)",
                    int(capped.sum()),
                    len(alloc),
                    float(max_part),
                    ", ".join(alloc.loc[capped, "symbol"].astype(str).head(5)),
                )
        else:
            alloc["adv_usd"] = np.nan
            alloc["cap_value_usd"] = np.nan

        alloc["raw_target_qty"] = (alloc["target_value_usd"] / denom.replace(0.0, np.nan)).fillna(0.0)

        if not bool(params.get("allow_short", False)):
            alloc["raw_target_qty"] = alloc["raw_target_qty"].clip(lower=0.0)

        alloc["target_qty"] = [
            _apply_qty_rules(float(q), float(lot), float(mq), float(step))
            for q, lot, mq, step in zip(
                alloc["raw_target_qty"], alloc["lot_size"], alloc["min_qty"], alloc["qty_step"], strict=False
            )
        ]

        targets = alloc[["symbol", "weight", "asof", "price", "target_qty"]].copy()
        a_targets = store.put_parquet("targets", targets)

        # orders
        cur = pos[["symbol", "qty"]].rename(columns={"qty": "cur_qty"})
        ords = targets.merge(cur, on="symbol", how="left")
        ords["cur_qty"] = ords["cur_qty"].fillna(0.0).astype(float)
        ords["delta_qty"] = ords["target_qty"].astype(float) - ords["cur_qty"]
        min_abs_qty = float(params.get("min_abs_qty", 0.0))
        ords = ords[ords["delta_qty"].abs() > max(min_abs_qty, 0.0)].copy()
        ords["side"] = ords["delta_qty"].apply(lambda x: "buy" if x > 0 else "sell")
        ords["qty"] = ords["delta_qty"].abs()

        ords = ords.merge(alloc[["symbol", "multiplier", "fx_to_usd", "min_notional"]], on="symbol", how="left")
        ords["multiplier"] = ords["multiplier"].fillna(1.0).astype(float)
        ords["fx_to_usd"] = ords["fx_to_usd"].fillna(1.0).astype(float)
        ords["min_notional"] = ords["min_notional"].fillna(0.0).astype(float)
        ords["notional_usd"] = (
            ords["qty"].astype(float) * ords["price"].astype(float) * ords["multiplier"] * ords["fx_to_usd"]
        )
        ords = ords[ords["notional_usd"] >= ords["min_notional"]].copy()

        orders = ords[["symbol", "side", "qty", "price"]].copy()
        orders["asof"] = asof
        a_orders = store.put_parquet("orders", orders)

        # extra debug artifacts (non-contract)
        store.put_parquet(
            "targets_ext",
            alloc[
                [
                    "symbol",
                    "asset_type",
                    "currency",
                    "multiplier",
                    "fx_to_usd",
                    "price",
                    "target_value_usd",
                    "raw_target_qty",
                    "target_qty",
                    "min_notional",
                    "adv_usd",
                    "cap_value_usd",
                ]
            ].copy(),
        )

        findings = []
        for rp in risk:
            try:
                findings.extend(rp.check_targets(targets, {}))
                findings.extend(rp.check_orders(orders, {}))
            except Exception:
                pass

        fills = pd.DataFrame(columns=["symbol", "side", "qty", "price"])
        approval_ok = bool(params.get("approval_ok", True))
        if mode in ("paper", "live") and broker is not None and len(orders):
            if bool(params.get("approval_required", False)) and not approval_ok:
                # Approval missing/mismatch -> do not execute
                pass
            else:
                fills = broker.place_orders(orders[["symbol", "side", "qty", "price"]])
        a_fills = store.put_parquet("fills", fills)

        # After-snapshot. This runs AFTER place_orders, so it is best-effort by
        # construction: a venue hiccup here must not kill a run whose orders
        # already executed, or the fills go unrecorded. `KrakenBroker.get_equity`
        # now RAISES rather than understating, which makes the try/except load
        # bearing rather than decorative. (`trading_pipeline` has always wrapped
        # its twin of this call; the two pipelines used to disagree.)
        portfolio_value_usd_post = portfolio_value_usd_pre
        cash_usd_post = cash_usd
        post_valuation: PortfolioValuation | None = None

        if broker is not None and mode in ("paper", "live"):
            try:
                cash2 = broker.get_cash() or {}
                cash_usd_post = 0.0
                for ccy, amt in cash2.items():
                    cash_usd_post += float(amt) * _fx_rate_to_usd(fx, str(ccy))

                if hasattr(broker, "get_equity"):
                    portfolio_value_usd_post = float(broker.get_equity())
                else:
                    pos2 = broker.get_positions()
                    if pos2 is None or len(pos2) == 0:
                        pos2 = pd.DataFrame({"symbol": [], "qty": []})
                    pos2 = pos2.copy()
                    pos2["symbol"] = pos2["symbol"].astype(str)
                    pos2["qty"] = pos2["qty"].astype(float)

                    pos2 = pos2.merge(alloc[["symbol", "price", "multiplier", "currency"]], on="symbol", how="left")
                    # NOT `fillna(0.0)`: that is the defect this change exists to
                    # remove. A holding with no price is worth an UNKNOWN amount,
                    # not zero, and this number is the NAV written to
                    # portfolio_daily -- a book recorded at cash because a ticker
                    # was missing is how the original understatement hid.
                    pos2["price"] = pos2["price"].astype(float)
                    pos2["multiplier"] = pos2["multiplier"].fillna(1.0).astype(float)
                    pos2["currency"] = pos2["currency"].fillna("USD").astype(str)
                    pos2["fx_to_usd"] = pos2["currency"].apply(lambda c: _fx_rate_to_usd(fx, c)).astype(float)

                    post_marks = _usd_marks(pos2)
                    post_valuation = value_holdings(
                        cash=cash_usd_post,
                        holdings=_summed_holdings(pos2),
                        get_price=lambda s: post_marks.get(s),
                    )
                    portfolio_value_usd_post = float(post_valuation.value)
                    if post_valuation.unpriced:
                        logger.error(
                            "Post-trade NAV is UNDERSTATED: %d of %d holding(s) could not be marked (%s). "
                            "portfolio_daily records the marked part only.",
                            len(post_valuation.unpriced),
                            post_valuation.n_holdings,
                            ", ".join(post_valuation.unpriced),
                        )
            except Exception as exc:
                # Deliberately broad: the orders are already placed.
                logger.error(
                    "Post-trade snapshot failed (%r); reporting the pre-trade value %.2f instead. "
                    "Orders were already executed and ARE recorded in fills.",
                    exc,
                    portfolio_value_usd_pre,
                )
                portfolio_value_usd_post = portfolio_value_usd_pre
                cash_usd_post = cash_usd

        portfolio_daily = pd.DataFrame(
            [
                {
                    "asof": asof,
                    "cash_usd": float(cash_usd_post),
                    "portfolio_value_usd": float(portfolio_value_usd_post),
                }
            ]
        )
        a_port = store.put_parquet("portfolio_daily", portfolio_daily)

        llm_notes_md = f"""# Run summary: {self.meta.name}

- asof: {asof}
- mode: {mode}
- portfolio_value_usd_pre: {portfolio_value_usd_pre:.2f}
- portfolio_value_usd_post: {portfolio_value_usd_post:.2f}
- portfolio_valuation: {valuation.state} (source={valuation.source}, \
{valuation.n_holdings} holding(s), {len(valuation.unpriced)} unpriced)
- orders: {len(orders)}
- fills: {len(fills)}
- fx_loaded: {fx is not None}
- instrument_map: {bool(params.get("instrument_map"))}
- approval_required: {bool(params.get("approval_required", False))}
- approval_ok: {approval_ok}
"""
        store.put_json("llm_notes", {"markdown": llm_notes_md})

        metrics = {
            "portfolio_value_usd_pre": float(portfolio_value_usd_pre),
            "portfolio_value_usd_post": float(portfolio_value_usd_post),
            "n_orders": float(len(orders)),
            "n_fills": float(len(fills)),
            # "nothing held", "fully marked" and "could not price" must stay
            # distinguishable in the run record, not collapse to one number.
            **valuation.as_metrics(),
        }

        return RunResult(
            run_id=store.run_id,
            pipeline_name=self.meta.name,
            mode=mode,
            asof=asof,
            artifacts={
                "targets": a_targets,
                "orders": a_orders,
                "fills": a_fills,
                "portfolio_daily": a_port,
            },
            metrics=metrics,
            notes={
                "kind": "trading",
                "risk_findings": findings,
                **valuation.as_notes(),
                "extra_artifacts": [
                    "targets_ext.parquet",
                    "llm_notes.json",
                    "orders_digest.json",
                    "approval_status.json",
                ],
            },
        )
