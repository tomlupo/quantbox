from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import math
from pathlib import Path
import pandas as pd
import yaml
import numpy as np

from quantbox.contracts import PluginMeta, RunResult, Mode, DataPlugin, ArtifactStore, BrokerPlugin, RiskPlugin

# --------------------------
# Helpers
# --------------------------

def _latest_prices(prices: pd.DataFrame) -> pd.DataFrame:
    px = prices.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1)
    return px[["symbol", "close"]].rename(columns={"close": "price"})

def _read_instrument_map(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["symbol","asset_type","currency","multiplier","lot_size","min_qty","qty_step","min_notional"])
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
    for c in ("min_qty","qty_step","min_notional"):
        if c not in df.columns:
            df[c] = 0.0
    if "asset_type" not in df.columns:
        df["asset_type"] = "spot"

    df["symbol"] = df["symbol"].astype(str)
    return df

def _fx_rate_to_usd(fx: Optional[pd.DataFrame], ccy: str) -> float:
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
        version="0.2.0",
        core_compat=">=0.1,<0.2",
        description="Bridge allocations -> targets/orders (+ execute). Supports multipliers, lot/step rules, FX (USD base).",
        tags=("trading","bridge"),
        capabilities=("paper","live","etf","stocks","futures","crypto"),
        schema_version="v1",
        params_schema={
            "type":"object",
            "properties":{
                "allocations_path":{"type":"string"},
                "instrument_map":{"type":["string","null"], "description":"YAML/CSV with symbol metadata"},
                "prices":{"type":"object","properties":{"lookback_days":{"type":"integer","minimum":1,"default":5}}},
                "base_currency":{"type":"string","default":"USD"},
                "min_abs_qty":{"type":"number","minimum":0,"default":0.0},
                "allow_short":{"type":"boolean","default":False},
                "cash_fallback_usd":{"type":"number","default":100000.0}
            },
            "required":["allocations_path"]
        },
        inputs=("allocations",),
        outputs=("targets","orders","fills","portfolio_daily"),
        examples=(
            "plugins:\n  pipeline:\n    name: trade.allocations_to_orders.v1\n    params:\n      allocations_path: ./artifacts/<run_id>/allocations.parquet\n      instrument_map: ./configs/instruments.yaml\n      prices:\n        lookback_days: 5",
        )
    )
    kind = "trading"

    def run(
        self,
        *,
        mode: Mode,
        asof: str,
        params: Dict[str, Any],
        data: DataPlugin,
        store: ArtifactStore,
        broker: Optional[BrokerPlugin],
        risk: List[RiskPlugin],
    ) -> RunResult:
        if mode in ("paper","live") and broker is None:
            raise ValueError("broker_required_for_paper_or_live")

        base_ccy = str(params.get("base_currency", "USD")).upper()
        if base_ccy != "USD":
            # keep artifact contracts stable (portfolio_daily is USD-based)
            base_ccy = "USD"

        alloc = pd.read_parquet(params["allocations_path"])
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

        prices = data.load_prices(universe, asof, params.get("prices", {"lookback_days": 5}))
        latest = _latest_prices(prices)
        store.put_parquet("prices", prices)

        alloc = alloc.merge(latest, on="symbol", how="left")
        if alloc["price"].isna().any():
            missing = alloc.loc[alloc["price"].isna(), "symbol"].tolist()
            raise ValueError(f"missing_prices_for: {missing}")

        fx = data.load_fx(asof, params.get("fx", {}))

        # cash
        if broker is None:
            cash = {"USD": float(params.get("cash_fallback_usd", 100000.0))}
            pos = pd.DataFrame(columns=["symbol","qty"])
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

        pos = pos.merge(alloc[["symbol","price","multiplier","currency"]], on="symbol", how="left")
        pos["price"] = pos["price"].fillna(0.0).astype(float)
        pos["multiplier"] = pos["multiplier"].fillna(1.0).astype(float)
        pos["currency"] = pos["currency"].fillna("USD").astype(str)
        pos["fx_to_usd"] = pos["currency"].apply(lambda c: _fx_rate_to_usd(fx, c))
        pos["value_usd"] = pos["qty"] * pos["price"] * pos["multiplier"] * pos["fx_to_usd"]
        current_value_usd = float(pos["value_usd"].sum())

        portfolio_value_usd_pre = float(cash_usd + current_value_usd)

        # targets
        alloc["fx_to_usd"] = alloc["currency"].apply(lambda c: _fx_rate_to_usd(fx, c))
        denom = alloc["price"].astype(float) * alloc["multiplier"].astype(float) * alloc["fx_to_usd"].astype(float)
        alloc["target_value_usd"] = alloc["weight"] * portfolio_value_usd_pre
        alloc["raw_target_qty"] = (alloc["target_value_usd"] / denom.replace(0.0, np.nan)).fillna(0.0)

        if not bool(params.get("allow_short", False)):
            alloc["raw_target_qty"] = alloc["raw_target_qty"].clip(lower=0.0)

        alloc["target_qty"] = [
            _apply_qty_rules(float(q), float(lot), float(mq), float(step))
            for q, lot, mq, step in zip(alloc["raw_target_qty"], alloc["lot_size"], alloc["min_qty"], alloc["qty_step"])
        ]

        targets = alloc[["symbol","weight","asof","price","target_qty"]].copy()
        a_targets = store.put_parquet("targets", targets)

        # orders
        cur = pos[["symbol","qty"]].rename(columns={"qty":"cur_qty"})
        ords = targets.merge(cur, on="symbol", how="left")
        ords["cur_qty"] = ords["cur_qty"].fillna(0.0).astype(float)
        ords["delta_qty"] = ords["target_qty"].astype(float) - ords["cur_qty"]
        min_abs_qty = float(params.get("min_abs_qty", 0.0))
        ords = ords[ords["delta_qty"].abs() > max(min_abs_qty, 0.0)].copy()
        ords["side"] = ords["delta_qty"].apply(lambda x: "buy" if x > 0 else "sell")
        ords["qty"] = ords["delta_qty"].abs()

        ords = ords.merge(alloc[["symbol","multiplier","fx_to_usd","min_notional"]], on="symbol", how="left")
        ords["multiplier"] = ords["multiplier"].fillna(1.0).astype(float)
        ords["fx_to_usd"] = ords["fx_to_usd"].fillna(1.0).astype(float)
        ords["min_notional"] = ords["min_notional"].fillna(0.0).astype(float)
        ords["notional_usd"] = ords["qty"].astype(float) * ords["price"].astype(float) * ords["multiplier"] * ords["fx_to_usd"]
        ords = ords[ords["notional_usd"] >= ords["min_notional"]].copy()

        orders = ords[["symbol","side","qty","price"]].copy()
        orders["asof"] = asof
        a_orders = store.put_parquet("orders", orders)

        # extra debug artifacts (non-contract)
        store.put_parquet("targets_ext", alloc[[
            "symbol","asset_type","currency","multiplier","fx_to_usd","price",
            "target_value_usd","raw_target_qty","target_qty","min_notional"
        ]].copy())

        findings = []
        for rp in risk:
            try:
                findings.extend(rp.check_targets(targets, {}))
                findings.extend(rp.check_orders(orders, {}))
            except Exception:
                pass

        fills = pd.DataFrame(columns=["symbol","side","qty","price"])
        if mode in ("paper","live") and broker is not None and len(orders):
            fills = broker.place_orders(orders[["symbol","side","qty","price"]])
        a_fills = store.put_parquet("fills", fills)

        # after snapshot
        portfolio_value_usd_post = portfolio_value_usd_pre
        cash_usd_post = cash_usd

        if broker is not None and mode in ("paper","live"):
            cash2 = broker.get_cash() or {}
            cash_usd_post = 0.0
            for ccy, amt in cash2.items():
                cash_usd_post += float(amt) * _fx_rate_to_usd(fx, str(ccy))

            pos2 = broker.get_positions()
            if pos2 is None or len(pos2) == 0:
                pos2 = pd.DataFrame({"symbol": [], "qty": []})
            pos2 = pos2.copy()
            pos2["symbol"] = pos2["symbol"].astype(str)
            pos2["qty"] = pos2["qty"].astype(float)

            pos2 = pos2.merge(alloc[["symbol","price","multiplier","currency"]], on="symbol", how="left")
            pos2["price"] = pos2["price"].fillna(0.0).astype(float)
            pos2["multiplier"] = pos2["multiplier"].fillna(1.0).astype(float)
            pos2["currency"] = pos2["currency"].fillna("USD").astype(str)
            pos2["fx_to_usd"] = pos2["currency"].apply(lambda c: _fx_rate_to_usd(fx, c))
            pos2["value_usd"] = pos2["qty"] * pos2["price"] * pos2["multiplier"] * pos2["fx_to_usd"]
            portfolio_value_usd_post = float(cash_usd_post + pos2["value_usd"].sum())

        portfolio_daily = pd.DataFrame([{
            "asof": asof,
            "cash_usd": float(cash_usd_post),
            "portfolio_value_usd": float(portfolio_value_usd_post),
        }])
        a_port = store.put_parquet("portfolio_daily", portfolio_daily)

        llm_notes_md = f"""# Run summary: {self.meta.name}

- asof: {asof}
- mode: {mode}
- portfolio_value_usd_pre: {portfolio_value_usd_pre:.2f}
- portfolio_value_usd_post: {portfolio_value_usd_post:.2f}
- orders: {len(orders)}
- fills: {len(fills)}
- fx_loaded: {fx is not None}
- instrument_map: {bool(params.get("instrument_map"))}
"""
        store.put_json("llm_notes", {"markdown": llm_notes_md})

        metrics = {
            "portfolio_value_usd_pre": float(portfolio_value_usd_pre),
            "portfolio_value_usd_post": float(portfolio_value_usd_post),
            "n_orders": float(len(orders)),
            "n_fills": float(len(fills)),
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
            notes={"kind":"trading", "risk_findings": findings, "extra_artifacts": ["targets_ext.parquet","llm_notes.json"]},
        )
