from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd

from quantbox.contracts import PluginMeta, RunResult, Mode, DataPlugin, ArtifactStore, BrokerPlugin, RiskPlugin

def _latest_prices(prices: pd.DataFrame) -> pd.DataFrame:
    px = prices.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1)
    return px[["symbol","close"]].rename(columns={"close":"price"})

@dataclass
class AllocationsToOrdersPipeline:
    meta = PluginMeta(
        name="trade.allocations_to_orders.v1",
        kind="pipeline",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Bridge allocations -> targets/orders (+ execute in paper/live)",
        tags=("trading","bridge"),
        capabilities=("paper","live"),
        schema_version="v1",
        params_schema={
            "type":"object",
            "properties":{
                "allocations_path":{"type":"string", "description":"Path to allocations.parquet (symbol, weight, ...)"},
                "universe":{"type":"object","properties":{"symbols":{"type":"array","items":{"type":"string"}}}},
                "prices":{"type":"object","properties":{"lookback_days":{"type":"integer","minimum":1,"default":5}}},
                "round_lot":{"type":"integer","minimum":1,"default":1},
                "min_abs_qty":{"type":"number","minimum":0,"default":0.0}
            },
            "required":["allocations_path"]
        },
        inputs=("allocations",),
        outputs=("targets","orders","fills","portfolio_daily"),
        examples=(
            "plugins:\n  pipeline:\n    name: trade.allocations_to_orders.v1\n    params:\n      allocations_path: ./artifacts/<run_id>/allocations.parquet\n      prices:\n        lookback_days: 5",
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

        alloc_path = params["allocations_path"]
        alloc = pd.read_parquet(alloc_path)
        if "symbol" not in alloc.columns or "weight" not in alloc.columns:
            raise ValueError("allocations_path must contain columns: symbol, weight")

        # Universe is implied by allocations unless explicitly provided
        symbols = alloc["symbol"].astype(str).tolist()
        universe = pd.DataFrame({"symbol": symbols})
        store.put_parquet("universe", universe)

        # Prices
        prices = data.load_prices(universe, asof, params.get("prices", {"lookback_days": 5}))
        latest = _latest_prices(prices)
        store.put_parquet("prices", prices)

        alloc2 = alloc.copy()
        alloc2["symbol"] = alloc2["symbol"].astype(str)
        alloc2 = alloc2.merge(latest, on="symbol", how="left")
        if alloc2["price"].isna().any():
            missing = alloc2[alloc2["price"].isna()]["symbol"].tolist()
            raise ValueError(f"missing_prices_for: {missing}")

        alloc2["asof"] = asof

        # Portfolio state (USD-only in this starter)
        if broker is None:
            cash = {"USD": 100000.0}
            pos = pd.DataFrame(columns=["symbol","qty"])
        else:
            cash = broker.get_cash()
            pos = broker.get_positions()
        cash_usd = float(cash.get("USD", 0.0))
        pos = pos.copy()
        if len(pos) == 0:
            pos = pd.DataFrame({"symbol": [], "qty": []})
        pos["symbol"] = pos["symbol"].astype(str)

        # Approx portfolio value using latest prices (ignore symbols not in alloc)
        pos_m = pos.merge(latest, on="symbol", how="left")
        pos_m["price"] = pos_m["price"].fillna(0.0)
        current_value = float((pos_m["qty"].astype(float) * pos_m["price"].astype(float)).sum())
        port_value = cash_usd + current_value

        # Targets: integer share sizing
        round_lot = int(params.get("round_lot", 1))
        min_abs_qty = float(params.get("min_abs_qty", 0.0))

        alloc2["target_value"] = alloc2["weight"].astype(float) * port_value
        alloc2["target_qty"] = (alloc2["target_value"] / alloc2["price"]).astype(float)
        # round to lot
        alloc2["target_qty"] = (alloc2["target_qty"] / round_lot).round() * round_lot

        targets = alloc2[["symbol","weight","asof","price","target_qty"]].copy()
        a_targets = store.put_parquet("targets", targets)

        # Orders: delta vs current positions
        cur = pos[["symbol","qty"]].rename(columns={"qty":"cur_qty"})
        ords = targets.merge(cur, on="symbol", how="left")
        ords["cur_qty"] = ords["cur_qty"].fillna(0.0).astype(float)
        ords["delta_qty"] = ords["target_qty"].astype(float) - ords["cur_qty"]
        ords = ords[ords["delta_qty"].abs() > max(min_abs_qty, 0.0)].copy()
        ords["side"] = ords["delta_qty"].apply(lambda x: "buy" if x > 0 else "sell")
        ords["qty"] = ords["delta_qty"].abs()
        orders = ords[["symbol","side","qty","price"]].copy()
        orders["asof"] = asof
        a_orders = store.put_parquet("orders", orders)

        # Risk checks (optional)
        findings = []
        for rp in risk:
            try:
                findings.extend(rp.check_targets(targets, {}))
                findings.extend(rp.check_orders(orders, {}))
            except Exception:
                pass

        fills = pd.DataFrame(columns=["symbol","side","qty","price"])
        if mode in ("paper","live") and broker is not None:
            fills = broker.place_orders(orders[["symbol","side","qty","price"]])
        a_fills = store.put_parquet("fills", fills)

        # Portfolio daily snapshot (best-effort)
        # After execution, ask broker again (if available)
        if broker is not None and mode in ("paper","live"):
            cash2 = float(broker.get_cash().get("USD", 0.0))
            pos2 = broker.get_positions()
            pos2 = pos2.merge(latest, on="symbol", how="left")
            pos2["price"] = pos2["price"].fillna(0.0)
            value2 = float((pos2["qty"].astype(float) * pos2["price"].astype(float)).sum())
            port_value2 = cash2 + value2
        else:
            cash2 = cash_usd
            port_value2 = port_value

        portfolio_daily = pd.DataFrame([{
            "asof": asof,
            "cash_usd": float(cash2),
            "portfolio_value_usd": float(port_value2),
        }])
        a_port = store.put_parquet("portfolio_daily", portfolio_daily)

        metrics = {
            "n_symbols": float(len(symbols)),
            "portfolio_value_usd": float(port_value),
            "n_orders": float(len(orders)),
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
            notes={"kind":"trading", "risk_findings": findings},
        )
