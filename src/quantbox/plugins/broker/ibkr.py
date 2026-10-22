from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantbox.contracts import PluginMeta
from quantbox.trade.book import BASIS_MARK

from ._fills import _CONFIRM_DELAY_S, _to_float, resolve_fill

try:
    from ib_insync import IB, LimitOrder, MarketOrder, Stock
except Exception:  # pragma: no cover
    IB = None  # type: ignore


# An IB ``OrderStatus.status`` in ccxt's unified statuses, so classify_fill /
# resolve_fill read it like every other venue's (TOM-1978). ``Inactive`` is how
# IB reports an order it rejected or will not work; an unrecognised or empty
# status (no status message has arrived yet) stays absent and reads UNKNOWN.
_IBKR_STATUS = {
    "Filled": "closed",
    "Submitted": "open",
    "PreSubmitted": "open",
    "PendingSubmit": "open",
    "ApiPending": "open",
    "PendingCancel": "open",
    "Cancelled": "canceled",
    "ApiCancelled": "canceled",
    "Inactive": "rejected",
}


def _num(value) -> float | None:
    """:func:`_to_float`, and ``None`` for IB's unset value (``UNSET_DOUBLE``, ~1.8e308)."""
    f = _to_float(value)
    return f if (f is not None and abs(f) < 1e300) else None


def ibkr_order(trade) -> dict | None:
    """An ib_insync ``Trade`` as a ccxt-shaped order for :func:`resolve_fill`.

    The status and filled quantity are the order's own (``trade.orderStatus``),
    never the requested quantity. ``average`` is IB's ``avgFillPrice``, or the
    share-weighted price of the trade's executions when IB has not reported one.
    ``None`` for no trade.
    """
    if trade is None:
        return None
    st = getattr(trade, "orderStatus", None)
    order = getattr(trade, "order", None)
    raw = str(getattr(st, "status", "") or "").strip()
    average = _num(getattr(st, "avgFillPrice", None))
    if not average or average <= 0:
        legs = [
            (_num(getattr(f.execution, "shares", None)), _num(getattr(f.execution, "price", None)))
            for f in getattr(trade, "fills", None) or []
        ]
        legs = [(q, p) for q, p in legs if q and p]
        total = sum(q for q, _ in legs)
        average = sum(q * p for q, p in legs) / total if total else None
    return {
        "id": getattr(order, "orderId", None),
        "status": _IBKR_STATUS.get(raw),
        "filled": _num(getattr(st, "filled", None)),
        "remaining": _num(getattr(st, "remaining", None)),
        "amount": _num(getattr(order, "totalQuantity", None)),
        "average": average,
        "price": _num(getattr(order, "lmtPrice", None)),
    }


def _ibkr_reason(trade, reason: str) -> str:
    """``reason`` plus IB's own status word and its last log message, when there is one."""
    raw = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "").strip()
    log = getattr(trade, "log", None) or []
    message = str(getattr(log[-1], "message", "") or "").strip() if log else ""
    extra = "; ".join(x for x in (f"IB status={raw}" if raw else "", message) if x)
    return f"{reason} [{extra}]" if (reason and extra) else reason


@dataclass
class IBKRBroker:
    """Interactive Brokers adapter using ib_insync.

    Implements the QuantBox BrokerPlugin interface:
    - get_positions
    - get_cash
    - get_market_snapshot
    - place_orders
    - fetch_fills

    This is a *scaffold*: good enough to start, but you will likely extend:
    - futures contracts (multiplier/exchange/currency)
    - FX conversion
    - smarter market data handling
    - robust reconciliation
    """

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 7
    account: str | None = None
    readonly: bool = False  # set True to block order placement

    # The venue's valuation basis -- see BrokerPlugin.valuation_basis.
    # Cash-account equities scaffold: net liquidation is cash plus positions.
    # The futures extension this docstring contemplates (multiplier, margin)
    # would have to revisit it.
    valuation_basis = BASIS_MARK

    meta = PluginMeta(
        name="ibkr.live.v1",
        kind="broker",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Interactive Brokers broker adapter (ib_insync)",
        tags=("ibkr", "broker"),
        capabilities=("paper", "live", "stocks", "etfs", "futures"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "default": "127.0.0.1",
                    "description": "TWS / IB Gateway host.",
                },
                "port": {
                    "type": "integer",
                    "default": 7497,
                    "description": "TWS / IB Gateway port (7497 paper, 7496 live).",
                },
                "client_id": {
                    "type": "integer",
                    "default": 7,
                    "description": "IB API client id.",
                },
                "account": {
                    "type": ["string", "null"],
                    "description": "IB account id; null = the default account.",
                },
                "readonly": {
                    "type": "boolean",
                    "default": False,
                    "description": "Block order placement (reads only).",
                },
            },
        },
        examples=(
            "plugins:\n  broker:\n    name: ibkr.live.v1\n    params_init:\n      host: 127.0.0.1\n      port: 7497\n      client_id: 7\n      account: DUXXXX",
        ),
    )

    def __post_init__(self):
        if IB is None:
            raise ImportError("ib_insync is required. pip install ib_insync")
        self.ib = IB()
        self.ib.connect(self.host, int(self.port), clientId=int(self.client_id))

    def _ensure_account(self):
        # If account not specified, keep None and rely on default account in TWS/Gateway
        return

    def get_positions(self) -> pd.DataFrame:
        self._ensure_account()
        positions = self.ib.positions()
        rows = []
        for p in positions:
            if self.account and getattr(p, "account", None) != self.account:
                continue
            sym = getattr(p.contract, "symbol", None) or getattr(p.contract, "localSymbol", "")
            rows.append({"symbol": sym, "qty": float(p.position)})
        return pd.DataFrame(rows)

    def get_cash(self) -> dict[str, float]:
        self._ensure_account()
        # accountSummary returns list of AccountValue
        vals = self.ib.accountSummary()
        cash = {}
        for v in vals:
            if self.account and getattr(v, "account", None) != self.account:
                continue
            if v.tag == "CashBalance":
                import contextlib

                with contextlib.suppress(Exception):
                    cash[v.currency] = float(v.value)
        return cash

    def get_market_snapshot(self, symbols: list[str]) -> pd.DataFrame:
        # Snapshot mid using reqMktData; in practice you may need exchange/currency routing.
        rows = []
        for sym in symbols:
            c = Stock(sym, "SMART", "USD")
            self.ib.qualifyContracts(c)
            t = self.ib.reqMktData(c, "", snapshot=True, regulatorySnapshot=False)
            self.ib.sleep(0.2)
            mid = None
            try:
                bid = float(t.bid) if t.bid is not None else None
                ask = float(t.ask) if t.ask is not None else None
                last = float(t.last) if t.last is not None else None
                if bid is not None and ask is not None:
                    mid = (bid + ask) / 2.0
                elif last is not None:
                    mid = last
            except Exception:
                mid = None
            rows.append({"symbol": sym, "mid": mid})
            self.ib.cancelMktData(c)
        return pd.DataFrame(rows)

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        if self.readonly:
            raise PermissionError("readonly broker: order placement disabled")

        fills = []
        for _, o in orders.iterrows():
            sym = str(o["symbol"])
            side = str(o["side"]).lower()
            qty = float(o["qty"])
            price = o.get("price", None)

            contract = Stock(sym, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            action = "BUY" if side == "buy" else "SELL"
            if price is None or (isinstance(price, float) and (price == 0.0)):
                order = MarketOrder(action, qty)
            else:
                order = LimitOrder(action, qty, float(price))

            trade = self.ib.placeOrder(contract, order)
            # Let the first status and execution messages arrive.
            self.ib.sleep(0.5)

            # One row per order, read from the order's OWN status and filled
            # quantity through resolve_fill (TOM-1978): a partial, a working
            # order and a rejection each come back as such, never as FILLED.
            # The re-read pumps ib_insync's event loop with ib.sleep (time.sleep
            # would block the updates), so resolve_fill itself waits 0 s.
            def refetch(trade=trade) -> dict | None:
                self.ib.sleep(_CONFIRM_DELAY_S)
                return ibkr_order(trade)

            ib_order = ibkr_order(trade)
            status, filled_qty, fill_price, reason = resolve_fill(ib_order, qty, refetch=refetch, confirm_delay=0)
            order_id = (ib_order or {}).get("id")
            fills.append(
                {
                    "symbol": sym,
                    "side": side,
                    "qty": filled_qty,
                    "price": fill_price,
                    "requested_qty": qty,
                    "order_id": "" if order_id is None else str(order_id),
                    "status": status,
                    "error": _ibkr_reason(trade, reason),
                }
            )

        return pd.DataFrame(fills)

    def fetch_fills(self, since: str) -> pd.DataFrame:
        # Best-effort: return executions; filtering by since is minimal here
        execs = self.ib.executions()
        rows = []
        for e in execs:
            sym = getattr(e.contract, "symbol", None) or getattr(e.contract, "localSymbol", "")
            side = "buy" if e.execution.side.upper() == "BOT" else "sell"
            rows.append(
                {"symbol": sym, "side": side, "qty": float(e.execution.shares), "price": float(e.execution.price)}
            )
        return pd.DataFrame(rows)
