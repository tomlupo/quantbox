"""A refused portfolio valuation must cost the cycle its ORDERS, not its BOOKS.

The gate added by this branch lets `resolve_portfolio_value` raise out of
Stage 5. That raise sat ABOVE Stage 6c -- resolving orders a previous cycle left
WORKING at the venue -- whose own `try/except` can only catch faults raised
INSIDE it. So a live book with a limit order that filled overnight would, on the
first day a held dust coin could not be marked, never book that fill; the queue's
`DEFAULT_MAX_AGE_DAYS` then drops the record after 7 days. Losing a real fill is
the one outcome that queue exists to prevent.

The fixture builds every path it asserts on and nothing is read from the repo
tree. The positive control is the second test: the same run WITHOUT a refusal
must still place orders, so a pipeline that silently stopped trading altogether
would fail here rather than pass this file.
"""

from __future__ import annotations

import pandas as pd

from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline
from quantbox.portfolio_value import PortfolioValuationError
from quantbox.reconciliation.working_orders import WorkingOrderStore

SYMBOLS = ["BTC"]
ASOF = "2026-09-22"
BOOK = "test-book"
ORDER_ID = "oid-1"


class _Store:
    """Minimal ArtifactStore: records what was written, writes nothing."""

    def __init__(self):
        self.run_id = "testrun"
        self.written: dict[str, object] = {}

    def put_parquet(self, name, df):
        self.written[name] = df
        return f"artifacts/{name}.parquet"

    def put_json(self, name, obj):
        self.written[name] = obj
        return f"artifacts/{name}.json"


class _Data:
    def load_universe(self, params):
        return pd.DataFrame({"symbol": SYMBOLS})

    def load_market_data(self, universe, asof, params):
        idx = pd.to_datetime([ASOF])
        return {"prices": pd.DataFrame({"BTC": [100.0]}, index=idx)}


class _Strategy:
    meta = type("M", (), {"name": "stub.strategy.v1"})()

    def run(self, *, data, params):
        idx = pd.to_datetime([ASOF])
        return {"weights": pd.DataFrame({"BTC": [1.0]}, index=idx)}


class _Broker:
    """Holds one order that the venue reports as FILLED."""

    def __init__(self):
        self.placed: list[pd.DataFrame] = []

    def get_positions(self):
        return pd.DataFrame(columns=["symbol", "qty"])

    def get_market_snapshot(self, symbols):
        syms = list(symbols)
        return pd.DataFrame({"symbol": syms, "mid": [100.0] * len(syms)})

    def get_cash(self):
        return {"USDC": 1000.0}

    def fetch_order_result(self, order_id, symbol):
        return {"status": "FILLED", "qty": 0.5, "price": 100.0, "error": ""}

    def place_orders(self, orders):
        self.placed.append(orders.copy())
        return pd.DataFrame(columns=["symbol", "side", "qty", "price", "status"])

    def notify(self, msg):
        return True


class _RefusingRebalancer:
    def generate_orders(self, weights, broker, params):
        raise PortfolioValuationError("cannot mark DUST", reason="unpriced_holdings", unpriced=("DUST",))


class _PlainRebalancer:
    def generate_orders(self, weights, broker, params):
        return {
            "weights": weights,
            "rebalancing": pd.DataFrame(),
            "orders": pd.DataFrame(
                [
                    {
                        "Asset": "BTC",
                        "Symbol": "BTC",
                        "Action": "Buy",
                        "Raw Quantity": 0.1,
                        "Adjusted Quantity": 0.1,
                        "Price": 100.0,
                        "Notional Value": 10.0,
                        "Order Status": "To be placed",
                        "Reason": "",
                        "Executable": True,
                    }
                ]
            ),
            "total_value": 10.0,
        }


def _queue_a_working_order(tmp_path):
    store = WorkingOrderStore(book_key=BOOK, root=str(tmp_path / "data"))
    store.record(
        symbol="BTC",
        side="buy",
        order_id=ORDER_ID,
        requested_qty=0.5,
        cycle_id="c-prev",
        order_ref="ref-1",
    )
    assert store.load(), "fixture queued nothing — the assertions below would be vacuous"
    return store


def _params(tmp_path):
    return {
        "book_key": BOOK,
        "data_dir": str(tmp_path / "data"),
        "capital_at_risk": 1.0,
        "stable_coin_symbol": "USDC",
        "trading_enabled": True,
    }


def _run(tmp_path, rebalancer):
    return TradingPipeline().run(
        mode="live",
        asof=ASOF,
        params=_params(tmp_path),
        data=_Data(),
        store=_Store(),
        broker=_Broker(),
        risk=[],
        strategies=[_Strategy()],
        rebalancer=rebalancer,
    )


def test_a_refused_valuation_still_resolves_last_cycle_working_orders(tmp_path):
    queue = _queue_a_working_order(tmp_path)

    result = _run(tmp_path, _RefusingRebalancer())

    assert not queue.load(), (
        "the overnight fill was never booked: the refusal aborted the run above "
        "Stage 6c and the queued order is still sitting there"
    )
    signals = result.notes.get("exceptions", {})
    assert signals, "no exception-signal block on the run record"
    assert signals.get("pipeline_ok") is False, "a cycle that sized nothing reported as a clean run"
    stages = [e.get("stage") for e in signals.get("api_errors", [])]
    assert "portfolio_valuation" in stages, "the refusal left no trace on the run record"


def test_a_refused_valuation_places_no_orders(tmp_path):
    _queue_a_working_order(tmp_path)
    broker = _Broker()
    pipeline = TradingPipeline()
    result = pipeline.run(
        mode="live",
        asof=ASOF,
        params=_params(tmp_path),
        data=_Data(),
        store=_Store(),
        broker=broker,
        risk=[],
        strategies=[_Strategy()],
        rebalancer=_RefusingRebalancer(),
    )
    assert broker.placed == [], "sized orders off a valuation the gate refused"
    assert result is not None


def test_positive_control_a_clean_cycle_still_trades(tmp_path):
    _queue_a_working_order(tmp_path)
    broker = _Broker()
    TradingPipeline().run(
        mode="live",
        asof=ASOF,
        params=_params(tmp_path),
        data=_Data(),
        store=_Store(),
        broker=broker,
        risk=[],
        strategies=[_Strategy()],
        rebalancer=_PlainRebalancer(),
    )
    assert broker.placed, "the harness never trades, so the refusal tests prove nothing"
