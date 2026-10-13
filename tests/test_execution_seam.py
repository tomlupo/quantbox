"""Order execution as its own module, one fill normaliser at the broker seam (TOM-1458).

Pinned here:

* every broker's ``place_orders`` rows pass ONE normaliser: an absent status, or
  a FILLED/PARTIAL row with no filled quantity, is UNKNOWN, never a fill. Each
  venue adapter reads its own reply through ``_fills`` (Binance live included,
  Q9: every fill of an order summed, a resting GTC limit WORKING);
* the trapped-residual counter reads the order's DECLARED intent, never its
  side: a declined OPEN is not a stuck close (TOM-901);
* a paper or live config without ``plugins.rebalancing`` is refused (Q4);
* the lot rules keep each venue's behaviour (Q5): Binance floors, Hyperliquid
  rounds to nearest.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantbox.plugins.broker import _fills
from quantbox.plugins.broker._fills import STATUS_UNKNOWN, STATUS_WORKING, normalise_results
from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline
from quantbox.plugins.rebalancing.futures_rebalancer import FuturesRebalancer
from quantbox.trade import lots
from quantbox.trade.executor import execute_orders
from quantbox.trade.notifier import Alerts, RecordingNotifier

# ---------------------------------------------------------------------------
# The normaliser at the seam
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "row",
    [
        {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 100.0},
        {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 100.0, "status": None},
        {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 100.0, "status": "  "},
        {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 100.0, "status": "done"},
        {"symbol": "BTC", "side": "buy", "price": 100.0, "status": "FILLED"},
        {"symbol": "BTC", "side": "buy", "qty": float("nan"), "price": 100.0, "status": "PARTIAL"},
    ],
    ids=["no-status", "none-status", "blank-status", "foreign-status", "filled-no-qty", "partial-nan-qty"],
)
def test_the_normaliser_makes_an_unconfirmed_row_unknown(row):
    out = normalise_results(pd.DataFrame([row]))
    assert out.iloc[0]["status"] == STATUS_UNKNOWN


def test_the_normaliser_keeps_a_confirmed_row_and_cleans_its_status():
    out = normalise_results(pd.DataFrame([{"symbol": "BTC", "side": " SELL ", "qty": 1.0, "status": " skipped "}]))
    assert out.iloc[0]["status"] == "SKIPPED"
    assert out.iloc[0]["side"] == " SELL "  # the report shows the broker's own side


class _ThirdPartyBroker:
    """A broker outside this repo: its rows carry no status, or no filled quantity."""

    def __init__(self, rows):
        self.rows = rows

    def place_orders(self, orders):
        return pd.DataFrame(self.rows)

    def get_positions(self):
        return pd.DataFrame([{"symbol": "BTC", "qty": 1.0}])


def _buy(symbol="BTC", **extra):
    return {
        "Asset": symbol,
        "Action": "Buy",
        "Adjusted Quantity": 1.0,
        "Price": 100.0,
        "Notional Value": 100.0,
        "Min Notional": 10.0,
        "Order Status": "To be placed",
        "Executable": True,
        **extra,
    }


@pytest.mark.parametrize(
    "row",
    [
        {"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 100.0},
        {"symbol": "BTC", "side": "buy", "price": 100.0, "status": "FILLED"},
    ],
    ids=["absent-status", "absent-filled"],
)
def test_the_seam_books_no_fill_for_any_brokers_unconfirmed_row(row):
    report = execute_orders(
        broker=_ThirdPartyBroker([row]),
        orders_df=pd.DataFrame([_buy()]),
        trading_enabled=True,
        alert=Alerts(RecordingNotifier()).send,
    )
    assert report["summary"]["total_executed"] == 0
    assert report["summary"]["total_failed"] == 1
    (detail,) = report["orders_details"]
    assert detail["status"] == "FAILED"
    assert detail["fill_status"] == STATUS_UNKNOWN


# ---------------------------------------------------------------------------
# Binance live: through _fills like every other venue (Q9)
# ---------------------------------------------------------------------------


class _BinanceClient:
    def __init__(self, reply, refetched=None, raises=None):
        self.reply, self.refetched, self.raises = reply, refetched, raises
        self.get_calls = 0

    def create_order(self, **kwargs):
        if self.raises:
            raise self.raises
        return dict(self.reply)

    def get_order(self, symbol, orderId):
        self.get_calls += 1
        return dict(self.refetched if self.refetched is not None else self.reply)


def _binance_live(client):
    from quantbox.plugins.broker.binance_live import BinanceLiveBroker

    broker = BinanceLiveBroker.__new__(BinanceLiveBroker)
    broker.paper_trading = False
    broker.readonly = False
    broker._client = client
    return broker


def _binance_scaffold(client):
    from quantbox.plugins.broker.binance import BinanceBroker

    broker = BinanceBroker.__new__(BinanceBroker)
    broker.readonly = False
    broker.client = client
    return broker


BINANCE = {"binance_live": _binance_live, "binance_scaffold": _binance_scaffold}
NO_STATUS = {"orderId": 7, "origQty": "1.0", "executedQty": "1.0", "cummulativeQuoteQty": "100"}
NO_FILLED = {"orderId": 8, "status": "FILLED", "origQty": "1.0"}


def _one_sell():
    return pd.DataFrame([{"symbol": "BTCUSDC", "side": "sell", "qty": 1.0, "price": 100.0}])


@pytest.mark.parametrize("make", list(BINANCE.values()), ids=list(BINANCE))
@pytest.mark.parametrize("reply", [NO_STATUS, NO_FILLED], ids=["no-status", "no-filled"])
def test_binance_reports_an_unconfirmed_reply_as_unknown_with_no_fill(make, reply):
    row = make(_BinanceClient(reply)).place_orders(_one_sell()).iloc[0]
    assert row["status"] == STATUS_UNKNOWN
    assert row["qty"] == 0.0
    assert row["order_id"] == str(reply["orderId"])


def test_binance_live_books_every_fill_of_an_order_not_the_first():
    reply = {
        "orderId": 9,
        "status": "FILLED",
        "origQty": "1.0",
        "executedQty": "1.0",
        "cummulativeQuoteQty": "100.5",
        "fills": [
            {"qty": "0.4", "price": "100.0", "commission": "0.04", "commissionAsset": "USDC"},
            {"qty": "0.6", "price": "100.8333333", "commission": "0.06", "commissionAsset": "USDC"},
        ],
    }
    out = _binance_live(_BinanceClient(reply)).place_orders(_one_sell())
    assert len(out) == 1
    row = out.iloc[0]
    assert row["status"] == "FILLED"
    assert row["qty"] == pytest.approx(1.0)
    assert row["price"] == pytest.approx(100.5)
    assert row["commission"] == pytest.approx(0.1)


def test_binance_live_reports_a_resting_limit_as_working_not_dropped(monkeypatch):
    monkeypatch.setattr(_fills.time, "sleep", lambda s: None)
    client = _BinanceClient({"orderId": 10, "status": "NEW", "origQty": "1.0", "executedQty": "0.0"})
    out = _binance_live(client).place_orders(_one_sell())
    assert len(out) == 1
    assert out.iloc[0]["status"] == STATUS_WORKING
    assert out.iloc[0]["order_id"] == "10"
    assert client.get_calls >= 1  # re-read from the venue before calling it working


def test_binance_live_reports_a_failed_submission_instead_of_dropping_it():
    out = _binance_live(_BinanceClient({}, raises=RuntimeError("insufficient balance"))).place_orders(_one_sell())
    assert len(out) == 1
    assert out.iloc[0]["status"] == "FAILED"
    assert "insufficient balance" in out.iloc[0]["error"]


def test_a_simulated_brokers_fill_passes_the_normaliser_unchanged():
    from quantbox.plugins.broker.sim import SimPaperBroker

    broker = SimPaperBroker()
    broker.set_prices({"BTC": 100.0})
    rows = broker.place_orders(pd.DataFrame([{"symbol": "BTC", "side": "buy", "qty": 1.0, "price": 100.0}]))
    assert list(normalise_results(rows)["status"]) == ["FILLED"]


# ---------------------------------------------------------------------------
# TOM-901: the trapped-residual counter reads declared intent, never side
# ---------------------------------------------------------------------------


def test_a_declined_open_does_not_read_as_a_stuck_close():
    """The whole path: the futures rebalancer declares a short ADD, the paper venue refuses it at its limit.

    Before TOM-901 the refused SELL counted as a skipped close-out, so a book that
    only failed to GROW a short was frozen as a trapped residual.
    """
    from quantbox.plugins.broker.futures_paper import FuturesPaperBroker

    broker = FuturesPaperBroker(
        margin_balance=1_000.0,
        positions={"ETH": -1.0},
        entry_prices={"ETH": 100.0},
        prices={"ETH": 100.0},
        position_limits={"ETH": 50.0},
    )
    orders = FuturesRebalancer().generate_orders(
        weights={"ETH": -0.5}, broker=broker, params={"mode": "paper", "max_leverage": 1.0}
    )["orders"]
    (row,) = orders.to_dict("records")
    assert row["Action"] == "Sell" and row["opens_exposure"] is True and row["Executable"]

    notifier = RecordingNotifier()
    report = execute_orders(broker=broker, orders_df=orders, trading_enabled=True, alert=Alerts(notifier).send)

    assert [d["status"] for d in report["orders_details"]] == ["SKIPPED"]
    assert not report.get("frozen"), report.get("freeze_reasons")
    assert notifier.messages == []


def test_a_declined_close_is_still_a_trapped_residual_whatever_its_side():
    """The other half: a BUY that closes a short, refused, IS the trapped residual #81 exists for."""

    class _Skips:
        def place_orders(self, orders):
            return pd.DataFrame(
                [
                    {"symbol": o["symbol"], "side": o["side"], "qty": o["qty"], "status": "SKIPPED", "error": "min"}
                    for _, o in orders.iterrows()
                ]
            )

    notifier = RecordingNotifier()
    report = execute_orders(
        broker=_Skips(),
        orders_df=pd.DataFrame([_buy("ETH", reduce_only=True, opens_exposure=False)]),
        trading_enabled=True,
        alert=Alerts(notifier).send,
    )
    assert report.get("frozen") is True
    assert report["freeze_reasons"]["skipped_sell"] == 1
    assert len(notifier.messages) == 1 and "trapped residual" in notifier.messages[0]


# ---------------------------------------------------------------------------
# Q4: no private order generator, no default plugin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["paper", "live"])
def test_a_paper_or_live_run_without_a_rebalancer_is_refused(mode):
    with pytest.raises(ValueError, match="plugins.rebalancing"):
        TradingPipeline().run(
            mode=mode,
            asof="2026-10-13",
            params={"strategies": [{"name": "x"}]},
            data=None,
            store=None,
            broker=object(),
            risk=[],
            rebalancer=None,
        )


def test_the_pipeline_carries_no_sizing_code():
    """The search sweep the card asks for, as a check: sizing lives in the rebalancing plugins."""
    for name in ("_generate_orders", "_create_executable_orders", "_build_rebalancing", "_apply_risk_transforms"):
        assert not hasattr(TradingPipeline, name), name


# ---------------------------------------------------------------------------
# Q5: one lot module, each venue's rule kept
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("qty", [0.3, 1.23456789, 0.0001, 12.5, 7.0, 0.29, 1e-05])
@pytest.mark.parametrize("step", [0.1, 0.001, 1.0, 0.25, 1e-05])
def test_the_binance_floor_never_rounds_up(qty, step):
    out = lots.floor_to_step(qty, step)
    assert out <= qty
    assert qty - out < 10 ** -lots.step_decimals(step) + 1e-12


def test_the_binance_floor_reads_the_decimal_not_the_binary_float():
    assert lots.floor_to_step(0.3, 0.1) == 0.3
    assert lots.floor_to_step(0.29, 0.01) == 0.29


def test_hyperliquid_still_rounds_to_nearest_until_tom_1721():
    assert lots.round_to_decimals(0.0156, lots.amount_decimals(0.001)) == 0.016
    assert lots.floor_to_decimals(0.0156, lots.amount_decimals(0.001)) == 0.015
