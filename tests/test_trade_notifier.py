"""The trading pipeline owns one alert channel (TOM-1720).

Before: the pipeline asked the broker to send (`getattr(broker, "notify")`) and
only Hyperliquid had a `notify`, so on Kraken — a live book — every HALT, FROZEN
and order-failure alert was dropped without a trace. These tests drive the REAL
brokers of the three live venues (their exchange stubbed) through the pipeline's
own alert sites, and assert the alert reaches the injected notifier on each.
"""

from __future__ import annotations

import inspect
import io
import json
import logging

import pandas as pd
import pytest

from quantbox.plugins.broker.binance_futures import BinanceFuturesBroker
from quantbox.plugins.broker.hyperliquid import HyperliquidBroker
from quantbox.plugins.broker.kraken import KrakenBroker
from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline
from quantbox.trade.executor import execute_orders
from quantbox.trade.notifier import Alerts, RecordingNotifier, TelegramNotifier


def _hyperliquid():
    return HyperliquidBroker.__new__(HyperliquidBroker)


def _binance_futures():
    return BinanceFuturesBroker.__new__(BinanceFuturesBroker)


def _kraken():
    class _Ex:
        def load_markets(self):
            return {}

    return KrakenBroker(_exchange=_Ex())


VENUES = {"hyperliquid": _hyperliquid, "binance_futures": _binance_futures, "kraken": _kraken}


def _pipeline() -> TradingPipeline:
    pipe = TradingPipeline()
    pipe.notifier = RecordingNotifier()
    return pipe


def _orders(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": sym,
                "Action": "Buy",
                "Adjusted Quantity": 1.0,
                "Price": 100.0,
                "Notional Value": 100.0,
                "Order Status": "To be placed",
                "Executable": True,
            }
            for sym in ("BTC", "ETH")[:n]
        ]
    )


def _frozen_orders() -> pd.DataFrame:
    """Every intended order suppressed below the venue minimum: the dead-man
    (the live 2026-06-15 shape, as in test_futures_rebalancer_freeze)."""
    return pd.DataFrame(
        [
            {
                "Asset": sym,
                "Action": "Sell",
                "Adjusted Quantity": 0.0,
                "Price": 1.0,
                "Notional Value": notional,
                "Order Status": "Below min notional",
                "Executable": False,
            }
            for sym, notional in (("ADA", 2.90), ("ETH", 3.78))
        ]
    )


@pytest.mark.parametrize("venue", sorted(VENUES))
def test_a_recon_halt_alerts_on_every_venue(venue):
    pipe = _pipeline()
    report = execute_orders(
        alert=Alerts(pipe.notifier).send,
        broker=VENUES[venue](),
        orders_df=_orders(),
        trading_enabled=True,
        gate_orders_allowed=False,
    )
    assert report["recon_gated"] == "halt"
    assert len(pipe.notifier.messages) == 1
    assert "RECON ENFORCE — HALT" in pipe.notifier.messages[0]


@pytest.mark.parametrize("venue", sorted(VENUES))
def test_a_frozen_book_alerts_on_every_venue(venue):
    pipe = _pipeline()
    report = execute_orders(
        alert=Alerts(pipe.notifier).send,
        broker=VENUES[venue](),
        orders_df=_frozen_orders(),
        trading_enabled=True,
    )
    assert report.get("frozen") is True
    assert len(pipe.notifier.messages) == 1
    assert "REBALANCER FROZEN" in pipe.notifier.messages[0]


def test_no_broker_carries_a_notify_or_its_own_telegram():
    """The search sweep the card asks for, as a check: the brokers alert nothing themselves."""
    for cls in (HyperliquidBroker, BinanceFuturesBroker, KrakenBroker):
        assert not hasattr(cls, "notify"), cls.__name__
        source = inspect.getsource(inspect.getmodule(cls))
        assert "api.telegram.org" not in source, cls.__name__
        assert "send_telegram" not in source, cls.__name__
    pipeline_source = inspect.getsource(inspect.getmodule(TradingPipeline))
    assert '"notify"' not in pipeline_source


def test_a_working_order_is_not_alerted_as_a_failure():
    """The PR #171 scar, on the pipeline's channel: a resting order is not a rejection."""

    class _Venue:
        def place_orders(self, orders):
            return pd.DataFrame(
                [
                    {"symbol": "BTC", "side": "buy", "qty": 0.0, "price": 0.0, "status": "WORKING", "order_id": "1"},
                    {"symbol": "ETH", "side": "buy", "qty": 0.0, "price": 0.0, "status": "FAILED", "error": "rejected"},
                ]
            )

    pipe = _pipeline()
    report = execute_orders(
        alert=Alerts(pipe.notifier).send, broker=_Venue(), orders_df=_orders(), trading_enabled=True
    )
    assert report["summary"]["total_working"] == 1
    assert len(pipe.notifier.messages) == 1
    assert "ETH" in pipe.notifier.messages[0]
    assert "BTC" not in pipe.notifier.messages[0]


def test_alerts_off_is_logged_once_and_recorded(caplog):
    with caplog.at_level(logging.WARNING, logger="quantbox.trade.notifier"):
        alerts = Alerts.from_env({})
        assert alerts.send("HALT", site="recon_halt") is False
        assert alerts.send("FROZEN", site="frozen") is False
    off_lines = [r for r in caplog.records if "alerts off" in r.getMessage()]
    assert len(off_lines) == 1
    assert "TELEGRAM_TOKEN" in off_lines[0].getMessage()
    record = alerts.record()
    assert record["on"] is False
    assert "TELEGRAM_TOKEN" in record["off_reason"]
    assert record["dropped"] == ["recon_halt", "frozen"]


def test_a_failing_notifier_never_fails_the_run():
    class _Down:
        def send(self, message):
            raise ConnectionError("telegram unreachable")

    alerts = Alerts(_Down())
    # The 0-fills site was the one unwrapped alert: a raising notify failed the run.
    report = execute_orders(
        alert=alerts.send,
        broker=type("_Empty", (), {"place_orders": lambda self, o: pd.DataFrame()})(),
        orders_df=_orders(),
        trading_enabled=True,
    )
    assert report["summary"]["total_executed"] == 0
    record = alerts.record()
    assert record["sent"] == []
    assert [f["site"] for f in record["failed"]] == ["zero_fills"]


def test_the_environment_selects_telegram_and_its_send_is_a_post(monkeypatch):
    alerts = Alerts.from_env({"TELEGRAM_TOKEN": "tok", "TELEGRAM_CHAT_ID": "42"})
    assert alerts.on
    seen = {}

    class _Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    def fake_urlopen(req, timeout):
        seen["url"], seen["body"] = req.full_url, json.loads(req.data)
        return _Resp(b'{"ok": true}')

    monkeypatch.setattr("quantbox.trade.notifier.urllib.request.urlopen", fake_urlopen)
    assert alerts.send("🛑 HALT", site="recon_halt") is True
    assert seen["url"] == "https://api.telegram.org/bottok/sendMessage"
    assert seen["body"] == {"chat_id": "42", "text": "🛑 HALT", "parse_mode": "HTML"}
    assert "tok" not in repr(TelegramNotifier(token="tok", chat_id="42"))
