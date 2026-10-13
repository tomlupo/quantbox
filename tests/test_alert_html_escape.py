"""Alert text is valid Telegram HTML at every alert site (TOM-1843).

`TelegramNotifier` sends with `parse_mode: HTML`. The alert sites put venue
error strings, symbols, reasons and book keys into the message unescaped, so one
`<` or `&` made Telegram answer 400 and the alert landed in
`notes.alerts.failed` instead of the chat. Each test below feeds a site hostile
text and checks two things on the message the notifier receives: Telegram's
HTML parser would accept it, and the site's own `<b>` headline is still markup.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pandas as pd
import pytest

from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline
from quantbox.plugins.publisher import telegram as publisher
from quantbox.trade.notifier import RecordingNotifier
from quantbox.trade.reconciliation import ReconEnforcementError, open_cycle
from quantbox.trade.reconciliation import cycle as cycle_mod

HOSTILE = "qty <0.001 & step<0.01>"
HOSTILE_ESCAPED = "qty &lt;0.001 &amp; step&lt;0.01&gt;"

# The tags Telegram's HTML mode accepts that our messages use, and its entities.
_TAG = re.compile(r"</?(b|i|pre|code)>")
_ENTITY = re.compile(r"&(lt|gt|amp|quot|#\d+);")


def assert_telegram_html(message: str) -> None:
    """Fail when Telegram's HTML parse mode would refuse *message* (400)."""
    bare = _ENTITY.sub("", _TAG.sub("", message))
    assert "<" not in bare and ">" not in bare, f"unescaped markup in: {message!r}"
    assert "&" not in bare, f"unescaped & in: {message!r}"


def _pipeline() -> TradingPipeline:
    pipe = TradingPipeline()
    pipe.notifier = RecordingNotifier()
    return pipe


def _order(symbol: str, action: str = "Buy", **extra) -> dict:
    return {
        "Asset": symbol,
        "Action": action,
        "Adjusted Quantity": 1.0,
        "Price": 100.0,
        "Notional Value": 100.0,
        "Order Status": "To be placed",
        "Executable": True,
        **extra,
    }


class _Venue:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def place_orders(self, orders):
        return pd.DataFrame(self._rows)


def _execute(pipe: TradingPipeline, broker, orders: list[dict]) -> dict:
    return pipe._execute_orders(
        broker=broker, orders_df=pd.DataFrame(orders), stable_coin="USD", trading_enabled=True, mode="live"
    )


def test_an_order_failure_with_a_venue_error_carrying_lt_and_amp_is_sent_escaped():
    """The card's acceptance case: the venue error reaches Telegram as text."""
    pipe = _pipeline()
    venue = _Venue(
        [
            {"symbol": "BTC", "side": "buy", "qty": 0.0, "price": 0.0, "status": "FILLED", "order_id": "1"},
            {"symbol": "ETH", "side": "buy", "qty": 0.0, "price": 0.0, "status": "FAILED", "error": HOSTILE},
        ]
    )
    _execute(pipe, venue, [_order("BTC"), _order("ETH")])

    assert pipe._alert_channel().record()["sent"] == ["order_failures"]
    (message,) = pipe.notifier.messages
    assert message.startswith("❌ <b>1 ORDER(S) FAILED</b>\n")
    assert f"BUY ETH ({HOSTILE_ESCAPED})" in message
    assert HOSTILE not in message
    assert_telegram_html(message)


def test_a_frozen_book_escapes_its_symbols_and_reasons():
    pipe = _pipeline()
    suppressed = {"Executable": False, "Order Status": "Below min notional", "Reason": HOSTILE}
    orders = [_order("A<B", action="Sell", **suppressed), _order("C&D", action="Sell", **suppressed)]
    report = _execute(pipe, _Venue([]), orders)

    assert report.get("frozen") is True
    (message,) = pipe.notifier.messages
    assert message.startswith("🧊 <b>REBALANCER FROZEN — 0 of 2 orders executable</b>\n")
    assert "A&lt;B" in message and "C&amp;D" in message
    assert HOSTILE_ESCAPED in message
    assert_telegram_html(message)


def test_a_trapped_residual_escapes_the_venue_skip_reason():
    pipe = _pipeline()
    venue = _Venue([{"symbol": "ADA", "side": "sell", "qty": 0.0, "price": 0.0, "status": "SKIPPED", "error": HOSTILE}])
    report = _execute(pipe, venue, [_order("ADA", action="Sell")])

    assert report.get("frozen") is True
    (message,) = pipe.notifier.messages
    assert message.startswith("🧊 <b>REBALANCER FROZEN — trapped residual</b>\n")
    assert HOSTILE_ESCAPED in message
    assert_telegram_html(message)


def test_zero_fills_escapes_the_submitted_symbols():
    pipe = _pipeline()
    _execute(pipe, _Venue([]), [_order("X<Y&Z")])

    (message,) = pipe.notifier.messages
    assert message.startswith("⚠️ <b>0 FILLS for 1 orders</b>\n")
    assert "X&lt;Y&amp;Z" in message
    assert_telegram_html(message)


def test_a_recon_halt_keeps_its_headline():
    """No dynamic text beyond a count; pinned so the sweep covers every site."""
    pipe = _pipeline()
    pipe._execute_orders(
        broker=_Venue([]),
        orders_df=pd.DataFrame([_order("A<B")]),
        stable_coin="USD",
        trading_enabled=True,
        mode="live",
        gate_orders_allowed=False,
    )
    (message,) = pipe.notifier.messages
    assert message.startswith("🛑 <b>RECON ENFORCE — HALT</b>\n")
    assert_telegram_html(message)


# --- the reconciliation cycle's sites ---------------------------------------


class _Alerts:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    def __call__(self, message, *, site):
        self.sent.append((site, message))
        return True


class _BrokerWithAPhantom:
    """Holds a position the book never ordered: a break, so the state moves."""

    def get_positions(self):
        return pd.DataFrame({"symbol": ["BT<C&"], "qty": [5.0]})

    def get_market_snapshot(self, symbols):
        return pd.DataFrame({"symbol": list(symbols), "mid": [100.0] * len(symbols)})


def _enforce(tmp_path, book_key: str) -> dict:
    return {
        "reconciliation": {
            "book_key": book_key,
            "data_dir": str(tmp_path),
            "mode": "enforce",
            "enforce_acknowledged": True,
        }
    }


def _close(run, alerts, broker) -> None:
    run.close(
        broker=broker,
        final_weights={},
        orders_df=pd.DataFrame(),
        execution_report={"orders_details": [], "summary": {}},
        portfolio_value=1000.0,
        stable_coin="USDC",
        alert=alerts,
    )


def test_recon_transition_and_preflight_escape_the_book_key_and_symbols(tmp_path):
    alerts = _Alerts()
    params = _enforce(tmp_path, "b<k&")
    _close(open_cycle(params, run_id="r1", asof="d", alert=alerts), alerts, _BrokerWithAPhantom())
    open_cycle(params, run_id="r2", asof="d", alert=alerts)  # the next cycle reads the persisted state

    sites = {site: message for site, message in alerts.sent}
    assert set(sites) == {"recon_transition", "recon_preflight"}
    assert "RECON [b&lt;k&amp;]" in sites["recon_transition"]
    assert "Symbol(s): BT&lt;C&amp;" in sites["recon_transition"]
    assert "RECON ENFORCE [b&lt;k&amp;]" in sites["recon_preflight"]
    for message in sites.values():
        assert_telegram_html(message)


def test_recon_persist_escapes_the_book_key_and_the_error(monkeypatch, tmp_path):
    def _boom(path, text):
        raise OSError("disk <full> & gone")

    monkeypatch.setattr(cycle_mod, "atomic_write_text", _boom)
    alerts = _Alerts()
    run = open_cycle(_enforce(tmp_path, "b<k&"), run_id="r1", asof="d", alert=alerts)
    with pytest.raises(ReconEnforcementError):
        _close(run, alerts, _BrokerWithAPhantom())

    ((site, message),) = alerts.sent
    assert site == "recon_persist"
    assert message.startswith("🛑 <b>RECON ENFORCE [b&lt;k&amp;] — state persist FAILED</b>\n")
    assert "disk &lt;full&gt; &amp; gone" in message
    assert_telegram_html(message)


# --- the Telegram publisher: the same parse mode, its own messages ------------


def test_the_publisher_escapes_failed_orders_and_keeps_its_markup():
    message = publisher._format_failed_orders(
        [{"action": "buy", "symbol": "A&B", "quantity": 1.0, "error": HOSTILE}], pipeline_name="p<1>"
    )
    assert message.startswith("<b>Trading Bot Report</b>\nPipeline: <b>p&lt;1&gt;</b>")
    assert f"BUY: A&amp;B 1 ({HOSTILE_ESCAPED})" in message
    assert_telegram_html(message)


def test_the_publisher_escapes_every_message_it_sends(monkeypatch):
    """Through `publish`: the new-token alert is escaped BEFORE its **bold** becomes <b>."""
    sent: list[str] = []
    monkeypatch.setattr(
        publisher, "_send_telegram_message", lambda token, chat, msg, parse_mode="HTML": sent.append(msg)
    )
    result = SimpleNamespace(
        run_id="r1",
        pipeline_name="p<1>",
        mode="live",
        metrics={},
        notes={
            "artifact_payload": {
                "portfolio_value": 1000.0,
                "execution_summary": {
                    "executed_orders": [{"action": "buy", "symbol": "A&B", "quantity": 1.0, "executed_price": 2.0}],
                    "failed_orders": [{"action": "sell", "symbol": "C<D", "quantity": 1.0, "error": HOSTILE}],
                },
                "rebalancing_table": [
                    {"asset": "E&F", "current_weight": 0.1, "target_weight": 0.2, "weight_delta": 0.1}
                ],
            },
            "new_token_alert": "**New <tokens> & co:**\n  • **X&Y**",
        },
    )
    publisher.TelegramPublisher().publish(result, {"telegram_token": "t", "telegram_chat_id": "c"})

    assert len(sent) == 5
    for message in sent:
        assert_telegram_html(message)
    assert sent[-1] == "<b>New &lt;tokens&gt; &amp; co:</b>\n  • <b>X&amp;Y</b>"


def test_the_validator_is_not_vacuous():
    """The check above must fail on the bug it guards."""
    with pytest.raises(AssertionError):
        assert_telegram_html(f"❌ <b>1 ORDER(S) FAILED</b>\nBUY ETH ({HOSTILE})")
