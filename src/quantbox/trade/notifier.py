"""The trading pipeline's one alert channel (TOM-1720).

Every alert a live run raises — an order failure, the reconciliation HALT gate,
the FROZEN dead-man, a trapped residual, zero fills, a reconciliation preflight,
persist or transition — goes through :class:`Alerts`, on every venue. Before
this, the pipeline asked the BROKER to send (``getattr(broker, "notify")``), and
only Hyperliquid had a ``notify``: on Kraken, a live book, every one of those
alerts was dropped without a trace.

Two rules hold for every alert:

* **Never silent when off.** No ``TELEGRAM_TOKEN`` / ``TELEGRAM_CHAT_ID`` logs
  "alerts off" once, and the run summary records it (:meth:`Alerts.record`).
* **Never fatal.** A send that fails is logged and recorded; the run goes on.

The notifier is injected: :class:`TradingPipeline` takes ``notifier=`` (a fake in
tests) and otherwise reads the environment.

Every alert is sent in Telegram's HTML parse mode, so the message owns its
``<b>`` markup and every DYNAMIC part (a venue error, a symbol, a reason, a book
key) goes through :func:`escape_html`. One unescaped ``<`` or ``&`` makes
Telegram answer 400 and the alert lands in ``failed`` instead of the chat
(TOM-1843).
"""

from __future__ import annotations

import html
import json
import logging
import os
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

TOKEN_ENV = "TELEGRAM_TOKEN"
CHAT_ENV = "TELEGRAM_CHAT_ID"


def escape_html(value: Any) -> str:
    """*value* as text that Telegram's HTML parse mode shows as written.

    Escapes ``&``, ``<`` and ``>`` — the three characters Telegram's HTML parser
    reads as markup. Quotes stay as they are: they mean nothing outside a tag.
    """
    return html.escape(str(value), quote=False)


class Notifier(Protocol):
    """Sends one message; raises when it was not delivered."""

    def send(self, message: str) -> None: ...


@dataclass(frozen=True)
class TelegramNotifier:
    """Telegram Bot API ``sendMessage``, HTML parse mode (the alerts' own markup)."""

    token: str = field(repr=False)
    chat_id: str
    timeout: float = 10.0

    def send(self, message: str) -> None:
        body = json.dumps({"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{self.token}/sendMessage",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            payload = json.loads(resp.read() or b"{}")
        if status != 200 or not payload.get("ok", False):
            raise RuntimeError(f"telegram answered {status}: {payload.get('description', payload)}")


@dataclass
class RecordingNotifier:
    """Keeps every message instead of sending it: tests and dry runs."""

    messages: list[str] = field(default_factory=list)

    def send(self, message: str) -> None:
        self.messages.append(message)


class Alerts:
    """One run's alert channel: sends through a notifier, records every outcome, never raises."""

    def __init__(self, notifier: Notifier | None, off_reason: str | None = None):
        self._notifier = notifier
        self.off_reason = off_reason if notifier is None else None
        self.sent: list[str] = []
        self.failed: list[dict[str, str]] = []
        self.dropped: list[str] = []
        if notifier is None:
            logger.warning("alerts off: %s", self.off_reason or "no notifier")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Alerts:
        env = os.environ if env is None else env
        token, chat_id = env.get(TOKEN_ENV, ""), env.get(CHAT_ENV, "")
        if not token or not chat_id:
            missing = [name for name, value in ((TOKEN_ENV, token), (CHAT_ENV, chat_id)) if not value]
            return cls(None, off_reason=f"{' and '.join(missing)} not set")
        return cls(TelegramNotifier(token=token, chat_id=chat_id))

    @property
    def on(self) -> bool:
        return self._notifier is not None

    def send(self, message: str, *, site: str) -> bool:
        """Send *message*; True when delivered. *site* names the alert in the run record."""
        if self._notifier is None:
            self.dropped.append(site)
            return False
        try:
            self._notifier.send(message)
        except Exception as exc:  # noqa: BLE001 - an alert must never fail the run
            logger.warning("alert %s not sent: %s", site, exc)
            self.failed.append({"site": site, "error": str(exc)})
            return False
        self.sent.append(site)
        return True

    def record(self) -> dict[str, Any]:
        """The run summary's ``alerts`` entry."""
        return {
            "on": self.on,
            "off_reason": self.off_reason,
            "sent": list(self.sent),
            "failed": list(self.failed),
            "dropped": list(self.dropped),
        }
