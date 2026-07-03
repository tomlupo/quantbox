"""Tests for the shared transient-error retry helper (quantbox.retry).

No network, no capital — a fake exchange raises a synthetic 429 (ccxt-style
``RateLimitExceeded``) once, then returns markets, and we assert the broker
startup retries through it. ``time.sleep`` is patched out so tests are instant.
"""

from __future__ import annotations

import httpx
import pytest

from quantbox.retry import is_transient, retryable, with_retry


# --- ccxt-style synthetic exceptions (matched by class NAME in is_transient) ---
class RateLimitExceeded(Exception):
    """Stand-in for ccxt.RateLimitExceeded (HTTP 429)."""


class AuthenticationError(Exception):
    """Stand-in for ccxt.AuthenticationError."""


class InsufficientFunds(Exception):
    """Stand-in for ccxt.InsufficientFunds."""


class InvalidOrder(Exception):
    """Stand-in for ccxt.InvalidOrder (order rejected)."""


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make backoff instantaneous."""
    monkeypatch.setattr("quantbox.retry.time.sleep", lambda *_: None)


# ---------------------------------------------------------------- is_transient
def test_transient_errors_classified():
    assert is_transient(RateLimitExceeded("429"))
    assert is_transient(ConnectionError())
    assert is_transient(TimeoutError())
    assert is_transient(httpx.ConnectError("boom"))
    resp = httpx.Response(429, request=httpx.Request("GET", "https://x"))
    assert is_transient(httpx.HTTPStatusError("429", request=resp.request, response=resp))
    # Binance transient status codes on .code
    err = Exception()
    err.code = -1003
    assert is_transient(err)


def test_genuine_errors_never_retry():
    # These are fail-closed: auth / funds / rejected order must NOT retry.
    assert not is_transient(AuthenticationError("bad key"))
    assert not is_transient(InsufficientFunds("no money"))
    assert not is_transient(InvalidOrder("rejected"))
    assert not is_transient(ValueError("bad config"))


# ------------------------------------------------------------------ with_retry
def test_with_retry_succeeds_after_transient():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitExceeded("429")
        return "ok"

    assert with_retry(flaky, base_delay=0.0) == "ok"
    assert calls["n"] == 3


def test_with_retry_reraises_non_transient_immediately():
    calls = {"n": 0}

    def bad_auth():
        calls["n"] += 1
        raise AuthenticationError("bad key")

    with pytest.raises(AuthenticationError):
        with_retry(bad_auth, base_delay=0.0)
    assert calls["n"] == 1  # never retried


def test_with_retry_exhausts_and_reraises():
    calls = {"n": 0}

    def always_throttled():
        calls["n"] += 1
        raise RateLimitExceeded("429")

    with pytest.raises(RateLimitExceeded):
        with_retry(always_throttled, attempts=3, base_delay=0.0)
    assert calls["n"] == 3


def test_retryable_decorator_forwards_args():
    calls = {"n": 0}

    @retryable(attempts=4, base_delay=0.0)
    def add(a, b):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RateLimitExceeded("429")
        return a + b

    assert add(2, 3) == 5
    assert calls["n"] == 2


# --------------------------------------- broker load_markets: 429-then-200 ----
class _FlakyExchange:
    """ccxt-kraken stand-in that 429s on the first load_markets, then succeeds."""

    def __init__(self):
        self.load_calls = 0
        self.markets = {
            "BTC/USD": {
                "spot": True,
                "base": "BTC",
                "quote": "USD",
                "precision": {"amount": 3},
                "limits": {"amount": {"min": 0.0001}, "cost": {"min": 5.0}},
            },
        }

    def load_markets(self):
        self.load_calls += 1
        if self.load_calls == 1:
            raise RateLimitExceeded("429 Too Many Requests")
        return self.markets


def test_kraken_broker_retries_load_markets_on_429():
    from quantbox.plugins.broker.kraken import KrakenBroker

    ex = _FlakyExchange()
    broker = KrakenBroker(_exchange=ex)  # __post_init__ calls _load_markets

    assert ex.load_calls == 2  # first 429, retried once, then 200
    assert "BTC/USD" in broker._markets
