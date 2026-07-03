"""Shared retry / backoff helper for transient upstream errors.

One helper — :func:`with_retry` — used across data-source and broker plugins so
a transient throttle (HTTP 429 / ccxt ``RateLimitExceeded`` / ``NetworkError`` /
a dropped socket) is retried with exponential backoff + jitter instead of
aborting a live daily run.

SCOPE — this is the whole safety contract:

* Only **transient** upstream errors retry (see :func:`is_transient`): rate
  limits, DDoS-protection throttles, timeouts, connection resets, and the
  transient HTTP/exchange status codes.
* Genuine failures are **never** retried — authentication errors, insufficient
  funds, and rejected / invalid orders propagate immediately so the caller
  fails closed. Retrying an auth error just burns the rate-limit budget; retrying
  an order-rejected can (with a non-idempotent endpoint) double-submit. Both are
  explicitly vetoed in :data:`_NON_RETRYABLE_EXC_NAMES`.

This module is the single source of truth. ``plugins/datasources/_utils.py`` and
the broker plugins import from here rather than carrying their own copies.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ccxt / exchange exception class names that indicate a transient condition and
# are safe to retry. Matched by class NAME so we never need a hard ccxt import
# (ccxt is an optional dependency; brokers import it lazily).
_TRANSIENT_EXC_NAMES = frozenset(
    {
        "RateLimitExceeded",
        "ExchangeNotAvailable",
        "RequestTimeout",
        "NetworkError",
        "DDoSProtection",
    }
)

# Exception class names that must NEVER retry, even if a broad handler upstream
# would otherwise treat them as retryable. These are genuine, non-transient
# failures — retrying wastes the rate-limit budget (auth) or risks a duplicate
# side-effect (order submission). This veto is checked FIRST in is_transient.
_NON_RETRYABLE_EXC_NAMES = frozenset(
    {
        "AuthenticationError",
        "PermissionDenied",
        "AccountSuspended",
        "AccountNotEnabled",
        "InsufficientFunds",
        "InvalidOrder",
        "OrderNotFound",
        "OrderImmediatelyFillable",
        "BadSymbol",
        "BadRequest",
        "InvalidAddress",
        "ArgumentsRequired",
        "NotSupported",
    }
)

# Transient HTTP / Binance API status codes (``exc.code``): -1003 rate limit,
# -1001 disconnected, -1000 unknown-but-retried, 503/504 upstream unavailable.
_TRANSIENT_CODES = frozenset({-1003, -1001, -1000, 503, 504})


def is_transient(exc: BaseException) -> bool:
    """Return True iff *exc* is a transient upstream error worth retrying.

    The non-retryable veto (auth / insufficient-funds / rejected-order) is
    checked first, so a genuine failure never retries even if it happens to
    subclass a broad exchange-error base.
    """
    name = type(exc).__name__
    # Fail-closed: genuine errors never retry.
    if name in _NON_RETRYABLE_EXC_NAMES:
        return False
    if isinstance(exc, (ConnectionError, TimeoutError, OSError, httpx.TransportError)):
        return True
    # ccxt HTTP 429 surfaces as RateLimitExceeded; httpx surfaces it as an
    # HTTPStatusError carrying a response.
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    if name in _TRANSIENT_EXC_NAMES:
        return True
    code = getattr(exc, "code", None)
    return code in _TRANSIENT_CODES


def _default_on_retry(exc: BaseException, attempt: int, attempts: int, delay: float, label: str | None) -> None:
    logger.warning(
        "Transient error%s (attempt %d/%d), retrying in %.1fs: %s: %s",
        f" on {label}" if label else "",
        attempt,
        attempts,
        delay,
        type(exc).__name__,
        exc,
    )


def with_retry(
    func: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retry_on: Callable[[BaseException], bool] = is_transient,
    on_retry: Callable[[BaseException, int, int, float, str | None], None] | None = None,
    label: str | None = None,
) -> T:
    """Call *func* (a zero-arg callable), retrying on transient errors.

    Exponential backoff with full jitter: attempt *n* (1-indexed) sleeps
    ``min(max_delay, base_delay * 2**(n-1))`` plus, when *jitter* is set, a
    random ``[0, base_delay)`` term to de-correlate concurrent retriers.

    Only errors for which ``retry_on(exc)`` is True are retried; everything else
    (and the final attempt) re-raises immediately, so genuine failures fail
    closed. Wrap the work in a ``lambda``/``functools.partial`` to pass args::

        markets = with_retry(exchange.load_markets, label="hyperliquid.load_markets")
        markets = with_retry(lambda: exchange.load_markets(reload=True))

    Returns *func*'s return value; re-raises the last exception if every attempt
    fails or the error is non-transient.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    notify = on_retry or _default_on_retry
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except BaseException as exc:  # noqa: BLE001 — re-raised unless transient
            last_exc = exc
            if attempt >= attempts or not retry_on(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            if jitter:
                delay += random.uniform(0, base_delay)
            notify(exc, attempt, attempts, delay, label)
            time.sleep(delay)
    # Unreachable: the final attempt either returns or re-raises above.
    raise last_exc  # type: ignore[misc]


def retryable(**config):
    """Decorator form of :func:`with_retry`.

    ``@retryable(attempts=4)`` wraps a function so each call is retried on
    transient errors with the same backoff contract. Accepts every keyword
    :func:`with_retry` does.

        @retryable(attempts=3, label="fetch_ohlcv")
        def fetch(symbol): ...
    """

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        cfg = dict(config)
        cfg.setdefault("label", fn.__name__)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            return with_retry(lambda: fn(*args, **kwargs), **cfg)

        return wrapper

    return deco


# Pre-built decorator matching the legacy ``retry_transient`` contract
# (4 attempts, exp backoff capped at 30s, transient-only, reraise). Kept as a
# module-level singleton so data-source plugins can ``@retry_transient`` cheaply.
retry_transient = retryable()
