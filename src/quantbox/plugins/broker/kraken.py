"""
Kraken Spot Broker Plugin

Live **spot** trading on Kraken via ccxt (``ccxt.kraken`` handles HMAC-SHA512
request signing and the monotonic nonce — no raw REST signing needed).

Scope: spot, **long-only, no leverage**. There is no ``get_equity`` /
signed-position / margin surface — ``get_positions`` reports current asset
balances mapped to ``[symbol, qty]`` (always non-negative) and ``get_cash``
reports the quote-currency balance.

## Authentication

Set environment variables (one API key per bot — Kraken's nonce is a single
monotonic counter per key, so two processes sharing a key will collide):

    KRAKEN_API_KEY=...      # public key
    KRAKEN_API_SECRET=...   # private key (base64)

There is **no passphrase** on Kraken (that's a Coinbase pattern).

## Quote currency

``quote_asset`` defaults to **USD** — Kraken's deep native large-cap books are
``*/USD``; USDC books are thin below the top names. Legacy Kraken asset codes
(``XXBT``/``ZUSD``) and earn/staking balances (``DOT.S``/``ETH.F``) are
normalised/filtered (see :func:`normalize_kraken_asset`).
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantbox.contracts import PluginMeta
from quantbox.retry import with_retry

from ..datasources.kraken_data import KRAKEN_BALANCE_SUFFIXES, normalize_kraken_asset
from ._fills import STATUS_WORKING, resolve_fill, trade_fee, trade_fee_currency

try:
    import ccxt
except ImportError:  # pragma: no cover
    ccxt = None  # type: ignore

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

# Kraken's Ledgers endpoint pages 50 rows at a time via an `ofs` offset. ccxt
# does NOT paginate — see KrakenBroker.fetch_ledger_entries.
LEDGER_PAGE_SIZE = 50
# ~10k entries. A ledger longer than this is real, but so is a pagination bug;
# we refuse to silently return a truncated history either way.
LEDGER_MAX_PAGES = 200

# Raw Kraken ledger types that represent a movement of value in/out of the
# trading account (as opposed to `trade`, `margin`, `rollover`, `staking`, ...).
# `transfer` is included because Kraken books some genuinely external movements
# under it — but it is NOT assumed external; the caller must classify it.
CASHFLOW_LEDGER_TYPES = frozenset({"deposit", "withdrawal", "transfer"})

# Quote-equivalent stablecoins, grouped BY PEG. A balance in a stablecoin pegged
# to the book's configured quote currency is cash-equivalent *dust*, NOT a trading
# position to liquidate. Treating e.g. a 0.0014 USDC residue on a USD book as a
# sellable position emits a guaranteed below-minimum order (Kraken's USDC min is
# 5.0) that can only fail — and on a low-exposure day it becomes the *only*
# executable order, masking an otherwise-clean no-trade signal.
#
# The exclusion is SCOPED TO THE QUOTE'S PEG, not "any stablecoin": on a USD-quoted
# book a EUR-pegged stable (EURT/EURC) is a genuine FX position that must remain
# liquidatable, and a *depegged* token (USTC) is not cash-equivalent at all — both
# must keep their place on the exit path. Excluded from get_positions analogous to
# the staking-suffix skip in _fetch_balances.
USD_PEGGED_STABLECOINS = frozenset(
    {
        "USDC",
        "USDT",
        "DAI",
        "TUSD",
        "BUSD",
        "FDUSD",
        "USDP",
        "GUSD",
        "PYUSD",
        "USD1",
        "USDS",
        "USDD",
        "FRAX",
        "LUSD",
    }
)
EUR_PEGGED_STABLECOINS = frozenset(
    {
        "EURT",
        "EURC",
        "EURS",
        "EURR",
    }
)
# Map a normalised fiat quote currency to the set of stables pegged to it. A quote
# not listed here (or a crypto quote) yields an empty set — nothing is excluded,
# which is the safe default (every balance stays liquidatable).
_PEG_BY_QUOTE = {
    "USD": USD_PEGGED_STABLECOINS,
    "EUR": EUR_PEGGED_STABLECOINS,
}


def quote_equivalent_stablecoins(quote_asset: str) -> frozenset:
    """Stablecoins that are cash-equivalent dust for the given quote currency.

    Scoped to the quote's peg: a USD book excludes USD-pegged stables only; a EUR
    book excludes EUR-pegged stables only. Depegged tokens (e.g. USTC) belong to no
    peg set and are therefore never excluded.
    """
    return _PEG_BY_QUOTE.get(normalize_kraken_asset(quote_asset), frozenset())


class _SkipOrder:
    """Sentinel: an order intentionally NOT placed (sub-minimum / sub-precision
    dust). A clean no-op — neither a fill nor a failure."""

    __slots__ = ()


# Singleton sentinel returned by _place_one for orders too small to place.
SKIP_ORDER = _SkipOrder()


@dataclass
class KrakenBroker:
    """Kraken spot broker adapter (ccxt, long-only).

    Interface (``BrokerPlugin``):
    - ``get_cash``: quote-currency (USD/USDC/EUR) balance as ``{currency: amount}``
    - ``get_positions``: non-quote asset balances as ``[symbol, qty]``
    - ``get_market_snapshot``: mid + min_qty/step_size/min_notional per symbol
    - ``place_orders``: MARKET orders by default, LIMIT when ``price`` is given
    - ``fetch_fills``: trade history since a timestamp
    """

    meta = PluginMeta(
        name="kraken.spot.v1",
        kind="broker",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Kraken spot broker adapter (ccxt, long-only, no leverage)",
        tags=("kraken", "broker", "crypto", "spot"),
        capabilities=("paper", "live", "crypto", "spot"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "quote_asset": {"type": "string", "default": "USD"},
                "api_key_env": {"type": "string", "default": "KRAKEN_API_KEY"},
                "api_secret_env": {"type": "string", "default": "KRAKEN_API_SECRET"},
                "readonly": {"type": "boolean", "default": False},
            },
        },
        examples=("plugins:\n  broker:\n    name: kraken.spot.v1\n    params_init:\n      quote_asset: USD",),
    )

    # Config
    quote_asset: str = "USD"
    api_key_env: str = "KRAKEN_API_KEY"
    api_secret_env: str = "KRAKEN_API_SECRET"
    readonly: bool = False

    # State (injectable for tests)
    _exchange: Any = field(default=None, repr=False)
    _markets: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self._exchange is None:
            if ccxt is None:
                raise ImportError("ccxt is required for KrakenBroker. pip install ccxt")
            api_key = os.environ.get(self.api_key_env)
            api_secret = os.environ.get(self.api_secret_env)
            if not api_key or not api_secret:
                raise OSError(f"Missing env vars: {self.api_key_env} / {self.api_secret_env}")
            self._exchange = ccxt.kraken(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                }
            )
        self._load_markets()

    def _load_markets(self) -> None:
        try:
            # Retry a transient 429 / DDoS-protection throttle with backoff so a
            # momentary rate limit doesn't blank the market map on startup.
            self._markets = with_retry(self._exchange.load_markets, label="kraken.load_markets") or {}
            logger.info("Connected to Kraken spot, %d markets loaded", len(self._markets))
        except Exception as e:  # pragma: no cover - network
            logger.error("Failed to load Kraken markets: %s", e)
            self._markets = {}

    def describe(self) -> dict[str, Any]:
        return {
            "name": "KrakenBroker",
            "exchange": "Kraken",
            "type": "spot",
            "quote_asset": self.quote_asset,
            "long_only": True,
            "leverage": False,
            "readonly": self.readonly,
        }

    # ------------------------------------------------------------------
    # Balances
    # ------------------------------------------------------------------

    def _fetch_balances(self) -> dict[str, float]:
        """Total balance per canonical asset, earn/staking folded into spot.

        ccxt already normalises most Kraken codes, but we defensively normalise
        legacy codes (``XXBT``->``BTC``) and fold ``.S/.F/...`` earn balances
        into their spot asset.
        """
        try:
            balance = self._exchange.fetch_balance()
        except Exception as e:  # pragma: no cover - network
            logger.error("Balance fetch failed: %s", e)
            return {}

        totals = balance.get("total", {}) or {}
        out: dict[str, float] = {}
        for raw_code, amount in totals.items():
            qty = float(amount or 0.0)
            if qty == 0.0:
                continue
            # Skip staking/earn sub-balances entirely (they are not spot-tradable
            # and folding them in would overstate the sellable position).
            if any(str(raw_code).upper().endswith(s) for s in KRAKEN_BALANCE_SUFFIXES):
                continue
            asset = normalize_kraken_asset(raw_code)
            out[asset] = out.get(asset, 0.0) + qty
        return out

    def get_cash(self) -> dict[str, float]:
        """Quote-currency balance as ``{currency: amount}``."""
        balances = self._fetch_balances()
        quote = normalize_kraken_asset(self.quote_asset)
        return {quote: float(balances.get(quote, 0.0))}

    def get_positions(self) -> pd.DataFrame:
        """Non-quote asset balances as ``[symbol, qty]`` (long-only, qty >= 0)."""
        balances = self._fetch_balances()
        quote = normalize_kraken_asset(self.quote_asset)
        # Only stables pegged to THIS book's quote are cash-equivalent dust; a
        # EUR-stable on a USD book (or a depegged token) stays liquidatable.
        quote_stables = quote_equivalent_stablecoins(self.quote_asset)
        rows = []
        for asset, qty in balances.items():
            if asset == quote:
                continue
            # Quote-equivalent stablecoins are cash-equivalent dust, not trading
            # positions to liquidate. A USDC residue on a USD-quoted book would
            # otherwise become a guaranteed sub-minimum sell.
            if asset in quote_stables:
                if qty > 0:
                    logger.info(
                        "Skipping quote-equivalent stablecoin balance %s=%s "
                        "(cash-equivalent dust, not a liquidatable position)",
                        asset,
                        qty,
                    )
                continue
            if qty > 0:
                rows.append({"symbol": asset, "qty": float(qty)})
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["symbol", "qty"])

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def _market_symbol(self, symbol: str) -> str | None:
        """Resolve a base ticker to a ccxt market symbol (``BTC/USD``)."""
        candidate = f"{symbol}/{self.quote_asset}"
        if candidate in self._markets:
            return candidate
        # Fall back: scan for a market whose normalised base + quote matches.
        target = normalize_kraken_asset(symbol)
        quote = normalize_kraken_asset(self.quote_asset)
        for ms, m in self._markets.items():
            if not m.get("spot", True):
                continue
            if normalize_kraken_asset(m.get("base", "")) == target and (
                normalize_kraken_asset(m.get("quote", "")) == quote
            ):
                return ms
        return None

    def get_market_snapshot(self, symbols: list[str]) -> pd.DataFrame:
        """Mid price + lot/notional limits per symbol."""
        rows = []
        for sym in symbols:
            ms = self._market_symbol(sym)
            mid = None
            min_qty = 0.0
            step_size = 0.0
            min_notional = 0.0
            if ms is not None:
                try:
                    ticker = self._exchange.fetch_ticker(ms)
                    mid = float(ticker.get("last") or ticker.get("close") or 0.0) or None
                except Exception as e:
                    logger.warning("Ticker fetch failed for %s: %s", sym, e)
                market = self._markets.get(ms, {})
                limits = market.get("limits", {})
                precision = market.get("precision", {})
                min_qty = (limits.get("amount", {}) or {}).get("min") or 0.0
                step_size = self._precision_to_step(precision.get("amount"))
                min_notional = (limits.get("cost", {}) or {}).get("min") or 0.0
            rows.append(
                {
                    "symbol": sym,
                    "mid": mid,
                    "min_qty": float(min_qty or 0.0),
                    "step_size": float(step_size or 0.0),
                    "min_notional": float(min_notional or 0.0),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _precision_to_step(precision: Any) -> float:
        """ccxt precision -> step size. ccxt may give a step float (0.001) or an
        integer number of decimals (3)."""
        if precision is None:
            return 0.0
        try:
            p = float(precision)
        except (TypeError, ValueError):
            return 0.0
        if p <= 0:
            return 0.0
        # Integer >= 1 means "number of decimals".
        if p >= 1 and float(p).is_integer():
            return 10 ** (-int(p))
        return p

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def _refetch_order(self, order: dict, symbol: str) -> dict | None:
        """Re-read an order from Kraken to confirm its fill (issues #68, #97).

        Called by ``resolve_fill`` when the initial result is non-terminal —
        either too ambiguous to classify (no status and no ``filled`` field) or
        accepted-but-still-settling (``status='open', filled=0``, Kraken's async
        settlement). Read-only; any failure returns None so the caller fails safe
        to not-filled.
        """
        order_id = order.get("id")
        ms = self._market_symbol(symbol)
        if not order_id or ms is None:
            return None
        try:
            return self._exchange.fetch_order(order_id, ms)
        except Exception as exc:  # noqa: BLE001 - confirmation must never crash execution
            logger.warning("Kraken fill confirmation fetch_order failed for %s: %s", symbol, exc)
            return None

    def fetch_order_result(self, order_id: str, symbol: str) -> dict | None:
        """Resolve a previously-placed order against the venue. Read-only.

        This is the second half of the WORKING outcome: an order that was still
        resting on the book when its run's confirmation window closed is recorded
        rather than alarmed, and the NEXT cycle calls this to find out what
        actually happened to it. Kraken keeps a closed order queryable by txid, so
        a fill that landed minutes after the run still reaches the books.

        Returns a dict of ``{status, qty, price, error}`` using the same emitted
        vocabulary as :func:`resolve_fill`, or ``None`` when the venue cannot be
        read at all — the caller must treat None as "still unresolved", never as a
        failure, so a transient API error cannot silently discard a real fill.
        """
        ms = self._market_symbol(symbol)
        if not order_id or ms is None:
            logger.warning("Cannot resolve working order for %s: missing order id or unknown market", symbol)
            return None
        try:
            order = self._exchange.fetch_order(order_id, ms)
        except Exception as exc:  # noqa: BLE001 - resolution must never crash a run
            logger.warning("Kraken fetch_order failed resolving %s (%s): %s", symbol, order_id, exc)
            return None
        if not order:
            return None
        # The venue's own reported size is the reference — the original request is
        # not in scope here, and using it would misread a floored full fill as a
        # partial. No refetch: this IS the refetch.
        requested = order.get("amount") or 0.0
        status, qty, price, reason = resolve_fill(order, requested, refetch=None)
        return {"status": status, "qty": qty, "price": price, "error": reason}

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Place spot orders (MARKET by default, LIMIT when ``price`` given).

        Long-only: SELL quantities are clamped to available balance is NOT done
        here (the rebalancer is responsible for sizing); we reject negative
        quantities defensively.
        """
        # NOTE: `pd.DataFrame(rows, columns=cols)` below SELECTS — a key added
        # to the row dict but not listed here is dropped silently, no error.
        # That is how the #92 fee was lost in fetch_fills. Keep them in step.
        cols = ["symbol", "side", "qty", "price", "order_id", "status", "error"]
        if self.readonly:
            raise PermissionError("readonly broker: order placement disabled")
        if orders is None or orders.empty:
            return pd.DataFrame(columns=cols)

        rows: list[dict[str, Any]] = []
        n_failed = 0
        n_skipped = 0
        # Orders the venue ACCEPTED that were still working when the confirmation
        # window closed. Counted apart from n_failed: a resting limit order is a
        # normal outcome, not an error, and must not colour the run red.
        n_working = 0
        residual_exits: list[str] = []
        for _, o in orders.iterrows():
            sym = str(o["symbol"])
            side = str(o["side"]).lower()
            qty = float(o["qty"])
            price = o.get("price", None)
            result = self._place_one(sym, side, qty, price)
            if result is SKIP_ORDER:
                # Sub-minimum / sub-precision dust: a clean no-op, NOT a failure.
                # Reported as SKIPPED so the pipeline counts it as neither a fill
                # nor a failed order, and it never blocks the rest of the batch.
                #
                # A skipped *SELL* is special: it is a close-out/reduce the book
                # WANTED to make but cannot (untradeable either way). Tradeability
                # is unchanged from the old FAILED path, but silently dropping it
                # could hide a trapped residual, so we surface it as a residual-
                # exposure note (see residual_exits below) — distinct from a quiet
                # all-cash day where only entries are skipped.
                is_exit = side == "sell"
                rows.append(
                    {
                        "symbol": sym,
                        "side": side,
                        "qty": qty,
                        "price": 0.0,
                        "order_id": None,
                        "status": "SKIPPED",
                        "error": (
                            "below exchange minimum (skipped; residual exposure retained)"
                            if is_exit
                            else "below exchange minimum (skipped)"
                        ),
                    }
                )
                n_skipped += 1
                if is_exit:
                    residual_exits.append(f"{sym} (~{qty:g})")
            elif result is not None:
                # Confirm the REAL fill (issue #68): an order Kraken accepted but
                # did not actually fill is reported FAILED/PARTIAL, never an
                # unconditional FILLED. Never assume requested == filled.
                status, filled_qty, fill_price, reason = resolve_fill(
                    result,
                    qty,
                    refetch=lambda r=result, s=sym: self._refetch_order(r, s),
                )
                rows.append(
                    {
                        "symbol": sym,
                        "side": side,
                        "qty": filled_qty,
                        "price": fill_price,
                        "order_id": result.get("id"),
                        "status": status,
                        "error": reason,
                    }
                )
                if status == "FAILED":
                    n_failed += 1
                elif status == STATUS_WORKING:
                    n_working += 1
                    logger.info(
                        "Kraken order still working at the venue: %s %s (id=%s) — %s",
                        side,
                        sym,
                        result.get("id"),
                        reason,
                    )
                elif status == "PARTIAL":
                    logger.warning("Kraken order partially filled: %s %s — %s", side, sym, reason)
            else:
                rows.append(
                    {
                        "symbol": sym,
                        "side": side,
                        "qty": qty,
                        "price": 0.0,
                        "order_id": None,
                        "status": "FAILED",
                        "error": "placement failed",
                    }
                )
                n_failed += 1

        if n_skipped:
            logger.info("Kraken orders skipped as sub-minimum dust (%d/%d)", n_skipped, len(orders))
        if residual_exits:
            # A close-out the book intended but could not place: the position is
            # untradeable (sub-min) yet still on the book. Surface it so a stuck
            # residual stays visible rather than vanishing as a silent SKIP.
            logger.warning(
                "Kraken close-out SELL(s) skipped sub-minimum — residual exposure RETAINED "
                "on %d position(s): %s. Untradeable at current size; monitor for a trapped residual.",
                len(residual_exits),
                ", ".join(residual_exits),
            )
        if n_working:
            logger.info(
                "Kraken orders still working at the venue (%d/%d) — resolved next cycle",
                n_working,
                len(orders),
            )
        if n_failed:
            logger.error("Kraken orders failed (%d/%d)", n_failed, len(orders))
        return pd.DataFrame(rows, columns=cols)

    def _place_one(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Any = None,
    ) -> dict | _SkipOrder | None:
        if side not in ("buy", "sell"):
            logger.error("Invalid side %r for %s (spot is long-only buy/sell)", side, symbol)
            return None
        if quantity <= 0:
            logger.warning("Non-positive quantity %s for %s, skipping", quantity, symbol)
            return None

        ms = self._market_symbol(symbol)
        if ms is None:
            logger.error("Unknown Kraken market for %s/%s", symbol, self.quote_asset)
            return None

        market = self._markets.get(ms, {})
        precision = (market.get("precision", {}) or {}).get("amount", 8)
        if isinstance(precision, float) and 0 < precision < 1:
            precision = max(0, -int(math.floor(math.log10(precision))))
        # Use ccxt's amount_to_precision (floors to the market's precision) so a
        # SELL can never round UP past the available balance. round() could push
        # a sell qty above what we hold; flooring keeps it safe. Fall back to a
        # floor-via-round only if amount_to_precision is unavailable.
        amount_to_precision = getattr(self._exchange, "amount_to_precision", None)
        if callable(amount_to_precision):
            try:
                quantity = float(amount_to_precision(ms, quantity))
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("amount_to_precision failed for %s (%s); flooring", symbol, e)
                factor = 10 ** int(precision)
                quantity = math.floor(quantity * factor) / factor
        else:
            factor = 10 ** int(precision)
            quantity = math.floor(quantity * factor) / factor
        if quantity <= 0:
            logger.warning("Quantity rounds to zero for %s (precision=%s)", symbol, precision)
            return SKIP_ORDER

        min_qty = (market.get("limits", {}).get("amount", {}) or {}).get("min") or 0
        if min_qty and quantity < min_qty:
            logger.warning("Quantity %s below Kraken minimum %s for %s", quantity, min_qty, symbol)
            return SKIP_ORDER

        # A NaN price must NOT route to a limit order (pd.notna(NaN) is False).
        has_price = pd.notna(price) and price > 0
        order_type = "limit" if has_price else "market"

        for attempt in range(MAX_RETRIES):
            try:
                order = self._exchange.create_order(
                    symbol=ms,
                    type=order_type,
                    side=side,
                    amount=quantity,
                    price=float(price) if has_price else None,
                )
                logger.info("Kraken order placed: %s %s %s @ %s", side, quantity, symbol, order_type)
                return order
            except Exception as e:
                # ccxt error classes exist only when ccxt is importable.
                if ccxt is not None and isinstance(e, (ccxt.InsufficientFunds, ccxt.InvalidOrder)):
                    logger.error("Order rejected for %s: %s", symbol, e)
                    return None
                if attempt < MAX_RETRIES - 1:
                    logger.warning("Order failed, retrying (%d/%d): %s", attempt + 1, MAX_RETRIES, e)
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    logger.error("Order failed after %d attempts: %s", MAX_RETRIES, e)
                    return None
        return None

    # ------------------------------------------------------------------
    # Cashflows (ledger)
    # ------------------------------------------------------------------

    def fetch_ledger_entries(
        self,
        since: str | None = None,
        max_pages: int = LEDGER_MAX_PAGES,
    ) -> list[dict[str, Any]]:
        """Raw ccxt ledger entries for this account, fully paginated.

        Kraken's ``Ledgers`` endpoint returns at most 50 rows per call and
        paginates with an ``ofs`` offset; **ccxt does not paginate for you**
        (``kraken.fetch_ledger`` issues exactly one ``privatePostLedgers`` and
        applies ``since``/``limit`` client-side after parsing). So we drive
        ``ofs`` ourselves and stop on a short/empty page.

        FAILS LOUD. Unlike :meth:`_fetch_balances` — where an empty dict is a
        recoverable "no positions" reading — a truncated ledger silently *omits
        deposits*, which would understate the cost basis of every downstream
        performance figure. Any upstream error (after :func:`with_retry` has
        exhausted its transient budget) propagates, and a run that hits
        ``max_pages`` raises rather than returning a partial history.
        """
        since_ts: int | None = None
        if since is not None:
            try:
                since_ts = int(pd.Timestamp(since).timestamp() * 1000)
            except Exception as exc:
                raise ValueError(f"Unparseable 'since' for Kraken ledger: {since!r}") from exc

        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for page in range(max_pages):
            ofs = page * LEDGER_PAGE_SIZE
            batch = with_retry(
                lambda o=ofs: self._exchange.fetch_ledger(code=None, since=since_ts, params={"ofs": o}),
                label="kraken.fetch_ledger",
            )
            batch = list(batch or [])
            new = [e for e in batch if str(e.get("id")) not in seen]
            for e in new:
                seen.add(str(e.get("id")))
            entries.extend(new)
            # A short page means we reached the end. An empty page, or a page that
            # is entirely duplicates (Kraken can re-serve rows when the ledger
            # grows under us), also terminates — otherwise we would loop forever.
            if len(batch) < LEDGER_PAGE_SIZE or not new:
                break
        else:
            raise RuntimeError(
                f"Kraken ledger pagination hit max_pages={max_pages} "
                f"({len(entries)} entries) without reaching the end. Refusing to "
                "return a truncated ledger — raise max_pages or narrow 'since'."
            )
        return entries

    def fetch_cashflows(self, since: str | None = None) -> pd.DataFrame:
        """External cash movements (deposits / withdrawals / transfers) from the
        Kraken ledger, normalised for performance accounting.

        Columns: ``date`` (UTC ``YYYY-MM-DD``), ``timestamp`` (ISO-8601),
        ``type`` (raw Kraken type: ``deposit`` / ``withdrawal`` / ``transfer``),
        ``currency`` (canonical asset code), ``amount`` (signed, in *that asset's*
        units — positive in, negative out), ``fee``, ``amount_net``
        (``amount - fee``, i.e. the actual balance delta), ``refid``, ``id``.

        Two deliberate non-conversions, because guessing either would fabricate
        the baseline the whole report is measured against:

        * **Amounts stay in their native asset.** A BTC deposit is not a USD
          cashflow; valuing it needs a historical price the broker has no
          business inventing. Callers must reject or explicitly price non-quote
          rows (see ``quantbox-live/scripts/sync_kraken_flows.py``).
        * **``transfer`` rows are returned, not classified.** Kraken uses
          ``transfer`` for both genuinely external movements and internal ones
          (spot↔futures wallet, staking migrations). Only the account's owner
          knows which; the caller decides.

        ``amount`` is taken from the RAW Kraken payload (``info.amount``), which
        is signed. ccxt's parsed ``amount`` is unsigned with the sign moved into
        ``direction`` — using it directly would turn every withdrawal into a
        deposit.
        """
        cols = ["date", "timestamp", "type", "currency", "amount", "fee", "amount_net", "refid", "id"]
        entries = self.fetch_ledger_entries(since=since)

        rows = []
        for e in entries:
            info = e.get("info", {}) or {}
            raw_type = str(info.get("type") or "").lower()
            if raw_type not in CASHFLOW_LEDGER_TYPES:
                continue
            ts = e.get("timestamp")
            if ts is None:
                raise ValueError(f"Kraken ledger entry {e.get('id')!r} has no timestamp: {e!r}")
            when = pd.Timestamp(int(ts), unit="ms", tz="UTC")
            amount = float(info.get("amount") or 0.0)
            fee = float(info.get("fee") or 0.0)
            rows.append(
                {
                    "date": when.strftime("%Y-%m-%d"),
                    "timestamp": when.isoformat(),
                    "type": raw_type,
                    "currency": normalize_kraken_asset(str(info.get("asset") or e.get("currency") or "")),
                    "amount": amount,
                    "fee": fee,
                    # Kraken books the fee separately from the amount, so the real
                    # balance delta is amount - fee for both directions (deposit
                    # credits amount then debits fee; withdrawal debits both).
                    "amount_net": amount - fee,
                    "refid": info.get("refid") or e.get("referenceId"),
                    "id": e.get("id"),
                }
            )

        if not rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows, columns=cols).sort_values("timestamp").reset_index(drop=True)
        return df

    def fetch_fills(self, since: str) -> pd.DataFrame:
        """Trade history since ``since`` (ISO timestamp) via Kraken TradesHistory."""
        # NOTE: `pd.DataFrame(rows, columns=cols)` SELECTS — any key not listed
        # here is dropped silently, no error. Adding a field to the row dict
        # below without adding it here makes that field a no-op (#92).
        cols = ["symbol", "side", "qty", "price", "timestamp", "fee", "fee_currency"]
        try:
            since_ts = int(pd.Timestamp(since).timestamp() * 1000)
        except Exception:
            since_ts = None
        try:
            trades = self._exchange.fetch_my_trades(symbol=None, since=since_ts)
        except Exception as e:
            logger.error("Error fetching Kraken fills: %s", e)
            return pd.DataFrame(columns=cols)

        rows = []
        for t in trades or []:
            sym = t.get("symbol", "")
            base = normalize_kraken_asset(sym.split("/")[0]) if "/" in sym else sym
            rows.append(
                {
                    "symbol": base,
                    "side": t.get("side", ""),
                    "qty": float(t.get("amount", 0) or 0),
                    "price": float(t.get("price", 0) or 0),
                    "timestamp": t.get("datetime", ""),
                    # #92: keep the venue-reported fee. None = UNKNOWN.
                    "fee": trade_fee(t),
                    "fee_currency": trade_fee_currency(t),
                }
            )
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
