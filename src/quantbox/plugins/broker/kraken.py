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

from ..datasources.kraken_data import KRAKEN_BALANCE_SUFFIXES, normalize_kraken_asset

try:
    import ccxt
except ImportError:  # pragma: no cover
    ccxt = None  # type: ignore

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

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
            self._markets = self._exchange.load_markets() or {}
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

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Place spot orders (MARKET by default, LIMIT when ``price`` given).

        Long-only: SELL quantities are clamped to available balance is NOT done
        here (the rebalancer is responsible for sizing); we reject negative
        quantities defensively.
        """
        cols = ["symbol", "side", "qty", "price", "order_id", "status", "error"]
        if self.readonly:
            raise PermissionError("readonly broker: order placement disabled")
        if orders is None or orders.empty:
            return pd.DataFrame(columns=cols)

        rows: list[dict[str, Any]] = []
        n_failed = 0
        n_skipped = 0
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
                fill_price = float(result.get("average", result.get("price", 0)) or 0)
                filled_qty = float(result.get("filled", qty) or qty)
                rows.append(
                    {
                        "symbol": sym,
                        "side": side,
                        "qty": filled_qty,
                        "price": fill_price,
                        "order_id": result.get("id"),
                        "status": "FILLED",
                        "error": "",
                    }
                )
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

    def fetch_fills(self, since: str) -> pd.DataFrame:
        """Trade history since ``since`` (ISO timestamp) via Kraken TradesHistory."""
        cols = ["symbol", "side", "qty", "price", "timestamp"]
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
                }
            )
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
