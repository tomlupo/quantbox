"""Financing of a backtest's cash: borrowed cash pays rf + spread, idle cash earns rf - spread (TOM-1429).

A spot book whose weights sum to more than 1 holds more than its equity: the
difference is borrowed. One that sums to less than 1 holds idle cash. Neither
is free, and a backtest that treats them as free (or, worse, cannot borrow and
cuts the last buys) is measuring a different book. The ``venue`` block says
what the venue charges::

    venue:
      allow_shorts: true
      financing:
        rate: "LT12TRUU Index"   # a ticker in the loaded prices: a cash total-return index,
                                 #   its bar-on-bar return is the rate; or a number: a constant
                                 #   annual rate (0.0 = free borrowing, idle cash earns nothing)
        borrow_spread_bps: 0     # borrowed cash costs rate + this
        lend_spread_bps: 0       # idle cash earns rate - this

It is a VENUE fact for the same reason ``allow_shorts`` is: it answers "could
this book have existed, and at what cost?" — a leveraged book exists only
where something lends, and the price of that lending is the venue's (or the
broker's), not the strategy's and not the execution timing's.

**Mechanism — engine-agnostic.** :func:`add_cash_legs` appends two synthetic
assets to the engine's book: ``LEND`` (held at ``max(1 - sum(w), 0)``, priced
by compounding ``rate - lend_spread``) and ``BORROW`` (held at
``min(1 - sum(w), 0)``, i.e. short, priced by compounding
``rate + borrow_spread``). The book the engine receives then sums to exactly 1,
so a cash-constrained engine (vectorbt cannot take cash below zero) never cuts a
buy, and the financing P&L is just the legs' mark-to-market. Splitting the
residual by sign makes the asymmetric spread exact: a leg's sign never changes.
Both legs trade without fees or slippage. The rate and spreads accrue ACT/365
on the calendar time between bars; a ticker rate is its own bar return.

Whether a book may hold net exposure above 1 at all is ``venue.leverage``
(:mod:`quantbox.execution_schedule`): ``normalize`` (the vectorbt default)
scales such a decision to net 1; ``borrow`` holds it, financed by this block —
or, without one, at an ASSUMED rate of 0 (:data:`ASSUMED_FREE`), recorded as
``assumed: true`` in the manifest and warned about. rsims is a margin
(notional) simulator with no cash floor, so its default is ``borrow``: its idle
cash earns nothing and its borrowing is free unless ``financing`` says otherwise.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: The synthetic cash legs' column names. Never a real ticker.
LEND = "__quantbox_cash_lend__"
BORROW = "__quantbox_cash_borrow__"
CASH_LEGS = (LEND, BORROW)

#: Net exposure above 1 by more than this needs borrowing (float noise below it does not).
NET_EXPOSURE_TOLERANCE = 1e-6

#: ``venue.leverage`` values; the default depends on the engine (:func:`resolve_leverage`).
LEVERAGE_MODES = ("normalize", "borrow")
LEVERAGE_SCHEMA: dict[str, Any] = {
    "enum": list(LEVERAGE_MODES),
    "description": (
        "What a decision with net exposure above 1 becomes (docs/adr/0007). normalize: the whole basket is "
        "scaled proportionally to net 1 on that rebalance (counted). borrow: held as decided; the excess is "
        "borrowed at venue.financing, or at an ASSUMED rate of 0 without that block (recorded and warned). "
        "Default: normalize on vectorbt (it cannot borrow), borrow on rsims (a margin simulator)."
    ),
}

FINANCING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rate"],
    "properties": {
        "rate": {
            "type": ["string", "number"],
            "description": (
                "The risk-free financing rate. A string: a ticker in the loaded prices, read as a cash "
                "total-return index whose bar-on-bar return IS the rate (e.g. 'LT12TRUU Index'). A number: "
                "a constant annual rate, accrued ACT/365 (0.0 = borrowing is free and idle cash earns nothing)."
            ),
        },
        "borrow_spread_bps": {
            "type": "number",
            "minimum": 0,
            "default": 0,
            "description": "Borrowed cash (weights summing above 1) costs rate + this, in bps per year.",
        },
        "lend_spread_bps": {
            "type": "number",
            "minimum": 0,
            "default": 0,
            "description": "Idle cash (weights summing below 1) earns rate - this, in bps per year.",
        },
    },
    "description": (
        "What the venue charges for cash (docs/adr/0007): borrowed cash pays rate + borrow_spread, idle cash "
        "earns rate - lend_spread. Without it, venue.leverage: borrow assumes a rate of 0 and records that."
    ),
}

#: How the block is spelled in an error message.
FINANCING_HINT = (
    "venue.financing: {rate: <ticker in the prices, or an annual number>, borrow_spread_bps, lend_spread_bps}"
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


@dataclass(frozen=True)
class Financing:
    """A resolved ``venue.financing`` block: exactly one of ``rate_ticker`` / ``rate_annual`` is set."""

    rate_ticker: str | None
    rate_annual: float | None
    borrow_spread_bps: float = 0.0
    lend_spread_bps: float = 0.0
    #: True when no block was declared and ``venue.leverage: borrow`` runs at an assumed rate of 0.
    assumed: bool = False

    def record(self) -> dict[str, Any]:
        """The block written to ``venue.financing`` in run@1 and explain@1."""
        rate: dict[str, Any] = {"ticker": self.rate_ticker} if self.rate_ticker is not None else {}
        if self.rate_annual is not None:
            rate["annual"] = self.rate_annual
        record = {
            "rate": rate,
            "borrow_spread_bps": self.borrow_spread_bps,
            "lend_spread_bps": self.lend_spread_bps,
            "day_count": "ACT/365",
        }
        if self.assumed:
            record["assumed"] = True
        return record


#: ``venue.leverage: borrow`` without a ``venue.financing`` block: free borrowing, idle cash earns nothing.
ASSUMED_FREE = Financing(rate_ticker=None, rate_annual=0.0, assumed=True)


def resolve_leverage(value: Any, engine: str) -> str:
    """``venue.leverage`` -> ``normalize`` | ``borrow``; absent: normalize on vectorbt, borrow on rsims."""
    if value is None:
        return "borrow" if engine == "rsims" else "normalize"
    if value not in LEVERAGE_MODES:
        raise ValueError(f"venue.leverage must be one of {list(LEVERAGE_MODES)}, got {value!r}")
    return str(value)


def resolve_financing(block: Any) -> Financing | None:
    """Validate a ``venue.financing`` block; None when absent. Raises ``ValueError`` on any other shape."""
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"venue.financing must be a mapping like {{rate: ..., borrow_spread_bps: 0}}, got {block!r}")
    unknown = sorted(set(block) - {"rate", "borrow_spread_bps", "lend_spread_bps"})
    if unknown:
        raise ValueError(
            f"venue.financing: unknown key(s) {unknown}; the keys are 'rate', 'borrow_spread_bps', 'lend_spread_bps'"
        )
    if "rate" not in block:
        raise ValueError(f"venue.financing: 'rate' is required — {FINANCING_HINT}")
    rate = block["rate"]
    if isinstance(rate, str):
        if not rate.strip():
            raise ValueError("venue.financing.rate: a ticker must be a non-empty string")
        ticker, annual = rate, None
    elif _is_number(rate):
        ticker, annual = None, float(rate)
    else:
        raise ValueError(f"venue.financing.rate must be a ticker (string) or an annual rate (number), got {rate!r}")
    spreads = {}
    for key in ("borrow_spread_bps", "lend_spread_bps"):
        value = block.get(key, 0)
        if not _is_number(value) or value < 0:
            raise ValueError(f"venue.financing.{key} must be a number >= 0, got {value!r}")
        spreads[key] = float(value)
    return Financing(ticker, annual, **spreads)


def _year_fractions(index: pd.Index) -> pd.Series:
    """ACT/365 calendar time from the previous bar to each bar; 0 on the first."""
    idx = pd.DatetimeIndex(index)
    days = pd.Series(idx, index=index).diff().dt.total_seconds().fillna(0.0) / 86400.0
    return days / 365.0


def rate_per_bar(index: pd.Index, financing: Financing, prices: pd.DataFrame | None = None) -> pd.Series:
    """The financing rate earned over each bar of *index* (0 on the first bar).

    A ticker rate is the bar-on-bar return of that column of *prices* (the
    loaded panel, forward-filled across its holidays). It must have printed by
    the first bar of *index*: before its first price the rate is unknown, and
    an unknown rate is refused, never assumed to be 0. Bars after its LAST
    price are counted (:func:`rate_stale_bars`): the forward-filled level makes
    the rate 0 there.
    """
    if financing.rate_annual is not None:
        return financing.rate_annual * _year_fractions(index)
    ticker = financing.rate_ticker
    if prices is None or ticker not in prices.columns:
        raise ValueError(
            f"venue.financing.rate: ticker {ticker!r} is not in the loaded prices — add it to the universe "
            "(it need not carry a weight), or give a constant annual rate"
        )
    raw = prices[ticker]
    first = raw.first_valid_index()
    if first is None or first > index[0]:
        raise ValueError(
            f"venue.financing.rate: ticker {ticker!r} has no price on or before the first backtest bar "
            f"{index[0]} (first price: {first}); the financing rate there is unknown"
        )
    level = raw.ffill().reindex(raw.index.union(index)).ffill().reindex(index)
    return level.pct_change(fill_method=None).fillna(0.0)


def rate_stale_bars(index: pd.Index, financing: Financing, prices: pd.DataFrame | None = None) -> int:
    """Bars of *index* after a ticker rate's last price, where it reads as 0; 0 for a constant rate."""
    if financing.rate_ticker is None or prices is None or financing.rate_ticker not in prices.columns:
        return 0
    last = prices[financing.rate_ticker].last_valid_index()
    return int((pd.DatetimeIndex(index) > last).sum()) if last is not None else len(index)


def add_cash_legs(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    financing: Financing,
    rate_source: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Append the LEND and BORROW legs to an engine-ready book; returns ``(prices, weights, record)``.

    *weights* must already be the book the engine trades (lagged, calendar
    applied, NaN policy materialised), so ``1 - sum(w)`` per row is exactly the
    cash the engine would hold after a rebalance. *rate_source* is the loaded
    prices panel a ticker rate is read from.
    """
    if weights.isna().any().any():
        raise ValueError("add_cash_legs: weights must be materialised (no NaN) before the cash legs are sized")
    clash = [c for c in CASH_LEGS if c in prices.columns]
    if clash:
        raise ValueError(f"add_cash_legs: {clash} is a reserved synthetic column name")
    rate = rate_per_bar(prices.index, financing, rate_source)
    years = _year_fractions(prices.index)
    lend_r = rate - financing.lend_spread_bps / 1e4 * years
    borrow_r = rate + financing.borrow_spread_bps / 1e4 * years
    cash = 1.0 - weights.sum(axis=1)

    out_prices = prices.copy()
    out_prices[LEND] = (1.0 + lend_r).cumprod()
    out_prices[BORROW] = (1.0 + borrow_r).cumprod()
    out_weights = weights.copy()
    out_weights[LEND] = cash.clip(lower=0.0)
    out_weights[BORROW] = cash.clip(upper=0.0)

    stale = rate_stale_bars(prices.index, financing, rate_source)
    if stale:
        logger.warning(
            "FINANCING: rate ticker %r stops printing %d bar(s) before the backtest ends; the rate is 0 there "
            "(counted as rate_stale_bars_at_end)",
            financing.rate_ticker,
            stale,
        )
    record = {
        **financing.record(),
        "rate_stale_bars_at_end": stale,
        "mean_cash_weight": float(cash.mean()) if len(cash) else 0.0,
        "min_cash_weight": float(cash.min()) if len(cash) else 0.0,
        "max_cash_weight": float(cash.max()) if len(cash) else 0.0,
        "borrow_bar_share": float((cash < -NET_EXPOSURE_TOLERANCE).mean()) if len(cash) else 0.0,
        "rate_annualised_mean": float(rate.sum() / years.sum()) if years.sum() > 0 else 0.0,
    }
    return out_prices, out_weights, record
