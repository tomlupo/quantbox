"""Pre-trade portfolio valuation — one implementation for every sizing site.

Why this module exists
----------------------
Four places in this codebase computed a portfolio value by hand as
``cash + sum(qty * price)``, and all four **skipped** any holding they could
not price::

    p = get_price(asset)
    if p is not None and qty > 0:
        total_value += qty * p

A book whose prices were unavailable therefore produced *exactly the same
number* as a book holding nothing at all: the cash balance. "Could not price"
and "nothing held" were indistinguishable, and every target was then sized off
that number (``target_value = weight * portfolio_value``). The live
``crypto-trend-kraken`` book sized off CASH instead of EQUITY since inception,
and because buying moves equity out of cash, the targets shrank as positions
accumulated — it chased its own tail.

Which rule applies is a property of the VENUE, not of the config
-----------------------------------------------------------------
The live incident was not "one call site marked its book wrong". It was that
the valuation rule came from **which rebalancer plugin someone named in a YAML
file**. ``configs/crypto_trend_kraken.yaml`` declared
``rebalancing.futures.v1`` while the broker was ``kraken.spot.v1``, and
``FuturesRebalancer`` does exactly what its own comment says::

    # For futures, portfolio value = margin balance (cash), not cash + positions
    total_value = max(0, cash_available)

That is correct for a perps book and wrong for a spot one, and the code had no
way to tell which it was holding. Swapping the config to a spot rebalancer
would have removed the symptom and left the trap armed for the next book.

So the rule is **derived from the broker**, which is the only object that knows
what venue it is talking to. A broker declares one class attribute:

``valuation_basis = BASIS_MARK`` (``"mark_to_market"``)
    A spot/cash book. Portfolio value is ``cash + sum(qty * price)``, so every
    held position must be markable.
``valuation_basis = BASIS_MARGIN`` (``"margin_balance"``)
    A derivatives book. Portfolio value is the margin balance (plus unrealised
    PnL), which is what ``get_equity()`` / ``get_cash()`` report; positions are
    leveraged and do not add to equity, so an unmarkable name does not
    understate it.

A broker that declares **neither** cannot be valued safely, so in live mode it
is a refusal rather than a guess — the same rule already applied to an
unmarkable position. Outside live, the call site's ``fallback_basis`` (what it
did before venue declarations existed) keeps paper and backtest runs working,
loudly.

The rule this module enforces
-----------------------------
A pre-trade valuation is either **complete** or it is a **refusal**. In live
mode a single unmarkable holding on a mark-to-market venue fails the run: it
does not warn, does not skip the name, and does not fall back to cash.
Under-valuing a book silently under-sizes every target, which is the defect
being fixed — so a fix that can still degrade into it has not fixed anything.

The three outcomes are distinguishable in the exit path, the metrics and the
logs:

``empty_book``
    Nothing held. A legitimate state; the value is the cash balance.
``unpriced_holdings``
    At least one held position could not be marked. Refusal.
``reconciliation_mismatch``
    The pipeline's valuation and the broker's own view disagree by more than
    the configured tolerance. Refusal.
``venue_valuation_basis_unknown``
    The broker does not say whether its book is marked or margined. Refusal on
    live.

This mirrors the refusal that ``quantbox-live``'s ``portfolio_snapshot.py``
already applies at the same venue ("equity would be understated. Refusing to
write."), rather than inventing a second convention for the same problem.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace

from quantbox.exceptions import QuantboxError

logger = logging.getLogger(__name__)

__all__ = [
    "BASIS_MARGIN",
    "BASIS_MARK",
    "DEFAULT_RECONCILIATION_TOLERANCE",
    "UNGATED_MODES",
    "PortfolioValuation",
    "PortfolioValuationError",
    "is_gated",
    "resolve_portfolio_value",
    "value_holdings",
    "venue_valuation_basis",
]

#: A spot/cash venue: portfolio value is ``cash + sum(qty * price)``, so every
#: held position must be markable.
BASIS_MARK = "mark_to_market"

#: A derivatives venue: portfolio value is the margin balance (plus unrealised
#: PnL). Positions are leveraged and do not add to equity.
BASIS_MARGIN = "margin_balance"

#: The name of the class attribute a broker declares. Kept as a constant so the
#: brokers, this module and the tests all spell it once.
BASIS_ATTR = "valuation_basis"

VALID_BASES = frozenset({BASIS_MARK, BASIS_MARGIN})

#: Relative tolerance for pre-trade reconciliation between the pipeline's own
#: valuation and the broker's ``get_equity()``. 0.5% absorbs the bid/ask and
#: timing gap between two price reads of the same book without absorbing a
#: missing position — the defect this gate exists to catch understates equity
#: by whole percent, not by basis points.
DEFAULT_RECONCILIATION_TOLERANCE = 0.005

#: Reasons a valuation refuses. Kept as literals so callers (and the run's
#: metrics) can branch on them without re-parsing a message.
REASON_UNPRICED = "unpriced_holdings"
REASON_MISMATCH = "reconciliation_mismatch"
REASON_BROKER_EQUITY_FAILED = "broker_equity_failed"
REASON_UNKNOWN_BASIS = "venue_valuation_basis_unknown"

#: The modes in which an incomplete valuation is NOT fatal. A simulation
#: computes and REPORTS the same thing but does not halt: a paper book sizing off
#: cash costs nothing, and gating it would make the simulation useless when a
#: venue ticker is briefly unavailable.
#:
#: The gate is expressed as an ALLOW-list of simulations rather than a deny-list
#: containing ``"live"``, so that anything the caller does not positively declare
#: a simulation is gated. A deny-list makes `mode="Live"`, `mode="LIVE"`,
#: `mode=None` and a caller that forgot to thread `mode` at all into silently
#: ungated live runs — which is the same shape as the defect this module exists
#: to close: a declaration read by the wrong consumer, failing open.
UNGATED_MODES = frozenset({"paper", "backtest", "dry_run", "dry-run"})


def is_gated(mode: str | None) -> bool:
    """Does this run mode halt on a valuation it cannot trust?

    Everything that is not positively declared a simulation does — including an
    empty or unrecognised mode. See :data:`UNGATED_MODES`.
    """
    if mode is None:
        return True
    return str(mode).strip().lower() not in UNGATED_MODES


class PortfolioValuationError(QuantboxError):
    """The pre-trade valuation cannot be trusted, so the run must not size off it.

    Carries the machine-readable ``reason`` and the offending symbols so the
    caller can put them in metrics and the operator can see WHICH name failed
    rather than only that something did.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        unpriced: Iterable[str] = (),
        computed: float | None = None,
        broker_equity: float | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.unpriced = tuple(unpriced)
        self.computed = computed
        self.broker_equity = broker_equity


@dataclass(frozen=True)
class PortfolioValuation:
    """The result of marking a book to market, including what could NOT be marked."""

    value: float
    cash: float
    marked_value: float
    priced: tuple[str, ...] = ()
    unpriced: tuple[str, ...] = ()
    n_holdings: int = 0
    source: str = "computed"
    broker_equity: float | None = None
    reconciled: bool = False
    #: Which venue rule produced :attr:`value` — see :data:`BASIS_MARK` /
    #: :data:`BASIS_MARGIN`. ``""`` when only :func:`value_holdings` ran, which
    #: marks a book without deciding what the mark MEANS for this venue.
    basis: str = ""

    @property
    def is_empty_book(self) -> bool:
        """True when nothing is held — NOT the same as 'nothing could be priced'."""
        return self.n_holdings == 0

    @property
    def is_complete(self) -> bool:
        """True when every held position was marked."""
        return not self.unpriced

    @property
    def state(self) -> str:
        """``empty_book`` | ``complete`` | ``unpriced_holdings``.

        The three used to be one number. They are now one word.
        """
        if self.is_empty_book:
            return "empty_book"
        if self.is_complete:
            return "complete"
        return REASON_UNPRICED

    def as_metrics(self) -> dict[str, float | None]:
        """Run metrics keeping the three outcomes distinguishable after the fact.

        ``RunResult.metrics`` is numeric by contract, so the state is carried as
        counts rather than as a label: an empty book is ``n_holdings == 0``, a
        complete one is ``n_unpriced == 0`` with holdings, and a refusal is
        ``n_unpriced > 0``. The label itself goes to :meth:`as_notes`.
        """
        return {
            "portfolio_value_cash_usd": float(self.cash),
            "portfolio_value_marked_usd": float(self.marked_value),
            "portfolio_value_n_holdings": float(self.n_holdings),
            "portfolio_value_n_unpriced": float(len(self.unpriced)),
            "portfolio_valuation_complete": 1.0 if self.is_complete else 0.0,
            "portfolio_value_broker_equity_usd": (
                float(self.broker_equity) if self.broker_equity is not None else None
            ),
            "portfolio_value_reconciled": 1.0 if self.reconciled else 0.0,
        }

    def as_notes(self) -> dict[str, object]:
        """The human-readable half, for ``RunResult.notes``.

        ``state`` always describes the MARK, on every venue. On a margined book
        ``unpriced_holdings`` therefore means "these names lost their target",
        not "equity is wrong" — which is why ``basis`` is recorded next to it
        rather than folded into it.
        """
        return {
            "portfolio_valuation_state": self.state,
            "portfolio_valuation_basis": self.basis,
            "portfolio_valuation_source": self.source,
            "portfolio_valuation_unpriced": list(self.unpriced),
            "portfolio_valuation_reconciled": self.reconciled,
        }


def value_holdings(
    *,
    cash: float,
    holdings: Mapping[str, float],
    get_price: Callable[[str], float | None],
    stable_coin: str | None = None,
    exclusions: Iterable[str] = (),
) -> PortfolioValuation:
    """Mark a book to market, COLLECTING what could not be marked.

    Unlike the four hand-rolled loops this replaces, a holding with no usable
    price is recorded in :attr:`PortfolioValuation.unpriced` instead of being
    added as zero. The caller decides what that means; this function never
    decides on its behalf.

    ``stable_coin`` is counted at par (it is the quote currency sitting in the
    positions table rather than the cash table). Symbols in ``exclusions`` are
    not part of the tradable book and are ignored entirely — they are neither
    marked nor reported as unpriced.
    """
    excluded = set(exclusions)
    cash_value = max(0.0, float(cash))

    marked = 0.0
    priced: list[str] = []
    unpriced: list[str] = []
    counted = 0

    for asset, raw_qty in holdings.items():
        try:
            qty = float(raw_qty)
        except (TypeError, ValueError):
            qty = float("nan")
        if not math.isfinite(qty):
            # A quantity we cannot read is not a quantity of zero. Marking it
            # would poison the total with NaN; skipping it would understate the
            # book — the exact move this module exists to stop.
            counted += 1
            unpriced.append(asset)
            continue
        if qty == 0:
            continue
        if stable_coin is not None and asset == stable_coin:
            # Quote currency held as a "position" is cash at par, not a mark.
            marked += max(0.0, qty)
            priced.append(asset)
            counted += 1
            continue
        if asset in excluded:
            continue

        counted += 1
        price = get_price(asset)
        if price is None or not _is_finite_positive(price):
            unpriced.append(asset)
            continue
        marked += qty * float(price)
        priced.append(asset)

    return PortfolioValuation(
        value=cash_value + marked,
        cash=cash_value,
        marked_value=marked,
        priced=tuple(sorted(priced)),
        unpriced=tuple(sorted(unpriced)),
        n_holdings=counted,
        source="computed",
    )


def venue_valuation_basis(broker: object | None) -> str | None:
    """Ask the BROKER how its own book is valued. ``None`` means it does not say.

    The declaration is a plain class attribute (:data:`BASIS_ATTR`) rather than
    a key in ``describe()``: ``describe()`` is a free-form introspection surface
    whose shape already varies across the brokers in this repo (Hyperliquid
    reports ``type: "decentralized"``, ``BinanceFuturesBroker`` has no ``type``
    at all, ``BinanceLiveBroker.describe()`` returns a whole account snapshot
    and hits the network). A valuation rule cannot key off a field like that,
    and it must not cost an API call to read.

    An unrecognised value is ``None``, not a guess: a declaration nobody can
    parse carries no more information than no declaration, and the caller's
    live-mode refusal is the right outcome for both.
    """
    if broker is None:
        return None
    declared = getattr(broker, BASIS_ATTR, None)
    if declared is None:
        return None
    if isinstance(declared, str) and declared in VALID_BASES:
        return declared
    logger.error(
        "Broker %s declares %s=%r, which is not one of %s — treating the venue as UNDECLARED.",
        type(broker).__name__,
        BASIS_ATTR,
        declared,
        sorted(VALID_BASES),
    )
    return None


def resolve_portfolio_value(
    *,
    broker: object | None,
    mode: str,
    cash: float,
    holdings: Mapping[str, float],
    get_price: Callable[[str], float | None],
    fallback_basis: str,
    stable_coin: str | None = None,
    exclusions: Iterable[str] = (),
    tolerance: float = DEFAULT_RECONCILIATION_TOLERANCE,
    require_reconciliation: bool = True,
) -> PortfolioValuation:
    """Resolve the pre-trade portfolio value, gating the run in live mode.

    The sequence, and why it is this order:

    1. Ask the broker which valuation basis its venue uses
       (:func:`venue_valuation_basis`). In a gated mode a venue that does not
       say is a refusal — guessing is how a spot book came to be valued by a
       futures rule.
    2. Mark the book with the caller's own price map, collecting what could not
       be marked. This runs on every venue: on a margined one the marks do not
       feed equity, but an unmarkable name still loses its target, and that
       belongs in the record.
    3. Ask the broker for its own view (``get_equity``), when it has one **and
       it declared its venue**. A broker that cannot value its own book RAISES;
       that is a refusal, not a reason to fall back to the pipeline's number —
       falling back is how the original defect stayed invisible.

       An UNDECLARED broker is not asked, even though it may answer. Its
       ``get_equity()`` is a number whose meaning is exactly the thing that was
       not declared: margin balance or marked value, and those differ by the
       whole value of the positions held. Reading it would also make
       ``fallback_basis`` unobservable — both bases would end at the broker's
       number — so the argument the call site passed to describe its historic
       behaviour would silently stop applying.
    4. Apply the venue's rule:

       * :data:`BASIS_MARK` — value is the mark. In a gated mode an incomplete
         mark refuses, and the broker's equity (where it has one) must
         reconcile with the mark to within ``tolerance``.
       * :data:`BASIS_MARGIN` — value is the broker's equity where it has one,
         otherwise ``max(0, cash)``, which is the margin balance the caller
         read. The mark is *not* a meaningful cross-check here (equity is margin
         plus unrealised PnL, not notional), so reconciling them would be a
         guaranteed false alarm, and an unmarkable name does not understate
         equity.

    Args:
        fallback_basis: the basis to use when the broker declares none AND the
            mode is not gated — i.e. what this call site did before venue
            declarations existed. It keeps paper and backtest runs working
            unchanged against third-party brokers; it never applies on live.
    """
    gated = is_gated(mode)
    if fallback_basis not in VALID_BASES:
        raise ValueError(f"fallback_basis must be one of {sorted(VALID_BASES)}, got {fallback_basis!r}")

    basis = venue_valuation_basis(broker)
    declared = basis is not None
    if basis is None:
        if gated:
            raise PortfolioValuationError(
                f"Broker {type(broker).__name__} does not declare a valuation basis "
                f"({BASIS_ATTR} is missing or unrecognised), so whether this book is worth "
                "its margin balance or its cash plus holdings is UNKNOWN. A live run will not "
                "guess: the two answers differ by the whole value of the positions held.",
                reason=REASON_UNKNOWN_BASIS,
            )
        logger.warning(
            "Broker %s does not declare %s; falling back to %r for this %s run. A live run would refuse here.",
            type(broker).__name__,
            BASIS_ATTR,
            fallback_basis,
            mode,
        )
        basis = fallback_basis

    valuation = value_holdings(
        cash=cash,
        holdings=holdings,
        get_price=get_price,
        stable_coin=stable_coin,
        exclusions=exclusions,
    )

    broker_equity: float | None = None
    if declared and broker is not None and hasattr(broker, "get_equity"):
        try:
            broker_equity = float(broker.get_equity())
        except PortfolioValuationError:
            # The broker refused to value its own book (e.g. an unmarkable
            # holding). Propagate it: it is the same refusal, raised closer to
            # the data.
            raise
        except Exception as exc:
            if gated:
                raise PortfolioValuationError(
                    f"Broker equity could not be read in {mode} mode ({exc!r}); "
                    "refusing to size targets off an unverified portfolio value.",
                    reason=REASON_BROKER_EQUITY_FAILED,
                    computed=valuation.value,
                ) from exc
            logger.warning("Broker equity unavailable (%r); using the computed valuation", exc)

    if basis == BASIS_MARGIN:
        return _resolve_margined(
            valuation=valuation,
            broker_equity=broker_equity,
            mode=mode,
        )
    return _resolve_marked(
        valuation=valuation,
        broker_equity=broker_equity,
        mode=mode,
        gated=gated,
        tolerance=tolerance,
        require_reconciliation=require_reconciliation,
    )


# ----------------------------------------------------------------------
# internals — one function per venue rule
# ----------------------------------------------------------------------


def _resolve_margined(
    *,
    valuation: PortfolioValuation,
    broker_equity: float | None,
    mode: str,
) -> PortfolioValuation:
    """A derivatives venue: the margin balance IS equity.

    Positions are leveraged and do not add to equity, so an unmarkable name does
    not understate the book — it only loses its own target. That is worth a log
    line and a metric, and it is NOT a valuation refusal, on live or anywhere
    else. Conflating the two is what made a spot book look like a perps book.

    ``broker_equity`` is preferred where the broker has one because it includes
    unrealised PnL. Where it does not, ``max(0, cash)`` is used — which is the
    margin balance the caller already read, and the number these call sites used
    before this module existed.
    """
    if valuation.unpriced:
        logger.warning(
            "%d of %d held position(s) on a margined venue could not be marked (%s). "
            "Equity is the margin balance so it is NOT understated, but these names lose "
            "their target this run.",
            len(valuation.unpriced),
            valuation.n_holdings,
            ", ".join(valuation.unpriced),
        )

    if broker_equity is None:
        logger.info(
            "Margined venue with no get_equity(): valuing the book at its margin balance %.2f (%s mode).",
            valuation.cash,
            mode,
        )
        return replace(valuation, value=valuation.cash, source="margin_balance", basis=BASIS_MARGIN)
    return replace(
        valuation,
        value=broker_equity,
        source="broker_equity",
        broker_equity=broker_equity,
        reconciled=False,
        basis=BASIS_MARGIN,
    )


def _resolve_marked(
    *,
    valuation: PortfolioValuation,
    broker_equity: float | None,
    mode: str,
    gated: bool,
    tolerance: float,
    require_reconciliation: bool,
) -> PortfolioValuation:
    """A spot/cash venue: value is ``cash + sum(qty * price)``, so the mark must be complete."""
    if gated and not valuation.is_complete:
        raise PortfolioValuationError(
            f"Pre-trade valuation is incomplete in {mode} mode: "
            f"{len(valuation.unpriced)} of {valuation.n_holdings} held position(s) "
            f"could not be marked ({', '.join(valuation.unpriced)}). "
            "Equity would be understated and every target under-sized. Refusing to trade.",
            reason=REASON_UNPRICED,
            unpriced=valuation.unpriced,
            computed=valuation.value,
            broker_equity=broker_equity,
        )
    if not valuation.is_complete:
        logger.warning(
            "Pre-trade valuation is INCOMPLETE (%d of %d held position(s) unmarkable: %s). "
            "%s mode does not gate on this, but the portfolio value is understated.",
            len(valuation.unpriced),
            valuation.n_holdings,
            ", ".join(valuation.unpriced),
            mode,
        )

    if broker_equity is None:
        return replace(valuation, basis=BASIS_MARK)

    reconciled = False
    if require_reconciliation:
        _reconcile_or_raise(
            computed=valuation.value,
            broker_equity=broker_equity,
            tolerance=tolerance,
            gated=gated,
            unpriced=valuation.unpriced,
        )
        reconciled = True

    return replace(
        valuation,
        value=broker_equity,
        source="broker_equity",
        broker_equity=broker_equity,
        reconciled=reconciled,
        basis=BASIS_MARK,
    )


def _reconcile_or_raise(
    *,
    computed: float,
    broker_equity: float,
    tolerance: float,
    gated: bool,
    unpriced: tuple[str, ...],
) -> None:
    """Compare the two views of the same book and refuse on a real disagreement."""
    denominator = max(abs(broker_equity), abs(computed))
    if denominator <= 0:
        # Both views say the book is worth nothing. That agrees, and the
        # zero-value guard downstream is what acts on it.
        return
    drift = abs(broker_equity - computed) / denominator
    if drift <= float(tolerance):
        logger.info(
            "Pre-trade reconciliation OK: computed %.2f vs broker %.2f (%.4f%% <= %.4f%%)",
            computed,
            broker_equity,
            drift * 100,
            float(tolerance) * 100,
        )
        return

    message = (
        f"Pre-trade reconciliation FAILED: pipeline valuation {computed:.2f} vs "
        f"broker equity {broker_equity:.2f} — {drift * 100:.3f}% apart, tolerance "
        f"{float(tolerance) * 100:.3f}%."
    )
    if gated:
        raise PortfolioValuationError(
            message + " Refusing to trade against an unreconciled portfolio value.",
            reason=REASON_MISMATCH,
            unpriced=unpriced,
            computed=computed,
            broker_equity=broker_equity,
        )
    logger.warning("%s Not gated in this mode.", message)


def _is_finite_positive(value: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0
