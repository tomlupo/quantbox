"""Cross-cycle store for orders left WORKING at the venue.

A priced order is submitted to Kraken as a LIMIT order, and a limit order may
legitimately rest on the book long after the run that placed it has exited. The
in-run confirmation window (``_fills.resolve_fill``) is bounded at a few seconds
because a run cannot block on the market; anything still alive when it closes is
emitted as ``WORKING``.

That outcome is only honest if somebody later finds out what happened. This
module is that somebody: the pipeline records each working order here at the end
of a run, and the NEXT run resolves each one against the venue and books the real
fill. Without it, ``WORKING`` would simply be a quieter way of losing a fill —
which is exactly the failure the old blanket ``FAILED`` produced, minus the alert.

Design constraints, mirroring ``ledger.OrderFillLedger``:

* **Per-book path**, namespaced by a validated ``book_key``, so two books can
  never share a file and a config-supplied key cannot escape the data root.
* **Whole-file rewrite**, not append: unlike the append-only intent ledger this
  is a *work queue* whose entries are removed once resolved. It is small (a
  handful of orders at most) and written by a single-process run loop.
* **Fail-soft on read**: a corrupt or missing file yields an empty queue and a
  warning, never an exception — a broken queue must not stop the book trading.
  It must NOT fail soft on write: silently failing to record a working order
  loses a real fill.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ledger import safe_book_key, to_ledger_status

logger = logging.getLogger(__name__)

FILENAME = "working_orders.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class WorkingOrderStore:
    """The per-book queue of orders still working at the venue."""

    book_key: str
    root: str | Path = "data"

    def __post_init__(self) -> None:
        safe = safe_book_key(self.book_key, self.root)
        self.path = Path(self.root) / safe / FILENAME

    # -- read ---------------------------------------------------------------
    def load(self) -> list[dict[str, Any]]:
        """Return the queued working orders. Never raises."""
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Working-order queue unreadable at %s: %s", self.path, exc)
            return []
        if not isinstance(data, list):
            logger.warning("Working-order queue at %s is not a list — ignoring", self.path)
            return []
        return [r for r in data if isinstance(r, dict)]

    # -- write --------------------------------------------------------------
    def save(self, records: list[dict[str, Any]]) -> None:
        """Replace the queue. Raises on failure — a lost record is a lost fill."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(records, indent=2, default=str))
        tmp.replace(self.path)

    def record(
        self,
        *,
        symbol: str,
        side: str,
        order_id: str,
        requested_qty: float,
        cycle_id: str,
        reason: str = "",
        order_ref: str | None = None,
    ) -> None:
        """Queue one order for resolution by a later cycle.

        ``order_ref`` is the intent reference from the order/fill ledger, carried
        so that when the fill is finally observed it can be bound back to the
        intent that produced it instead of landing as an orphan result.
        """
        records = self.load()
        records.append(
            {
                "symbol": str(symbol),
                "side": str(side),
                "order_id": str(order_id),
                "requested_qty": float(requested_qty),
                "cycle_id": str(cycle_id),
                "reason": str(reason),
                "order_ref": str(order_ref) if order_ref else None,
                "recorded_at": _utc_now_iso(),
            }
        )
        self.save(records)

    def drop(self, order_ids: set[str]) -> None:
        """Remove resolved orders from the queue."""
        if not order_ids:
            return
        self.save([r for r in self.load() if str(r.get("order_id")) not in order_ids])


# ===========================================================================
# Resolution against the venue
# ===========================================================================
# The semantics below are the WHOLE contract of the working-order queue, and
# they have exactly one implementation because two callers need them:
#
#   * the trading pipeline's Stage 6c, on the NEXT daily cycle; and
#   * the out-of-cycle follow-up job, an hour after a cycle, which exists so a
#     late fill is booked in an hour rather than a day.
#
# A second copy of these rules in a script would be a second definition of what
# "could not check" means, which is the one distinction the queue is for.


# Orders left WORKING are queued for a few days before we stop expecting them;
# a limit order that has not resolved by then is either long dead or long
# filled, and position reconciliation covers the book either way. The cap exists
# so an order the venue never resolves cannot grow the queue without bound.
DEFAULT_MAX_AGE_DAYS = 7


@dataclass
class WorkingOrderResolution:
    """What one resolution pass learned. Every queued order lands in exactly one
    bucket, so a caller can tell the three outcomes apart:

    * ``resolved`` — the venue gave a terminal answer; booked and dequeued.
    * ``still_working`` — the venue answered "still on the book". A fact, not a
      failure.
    * ``unreadable`` — the venue could NOT be read. This is the bucket that
      separates "checked and found nothing" from "did not manage to check", and
      a caller that collapses it into either of the other two is lying.

    ``dropped`` lists the records removed from the queue this pass, whether
    booked or aged out; ``expired`` is the loud subset that aged out unresolved.
    """

    queued: int = 0
    resolved: list[dict[str, Any]] = field(default_factory=list)
    still_working: list[dict[str, Any]] = field(default_factory=list)
    unreadable: list[dict[str, Any]] = field(default_factory=list)
    expired: list[dict[str, Any]] = field(default_factory=list)
    checked: bool = False

    @property
    def unresolved(self) -> list[dict[str, Any]]:
        """Queued orders that did not reach a terminal state this pass."""
        return [*self.still_working, *self.unreadable]

    def older_than(self, seconds: float, *, now: datetime | None = None) -> list[dict[str, Any]]:
        """Unresolved orders queued more than ``seconds`` ago.

        This is the "still open an hour later" question. An order whose
        ``recorded_at`` cannot be parsed is INCLUDED — an unreadable age must
        not silently read as "young enough to ignore".
        """
        moment = now or datetime.now(timezone.utc)
        out = []
        for rec in self.unresolved:
            age = record_age_seconds(rec, now=moment)
            if age is None or age > seconds:
                out.append(rec)
        return out


def record_age_seconds(rec: dict[str, Any], *, now: datetime | None = None) -> float | None:
    """Seconds since the order was queued, or None when that cannot be known."""
    raw = rec.get("recorded_at")
    if not raw:
        return None
    try:
        recorded = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
    if recorded.tzinfo is None:
        recorded = recorded.replace(tzinfo=timezone.utc)
    moment = now or datetime.now(timezone.utc)
    return (moment - recorded).total_seconds()


def _expired(rec: dict[str, Any], now: datetime, max_age_days: int) -> bool:
    """True when a queued order is older than the retention cap.

    An order whose ``recorded_at`` is missing or unparseable is NOT expired: we
    do not know its age, and dropping a real fill on an unreadable timestamp is
    the one outcome this queue exists to prevent.
    """
    age = record_age_seconds(rec, now=now)
    return age is not None and age > max_age_days * 86400


def resolve_working_orders(
    store: WorkingOrderStore,
    broker: Any,
    *,
    ledger: Any = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    now: datetime | None = None,
) -> WorkingOrderResolution:
    """Ask the venue what became of orders a previous cycle left working.

    An order the venue cannot be read for stays queued (an API blip must never
    discard a real fill); an order still working stays queued too. A terminal
    answer is recorded in ``ledger`` against the carried ``order_ref`` and
    dropped. Past ``max_age_days`` an unresolved order is dropped LOUDLY.

    Returns a :class:`WorkingOrderResolution`. ``checked`` is False when no
    resolution was even attempted (nothing queued, or a broker that cannot
    resolve) — a caller reporting "clean" must consult it.
    """
    queued = store.load()
    result = WorkingOrderResolution(queued=len(queued))
    if not queued:
        result.checked = True  # nothing to check IS a complete check
        return result

    resolver = getattr(broker, "fetch_order_result", None)
    if not callable(resolver):
        logger.warning(
            "%d working order(s) queued but broker %s cannot resolve orders — their fills will not be booked",
            len(queued),
            type(broker).__name__,
        )
        # checked stays False: we did not look, and that is not "nothing found".
        result.unreadable = list(queued)
        return result

    drop_ids: set[str] = set()
    moment = now or datetime.now(timezone.utc)
    for rec in queued:
        order_id = str(rec.get("order_id") or "")
        symbol = str(rec.get("symbol") or "")
        if not order_id:
            # An entry with no order id can never be resolved; drop it rather
            # than carry an unanswerable record forever.
            logger.error("Working-order entry for %s carries no order_id — dropping it", symbol or "?")
            drop_ids.add(order_id)
            continue

        try:
            outcome = resolver(order_id, symbol)
        except Exception as exc:  # noqa: BLE001 - a venue error is "could not check"
            logger.warning(
                "Working order %s %s (id=%s) raised while resolving — left queued: %s",
                rec.get("side"),
                symbol,
                order_id,
                exc,
            )
            outcome = None

        if outcome is None:
            # Could not read the venue. NOT a failure — keep it queued so the
            # next pass tries again. This is the branch that separates
            # "cannot check" from "checked and found nothing".
            logger.warning(
                "Working order %s %s (id=%s) could not be resolved this pass — left queued",
                rec.get("side"),
                symbol,
                order_id,
            )
            result.unreadable.append(rec)
            if _expired(rec, moment, max_age_days):
                logger.error(
                    "Working order %s %s (id=%s) has been unresolved for over %d days "
                    "— dropping from the queue; reconcile it by hand",
                    rec.get("side"),
                    symbol,
                    order_id,
                    max_age_days,
                )
                result.expired.append(rec)
                drop_ids.add(order_id)
            continue

        status = str(outcome.get("status", "")).strip().upper()
        if status == "WORKING":
            logger.info(
                "Working order %s %s (id=%s) is STILL working at the venue",
                rec.get("side"),
                symbol,
                order_id,
            )
            result.still_working.append(rec)
            if _expired(rec, moment, max_age_days):
                logger.error(
                    "Working order %s %s (id=%s) has rested for over %d days — "
                    "dropping from the queue; reconcile it by hand",
                    rec.get("side"),
                    symbol,
                    order_id,
                    max_age_days,
                )
                result.expired.append(rec)
                drop_ids.add(order_id)
            continue

        # Terminal: FILLED / PARTIAL / FAILED. Book it and stop tracking it.
        drop_ids.add(order_id)
        entry = {
            "symbol": symbol,
            "side": str(rec.get("side", "")),
            "order_id": order_id,
            "status": status,
            "quantity": float(outcome.get("qty", 0) or 0.0),
            "price": float(outcome.get("price", 0) or 0.0),
            "error": str(outcome.get("error", "")),
            "placed_cycle_id": str(rec.get("cycle_id", "")),
        }
        result.resolved.append(entry)
        logger.info(
            "Working order resolved LATE: %s %s (id=%s) -> %s qty=%s @ %s",
            entry["side"],
            symbol,
            order_id,
            status,
            entry["quantity"],
            entry["price"],
        )
        order_ref = rec.get("order_ref")
        if ledger is not None and order_ref:
            try:
                ledger.record_result(
                    order_ref=str(order_ref),
                    cycle_id=str(rec.get("cycle_id") or ""),
                    status=to_ledger_status(status),
                    filled_qty=entry["quantity"],
                    avg_px=entry["price"],
                    resolved_late=True,
                )
            except Exception:  # noqa: BLE001 - ledger write must not lose the resolution
                logger.exception(
                    "Failed to record LATE result for %s (ref=%s) in the ledger",
                    symbol,
                    order_ref,
                )

    store.drop(drop_ids)
    result.checked = True
    return result
