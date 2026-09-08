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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ledger import safe_book_key

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
