"""Append-only per-book order/fill ledger — the authoritative INTENT record.

This is deliverable #1 of issue #87. It records, per book, *what the system
meant to do* (intent, on submit) and *what it observed back* (result, on fill
or failure) in an append-only JSONL log. It is the only artifact that can prove
an order that SHOULD have filled did not — external exchange state alone cannot,
because a missed fill leaves no trace on the exchange.

Authority rule (encoded here and in the detector):

    exchange = truth for HOLDINGS  (trade the reconciled real book)
    internal ledger = truth for INTENT  (detects missed / phantom fills)

Design constraints:

* **Append-only.** Records are only ever appended; nothing is mutated or
  deleted in place. A result record references its intent by ``order_ref``.
* **Single-writer.** One live loop owns one book's ledger. We do not attempt
  multi-process locking; the per-book path (namespaced by ``book_key``, matching
  quantbox-live#38's per-book pattern) guarantees no two books share a file, and
  the trading loop for a book is single-threaded and single-process.
* **JSONL.** One JSON object per line, flushed on every write, so a crash mid-run
  never corrupts earlier records and a tail-follower sees each event immediately.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# JSONL record "kind" discriminator.
KIND_INTENT = "intent"
KIND_RESULT = "result"

# Terminal result statuses. ``filled`` / ``partial`` mean the exchange confirmed
# (some) execution; the rest are non-fills the intent record lets us detect.
RESULT_STATUSES = frozenset({"filled", "partial", "rejected", "failed", "timeout", "skipped"})


@dataclass
class OrderFillLedger:
    """Append-only JSONL order/fill ledger for a single book.

    Path is ``<root>/<book_key>/orders.jsonl`` — per-book namespacing means each
    book is an independent single-writer stream. Instantiate one per book per run.

    Parameters
    ----------
    book_key:
        Stable identifier for the book (e.g. ``"carver-HL"``). Namespaces the path.
    root:
        Base data directory. The book gets its own subdirectory beneath it.
    clock:
        Callable returning an ISO-8601 UTC timestamp string. Injectable so tests
        are deterministic; defaults to wall-clock UTC.
    """

    book_key: str
    root: str | os.PathLike[str]
    clock: Any = None
    path: Path = field(init=False)

    def __post_init__(self) -> None:
        if not self.book_key:
            raise ValueError("book_key is required for a per-book ledger")
        # book_key is YAML/config-controlled and used as a path segment, so it
        # must NOT be able to escape ``root``. Reject anything that isn't a single
        # safe segment (path separators, "..", or an absolute path) — fail closed
        # rather than silently writing the ledger/state outside the recon root.
        bk = str(self.book_key)
        if bk in (".", "..") or "/" in bk or "\\" in bk or os.sep in bk or (os.altsep and os.altsep in bk):
            raise ValueError(
                f"book_key must be a single safe path segment, got {bk!r} "
                "(no path separators or '..' — it namespaces a directory under root)"
            )
        book_dir = Path(self.root) / bk
        # Defense in depth: the resolved dir must stay within root.
        root_resolved = Path(self.root).resolve()
        if root_resolved not in book_dir.resolve().parents and book_dir.resolve() != root_resolved:
            raise ValueError(f"book_key {bk!r} escapes the reconciliation root {self.root!r}")
        book_dir.mkdir(parents=True, exist_ok=True)
        self.path = book_dir / "orders.jsonl"
        if self.clock is None:
            self.clock = _utc_now_iso

    # -- writes -----------------------------------------------------------
    def record_intent(
        self,
        *,
        cycle_id: str,
        symbol: str,
        side: str,
        order_ref: str,
        target_qty: float | None = None,
        target_wt: float | None = None,
        limit_px: float | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Append an INTENT record on order submit.

        ``order_ref`` must be unique within the book so the later result record
        can be matched back to this intent. Returns the written record.
        """
        record = {
            "kind": KIND_INTENT,
            "ts": self.clock(),
            "cycle_id": str(cycle_id),
            "book": self.book_key,
            "symbol": str(symbol),
            "side": str(side).lower(),
            "target_qty": _opt_float(target_qty),
            "target_wt": _opt_float(target_wt),
            "limit_px": _opt_float(limit_px),
            "order_ref": str(order_ref),
            "intended": True,
        }
        record.update(extra)
        self._append(record)
        return record

    def record_result(
        self,
        *,
        order_ref: str,
        status: str,
        filled_qty: float | None = None,
        avg_px: float | None = None,
        cycle_id: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Append a RESULT record on order outcome, keyed to an ``order_ref``."""
        status_l = str(status).lower()
        if status_l not in RESULT_STATUSES:
            # Do not reject — an unknown status is still an observed outcome we
            # want on the record — but flag it so it is not silently normalised.
            logger.warning("Ledger result with unknown status %r for %s", status, order_ref)
        record = {
            "kind": KIND_RESULT,
            "ts": self.clock(),
            "book": self.book_key,
            "order_ref": str(order_ref),
            "status": status_l,
            "filled_qty": _opt_float(filled_qty),
            "avg_px": _opt_float(avg_px),
        }
        if cycle_id is not None:
            record["cycle_id"] = str(cycle_id)
        record.update(extra)
        self._append(record)
        return record

    def _append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"), sort_keys=False)
        # Open-append-close per write: cheap at cycle cadence and guarantees the
        # OS flushes each record, so a crash cannot corrupt earlier lines.
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    # -- reads ------------------------------------------------------------
    def read_all(self) -> list[dict[str, Any]]:
        """Read every record in append order. Skips blank/corrupt trailing lines."""
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with open(self.path, encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    out.append(json.loads(raw))
                except json.JSONDecodeError:
                    # An append-only log can only ever be corrupt on the final
                    # line (a crash mid-write). Log and stop rather than raise.
                    logger.warning("Skipping corrupt ledger line %d in %s", lineno, self.path)
                    break
        return out

    def intents_for_cycle(self, cycle_id: str) -> list[dict[str, Any]]:
        """All INTENT records for a given cycle."""
        cid = str(cycle_id)
        return [r for r in self.read_all() if r.get("kind") == KIND_INTENT and r.get("cycle_id") == cid]

    def match_intents_to_results(self) -> dict[str, dict[str, Any]]:
        """Join intents to their results by ``order_ref``.

        Returns ``{order_ref: {"intent": <record>, "result": <record|None>}}``.
        An intent with ``result=None`` is a submitted-but-unresolved order — the
        exact signature of a *missed* fill that the internal ledger can prove and
        the exchange alone cannot.
        """
        joined: dict[str, dict[str, Any]] = {}
        for r in self.read_all():
            ref = r.get("order_ref")
            if ref is None:
                continue
            slot = joined.setdefault(ref, {"intent": None, "result": None})
            if r.get("kind") == KIND_INTENT:
                slot["intent"] = r
            elif r.get("kind") == KIND_RESULT:
                # Last result wins if an order is resolved more than once.
                slot["result"] = r
        return joined


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
