"""Reconciliation-break enforcement + append-only order/fill ledger (issue #87).

Two cooperating pieces that let a live book be safely re-armed:

* :class:`OrderFillLedger` — append-only per-book JSONL record of INTENT and
  OBSERVED RESULT (on fill/failure). Intent is captured at the broker submission
  call inside the pipeline's execution stage (crash-durable audit trail, #90);
  the post-execution reconciliation stage reads it back rather than reconstructing.
* :class:`ReconciliationStateMachine` + :func:`classify_breaks` — classify
  reconciliation breaks and compute the NORMAL → DEGRADED → HALT/FLATTEN
  transition. Observe by default (logs/alerts, never gates). ``enforce`` is a
  REAL mode (#90) whose gate (:func:`preflight_gate`) is consumed PRE-EXECUTION —
  the pipeline persists state and reads it back before the next cycle's orders.
  Enabling enforce on a live book is a deliberate, Tom-gated action.

Authority rule (encoded throughout): exchange = truth for HOLDINGS, internal
ledger = truth for INTENT.
"""

from quantbox.reconciliation.ledger import (
    KIND_INTENT,
    KIND_RESULT,
    RESULT_STATUSES,
    OrderFillLedger,
)
from quantbox.reconciliation.state_machine import (
    BookTolerances,
    Break,
    BreakClass,
    ReconciliationStateMachine,
    ReconDecision,
    ReconState,
    classify_breaks,
    preflight_gate,
)

__all__ = [
    "OrderFillLedger",
    "KIND_INTENT",
    "KIND_RESULT",
    "RESULT_STATUSES",
    "Break",
    "BreakClass",
    "BookTolerances",
    "ReconDecision",
    "ReconState",
    "ReconciliationStateMachine",
    "classify_breaks",
    "preflight_gate",
]
