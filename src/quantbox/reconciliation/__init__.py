"""Reconciliation-break enforcement + append-only order/fill ledger (issue #87).

Two cooperating pieces that let a live book be safely re-armed:

* :class:`OrderFillLedger` — append-only per-book JSONL record of INTENT and
  OBSERVED RESULT (on fill/failure). In the current pipeline wiring intent is
  reconstructed post-execution (observe-v1), not captured at submission — see the
  ``_run_reconciliation_impl`` caveat; true submission-time capture is follow-up.
* :class:`ReconciliationStateMachine` + :func:`classify_breaks` — classify
  reconciliation breaks and compute the NORMAL → DEGRADED → HALT/FLATTEN
  transition the machine WOULD take. OBSERVE-ONLY: it only logs/alerts and never
  gates orders. ``enforce`` is REJECTED (this stage is post-execution, so gating
  here is false safety); real, Tom-gated enforcement is a future issue.

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
]
