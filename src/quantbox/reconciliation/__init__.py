"""Reconciliation-break enforcement + append-only order/fill ledger (issue #87).

Two cooperating pieces that let a live book be safely re-armed:

* :class:`OrderFillLedger` — append-only per-book JSONL record of INTENT (on
  submit) and OBSERVED RESULT (on fill/failure). The authoritative intent record.
* :class:`ReconciliationStateMachine` + :func:`classify_breaks` — classify
  reconciliation breaks and compute the NORMAL → DEGRADED → HALT/FLATTEN
  enforcement transition. Runs in ``observe`` mode by default (log/alert only,
  zero order-behavior change); ``enforce`` gates orders and is Tom-only.

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
