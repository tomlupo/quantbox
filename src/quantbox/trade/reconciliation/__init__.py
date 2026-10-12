"""Reconciliation of a live book: the cycle, the order/fill ledger, the break state machine, the working-order queue.

``quantbox.reconciliation`` until 0.16.0 (TOM-1455, ADR-0011). The package owns
the whole cycle the trading pipeline runs once per run — load → gate → record →
close (:mod:`.cycle`, :func:`open_cycle`) — and the parts it is made of:

* :class:`OrderFillLedger` — append-only per-book JSONL record of INTENT and
  OBSERVED RESULT (on fill/failure). Intent is captured at the broker submission
  call inside the pipeline's execution stage (crash-durable audit trail, #90);
  the cycle's close reads it back rather than reconstructing.
* :class:`ReconciliationStateMachine` + :func:`classify_breaks` — classify
  reconciliation breaks and compute the NORMAL → DEGRADED → HALT/FLATTEN
  transition. Observe by default (logs/alerts, never gates). ``enforce`` is a
  REAL mode (#90) whose gate (:func:`preflight_gate`) is consumed PRE-EXECUTION —
  the cycle persists state and reads it back before the next cycle's orders.
  Enabling enforce on a live book is a deliberate, Tom-gated action.
* :class:`WorkingOrderStore` + :func:`resolve_working_orders` — orders left
  resting at the venue, resolved by a later cycle or by quantbox-live's
  out-of-cycle job.

Authority rule (encoded throughout): exchange = truth for HOLDINGS, internal
ledger = truth for INTENT.
"""

from quantbox.trade.reconciliation.cycle import (
    CycleRun,
    ReconCycle,
    ReconEnforcementError,
    filter_reduce_only,
    open_cycle,
)
from quantbox.trade.reconciliation.ledger import (
    KIND_INTENT,
    KIND_RESULT,
    RESULT_STATUSES,
    OrderFillLedger,
    safe_book_key,
)
from quantbox.trade.reconciliation.state_machine import (
    BookTolerances,
    Break,
    BreakClass,
    ReconciliationStateMachine,
    ReconDecision,
    ReconState,
    classify_breaks,
    preflight_gate,
)
from quantbox.trade.reconciliation.working_orders import (
    DEFAULT_MAX_AGE_DAYS,
    WorkingOrderResolution,
    WorkingOrderStore,
    record_age_seconds,
    resolve_working_orders,
)

__all__ = [
    "CycleRun",
    "DEFAULT_MAX_AGE_DAYS",
    "OrderFillLedger",
    "KIND_INTENT",
    "KIND_RESULT",
    "RESULT_STATUSES",
    "Break",
    "BreakClass",
    "BookTolerances",
    "ReconCycle",
    "ReconDecision",
    "ReconEnforcementError",
    "ReconState",
    "ReconciliationStateMachine",
    "WorkingOrderResolution",
    "WorkingOrderStore",
    "classify_breaks",
    "filter_reduce_only",
    "open_cycle",
    "preflight_gate",
    "record_age_seconds",
    "resolve_working_orders",
    "safe_book_key",
]
