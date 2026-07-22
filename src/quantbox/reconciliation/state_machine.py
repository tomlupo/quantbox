"""Reconciliation-break detector + enforcement STATE MACHINE (issue #87, #90).

This is the ACTION layer. Each cycle, after external truth is fetched, the
detector classifies breaks and the state machine computes the transition
NORMAL -> DEGRADED -> HALT / FLATTEN and the action to take.

**Observe by default; real enforcement is Tom-gated (issue #90).** The machine
ALWAYS computes the full transition and the action it *would* take
(``would_be_action``), so shadow alerting and the carver-HL replay test work.

* ``mode="observe"`` (default; ``"shadow"`` alias) — NEVER gates orders:
  ``orders_allowed`` stays True and ``reduce_only`` stays False. ZERO
  order-behavior change.
* ``mode="enforce"`` — the machine's resulting state DOES set the gate
  (``orders_allowed`` / ``reduce_only``). This is only safe because the gate is
  consumed PRE-EXECUTION: the pipeline persists the state after each cycle and
  reads it back BEFORE constructing/executing the next cycle's orders (the
  "gate the NEXT cycle" model — see :func:`preflight_gate`). Enforcing here is
  NOT false safety the way a post-execution gate would be, because no order is
  sent on a decision computed too late. Enabling enforce on a live book is a
  deliberate, Tom-gated action (an explicit config acknowledgment plus a deploy),
  never a casual flag flip — the pipeline REFUSES ``enforce`` unless the book
  config carries an explicit acknowledgment (see the trading pipeline).

Authority rule (see also ledger.py):

    exchange = truth for HOLDINGS  (trade the reconciled real book)
    internal ledger = truth for INTENT  (detects missed / phantom fills)

The detector diffs internal intent (ledger) against external truth (exchange),
but any resulting *trading* decision is expressed against the reconciled real
book — we never act on internal-only state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ReconState(str, Enum):
    """Per-book enforcement state.

    Ordered by severity via :data:`_STATE_RANK`:
    ``NORMAL < DEGRADED < FLATTEN < HALT``.

    * ``NORMAL``   — reconciled; trade normally.
    * ``DEGRADED`` — soft break(s); alert + bounded retry/backoff, orders still
      allowed. Auto-recovers to NORMAL on a clean cycle.
    * ``FLATTEN``  — reduce-only: only orders that reduce/exit existing exposure
      are allowed (get out of trapped/phantom positions, take on nothing new).
    * ``HALT``     — no new orders at all; the order path or the books are not
      trustworthy. Requires manual clear (Tom) to leave.
    """

    NORMAL = "normal"
    DEGRADED = "degraded"
    FLATTEN = "flatten"
    HALT = "halt"


_STATE_RANK = {
    ReconState.NORMAL: 0,
    ReconState.DEGRADED: 1,
    ReconState.FLATTEN: 2,
    ReconState.HALT: 3,
}

# A hard halt/flatten does not auto-clear: reconciling for a cycle is not proof
# the underlying fault is gone. Only DEGRADED auto-recovers.
_HARD_STATES = {ReconState.FLATTEN, ReconState.HALT}

# Recognised enforcement modes. ``shadow`` is an alias for ``observe``.
_OBSERVE_MODES = frozenset({"observe", "shadow"})
_ENFORCE_MODE = "enforce"
_VALID_MODES = _OBSERVE_MODES | {_ENFORCE_MODE}


def preflight_gate(state: ReconState) -> tuple[bool, bool]:
    """Map a persisted per-book state to a PRE-EXECUTION order gate.

    Returns ``(orders_allowed, reduce_only)``:

    * ``HALT``    → ``(False, False)`` — no new orders at all.
    * ``FLATTEN`` → ``(True, True)``   — reduce-only: only orders that reduce or
      exit existing exposure are allowed.
    * ``DEGRADED`` / ``NORMAL`` → ``(True, False)`` — trade normally.

    This is a PURE function of the state so the pipeline can consume it before it
    constructs/executes a cycle's orders. Whether the gate is actually APPLIED is
    the caller's decision (only in enforce mode); observe mode computes it for
    reporting but never applies it.
    """
    if state == ReconState.HALT:
        return (False, False)
    if state == ReconState.FLATTEN:
        return (True, True)
    return (True, False)


class BreakClass(str, Enum):
    """The break taxonomy from issue #87."""

    DRIFT = "drift"  # per-symbol target-vs-actual drift > tolerance
    CONSECUTIVE_FAILED = "consecutive_failed"  # N consecutive failed/zero-fill orders
    PHANTOM_POSITION = "phantom_position"  # holding with no matching intent
    EQUITY_MISMATCH = "equity_mismatch"  # start + flows + pnl != equity beyond eps


@dataclass(frozen=True)
class Break:
    """A single classified reconciliation break."""

    klass: BreakClass
    symbol: str | None
    severity: str  # "soft" | "hard"
    detail: str
    recommended_action: ReconState


@dataclass
class BookTolerances:
    """Per-book tolerances + enforcement mode. Conservative defaults.

    A drift above ``max_drift`` is a soft break; above ``drift_halt`` it is hard
    (the book is materially off its target and could be trapped). Consecutive
    failed/zero-fill orders escalate from soft (at ``degraded_failed_streak``) to
    hard (at ``halt_failed_streak``). A phantom position and an equity mismatch
    are always hard.
    """

    mode: str = "observe"  # "observe" only — see __post_init__ (#87)
    max_drift: float = 0.10  # 10% target-vs-actual drift → soft break
    drift_halt: float = 0.25  # 25% drift → hard break (trapped/uncontrolled)
    degraded_failed_streak: int = 1  # 1 failed/zero-fill → soft
    halt_failed_streak: int = 3  # 3 consecutive → hard (order path broken)
    equity_eps: float = 0.02  # 2% equity reconciliation tolerance
    max_degraded_cycles: int = 3  # DEGRADED this many cycles → escalate to HALT
    # Dust guard, in quote currency. Drift and phantom detection ignore positions
    # worth less than this. Without it a $2 residual against a zero target reads as
    # a 100% drift and a hard phantom every single cycle — and a book that halts on
    # dust is a book whose halt gets switched off (quantbox#87).
    drift_notional_floor: float = 10.0
    phantom_lookback: int = 2  # a holding intended within this many recent cycles
    # is explained (carryover); older reappearances count as phantom.

    def __post_init__(self) -> None:
        # Observe by default; ``enforce`` is now a real mode (#90) because the
        # pipeline consumes the gate PRE-EXECUTION (persist state, read it back
        # before the next cycle's orders — see preflight_gate). A post-execution
        # gate would still be false safety, but that is no longer the enforcement
        # path. Enabling enforce on a live book stays a deliberate, Tom-gated
        # action: the trading pipeline additionally REFUSES enforce unless the
        # book config carries an explicit acknowledgment. Here we only validate
        # the mode string.
        if self.mode not in _VALID_MODES:
            raise ValueError(f"reconciliation mode must be one of {sorted(_VALID_MODES)} (got {self.mode!r}).")

    @property
    def is_enforce(self) -> bool:
        return self.mode == _ENFORCE_MODE


@dataclass
class ReconDecision:
    """The outcome of one cycle's evaluation."""

    book_key: str
    from_state: ReconState
    to_state: ReconState
    breaks: list[Break]
    mode: str
    # would_be_action is the state the machine computed (the action it WOULD take).
    # Populated in every mode so the replay/shadow test can prove the machine
    # "would have" acted (observe-only never acts on it).
    would_be_action: ReconState
    alert: str | None
    # Observe-only gate: ALWAYS the permissive values (allow everything) — ZERO
    # order-behavior change. Kept on the dataclass for the future enforcement
    # wiring (a pre-execution consumer), which does not exist yet.
    orders_allowed: bool
    reduce_only: bool

    @property
    def enforced(self) -> bool:
        # True iff this decision was computed under enforce mode. The gate
        # (orders_allowed / reduce_only) is only actually consumed by the pipeline
        # PRE-EXECUTION (next-cycle gating); see preflight_gate.
        return self.mode == _ENFORCE_MODE

    @property
    def transitioned(self) -> bool:
        return self.from_state != self.to_state


# ---------------------------------------------------------------------------
# Detector — classify breaks from external truth vs internal intent
# ---------------------------------------------------------------------------
def classify_breaks(
    *,
    tol: BookTolerances,
    drifts: dict[str, float] | None = None,
    failed_streaks: dict[str, int] | None = None,
    phantom_symbols: list[str] | None = None,
    equity_mismatch: float | None = None,
) -> list[Break]:
    """Classify a cycle's reconciliation breaks.

    Inputs are already-computed reconciliation signals (the trading pipeline
    already computes target-vs-actual drift and unexpected positions):

    * ``drifts`` — ``{symbol: abs_fractional_drift}`` of target vs actual weight.
    * ``failed_streaks`` — ``{symbol: consecutive_failed_or_zero_fill_count}``.
    * ``phantom_symbols`` — holdings on the exchange with no matching intent in
      the ledger (authority rule: exchange=holdings truth, ledger=intent truth).
    * ``equity_mismatch`` — ``abs((start+flows+pnl) - equity) / equity``; a
      fractional residual beyond ``equity_eps`` is a hard break.
    """
    breaks: list[Break] = []

    for sym, d in (drifts or {}).items():
        ad = abs(float(d))
        if ad >= tol.drift_halt:
            breaks.append(
                Break(
                    BreakClass.DRIFT,
                    sym,
                    "hard",
                    f"drift {ad:.1%} >= halt tol {tol.drift_halt:.1%}",
                    # Large drift on a held name → get out of it: reduce-only.
                    ReconState.FLATTEN,
                )
            )
        elif ad > tol.max_drift:
            breaks.append(
                Break(
                    BreakClass.DRIFT,
                    sym,
                    "soft",
                    f"drift {ad:.1%} > tol {tol.max_drift:.1%}",
                    ReconState.DEGRADED,
                )
            )

    for sym, streak in (failed_streaks or {}).items():
        n = int(streak)
        if n >= tol.halt_failed_streak:
            breaks.append(
                Break(
                    BreakClass.CONSECUTIVE_FAILED,
                    sym,
                    "hard",
                    f"{n} consecutive failed/zero-fill orders >= {tol.halt_failed_streak}",
                    # A broken order path: stop sending orders entirely.
                    ReconState.HALT,
                )
            )
        elif n >= tol.degraded_failed_streak:
            breaks.append(
                Break(
                    BreakClass.CONSECUTIVE_FAILED,
                    sym,
                    "soft",
                    f"{n} consecutive failed/zero-fill orders >= {tol.degraded_failed_streak}",
                    ReconState.DEGRADED,
                )
            )

    for sym in phantom_symbols or []:
        breaks.append(
            Break(
                BreakClass.PHANTOM_POSITION,
                sym,
                "hard",
                "position held with no matching intent in ledger",
                # Unexpected exposure we never asked for → reduce it out.
                ReconState.FLATTEN,
            )
        )

    if equity_mismatch is not None and abs(float(equity_mismatch)) > tol.equity_eps:
        breaks.append(
            Break(
                BreakClass.EQUITY_MISMATCH,
                None,
                "hard",
                f"equity residual {abs(float(equity_mismatch)):.1%} > eps {tol.equity_eps:.1%}",
                # Books don't reconcile → trust nothing, halt.
                ReconState.HALT,
            )
        )

    return breaks


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
@dataclass
class ReconciliationStateMachine:
    """Per-book reconciliation enforcement state machine.

    Stateful across cycles: holds the current state and a DEGRADED-cycle counter
    so a book that keeps limping in DEGRADED eventually escalates to HALT — the
    exact failure mode carver-HL exhibited (limping cycle after cycle).
    """

    book_key: str
    tol: BookTolerances = field(default_factory=BookTolerances)
    state: ReconState = ReconState.NORMAL
    degraded_cycles: int = 0

    def evaluate(self, breaks: list[Break]) -> ReconDecision:
        """Advance the machine by one cycle given this cycle's breaks."""
        from_state = self.state
        target = self._compute_target(breaks)

        # Update the machine's own state (this is bookkeeping, not order-gating,
        # so it runs in observe mode too — observe still tracks what WOULD happen).
        self.state = target
        if target == ReconState.DEGRADED:
            self.degraded_cycles += 1
        elif target == ReconState.NORMAL:
            self.degraded_cycles = 0

        would_be_action = target
        enforce = self.tol.mode == _ENFORCE_MODE

        # The gate for THIS decision reflects the resulting state — but only under
        # enforce mode does it carry non-permissive values. Observe/shadow ALWAYS
        # report the permissive gate (zero order-behavior change), while still
        # surfacing the would_be_action + alert. The pipeline consumes the gate
        # PRE-EXECUTION on the NEXT cycle (persist-then-read), never here.
        if enforce:
            orders_allowed, reduce_only = preflight_gate(target)
        else:
            orders_allowed, reduce_only = True, False

        alert = self._build_alert(from_state, target, breaks, enforce)

        return ReconDecision(
            book_key=self.book_key,
            from_state=from_state,
            to_state=target,
            breaks=breaks,
            mode=self.tol.mode,
            would_be_action=would_be_action,
            alert=alert,
            orders_allowed=orders_allowed,
            reduce_only=reduce_only,
        )

    def clear(self) -> None:
        """Manual clear of a hard halt/flatten (Tom-only in production)."""
        self.state = ReconState.NORMAL
        self.degraded_cycles = 0

    def _compute_target(self, breaks: list[Break]) -> ReconState:
        # Most-severe recommended action across this cycle's breaks.
        worst = ReconState.NORMAL
        for b in breaks:
            if _STATE_RANK[b.recommended_action] > _STATE_RANK[worst]:
                worst = b.recommended_action

        # Sticky hard states: a hard halt/flatten never auto-downgrades just
        # because a single cycle looks clean — only an explicit clear() leaves it.
        if self.state in _HARD_STATES and _STATE_RANK[worst] < _STATE_RANK[self.state]:
            return self.state

        # DEGRADED that persists too long escalates to HALT (the anti-limp rule).
        if worst == ReconState.DEGRADED and self.degraded_cycles + 1 > self.tol.max_degraded_cycles:
            return ReconState.HALT

        return worst

    def _build_alert(
        self,
        from_state: ReconState,
        to_state: ReconState,
        breaks: list[Break],
        enforce: bool,
    ) -> str | None:
        # Tier-2 hard alert on any transition INTO a non-NORMAL state, or while
        # sitting in a hard state with fresh breaks. No alert on a clean NORMAL
        # cycle (that is the daily-glance's job, not a hard alert).
        if to_state == ReconState.NORMAL and from_state == ReconState.NORMAL:
            return None

        classes = sorted({b.klass.value for b in breaks})
        symbols = sorted({b.symbol for b in breaks if b.symbol})
        action_word = {
            ReconState.NORMAL: "recovered to NORMAL",
            ReconState.DEGRADED: "DEGRADED (alert + bounded retry)",
            ReconState.FLATTEN: "FLATTEN-ONLY (reduce/exit only)",
            ReconState.HALT: "HALT (no new orders)",
        }[to_state]
        mode_note = "ENFORCE — action taken" if enforce else "OBSERVE — would have acted; orders NOT gated"
        return (
            f"🚨 RECON [{self.book_key}] {from_state.value} → {to_state.value}: {action_word}\n"
            f"Break class(es): {', '.join(classes) or 'none'}\n"
            f"Symbol(s): {', '.join(symbols) or '—'}\n"
            f"Mode: {mode_note}"
        )
