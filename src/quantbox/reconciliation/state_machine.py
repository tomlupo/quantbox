"""Reconciliation-break detector + enforcement STATE MACHINE (issue #87, #2).

This is the ACTION layer. Each cycle, after external truth is fetched, the
detector classifies breaks and the state machine computes the transition
NORMAL -> DEGRADED -> HALT / FLATTEN and the action to take.

**OBSERVE-ONLY (by design, not by default).** The machine ALWAYS computes the
full transition and the action it *would* take (``would_be_action``), so shadow
alerting and the carver-HL replay test work. But it NEVER gates orders here:
``orders_allowed`` stays True and ``reduce_only`` stays False — ZERO
order-behavior change. ``mode`` accepts only ``"observe"`` (``"shadow"`` alias);
``BookTolerances`` REJECTS ``"enforce"``.

Why reject rather than default-off: this stage runs AFTER order execution in the
pipeline, so an "enforce" gate here would be FALSE SAFETY — it would report
orders_allowed=False / "action taken" while the orders have already been sent.
Real enforcement (consume the gate pre-execution, or gate the NEXT cycle) is a
separate future issue; until it's wired, no config may claim orders are gated.

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

    def __post_init__(self) -> None:
        # OBSERVE-ONLY by design (#87). ``enforce`` is deliberately REJECTED, not
        # merely defaulted-off: this reconciliation stage runs AFTER order
        # execution, so an "enforce" gate here would be FALSE SAFETY — it would
        # report orders_allowed=False and alert "action taken" while the orders
        # have already been sent. Real enforcement (consume the gate pre-execution
        # or gate the next cycle) is a separate future issue; until it exists,
        # accepting mode="enforce" anywhere is unsafe. The machine still computes
        # the full would_be_action so observe/shadow alerting + the replay test work.
        if self.mode not in ("observe", "shadow"):
            raise ValueError(
                f"reconciliation mode must be 'observe' (got {self.mode!r}). "
                "'enforce' is NOT accepted: this stage runs post-execution, so a "
                "gate here is false safety. Real enforcement is a future issue."
            )


@dataclass
class ReconDecision:
    """The outcome of one cycle's evaluation."""

    book_key: str
    from_state: ReconState
    to_state: ReconState
    breaks: list[Break]
    mode: str
    # would_be_action is the state the machine computed (the action it WOULD take).
    # In enforce mode it equals to_state; in observe mode it is still populated so
    # the replay/shadow test can prove the machine "would have" acted.
    would_be_action: ReconState
    alert: str | None
    # Effective gate the caller must honour. In observe mode these are always the
    # permissive values (allow everything) so there is ZERO order-behavior change.
    orders_allowed: bool
    reduce_only: bool

    @property
    def enforced(self) -> bool:
        return self.mode == "enforce"

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
        # enforce is ALWAYS False here: BookTolerances rejects mode="enforce"
        # (this stage is post-execution, so gating here would be false safety —
        # #87). The gate stays permissive; we only surface the WOULD-BE action.
        # A future issue that wires real pre-execution gating flips this on.
        enforce = False

        # Observe-only: NEVER restrict orders — zero order-behavior change — but
        # still surface the alert describing the action the machine WOULD take.
        orders_allowed = True
        reduce_only = False

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
