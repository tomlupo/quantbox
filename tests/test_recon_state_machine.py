"""Reconciliation-break state machine tests (issue #87, deliverable #2).

Covers: each break class -> correct transition + action; observe-mode never
gates orders; and a REPLAY of carver-HL's real failing pattern proving the
machine WOULD have transitioned to HALT/FLATTEN instead of limping.
"""

from __future__ import annotations

from quantbox.reconciliation import (
    BookTolerances,
    BreakClass,
    ReconciliationStateMachine,
    ReconState,
    classify_breaks,
)


def _machine(mode="observe", **tol):
    return ReconciliationStateMachine(book_key="carver-HL", tol=BookTolerances(mode=mode, **tol))


# ---------------------------------------------------------------------------
# classify_breaks — each break class
# ---------------------------------------------------------------------------
def test_soft_drift_is_degraded():
    tol = BookTolerances()
    breaks = classify_breaks(tol=tol, drifts={"DOGE": 0.15})
    assert len(breaks) == 1
    assert breaks[0].klass == BreakClass.DRIFT
    assert breaks[0].severity == "soft"
    assert breaks[0].recommended_action == ReconState.DEGRADED


def test_large_drift_is_hard_flatten():
    breaks = classify_breaks(tol=BookTolerances(), drifts={"DOGE": 0.40})
    assert breaks[0].severity == "hard"
    assert breaks[0].recommended_action == ReconState.FLATTEN


def test_drift_within_tol_no_break():
    assert classify_breaks(tol=BookTolerances(), drifts={"DOGE": 0.05}) == []


def test_consecutive_failed_escalates():
    tol = BookTolerances()
    assert classify_breaks(tol=tol, failed_streaks={"DOGE": 1})[0].severity == "soft"
    hard = classify_breaks(tol=tol, failed_streaks={"DOGE": 3})[0]
    assert hard.severity == "hard"
    assert hard.recommended_action == ReconState.HALT


def test_phantom_position_is_hard_flatten():
    b = classify_breaks(tol=BookTolerances(), phantom_symbols=["ETH"])[0]
    assert b.klass == BreakClass.PHANTOM_POSITION
    assert b.severity == "hard"
    assert b.recommended_action == ReconState.FLATTEN


def test_equity_mismatch_is_hard_halt():
    b = classify_breaks(tol=BookTolerances(), equity_mismatch=0.10)[0]
    assert b.klass == BreakClass.EQUITY_MISMATCH
    assert b.recommended_action == ReconState.HALT


def test_equity_within_eps_no_break():
    assert classify_breaks(tol=BookTolerances(), equity_mismatch=0.01) == []


# ---------------------------------------------------------------------------
# state machine transitions
# ---------------------------------------------------------------------------
def test_clean_cycle_stays_normal():
    m = _machine()
    d = m.evaluate([])
    assert d.to_state == ReconState.NORMAL
    assert d.alert is None


def test_normal_to_degraded_on_soft():
    m = _machine()
    d = m.evaluate(classify_breaks(tol=m.tol, drifts={"DOGE": 0.15}))
    assert d.from_state == ReconState.NORMAL
    assert d.to_state == ReconState.DEGRADED
    assert d.transitioned
    assert d.alert is not None


def test_degraded_auto_recovers_on_clean_cycle():
    m = _machine()
    m.evaluate(classify_breaks(tol=m.tol, drifts={"DOGE": 0.15}))
    assert m.state == ReconState.DEGRADED
    d = m.evaluate([])
    assert d.to_state == ReconState.NORMAL


def test_hard_state_is_sticky_until_clear():
    m = _machine()
    m.evaluate(classify_breaks(tol=m.tol, equity_mismatch=0.10))
    assert m.state == ReconState.HALT
    # A subsequent clean cycle must NOT auto-downgrade a hard halt.
    d = m.evaluate([])
    assert d.to_state == ReconState.HALT
    m.clear()
    assert m.state == ReconState.NORMAL


def test_persistent_degraded_escalates_to_halt():
    m = _machine(max_degraded_cycles=2)
    for _ in range(2):
        m.evaluate(classify_breaks(tol=m.tol, drifts={"DOGE": 0.15}))
        assert m.state == ReconState.DEGRADED
    d = m.evaluate(classify_breaks(tol=m.tol, drifts={"DOGE": 0.15}))
    assert d.to_state == ReconState.HALT  # anti-limp rule


def test_halt_dominates_flatten():
    m = _machine()
    breaks = classify_breaks(tol=m.tol, drifts={"DOGE": 0.40}, failed_streaks={"SOL": 3})
    d = m.evaluate(breaks)
    assert d.to_state == ReconState.HALT  # HALT (rank 3) > FLATTEN (rank 2)


# ---------------------------------------------------------------------------
# OBSERVE mode never changes orders
# ---------------------------------------------------------------------------
def test_observe_mode_never_gates_orders():
    m = _machine(mode="observe")
    breaks = classify_breaks(tol=m.tol, drifts={"DOGE": 0.40}, phantom_symbols=["ETH"])
    d = m.evaluate(breaks)
    # It classifies + computes the would-be action ...
    assert d.would_be_action in (ReconState.FLATTEN, ReconState.HALT)
    assert d.to_state == d.would_be_action
    # ... but does NOT gate anything.
    assert d.orders_allowed is True
    assert d.reduce_only is False
    assert d.enforced is False
    # and the alert flags it as observe-only.
    assert "OBSERVE" in d.alert


def test_enforce_mode_is_rejected_at_config():
    """enforce is REJECTED, not defaulted-off: this stage is post-execution, so a
    gate here would be false safety. No config can construct an enforcing book."""
    import pytest

    with pytest.raises(ValueError, match="enforce"):
        BookTolerances(mode="enforce")


def test_hard_break_computes_would_be_action_without_gating():
    """Even for a hard break, observe-only NEVER gates — it only reports the
    would-be action (the shadow signal the replay test relies on)."""
    m = _machine(mode="observe")
    d = m.evaluate(classify_breaks(tol=m.tol, failed_streaks={"SOL": 3}))
    assert d.to_state == ReconState.HALT
    assert d.would_be_action == ReconState.HALT
    assert d.orders_allowed is True  # never gated here
    assert d.reduce_only is False
    assert d.enforced is False


# ---------------------------------------------------------------------------
# REPLAY: carver-HL's actual failure pattern
# ---------------------------------------------------------------------------
def test_carver_hl_replay_would_have_halted_or_flattened():
    """Seed the machine with carver-HL's real failing cycles.

    carver-HL saw ~40% drift on DOGE, 2 failed orders / 0 fills, and unexpected
    positions (ETH, SOL), yet kept limping cycle after cycle to -29.81% DD. In
    OBSERVE mode the machine only *would have* acted — it does not gate — so we
    prove the transition WITHOUT it changing any orders.
    """
    m = _machine(mode="observe")

    # Cycle 1: drift builds on DOGE, first failed order, ETH appears unexpected.
    d1 = m.evaluate(
        classify_breaks(
            tol=m.tol,
            drifts={"DOGE": 0.18},
            failed_streaks={"DOGE": 1},
            phantom_symbols=["ETH"],
        )
    )
    # A phantom position alone is a hard break -> would flatten from cycle 1.
    assert d1.would_be_action in (ReconState.FLATTEN, ReconState.HALT)

    # Cycle 2: DOGE drift now ~40%, 2 consecutive failed, SOL also unexpected.
    d2 = m.evaluate(
        classify_breaks(
            tol=m.tol,
            drifts={"DOGE": 0.40},
            failed_streaks={"DOGE": 2},
            phantom_symbols=["ETH", "SOL"],
        )
    )

    # The machine WOULD have transitioned to a hard protective state (HALT or
    # FLATTEN) — i.e. it would have caught the real incident instead of limping.
    assert d2.would_be_action in (ReconState.HALT, ReconState.FLATTEN)
    assert m.state in (ReconState.HALT, ReconState.FLATTEN)

    # OBSERVE guarantee: despite the hard would-be action, orders are untouched.
    assert d2.orders_allowed is True
    assert d2.reduce_only is False
    assert d2.enforced is False

    # And the classified breaks name the offending symbols + classes.
    classes = {b.klass for b in d2.breaks}
    assert BreakClass.DRIFT in classes
    assert BreakClass.PHANTOM_POSITION in classes
