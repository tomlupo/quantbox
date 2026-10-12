"""The reconciliation cycle of one book: load -> gate -> record -> close (TOM-1455).

The trading pipeline runs one cycle per run and holds nothing of it but a
:class:`CycleRun`:

1. **load** — :func:`open_cycle` parses the ``reconciliation`` block ONCE:
   tolerances, the ledger, the persisted state and the state machine
   (:meth:`ReconCycle.load`). Enforce is honoured only with
   ``enforce_acknowledged: true``.
2. **gate** — the same call derives the PRE-EXECUTION gate from the persisted
   prior-cycle state (:meth:`ReconCycle.gate`). In enforce mode a HALT blocks
   every order and a FLATTEN keeps only exposure-reducing ones
   (:func:`filter_reduce_only`); observe mode records the gate and changes
   nothing. A load or gate fault fails CLOSED under acknowledged enforce and
   open in observe.
3. **record** — the execution stage writes intent and result to
   :attr:`CycleRun.ledger` at the broker call (crash-durable, #90).
4. **close** — :meth:`CycleRun.close` classifies the breaks against the marked
   book (``quantbox.trade.book``: a held name with no mark has UNKNOWN drift,
   never a -100% drift), computes the NORMAL → DEGRADED → HALT/FLATTEN
   transition, persists it atomically for the next cycle's gate, and alerts a
   transition. It never fails the run, except :class:`ReconEnforcementError`
   (the enforcement authority itself is unwritable under enforce).

Authority rule: exchange = truth for HOLDINGS, the ledger = truth for INTENT.

This was ``TradingPipeline._recon_load``, ``_recon_preflight``,
``_filter_reduce_only``, ``_run_reconciliation`` and ``_ReconCtx`` until 0.16.0;
``_recon_load`` then ran twice per run (the gate's and the close's), against
its own "built once" docstring.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from quantbox.trade.book import read_book
from quantbox.trade.reconciliation.ledger import KIND_INTENT, NON_TERMINAL_RESULT_STATUSES, OrderFillLedger
from quantbox.trade.reconciliation.state_machine import (
    BookTolerances,
    ReconciliationStateMachine,
    ReconState,
    classify_breaks,
    preflight_gate,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Alert",
    "CycleRun",
    "ReconCycle",
    "ReconEnforcementError",
    "atomic_write_text",
    "enforce_requested",
    "filter_reduce_only",
    "open_cycle",
]

#: The alert channel the cycle sends through: ``alert(message, site=...)``,
#: never raising (``quantbox.trade.notifier.Alerts.send``).
Alert = Callable[..., Any]


class ReconEnforcementError(RuntimeError):
    """A CRITICAL enforce-mode reconciliation failure that must NOT be swallowed.

    Raised when the enforcement authority itself is untrustworthy under acknowledged
    enforce (e.g. the persisted state file cannot be written). Unlike ordinary recon
    faults — which are observability losses and never fatal — this must escape the
    guard and fail the run, so the book stops rather than silently trading the next
    cycle from stale/absent enforcement state.
    """


def atomic_write_text(path: Any, text: str) -> None:
    """Durably replace ``path`` with ``text``: write a temp file in the same dir,
    fsync it, then ``os.replace`` (atomic on POSIX). A crash can never leave the
    target truncated/half-written — the reader sees either the old file or the new
    one, never a corrupt one. Used for the recon state file, which IS the
    enforcement authority (a corrupt state must never silently fail open).
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        # fsync the parent directory so the rename itself is crash-durable on POSIX
        # (a bare os.replace is not persisted until the dir entry is synced).
        with contextlib.suppress(OSError, AttributeError):
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def enforce_requested(params: Mapping[str, Any]) -> bool:
    """True iff the config asks for acknowledged enforce, WITHOUT loading the
    ledger. Decides fail-open vs fail-closed when :meth:`ReconCycle.load` itself
    raised (so its parsed mode cannot be trusted)."""
    cfg = params.get("reconciliation")
    if not isinstance(cfg, dict):
        return False
    tol_cfg = cfg.get("tolerances")
    tol_mode = tol_cfg.get("mode") if isinstance(tol_cfg, dict) else None
    mode = str(cfg.get("mode", tol_mode or "observe"))
    return mode == "enforce" and bool(cfg.get("enforce_acknowledged", False))


@dataclass
class ReconCycle:
    """One book's loaded cycle: config parsed once, the SAME ledger and persisted
    state behind the gate and the close."""

    book_key: str
    root: Path
    tol: BookTolerances
    cycle_id: str
    mode: str
    enforce_refused: bool
    ledger: OrderFillLedger
    machine: ReconciliationStateMachine
    state_path: Path
    persisted: dict[str, Any]

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, params: Mapping[str, Any], *, run_id: str, asof: str) -> ReconCycle | None:
        """Parse the ``reconciliation`` config block and build the cycle
        (tolerances, ledger, persisted state, state machine).

        Returns ``None`` when no ``reconciliation`` block is present (no-op default).
        Raises on a bad config (bad tolerance key, unsafe book_key) — :func:`open_cycle`
        guards it so a recon fault is never fatal to the trading run.

        ENFORCE IS TOM-GATED. ``mode="enforce"`` is honored ONLY when the book
        config also carries ``enforce_acknowledged: true`` — an explicit, auditable
        opt-in that no casual config flip satisfies. Absent the acknowledgment,
        enforce is refused and forced back to observe (``enforce_refused=True``), so
        a book can never silently start gating live capital. Enabling enforce on a
        live book remains a deliberate action (this ack + a deploy), never automatic.
        """
        cfg = params.get("reconciliation")
        if not cfg:
            return None

        book_key = str(cfg.get("book_key") or params.get("book_key") or "default")
        root = Path(str(cfg.get("data_dir", "data")))
        tol_cfg = dict(cfg.get("tolerances", {}))
        # `mode` may sit at the block top-level or inside tolerances; block wins.
        mode = str(cfg.get("mode", tol_cfg.pop("mode", "observe")))

        enforce_refused = False
        if mode == "enforce" and not bool(cfg.get("enforce_acknowledged", False)):
            enforce_refused = True
            logger.error(
                "RECON [%s]: mode='enforce' requested WITHOUT enforce_acknowledged=true — "
                "refusing to gate live capital on an unacknowledged config. Forcing OBSERVE. "
                "Enabling enforce is a deliberate, Tom-gated action.",
                book_key,
            )
            mode = "observe"

        tol = BookTolerances(mode=mode, **tol_cfg)
        # cycle_id namespaces order_refs in the ledger. A STATIC/reused cycle_id
        # conflates old and new intents/results (ledger matching is by order_ref),
        # which corrupts the failure/missed-fill signals the enforcement gate acts
        # on. Under enforce we therefore IGNORE any config-supplied cycle_id and
        # force the run-unique run_id (issue #90 review). Observe keeps the flexible
        # config override for testing/backfills.
        cfg_cycle = cfg.get("cycle_id")
        if tol.is_enforce:
            if cfg_cycle:
                logger.warning(
                    "RECON ENFORCE [%s]: ignoring config cycle_id %r — forcing run-unique run_id for live gating.",
                    book_key,
                    cfg_cycle,
                )
            if not run_id:
                raise ReconEnforcementError("enforce mode requires a run-unique run_id for cycle_id; none was provided")
            cycle_id = str(run_id)
        else:
            cycle_id = str(cfg_cycle or run_id or asof)
        ledger = OrderFillLedger(book_key=book_key, root=root)
        state_path = ledger.path.parent / "recon_state.json"

        persisted: dict[str, Any] = {}
        state_corrupt = False
        if state_path.exists():
            try:
                persisted = json.loads(state_path.read_text())
                if not isinstance(persisted, dict):
                    raise ValueError("recon state is not a JSON object")
            except Exception as exc:
                state_corrupt = True
                persisted = {}
                logger.error("Reconciliation state at %s is unreadable/corrupt: %s", state_path, exc)

        # FAIL CLOSED under enforce: the persisted state IS the enforcement
        # authority, so a corrupt/unreadable state must NOT silently reset to
        # NORMAL and let the next cycle trade freely. Force HALT so the book stops
        # until Tom clears it. In observe mode a corrupt state is only an
        # observability loss, so we fall back to a fresh NORMAL.
        if state_corrupt and tol.is_enforce:
            logger.error(
                "RECON ENFORCE [%s]: corrupt persisted state — failing CLOSED to HALT until manually cleared.",
                book_key,
            )
            persisted = {"state": "halt", "degraded_cycles": 0, "streaks": {}}

        machine = ReconciliationStateMachine(book_key=book_key, tol=tol)
        # Validate the persisted STATE VALUE (not just JSON well-formedness): a
        # readable file with a bogus state like {"state": "bogus"} is just as
        # dangerous as corrupt JSON — under enforce it must fail CLOSED to HALT,
        # never silently fall back to the default NORMAL and permit trading.
        raw_state = persisted.get("state", "normal")
        try:
            machine.state = ReconState(raw_state)
        except (ValueError, KeyError):
            if tol.is_enforce:
                logger.error(
                    "RECON ENFORCE [%s]: invalid persisted state value %r — failing CLOSED to HALT.",
                    book_key,
                    raw_state,
                )
                machine.state = ReconState.HALT
            else:
                logger.warning("Reconciliation state value %r invalid — resetting to NORMAL (observe).", raw_state)
                machine.state = ReconState.NORMAL
        try:
            machine.degraded_cycles = int(persisted.get("degraded_cycles", 0))
        except (TypeError, ValueError):
            machine.degraded_cycles = 0

        return cls(
            book_key=book_key,
            root=root,
            tol=tol,
            cycle_id=cycle_id,
            mode=mode,
            enforce_refused=enforce_refused,
            ledger=ledger,
            machine=machine,
            state_path=state_path,
            persisted=persisted,
        )

    # ------------------------------------------------------------------ gate
    def gate(self, *, alert: Alert) -> dict[str, Any]:
        """The PRE-EXECUTION gate from the PERSISTED (prior-cycle) state.

        This is the real enforcement touchpoint (issue #90): the state machine
        persisted the outcome of the PREVIOUS cycle, and we read it back HERE,
        before this cycle's orders are executed. A HALT persisted last cycle
        blocks all orders now; a FLATTEN persisted last cycle makes this cycle
        reduce-only. Because the decision is durable state consumed strictly
        before execution, enforcing it is real safety, not the post-execution
        false safety #87 rejected.

        The gate is only APPLIED in enforce mode; observe mode computes it for the
        record (``applied=False``) and changes no orders.
        """
        orders_allowed, reduce_only = preflight_gate(self.machine.state)
        enforced = self.tol.is_enforce
        applied = bool(enforced and (not orders_allowed or reduce_only))
        note = {
            "gate_from_state": self.machine.state.value,
            "orders_allowed": orders_allowed,
            "reduce_only": reduce_only,
            "enforced": enforced,
            "applied": applied,
        }
        if applied:
            action = "HALT (no new orders)" if not orders_allowed else "FLATTEN (reduce-only)"
            msg = (
                f"🛑 RECON ENFORCE [{self.book_key}] PRE-EXECUTION gate from prior state "
                f"'{self.machine.state.value}': {action}. This cycle's orders are gated BEFORE execution."
            )
            logger.error(msg)
            alert(msg, site="recon_preflight")
        elif enforced:
            logger.info(
                "RECON ENFORCE [%s] preflight: prior state '%s' permits normal trading.",
                self.book_key,
                self.machine.state.value,
            )
        return note

    # ----------------------------------------------------------------- close
    def close(
        self,
        *,
        broker: Any,
        final_weights: Mapping[str, float],
        orders_df: pd.DataFrame,
        execution_report: Mapping[str, Any],
        portfolio_value: float,
        stable_coin: str,
        intent_captured: bool,
        alert: Alert,
    ) -> dict[str, Any]:
        """Evaluate the break state machine against the ledger and the marked book.

        Classifies breaks, computes the NORMAL→DEGRADED→HALT/FLATTEN transition,
        logs/alerts, and PERSISTS the resulting state, which the NEXT cycle's
        :meth:`gate` reads back — that persist-then-read is the real enforcement
        path (issue #90).

        LEDGER SOURCE (issue #90): when the execution stage captured intent at the
        broker submission call (``intent_captured``), this READS those records
        back rather than reconstructing them, so the audit trail reflects the
        exact submission path. Otherwise it reconstructs intent + result from
        ``orders_df`` / ``execution_report`` so observe-mode shadow detection
        still works — except on a cycle the gate blocked, which submitted nothing.
        """
        book_key = self.book_key
        tol = self.tol
        cycle_id = self.cycle_id
        ledger = self.ledger
        machine = self.machine

        # Failed/zero-fill counting for the consecutive-failure break class.
        # `attempted_syms` = every symbol we submitted an intent for THIS cycle;
        # only these keep/accrue a streak, so a symbol that is simply no longer
        # traded cannot leave a stale streak that falsely escalates to HALT.
        this_cycle_failed: dict[str, int] = {}
        filled_syms: set[str] = set()
        attempted_syms: set[str] = set()
        missed_fills: list[str] = []

        # Did the execution stage already capture intent at submission for THIS
        # cycle? The caller says so explicitly — we do NOT infer it from the
        # presence of same-cycle_id intents, because a bare cycle_id can be reused
        # across cycles. When capture was active we ALWAYS read the ledger back
        # (never reconstruct), so a HALTED cycle that submitted nothing correctly
        # reads as zero attempts rather than re-recording orders never sent.
        already_captured = bool(intent_captured)

        if already_captured:
            # --- Read the submission-time-captured ledger (authoritative) ---
            matched = ledger.match_intents_to_results()
            for _ref, slot in matched.items():
                intent = slot.get("intent")
                if intent is None or str(intent.get("cycle_id")) != cycle_id:
                    continue
                sym = str(intent.get("symbol", ""))
                attempted_syms.add(sym)
                result = slot.get("result")
                status = str(result.get("status")).lower() if result else None
                if status in ("filled", "partial"):
                    filled_syms.add(sym)
                elif status in NON_TERMINAL_RESULT_STATUSES:
                    # `working`: accepted and still live on the book. `unknown`:
                    # unconfirmed but queued for resolution (TOM-1336). Neither is
                    # a fill (nothing confirmed executed) nor a missed fill
                    # (nothing is known to have gone wrong) -- so it must touch
                    # neither `filled_syms` nor the failure streak. Stage 6c
                    # resolves it against the venue next cycle and records the
                    # terminal result against this same ref.
                    continue
                elif status in ("timeout", None):
                    # Submitted but no (real) result observed = missed fill.
                    this_cycle_failed[sym] = this_cycle_failed.get(sym, 0) + 1
                    missed_fills.append(sym)
                else:  # failed / skipped / rejected / anything non-fill
                    this_cycle_failed[sym] = this_cycle_failed.get(sym, 0) + 1
        elif execution_report.get("recon_gated"):
            # --- Cycle was GATED pre-execution: submit-set is NOT orders_df -----
            # The pre-execution gate blocked orders (HALT) or filtered them
            # (FLATTEN), so the broker never received orders_df. Reconstructing
            # intents/results from orders_df here would fabricate ledger records for
            # orders that were NEVER submitted — corrupting the audit trail and
            # escalating failure/missed-fill streaks off a SAFETY GATE rather than
            # real broker behavior. Record NOTHING; leave attempted_syms empty so no
            # streak accrues. (This is the enforce fail-closed path, where the
            # execution stage had no ledger, so `already_captured` is False but
            # nothing was actually sent.)
            logger.info(
                "RECON [%s]: cycle gated pre-execution (%s) — no orders submitted; "
                "skipping ledger reconstruction (no synthetic intents/results).",
                book_key,
                execution_report.get("recon_gated"),
            )
        else:
            # --- Reconstruction fallback: record intent + result here -------
            details = execution_report.get("orders_details", [])
            executable = (
                orders_df[orders_df.get("Executable", pd.Series(dtype=bool))]
                if isinstance(orders_df, pd.DataFrame) and "Executable" in orders_df.columns
                else orders_df
            )
            # Match results to intents by (symbol, side) CONSUMING each result once,
            # so N same-(symbol, side) orders in a cycle can't all bind to the first
            # result (duplicating a fill or masking a missed one).
            details_pool: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for d in details:
                k = (str(d.get("symbol", "")), str(d.get("side", d.get("action", ""))).lower())
                details_pool[k].append(d)

            if isinstance(executable, pd.DataFrame) and not executable.empty:
                for i, (_, row) in enumerate(executable.iterrows()):
                    sym = str(row.get("Asset", ""))
                    side = str(row.get("Action", "")).lower()
                    attempted_syms.add(sym)
                    order_ref = f"{cycle_id}:{i}:{sym}:{side}"
                    ledger.record_intent(
                        cycle_id=cycle_id,
                        symbol=sym,
                        side=side,
                        order_ref=order_ref,
                        target_qty=row.get("Adjusted Quantity"),
                        target_wt=final_weights.get(sym),
                        limit_px=row.get("Price"),
                    )
                    pool = details_pool.get((sym, side))
                    match = pool.pop(0) if pool else None
                    if match is not None:
                        status = str(match.get("status", "failed")).upper()
                        if str(match.get("fill_status", "")).upper() == "UNKNOWN" and match.get("queued"):
                            # Unconfirmed but queued for next-cycle resolution: the
                            # same non-terminal rule as the captured path (TOM-1336).
                            status = "UNKNOWN"
                        ledger.record_result(
                            order_ref=order_ref,
                            cycle_id=cycle_id,
                            status=status.lower(),
                            filled_qty=match.get("executed_quantity"),
                            avg_px=match.get("executed_price"),
                        )
                        if status in ("FILLED", "PARTIAL"):
                            filled_syms.add(sym)
                        elif status.lower() in NON_TERMINAL_RESULT_STATUSES:
                            # WORKING: accepted, still live on the book. UNKNOWN
                            # (queued): unconfirmed, resolved next cycle. Neither a
                            # fill nor a failure — the same rule as the captured-
                            # intent path above. Without this the reconstruction
                            # path would count every resting limit order, and every
                            # queued unconfirmed one, as a failed order.
                            pass
                        else:  # FAILED / SKIPPED / anything non-fill
                            this_cycle_failed[sym] = this_cycle_failed.get(sym, 0) + 1
                    else:
                        # Intent written but NO result observed = missed fill.
                        ledger.record_result(order_ref=order_ref, cycle_id=cycle_id, status="timeout")
                        this_cycle_failed[sym] = this_cycle_failed.get(sym, 0) + 1
                        missed_fills.append(sym)

        # --- Reconcile against external truth (holdings), from the marked book ---
        pv = float(portfolio_value) if portfolio_value else 0.0
        actual_wt, unpriced = _held_weights(broker, pv, stable_coin)
        if unpriced:
            logger.warning(
                "RECON [%s]: drift UNKNOWN for %d unmarkable holding(s): %s",
                book_key,
                len(unpriced),
                ", ".join(unpriced),
            )

        # Drift must be FRACTIONAL (|actual - target| / |target|), because that is
        # what BookTolerances documents and what `max_drift=0.10` / `drift_halt=0.25`
        # mean. The previous absolute weight delta made both tolerances wrong in
        # opposite directions: on a book whose largest target is ~5% of equity an
        # absolute delta can never reach 0.25 (the halt was dead), while a 30%
        # holding against a zero target tripped FLATTEN instantly.
        #
        # `drift_notional_floor` keeps dust out of the signal: a $2 residual against
        # a zero target is fractionally infinite but economically irrelevant, and a
        # book that halts on dust is a book whose halt gets switched off.
        drift_floor = float(getattr(tol, "drift_notional_floor", 10.0))
        drifts: dict[str, float] = {}
        for sym in set(final_weights) | set(actual_wt):
            if sym == stable_coin or sym in unpriced:
                continue
            actual = actual_wt.get(sym, 0.0)
            target = float(final_weights.get(sym, 0.0))
            # Ignore positions too small to matter (both legs sub-floor in notional).
            if pv > 0 and max(abs(actual), abs(target)) * pv < drift_floor:
                continue
            denom = abs(target)
            if denom < 1e-9:
                # No target at all: any real holding is a 100% drift (fully
                # un-intended exposure), which is the honest fractional reading.
                drifts[sym] = 1.0 if abs(actual) > 1e-9 else 0.0
            else:
                drifts[sym] = (actual - target) / denom

        # Phantom = an exchange holding with no matching intent in the ledger over
        # a RECENT window (authority rule: internal ledger is truth for intent).
        # Scoping matters both ways: using final_weights (round-1 logic) reflags
        # every legitimate carryover/residual as phantom; using ALL history
        # (round-2 logic) means a name traded once can NEVER be phantom again,
        # masking a genuine re-appearing phantom on a rotating book. So we count a
        # holding as explained only if it was intended within the last
        # `phantom_lookback` cycles (default 2) — recent carryover is explained, a
        # stale reappearance is not. The current cycle's intents were just written.
        phantom: list[str] = []
        recs = ledger.read_all()
        cycle_order = sorted(
            {r.get("cycle_id") for r in recs if r.get("cycle_id")},
            key=lambda c: max((r.get("ts", "") for r in recs if r.get("cycle_id") == c), default=""),
        )
        recent_cycles = set(cycle_order[-max(1, tol.phantom_lookback) :])
        intended_symbols = {
            r.get("symbol") for r in recs if r.get("kind") == KIND_INTENT and r.get("cycle_id") in recent_cycles
        }
        for sym in actual_wt:
            if sym == stable_coin:
                continue
            # Same dust guard as drift: a sub-floor residual is not a phantom worth
            # flattening the book over. Every phantom is a HARD break, so without
            # this one stray $0.30 dust position sends the book to FLATTEN.
            if pv > 0 and abs(actual_wt.get(sym, 0.0)) * pv < drift_floor:
                continue
            if sym not in intended_symbols and abs(actual_wt.get(sym, 0.0)) > 1e-6:
                phantom.append(sym)

        # --- Persistent per-book state (streaks + machine state) -----------
        # `persisted` / `state_path` / `machine` (with prior state loaded) are the
        # ones the gate read, so the gate and this close agree on the same prior
        # state. Only carry a streak forward for a symbol we ATTEMPTED again this
        # cycle; a symbol we stopped trading has, by definition, no *consecutive*
        # failure to escalate on (prevents a stale streak from falsely reaching HALT).
        prior_streaks: dict[str, int] = dict(self.persisted.get("streaks", {}))
        streaks: dict[str, int] = {s: n for s, n in prior_streaks.items() if s in attempted_syms}
        for sym in filled_syms:
            streaks.pop(sym, None)  # a fill breaks the failure streak
        for sym, n in this_cycle_failed.items():
            streaks[sym] = streaks.get(sym, 0) + n

        breaks = classify_breaks(
            tol=tol,
            drifts=drifts,
            failed_streaks=streaks,
            phantom_symbols=phantom,
        )

        decision = machine.evaluate(breaks)

        state_payload = json.dumps(
            {
                "state": machine.state.value,
                "degraded_cycles": machine.degraded_cycles,
                "streaks": streaks,
            }
        )
        # Atomic write (temp + fsync + os.replace): the state file is the
        # enforcement authority, so a crash mid-persist must never leave it
        # truncated/corrupt. Retry once on failure.
        persist_err: Exception | None = None
        for _attempt in range(2):
            try:
                atomic_write_text(self.state_path, state_payload)
                persist_err = None
                break
            except Exception as exc:  # noqa: BLE001
                persist_err = exc
        if persist_err is not None:
            if tol.is_enforce:
                # ENFORCE: the next cycle's gate reads this file. If we cannot
                # persist a (possibly more severe) state, the enforcement authority
                # is unwritable — do NOT continue silently. Alert hard and RAISE so
                # the run surfaces as failed and the operator halts the book; a
                # silent continue would let the next cycle trade from stale state.
                logger.error(
                    "RECON ENFORCE [%s]: could not persist recon state (%s) — enforcement "
                    "authority unwritable. Raising to fail the run.",
                    book_key,
                    persist_err,
                )
                alert(
                    f"🛑 <b>RECON ENFORCE [{book_key}] — state persist FAILED</b>\n"
                    f"Could not write recon_state.json ({persist_err}). The next cycle's "
                    "enforcement gate cannot be trusted — HALT the book and investigate disk.",
                    site="recon_persist",
                )
                raise ReconEnforcementError(f"recon state persist failed under enforce: {persist_err}") from persist_err
            logger.warning("Reconciliation state persist failed (observe): %s", persist_err)

        # --- Alert (observe = log/alert only, no gating) -------------------
        if decision.alert:
            log_fn = logger.error if decision.to_state.value in ("halt", "flatten") else logger.warning
            log_fn("RECON [%s] %s", book_key, decision.alert.replace("\n", " | "))
            if decision.transitioned:
                alert(decision.alert, site="recon_transition")

        return {
            "book_key": book_key,
            "mode": decision.mode,
            "enforced": decision.enforced,
            # True iff a config asked for enforce WITHOUT enforce_acknowledged=true
            # and we forced it back to observe. Surfaced so a monitor can flag a
            # misconfigured book that believes it is gating when it is not.
            "enforce_refused": self.enforce_refused,
            # Whether the ledger for this cycle was captured at submission time
            # (crash-durable) vs reconstructed post-execution (fallback).
            "intent_captured_at_submission": already_captured,
            "from_state": decision.from_state.value,
            "to_state": decision.to_state.value,
            "would_be_action": decision.would_be_action.value,
            "transitioned": decision.transitioned,
            "orders_allowed": decision.orders_allowed,
            "reduce_only": decision.reduce_only,
            "n_breaks": len(breaks),
            "breaks": [
                {
                    "class": b.klass.value,
                    "symbol": b.symbol,
                    "severity": b.severity,
                    "detail": b.detail,
                    "recommended_action": b.recommended_action.value,
                }
                for b in breaks
            ],
            "missed_fills": missed_fills,
            # Held symbols whose drift is UNKNOWN (no mark) — not breaks, not zero.
            "unpriced": unpriced,
            "ledger_path": str(ledger.path),
        }


def _held_weights(broker: Any, pv: float, stable_coin: str) -> tuple[dict[str, float], list[str]]:
    """The held book as weights of ``pv``, read as a marked book; plus the held names with no mark.

    A held name with no mark has an UNKNOWN weight (TOM-1335): it is kept out of
    the weights and reported, never read as zero — reading a missing mid as 0
    made a fully-held name a -100% drift, a hard break that alone sends the book
    to FLATTEN. A failed snapshot now leaves every held name unmarked, so it
    too reads as unknown drift rather than as an empty book. A failed positions
    read is logged and reads as no holdings, as before.
    """
    actual_wt: dict[str, float] = {}
    unpriced: list[str] = []
    if not callable(getattr(broker, "get_positions", None)) or pv <= 0:
        return actual_wt, unpriced
    try:
        if not hasattr(broker, "get_market_snapshot"):
            return actual_wt, unpriced
        book = read_book(broker)
        for sym in sorted(book.positions):
            value = book.value_of(sym)
            if value is not None:
                actual_wt[sym] = value / pv
            elif sym != stable_coin:
                unpriced.append(sym)
    except Exception as exc:  # never let recon crash the run
        logger.warning("Reconciliation position read failed: %s", exc)
    return actual_wt, sorted(unpriced)


def filter_reduce_only(executable: pd.DataFrame, broker: Any) -> pd.DataFrame:
    """Keep only orders that REDUCE existing exposure, clamped so they never
    flip a position past zero. The FLATTEN gate's filter.

    An order reduces |position| iff its signed delta opposes the current
    signed position. Opening/increasing legs are dropped; a reducing leg is
    clamped to at most the current position magnitude. Fails SAFE: if we cannot
    read positions, we drop everything (send nothing) rather than risk opening.
    """
    positions: dict[str, float] = {}
    get_positions = getattr(broker, "get_positions", None)
    if callable(get_positions):
        try:
            pos = get_positions()
            if pos is not None and len(pos) > 0:
                for _, r in pos.iterrows():
                    positions[str(r.get("symbol", ""))] = float(r.get("qty", 0) or 0)
        except Exception as exc:  # fail safe: no positions → reduce nothing
            logger.warning("Reduce-only gate could not read positions (dropping all orders): %s", exc)
            return executable.iloc[0:0]
    else:
        logger.warning("Reduce-only gate: broker exposes no get_positions — dropping all orders (fail safe).")
        return executable.iloc[0:0]

    kept: list[Any] = []
    for _, row in executable.iterrows():
        sym = str(row.get("Asset", ""))
        action = str(row.get("Action", "")).strip().lower()
        qty = float(row.get("Adjusted Quantity", 0) or 0)
        cur = positions.get(sym, 0.0)
        if qty <= 0 or cur == 0:
            continue
        # FAIL CLOSED: only a recognised buy/sell can be judged reduce-or-not.
        # An unknown action under a FLATTEN gate must be DROPPED, never assumed
        # to be exposure-reducing (assuming sell could send an opening order).
        if action not in ("buy", "sell"):
            logger.warning(
                "Reduce-only gate: dropping order with unrecognised action %r for %s (fail closed).",
                action,
                sym,
            )
            continue
        delta = qty if action == "buy" else -qty
        # Reduces only if the order's signed delta opposes the current position.
        if (delta > 0) == (cur > 0):
            continue
        new_qty = min(qty, abs(cur))  # clamp: never cross zero
        if new_qty <= 0:
            continue
        r = row.copy()
        r["Adjusted Quantity"] = new_qty
        # Exposure-reducing under a FLATTEN gate, i.e. an emergency exit. Mark
        # reduce_only so the broker sends reduceOnly: the venue exempts a
        # sub-min close from its min-notional floor and it can never flip
        # through zero. Without this the gate's own exits could be rejected as
        # ordinary sub-min orders (quantbox#87 / #138 review).
        r["reduce_only"] = True
        kept.append(r)
    return pd.DataFrame(kept) if kept else executable.iloc[0:0]


# The gate recorded when the gate itself could not be computed under acknowledged
# enforce: FAIL CLOSED, no order this cycle.
_FAIL_CLOSED_GATE: dict[str, Any] = {
    "applied": True,
    "orders_allowed": False,
    "reduce_only": False,
    "enforced": True,
    "gate_from_state": "unknown",
    "fail_closed": True,
}


@dataclass
class CycleRun:
    """One run's reconciliation cycle, as the pipeline holds it (:func:`open_cycle`).

    ``loaded`` is the cycle, or ``None`` (no ``reconciliation`` block, or the load
    failed: ``load_error``). ``armed`` is True when the load AND the gate
    succeeded: only then does the execution stage write the ledger at submission.
    ``preflight`` is the gate's record (``{}`` with no gate).
    """

    params: Mapping[str, Any]
    loaded: ReconCycle | None = None
    armed: bool = False
    preflight: dict[str, Any] = field(default_factory=dict)
    load_error: Exception | None = None

    @property
    def gate_applied(self) -> bool:
        return bool(self.preflight.get("applied"))

    @property
    def orders_allowed(self) -> bool:
        """The gate's verdict for this cycle's orders (True unless an enforce gate applies)."""
        return bool(self.preflight.get("orders_allowed", True)) if self.gate_applied else True

    @property
    def reduce_only(self) -> bool:
        return bool(self.preflight.get("reduce_only", False)) if self.gate_applied else False

    @property
    def ledger(self) -> OrderFillLedger | None:
        """The ledger the execution stage records intent and result to, or ``None``."""
        return self.loaded.ledger if self.armed and self.loaded is not None else None

    @property
    def cycle_id(self) -> str | None:
        return self.loaded.cycle_id if self.armed and self.loaded is not None else None

    @property
    def capture_fail_closed(self) -> bool:
        """Under enforce a failed intent write drops the order: never trade live without the record."""
        return bool(self.armed and self.loaded is not None and self.loaded.tol.is_enforce)

    def close(
        self,
        *,
        broker: Any,
        final_weights: Mapping[str, float],
        orders_df: pd.DataFrame,
        execution_report: Mapping[str, Any],
        portfolio_value: float,
        stable_coin: str,
        alert: Alert,
    ) -> dict[str, Any]:
        """Close the cycle (:meth:`ReconCycle.close`), guarded: it NEVER fails the run.

        This runs AFTER orders executed. A config typo, a bad tolerance key, a
        permission/disk-full error on the ledger write, or any other fault in
        the reconciliation layer must not abort the run and lose its artifacts:
        it is logged and returned as ``{"error": ...}``. The one exception is
        :class:`ReconEnforcementError` (the enforcement authority is unwritable
        or untrustworthy under enforce), which escapes and fails the run.

        Returns ``{}`` when the book has no ``reconciliation`` block. This
        cycle's gate is attached as ``preflight_gate`` / ``gate_applied`` (and
        ``recon_gated`` when the execution stage gated orders).
        """
        notes: dict[str, Any]
        if self.loaded is None:
            if isinstance(self.load_error, ReconEnforcementError):
                raise self.load_error
            notes = {} if self.load_error is None else {"error": str(self.load_error)}
            if self.load_error is not None:
                logger.error("Reconciliation stage failed (non-fatal to the run): %s", self.load_error)
        else:
            try:
                notes = self.loaded.close(
                    broker=broker,
                    final_weights=final_weights,
                    orders_df=orders_df,
                    execution_report=execution_report,
                    portfolio_value=portfolio_value,
                    stable_coin=stable_coin,
                    intent_captured=self.armed,
                    alert=alert,
                )
            except ReconEnforcementError:
                # CRITICAL under enforce: must NOT be swallowed. Let it escape and
                # fail the run so the book stops, rather than completing
                # "successfully" while the next cycle trades from stale/absent
                # enforcement state. close() already alerted hard.
                raise
            except Exception as exc:  # noqa: BLE001 - ordinary recon faults are never fatal
                logger.error("Reconciliation stage failed (non-fatal to the run): %s", exc)
                notes = {"error": str(exc)}
        if self.preflight:
            notes["preflight_gate"] = self.preflight
            notes["gate_applied"] = self.gate_applied
            if execution_report.get("recon_gated"):
                notes["recon_gated"] = execution_report["recon_gated"]
        return notes


def open_cycle(params: Mapping[str, Any], *, run_id: str, asof: str, alert: Alert) -> CycleRun:
    """Load this run's cycle and compute its PRE-EXECUTION gate (Stage 7a).

    Guarded so a recon fault is never fatal: under acknowledged enforce an
    inability to load or compute the persisted gate FAILS CLOSED (HALT, no order
    this cycle); in observe it fails open to no gate.
    """
    run = CycleRun(params=params)
    try:
        run.loaded = ReconCycle.load(params, run_id=run_id, asof=asof)
        if run.loaded is not None:
            run.preflight = run.loaded.gate(alert=alert)
            run.armed = True
    except Exception as exc:  # noqa: BLE001 - recon must never be fatal
        run.armed = False
        if run.loaded is None:
            run.load_error = exc
        if enforce_requested(params):
            # FAIL CLOSED: under acknowledged enforce, an inability to load or
            # compute the persisted gate must NOT let orders through — block
            # this cycle. (Observe mode fails open to no-gate: no gating claimed.)
            logger.error(
                "RECON ENFORCE: preflight failed under acknowledged enforce — failing CLOSED (HALT) "
                "this cycle; no orders will be sent. Error: %s",
                exc,
            )
            run.preflight = dict(_FAIL_CLOSED_GATE)
        else:
            logger.error("Recon preflight failed (observe, non-fatal, fail-safe to no-gate): %s", exc)
            run.preflight = {}
    return run
