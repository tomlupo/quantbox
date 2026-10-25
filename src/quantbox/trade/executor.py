"""Order execution: gate → submit → classify → working orders → report (TOM-1458).

The trading pipeline sizes nothing and executes nothing itself: a rebalancing
plugin turns target weights into orders, and :func:`execute_orders` sends them.
It used to be ``TradingPipeline._execute_orders``, ~850 lines inside a 3,000-line
orchestrator; the stages, in order:

1. **gate** — ``trading_enabled``, the reconciliation HALT and FLATTEN gates
   (enforce mode, issue #90), and the dead-man when no order is executable: a
   residual day (TOM-1778, TOM-2212), a quiet day, or a FROZEN book. The book
   freezes only when a suppressed row is not proven to trap nothing.
2. **submit** — sell before buy, in a stable sort that keeps a flip's close
   leg before its open leg (TOM-2396); the order INTENT is written to the
   ledger before ``broker.place_orders`` (crash-durable, issue #90).
3. **classify** — every broker row passes :func:`normalise_results` (the one
   normaliser at the broker seam: an absent status or filled quantity is
   UNKNOWN, never a fill), then books as FILLED / PARTIAL / WORKING / UNKNOWN /
   FAILED / SKIPPED, bound one-to-one to its intent. A (symbol, side) whose
   result rows do not number its submitted orders binds nothing: each of its
   orders books FAILED with an UNKNOWN fill (:func:`unbound_keys`, TOM-2396).
4. **working orders** — an order still resting at the venue, or an unconfirmed
   one with an id, is queued so the next cycle books its real outcome.
5. **report** — the execution report the pipeline stores, and an alert for
   every order failure.

Every alert goes through the *alert* callable the caller passes (the pipeline's
one alert channel, :mod:`quantbox.trade.notifier`); it must never raise.

**Intent is declared, never inferred from side** (TOM-901). The trapped-residual
freeze (#81) counts an order the broker skipped only when the order EXITS held
exposure. A rebalancer declares that on each order row: ``opens_exposure`` is
True for an open from flat or a same-side add
(:func:`quantbox.plugins.rebalancing.futures_rebalancer.declares_open`). A
declined OPEN is therefore never read as a stuck close. A row that declares
nothing (a third-party rebalancer) keeps the long-only spot reading, where a
sell is the exit.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

import pandas as pd

from quantbox.contracts import BrokerPlugin
from quantbox.plugins.broker._fills import RAW_STATUS, normalise_results
from quantbox.plugins.rebalancing.futures_rebalancer import RESIDUAL_STATUS
from quantbox.trade.notifier import escape_html
from quantbox.trade.reconciliation.cycle import filter_reduce_only
from quantbox.trade.reconciliation.ledger import EXEC_STATUS_TO_LEDGER

logger = logging.getLogger(__name__)

#: ``alert(message, *, site) -> bool``: sends one alert, never raises.
Alert = Callable[..., bool]

#: Order statuses that mean "the strategy WANTED to trade but a guard suppressed
#: the order" (vs. a legitimately quiet day of zero-deltas).
SUPPRESSED_STATUSES = frozenset(
    {
        "Below min notional",
        "Below threshold",
        "Below min qty",
        "Zero quantity",
        "Zero price",
        "Invalid (NaN)",
    }
)


def _declares_close(row: Any, side: str) -> bool:
    """The order row's declared intent: True when the order exits held exposure (TOM-901).

    ``opens_exposure`` declared (True or False) decides. A row that declares
    nothing keeps the long-only spot reading: a sell exits, a buy opens.
    """
    declared = row.get("opens_exposure") if hasattr(row, "get") else None
    if declared is not None and not pd.isna(declared):
        return not bool(declared)
    return side == "sell"


def _safe_float(v: Any) -> float | None:
    """Best-effort float coercion for ledger records; None on failure/None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_fee(value: Any) -> float | None:
    """A fee that was not reported is UNKNOWN, never 0.0 (#92).

    The `float(x or 0)` idiom this replaces mapped both "no fee key" and
    "fee was genuinely zero" onto 0.0. Only the second is data. Conflating them
    is what produced 165 days of `Fees (cumulative) $0.00` on a book with ~24x
    equity turnover.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN is not a measurement


def suppressed_order_records(suppressed: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-order detail for a freeze, as structured records.

    A freeze alert used to carry only a status HISTOGRAM ("Below min notional=2,
    Below threshold=1"). That says how many orders died and nothing about which,
    how big, or how far under the bar they were — so the first question a human
    asks on being paged ("what could not trade, and by how much?") required
    opening the run log every time. Everything needed is already on the row; it
    just never left the pipeline.

    ``shortfall_usd`` is what makes it actionable: it is the gap between the
    order and the venue minimum, i.e. exactly how much the book would have to
    grow for this leg to become executable.
    """
    records: list[dict[str, Any]] = []
    for _, row in suppressed.iterrows():
        notional = float(row.get("Notional Value", 0) or 0)
        min_notional = float(row.get("Min Notional", 0) or 0)
        record = {
            "symbol": str(row.get("Asset", "") or row.get("Symbol", "")),
            "action": str(row.get("Action", "")).upper(),
            "notional_usd": notional,
            "min_notional_usd": min_notional,
            "status": str(row.get("Order Status", "")),
            "reason": str(row.get("Reason", "")),
        }
        if min_notional > 0 and 0 < notional < min_notional:
            record["shortfall_usd"] = min_notional - notional
        records.append(record)
    return records


def format_suppressed_line(rec: dict[str, Any]) -> str:
    """One human-readable line per suppressed order, for chat alerts."""
    line = f"{rec['action']} {rec['symbol']} ${rec['notional_usd']:,.2f}"
    shortfall = rec.get("shortfall_usd")
    if shortfall:
        line += f" (min ${rec['min_notional_usd']:,.2f}, short ${shortfall:,.2f})"
    detail = rec.get("reason") or rec.get("status")
    if detail:
        line += f" — {detail}"
    return line


def empty_report() -> dict[str, Any]:
    """An execution report with nothing sent: the shape every exit of :func:`execute_orders` starts from."""
    return {
        "executed_orders": [],
        "failed_orders": [],
        "summary": {
            "total_executed": 0,
            "total_partial": 0,
            "total_failed": 0,
            # Accepted by the venue, still resting on the book when the run
            # ended. Neither executed nor failed — resolved next cycle.
            "total_working": 0,
            "total_value": 0.0,
            "total_cost": 0.0,
        },
        "orders_details": [],
        "api_errors": [],  # broker API errors caught during execution (#62)
    }


def report_order_failures(report: dict[str, Any], alert: Alert) -> None:
    """Record + alert this cycle's order failures. Idempotent; no-op at zero.

    The two freeze guards are both CONJUNCTIONS requiring a total standstill:
    one needs ``total_executed == 0`` AND a skipped close-out, the other
    needs the broker to return an EMPTY frame. Neither fires on the commonest
    real failure — "some orders filled, some were rejected" — which is how
    carver-HL emitted the same rejected ETH/SOL close-outs for 33 consecutive
    days while every run was recorded as healthy (quantbox#87).

    Called from EVERY exit of :func:`execute_orders` that can leave failures on
    the report: the normal path, the ``place_orders`` exception path and the
    all-dropped-under-enforce path. The exception path is the worst case — the
    whole batch failed at submission and nothing traded — so it must alert too,
    rather than returning early past the report.
    """
    n_failed = int(report["summary"]["total_failed"])
    if not n_failed:
        return
    failed_list = ", ".join(
        f"{str(d.get('action', '')).upper()} {d.get('symbol', '')} ({d.get('error', 'unknown')})"
        for d in report["orders_details"]
        if d.get("status") == "FAILED"
    )
    report["order_failures"] = n_failed
    report["order_failure_detail"] = failed_list
    logger.error(
        "ORDER FAILURES: %d of %d submitted order(s) failed — %s",
        n_failed,
        n_failed + int(report["summary"]["total_executed"]),
        failed_list,
    )
    alert(
        f"❌ <b>{n_failed} ORDER(S) FAILED</b>\n{escape_html(failed_list)}\n"
        "Targets NOT reached for these symbols; exposure is unchanged.",
        site="order_failures",
    )


def _untrapped(suppressed: pd.DataFrame) -> pd.Series:
    """Per suppressed row: True when its DECLARED intent proves the row traps nothing (TOM-1778, TOM-2212).

    Two kinds, both declared by the rebalancer in ``opens_exposure``:

    * a sub-min ENTRY — an open from flat or a same-side add (``opens_exposure``
      True) below the venue minimum or inside the churn band. Not trading it
      leaves the book where it is; nothing it holds is stuck.
    * a CHURN-BAND TRIM — a row the band held back (``Below threshold``) whose
      intent is declared and which declares it is NOT a full close
      (``full_close`` False). The band is the book's own churn policy, not a
      refusal, and once the drift grows past it the trade goes out. A band row
      that is, or may be, a full close is not cleared: the futures rebalancer
      exempts every full close from the band, but the standard rebalancer
      exempts only a long one, so its short close can sit in the band.

    Everything else is NOT proven untrapped: a declared exit any floor or the
    venue blocked (``Below min notional``, ``Below min qty``, ``Zero quantity``),
    a data fault (``Zero price``, ``Invalid (NaN)``) on any row, and any row that
    declares no intent (a third-party rebalancer). A full close the venue can
    never accept is not in this frame at all: the rebalancer reports it as a
    residual (``RESIDUAL_STATUS``), and :func:`_gate_unexecutable` lists it.
    """
    if "opens_exposure" not in suppressed.columns:
        return pd.Series(False, index=suppressed.index)
    declared = suppressed["opens_exposure"].notna()
    status = suppressed["Order Status"]
    entry = status.isin({"Below min notional", "Below threshold", "Below min qty"}) & suppressed["opens_exposure"].eq(
        True
    )
    not_a_close = (
        suppressed["full_close"].eq(False)
        if "full_close" in suppressed.columns
        else pd.Series(False, index=suppressed.index)
    )
    band = (status == "Below threshold") & declared & not_a_close
    return entry | band


def _gate_unexecutable(
    report: dict[str, Any], orders_df: pd.DataFrame, residuals: pd.DataFrame, broker: BrokerPlugin, alert: Alert
) -> dict[str, Any]:
    """No order is executable: a residual day, a quiet day, or the dead-man FREEZE.

    The book freezes only when a suppressed row may hold something the venue
    could accept: see :func:`_untrapped` for the rows that cannot. The order of
    the checks below is part of the contract.
    """
    reasons: dict[str, Any] = {}
    suppressed = pd.DataFrame()
    if "Order Status" in orders_df.columns:
        reasons = orders_df["Order Status"].value_counts().to_dict()
        suppressed = orders_df[orders_df["Order Status"].isin(SUPPRESSED_STATUSES)]
    untrapped = _untrapped(suppressed) if not suppressed.empty else pd.Series(dtype=bool)
    nothing_trapped = bool(untrapped.all())

    # RESIDUAL DAY (TOM-1778; Tom, 2026-10-12: "Residual is not a freeze";
    # extended Tom, 2026-10-23, TOM-2212). The book's only untraded EXITS are
    # residuals — partial reduces below the venue minimum, and full closes the
    # venue can never accept (dust) — and everything else suppressed is a
    # sub-min entry or a churn-band trim (see _untrapped). Nothing is trapped:
    # the venue cannot take those exits at this size. Drift stays recon's job.
    # Anything else suppressed still goes to the dead-man below.
    if not residuals.empty and nothing_trapped:
        report["residual_day"] = True
        report["residual_reasons"] = reasons
        logger.info(
            "RESIDUAL DAY: no order to send — %d residual exit(s) the venue rejects at this size "
            "and %d sub-min entry or churn-band leg(s). Not a freeze.",
            len(residuals),
            len(suppressed),
        )
        return report

    if len(suppressed) == 0:
        logger.warning("All %d orders filtered out: %s", len(orders_df), reasons)
        return report

    # A QUIET DAY on a held book (TOM-2212): every suppressed row declares an
    # intent that traps nothing — holdings inside the churn band, sub-min
    # entries — and there is no residual. No position read is needed: the
    # declared intents already say what the position read below only guesses
    # (issue #82), so its fail-safe stays for rows that declare nothing.
    if nothing_trapped:
        report["quiet_day"] = True
        report["quiet_reasons"] = reasons
        logger.info(
            "QUIET DAY: no order to send — %d sub-min entry or churn-band leg(s), nothing trapped. Not a freeze.",
            len(suppressed),
        )
        return report

    # From here on, only the rows not proven untrapped are judged, and only
    # they are named on a freeze page: a band trim or a declared entry beside
    # them never makes the book trapped by itself.
    suppressed = suppressed[~untrapped]

    # Distinguish a TRAPPED book (real freeze) from a QUIET DAY
    # (all-cash, nothing to do). The dead-man freeze exists to catch
    # a book that SHOULD be rebalanced/exited but CANNOT — i.e. a
    # held position whose exit/reduce order is sub-min and therefore
    # suppressed. That manifests as a suppressed SELL leg, and/or the
    # broker still reporting liquidatable positions (post stablecoin-
    # dust exclusion, see kraken.get_positions).
    #
    # By contrast, on a weak-signal day an all-cash book produces a
    # ~few-percent gross target where every ENTRY (buy) leg is below
    # min_notional. Nothing is stuck — there are no positions to exit
    # and no suppressed sells. Staying in cash is the correct, clean
    # outcome, not a frozen rebalancer. Classify it as a QUIET DAY at
    # INFO level so flat-trend days don't fire the loud freeze alert.
    actions = suppressed.get("Action", pd.Series(dtype=object)).astype(str).str.lower()
    has_suppressed_sell = bool((actions == "sell").any())

    has_liquidatable_positions = False
    cannot_confirm_flat = False
    try:
        # `get_positions` is a REQUIRED BrokerPlugin method: called, never
        # probed (TOM-1459). A broker without it raises here and takes the
        # same fail-safe branch as a failed read.
        pos = broker.get_positions()
        has_liquidatable_positions = pos is not None and len(pos) > 0
    except Exception as exc:  # never let a position read crash the run
        # Fail SAFE (issue #82): if we cannot confirm the book is flat, treat
        # it as potentially trapped (do not downgrade to quiet). On a
        # long-only spot book a suppressed BUY is always a (harmless) entry,
        # but on a futures book a BUY can be a short-close EXIT that the
        # has_suppressed_sell heuristic does NOT catch.
        logger.warning(
            "Position read for quiet/freeze classification failed on %s: %s — failing safe to trapped.",
            type(broker).__name__,
            exc,
        )
        cannot_confirm_flat = True

    trapped = has_suppressed_sell or has_liquidatable_positions or cannot_confirm_flat

    if not trapped:
        # QUIET DAY: all-cash, every target leg sub-min. Stay in cash.
        report["quiet_day"] = True
        report["quiet_reasons"] = reasons
        order_list = ", ".join(
            f"{str(r.get('Action', '')).upper()} {r.get('Asset', '')} "
            f"(${float(r.get('Notional Value', 0) or 0):.2f}, {r.get('Order Status', '')})"
            for _, r in suppressed.iterrows()
        )
        logger.info(
            "QUIET DAY: all-cash, %d sub-min target leg(s) — staying in cash. "
            "No liquidatable positions and no suppressed exits, so the book is "
            "NOT frozen. Suppressed targets: %s",
            len(suppressed),
            order_list,
        )
        return report

    # DEAD-MAN: a real rebalance was intended but EVERY order was
    # filtered. The book is NOT tracking its targets — it is frozen
    # on stale positions. Do not exit quietly: flag the run and alert.
    report["frozen"] = True
    report["freeze_reasons"] = reasons
    # Structured per-order detail so the notifier can explain WHICH
    # orders died and by how much, without re-reading the run log.
    report["frozen_orders"] = suppressed_order_records(suppressed)
    order_lines = [format_suppressed_line(r) for r in report["frozen_orders"]]
    order_list = "; ".join(order_lines)
    logger.error(
        "REBALANCER FROZEN: no order executable, %d suppressed order(s) not proven untrapped %s. "
        "Portfolio NOT rebalanced — holding stale positions. Orders: %s",
        len(suppressed),
        reasons,
        order_list,
    )
    alert(
        f"🧊 <b>REBALANCER FROZEN — 0 of {len(orders_df)} orders executable</b>\n"
        f"Reasons: {escape_html(reasons)}\n"
        "Suppressed:\n" + "\n".join(f"• {escape_html(line)}" for line in order_lines) + "\n"
        "Portfolio NOT rebalanced; holding stale positions. "
        "Likely min_notional too high for account size (or stale/NaN data).",
        site="frozen",
    )
    return report


def _result_key(row: Any) -> tuple[str, str]:
    return (str(row.get("symbol", "")).strip(), str(row.get("side", "")).strip().lower())


def unbound_keys(submitted: pd.DataFrame, results: pd.DataFrame) -> set[tuple[str, str]]:
    """The (symbol, side) keys whose broker results cannot be bound to their orders (TOM-2396).

    The executor binds the k-th result row of a key to the k-th order of that
    key it submitted. That is exact only when the broker returns ONE row per
    submitted order of the key. Brokers carry no client order id that echoes
    the submitted row, so the count is the check. A key with more or fewer
    result rows than orders is unbound: a missing rejection row would hand the
    next same-key order's fill to the rejected one. A flip's close and open
    legs share a key, which is how PR #310's review found it.
    """
    want: dict[tuple[str, str], int] = defaultdict(int)
    got: dict[tuple[str, str], int] = defaultdict(int)
    for _, row in submitted.iterrows():
        want[_result_key(row)] += 1
    for _, row in results.iterrows():
        got[_result_key(row)] += 1
    return {key for key in set(want) | set(got) if want[key] != got[key]}


def _fail_closed_unbound(
    report: dict[str, Any],
    submitted: pd.DataFrame,
    results: pd.DataFrame,
    unbound: set[tuple[str, str]],
    intent_refs: dict[tuple[str, str], list[str | None]],
    declared_closes: dict[tuple[str, str], deque[bool]],
    ledger: Any,
    cycle_id: str | None,
) -> pd.DataFrame:
    """Book every order of an unbound key as FAILED with an UNKNOWN fill; return the results left to bind.

    No fill of an unbound key is booked to any order, because none can be told
    apart. Each submitted order of the key gets its own FAILED row (fill
    UNKNOWN, no quantity) and its ledger intent closes ``failed``. The broker's
    own rows, order ids included, go on the report under ``unbound_results``
    for the operator to reconcile against the venue. The failure alert fires
    from :func:`report_order_failures` at the end of the run.
    """
    keep = [_result_key(r) not in unbound for _, r in results.iterrows()]
    dropped = results[[not k for k in keep]]
    report["unbound_results"] = [
        {
            "symbol": str(r.get("symbol", "")),
            "side": str(r.get("side", "")),
            "status": str(r.get("status", "")),
            "qty": _safe_float(r.get("qty")),
            "order_id": None if r.get("order_id") is None or pd.isna(r.get("order_id")) else str(r.get("order_id")),
        }
        for _, r in dropped.iterrows()
    ]
    for key in sorted(unbound):
        n_sent = int(sum(_result_key(r) == key for _, r in submitted.iterrows()))
        n_back = int(sum(_result_key(r) == key for _, r in dropped.iterrows()))
        error = (
            f"broker returned {n_back} result row(s) for {n_sent} order(s) on {key[0]} {key[1]}: "
            "no result can be bound to its order, no fill booked; reconcile against the venue"
        )
        logger.error("RESULTS UNBOUND: %s", error)
        for _, row in submitted.iterrows():
            if _result_key(row) != key:
                continue
            report["orders_details"].append(
                {
                    "symbol": key[0],
                    "action": key[1],
                    "quantity": 0.0,
                    "status": "FAILED",
                    "fill_status": "UNKNOWN",
                    "error": error,
                }
            )
            report["summary"]["total_failed"] += 1
        for ref in intent_refs.pop(key, []):
            if ref is None or ledger is None or cycle_id is None:
                continue
            try:
                ledger.record_result(order_ref=ref, cycle_id=cycle_id, status="failed")
            except Exception as exc:  # never let the ledger crash the run
                logger.warning("Result capture failed for %s %s: %s", key[1], key[0], exc)
        declared_closes.pop(key, None)
    return results[keep]


def execute_orders(
    *,
    broker: BrokerPlugin,
    orders_df: pd.DataFrame,
    trading_enabled: bool,
    alert: Alert,
    ledger: Any = None,
    cycle_id: str | None = None,
    gate_orders_allowed: bool = True,
    gate_reduce_only: bool = False,
    capture_fail_closed: bool = False,
    working_store: Any = None,
) -> dict[str, Any]:
    """Execute a rebalancer's orders via *broker*, sells before buys, and report what happened.

    Ported from quantlab ``orders.py:execute_orders()``; it was
    ``TradingPipeline._execute_orders`` until TOM-1458.

    Reconciliation (issue #90):

    * If ``ledger`` is supplied, the order INTENT is captured at the broker
      submission call — immediately before ``broker.place_orders`` — and each
      order's RESULT is recorded as fills are processed. This makes the ledger
      a crash-durable audit trail: if the run dies between submission and the
      reconciliation close, the intent is already on disk.
    * ``gate_orders_allowed`` / ``gate_reduce_only`` are the PRE-EXECUTION gate
      the caller derived from the persisted prior-cycle recon state (enforce
      mode only). ``gate_orders_allowed=False`` halts all orders this cycle;
      ``gate_reduce_only=True`` keeps only exposure-reducing orders. In observe
      mode the caller passes the permissive defaults, so behavior is unchanged.
    * ``working_store`` (optional): queue for orders the venue ACCEPTED that
      were still resting on the book when the confirmation window closed. They
      are recorded here so the next cycle can resolve them against the venue
      and book the real fill; without a store they are still reported, but
      their eventual fill is not recovered.
    * ``capture_fail_closed`` (enforce mode): if intent capture fails for an
      order, that order is DROPPED (not submitted) — under enforcement we must
      never trade live without the crash-durable intent record. In observe mode
      (default) a capture failure is only an observability loss and the order
      still sends.

    *alert* is ``alert(message, *, site) -> bool`` and must never raise.
    """
    # Orders left working at the venue this cycle. Defined before any early
    # return so the persist step at the end is always safe to run.
    working_to_queue: list[dict[str, Any]] = []
    report = empty_report()

    if orders_df is None or orders_df.empty:
        logger.info("No orders generated")
        return report

    if not trading_enabled:
        logger.info("trading_enabled=False, skipping execution")
        return report

    # --- Pre-execution HALT gate (enforce mode, issue #90) -------------
    # Derived by the caller from the persisted prior-cycle recon state. A HALT
    # means the order path / books are not trustworthy: send nothing this cycle.
    if not gate_orders_allowed:
        report["recon_gated"] = "halt"
        logger.error(
            "RECON ENFORCE: %d order(s) HALTED pre-execution by the prior-cycle gate — nothing sent this cycle.",
            len(orders_df),
        )
        alert(
            f"🛑 <b>RECON ENFORCE — HALT</b>\n{len(orders_df)} order(s) NOT sent; "
            "book is in HALT from the prior cycle. Manual clear required.",
            site="recon_halt",
        )
        return report

    executable = (
        orders_df[orders_df.get("Executable", pd.Series(dtype=bool))].copy()
        if "Executable" in orders_df.columns
        else orders_df
    )

    # Exits the venue rejects at this size: partial reduces below the venue
    # minimum (TOM-1778) and full closes the venue can never accept (TOM-2212).
    # Never sent. A known residual — listed, never a freeze.
    residuals = (
        orders_df[orders_df["Order Status"] == RESIDUAL_STATUS]
        if "Order Status" in orders_df.columns
        else pd.DataFrame()
    )
    if not residuals.empty:
        report["residuals"] = suppressed_order_records(residuals)
        logger.info(
            "RESIDUAL: %d exit(s) the venue rejects at this size not sent, residual retained: %s",
            len(residuals),
            "; ".join(format_suppressed_line(r) for r in report["residuals"]),
        )

    # --- Pre-execution FLATTEN gate (enforce mode, issue #90) ----------
    # Keep only exposure-reducing orders (clamped so they never cross zero).
    if gate_reduce_only and isinstance(executable, pd.DataFrame) and not executable.empty:
        before_n = len(executable)
        executable = filter_reduce_only(executable, broker)
        report["recon_gated"] = "flatten"
        logger.warning(
            "RECON ENFORCE: reduce-only gate — %d of %d order(s) retained (exposure-reducing only).",
            len(executable),
            before_n,
        )
        if executable.empty:
            # All orders were opening/increasing — nothing to reduce. This is a
            # clean gated no-op, NOT a freeze.
            report["recon_reduce_only_noop"] = True
            logger.warning("RECON ENFORCE reduce-only: no exposure-reducing orders to send.")
            return report

    if executable.empty:
        return _gate_unexecutable(report, orders_df, residuals, broker, alert)

    # Sell before buy: sort by Action descending (Sell > Buy). The sort is
    # STABLE, so rows of one side keep the rebalancer's order. A flip is a
    # close leg then an open leg of the SAME side (TOM-2396), and the close
    # must reach the venue first: every broker sends rows one by one, in order.
    executable = executable.sort_values(by="Action", ascending=False, kind="stable")

    # Convert to broker-compatible format
    broker_orders_data: list[dict[str, Any]] = []
    # Submission INTENT, keyed by (symbol, side). Brokers hand back skipped and
    # failed rows carrying only symbol/side/qty/price/status/error — the
    # notional, the venue minimum and the order's declared intent are dropped
    # at that boundary. Without this map the broker-side freeze alert reports
    # "SELL ADA $0.00" with no shortfall, in precisely the trapped-residual case
    # the alert exists to explain (caught in review of #143).
    #
    # Deliberately a SIDE MAP rather than extra columns on broker_orders:
    # that frame is handed to third-party adapters, and widening it risks a
    # broker treating an unexpected column as an order field.
    #
    # A per-key FIFO, not a single dict entry: two orders can share
    # (symbol, side) — the intent/result matching below already carries a FIFO
    # for exactly that shape. With last-wins, both frozen rows would report the
    # LAST order's notional, so a $12 trapped residual reads as $1.20 (caught in
    # review of #144).
    order_intent: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for _, row in executable.iterrows():
        action = str(row.get("Action", "")).lower()
        side = "sell" if action == "sell" else "buy"
        symbol = str(row.get("Asset", ""))
        # NaN-safe: a mixed-column DataFrame fills a missing reduce_only with
        # NaN, and bool(NaN) is True — which would send OPENS reduce-only.
        _ro = row.get("reduce_only", False)
        reduce_only = bool(_ro) if pd.notna(_ro) else False
        broker_orders_data.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": float(row.get("Adjusted Quantity", 0)),
                "price": float(row.get("Price", 0)),
                # Thread the rebalancer's close flag through so the broker sends
                # a flat-target exit reduce-only (venue exempts it from the $10
                # minimum; can't flip through zero). Defaults False for opens.
                "reduce_only": reduce_only,
            }
        )
        order_intent[(symbol, side)].append(
            {
                "notional": float(row.get("Notional Value", 0) or 0),
                "min_notional": float(row.get("Min Notional", 0) or 0),
                # The order's DECLARED intent (TOM-901): does it exit held exposure?
                "closes": _declares_close(row, side),
            }
        )

    broker_orders = pd.DataFrame(broker_orders_data)
    if broker_orders.empty:
        return report

    # --- Submission-time INTENT capture (issue #90) --------------------
    # Record every order's intent to the append-only ledger at the broker
    # submission call, BEFORE place_orders. This is the crash-durable audit
    # trail: an intent on disk before submission proves what the system meant
    # to do even if the run dies mid-submission. `intent_refs` is a per-(symbol,
    # side) FIFO of order_refs so results bind one-to-one to intents below.
    #
    # Per-(symbol, side) FIFO of order_refs. A None entry is a SENTINEL for an
    # order that executed but whose intent write failed (observe) — it reserves
    # the positional slot so results stay 1:1 without recording against a
    # never-written intent.
    intent_refs: dict[tuple[str, str], list[str | None]] = defaultdict(list)
    if ledger is not None and cycle_id is not None:
        dropped_idx: list[int] = []
        # Intents of the orders that SURVIVE to submission, rebuilt as we go.
        #
        # An order dropped under enforce still gets an `orders_details` row
        # (status FAILED), so the freeze builder processes it — but it never
        # consumed its own intent, which left the FIFO one ahead and handed
        # every later order the WRONG notional. Reproduced in review of #145:
        # a submitted-then-skipped ADA SELL of $12.00 was reported as $1.20,
        # the notional of a different, dropped order. A residual figure that
        # is confidently wrong is worse than none — it is the number a human
        # would act on.
        #
        # popleft-per-visit is positionally exact: this loop walks
        # broker_orders in submission order, which is the order the intents
        # were appended in, so the k-th visit to a (symbol, side) pops that
        # key's k-th intent.
        surviving_intent: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
        for i, (_, brow) in enumerate(broker_orders.iterrows()):
            sym = str(brow["symbol"])
            side = str(brow["side"]).strip().lower()
            this_intent: dict[str, Any] = order_intent[(sym, side)].popleft() if order_intent[(sym, side)] else {}
            order_ref = f"{cycle_id}:{i}:{sym}:{side}"
            try:
                ledger.record_intent(
                    cycle_id=cycle_id,
                    symbol=sym,
                    side=side,
                    order_ref=order_ref,
                    target_qty=_safe_float(brow.get("qty")),
                    limit_px=_safe_float(brow.get("price")),
                )
            except Exception as exc:
                if capture_fail_closed:
                    # ENFORCE: never trade live without the durable intent
                    # record — DROP the order (do not submit it).
                    logger.error(
                        "RECON ENFORCE: intent capture failed for %s %s — DROPPING order (fail closed; not sent): %s",
                        side,
                        sym,
                        exc,
                    )
                    report["orders_details"].append(
                        {
                            "symbol": sym,
                            "action": side,
                            "status": "FAILED",
                            "error": f"intent capture failed under enforce (order dropped): {exc}",
                            # Its OWN intent, popped above — so the freeze
                            # alert reports this order's notional and not a
                            # neighbour's.
                            "notional": this_intent.get("notional", 0.0),
                            "min_notional": this_intent.get("min_notional", 0.0),
                        }
                    )
                    report["summary"]["total_failed"] += 1
                    report["recon_capture_dropped"] = report.get("recon_capture_dropped", 0) + 1
                    dropped_idx.append(i)
                    continue
                # OBSERVE: ledger must never block execution — the order still
                # executes. But we MUST still reserve a positional slot so result
                # matching stays 1:1: append a None SENTINEL. Without it, a later
                # fill for another same-(symbol, side) order would pop THIS
                # order's would-be slot and bind to the WRONG intent. The sentinel
                # absorbs this order's own result (recording nothing, since its
                # intent was never durably written) and keeps every other order's
                # result bound to its own ref.
                logger.warning("Intent capture failed for %s %s (order still sent): %s", side, sym, exc)
                intent_refs[(sym, side)].append(None)
                surviving_intent[(sym, side)].append(this_intent)
                continue
            intent_refs[(sym, side)].append(order_ref)
            surviving_intent[(sym, side)].append(this_intent)

        # The freeze builder must consume only the intents of orders that were
        # actually submitted; the dropped ones now travel on their own
        # orders_details rows instead.
        order_intent = surviving_intent

        # Under enforce, actually remove the dropped orders before submission.
        if dropped_idx:
            broker_orders = broker_orders.drop(broker_orders.index[dropped_idx]).reset_index(drop=True)
            if broker_orders.empty:
                logger.error("RECON ENFORCE: all orders dropped (intent capture failed) — nothing sent.")
                # Same rule as the batch-exception path: every exit from this
                # function that leaves failures on the report must report them.
                # Dropping the entire batch under enforce IS a total trading
                # standstill — precisely when a human must be told.
                report_order_failures(report, alert)
                return report

    # The declared intent of each SUBMITTED order, per (symbol, side), in
    # submission order. Every broker row consumes one slot, whatever its status,
    # so the k-th row of a key reads the k-th submitted order's intent (TOM-901).
    declared_closes: dict[tuple[str, str], deque[bool]] = defaultdict(deque)
    for key, intents in order_intent.items():
        declared_closes[key].extend(bool(it.get("closes", key[1] == "sell")) for it in intents)

    def _record_result(symbol: str, side: str, status: str, filled_qty: Any = None, avg_px: Any = None) -> None:
        """Bind a broker result back to a captured intent (one-to-one, FIFO)."""
        if ledger is None or cycle_id is None:
            return
        pool = intent_refs.get((str(symbol), str(side).strip().lower()))
        if not pool:
            return
        ref = pool.pop(0)
        if ref is None:
            # Sentinel slot (intent write had failed): consume it to keep the
            # FIFO aligned, but record nothing — there is no durable intent to
            # bind this result to.
            return
        try:
            ledger.record_result(
                order_ref=ref,
                cycle_id=cycle_id,
                status=EXEC_STATUS_TO_LEDGER.get(str(status).strip().upper(), str(status).strip().lower()),
                filled_qty=_safe_float(filled_qty),
                avg_px=_safe_float(avg_px),
            )
        except Exception as exc:  # never let the ledger crash the run
            logger.warning("Result capture failed for %s %s: %s", side, symbol, exc)

    try:
        # THE BROKER SEAM (TOM-1458): every broker's rows, one normaliser.
        fills = normalise_results(broker.place_orders(broker_orders))
    except Exception as exc:
        logger.error("Broker place_orders failed: %s", exc)
        report["api_errors"].append({"stage": "place_orders", "error": str(exc)})
        for _, row in broker_orders.iterrows():
            report["orders_details"].append(
                {
                    "symbol": row["symbol"],
                    "action": row["side"],
                    "status": "FAILED",
                    "error": str(exc),
                }
            )
            report["summary"]["total_failed"] += 1
            # The whole batch failed at submission — record a failed result so
            # the ledger closes each intent (no dangling missed-fill timeouts).
            _record_result(row["symbol"], row["side"], "FAILED")
        # A batch-level submission failure is the WORST case (nothing traded
        # at all), so it must alert on the same path as a per-order rejection.
        report_order_failures(report, alert)
        return report

    # Results bind to orders by position within (symbol, side). That holds only
    # when the broker returns one row per submitted order of each key. Where it
    # does not, no row of that key can be bound: fail closed (TOM-2396).
    unbound: set[tuple[str, str]] = set()
    if fills is not None and not fills.empty:
        unbound = unbound_keys(broker_orders, fills)
        if unbound:
            fills = _fail_closed_unbound(
                report, broker_orders, fills, unbound, intent_refs, declared_closes, ledger, cycle_id
            )

    # Process fills and failures
    if fills is not None and not fills.empty:
        n_skipped = 0
        n_skipped_close = 0  # skipped orders that DECLARED an exit (#81, TOM-901)
        for _, fill_row in fills.iterrows():
            # The normaliser already stripped and upper-cased the status, and
            # made an absent, empty or unrecognised one UNKNOWN (TOM-1336): a
            # padded "SKIPPED " cannot slip past the freeze counter (#81), and
            # an order the venue never confirmed never books as a fill.
            status = str(fill_row.get("status"))
            _raw_status = fill_row.get(RAW_STATUS, fill_row.get("status"))
            side = str(fill_row.get("side", "")).strip().lower()
            _declared = declared_closes.get((str(fill_row.get("symbol", "")).strip(), side))
            closes = _declared.popleft() if _declared else side == "sell"
            if status == "SKIPPED":
                # Broker intentionally did not place this order (sub-minimum /
                # sub-precision dust). A clean no-op: neither executed nor
                # failed — record for visibility but do not count it.
                report["orders_details"].append(
                    {
                        "symbol": str(fill_row.get("symbol", "")),
                        "action": str(fill_row.get("side", "")),
                        "quantity": float(fill_row.get("qty", 0)),
                        "status": "SKIPPED",
                        "error": str(fill_row.get("error", "below exchange minimum (skipped)")),
                    }
                )
                n_skipped += 1
                _record_result(fill_row.get("symbol", ""), side, "SKIPPED")
                if closes:
                    # An exit the book WANTED to make but the broker could not
                    # place (sub-exchange-min). Tracked so an all-skipped batch
                    # with a trapped residual is not mistaken for a healthy run
                    # (#81). Counted on the order's DECLARED intent, never on its
                    # side: a declined OPEN is not a stuck close (TOM-901).
                    n_skipped_close += 1
                continue
            if status == "WORKING":
                # The venue accepted this order and it was still on the book
                # when the confirmation window closed. NOT a failure and NOT a
                # fill: record it for visibility, queue it for resolution by
                # the next cycle, and leave the ledger intent deliberately
                # unmatched — the order has no terminal result yet, and
                # writing one would forge an outcome we have not observed.
                order_id = str(fill_row.get("order_id") or "")
                report["orders_details"].append(
                    {
                        "symbol": str(fill_row.get("symbol", "")),
                        "action": str(fill_row.get("side", "")),
                        "quantity": float(fill_row.get("qty", 0)),
                        "status": "WORKING",
                        "order_id": order_id,
                        "error": str(fill_row.get("error", "still working at the venue")),
                    }
                )
                report["summary"]["total_working"] += 1
                # Consume this order's intent slot to keep the per-(symbol,
                # side) FIFO aligned with the results that follow, and carry
                # the ref forward so the late fill binds to THIS intent.
                _pool = intent_refs.get((str(fill_row.get("symbol", "")), side))
                intent_ref = _pool.pop(0) if _pool else None
                # Record an explicit NON-TERMINAL `working` result. Leaving the
                # intent unmatched would be read by the reconciliation close as
                # "submitted, no result observed" -- a MISSED FILL -- and with
                # degraded_failed_streak=1 a single resting limit order would
                # open a reconciliation break every cycle. That is the same
                # cry-wolf this change exists to remove, moved from Telegram
                # into recon. The record claims no outcome; it states the
                # observed fact that the order is alive.
                if ledger is not None and cycle_id is not None and intent_ref:
                    try:
                        ledger.record_result(
                            order_ref=str(intent_ref),
                            cycle_id=cycle_id,
                            status="working",
                            order_id=order_id,
                        )
                    except Exception:  # noqa: BLE001 - ledger must not break the run
                        logger.exception(
                            "Failed to record the working result for %s",
                            fill_row.get("symbol", ""),
                        )
                if order_id:
                    working_to_queue.append(
                        {
                            "symbol": str(fill_row.get("symbol", "")),
                            "side": side,
                            "order_id": order_id,
                            "requested_qty": float(fill_row.get("requested_qty", 0) or 0.0),
                            "reason": str(fill_row.get("error", "")),
                            "order_ref": intent_ref,
                        }
                    )
                else:
                    # Working but untrackable: no id means no later resolution,
                    # so this one really can go missing. Say so loudly rather
                    # than let it look like the handled case.
                    logger.error(
                        "Order for %s is working at the venue but carries NO order id — "
                        "it cannot be resolved next cycle and its fill may go unrecorded",
                        fill_row.get("symbol", ""),
                    )
                continue
            if status == "UNKNOWN":
                # The broker could not confirm what happened. Book NO fill.
                # It is reported as a failure (so the operator is alerted and
                # every FAILED consumer downstream sees a non-fill), and when
                # it carries an order id it joins the working-order queue so
                # the next cycle books the venue's real outcome.
                _oid = fill_row.get("order_id")
                order_id = "" if _oid is None or pd.isna(_oid) else str(_oid)
                _err = fill_row.get("error")
                _err = "" if _err is None or pd.isna(_err) else str(_err)
                # A QUEUED unconfirmed order is non-terminal `unknown` in the
                # ledger, the same rule as WORKING: the resolver books its real
                # outcome against this order_ref next cycle. Booking it terminal
                # `failed` let the reconciliation close count it into the
                # failure streak — one unconfirmed order opened a
                # consecutive_failed break, three escalated to HALT. With no id
                # or no queue nothing will ever resolve it, so it stays terminal
                # `failed` and loud.
                queued = bool(order_id) and working_store is not None
                report["orders_details"].append(
                    {
                        "symbol": str(fill_row.get("symbol", "")),
                        "action": str(fill_row.get("side", "")),
                        "quantity": 0.0,
                        "status": "FAILED",
                        "fill_status": "UNKNOWN",
                        "queued": queued,
                        "order_id": order_id,
                        "error": (
                            f"fill UNCONFIRMED (broker status={_raw_status!r}); no fill booked"
                            + (f" — {_err}" if _err else "")
                        ),
                    }
                )
                report["summary"]["total_failed"] += 1
                _pool = intent_refs.get((str(fill_row.get("symbol", "")), side))
                intent_ref = _pool.pop(0) if _pool else None
                if ledger is not None and cycle_id is not None and intent_ref:
                    try:
                        if queued:
                            ledger.record_result(
                                order_ref=str(intent_ref), cycle_id=cycle_id, status="unknown", order_id=order_id
                            )
                        else:
                            ledger.record_result(order_ref=str(intent_ref), cycle_id=cycle_id, status="failed")
                    except Exception:  # noqa: BLE001 - ledger must not break the run
                        logger.exception("Failed to record the unconfirmed result for %s", fill_row.get("symbol", ""))
                if order_id:
                    working_to_queue.append(
                        {
                            "symbol": str(fill_row.get("symbol", "")),
                            "side": side,
                            "order_id": order_id,
                            "requested_qty": float(fill_row.get("requested_qty", 0) or 0.0),
                            "reason": f"fill unconfirmed: {_err}",
                            "order_ref": intent_ref,
                        }
                    )
                else:
                    logger.error(
                        "Order for %s has an UNCONFIRMED fill and NO order id — it cannot be "
                        "resolved next cycle; reconcile it against the venue by hand",
                        fill_row.get("symbol", ""),
                    )
                continue
            if status == "FAILED":
                report["orders_details"].append(
                    {
                        "symbol": str(fill_row.get("symbol", "")),
                        "action": str(fill_row.get("side", "")),
                        "quantity": float(fill_row.get("qty", 0)),
                        "status": "FAILED",
                        "error": str(fill_row.get("error", "placement failed")),
                    }
                )
                report["summary"]["total_failed"] += 1
                _record_result(fill_row.get("symbol", ""), side, "FAILED")
                continue

            # FILLED or PARTIAL: a real (possibly partial) fill. Record the
            # ACTUAL filled qty the broker confirmed (issue #68) so the book's
            # recorded position matches reality; a partial is flagged so
            # monitors can see the unfilled remainder.
            is_partial = status == "PARTIAL"
            if is_partial:
                logger.warning(
                    "Partial fill: %s %s — %s",
                    side,
                    str(fill_row.get("symbol", "")),
                    str(fill_row.get("error", "")),
                )
            detail = {
                "symbol": str(fill_row.get("symbol", "")),
                "action": str(fill_row.get("side", "")),
                "side": str(fill_row.get("side", "")),
                "executed_quantity": float(fill_row.get("qty", 0)),
                "executed_price": float(fill_row.get("price", 0)),
                # #92: absent means the broker did not report one — UNKNOWN,
                # not free. No LIVE broker's place_orders emits a fee key;
                # only futures_paper does. The old `, 0` default is what made
                # `fees_this_run` read $0.00 on every live run.
                "fee": to_fee(fill_row.get("fee")),
                "status": "PARTIAL" if is_partial else "FILLED",
            }
            # Find reference price
            ref_price = 0.0
            matching = broker_orders[broker_orders["symbol"] == detail["symbol"]]
            if not matching.empty:
                ref_price = float(matching.iloc[0]["price"])
            detail["reference_price"] = ref_price
            exec_price = detail["executed_price"]
            if ref_price > 0 and exec_price > 0:
                detail["spread_pct"] = abs(exec_price - ref_price) / ref_price
            else:
                detail["spread_pct"] = 0.0

            report["orders_details"].append(detail)
            report["summary"]["total_executed"] += 1
            _record_result(
                detail["symbol"],
                side,
                "PARTIAL" if is_partial else "FILLED",
                filled_qty=detail["executed_quantity"],
                avg_px=detail["executed_price"],
            )
            if is_partial:
                # A partial fill DID execute (so it counts as executed and keeps
                # the freeze logic honest), but the unfilled remainder is real
                # residual exposure — surface it as a first-class incomplete
                # outcome so the notifier can raise an exception on it instead of
                # reading the run as a clean success.
                report["summary"]["total_partial"] += 1

        # Freeze guard (issue #81): the broker returned rows, but if EVERY
        # submitted order was SKIPPED (zero fills, zero hard failures) and at
        # least one of those skips was a declared EXIT, the book has a
        # trapped residual it wanted to exit but could not place at the venue.
        # Post the dust-fix this surfaces as SKIPPED instead of the pre-PR
        # FAILED count, so without this it would read as a healthy run. Treat
        # it as a freeze and alert — the trapped-residual signal must not be
        # lost just because the orders were *executable* upstream.
        # Fires whenever zero orders filled AND at least one declared exit was
        # skipped — regardless of whether other orders ALSO hard-failed. The
        # trapped-residual signal must NOT be masked just because total_failed>0
        # (a concurrent placement failure would otherwise hide the retained
        # residual behind the generic FAILED alert).
        if report["summary"]["total_executed"] == 0 and n_skipped_close >= 1:
            _freeze_trapped_residual(report, order_intent, n_skipped, n_skipped_close, alert)
    elif not unbound:
        # Submitted orders but received zero fills — broker-level failure.
        order_list = ", ".join(f"{r['side'].upper()} {r['qty']:.4f} {r['symbol']}" for _, r in broker_orders.iterrows())
        logger.error(
            "Broker returned 0 fills for %d submitted orders: %s",
            len(broker_orders),
            order_list,
        )
        # Was the one unwrapped site: a raising notify here failed the run.
        alert(
            f"⚠️ <b>0 FILLS for {len(broker_orders)} orders</b>\n"
            f"Submitted: {escape_html(order_list)}\n"
            "Check broker connectivity — portfolio NOT rebalanced.",
            site="zero_fills",
        )

    # Close any INTENT with no observed result as a TIMEOUT (a missed fill:
    # submitted but the broker returned nothing for it). Authority rule: the
    # internal ledger is the truth for intent — this is the class the exchange
    # alone cannot prove. Done after all fills are matched so only genuinely
    # unresolved intents remain.
    if ledger is not None and cycle_id is not None:
        for (sym, side), refs in intent_refs.items():
            for ref in refs:
                if ref is None:
                    continue  # sentinel: no durable intent to close
                try:
                    ledger.record_result(order_ref=ref, cycle_id=cycle_id, status="timeout")
                except Exception as exc:  # never let the ledger crash the run
                    logger.warning("Timeout capture failed for %s %s: %s", side, sym, exc)
        intent_refs.clear()

    report["summary"]["total_value"] = sum(
        d.get("executed_quantity", 0) * d.get("executed_price", 0)
        for d in report["orders_details"]
        if d.get("status") in ("FILLED", "PARTIAL")
    )

    # Persist BEFORE the alerting call below: queuing a working order is the
    # durable step that lets its fill be recovered next cycle, and it must not
    # be skipped because a notification path raised.
    if working_to_queue and working_store is not None:
        for rec in working_to_queue:
            try:
                working_store.record(cycle_id=str(cycle_id or ""), **rec)
            except Exception:
                # A lost queue entry is a lost fill, so this is loud and it
                # does NOT swallow: the run continues (the order is live and
                # the position reconciles next cycle regardless), but the
                # operator must see that automatic resolution will not happen.
                logger.exception(
                    "FAILED to queue working order %s %s (id=%s) for next-cycle "
                    "resolution — its fill will not be booked automatically",
                    rec.get("side"),
                    rec.get("symbol"),
                    rec.get("order_id"),
                )
    elif working_to_queue:
        logger.warning(
            "%d order(s) working at the venue but no working-order store is "
            "configured — their fills will not be booked automatically",
            len(working_to_queue),
        )

    report_order_failures(report, alert)
    return report


def _freeze_trapped_residual(
    report: dict[str, Any],
    order_intent: dict[tuple[str, str], deque[dict[str, Any]]],
    n_skipped: int,
    n_skipped_close: int,
    alert: Alert,
) -> None:
    """The #81 broker-side freeze: 0 fills and at least one declared exit skipped."""
    report["frozen"] = True
    reasons = {
        # The key keeps its name for every reader downstream; it counts skipped
        # EXITS, by declared intent (TOM-901), not skipped sells.
        "skipped_sell": n_skipped_close,
        "skipped_total": n_skipped,
        "failed": int(report["summary"]["total_failed"]),
    }
    report["freeze_reasons"] = reasons

    # Same explainability contract as the upstream freeze: name each
    # order and why it could not go through, not just a count. These
    # rows come from the BROKER (post-submission), so the fields are
    # the order_details shape rather than the rebalancer's columns.
    # Recover notional + venue minimum from the submission intent: the
    # broker's own rows do not carry them (see order_intent above), so
    # reading only `d` yields $0.00 and no shortfall — useless in exactly
    # the trapped-residual case this alert exists for.
    def _frozen_record(d: dict[str, Any]) -> dict[str, Any]:
        # Strip before matching: `orders_details` carries the broker's
        # RAW side, and a padded " SELL " (#81's shape) misses the
        # intent key — reinstating the "$0.00" bug in the very case
        # the recovery exists for (caught in review of #144).
        symbol = str(d.get("symbol", "")).strip()
        action = str(d.get("action", "")).strip()
        # Consume the FIFO so duplicate (symbol, side) orders each get
        # their OWN intent rather than all reading the last one.
        #
        # ONLY when this row does not already carry its own numbers.
        # An enforce-DROPPED order brings its intent along on its own
        # details row and was never added to the surviving FIFO, so
        # popping for it would consume a SUBMITTED order's intent and
        # leave that order reading $0.00 — the same misalignment one
        # step removed.
        own_notional = d.get("notional") or d.get("target_value")
        own_min = d.get("min_notional")
        if own_notional is None or own_min is None:
            pending = order_intent.get((symbol, action.lower()))
            intent = pending.popleft() if pending else {}
        else:
            intent = {}
        notional = own_notional or intent.get("notional") or 0
        min_notional = own_min or intent.get("min_notional") or 0
        return {
            "symbol": symbol,
            "action": action.upper(),
            "notional_usd": float(notional),
            "min_notional_usd": float(min_notional),
            "status": str(d.get("status", "")),
            "reason": str(d.get("reason") or d.get("error") or ""),
        }

    report["frozen_orders"] = [
        _frozen_record(d) for d in report["orders_details"] if d.get("status") in {"SKIPPED", "FAILED"}
    ]
    for rec in report["frozen_orders"]:
        mn, n = rec["min_notional_usd"], rec["notional_usd"]
        if mn > 0 and 0 < n < mn:
            rec["shortfall_usd"] = mn - n
    skipped_lines = [format_suppressed_line(r) for r in report["frozen_orders"] if r["status"] == "SKIPPED"]
    skipped_list = "; ".join(skipped_lines)
    failed_n = int(report["summary"]["total_failed"])
    failed_note = f"; {failed_n} order(s) also hard-failed" if failed_n else ""
    logger.error(
        "REBALANCER FROZEN: 0 fills; %d close-out order(s) skipped "
        "sub-minimum (trapped residual)%s. Residual exposure RETAINED — "
        "book NOT rebalanced. Skipped: %s",
        n_skipped_close,
        failed_note,
        skipped_list,
    )
    alert(
        "🧊 <b>REBALANCER FROZEN — trapped residual</b>\n"
        f"0 fills; {n_skipped_close} close-out SELL(s) below exchange "
        f"minimum{failed_note}; residual exposure RETAINED.\n"
        f"Skipped: {escape_html(skipped_list)}\n"
        "Book NOT rebalanced — monitor for a trapped residual.",
        site="trapped_residual",
    )
