#!/usr/bin/env python3
"""Deterministic "Live book health" pre-compute (TOM-146).

CORE ONLY: runs as user prod from core's box-job `live-books-publish` (bridge jobs/live_books_publish.py, 07:30),
which publishes its --json as GET /data/live-books. Agents on forge2 never run it and have no /home/prod; they read
that route. It reads the live-book configs + latest daily reports directly from
/home/prod/workspace/prod/quantbox-live (READ ONLY — never write under
/home/prod) and emits one health record per book:

  - equity (current) + peak equity + current drawdown-from-peak, both taken
    from the SAME series — the report's own equity-curve ASCII chart — so
    current and peak are guaranteed like-for-like. (Codex review 2026-07-22:
    `Portfolio (post)` is NOT the right basis — for a margined book it can
    diverge from the reconciled `Account equity` the curve is actually built
    from; e.g. carver-HL 2026-07-22 showed Portfolio(post)=$85.73 vs the
    curve's own last point / Account equity ~$83.8-84.0, understating
    drawdown. The curve's last point matches Account equity when a Margin
    Status section exists, and matches Portfolio (post) exactly for a
    non-margined spot book (Kraken) — so using it is correct for both book
    shapes without needing to special-case which table exists.)
  - the configured drawdown_from_peak_pct limit (alert-only, no kill-switch)
  - trading_enabled flag from the book's config
  - last fill: this run's fill count, and if 0, the most recent PRIOR report
    (its own long lookback, LAST_FILL_LOOKBACK_DAYS) that had fills > 0, with
    days_ago
  - breach_start / over_limit_dates: the over-limit history on that same curve
    (KPI-writer contract, TOM-1025); null when curve or limit is unreadable
  - failed/unconfirmed orders from the latest run's "Failed Orders" table
  - staleness: flags loudly if today's report is missing, if the most recent
    report found is more than 1 day old BY FILENAME, or if the report's own
    "Generated:" timestamp disagrees with its filename date (a mis-dated or
    stale-content file passing the filename check)

No network calls, no fabrication — every field traces to a report/config
line; a missing/unreadable/unparseable source is reported as such LOUDLY,
never silently dropped, defaulted, or guessed.

Usage: python3 live-book-health.py [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path

# Prod on core, READ ONLY: the one place this default is right. QUANTBOX_LIVE points elsewhere (tests, a copy).
LIVE_ROOT = Path(os.environ.get("QUANTBOX_LIVE", "/home/prod/workspace/prod/quantbox-live"))
CONFIGS_DIR = LIVE_ROOT / "configs"
REPORTS_DIR = LIVE_ROOT / "reports"

# book_id -> (config file, reports subdir)
BOOKS = {
    "carver-HL": ("carver_hyperliquid.yaml", "carver-HL"),
    "crypto_trend-Kraken": ("crypto_trend_kraken.yaml", "crypto-trend-kraken"),
}

# A MUTED BOOK IS A TEST BOOK: it is still measured and still rendered, but it
# cannot raise the exit code, so nothing escalates off it.
#
# Tom, 2026-09-25 (voice): "czemu kurwa caly czas jestes cos z tym karwerem
# Hyperliquid? Wez sobie to gdzies oznacz, ze to jest taki trochę na razie
# testowo i nic z tym nie bedziemy robic. Nie masz mi o tym na razie nic mowic."
#
# carver-HL had been over its drawdown limit for 121 consecutive days, so it
# emitted a line every single morning about a book nobody intended to act on --
# the alarm nobody reads, and the very shape the 15/16/17.09 note in
# references/daily-brief.md already warned about. Muting is done HERE, in the
# exit code, and not as a sentence in the skill, because a rule in prose is not
# an enforcer: a caller that scheduled this script would still have seen
# EXIT_FINDING every day.
#
# To unmute, delete the entry -- do not add a flag to override it.
MUTED_BOOKS = {
    "carver-HL": "test book, parked by Tom 2026-09-25 -- measured, never escalated",
}

# `### Failed Orders` IS AN OPTIONAL SECTION, AND ITS ABSENCE IS NOT A LOST ONE.
#
# Corrected 2026-09-08 after I got this wrong. On 07-09 I saw the section absent
# from a Kraken report that had carried it "daily for months" and called it a
# lost section -- BAD_SOURCE. Checking all 69 reports says otherwise: the heading
# has NEVER once appeared with an empty table, and it is absent on exactly the
# days with no failed orders (08-22, 08-24, 08-25, 09-03, 09-07). The template
# emits the section only when there is something to put in it.
#
# So absence means "no failed orders", and the guard I wrote would have fired
# BAD_SOURCE on 5 of the last 20 Kraken days -- a false alarm on every clean day,
# which is how an alarm stops being read at all.
#
# What CANNOT be told apart from the report alone is "no failures" from "the
# failed-order check never ran". That is a real limit and it is stated in the
# rendered line rather than papered over with a verified-sounding "none".
STALE_LOOKBACK_DAYS = 14
# The last-fill search has its OWN, much longer lookback (2026-09-17). It used to
# share STALE_LOOKBACK_DAYS, so once the last fill was older than 14 days the
# block said "none in the last 14d" forever and the DATE was lost -- carver-HL's
# last fill was 2026-08-30 and the brief said "no fill in 14 days" on 14.09,
# 16.09 and 17.09: a counter that does not advance. Staleness keeps its 14.
LAST_FILL_LOOKBACK_DAYS = 400
# `over_limit_dates` covers this many calendar days, INCLUSIVE of the report date.
OVER_LIMIT_WINDOW_DAYS = 7
# A report whose own "Generated:" date differs from its filename date by more
# than this is flagged CONTENT_DATE_MISMATCH regardless of what the filename
# says (fix #3 — staleness must not key on the filename alone).
GENERATED_DATE_TOLERANCE_DAYS = 1
# Equity curves shorter than this look potentially windowed/truncated rather
# than full history — not proof, just a loud caveat (fix: the peak-window
# assertion codex asked for; we have no documented per-book inception date to
# assert against, so this is an honest "verify" flag, not a hard assertion).
SHORT_CURVE_WARN_DAYS = 30


def read_config_flags(config_path: Path) -> dict:
    """Pull trading_enabled + drawdown_from_peak_pct out of the YAML by grep,
    not a full YAML parse — these configs are prod-owned and read-only; a
    lightweight regex avoids adding a yaml dependency for two scalars.
    A config that exists but doesn't match either pattern is a LOUD parse
    failure (fix #5), not a silent None — different nesting or a YAML `yes`/
    `on` boolean must not render as "unknown" with no signal."""
    out: dict = {"config_error": None}
    if not config_path.exists():
        out["config_error"] = f"config not found: {config_path}"
        out["trading_enabled"] = None
        out["drawdown_from_peak_pct_limit"] = None
        return out

    text = config_path.read_text(errors="replace")
    errors = []

    m = re.search(r"trading_enabled:\s*(true|false)", text)
    if m:
        out["trading_enabled"] = m.group(1) == "true"
    else:
        out["trading_enabled"] = None
        errors.append("trading_enabled: not found (or non-boolean/nested form)")

    m = re.search(r"drawdown_from_peak_pct:\s*([\d.]+)", text)
    if m:
        out["drawdown_from_peak_pct_limit"] = float(m.group(1))
    else:
        out["drawdown_from_peak_pct_limit"] = None
        errors.append("drawdown_from_peak_pct: not found")

    out["config_error"] = "; ".join(errors) if errors else None
    return out


def _parse_generated(raw: str | None) -> datetime | None:
    if not raw:
        return None
    # "2026-07-22 04:04 UTC"
    m = re.match(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})", raw)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def _equity_curve_from_performance_json(
    report_path: Path,
) -> list[tuple[str, str]]:
    """The full dated equity series for the run whose report is `report_path`.

    Returns `(date, equity-as-string)` pairs so the caller's existing
    comma-stripping float parse is untouched, and `[]` when the file is absent
    or unreadable -- which the caller turns into a LOUD `missing_sections`
    entry, never a clean reading.

    `reports/<book>/<YYYY-MM-DD>.md` is the rendering; `reports/<book>/<YYYY-MM-DD>/
    performance.json` is the data behind it, written by the same run.
    """
    day_dir = report_path.with_suffix("")
    perf_path = day_dir / "performance.json"
    try:
        payload = json.loads(perf_path.read_text())
    except (OSError, ValueError):
        # Unreadable and absent are the same answer HERE -- "no series" -- and
        # the caller makes that loud. It must not raise: one corrupt file for
        # one book cannot take down the check for the others.
        return []
    out: list[tuple[str, str]] = []
    for point in payload.get("equity_curve") or []:
        d, e = point.get("date"), point.get("equity")
        # A null/NaN mark is dropped rather than coerced: `float("nan")` would
        # poison max() and every comparison against it is False, so a single bad
        # mark would silently flatten the peak.
        if not d or e is None or e != e:
            continue
        out.append((str(d), str(e)))
    return out


def parse_report(report_path: Path) -> dict:
    text = report_path.read_text(errors="replace")
    out: dict = {"path": str(report_path)}

    m = re.search(r"\*\*Generated:\*\*\s*(.+)", text)
    out["generated_raw"] = m.group(1).strip() if m else None
    out["generated_dt"] = _parse_generated(out["generated_raw"])

    # The row was RENAMED. `generate_report.py` wrote `Portfolio (post)` until
    # the v0.7.0 report work (PR #133/#135, merged 2026-09-23) replaced it with
    # `Portfolio value (run end)` and added `Sizing base (marked equity)` above
    # it. This parse kept asking for the old name, so from 2026-09-24 BOTH live
    # books reported BAD_SOURCE and my kpi-1 read BLIND every day -- my own
    # change blinded my own monitor, and nothing connected the two because the
    # report still looked perfectly healthy to a human.
    #
    # Both names are accepted: the historical reports on disk still carry the
    # old one, and a parser that can only read today cannot answer a question
    # about last month.
    m = re.search(
        r"\|\s*(?:Portfolio \(post\)|Portfolio value \(run end\))\s*\|\s*\$(-?[\d,.]+)\s*\|",
        text,
    )
    out["portfolio_post"] = float(m.group(1).replace(",", "")) if m else None

    m = re.search(r"\|\s*Fills\s*\|\s*(\d+)\s*\|", text)
    out["fills"] = int(m.group(1)) if m else None

    m = re.search(r"\|\s*Max Drawdown\s*\|\s*(-?[\d.]+)%\s*\|", text)
    out["max_drawdown_pct_reported"] = float(m.group(1)) if m else None

    # THE EQUITY SERIES COMES FROM `performance.json`, NOT FROM THIS REPORT.
    #
    # It used to be scraped out of the report's ASCII bar chart. That was the
    # wrong design and it cost two days of blind KPI readings: the v0.7.0 report
    # work (2026-09-23) replaced the chart with a table of the "last 14 of 84
    # marks", and a monitor whose input is a RENDERING breaks whenever the
    # rendering changes. Tom, 2026-09-25: "to ze liczymy performance to sobie
    # liczmy a co raportujemy w equity curve w raporcie to inne -- raport nie
    # powinien byc zrodlem do wyliczen performance."
    #
    # Reading the report's table instead would have been worse than blind: it
    # holds 14 of 84 marks, so the peak -- and therefore drawdown against the
    # limit -- would have been computed over a fortnight and come out
    # UNDERSTATED, with nothing to say so.
    #
    # `performance.json` sits in `<book>/<date>/` beside the report it feeds and
    # carries the whole series. Absence is still LOUD (`equity curve` lands in
    # `missing_sections` exactly as before): a monitor that cannot find its
    # source must not report a clean book.
    curve = _equity_curve_from_performance_json(report_path)
    out["equity_curve_points"] = len(curve)
    out["equity_curve_first_date"] = curve[0][0] if curve else None
    out["equity_curve_last_date"] = curve[-1][0] if curve else None
    values = [float(v.replace(",", "")) for v in (c[1] for c in curve)]
    # The dated series itself, for the over-limit history -- the SAME parse as
    # the peak/current below, never a second one.
    out["equity_curve"] = [(c[0], v) for c, v in zip(curve, values)]
    out["equity_current"] = values[-1] if values else None
    # explicit is-not-None + positive check (fix #4) — a zero/negative peak
    # (or a peak that's merely falsy-but-present, e.g. 0.0) must not be
    # silently treated as "missing" or produce signed nonsense.
    positive_values = [v for v in values if v is not None and v > 0]
    out["equity_peak"] = max(positive_values) if positive_values else None
    out["equity_curve_has_nonpositive"] = (
        any(v <= 0 for v in values) if values else False
    )

    # Required-sections check (fix #2): a truncated report (title +
    # Generated only) must not silently parse to an "OK" record with null
    # equity/drawdown — missing BOTH the summary line and the curve is the
    # signal of a bad/incomplete source.
    out["missing_sections"] = []
    if out["portfolio_post"] is None:
        out["missing_sections"].append("Portfolio Summary (Portfolio (post))")
    if not values:
        out["missing_sections"].append("equity curve")
    if out["fills"] is None:
        out["missing_sections"].append("Portfolio Summary (Fills)")

    # Failed / unconfirmed orders table. PRESENCE is recorded separately from
    # CONTENTS: `failed == []` from a section that was there and empty is a real
    # all-clear, while `failed == []` from a section that was never written is a
    # question nobody asked. Collapsing them is how this check reported clean.
    out["failed_orders_section_present"] = "### Failed Orders" in text
    failed = []
    fm = re.search(r"### Failed Orders\n\n(.*?)\n\n", text, re.S)
    if fm:
        for line in fm.group(1).splitlines():
            if (
                line.startswith("|")
                and "---" not in line
                and "symbol" not in line.lower()
            ):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) >= 4:
                    failed.append(
                        {
                            "symbol": cells[0],
                            "action": cells[1],
                            "error": cells[2],
                            "status": cells[3],
                        }
                    )
    out["failed_orders"] = failed

    return out


def over_limit_history(
    curve: list[tuple[str, float]], limit: float | None, report_date: date
) -> tuple[str | None, list[str] | None]:
    """(breach_start, over_limit_dates) from the report's own equity curve.

    Drawdown is measured from the RUNNING peak. Non-positive points are excluded
    exactly as the peak logic in parse_report excludes them: they neither move
    the peak nor count as a day over or under the limit, so they do not break a
    run either. The comparison is the one `over_limit` uses (2-dp rounded pct
    / 100 > limit), so breach_start is non-null exactly when over_limit is true.

    breach_start     first date of the CURRENT unbroken over-limit run; None when
                     the last point is not over the limit.
    over_limit_dates every curve date in the last OVER_LIMIT_WINDOW_DAYS calendar
                     days (inclusive of report_date) that was over the limit.

    Both are None -- never [] and never a guessed "not over" -- when the curve
    has no positive point or the limit could not be read: could-not-look is not
    the same answer as looked-and-clean.
    """
    points = [(d, v) for d, v in curve if v is not None and v > 0]
    if limit is None or not points:
        return None, None
    flags: list[tuple[str, bool]] = []
    peak = 0.0
    for d, v in points:
        peak = max(peak, v)
        dd_pct = round((peak - v) / peak * 100, 2)
        flags.append((d, (dd_pct / 100.0) > limit))

    breach_start = None
    for d, over in reversed(flags):
        if not over:
            break
        breach_start = d

    window_start = (
        report_date - timedelta(days=OVER_LIMIT_WINDOW_DAYS - 1)
    ).isoformat()
    window_end = report_date.isoformat()
    dates = [d for d, over in flags if over and window_start <= d <= window_end]
    return breach_start, dates


def find_last_fill(
    reports_dir: Path, before_date: date, max_lookback: int
) -> tuple[dict | None, int]:
    """(most recent prior report with fills > 0, how many reports were READ).
    The count is returned so "none found" can say how much it looked at."""
    reports_read = 0
    for i in range(1, max_lookback + 1):
        d = before_date - timedelta(days=i)
        f = reports_dir / f"{d.isoformat()}.md"
        if not f.exists():
            continue
        r = parse_report(f)
        reports_read += 1
        if r.get("fills"):
            return {"date": d.isoformat(), "fills": r["fills"]}, reports_read
    return None, reports_read


def health_for_book(
    book_id: str, config_file: str, reports_subdir: str, today: date
) -> dict:
    # The two KPI-contract fields exist on EVERY record, null until computed, so
    # an early MISSING / BAD_SOURCE return reads as "could not look", not omitted.
    rec: dict = {"book": book_id, "breach_start": None, "over_limit_dates": None}
    # A muted book still carries its breach history, and says it is muted, so a
    # consumer (kpi_quark's KPI #1) can exclude it by NAME instead of guessing.
    rec["muted"] = MUTED_BOOKS.get(book_id)

    config_path = CONFIGS_DIR / config_file
    rec["config"] = read_config_flags(config_path)

    reports_dir = REPORTS_DIR / reports_subdir
    if not reports_dir.exists():
        rec["status"] = "MISSING"
        rec["message"] = f"reports dir not found: {reports_dir}"
        return rec

    today_report = reports_dir / f"{today.isoformat()}.md"
    latest_path = None
    latest_date = None
    if today_report.exists():
        latest_path, latest_date = today_report, today
    else:
        for i in range(1, STALE_LOOKBACK_DAYS + 1):
            d = today - timedelta(days=i)
            cand = reports_dir / f"{d.isoformat()}.md"
            if cand.exists():
                latest_path, latest_date = cand, d
                break

    if latest_path is None:
        rec["status"] = "MISSING"
        rec["message"] = (
            f"no report found in the last {STALE_LOOKBACK_DAYS} days under {reports_dir}"
        )
        return rec

    filename_age_days = (today - latest_date).days
    report = parse_report(latest_path)
    rec["report_date"] = latest_date.isoformat()
    rec["report_filename_age_days"] = filename_age_days
    rec["equity"] = report.get("equity_current")
    rec["equity_peak"] = report.get("equity_peak")
    rec["equity_basis"] = (
        "equity_curve_last_point (== Account equity for margined books, == Portfolio (post) for spot)"
    )
    rec["portfolio_post_for_reference"] = report.get("portfolio_post")
    rec["max_drawdown_pct_reported"] = report.get("max_drawdown_pct_reported")
    rec["fills_this_run"] = report.get("fills")
    rec["failed_orders"] = report.get("failed_orders")
    rec["generated_raw"] = report.get("generated_raw")
    rec["missing_sections"] = report.get("missing_sections") or []
    rec["failed_orders_section_present"] = report.get("failed_orders_section_present")

    # NO `missing_sections` ENTRY FOR THIS. See the note at the top: the section
    # is omitted on days with nothing to report, so requiring it would alarm on
    # every clean day.
    rec["equity_curve_points"] = report.get("equity_curve_points")
    rec["equity_curve_first_date"] = report.get("equity_curve_first_date")
    rec["equity_curve_has_nonpositive"] = report.get("equity_curve_has_nonpositive")

    # --- fix #3: content-date check, not just filename ---
    gen_dt = report.get("generated_dt")
    content_date_mismatch = None
    if gen_dt is not None:
        gen_date = gen_dt.date()
        drift = abs((gen_date - latest_date).days)
        if drift > GENERATED_DATE_TOLERANCE_DAYS:
            content_date_mismatch = (
                f"filename says {latest_date.isoformat()} but report body says "
                f"Generated: {report.get('generated_raw')} ({drift}d apart)"
            )
    else:
        content_date_mismatch = "no parseable 'Generated:' timestamp in report body"
    rec["content_date_mismatch"] = content_date_mismatch

    # --- fix #2: missing required sections => loud bad-source status ---
    if rec["missing_sections"]:
        rec["status"] = "BAD_SOURCE"
        rec["message"] = (
            f"{latest_path.name} is missing required section(s): "
            + ", ".join(rec["missing_sections"])
            + " — treating as unreliable, not as a clean current reading"
        )
        return rec

    # --- fix #1 (cont'd): drawdown from the same series, with explicit
    # is-not-None + positive-peak guards (fix #4) ---
    if (
        rec["equity"] is not None
        and rec["equity_peak"] is not None
        and rec["equity_peak"] > 0
    ):
        rec["drawdown_from_peak_pct"] = round(
            (rec["equity_peak"] - rec["equity"]) / rec["equity_peak"] * 100, 2
        )
    else:
        rec["drawdown_from_peak_pct"] = None

    if not rec["fills_this_run"]:
        lf, reports_read = find_last_fill(
            reports_dir, latest_date, LAST_FILL_LOOKBACK_DAYS
        )
        rec["last_fill"] = lf  # None if no fill in lookback window either
        rec["last_fill_search"] = {
            "lookback_days": LAST_FILL_LOOKBACK_DAYS,
            "prior_reports_read": reports_read,
        }
    else:
        rec["last_fill"] = {
            "date": latest_date.isoformat(),
            "fills": rec["fills_this_run"],
        }
    if rec["last_fill"]:
        rec["last_fill"]["days_ago"] = (
            today - date.fromisoformat(rec["last_fill"]["date"])
        ).days

    # --- staleness verdict now considers BOTH signals (fix #3) ---
    if filename_age_days == 0 and content_date_mismatch is None:
        if rec["failed_orders"]:
            rec["status"] = "OK_WITH_FAILED_ORDERS"
        elif not rec["failed_orders_section_present"]:
            # Healthy, and the section's absence is the template saying "none".
            # Kept as its own status so a reader can still tell this apart from
            # a report that explicitly listed zero.
            rec["status"] = "OK_NO_FAILED_ORDERS_SECTION"
        else:
            rec["status"] = "OK"
    elif content_date_mismatch is not None and filename_age_days == 0:
        rec["status"] = "CONTENT_DATE_MISMATCH"
        rec["message"] = content_date_mismatch
    elif filename_age_days == 1:
        rec["status"] = "STALE_1D"
        rec["message"] = (
            f"no report since {latest_date.isoformat()} — today's report missing"
        )
    else:
        rec["status"] = "STALE"
        rec["message"] = (
            f"no report since {latest_date.isoformat()} ({filename_age_days}d ago)"
        )

    if content_date_mismatch is not None and rec["status"] not in (
        "CONTENT_DATE_MISMATCH",
    ):
        rec["message"] = (
            rec.get("message", "") + f"; also: {content_date_mismatch}"
        ).strip("; ")

    limit = rec["config"].get("drawdown_from_peak_pct_limit")
    if limit is not None and rec["drawdown_from_peak_pct"] is not None:
        rec["over_limit"] = (rec["drawdown_from_peak_pct"] / 100.0) > limit
    else:
        rec["over_limit"] = None

    rec["breach_start"], rec["over_limit_dates"] = over_limit_history(
        report.get("equity_curve") or [], limit, latest_date
    )

    # --- curve-window caveat (codex's peak-window ask) ---
    rec["curve_window_caveat"] = None
    if (
        rec.get("equity_curve_points")
        and rec["equity_curve_points"] < SHORT_CURVE_WARN_DAYS
    ):
        rec["curve_window_caveat"] = (
            f"equity curve is only {rec['equity_curve_points']} points "
            f"(first {rec['equity_curve_first_date']}) — if this is a young book "
            "(post-inception) the peak is genuine; if not, verify the report "
            "template isn't windowing the chart, since a truncated window "
            "silently understates the true peak/drawdown"
        )
    if rec.get("equity_curve_has_nonpositive"):
        rec["curve_window_caveat"] = (
            rec["curve_window_caveat"] + " " if rec["curve_window_caveat"] else ""
        ) + "curve also contains a zero/negative point — excluded from the peak."

    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        action="store_true",
        help="emit raw JSON instead of the rendered block",
    )
    ap.add_argument(
        "--date", default=None, help="override 'today' (YYYY-MM-DD), for testing"
    )
    args = ap.parse_args()

    today = date.fromisoformat(args.date) if args.date else date.today()

    records = [
        health_for_book(book_id, cfg, sub, today)
        for book_id, (cfg, sub) in BOOKS.items()
    ]

    if args.json:
        print(json.dumps(records, indent=2, default=str))
        raise SystemExit(exit_code(records))

    for rec in records:
        print(render_block(rec))
        print()

    raise SystemExit(exit_code(records))


# "Could not look" must exit DIFFERENTLY from "looked and it is fine". Until
# this existed, main() returned None on every path, so the status it had just
# computed never reached the caller at all: a MISSING reports dir and a clean
# book both exited 0, and anything scheduling this saw success either way.
EXIT_OK, EXIT_FINDING, EXIT_BLIND = 0, 1, 2

# Statuses that mean I COULD NOT LOOK, as opposed to looked-and-found-something.
_BLIND_STATUSES = {"MISSING", "BAD_SOURCE"}


def exit_code(records: list[dict]) -> int:
    # Muted books are dropped BEFORE any status is read: they are measured for
    # the rendered block, but they never move the exit code.
    #
    # The order matters. "Zero books examined" below means an empty FIELD OF
    # VIEW and must stay EXIT_BLIND -- but every book being muted is a
    # deliberate configuration, not blindness, so that case is answered first.
    # Without this, muting the last unmuted book would make the script report
    # EXIT_BLIND forever and read as broken.
    if records and all(r.get("book") in MUTED_BOOKS for r in records):
        return EXIT_OK
    records = [r for r in records if r.get("book") not in MUTED_BOOKS]
    statuses = [r.get("status") for r in records]
    if not records:
        # Zero books examined is not a clean estate; it is an empty field of view.
        return EXIT_BLIND
    if any(st in _BLIND_STATUSES for st in statuses):
        return EXIT_BLIND
    if any(
        st in {"OK_WITH_FAILED_ORDERS", "CONTENT_DATE_MISMATCH"}
        or (st or "").startswith("STALE")
        for st in statuses
    ) or any(r.get("over_limit") for r in records):
        return EXIT_FINDING
    return EXIT_OK


def render_block(rec: dict) -> str:
    book = rec["book"]
    # A muted book still renders -- it is measured, and hiding it would turn a
    # parked book into an invisible one -- but it says so on its own first line,
    # so the reader never has to remember which books are live.
    if book in MUTED_BOOKS:
        book = f"{book} [TEST -- parked, nie raportowac]"
    status = rec.get("status")

    if status == "MISSING":
        return f"**{book}** — ⚠️ NO DATA: {rec['message']}"
    if status == "BAD_SOURCE":
        return f"**{book}** — ⚠️ BAD SOURCE: {rec['message']}"

    cfg = rec.get("config", {})
    lines = [f"**{book}**"]

    if cfg.get("config_error"):
        lines.append(f"- ⚠️ CONFIG PARSE FAILURE: {cfg['config_error']}")

    limit = cfg.get("drawdown_from_peak_pct_limit")
    limit_str = (
        f"{limit * 100:.0f}%"
        if limit is not None
        else "— (unparseable, see config error above)"
    )
    dd = rec.get("drawdown_from_peak_pct")
    dd_str = f"{dd:.1f}%" if dd is not None else "—"
    over = " ⚠️ OVER LIMIT" if rec.get("over_limit") else ""

    if status == "CONTENT_DATE_MISMATCH":
        lines.append(f"- ⚠️ CONTENT/FILENAME DATE MISMATCH: {rec['message']}")
    elif status and status.startswith("STALE"):
        lines.append(f"- ⚠️ STALE: {rec['message']}")
    elif rec.get("content_date_mismatch"):
        lines.append(f"- ⚠️ ALSO: {rec['content_date_mismatch']}")

    lines.append(
        f"- Drawdown from peak: {dd_str} vs {limit_str} limit (alert-only, no kill-switch){over}"
    )
    if rec.get("breach_start"):
        day_n = (
            date.fromisoformat(rec["report_date"])
            - date.fromisoformat(rec["breach_start"])
        ).days + 1
        lines.append(f"- Over limit since {rec['breach_start']} (day {day_n})")
    eq = rec.get("equity")
    peak = rec.get("equity_peak")
    eq_str = f"${eq:.2f}" if eq is not None else "—"
    peak_str = f"${peak:.2f}" if peak is not None else "—"
    lines.append(
        f"- Equity: {eq_str} (peak {peak_str}, basis: equity-curve last point) "
        f"— as of {rec.get('report_date', '?')}"
    )

    te = cfg.get("trading_enabled")
    lines.append(
        f"- trading_enabled: {te if te is not None else 'unknown (config parse failure)'}"
    )

    lf = rec.get("last_fill")
    if lf:
        n = lf.get("days_ago")
        ago = f"{n} day{'' if n == 1 else 's'} ago, " if n is not None else ""
        # The "check execution" flag used to come from the 14d search coming back
        # empty; now that the date survives, the flag keys on its age instead.
        warn = (
            " — ⚠️ check execution" if n is not None and n > STALE_LOOKBACK_DAYS else ""
        )
        lines.append(f"- Last fill: {lf['date']} ({ago}{lf['fills']} filled){warn}")
    else:
        search = rec.get("last_fill_search") or {}
        lines.append(
            f"- Last fill: none in the last "
            f"{search.get('lookback_days', LAST_FILL_LOOKBACK_DAYS)}d "
            f"({search.get('prior_reports_read', '?')} prior reports read) "
            "— ⚠️ check execution"
        )

    failed = rec.get("failed_orders") or []
    if failed:
        fs = "; ".join(f"{o['symbol']} {o['action']} ({o['error']})" for o in failed)
        lines.append(f"- Failed/unconfirmed orders ({len(failed)}): {fs}")
    elif rec.get("failed_orders_section_present"):
        lines.append("- Failed/unconfirmed orders: none")
    else:
        lines.append(
            "- Failed/unconfirmed orders: none reported — this template omits "
            "the section when there are none, so this is the report's way of "
            "saying zero, not an independent verification"
        )

    if rec.get("curve_window_caveat"):
        lines.append(f"- ℹ️ {rec['curve_window_caveat']}")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
