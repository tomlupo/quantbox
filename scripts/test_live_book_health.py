"""Tests for live-book-health.py: the over-limit history and the last-fill search.

Stdlib only, no prod files: every report and config is written into a temp dir
and the module's LIVE paths are pointed at it. Every end-to-end read first
asserts those paths sit UNDER the sandbox, and the fixture book id (`bk`) does
not exist in prod, so a failed redirect cannot pass by reading the real books.

Run: `python3 -m unittest test_live_book_health.py` (pytest collects it too).
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "live_book_health", HERE / "live-book-health.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lbh = _load()


def report_text(day: date, curve: list[tuple[str, float]], fills: int) -> str:
    """The RENDERING only. The equity series is no longer scraped from here --
    it comes from `performance.json`, which `write_performance` creates beside
    the report. The prose keeps the modern row name (`Portfolio value (run
    end)`); `report_text_legacy_rows` covers the old one, which still has to
    parse because historical reports on disk carry it."""
    return (
        f"# Daily report\n\n**Generated:** {day.isoformat()} 04:04 UTC\n\n"
        "## Portfolio Summary\n\n| Metric | Value |\n|---|---|\n"
        f"| Portfolio value (run end) | ${curve[-1][1]:,.2f} |\n"
        f"| Fills | {fills} |\n"
        "| Max Drawdown | -12.00% |\n"
    )


def report_text_legacy_rows(day: date, curve: list[tuple[str, float]], fills: int) -> str:
    """A pre-2026-09-23 report, carrying `Portfolio (post)`."""
    return report_text(day, curve, fills).replace(
        "Portfolio value (run end)", "Portfolio (post)"
    )


def write_performance(day_dir: Path, curve: list[tuple[str, float]]) -> None:
    """The DATA the health check reads, written where the run writes it.

    The fixture CREATES what the code under test consumes. It used to encode the
    series as ASCII bars inside the report and let the parser scrape them back
    out -- so when the report stopped carrying the chart, these tests went on
    passing against a shape production no longer produced.
    """
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / "performance.json").write_text(
        json.dumps({"equity_curve": [{"date": d, "equity": v} for d, v in curve]})
    )


def curve_from(start: date, values: list[float]) -> list[tuple[str, float]]:
    return [((start + timedelta(days=i)).isoformat(), v) for i, v in enumerate(values)]


# limit 10%. Day: 0    1    2     3     4    5    6     7
# value           100  95   85    88    96   110  98    97
# dd from peak    0    5    15!   12!   4    0    10.9! 11.8!
BREACH_RECOVER_BREACH = [100, 95, 85, 88, 96, 110, 98, 97]
START = date(2026, 3, 1)


class OverLimitHistory(unittest.TestCase):
    def test_breach_start_is_the_LAST_runs_start_not_the_first(self):
        curve = curve_from(START, BREACH_RECOVER_BREACH)
        report_date = START + timedelta(days=7)
        start, dates = lbh.over_limit_history(curve, 0.10, report_date)
        self.assertEqual(start, "2026-03-07")  # not 2026-03-03, the first run
        # 7-day window = 03-02..03-08: both runs' days that fall inside it
        self.assertEqual(
            dates, ["2026-03-03", "2026-03-04", "2026-03-07", "2026-03-08"]
        )

    def test_window_is_seven_calendar_days_inclusive_of_report_date(self):
        curve = curve_from(START, [100] + [80] * 9)  # over from 03-02 to 03-10
        start, dates = lbh.over_limit_history(curve, 0.10, date(2026, 3, 10))
        self.assertEqual(start, "2026-03-02")
        self.assertEqual(dates[0], "2026-03-04")
        self.assertEqual(len(dates), 7)

    def test_recovered_at_the_end_means_null_start_but_dates_still_listed(self):
        curve = curve_from(START, [100, 85, 99])
        start, dates = lbh.over_limit_history(curve, 0.10, date(2026, 3, 3))
        self.assertIsNone(start)
        self.assertEqual(dates, ["2026-03-02"])

    def test_never_breaches_is_null_and_EMPTY_LIST(self):
        curve = curve_from(START, [100, 99, 101, 95, 102])
        self.assertEqual(
            lbh.over_limit_history(curve, 0.10, date(2026, 3, 5)), (None, [])
        )

    def test_exactly_at_the_limit_is_not_over(self):
        curve = curve_from(START, [100, 90])
        self.assertEqual(
            lbh.over_limit_history(curve, 0.10, date(2026, 3, 2)), (None, [])
        )

    def test_unreadable_limit_or_curve_is_null_null_never_empty(self):
        curve = curve_from(START, [100, 50])
        self.assertEqual(lbh.over_limit_history(curve, None, START), (None, None))
        self.assertEqual(lbh.over_limit_history([], 0.10, START), (None, None))
        self.assertEqual(
            lbh.over_limit_history(curve_from(START, [0, -5]), 0.10, START),
            (None, None),
        )

    def test_nonpositive_point_neither_moves_the_peak_nor_breaks_a_run(self):
        curve = curve_from(START, [100, 80, 0, 80])
        start, dates = lbh.over_limit_history(curve, 0.10, date(2026, 3, 4))
        self.assertEqual(start, "2026-03-02")
        self.assertEqual(dates, ["2026-03-02", "2026-03-04"])


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self._saved = (lbh.CONFIGS_DIR, lbh.REPORTS_DIR)
        lbh.CONFIGS_DIR = self.root / "configs"
        lbh.REPORTS_DIR = self.root / "reports"
        self.addCleanup(self._restore)
        lbh.CONFIGS_DIR.mkdir()
        (lbh.REPORTS_DIR / "bk").mkdir(parents=True)

    def _restore(self):
        lbh.CONFIGS_DIR, lbh.REPORTS_DIR = self._saved

    def _config(self, body="trading_enabled: true\ndrawdown_from_peak_pct: 0.10\n"):
        (lbh.CONFIGS_DIR / "bk.yaml").write_text(body)

    def _report(self, day: date, values: list[float], fills: int, *, legacy=False):
        start = day - timedelta(days=len(values) - 1)
        curve = curve_from(start, values)
        render = report_text_legacy_rows if legacy else report_text
        book_dir = lbh.REPORTS_DIR / "bk"
        (book_dir / f"{day.isoformat()}.md").write_text(render(day, curve, fills))
        write_performance(book_dir / day.isoformat(), curve)

    def _health(self, today: date) -> dict:
        self.assertIn(self.root, lbh.REPORTS_DIR.parents)
        self.assertIn(self.root, lbh.CONFIGS_DIR.parents)
        return lbh.health_for_book("bk", "bk.yaml", "bk", today)

    def test_breaching_book_record_and_rendered_block(self):
        today = date(2026, 3, 8)
        self._config()
        self._report(today, BREACH_RECOVER_BREACH, fills=2)
        rec = self._health(today)
        self.assertEqual(rec["status"], "OK_NO_FAILED_ORDERS_SECTION")  # it PARSED
        self.assertTrue(rec["over_limit"])
        self.assertEqual(rec["breach_start"], "2026-03-07")
        self.assertEqual(
            rec["over_limit_dates"],
            ["2026-03-03", "2026-03-04", "2026-03-07", "2026-03-08"],
        )
        block = lbh.render_block(rec)
        self.assertIn("- Over limit since 2026-03-07 (day 2)", block)
        self.assertIn("- Last fill: 2026-03-08 (0 days ago, 2 filled)", block)
        self.assertNotIn("check execution", block)

    def test_clean_book_has_null_start_empty_dates_and_no_over_limit_line(self):
        today = date(2026, 3, 5)
        self._config()
        self._report(today, [100, 99, 101, 95, 102], fills=1)
        rec = self._health(today)
        self.assertEqual(rec["status"], "OK_NO_FAILED_ORDERS_SECTION")
        self.assertIs(rec["over_limit"], False)
        self.assertIsNone(rec["breach_start"])
        self.assertEqual(rec["over_limit_dates"], [])
        self.assertNotIn("Over limit since", lbh.render_block(rec))

    def test_unreadable_limit_yields_null_for_both_fields(self):
        today = date(2026, 3, 8)
        self._config("trading_enabled: true\n")
        self._report(today, BREACH_RECOVER_BREACH, fills=1)
        rec = self._health(today)
        self.assertIsNotNone(
            rec["equity"]
        )  # the curve WAS read; only the limit is blind
        self.assertIsNone(rec["breach_start"])
        self.assertIsNone(rec["over_limit_dates"])

    def test_missing_and_bad_source_records_carry_both_keys_as_null(self):
        self._config()
        missing = self._health(date(2026, 3, 8))
        self.assertEqual(missing["status"], "MISSING")
        (lbh.REPORTS_DIR / "bk" / "2026-03-08.md").write_text(
            "# x\n\n**Generated:** 2026-03-08 04:04 UTC\n"
        )
        bad = self._health(date(2026, 3, 8))
        self.assertEqual(bad["status"], "BAD_SOURCE")
        for rec in (missing, bad):
            self.assertIn("breach_start", rec)
            self.assertIn("over_limit_dates", rec)
            self.assertIsNone(rec["breach_start"])
            self.assertIsNone(rec["over_limit_dates"])

    def test_last_fill_older_than_14_days_keeps_its_DATE_and_advances(self):
        self._config()
        fill_day = date(2026, 3, 1)
        self._report(fill_day, [100, 101], fills=3)
        for offset, expect in ((20, 20), (21, 21)):
            today = fill_day + timedelta(days=offset)
            self._report(today, [100, 101], fills=0)
            rec = self._health(today)
            self.assertEqual(
                rec["last_fill"], {"date": "2026-03-01", "fills": 3, "days_ago": expect}
            )
            self.assertIn(
                f"- Last fill: 2026-03-01 ({expect} days ago, 3 filled) — ⚠️ check execution",
                lbh.render_block(rec),
            )

    def test_no_fill_anywhere_says_the_lookback_and_how_much_it_read(self):
        self._config()
        today = date(2026, 3, 8)
        self._report(today - timedelta(days=1), [100, 101], fills=0)
        self._report(today, [100, 101], fills=0)
        rec = self._health(today)
        self.assertIsNone(rec["last_fill"])
        self.assertEqual(rec["last_fill_search"]["prior_reports_read"], 1)
        self.assertIn(
            f"- Last fill: none in the last {lbh.LAST_FILL_LOOKBACK_DAYS}d "
            "(1 prior reports read) — ⚠️ check execution",
            lbh.render_block(rec),
        )


if __name__ == "__main__":
    unittest.main()


class TheSeriesComesFromDataNotFromTheRendering(EndToEnd):
    """The 2026-09-23 regression, pinned.

    The v0.7.0 report work replaced the full ASCII equity chart with a table of
    the last 14 marks and renamed a summary row. Both books then read
    BAD_SOURCE for two days and kpi-1 read BLIND, because the monitor's input
    was a RENDERING. These tests fail if anyone points it back at the report.
    """

    def test_a_truncated_table_in_the_report_does_not_shorten_the_series(self):
        """The one that would have caught it -- and the one that stops the
        tempting wrong fix.

        The report shows only the last 3 marks, as production's now does. If the
        peak were taken from what the report DISPLAYS, it would be 96 and the
        drawdown a comfortable 0%; the whole series peaks at 110, which is a
        real 11.8% breach of the 10% limit. Understating a drawdown is worse
        than failing to read one, so this asserts the breach is still seen.
        """
        today = date(2026, 3, 8)
        self._config()
        curve = curve_from(today - timedelta(days=7), BREACH_RECOVER_BREACH)
        book_dir = lbh.REPORTS_DIR / "bk"
        book_dir.mkdir(parents=True, exist_ok=True)
        visible = curve[-3:]
        (book_dir / f"{today.isoformat()}.md").write_text(
            report_text(today, curve, fills=2)
            + "\n**Recent equity** (last 3 of 8 marks):\n\n| Date | Equity |\n|---|---|\n"
            + "".join(f"| {d} | ${v:,.2f} |\n" for d, v in visible)
        )
        write_performance(book_dir / today.isoformat(), curve)

        rec = self._health(today)
        self.assertEqual(rec["equity_curve_points"], len(curve))  # 8, not 3
        self.assertEqual(rec["equity_peak"], 110)  # not 96
        self.assertTrue(rec["over_limit"])

    def test_a_missing_performance_json_is_loud_not_a_clean_book(self):
        """"Could not look" must not exit like "looked and it is fine"."""
        today = date(2026, 3, 8)
        self._config()
        curve = curve_from(today - timedelta(days=7), BREACH_RECOVER_BREACH)
        book_dir = lbh.REPORTS_DIR / "bk"
        book_dir.mkdir(parents=True, exist_ok=True)
        (book_dir / f"{today.isoformat()}.md").write_text(
            report_text(today, curve, fills=2)
        )
        # deliberately no write_performance()

        rec = self._health(today)
        self.assertEqual(rec["status"], "BAD_SOURCE")
        self.assertIn("equity curve", rec["missing_sections"])
        self.assertEqual(lbh.exit_code([rec]), lbh.EXIT_BLIND)

    def test_a_pre_rename_report_row_still_parses(self):
        """Historical reports carry `Portfolio (post)`. A parser that can only
        read today cannot answer a question about last month."""
        today = date(2026, 3, 8)
        self._config()
        self._report(today, BREACH_RECOVER_BREACH, fills=2, legacy=True)
        rec = self._health(today)
        self.assertEqual(rec["status"], "OK_NO_FAILED_ORDERS_SECTION")
        self.assertEqual(rec["portfolio_post_for_reference"], 97)
