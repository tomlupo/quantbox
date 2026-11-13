"""The report names its numbers honestly (TOM-3764).

H58's page printed ``total_return`` 303.58 as a bare number a reader took for a
303.6x multiple (it is +30358 %, 304.58x wealth), and a gate row
``0.829 | >= 0.95 | na`` that reads as a pass. The export now labels the return
with its wealth multiple, derives a gate's status from its value and comparator
(never the results file's word), and fills a fair-baseline table from
``arms_summary.json``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from quantbox.finding_export import _checked_gate, export_finding_report
from quantbox.metrics import total_return_label
from quantbox.plugins.pipeline._report import generate_summary_md

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_run"


# -- total return with its wealth multiple -------------------------------------


@pytest.mark.parametrize(
    ("total_return", "label"),
    [
        (303.579388075131, "30358% (304.58x wealth)"),  # H58's dp-timing arm
        (333.810998238666, "33381% (334.81x wealth)"),  # H58's buy-and-hold arm
        (1.0, "100% (2.00x wealth)"),
        (0.25, "25.00% (1.25x wealth)"),
        (-0.4, "-40.00% (0.60x wealth)"),
        (0.0, "0.00% (1.00x wealth)"),
        (float("nan"), "n/a"),
    ],
)
def test_a_total_return_is_printed_with_its_wealth_multiple(total_return, label):
    assert total_return_label(total_return) == label


def test_the_metrics_table_labels_total_return_and_leaves_every_other_row_as_numbers():
    metrics = json.loads((GOLDEN / "metrics.json").read_text())
    rows = {r["label"]: r for r in export_finding_report(GOLDEN)["metrics"]["rows"]}
    assert rows["Total return"] == {
        "label": "Total return",
        "format": "text",
        "values": [total_return_label(metrics["total_return"])],
    }
    assert rows["CAGR"] == {"label": "CAGR", "format": "pct", "values": [pytest.approx(metrics["cagr"])]}


def test_summary_md_prints_total_return_with_its_wealth_multiple():
    md = generate_summary_md("r", "2024-10-31", {"total_return": 303.579388075131}, [], "1966-01-31", "2024-10-31")
    assert "| Total return | 30358% (304.58x wealth) |" in md
    assert "30357.9%" not in md


# -- a gate's status comes from its value and comparator -----------------------


def _gate(value, threshold, status="na", note="the line's note"):
    return {"name": "g", "value": value, "threshold": threshold, "status": status, "note": note}


@pytest.mark.parametrize(
    ("value", "threshold", "status", "shown"),
    [
        (0.829, ">= 0.95", "fail", "< 0.95"),  # H58's DSR row
        (3.37, "> 2.0", "pass", "> 2.0"),  # H58's NW row
        (4.42, ">= 1.645", "pass", ">= 1.645"),
        (0.771, "> 0.10", "pass", "> 0.10"),  # the bar as written, not 0.1
        (0.95, ">= 0.95", "pass", ">= 0.95"),  # equality: >= holds
        (0.95, "> 0.95", "fail", "<= 0.95"),  # equality: > does not
        (0.04, "<= 0.05", "pass", "<= 0.05"),
        (0.06, "<= 0.05", "fail", "> 0.05"),
        (0.06, "< 0.05", "fail", ">= 0.05"),
        (-0.3, ">= -0.25", "fail", "< -0.25"),  # a negative bar keeps its sign
        (0.96, "≥ 0.95", "pass", ">= 0.95"),
        (1e-3, "<= 5e-2", "pass", "<= 5e-2"),
        ("0.829", ">= 0.95", "fail", "< 0.95"),  # a numeric string is a number
    ],
)
def test_a_one_comparator_gate_is_checked_and_shows_the_comparator_that_holds(value, threshold, status, shown):
    out = _checked_gate(_gate(value, threshold))
    assert (out["status"], out["threshold"]) == (status, shown)
    assert out["value"] == value  # the number is never touched


def test_a_failed_gate_keeps_its_bar_and_what_the_line_wrote_in_the_note():
    out = _checked_gate(_gate(0.829, ">= 0.95", status="na", note="Tests a positive Sharpe."))
    assert out["note"] == "needs >= 0.95; the line wrote 'na'. Tests a positive Sharpe."


def test_a_gate_whose_status_agrees_keeps_its_note_unchanged():
    gate = _gate(4.42, ">= 1.645", status="PASS", note="Paper 4.67.")
    assert _checked_gate(gate) == {**gate, "status": "pass"}


@pytest.mark.parametrize(
    ("value", "threshold", "why"),
    [
        (0.009, "< 0.10 and P < 0.90", "is not one comparator"),
        (0.5, 0.95, "is not one comparator"),  # a bare number names no direction
        (None, ">= 0.95", "is not a number"),
        ("n/a", ">= 0.95", "is not a number"),
        (True, ">= 0.5", "is not a number"),
    ],
)
def test_a_gate_quantbox_cannot_check_keeps_the_lines_status_and_says_so(value, threshold, why):
    out = _checked_gate(_gate(value, threshold, status="pass", note="P = 0.00."))
    assert out["status"] == "pass" and out["threshold"] == threshold
    assert out["note"].startswith("status as the line wrote it (")
    assert why in out["note"] and out["note"].endswith("not checked). P = 0.00.")


def test_the_export_checks_the_gates_of_a_line_results_file(tmp_path):
    extra = tmp_path / "results.json"
    gates = [_gate(0.829, ">= 0.95"), _gate(0.009, "< 0.10 and P < 0.90", status="pass")]
    extra.write_text(json.dumps({"audit": {"gates": gates}}), encoding="utf-8")
    out = export_finding_report(GOLDEN, extra=extra)["audit"]["gates"]
    assert [(g["status"], g["threshold"]) for g in out] == [("fail", "< 0.95"), ("pass", "< 0.10 and P < 0.90")]
    assert json.loads(extra.read_text())["audit"]["gates"] == gates  # the file is read, never changed


# -- the fair baseline, from arms_summary.json ---------------------------------


def _arms_batch(tmp_path: Path, names: list[str], *, summary: bool = True) -> Path:
    """A ``quantbox arms`` batch: one run per arm and the arms_summary.json it writes."""
    batch = tmp_path / "arms__20261113T170702317939Z"
    metrics = json.loads((GOLDEN / "metrics.json").read_text())
    arms = []
    for i, name in enumerate(names):
        shutil.copytree(GOLDEN, batch / name / "2024-02-09__backtest_pipeline_v1__aaaa__20260101T000000Z")
        arms.append(
            {"name": name, "status": "ok", "metrics": {**metrics, "sharpe": 0.5 + i / 10, "total_return": 300.0 + i}}
        )
    if summary:
        doc = {"schema": "quantbox/arms@1", "batch_id": batch.name, "arms": arms}
        (batch / "arms_summary.json").write_text(json.dumps(doc), encoding="utf-8")
    return batch


def _baseline(payload: dict) -> dict | None:
    return next((t for t in payload["tables"] if t["title"].startswith("Fair baseline")), None)


def test_a_buy_and_hold_arm_fills_the_fair_baseline_table(tmp_path):
    payload = export_finding_report(_arms_batch(tmp_path, ["dp-timing", "buyhold"]))
    table = _baseline(payload)
    assert table["title"] == "Fair baseline: buyhold"
    assert payload["tables"][0] is table  # first: the fair comparison before the line's tables
    assert table["columns"] == ["arm", "role", "Sharpe", "CAGR", "Max drawdown", "Total return"]
    (cand, base) = table["rows"]
    assert cand[:3] == ["dp-timing", "candidate", 0.5]
    assert base[:3] == ["buyhold", "fair baseline", 0.6]
    assert base[5] == "30100% (302.00x wealth)"
    assert "arms__20261113T170702317939Z/arms_summary.json" in table["note"]


def test_benchmark_names_the_fair_baseline(tmp_path):
    table = _baseline(export_finding_report(_arms_batch(tmp_path, ["a", "b"]), benchmark="a"))
    assert [r[:2] for r in table["rows"]] == [["a", "fair baseline"], ["b", "candidate"]]


@pytest.mark.parametrize(
    ("names", "summary"),
    [
        (["a", "b"], True),  # no arm is a baseline by name: no guess
        (["bench_a", "buy-and-hold", "c"], True),  # two baselines by name: no guess
        (["dp-timing", "buyhold"], False),  # not an arms batch
    ],
)
def test_there_is_no_fair_baseline_table_rather_than_a_guess(tmp_path, names, summary):
    assert _baseline(export_finding_report(_arms_batch(tmp_path, names, summary=summary))) is None
