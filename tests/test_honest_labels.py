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

import numpy as np
import pandas as pd
import pytest

from quantbox.finding_export import _checked_gate, export_finding_report
from quantbox.metrics import total_return_label
from quantbox.plugins.pipeline._report import _TEMPLATE_PATH, generate_report_data, generate_summary_md

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


def test_the_metrics_table_keeps_total_return_a_number_and_puts_the_wealth_multiple_beside_it():
    # The card's "numbers unchanged": finding_report.json keeps total_return as the number
    # it was (format pct), and the wealth multiple is a NEW row next to it, a number too.
    # Every label names the bar it was sampled at (TOM-3718).
    metrics = json.loads((GOLDEN / "metrics.json").read_text())
    rows = export_finding_report(GOLDEN)["metrics"]["rows"]
    assert rows[0] == {"label": "Total return (daily bars)", "format": "pct", "values": [metrics["total_return"]]}
    assert rows[1] == {
        "label": "Wealth multiple (1 + total return, daily bars)",
        "format": "num",
        "values": [pytest.approx(1 + metrics["total_return"])],
    }
    assert rows[2] == {"label": "CAGR (daily bars)", "format": "pct", "values": [pytest.approx(metrics["cagr"])]}
    assert all(not isinstance(v, str) for r in rows for v in r["values"])


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


def _with_variants(run_dir: Path, variants: list[str]) -> None:
    """Turn a copied golden run into a run with *variants* (the files ``_run_arms`` reads)."""
    returns = pd.read_parquet(run_dir / "returns.parquet")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    long = pd.concat([returns.assign(variant=v) for v in variants], ignore_index=True)
    long[["date", "variant", "returns"]].to_parquet(run_dir / "variant_returns.parquet")
    pd.DataFrame([{"variant": v, **metrics} for v in variants]).to_parquet(run_dir / "variant_metrics.parquet")


def test_a_benchmark_variant_names_its_arm_as_the_fair_baseline(tmp_path):
    # --benchmark is checked against load_arms' names, "arm/variant" for a run with
    # variants; arms_summary.json names the arm. The table used to vanish silently.
    batch = _arms_batch(tmp_path, ["dp-timing", "buyhold"])
    for arm in ("dp-timing", "buyhold"):
        (run,) = (batch / arm).iterdir()
        _with_variants(run, ["main", "benchmark_5050"])
    table = _baseline(export_finding_report(batch, benchmark="buyhold/benchmark_5050"))
    assert table is not None
    assert table["title"] == "Fair baseline: buyhold"
    assert [r[:2] for r in table["rows"]] == [["dp-timing", "candidate"], ["buyhold", "fair baseline"]]


@pytest.mark.parametrize("name", ["spy-bench", "benchmark", "buyhold", "Buy-and-Hold"])
def test_the_arm_drawn_as_the_benchmark_is_the_fair_baseline(tmp_path, name):
    # One rule for one concept: the series' dashed benchmark line and the baseline table.
    payload = export_finding_report(_arms_batch(tmp_path, ["dp-timing", name]))
    assert {ln["name"]: ln["benchmark"] for ln in payload["series"]["lines"]} == {"dp-timing": False, name: True}
    assert _baseline(payload)["title"] == f"Fair baseline: {name}"


# -- the HTML report strip prints Python's label, not a copy of its rule -------


def _report_kwargs(variant_results: dict) -> dict:
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    rets = pd.Series(np.linspace(-0.01, 0.02, len(dates)), index=dates)
    nav = (1 + rets).cumprod()
    primary = next(iter(variant_results.values()), None)
    return dict(
        run_id="r",
        asof="2020-03-01",
        metrics=primary["metrics"] if primary else {"total_return": 303.579388075131},
        portfolio_daily=nav.rename("portfolio_value").rename_axis("date").reset_index(),
        returns=rets,
        weights_history=pd.DataFrame(1.0, index=dates, columns=["A"]),
        bt_prices=pd.DataFrame(100 * nav.to_numpy(), index=dates, columns=["A"]),
        strategy_names=list(variant_results) or ["s"],
        period_start="2020-01-01",
        period_end="2020-02-29",
        variant_results=variant_results or None,
    )


def test_the_html_strip_gets_total_return_label_from_python():
    kwargs = _report_kwargs({})
    metrics = kwargs["metrics"]
    rd = generate_report_data(**kwargs)
    assert rd["metrics"]["_total_return_label"] == "30358% (304.58x wealth)"
    assert "_total_return_label" not in metrics  # the run's own metrics are not written to
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    assert "metricsObj._total_return_label" in template
    assert "x wealth" not in template  # no second copy of total_return_label's rule to drift


def test_each_variant_strip_gets_its_own_total_return_label():
    k = _report_kwargs({})
    variant = {
        "returns": k["returns"],
        "portfolio_daily": k["portfolio_daily"],
        "weights_history": k["weights_history"],
        "bt_prices": k["bt_prices"],
        "strategy_details": {},
        "config": {},
    }
    vr = {
        "a": {**variant, "name": "a", "metrics": {"total_return": -0.4}},
        "b": {**variant, "name": "b", "metrics": {"total_return": 1.0}},
    }
    rd = generate_report_data(**_report_kwargs(vr))
    assert {v: m["_total_return_label"] for v, m in rd["variant_metrics"].items()} == {
        "a": "-40.00% (0.60x wealth)",
        "b": "100% (2.00x wealth)",
    }
