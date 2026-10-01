"""``quantbox report export --format qute-research/finding-report@1`` (TOM-1365).

quantbox emits the DATA the qute-research finding page renders; the plugin owns
the page. The export reads a run directory (or a directory of arms, one run each)
through its ``quantbox/run@1`` manifest only, and every run writes the export as
its slim default report — the heavy ``report.html`` / ``report_data.json`` pair
is opt-in (``full_report: true``).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from quantbox.cli import app
from quantbox.finding_export import SCHEMA_ID, export_finding_report
from quantbox.registry import PluginRegistry
from quantbox.runner import run_from_config

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_run"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _walk(obj, path="$"):
    """Yield (path, value) for every leaf of a JSON-shaped object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def _containers(obj, path="$"):
    if isinstance(obj, (dict, list)):
        yield path, obj
        items = obj.items() if isinstance(obj, dict) else enumerate(obj)
        for k, v in items:
            yield from _containers(v, f"{path}.{k}" if isinstance(obj, dict) else f"{path}[{k}]")


def _inputs(root: Path, n: int = 400) -> Path:
    idx = pd.date_range("2023-06-01", periods=n, freq="D")
    rng = np.random.default_rng(11)
    prices = pd.DataFrame(
        {
            "A": 100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n)),
            "B": 50.0 * np.cumprod(1 + rng.normal(0.0, 0.02, n)),
        },
        index=idx,
    )
    path = root / "prices.parquet"
    prices.rename_axis("date").reset_index().melt("date", var_name="symbol", value_name="close").to_parquet(
        path, index=False
    )
    return path


def _run(tmp_path: Path, params: dict, strategies: list | None = None) -> Path:
    prices_path = _inputs(tmp_path)
    cfg = {
        "run": {"mode": "backtest", "asof": "2024-07-01", "pipeline": "backtest.pipeline.v1"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "plugins": {
            "pipeline": {
                "name": "backtest.pipeline.v1",
                "params": {"engine": "vectorbt", "fees": 0.0, "universe": {"symbols": ["A", "B"]}, **params},
            },
            "strategies": strategies
            if strategies is not None
            else [{"name": "strategy.static_weights.v1", "weight": 1.0, "params_init": {"weights": {"A": 1.0}}}],
            "data": {"name": "local_file_data", "params_init": {"prices_path": str(prices_path)}},
        },
    }
    result = run_from_config(cfg, PluginRegistry.discover())
    return tmp_path / "artifacts" / result.run_id


def _variants_run(tmp_path: Path) -> Path:
    variants = [
        {"name": "all_a", "strategy": {"name": "strategy.static_weights.v1", "params_init": {"weights": {"A": 1.0}}}},
        {
            "name": "benchmark_5050",
            "strategy": {"name": "strategy.static_weights.v1", "params_init": {"weights": {"A": 0.5, "B": 0.5}}},
        },
    ]
    return _run(tmp_path, {"variants": variants}, strategies=[])


# ----------------------------------------------------------------------
# the payload, from a single run (the golden run@1 directory)
# ----------------------------------------------------------------------


def test_export_of_a_run_carries_its_returns_metrics_and_provenance():
    payload = export_finding_report(GOLDEN)
    manifest = json.loads((GOLDEN / "run_manifest.json").read_text())
    metrics = json.loads((GOLDEN / "metrics.json").read_text())
    returns = pd.read_parquet(GOLDEN / "returns.parquet")

    assert payload["schema"] == SCHEMA_ID == "qute-research/finding-report@1"
    assert payload["run_id"] == manifest["run_id"]

    # series: growth of 1 from the run's own returns, one point per return
    series = payload["series"]
    assert len(series["dates"]) == len(returns)
    assert series["dates"] == sorted(set(series["dates"]))
    (line,) = series["lines"]
    expected = (1 + returns["returns"].fillna(0)).cumprod()
    assert line["equity"] == pytest.approx(expected.tolist(), abs=1e-6)
    assert len(line["drawdown"]) == len(series["dates"])
    assert min(line["drawdown"]) == pytest.approx(min(expected / expected.cummax() - 1), abs=1e-6)

    # the hero Sharpe reports the finding's gate metric, with the engine's number
    sharpe = next(k for k in payload["kpis"] if k.get("key") == "backtest_sharpe")
    assert sharpe["value"] == pytest.approx(metrics["sharpe"])

    # metrics table: one column per arm, the engine's numbers
    table = payload["metrics"]
    assert table["columns"] == [line["name"]]
    cagr = next(r for r in table["rows"] if r["label"] == "CAGR")
    assert cagr["values"] == [pytest.approx(metrics["cagr"])] and cagr["format"] == "pct"

    # provenance from the manifest — engine, execution, dataset, funding
    prov = next(t for t in payload["tables"] if t["title"] == "Run provenance")
    rows = {r[0]: r[1:] for r in prov["rows"]}
    assert rows["engine"] == ["rsims 0.7.0"]
    assert rows["execution.lag_bars"] == [1]
    assert rows["dataset.sha256"] == [manifest["dataset"]["sha256"]]
    assert rows["funding.modelled"] == ["true"]
    assert rows["n_trials"] == [3]
    assert rows["config.sha256"] == [manifest["config"]["sha256"]]


def test_export_obeys_the_renderers_strict_rules():
    """What the plugin refuses, the export never ships: booleans as values,
    empty sections, NaN/Infinity (strict JSON), statuses outside the set."""
    payload = export_finding_report(GOLDEN)
    json.dumps(payload, allow_nan=False)  # strict JSON or it raises
    for path, value in _walk(payload):
        if path.endswith(".benchmark"):
            continue  # series.lines[].benchmark is the one boolean field
        assert not isinstance(value, bool), f"{path}: a boolean is refused by the renderer"
    for path, value in _containers(payload):
        assert value, f"{path}: an empty section is refused by the renderer"
    for cell in payload["robustness"]["cells"]:
        assert cell["status"] in {"pass", "fail", "na"}


def test_robustness_is_a_grid_of_arms_by_calendar_year(tmp_path):
    run_dir = _variants_run(tmp_path)
    payload = export_finding_report(run_dir)
    grid = payload["robustness"]
    assert grid["rows"] == ["all_a", "benchmark_5050"]
    assert grid["cols"] == ["2023", "2024"]
    returns = pd.read_parquet(run_dir / "variant_returns.parquet").set_index("date")
    cell = next(c for c in grid["cells"] if c["row"] == "all_a" and c["col"] == "2024")
    r = returns["all_a"]
    expected = float((1 + r[r.index.year == 2024].fillna(0)).prod() - 1)
    assert cell["value"] == pytest.approx(expected)
    assert cell["status"] == ("pass" if expected > 0 else "fail")


# ----------------------------------------------------------------------
# arms: a variants run, and a directory of runs
# ----------------------------------------------------------------------


def test_a_variants_run_exports_one_arm_per_variant(tmp_path):
    run_dir = _variants_run(tmp_path)
    assert (run_dir / "variant_returns.parquet").exists()
    payload = export_finding_report(run_dir)
    lines = {ln["name"]: ln for ln in payload["series"]["lines"]}
    assert set(lines) == {"all_a", "benchmark_5050"}
    assert lines["benchmark_5050"]["benchmark"] is True
    assert lines["all_a"]["benchmark"] is False
    assert payload["metrics"]["columns"] == ["all_a", "benchmark_5050"]
    vm = pd.read_parquet(run_dir / "variant_metrics.parquet").set_index("variant")
    sharpe = next(r for r in payload["metrics"]["rows"] if r["label"] == "Sharpe")
    assert sharpe["values"] == [
        pytest.approx(vm.loc["all_a", "sharpe"]),
        pytest.approx(vm.loc["benchmark_5050", "sharpe"]),
    ]


def test_a_directory_of_runs_is_one_arm_per_run(tmp_path):
    arms = tmp_path / "arms"
    shutil.copytree(GOLDEN, arms / "arm_a")
    nested = arms / "arm_b" / "2024-02-09__older"
    shutil.copytree(GOLDEN, nested)
    newer = arms / "arm_b" / "2024-02-09__newer"
    shutil.copytree(GOLDEN, newer)
    (arms / "not_a_run").mkdir()

    payload = export_finding_report(arms)
    assert [ln["name"] for ln in payload["series"]["lines"]] == ["arm_a", "arm_b"]
    assert payload["metrics"]["columns"] == ["arm_a", "arm_b"]
    prov = next(t for t in payload["tables"] if t["title"] == "Run provenance")
    assert prov["columns"] == ["field", "arm_a", "arm_b"]
    assert "run_id" not in payload  # an arms export is not one run


def test_primary_picks_the_arm_the_hero_cards_report(tmp_path):
    run_dir = _variants_run(tmp_path)
    vm = pd.read_parquet(run_dir / "variant_metrics.parquet").set_index("variant")
    payload = export_finding_report(run_dir, primary="benchmark_5050")
    sharpe = next(k for k in payload["kpis"] if k.get("key") == "backtest_sharpe")
    assert sharpe["value"] == pytest.approx(vm.loc["benchmark_5050", "sharpe"])
    with pytest.raises(ValueError, match="no arm named"):
        export_finding_report(run_dir, primary="nope")


def test_a_directory_without_runs_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="no run"):
        export_finding_report(tmp_path)


# ----------------------------------------------------------------------
# the slim default report; the full pair is opt-in
# ----------------------------------------------------------------------


def test_a_run_writes_the_slim_report_by_default_and_not_the_heavy_pair(tmp_path):
    run_dir = _run(tmp_path, {})
    assert not (run_dir / "report.html").exists()
    assert not (run_dir / "report_data.json").exists()
    written = json.loads((run_dir / "finding_report.json").read_text())
    assert written == export_finding_report(run_dir)
    total = sum(p.stat().st_size for p in run_dir.iterdir())
    assert total < 2 * 1024 * 1024


def test_full_report_true_also_writes_the_heavy_pair(tmp_path):
    run_dir = _run(tmp_path, {"full_report": True})
    assert (run_dir / "report.html").exists()
    assert (run_dir / "report_data.json").exists()
    assert (run_dir / "finding_report.json").exists()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def test_cli_report_export_writes_the_payload(tmp_path):
    out = tmp_path / "finding.json"
    res = CliRunner().invoke(app, ["report", "export", str(GOLDEN), "--format", SCHEMA_ID, "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert json.loads(out.read_text()) == export_finding_report(GOLDEN)


def test_cli_report_export_prints_to_stdout_without_out():
    res = CliRunner().invoke(app, ["report", "export", str(GOLDEN), "--format", SCHEMA_ID])
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["schema"] == SCHEMA_ID


def test_cli_report_export_refuses_an_unknown_format():
    res = CliRunner().invoke(app, ["report", "export", str(GOLDEN), "--format", "qute-research/finding-report@2"])
    assert res.exit_code == 2


def test_cli_report_export_exits_1_on_a_directory_without_runs(tmp_path):
    res = CliRunner().invoke(app, ["report", "export", str(tmp_path), "--format", SCHEMA_ID])
    assert res.exit_code == 1


# ----------------------------------------------------------------------
# the plugin renders it — no lab-side adapter
# ----------------------------------------------------------------------


def _renderer() -> Path | None:
    """The qute-research renderer: $QUTE_RESEARCH_ROOT, else a qute-plugins checkout beside an ancestor."""
    rel = Path("scripts") / "finding_report.py"
    env = os.environ.get("QUTE_RESEARCH_ROOT")
    if env:
        return Path(env) / rel
    for parent in Path(__file__).resolve().parents:
        cand = parent / "qute-plugins" / "plugins" / "qute-research" / rel
        if cand.is_file():
            return cand
    return None


def _finding(path: Path, sharpe: float) -> Path:
    path.write_text(
        "---\n"
        "line: export-check\n"
        "date: 2026-10-01\n"
        "verdict: inconclusive\n"
        f"backtest_sharpe: {sharpe:.2f}\n"
        "review:\n  status: pending\n"
        "---\n\n# Export check\n\n## Result\n\nRendered from a quantbox export.\n\n## Engine gap\n\nnone\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("source", ["golden", "variants"])
def test_the_qute_research_renderer_accepts_the_export(tmp_path, source):
    script = _renderer()
    if script is None or not script.is_file():
        pytest.skip("qute-research renderer not found (set QUTE_RESEARCH_ROOT) — NOT CHECKED")
    run_dir = GOLDEN if source == "golden" else _variants_run(tmp_path)
    data = tmp_path / "finding_report.json"
    payload = export_finding_report(run_dir)
    data.write_text(json.dumps(payload), encoding="utf-8")
    sharpe = next(k for k in payload["kpis"] if k.get("key") == "backtest_sharpe")["value"]
    assert math.isfinite(sharpe)
    finding = _finding(tmp_path / "2026-10-01-inconclusive-export-check.md", sharpe)
    page = tmp_path / "finding.html"
    proc = subprocess.run(
        [sys.executable, str(script), str(finding), "--data", str(data), "--out", str(page)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    html = page.read_text(encoding="utf-8")
    assert 'id="robustness"' in html and 'id="provenance"' in html
