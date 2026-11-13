"""``quantbox compare A B --rtol`` (TOM-3764).

"Bit for bit" and "97 of 97 values identical" were claims nobody could check; of
H58's 75 numeric analysis leaves, 37 differed at up to 9.3e-14. The verb measures
it: every number, how many differ, the largest relative difference and where.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from quantbox.cli import app
from quantbox.compare import SCHEMA_ID, compare, headline
from quantbox.finding_export import load_arms

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_run"
runner = CliRunner()


def _json_file(path: Path, doc) -> Path:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


A = {"leg1": {"t": 4.4244, "p": [0.04, 0.0]}, "n": 660, "label": "1946-2000", "ok": True}


def test_two_json_files_are_compared_leaf_by_leaf(tmp_path):
    b = json.loads(json.dumps(A))
    b["leg1"]["t"] = 4.4244 * (1 + 9.3e-14)
    out = compare(_json_file(tmp_path / "a.json", A), _json_file(tmp_path / "b.json", b))
    assert out["schema"] == SCHEMA_ID and out["kind"] == "json"
    assert (out["n_values"], out["n_differ"], out["max_rel_diff_at"]) == (4, 1, "leg1.t")
    assert out["max_rel_diff"] == pytest.approx(9.3e-14, rel=1e-3)
    assert out["equal"] and out["mismatches"] == []
    assert headline(out) == "equal within rtol 1e-09: 4 numbers, 1 differ, max rel diff 9.3e-14 (leg1.t)"


def test_a_difference_above_rtol_is_not_equal(tmp_path):
    b = {**A, "n": 661}
    out = compare(_json_file(tmp_path / "a.json", A), _json_file(tmp_path / "b.json", b), rtol=1e-3)
    assert not out["equal"] and out["max_rel_diff"] == pytest.approx(1 / 661)
    assert headline(out).startswith("NOT equal within rtol 0.001: 4 numbers, 1 differ, max rel diff 0.0015 (n)")


def test_identical_files_have_no_difference(tmp_path):
    out = compare(_json_file(tmp_path / "a.json", A), _json_file(tmp_path / "b.json", A), rtol=0)
    assert out["equal"] and (out["n_differ"], out["max_rel_diff"], out["max_rel_diff_at"]) == (0, 0.0, None)
    assert headline(out) == "equal within rtol 0: 4 numbers, 0 differ, max rel diff 0"


@pytest.mark.parametrize(
    ("b", "mismatch"),
    [
        ({**A, "extra": 1}, "extra: only in B"),
        ({k: v for k, v in A.items() if k != "n"}, "n: only in A"),
        ({**A, "leg1": {"t": 4.4244, "p": [0.04]}}, "leg1.p: 2 items vs 1"),
        ({**A, "label": "1946-2001"}, "label: '1946-2000' vs '1946-2001'"),
        ({**A, "ok": 1}, "ok: True vs 1"),  # a bool is not the number 1
        ({**A, "n": None}, "n: 660 vs None"),
        ({**A, "n": float("nan")}, "n: 660.0 vs nan"),
        ({**A, "n": float("inf")}, "n: 660.0 vs inf"),
    ],
)
def test_a_structural_difference_is_never_equal_whatever_rtol(tmp_path, b, mismatch):
    out = compare(_json_file(tmp_path / "a.json", A), _json_file(tmp_path / "b.json", b), rtol=1.0)
    assert not out["equal"] and out["mismatches"] == [mismatch]


def test_nan_against_nan_and_the_same_infinity_are_equal(tmp_path):
    doc = {"x": float("nan"), "y": float("inf"), "z": 0.0}
    out = compare(_json_file(tmp_path / "a.json", doc), _json_file(tmp_path / "b.json", doc), rtol=0)
    assert out["equal"] and out["n_values"] == 3 and out["n_differ"] == 0


def test_two_runs_compare_their_metrics_and_returns_by_date(tmp_path):
    same = compare(GOLDEN, shutil.copytree(GOLDEN, tmp_path / "copy"), rtol=0)
    (arm,) = load_arms(GOLDEN)  # the metrics as the export serves them (both drawdown names)
    returns = pd.read_parquet(GOLDEN / "returns.parquet")
    n_metrics = sum(1 for v in arm.metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
    assert same["kind"] == "runs" and same["equal"]
    assert same["n_values"] == n_metrics + len(returns) and same["n_differ"] == 0

    moved = shutil.copytree(GOLDEN, tmp_path / "moved")
    changed = returns.copy()
    changed.loc[10, "returns"] = changed.loc[10, "returns"] * (1 + 1e-6)
    changed.to_parquet(moved / "returns.parquet", index=False)
    out = compare(GOLDEN, moved, rtol=1e-9)
    when = pd.Timestamp(returns.loc[10, "date"]).isoformat()
    assert not out["equal"] and out["n_differ"] == 1
    assert out["max_rel_diff_at"] == f"Strategy.returns[{when}]"
    assert out["max_rel_diff"] == pytest.approx(1e-6, rel=1e-3)


def test_runs_whose_dates_differ_are_a_structural_difference(tmp_path):
    short = shutil.copytree(GOLDEN, tmp_path / "short")
    pd.read_parquet(GOLDEN / "returns.parquet").iloc[:-1].to_parquet(short / "returns.parquet", index=False)
    out = compare(GOLDEN, short)
    assert not out["equal"]
    assert any(m.startswith("Strategy.returns: dates differ") and "1 only in A" in m for m in out["mismatches"])


def test_two_arms_directories_pair_their_arms_by_name(tmp_path):
    for side in ("a", "b"):
        shutil.copytree(GOLDEN, tmp_path / side / "dp-timing")
    shutil.copytree(GOLDEN, tmp_path / "b" / "buyhold")
    out = compare(tmp_path / "a", tmp_path / "b")
    assert out["mismatches"] == ["arm 'buyhold': only in B"] and not out["equal"]


@pytest.mark.parametrize("rtol", [-1.0, math.nan, math.inf])
def test_a_bad_rtol_is_refused(tmp_path, rtol):
    with pytest.raises(ValueError, match="rtol"):
        compare(GOLDEN, GOLDEN, rtol=rtol)


def test_a_file_and_a_directory_are_refused(tmp_path):
    with pytest.raises(ValueError, match="not the same kind"):
        compare(_json_file(tmp_path / "a.json", A), GOLDEN)


# -- the CLI: exit 0 equal, 1 not, 2 refused ------------------------------------


def test_cli_prints_the_headline_and_exits_by_the_verdict(tmp_path):
    a = _json_file(tmp_path / "a.json", A)
    b = _json_file(tmp_path / "b.json", {**A, "n": 660 * (1 + 1e-10)})
    res = runner.invoke(app, ["compare", str(a), str(b)])
    assert res.exit_code == 0, res.output
    assert res.stdout.startswith("equal within rtol 1e-09: 4 numbers, 1 differ, max rel diff 1e-10 (n)")
    res = runner.invoke(app, ["compare", str(a), str(b), "--rtol", "1e-12"])
    assert res.exit_code == 1 and res.stdout.startswith("NOT equal within rtol 1e-12")


def test_cli_json_is_the_verdict(tmp_path):
    a = _json_file(tmp_path / "a.json", A)
    res = runner.invoke(app, ["compare", str(a), str(a), "--json"])
    assert res.exit_code == 0
    assert json.loads(res.stdout) == compare(a, a)


def test_cli_lists_the_structural_differences(tmp_path):
    a = _json_file(tmp_path / "a.json", A)
    b = _json_file(tmp_path / "b.json", {**A, "label": "x"})
    res = runner.invoke(app, ["compare", str(a), str(b)])
    assert res.exit_code == 1 and "  label: '1946-2000' vs 'x'" in res.stdout


@pytest.mark.parametrize("b", ["missing.json", "not.json"])
def test_cli_refuses_an_input_it_cannot_read(tmp_path, b):
    (tmp_path / "not.json").write_text("{nope", encoding="utf-8")
    a = _json_file(tmp_path / "a.json", A)
    res = runner.invoke(app, ["compare", str(a), str(tmp_path / b)])
    assert res.exit_code == 2 and "ERROR:" in res.stderr


def test_return_dates_are_labelled_only_where_a_difference_is_reported(tmp_path, monkeypatch):
    # The reviewer's nit on TOM-3764: one isoformat per return date of every arm, even
    # when nothing differs. Only the date a difference is reported at is formatted now.
    calls: list[pd.Timestamp] = []
    isoformat = pd.Timestamp.isoformat

    def counted(self, *args, **kwargs):
        calls.append(self)
        return isoformat(self, *args, **kwargs)

    monkeypatch.setattr(pd.Timestamp, "isoformat", counted)
    assert compare(GOLDEN, shutil.copytree(GOLDEN, tmp_path / "copy"), rtol=0)["equal"]
    assert calls == []

    moved = shutil.copytree(GOLDEN, tmp_path / "moved")
    returns = pd.read_parquet(GOLDEN / "returns.parquet")
    returns.loc[10, "returns"] = returns.loc[10, "returns"] * (1 + 1e-6)
    returns.to_parquet(moved / "returns.parquet", index=False)
    out = compare(GOLDEN, moved, rtol=1e-9)
    assert out["max_rel_diff_at"] == f"Strategy.returns[{isoformat(pd.Timestamp(returns.loc[10, 'date']))}]"
    assert 1 <= len(calls) <= 2
