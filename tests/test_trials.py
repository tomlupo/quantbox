"""The line's trial log, trials.jsonl (TOM-3929): the format, the count, the line, the DSR gate.

The format is the owner's and other repos read it (qute-research quant_preflight,
the lab's gauntlet script): one JSON object per line, n_trials = distinct
``config_sha256``, blank lines ignored, a line that is not JSON an error.
The end-to-end proof (40 real runs, then ``gates dsr --line``) is in test_arms.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quantbox import __version__, trials
from quantbox.cli import app

PREREG_README = "# a line\n\n```yaml\nprereg:\n  id: h1\n```\n"
# Annualised 1.0 over five daily years: PASS at 1 trial, FAIL at 40 (dsr_conservative 0.987 vs 0.752).
SUMMARY = ["--sharpe", "1.0", "--skew", "0", "--kurtosis", "3", "--n-obs", "1260", "--periods", "252"]
SUMMARY += ["--trials-sr-std", "0.02"]


def _line(tmp_path: Path, n_configs: int = 0, rows_per_config: int = 1) -> Path:
    line = tmp_path / "line"
    line.mkdir()
    (line / "README.md").write_text(PREREG_README)
    rows = [
        trials.trial_row("run", {"p": i}, data_pins={"demo": "ab" * 32}, sharpe=0.5)
        for i in range(n_configs)
        for _ in range(rows_per_config)
    ]
    trials.append(line, rows)
    return line


def _dsr(*args: str):
    return CliRunner().invoke(app, ["gates", "dsr", *SUMMARY, *args])


# ── the format ────────────────────────────────────────────────────────────


def test_a_row_is_the_owners_format():
    row = trials.trial_row("arms", {"p": 1}, data_pins={"demo": "ab" * 32}, sharpe=1.25, arm="p-1")
    assert set(row) == {"ts", "command", "config_sha256", "quantbox_version", "data_pins", "metric", "arm"}
    assert row["command"] == "arms" and row["arm"] == "p-1"
    assert row["quantbox_version"] == __version__
    assert row["data_pins"] == {"demo": "ab" * 32}
    assert row["metric"] == {"sharpe": 1.25}
    assert len(row["config_sha256"]) == 64
    assert row["ts"].endswith("Z") and row["ts"][10] == "T"  # UTC ISO-8601
    assert "arm" not in trials.trial_row("run", {"p": 1}, data_pins={}, sharpe=1.0)


def test_a_non_finite_sharpe_is_null_so_the_log_stays_strict_json(tmp_path):
    line = _line(tmp_path)
    trials.append(line, [trials.trial_row("run", {"p": 1}, data_pins={}, sharpe=float("nan"))])
    text = (line / trials.LOG_NAME).read_text()
    assert "NaN" not in text
    assert json.loads(text)["metric"] == {"sharpe": None}


def test_append_writes_one_json_object_per_line_and_never_rewrites(tmp_path):
    line = _line(tmp_path, n_configs=2)
    before = (line / trials.LOG_NAME).read_text()
    trials.append(line, [trials.trial_row("sweep", {"p": 9}, data_pins={}, sharpe=0.1, arm="p=9")])
    after = (line / trials.LOG_NAME).read_text()
    assert after.startswith(before)
    assert [json.loads(raw)["command"] for raw in after.splitlines()] == ["run", "run", "sweep"]


# ── the count ─────────────────────────────────────────────────────────────


def test_n_trials_is_the_distinct_configs_not_the_rows(tmp_path):
    assert trials.count(_line(tmp_path, n_configs=3, rows_per_config=4)) == 3


def test_blank_lines_are_ignored(tmp_path):
    line = _line(tmp_path, n_configs=2)
    with (line / trials.LOG_NAME).open("a") as f:
        f.write("\n   \n")
    assert trials.count(line) == 2


@pytest.mark.parametrize(
    "bad, message",
    [
        ("not json at all", "line 3 is not JSON"),
        ('{"command": "run"}', "line 3 names no `config_sha256`"),
        ("[1]", "line 3 names no `config_sha256`"),
    ],
)
def test_a_line_that_is_not_a_trial_row_is_an_error_not_a_skip(tmp_path, bad, message):
    line = _line(tmp_path, n_configs=2)
    with (line / trials.LOG_NAME).open("a") as f:
        f.write(bad + "\n")
    with pytest.raises(trials.TrialLogError, match=message):
        trials.count(line)


def test_no_log_is_an_error(tmp_path):
    with pytest.raises(trials.TrialLogError, match="no trial log"):
        trials.count(tmp_path)


def test_the_hash_ignores_where_the_output_lands_and_the_batch_count():
    cfg = {"run": {"mode": "backtest", "asof": "2024-02-29"}, "plugins": {"strategies": [{"params": {"w": 20}}]}}
    moved = {**cfg, "artifacts": {"root": "/elsewhere/20261114T101010Z"}, "run": {**cfg["run"], "n_trials": 9}}
    assert trials.config_sha256(moved) == trials.config_sha256(cfg)
    other = {**cfg, "plugins": {"strategies": [{"params": {"w": 21}}]}}
    assert trials.config_sha256(other) != trials.config_sha256(cfg)
    assert trials.config_sha256({**cfg, "run": {**cfg["run"], "asof": "2024-03-31"}}) != trials.config_sha256(cfg)


# ── the line ──────────────────────────────────────────────────────────────


def test_the_line_is_the_nearest_readme_with_a_prereg_block(tmp_path):
    line = _line(tmp_path)
    (line / "configs" / "deep").mkdir(parents=True)
    assert trials.find_line(line / "configs" / "deep") == line.resolve()
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "README.md").write_text("# no prereg here\n\nprereg: in prose is not a block\n")
    assert trials.find_line(tmp_path / "plain") is None


def test_the_walk_stops_at_a_git_root(tmp_path):
    (tmp_path / "README.md").write_text(PREREG_README)  # a line ABOVE the repo is not this repo's
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "configs").mkdir()
    assert trials.find_line(repo / "configs") is None


def test_an_explicit_line_must_be_a_directory(tmp_path):
    with pytest.raises(trials.TrialLogError, match="not a directory"):
        trials.resolve_line(str(tmp_path / "missing"), tmp_path / "c.yaml")
    assert trials.resolve_line(str(tmp_path), tmp_path / "x" / "c.yaml") == tmp_path.resolve()


# ── gates dsr --line ──────────────────────────────────────────────────────


def test_without_line_dsr_is_unchanged():
    """The Before of the proof: one trial claimed, the gate passes, whatever was run."""
    result = _dsr("--n-trials", "1")
    assert result.exit_code == 0, result.output
    assert result.output.startswith("dsr: PASS")
    default = json.loads(_dsr("--json").stdout)
    assert set(default["by_n_trials"]) == {"1", "5", "10", "20", "50", "100"}
    assert "trial_log" not in default


def test_a_claim_below_the_log_is_refused_naming_the_logged_count(tmp_path):
    line = _line(tmp_path, n_configs=40, rows_per_config=2)
    result = _dsr("--line", str(line), "--n-trials", "1")
    assert result.exit_code == 2, result.output
    assert "40 logged trials" in result.output and "40 distinct configs" in result.output
    # a range is judged by its maximum: the verdict is read there
    assert _dsr("--line", str(line), "--n-trials", "1,5,39").exit_code == 2


def test_with_the_log_the_verdict_equals_the_old_one_at_the_logged_count(tmp_path):
    line = _line(tmp_path, n_configs=40, rows_per_config=2)
    logged = _dsr("--line", str(line), "--json")
    old = _dsr("--n-trials", "40", "--json")
    assert logged.exit_code == old.exit_code == 1
    got = json.loads(logged.stdout)
    assert got.pop("trial_log") == {"path": str(line / trials.LOG_NAME), "n_logged": 40}
    assert got == json.loads(old.stdout)
    text = _dsr("--line", str(line))
    assert "trials: 40 logged in" in text.output


def test_a_claim_at_or_above_the_log_is_used_as_given(tmp_path):
    line = _line(tmp_path, n_configs=40)
    for claim in ("40", "1,5,100"):
        got = json.loads(_dsr("--line", str(line), "--n-trials", claim, "--json").stdout)
        got.pop("trial_log")
        assert got == json.loads(_dsr("--n-trials", claim, "--json").stdout), claim


@pytest.mark.parametrize("content", [None, "", "garbage\n"])
def test_a_line_with_no_readable_log_cannot_compute(tmp_path, content):
    if content is not None:
        (tmp_path / trials.LOG_NAME).write_text(content)
    result = _dsr("--line", str(tmp_path))
    assert result.exit_code == 2, result.output
    assert "CANNOT COMPUTE" in result.output
