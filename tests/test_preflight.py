"""`quantbox preflight` — every refusal fires, and blindness never exits 0.

Ported from qute-research 0.13.0 `tests/test_quant_preflight.py` with the gate
(ADR-0013). Every case is kept, except the ones that drove the plugin's subprocess
path through a stand-in CLI (`fixtures/fake_quantbox.py`): the port asks quantbox
in-process, so those are replaced by the in-process cases at the end of the file.

WHY THESE TESTS EXIST. This gate's failure is the silent kind the suite keeps
tests for: a guard that wrongly ALLOWS looks exactly like a clean line, and the
backtest it should have stopped then produces a number somebody believes. So
each refusal has a POSITIVE CONTROL built by breaking exactly one thing in a
lab that otherwise passes — and `test_the_clean_lab_passes` is what proves the
fixture is not refusing for some other reason. Every assertion names the check
id AND a phrase of the reason, because "exit 1" alone is reached by any branch.

Each test drives a real git repo: the timing check is a statement about
history, and a fixture that fakes history would be testing the fixture.

What the RUNNER will do with a config is quantbox's answer (`quantbox config
explain --json`, explain@1 — TOM-1367), so each config's plan is a dict the test
writes (`lab.plans`), handed to `run(explain=...)`.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quantbox import preflight

SYSTEM = """\
system: toy-lab
mandate: alpha
production: toy-live
books:
  - name: perp-book
    market: perp
    venue:
      allow_shorts: true
      leverage: 3
      funding_modelled: true
    primary_dataset: perp-daily
  - name: spot-book
    market: spot
    venue: {allow_shorts: false, leverage: 1, funding_modelled: false}
    primary_dataset: spot-daily
data:
  canonical_root: datasets
  on_demand_dir: datasets
liveness:
  min_realized_vol_ratio: 0.5
  max_short_gross_share: 0.6
  max_abs_net_exposure_for_long_only: 1.0
modes: {monitor: true, scout: false, dev_loop: true}
"""

PREREG = """\
prereg:
  id: H1
  book: perp-book
  primary_dataset: perp-daily
  secondary_datasets: [spot-daily]
  criterion:
    pass_rule: "all legs"
    legs:
      - name: mdd
        metric: max_drawdown
        threshold: 0.70
  claim_test:
    method: paired-stationary-block-bootstrap
    draws: 2000
    mean_block: 30
    seed: 7
    min_probability: 0.9
  ex_largest_episode:
    relabel_to: inconclusive
  n_trials: 4
  liveness:
    min_realized_vol_ratio: 0.5
    max_short_gross_share: 0.5
    max_abs_net_exposure_for_long_only: 1.0
"""

CONFIG = """\
run:
  mode: backtest
artifacts:
  root: "./artifacts/arm"
plugins:
  pipeline:
    params:
      engine: rsims
      execution:
        lag_bars: 1
      risk:
        max_leverage: 3
  strategies:
    - name: s
      params:
        spans: [[8, 32], [16, 64]]
  data:
    params_init:
      dataset_root: "../../datasets"
      dataset: "perp-daily"
"""

FUNDING = "PAR1" + "funding-rates " * 4 + "PAR1"  # parquet magic at both ends, not a 4-byte stub
T0, T1, T2 = "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"


def _git(repo: Path, *args: str, date: str = T0) -> str:
    env = dict(os.environ, GIT_COMMITTER_DATE=date, GIT_AUTHOR_DATE=date)
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _readme(prereg: str) -> str:
    return f"# line\n\nProse for people.\n\n```yaml\n{prereg}```\n"


def make_plan(lab: Lab, arm: str = "arm") -> dict:
    """quantbox's explain@1 plan for the clean config: a perp dataset pinned by its
    datasets.lock, the funding engine with a funding file, t+1 execution, venue
    as the perp book declares it."""
    return {
        "schema": "quantbox/explain@1",
        "ok": True,
        "errors": [],
        "engine": {"name": "rsims", "version": "0.0-test"},
        "dataset": {"name": "perp-daily", "sha256": "b" * 64, "source": "lock", "tier": "lock", "market": "futures"},
        "funding": {
            "modelled": True,
            "source_path": str(lab.root / "datasets" / "perp-daily" / "funding_rates.parquet"),
            "sha256": None,
        },
        "execution": {
            "lag_bars": 1,
            "fill": "close",
            "same_bar": False,
            "description": "decide at t, fill at t+1 close",
        },
        "venue": {"declared": True, "allow_shorts": True, "max_leverage": 3},
        "n_trials": None,
        "artifacts_root": str(lab.line / "artifacts" / arm),
    }


def spot_plan(lab: Lab) -> dict:
    plan = make_plan(lab)
    plan["dataset"].update(name="spot-daily", market="spot")
    plan["venue"].update(allow_shorts=False, max_leverage=1)
    plan["funding"] = {"modelled": False, "source_path": None, "sha256": None}
    return plan


class Lab:
    def __init__(self, root: Path):
        self.root = root
        self.line = root / "research" / "line"
        self.config = self.line / "configs" / "arm.yaml"
        self.readme = self.line / "README.md"
        self.plans: dict[str, dict] = {"arm.yaml": make_plan(self)}
        self.resolutions: dict[str, dict] = {}
        self.asked: list[tuple[Path, Path]] = []

    @property
    def plan(self) -> dict:
        return self.plans["arm.yaml"]

    def explain(self, path: Path, cwd: Path) -> dict:
        self.asked.append((path, cwd))
        return copy.deepcopy(self.plans[path.name])

    def resolve(self, name: str, config: Path, cwd: Path) -> dict:
        self.asked.append((config, cwd))
        if name not in self.resolutions:
            raise preflight.Unavailable(f"dataset {name!r} not found")
        return copy.deepcopy(self.resolutions[name])

    def commit(self, message: str = "c", date: str = T0) -> None:
        _git(self.root, "add", "-A", date=date)
        _git(self.root, "commit", "-q", "-m", message, date=date)

    def add_run(self, stamp: str = "20260102T000000Z", arm: str = "arm") -> None:
        _write(self.line / "artifacts" / arm / f"x__{stamp}" / "run_manifest.json", "{}")

    def run(self, *configs: Path, **kwargs):
        kwargs.setdefault("explain", self.explain)
        kwargs.setdefault("resolve", self.resolve)
        return preflight.run(self.root, self.line, list(configs or [self.config]), **kwargs)


@pytest.fixture
def lab(tmp_path) -> Lab:
    made = Lab(tmp_path / "lab")
    _write(made.root / "system.yaml", SYSTEM)
    _write(made.root / "datasets" / "perp-daily" / "funding_rates.parquet", FUNDING)
    _write(made.root / ".gitignore", "artifacts/\n")
    _write(made.readme, _readme(PREREG))
    _write(made.config, CONFIG)
    _git(made.root, "init", "-q")
    made.commit()
    return made


def _rows(report, status: str) -> dict[str, str]:
    return {check: reason for st, check, reason in report.rows if st == status}


def _refused(report, check: str, phrase: str) -> None:
    refused = _rows(report, preflight.REFUSE)
    assert check in refused, f"{check} not refused; rows={report.rows}"
    assert phrase in refused[check], refused[check]
    assert report.exit_code == 1


def _only_refusal(report, check: str, phrase: str) -> None:
    """The ONE thing broken is the one thing refused — nothing else reached exit 1."""
    _refused(report, check, phrase)
    assert list(_rows(report, preflight.REFUSE)) == [check]


# ── the control the positive controls stand on ─────────────────────────────


def test_the_clean_lab_passes(lab):
    report = lab.run()
    assert report.exit_code == 0, report.rows
    assert {st for st, _, _ in report.rows} == {preflight.PASS}
    checks = {check for _, check, _ in report.rows}
    # every family of check RAN — a pass that skipped one is not a pass
    for family in (
        "system",
        "prereg",
        "criterion",
        "claim-test",
        "ex-largest-episode",
        "trial-count",
        "prereg-committed",
        "primary-dataset",
        "liveness",
        "market[arm.yaml]",
        "data[arm.yaml]",
        "shorts[arm.yaml]",
        "leverage[arm.yaml]",
        "execution[arm.yaml]",
        "funding[arm.yaml]",
        "placeholders",
        "secondary-datasets",
        "run-manifest",
    ):
        assert family in checks, family
    # quantbox was asked, from the line, about the config itself
    assert lab.asked == [(lab.config, lab.line)]


def test_a_later_run_after_the_frozen_criterion_passes(lab):
    lab.add_run("20260102T000000Z")
    assert lab.run().exit_code == 0


# ── system.yaml ────────────────────────────────────────────────────────────


def test_no_system_yaml_refuses_and_says_cannot_determine(lab):
    (lab.root / "system.yaml").unlink()
    lab.commit()
    _refused(lab.run(), "system", "no system.yaml — cannot determine")


def test_a_system_yaml_missing_a_venue_answer_refuses(lab):
    _write(lab.root / "system.yaml", SYSTEM.replace("      allow_shorts: true\n", ""))
    _refused(lab.run(), "system", "venue.allow_shorts must be true|false")


def test_a_book_the_lab_does_not_declare_refuses(lab):
    _write(lab.readme, _readme(PREREG.replace("book: perp-book", "book: ghost")))
    lab.commit()
    _only_refusal(lab.run(), "book", "not declared in system.yaml")


# ── the primary dataset is the book's market ───────────────────────────────


def test_a_spot_primary_for_a_perp_book_refuses(lab):
    swapped = PREREG.replace("primary_dataset: perp-daily", "primary_dataset: spot-daily")
    _write(lab.readme, _readme(swapped.replace("[spot-daily]", "[perp-daily]")))
    lab.plan["dataset"].update(name="spot-daily", market="spot")
    lab.commit()
    report = lab.run()
    _refused(report, "primary-dataset", "can only be secondary")
    _refused(report, "market[arm.yaml]", "declared PRIMARY, but spot-daily is a spot dataset")


def test_a_primary_whose_manifest_is_another_market_refuses(lab):
    """The NAME matches the book's; the dataset itself says it is spot."""
    lab.plan["dataset"]["market"] = "spot"
    report = lab.run()
    _refused(report, "market[arm.yaml]", "trades perp")
    assert "primary-dataset" in _rows(report, preflight.PASS)


def test_a_dataset_the_prereg_never_named_refuses(lab):
    _write(lab.readme, _readme(PREREG.replace("  secondary_datasets: [spot-daily]\n", "")))
    lab.plan["dataset"].update(name="spot-daily", market="spot")
    lab.plan["venue"]["allow_shorts"] = False
    lab.commit()
    _only_refusal(lab.run(), "dataset[arm.yaml]", "neither the pre-registered primary")


# ── what the venue cannot hold is OFF in the config ────────────────────────

SPOT_PREREG = (
    PREREG.replace("book: perp-book", "book: spot-book")
    .replace("primary_dataset: perp-daily", "primary_dataset: spot-daily")
    .replace("[spot-daily]", "[]")
)


def _spot(lab: Lab) -> None:
    _write(lab.readme, _readme(SPOT_PREREG))
    lab.plans["arm.yaml"] = spot_plan(lab)
    lab.commit()


def test_a_long_only_book_whose_runner_applies_shorts_off_passes(lab):
    _spot(lab)
    report = lab.run()
    assert report.exit_code == 0, report.rows
    assert "the runner applies venue.allow_shorts: false" in _rows(report, preflight.PASS)["shorts[arm.yaml]"]


def test_a_long_only_book_whose_runner_allows_shorts_refuses(lab):
    _spot(lab)
    lab.plan["venue"]["allow_shorts"] = True
    _only_refusal(lab.run(), "shorts[arm.yaml]", "the runner will allow shorts")


def test_shorts_on_a_spot_secondary_of_a_shorting_book_refuse(lab):
    """The book may short — on ITS market. A spot proxy arm may not."""
    lab.plan["dataset"].update(name="spot-daily", market="spot")
    _only_refusal(lab.run(), "shorts[arm.yaml]", "this config trades spot, not the book's perp")


def test_leverage_above_the_venue_refuses(lab):
    lab.plan["venue"]["max_leverage"] = 5
    _only_refusal(lab.run(), "leverage[arm.yaml]", "caps gross leverage at 5 > venue leverage 3")


# ── execution timing ───────────────────────────────────────────────────────

STATED = PREREG + "  execution:\n    timing: same-bar at the pinned engine\n    plus_one_bar_result: true\n"


def test_absent_execution_is_unknown_timing_and_refuses(lab):
    del lab.plan["execution"]
    _only_refusal(lab.run(), "execution[arm.yaml]", "timing is UNKNOWN")


def test_absent_execution_passes_when_timing_and_plus_one_bar_are_preregistered(lab):
    del lab.plan["execution"]
    _write(lab.readme, _readme(STATED))
    lab.commit()
    assert lab.run().exit_code == 0


def test_a_timing_statement_without_the_plus_one_bar_promise_still_refuses(lab):
    del lab.plan["execution"]
    _write(
        lab.readme,
        _readme(STATED.replace("plus_one_bar_result: true", "plus_one_bar_result: false")),
    )
    lab.commit()
    _only_refusal(lab.run(), "execution[arm.yaml]", "timing is UNKNOWN")


@pytest.mark.parametrize("same_bar", [{"lag_bars": 0}, {"same_bar": True}], ids=["lag-0", "same-bar"])
def test_lag_zero_needs_a_written_reason(lab, same_bar):
    lab.plan["execution"].update(same_bar)
    _only_refusal(lab.run(), "execution[arm.yaml]", "lag0_reason")
    _write(
        lab.readme,
        _readme(PREREG + "  execution:\n    lag0_reason: fills at the open auction\n"),
    )
    lab.commit()
    assert lab.run().exit_code == 0


# ── what the pre-registration must carry ───────────────────────────────────


def test_a_prose_only_preregistration_refuses(lab):
    _write(lab.readme, "# line\n\nMDD reduction >= 30% AND Calmar improves >= 20%.\n")
    lab.commit()
    _refused(lab.run(), "prereg", "criterion in prose cannot be checked")


@pytest.mark.parametrize(
    ("broken", "check", "phrase"),
    [
        (
            PREREG.replace("        threshold: 0.70\n", ""),
            "criterion",
            "numeric `threshold`",
        ),
        (
            PREREG.replace("        threshold: 0.70", "        threshold: a lot"),
            "criterion",
            "numeric `threshold`",
        ),
        (
            PREREG.replace('    pass_rule: "all legs"\n', ""),
            "criterion",
            "pass_rule missing",
        ),
        (
            PREREG.replace("    method: paired-stationary-block-bootstrap\n", ""),
            "claim-test",
            "no significance test of its own",
        ),
        (PREREG.replace("    seed: 7\n", ""), "claim-test", "claim_test.seed"),
        (
            PREREG.replace("    min_probability: 0.9\n", ""),
            "claim-test",
            "min_probability",
        ),
        (
            PREREG.replace("  ex_largest_episode:\n    relabel_to: inconclusive\n", ""),
            "ex-largest-episode",
            "largest drawdown episode",
        ),
        (PREREG.replace("  n_trials: 4\n", ""), "trial-count", "no integer `n_trials`"),
        (PREREG.replace("n_trials: 4", "n_trials: 0"), "trial-count", "no integer `n_trials`"),
        (PREREG.replace("n_trials: 4", "n_trials: true"), "trial-count", "no integer `n_trials`"),
        (
            PREREG.replace("relabel_to: inconclusive", "relabel_to: maybe"),
            "ex-largest-episode",
            "largest drawdown episode",
        ),
        (
            PREREG.replace("    max_short_gross_share: 0.5\n", ""),
            "liveness",
            "max_short_gross_share not pre-registered",
        ),
        (
            PREREG.replace("max_short_gross_share: 0.5", "max_short_gross_share: 0.9"),
            "liveness",
            "looser",
        ),
        (
            PREREG.replace(
                "min_realized_vol_ratio: 0.5\n    max_short",
                "min_realized_vol_ratio: 0.1\n    max_short",
            ),
            "liveness",
            "min_realized_vol_ratio=0.1 is looser",
        ),
    ],
    ids=[
        "threshold-absent",
        "threshold-not-a-number",
        "no-pass-rule",
        "no-claim-test-method",
        "bootstrap-without-seed",
        "no-min-probability",
        "no-ex-largest-episode",
        "no-n-trials",
        "n-trials-zero",
        "n-trials-boolean",
        "ex-largest-episode-bad-label",
        "liveness-key-absent",
        "liveness-max-looser",
        "liveness-min-looser",
    ],
)
def test_a_preregistration_missing_a_piece_refuses(lab, broken, check, phrase):
    assert broken != PREREG, "the replacement did not apply — this case would test nothing"
    _write(lab.readme, _readme(broken))
    lab.commit()
    _only_refusal(lab.run(), check, phrase)


def test_fewer_trials_than_configs_about_to_run_refuses(lab):
    _write(lab.readme, _readme(PREREG.replace("n_trials: 4", "n_trials: 1")))
    second = lab.line / "configs" / "arm2.yaml"
    _write(second, CONFIG.replace("artifacts/arm", "artifacts/arm2"))
    lab.plans["arm2.yaml"] = make_plan(lab, "arm2")
    lab.commit()
    _only_refusal(lab.run(lab.config, second), "trial-count", "n_trials=1 < the 2 config(s)")


# ── the criterion was committed BEFORE the first run ───────────────────────


def test_an_uncommitted_criterion_refuses(lab):
    _write(lab.readme, _readme(PREREG.replace("threshold: 0.70", "threshold: 0.80")))
    _only_refusal(lab.run(), "prereg-committed", "uncommitted changes")


def test_a_run_older_than_the_criterion_refuses(lab):
    lab.add_run("20251231T235959Z")  # the fixture's commit is T0 = 2026-01-01
    _only_refusal(lab.run(), "prereg-committed", "predates the criterion's last change")


def test_a_criterion_edited_after_the_first_run_refuses(lab):
    lab.add_run("20260102T000000Z")
    _write(lab.readme, _readme(PREREG.replace("threshold: 0.70", "threshold: 0.80")))
    lab.commit("move the goalposts", date=T2)
    _only_refusal(lab.run(), "prereg-committed", "post-registration")


def test_growing_the_trial_count_after_a_run_is_not_a_post_registration(lab):
    """`n_trials` is outside the frozen part on purpose — it grows with every arm."""
    lab.add_run("20260102T000000Z")
    _write(lab.readme, _readme(PREREG.replace("n_trials: 4", "n_trials: 6")))
    lab.commit("count the extra arms", date=T2)
    report = lab.run()
    assert report.exit_code == 0, report.rows
    assert "frozen at" in _rows(report, preflight.PASS)["prereg-committed"]


def test_a_runs_dir_outside_the_config_is_searched_too(lab, tmp_path):
    _write(tmp_path / "elsewhere" / "r__20251231T000000Z" / "run_manifest.json", "{}")
    report = lab.run(runs_dirs=[tmp_path / "elsewhere"])
    _only_refusal(report, "prereg-committed", "predates")


# ── pinned research-plane data ─────────────────────────────────────────────


def test_a_dataset_pinned_by_its_lock_passes_and_says_so(lab):
    said = _rows(lab.run(), preflight.PASS)["data[arm.yaml]"]
    assert "perp-daily read by name at the build datasets.lock pins" in said


def test_a_dataset_read_by_name_but_not_pinned_refuses(lab):
    lab.plan["dataset"].update(tier="raw", sha256="c" * 64)
    _only_refusal(lab.run(), "data[arm.yaml]", "NOT pinned in a datasets.lock")


UNRESOLVED = {"tier": "raw", "sha256": None, "market": None, "warning": "no dataset plugin used"}


def test_a_by_name_read_the_plan_left_unresolved_is_settled_by_dataset_resolve(lab):
    """Seen on quantbox-lab (`dataset.curated.v1`): explain reports tier raw, sha256 null
    and no market. quantbox's `dataset resolve` answers the narrower question — the
    build a run binds — and the market comes from that answer."""
    lab.plan["dataset"].update(UNRESOLVED)
    lab.resolutions["perp-daily"] = {
        "sha256": "b" * 64,
        "matches": True,
        "restored": False,
        "lock": "/lab/datasets.lock",
        "market": "futures",
    }
    report = lab.run()
    assert report.exit_code == 0, report.rows
    passed = _rows(report, preflight.PASS)
    assert "`quantbox dataset resolve` binds the build /lab/datasets.lock pins" in passed["data[arm.yaml]"]
    assert passed["market[arm.yaml]"] == "perp-daily: perp (primary)"
    assert (lab.config, lab.line) in lab.asked


@pytest.mark.parametrize(
    ("resolution", "phrase"),
    [
        ({"sha256": None, "matches": None, "lock": None, "market": "futures"}, "NOT pinned"),
        ({"sha256": "b" * 64, "matches": False, "error": "has sha256 x", "market": "futures"}, "not the pinned build"),
    ],
    ids=["no-pin", "bytes-moved"],
)
def test_an_unresolved_by_name_read_that_resolve_finds_unpinned_refuses(lab, resolution, phrase):
    lab.plan["dataset"].update(UNRESOLVED)
    lab.resolutions["perp-daily"] = resolution
    _only_refusal(lab.run(), "data[arm.yaml]", phrase)


def test_an_unresolved_by_name_read_that_resolve_cannot_settle_is_blind(lab):
    """Whether a lock pins it is then unknown — "not pinned" would be a false refusal,
    "pinned" a false pass."""
    lab.plan["dataset"].update(UNRESOLVED)
    report = lab.run()
    _blind(report, "data[arm.yaml]")
    said = _rows(report, preflight.BLIND)["data[arm.yaml]"]
    assert "no dataset plugin used" in said and "`dataset resolve` could not either" in said


INLINE_SHA = "d" * 64
HASHED = (
    "inputs:\n  - dataset: perp-daily\n    market: perp\n    path: datasets/perp-daily/prices.parquet\n    sha256: "
    + INLINE_SHA
    + "\n"
)


def _inline(lab: Lab, data_inputs: str | None) -> None:
    lab.plan["dataset"] = {
        "name": "prices.parquet",
        "sha256": INLINE_SHA,
        "source": "inline",
        "tier": "raw",
        "market": None,
    }
    if data_inputs is not None:
        _write(lab.line / "data_inputs.yaml", data_inputs)
        lab.commit()


def test_a_file_read_by_path_whose_hash_the_line_declares_passes(lab):
    """The dataset's name and market then come from that data_inputs.yaml row."""
    _inline(lab, HASHED)
    report = lab.run()
    assert report.exit_code == 0, report.rows
    passed = _rows(report, preflight.PASS)
    assert "declared in data_inputs.yaml" in passed["data[arm.yaml]"]
    assert passed["market[arm.yaml]"] == "perp-daily: perp (primary)"


@pytest.mark.parametrize(
    "data_inputs",
    [None, HASHED.replace(INLINE_SHA, "null"), HASHED.replace(INLINE_SHA, "e" * 64), "inputs: []\n"],
    ids=["no-file", "unhashed", "another-hash", "empty"],
)
def test_a_file_read_by_path_whose_hash_is_declared_nowhere_refuses(lab, data_inputs):
    _inline(lab, data_inputs)
    _refused(lab.run(), "data[arm.yaml]", "that content hash is declared nowhere")


def test_a_dataset_source_quantbox_does_not_name_is_blind(lab):
    lab.plan["dataset"]["source"] = "warehouse"
    _blind(lab.run(), "data[arm.yaml]")


# ── blind is not clean ─────────────────────────────────────────────────────


def _blind(report, check: str) -> None:
    assert check in _rows(report, preflight.BLIND), report.rows
    assert report.exit_code == 2


def test_a_system_yaml_the_parser_cannot_read_is_blind_not_absent(lab, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)  # CI has no PyYAML; make that true here
    _write(lab.root / "system.yaml", "base: &b {x: 1}\n" + SYSTEM)
    _blind(lab.run(), "system")


def test_quantbox_raising_instead_of_planning_is_blind_never_a_pass(lab, monkeypatch):
    """The port's "no quantbox to ask": the in-process plan raises (seen for real: a data
    plugin whose import fails, `No module named 'yfinance'`)."""

    def crash(_path):
        raise ModuleNotFoundError("No module named 'yfinance'")

    monkeypatch.setattr(preflight._explain, "explain_path", crash)
    report = lab.run(explain=preflight.explain_config)
    _blind(report, "config[arm.yaml]")
    said = _rows(report, preflight.BLIND)["config[arm.yaml]"]
    assert "could not ask quantbox" in said and "raised ModuleNotFoundError" in said


@pytest.mark.parametrize(
    "schema", ["quantbox/explain@2", "quantbox/run@1", None], ids=["next-major", "other-kind", "none"]
)
def test_a_plan_of_a_schema_this_plugin_does_not_read_is_blind(lab, schema):
    lab.plan["schema"] = schema
    report = lab.run()
    _blind(report, "config[arm.yaml]")
    assert not any(check.endswith("[arm.yaml]") and check != "config[arm.yaml]" for _, check, _ in report.rows), (
        "a plan that was not read must not be judged"
    )


def test_a_run_quantbox_cannot_start_refuses_with_quantbox_reasons(lab):
    lab.plans["arm.yaml"] = {"schema": "quantbox/explain@1", "ok": False, "errors": ["data: unknown plugin 'x.v1'"]}
    _only_refusal(lab.run(), "config[arm.yaml]", "quantbox cannot start this run: data: unknown plugin 'x.v1'")


def test_a_dataset_with_no_market_is_blind(lab):
    lab.plan["dataset"]["market"] = None
    _blind(lab.run(), "market[arm.yaml]")


def test_no_artifacts_root_means_nowhere_to_look_for_runs(lab):
    del lab.plan["artifacts_root"]
    _blind(lab.run(), "prereg-committed")


def test_a_line_outside_git_cannot_prove_its_timing(tmp_path):
    made = Lab(tmp_path / "lab")
    _write(made.root / "system.yaml", SYSTEM)
    _write(made.root / "datasets" / "perp-daily" / "funding_rates.parquet", FUNDING)
    _write(made.readme, _readme(PREREG))
    _write(made.config, CONFIG)
    _blind(made.run(), "prereg-committed")


def test_several_prereg_blocks_without_an_id_is_blind_and_with_one_passes(lab):
    _write(lab.readme, _readme(PREREG) + _readme(PREREG.replace("id: H1", "id: H2")))
    lab.commit()
    _blind(lab.run(), "prereg")
    assert lab.run(wanted="H1").exit_code == 0


def test_a_refusal_outranks_blindness(lab):
    lab.plan["dataset"]["market"] = None
    lab.plan["venue"]["max_leverage"] = 9
    report = lab.run()
    assert _rows(report, preflight.BLIND) and report.exit_code == 1
    assert "could not check" in preflight.render(report)


def test_an_empty_report_is_not_a_pass():
    assert preflight.Report().exit_code == 2


# ── the parser refuses to guess ────────────────────────────────────────────


def test_the_subset_parser_reads_what_the_gate_needs():
    parsed = preflight.parse_yaml(CONFIG)
    assert parsed["plugins"]["strategies"][0]["params"]["spans"] == [[8, 32], [16, 64]]
    assert parsed["plugins"]["pipeline"]["params"]["execution"]["lag_bars"] == 1
    folded = preflight.parse_yaml("a: one two\n  three\nb: 'x # y'  # c\nc:\n- 1\n- k: v\n  l: w\n")
    assert folded == {
        "a": "one two three",
        "b": "x # y",
        "c": [1, {"k": "v", "l": "w"}],
    }


@pytest.mark.parametrize("text", ["a: &x 1\n", "b: *x\n", "<<: {a: 1}\n", "a: [1,\n  2]\n", "a: !!str 1\n"])
def test_the_subset_parser_raises_on_what_it_does_not_model(text):
    with pytest.raises(preflight.YamlUnsupported):
        preflight.parse_yaml(text)


# ── the gate trusts no TYPE (M1) ───────────────────────────────────────────
#
# The config's own spellings (`yes`, `1_0`, an `ALLOW_SHORT` key) are quantbox's to
# read now; what this gate reads is the PLAN, and a plan field of the wrong type is
# BLIND — never coerced into a pass.


@pytest.mark.parametrize(
    ("where", "value", "check"),
    [
        (("venue", "allow_shorts"), "false", "shorts[arm.yaml]"),
        (("venue", "allow_shorts"), 0, "shorts[arm.yaml]"),
        (("venue", "max_leverage"), "3", "leverage[arm.yaml]"),
        (("venue", "max_leverage"), float("inf"), "leverage[arm.yaml]"),
        (("venue", "max_leverage"), True, "leverage[arm.yaml]"),
        (("execution", "lag_bars"), "0", "execution[arm.yaml]"),
        (("execution", "lag_bars"), 1.0, "execution[arm.yaml]"),
        (("execution", "lag_bars"), -1, "execution[arm.yaml]"),
        (("execution", "lag_bars"), True, "execution[arm.yaml]"),
    ],
    ids=[
        "shorts-string",
        "shorts-int",
        "leverage-string",
        "leverage-inf",
        "leverage-bool",
        "lag-string",
        "lag-float",
        "lag-negative",
        "lag-bool",
    ],
)
def test_a_plan_field_of_the_wrong_type_is_blind(lab, where, value, check):
    lab.plan[where[0]][where[1]] = value
    _blind(lab.run(), check)


@pytest.mark.parametrize("spelling", ["1_0", "0x10", "010", "1:30", "1e3", "1.0e3"])
def test_a_number_yaml_readers_disagree_about_is_never_read_as_a_string(spelling):
    with pytest.raises(preflight.YamlUnsupported):
        preflight.parse_yaml(f"leverage: {spelling}\n")


def test_yaml_1_1_booleans_are_booleans_exactly_as_the_engine_reads_them():
    parsed = preflight.parse_yaml("a: yes\nb: Off\nc: TRUE\nd: tRuE\ne: 'yes'\nf: {g: on}\n")
    assert parsed == {
        "a": True,
        "b": False,
        "c": True,
        "d": "tRuE",
        "e": "yes",
        "f": {"g": True},
    }


@pytest.mark.parametrize(
    ("broken", "check", "phrase"),
    [
        (
            PREREG.replace("threshold: 0.70", "threshold: .inf"),
            "criterion",
            "not a finite numeric",
        ),
        (
            PREREG.replace("threshold: 0.70", "threshold: .nan"),
            "criterion",
            "not a finite numeric",
        ),
        (
            PREREG.replace("threshold: 0.70", "threshold: 1.0e+999"),
            "criterion",
            "not a finite numeric",
        ),
        (
            PREREG.replace("threshold: 0.70", 'threshold: "0.70"'),
            "criterion",
            "not a finite numeric",
        ),
        (
            PREREG.replace("min_probability: 0.9", "min_probability: .nan"),
            "claim-test",
            "min_probability is nan",
        ),
        (
            PREREG.replace("n_trials: 4", "n_trials: 4.0"),
            "trial-count",
            "`n_trials` is 4.0",
        ),
        (
            PREREG.replace("n_trials: 4", 'n_trials: "4"'),
            "trial-count",
            "`n_trials` is '4'",
        ),
        (
            PREREG.replace("max_short_gross_share: 0.5", 'max_short_gross_share: "0.5"'),
            "liveness",
            "is '0.5', not a finite number",
        ),
    ],
    ids=[
        "inf",
        "nan",
        "overflow",
        "threshold-string",
        "probability-nan",
        "trials-float",
        "trials-string",
        "liveness-string",
    ],
)
def test_a_value_of_the_wrong_type_refuses_by_name(lab, broken, check, phrase):
    assert broken != PREREG
    _write(lab.readme, _readme(broken))
    lab.commit()
    _only_refusal(lab.run(), check, phrase)


# ── review of 2d7fbab: run evidence is searched for, not pointed at (M2) ───

EDITED = PREREG.replace("threshold: 0.70", "threshold: 0.95")


def test_a_fresh_artifacts_root_does_not_hide_the_earlier_runs(lab):
    lab.add_run("20260102T000000Z")  # under ./artifacts/arm
    _write(lab.readme, _readme(EDITED))
    lab.plan["artifacts_root"] = str(lab.line / "artifacts" / "arm_v2")
    lab.commit("move the goalposts AND the artifacts root", date=T2)
    _only_refusal(lab.run(), "prereg-committed", "post-registration")


def test_a_manifest_under_another_name_is_still_a_run(lab):
    # OUTSIDE the declared root (where every file counts) and with no run stamp anywhere:
    # only the manifest NAME makes this evidence
    manifest = lab.line / "results" / "run1" / "manifest.json"
    _write(manifest, "{}")
    os.utime(manifest, (1767312000, 1767312000))  # 2026-01-02, before the edit at T2
    _write(lab.readme, _readme(EDITED))
    lab.commit("edit", date=T2)
    _only_refusal(lab.run(), "prereg-committed", "post-registration")


def test_a_run_stamped_directory_with_no_manifest_is_still_a_run(lab):
    (lab.line / "results" / "x__20251231T000000Z").mkdir(parents=True)
    _only_refusal(lab.run(), "prereg-committed", "predates")


def test_run_evidence_dated_in_the_future_refuses(lab):
    lab.add_run("20990101T000000Z")
    _only_refusal(lab.run(), "prereg-committed", "dated in the FUTURE")


def test_an_unstamped_run_is_dated_by_mtime_and_the_pass_says_so(lab):
    manifest = lab.line / "artifacts" / "arm" / "run1" / "run_manifest.json"
    _write(manifest, "{}")
    os.utime(manifest, (1767400000, 1767400000))  # 2026-01-03, after T0
    report = lab.run()
    assert report.exit_code == 0, report.rows
    said = _rows(report, preflight.PASS)["prereg-committed"]
    assert "from mtime" in said and "earliest of 1 counted item(s)" in said
    assert "walked:" in said and "1 evidence item(s) found" in said
    os.utime(manifest, (4102444800, 4102444800))  # touched to 2100
    _only_refusal(lab.run(), "prereg-committed", "dated in the FUTURE")


def test_the_pass_row_names_what_it_searched_when_it_found_nothing(lab):
    said = _rows(lab.run(), preflight.PASS)["prereg-committed"]
    assert "no run evidence" in said and str(lab.line) in said


def _two_blocks(h1: str = "", h2: str = "") -> str:
    return _readme(PREREG + h1) + _readme(PREREG.replace("id: H1", "id: H2") + h2)


H1_RUNS = "  runs: [artifacts/h1]\n"


def test_attribution_is_declared_a_run_under_another_blocks_runs_is_theirs(lab):
    """H1 declared where its runs land BEFORE it ran; H2 is registered later. H1's run
    is H1's — by declaration, and the row says so."""
    _write(lab.readme, _readme(PREREG + H1_RUNS))
    lab.commit("H1 declares its runs")
    lab.add_run("20260102T000000Z", arm="h1/arm")
    _write(lab.readme, _two_blocks(H1_RUNS))
    lab.commit("register H2", date=T2)
    report = lab.run(wanted="H2")
    assert report.exit_code == 0, report.rows
    assert "1 attributed to block 'H1' by its `runs`" in _rows(report, preflight.PASS)["prereg-committed"]


@pytest.mark.parametrize("labname", ["wt-h2-followup", "plainlab"])
def test_no_path_text_can_attribute_a_run_to_another_block(tmp_path, labname):
    """The reviewed design matched the other block's id as a SUBSTRING of the absolute
    path: a worktree called `wt-h2-followup` disowned every run of H1."""
    made = Lab(tmp_path / labname)
    _write(made.root / "system.yaml", SYSTEM)
    _write(made.root / "datasets" / "perp-daily" / "funding_rates.parquet", FUNDING)
    _write(made.root / ".gitignore", "artifacts/\n")
    _write(made.readme, _two_blocks())
    _write(made.config, CONFIG)
    _git(made.root, "init", "-q")
    made.commit()
    made.add_run("20260102T000000Z", arm="arm/H2_renamed_run")  # the run dir even NAMES H2
    _write(made.readme, _two_blocks().replace("threshold: 0.70", "threshold: 0.95", 1))
    made.commit("edit H1 after its run", date=T2)
    _only_refusal(made.run(wanted="H1"), "prereg-committed", "post-registration")


@pytest.mark.parametrize("dummy", ["arm", "line", "artifacts", "x"])
def test_a_dummy_block_whose_id_is_a_path_fragment_disowns_nothing(lab, dummy):
    lab.add_run("20260102T000000Z")
    _write(lab.readme, _readme(EDITED) + _readme(PREREG.replace("id: H1", f"id: {dummy}")))
    lab.commit("edit + dummy block", date=T2)
    _only_refusal(lab.run(wanted="H1"), "prereg-committed", "post-registration")


def test_a_runs_declaration_added_after_the_run_cannot_disown_it(lab):
    """Declared attribution is only as old as the declaration: a block that claims a
    directory AFTER a run landed there does not take that run off anyone's hands."""
    lab.add_run("20260102T000000Z", arm="old")
    _write(lab.readme, _readme(EDITED) + _readme(PREREG.replace("id: H1", "id: DUMMY") + "  runs: [artifacts/old]\n"))
    lab.commit("edit H1, add a block that claims the old runs", date=T2)
    report = lab.run(wanted="H1")
    _only_refusal(report, "prereg-committed", "NOT newer than that declaration, so counted here")


def test_a_new_h1_after_h12s_runs_is_not_refused_by_a_substring(lab):
    _write(lab.readme, _readme(PREREG.replace("id: H1", "id: H12") + "  runs: [artifacts/h12_arm]\n"))
    lab.commit("H12")
    lab.add_run("20260102T000000Z", arm="h12_arm")
    _write(lab.readme, lab.readme.read_text() + _readme(PREREG))
    lab.commit("a new H1", date=T2)
    assert lab.run(wanted="H1").exit_code == 0


def test_a_second_hypothesis_in_a_legacy_line_has_a_reasoned_way_in(lab):
    """H1 ran before any block declared `runs`. H2 cannot inherit that silently — and is
    not locked out either: it excludes H1's directory by name, with a reason, in the open."""
    lab.add_run("20260102T000000Z", arm="h1_arm")
    h2 = PREREG.replace("id: H1", "id: H2")
    _write(lab.readme, _readme(PREREG) + _readme(h2))
    lab.commit("register H2", date=T2)
    _only_refusal(lab.run(wanted="H2"), "prereg-committed", "list it under `not_runs:` with a reason")
    excluded = (
        h2 + "  not_runs:\n    - path: artifacts/h1_arm\n      reason: runs of H1, made before blocks declared runs\n"
    )
    _write(lab.readme, _readme(PREREG) + _readme(excluded))
    lab.commit("H2 says what is not its run", date=T2)
    report = lab.run(wanted="H2")
    assert report.exit_code == 0, report.rows
    assert "excluded by not_runs" in _rows(report, preflight.PASS)["prereg-committed"]


def test_two_blocks_claiming_overlapping_runs_refuse(lab):
    _write(lab.readme, _two_blocks("  runs: [artifacts]\n", "  runs: [artifacts/arm]\n"))
    lab.commit()
    _only_refusal(lab.run(wanted="H1"), "prereg-committed", "cannot belong to two hypotheses")


def test_a_config_whose_artifacts_root_is_outside_the_blocks_runs_refuses(lab):
    _write(lab.readme, _readme(PREREG + H1_RUNS))
    lab.commit()
    _only_refusal(lab.run(), "prereg-committed", "is outside this block's declared `runs`")
    lab.plan["artifacts_root"] = str(lab.line / "artifacts" / "h1" / "arm")
    assert lab.run().exit_code == 0


def test_runs_is_frozen_so_a_run_cannot_be_disowned_by_editing_it(lab):
    _write(lab.readme, _readme(PREREG + "  runs: [artifacts]\n"))
    lab.commit()
    lab.add_run("20260102T000000Z", arm="old")
    _write(lab.readme, _readme(PREREG + "  runs: [artifacts/arm]\n"))
    lab.commit("narrow `runs` to leave the old run outside", date=T2)
    _only_refusal(lab.run(), "prereg-committed", "post-registration")


@pytest.mark.parametrize(
    "parked",
    [".runs", "datasets/runs", "node_modules", "__pycache__", "deep/.hidden/more"],
)
def test_an_old_run_parked_anywhere_in_the_line_is_found(lab, parked):
    _write(lab.line / parked / "x__20251231T000000Z" / "run_manifest.json", "{}")
    _only_refusal(lab.run(), "prereg-committed", "predates")


@pytest.mark.parametrize("root", ["./.artifacts/arm", "./datasets/artifacts/arm", "./node_modules/arm"])
def test_a_declared_artifacts_root_is_always_walked_whatever_it_is_called(lab, root):
    lab.plan["artifacts_root"] = str(lab.line / root)
    _write(lab.line / root / "run1" / "results.parquet", "x")  # no manifest, no stamp: still a run's output
    os.utime(lab.line / root / "run1" / "results.parquet", (1735689600, 1735689600))  # 2025-01-01
    _only_refusal(lab.run(), "prereg-committed", "predates")


def test_inside_a_declared_root_every_file_is_a_runs_output_but_a_gitkeep_is_not(lab):
    _write(lab.line / "artifacts" / "arm" / ".gitkeep", "")
    os.utime(lab.line / "artifacts" / "arm" / ".gitkeep", (1735689600, 1735689600))
    assert lab.run().exit_code == 0


def test_vcs_and_env_directories_are_not_walked_and_the_row_says_so(lab):
    _write(lab.line / ".venv" / "lib" / "pkg" / "manifest.json", "{}")
    os.utime(lab.line / ".venv" / "lib" / "pkg" / "manifest.json", (1735689600, 1735689600))
    report = lab.run()
    assert report.exit_code == 0, report.rows
    assert "never walked: ['.venv']" in _rows(report, preflight.PASS)["prereg-committed"]


def test_a_package_manifest_under_node_modules_is_not_a_run(lab):
    _write(lab.line / "node_modules" / "left-pad" / "manifest.json", "{}")
    os.utime(lab.line / "node_modules" / "left-pad" / "manifest.json", (1735689600, 1735689600))
    assert lab.run().exit_code == 0


OLD = (1735689600, 1735689600)  # 2025-01-01: older than every commit in these fixtures


def test_an_old_manifest_that_is_not_a_run_refuses_until_it_is_declared_not_a_run(lab):
    """A false refusal with a way out: `not_runs`, reasoned, frozen, and printed."""
    _write(lab.line / "tests" / "fixtures" / "run_manifest.json", "{}")
    os.utime(lab.line / "tests" / "fixtures" / "run_manifest.json", OLD)
    _only_refusal(lab.run(), "prereg-committed", "list it under `not_runs:` with a reason")
    _write(lab.readme, _readme(PREREG + "  not_runs:\n    - path: tests/fixtures\n      reason: a test fixture\n"))
    lab.commit()
    report = lab.run()
    assert report.exit_code == 0, report.rows
    assert "1 excluded by not_runs" in _rows(report, preflight.PASS)["prereg-committed"]
    assert "(a test fixture)" in _rows(report, preflight.PASS)["prereg-committed"]


def test_a_not_runs_entry_without_a_reason_refuses(lab):
    _write(lab.readme, _readme(PREREG + "  not_runs:\n    - path: tests/fixtures\n"))
    lab.commit()
    _only_refusal(lab.run(), "prereg-committed", "needs a `path` and a written `reason`")


def test_a_manifest_declared_as_a_data_input_is_excluded_and_said(lab):
    _write(lab.line / "inputs" / "universe_manifest.json", "{}")
    os.utime(lab.line / "inputs" / "universe_manifest.json", OLD)
    _only_refusal(lab.run(), "prereg-committed", "predates")
    _write(lab.line / "data_inputs.yaml", "inputs:\n  - path: inputs/universe_manifest.json\n    sha256: TBD\n")
    lab.commit()
    report = lab.run()
    assert report.exit_code == 0, report.rows
    assert "1 excluded as a declared data input" in _rows(report, preflight.PASS)["prereg-committed"]


def test_the_parent_of_a_declared_root_is_walked_inside_the_lab_and_only_there(lab, tmp_path):
    lab.plan["artifacts_root"] = str(lab.root / "artifacts" / "line" / "arm")
    _write(lab.root / "artifacts" / "other-line" / "armz" / "x__20250101T000000Z" / "run_manifest.json", "{}")
    assert lab.run().exit_code == 0  # another line's runs are not under THIS root's parent
    _write(lab.root / "artifacts" / "line" / "arm_old" / "x__20250101T000000Z" / "run_manifest.json", "{}")
    _only_refusal(lab.run(), "prereg-committed", "predates")  # a sibling of the declared root is
    outside = tmp_path / "shared-runs"
    _write(outside / "someone-elses-lab" / "x__20250101T000000Z" / "run_manifest.json", "{}")
    lab.plan["artifacts_root"] = str(outside / "mine")
    report = lab.run()
    assert report.exit_code == 0, report.rows
    assert (
        "outside the lab, only the declared root itself was walked" in _rows(report, preflight.PASS)["prereg-committed"]
    )


def test_an_on_demand_dataset_manifest_is_a_declared_input_not_run_evidence(lab):
    listed = (
        HASHED + "  - dataset: perp-daily\n    path: datasets/perp-daily/manifest.json\n    sha256: " + "c" * 64 + "\n"
    )
    _inline(lab, listed)
    _write(lab.line / "datasets" / "perp-daily" / "manifest.json", "{}")
    os.utime(lab.line / "datasets" / "perp-daily" / "manifest.json", (1, 1))  # 1970: would refuse
    report = lab.run()
    assert report.exit_code == 0, report.rows
    assert "excluded as a declared data input" in _rows(report, preflight.PASS)["prereg-committed"]


@pytest.mark.parametrize(
    ("after", "what"),
    [
        (
            PREREG.replace("max_short_gross_share: 0.5", "max_short_gross_share: 0.6"),
            "liveness",
        ),
        (PREREG.replace("[spot-daily]", "[spot-daily, other]"), "secondary"),
        (PREREG + "  execution:\n    lag0_reason: because\n", "execution"),
    ],
    ids=["liveness-loosened", "secondary-added", "lag0-reason-added"],
)
def test_everything_a_check_compares_against_is_frozen(lab, after, what):
    lab.add_run("20260102T000000Z")
    _write(lab.readme, _readme(after))
    lab.commit(f"touch {what} after the run", date=T2)
    _only_refusal(lab.run(), "prereg-committed", "post-registration")


# ── review of 2d7fbab: reasons, never tracebacks (m1, m4–m8) ───────────────


def test_an_unexpected_exception_is_a_blind_row_and_the_rows_so_far_still_print(lab, monkeypatch, capsys):
    def boom(*_args, **_kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(preflight, "check_book", boom)
    monkeypatch.setattr(preflight, "explain_config", lab.explain)
    try:
        code = preflight.main(["--lab", str(lab.root), "--line", str(lab.line), str(lab.config)])
    except RuntimeError:
        pytest.fail("a traceback reached the caller — a traceback is not a reason")
    out = capsys.readouterr().out
    assert code == 2
    assert "BLIND  internal" in out and "unexpected RuntimeError: kaboom" in out
    assert "PASS   system" in out, "the rows collected before the crash were dropped"
    assert "PREFLIGHT: COULD NOT CHECK" in out


@pytest.mark.parametrize("value", ["5", "true", "spot-daily-long-history"], ids=["int", "bool", "string"])
def test_secondary_datasets_that_is_not_a_list_refuses_without_substring_matching(lab, value):
    _write(
        lab.readme,
        _readme(PREREG.replace("secondary_datasets: [spot-daily]", f"secondary_datasets: {value}")),
    )
    lab.plan["dataset"].update(name="spot-daily", market="spot")
    lab.plan["venue"]["allow_shorts"] = False
    lab.commit()
    report = lab.run()
    _refused(report, "secondary-datasets", "not a list of dataset names")
    _refused(report, "dataset[arm.yaml]", "neither the pre-registered primary")


@pytest.mark.parametrize("text", ["- a\n", "inputs: 5\n"], ids=["a-list", "inputs-int"])
def test_a_data_inputs_file_of_the_wrong_shape_refuses_with_a_reason(lab, text):
    _inline(lab, text)
    _refused(lab.run(), "data[arm.yaml]", "has no `inputs:` list")


@pytest.mark.parametrize(
    ("system", "phrase"),
    [
        (
            SYSTEM.replace("  - name: spot-book", "  - name: perp-book"),
            "duplicate book name(s) ['perp-book']",
        ),
        (
            SYSTEM.replace(
                "  max_short_gross_share: 0.6\n",
                "  max_short_gross_share: 0.6\n  floor_turnover: 5\n",
            ),
            "liveness key 'floor_turnover' must start with min_ or max_",
        ),
        (
            SYSTEM.replace("liveness:\n", "liveness:\n  1: 2\n"),
            "liveness key '1' must start with min_ or max_",
        ),
        (
            SYSTEM.replace("min_realized_vol_ratio: 0.5", 'min_realized_vol_ratio: "0.5"'),
            "liveness.min_realized_vol_ratio is '0.5', not a finite number",
        ),
        (
            SYSTEM.replace("    market: perp\n", "    market: perp\n    mandate: whatever\n"),
            "`mandate` override must be one of",
        ),
        (
            SYSTEM.replace("  - name: spot-book", "  - just-a-string\n  - name: spot-book"),
            "is not a mapping",
        ),
    ],
    ids=[
        "duplicate-book",
        "liveness-no-direction",
        "liveness-non-name-key",
        "liveness-string",
        "book-mandate-invalid",
        "book-not-a-mapping",
    ],
)
def test_a_declaration_that_cannot_be_compared_refuses_by_name(lab, system, phrase):
    assert system != SYSTEM
    _write(lab.root / "system.yaml", system)
    _refused(lab.run(), "system", phrase)


def test_a_per_book_mandate_override_is_accepted(lab):
    _write(
        lab.root / "system.yaml",
        SYSTEM.replace("    market: perp\n", "    market: perp\n    mandate: zero-edge-test\n"),
    )
    assert lab.run().exit_code == 0


def test_a_prereg_block_inside_a_longer_fence_is_documentation_not_the_preregistration(lab):
    _write(lab.readme, "# line\n\n````markdown\nexample:\n```yaml\n" + PREREG + "```\n````\n")
    lab.commit()
    _refused(lab.run(), "prereg", "no top-level fenced yaml block")
    # a SHORTER bare fence inside does not close the longer one — or the example escapes
    _write(lab.readme, "# line\n\n````markdown\n```\n```yaml\n" + PREREG + "```\n````\n")
    lab.commit()
    _refused(lab.run(), "prereg", "no top-level fenced yaml block")


def test_a_tilde_fence_is_a_fence(lab):
    _write(lab.readme, "# line\n\n~~~yaml\n" + PREREG + "~~~\n")
    lab.commit()
    assert lab.run().exit_code == 0


def test_a_second_yaml_document_is_blind_even_where_pyyaml_could_answer(lab):
    _write(lab.root / "system.yaml", SYSTEM + "---\nnotes: a second document\n")
    _blind(lab.run(), "system")


def test_a_duplicate_key_is_blind_not_last_one_wins(lab):
    _write(lab.root / "system.yaml", SYSTEM.replace("      leverage: 3\n", "      leverage: 3\n      leverage: 9\n"))
    _blind(lab.run(), "system")


# ── a book that pays funding gets results that include it ──────────────────


NOT_CHARGED = {"modelled": False, "source_path": None, "sha256": None}


def test_a_funding_book_whose_run_will_not_charge_funding_refuses(lab):
    lab.plan["engine"]["name"] = "vectorbt"
    lab.plan["funding"] = dict(NOT_CHARGED)
    _only_refusal(lab.run(), "funding[arm.yaml]", "quantbox will not charge funding (engine 'vectorbt'")


@pytest.mark.parametrize(
    ("content", "phrase"),
    [
        (b"", "is 0 byte(s) — a stub"),
        (b"stub", "is 4 byte(s) — a stub"),
        (b"not-a-parquet-file-at-all", "no PAR1 magic"),
        (None, "No such file"),
    ],
    ids=["zero-bytes", "four-byte-stub", "no-magic", "gone"],
)
def test_a_funding_file_that_cannot_be_funding_rates_refuses(lab, content, phrase):
    """quantbox's plan says a file EXISTS; a stub is not funding rates."""
    target = lab.root / "datasets" / "perp-daily" / "funding_rates.parquet"
    if content is None:
        target.unlink()
    else:
        target.write_bytes(content)
    lab.commit()
    _only_refusal(lab.run(), "funding[arm.yaml]", phrase)


def test_the_funding_engine_with_a_funding_input_passes_and_names_it(lab):
    said = _rows(lab.run(), preflight.PASS)["funding[arm.yaml]"]
    assert said.startswith("engine rsims charges funding from ") and "funding_rates.parquet (" in said


def test_funding_excluded_by_preregistration_passes_and_says_so(lab):
    lab.plan["funding"] = dict(NOT_CHARGED)
    _write(lab.readme, _readme(PREREG + "  funding:\n    mode: excluded\n    reason: overlay effect only\n"))
    lab.commit()
    report = lab.run()
    assert report.exit_code == 0, report.rows
    assert _rows(report, preflight.PASS)["funding[arm.yaml]"].startswith(
        "funding EXCLUDED by pre-registration: overlay effect only"
    )


def test_an_exclusion_with_no_reason_is_not_an_exclusion(lab):
    lab.plan["funding"] = dict(NOT_CHARGED)
    _write(lab.readme, _readme(PREREG + "  funding:\n    mode: excluded\n"))
    lab.commit()
    _only_refusal(lab.run(), "funding[arm.yaml]", "pre-register `funding:")


def test_a_book_that_pays_no_funding_owes_no_funding_row(lab):
    _spot(lab)
    report = lab.run()
    assert report.exit_code == 0 and not any("funding" in check for _, check, _ in report.rows)


# ── found on a real stamped line: placeholders, and the supported data path ─


@pytest.mark.parametrize(
    ("broken", "where"),
    [
        (
            STATED.replace(
                "timing: same-bar at the pinned engine",
                "timing: <what the pinned engine does>",
            ),
            "prereg.execution.timing",
        ),
        (
            STATED.replace("timing: same-bar at the pinned engine", "timing: TBD"),
            "prereg.execution.timing",
        ),
        (
            PREREG.replace('pass_rule: "all legs"', 'pass_rule: "TODO decide"'),
            "prereg.criterion.pass_rule",
        ),
        (PREREG + "  notes: <anything else the template left>\n", "prereg.notes"),
        (
            PREREG.replace(
                "    max_short_gross_share: 0.5\n",
                "    max_short_gross_share: 0.5\n    <key>: 1\n",
            ),
            "prereg.liveness.<key> (key)",
        ),
    ],
    ids=[
        "angle-brackets",
        "tbd",
        "todo",
        "a-field-no-check-reads",
        "placeholder-key",
    ],
)
def test_a_placeholder_left_anywhere_in_the_block_refuses_by_name(lab, broken, where):
    _write(lab.readme, _readme(broken))
    del lab.plan["execution"]  # so `execution.timing` is what the run stands on
    lab.commit()
    report = lab.run()
    _refused(report, "placeholders", where)


@pytest.mark.parametrize(
    "legit",
    [
        PREREG.replace('pass_rule: "all legs"', 'pass_rule: "MDD < 30% and Calmar > 1"'),
        PREREG.replace('pass_rule: "all legs"', 'pass_rule: "treated<baseline on all legs"'),
        PREREG.replace('pass_rule: "all legs"', 'pass_rule: "a<b and c>d"'),
        PREREG.replace("        threshold: 0.70", '        op: "<="\n        threshold: 0.70'),
        PREREG.replace("name: mdd", "name: TODOS-cleanup"),
        PREREG.replace("name: mdd", "name: xxx_momentum"),
        PREREG + '  execution:\n    lag0_reason: ""\n',
        PREREG.replace("        threshold: 0.70", '        threshold: 0.70\n        versus: ""'),
        PREREG + "  note:\n",
    ],
    ids=[
        "two-comparisons",
        "less-than-in-prose",
        "a<b-and-c>d",
        "op-le",
        "TODOS-word",
        "xxx_word",
        "empty-optional",
        "empty-versus",
        "null-note",
    ],
)
def test_a_legitimate_value_is_not_a_placeholder(lab, legit):
    _write(lab.readme, _readme(legit))
    lab.commit()
    report = lab.run()
    assert report.exit_code == 0, report.rows


@pytest.mark.parametrize(
    "template_text",
    [
        "<number>",
        "<a books[].name from system.yaml>",
        "<owner>/<prod-repo>",
        "docs/mandates/<book>.md",
        "<number in (0, 1]>",
        "<all legs | k of n arms>",
        "TBD-style window",
        "todo",
        "<k>",
    ],
)
def test_everything_the_scaffold_templates_emit_is_a_placeholder(template_text):
    assert preflight._placeholders({"field": template_text}, "prereg") == ["prereg.field"]


# ── run manifests: the version contract ────────────────────────────────────


def _manifest(lab: Lab, doc: dict, stamp: str = "20260102T000000Z") -> None:
    _write(lab.line / "artifacts" / "arm" / f"x__{stamp}" / "run_manifest.json", json.dumps(doc))


def test_a_run_manifest_of_a_major_this_plugin_reads_is_counted(lab):
    _manifest(lab, {"schema": "quantbox/run@1"})
    report = lab.run()
    assert report.exit_code == 0, report.rows
    assert _rows(report, preflight.PASS)["run-manifest"].startswith(
        "1 quantbox run manifest(s), every one quantbox/run@1"
    )


@pytest.mark.parametrize("schema", ["quantbox/run@2", "quantbox/run@0", "quantbox/run@one"])
def test_a_run_manifest_of_an_unknown_major_is_blind_never_read_as_the_nearest(lab, schema):
    _manifest(lab, {"schema": schema})
    report = lab.run()
    _blind(report, "run-manifest")
    assert schema in _rows(report, preflight.BLIND)["run-manifest"]


def test_a_manifest_that_claims_no_quantbox_schema_is_only_run_evidence(lab):
    _manifest(lab, {"legacy": True})
    report = lab.run()
    assert report.exit_code == 0, report.rows
    assert _rows(report, preflight.PASS)["run-manifest"].startswith("0 quantbox run manifest(s)")


# ── quantbox, asked in-process (replaces the plugin's stand-in CLI cases) ────


@pytest.fixture
def planned(lab, monkeypatch):
    """quantbox's real explain entry point answers with `lab.plans`, and records how it
    was asked: the config path it got, the cwd, and that it printed to stderr only."""
    calls: list[tuple[str, Path]] = []

    def explain_path(path):
        calls.append((path, Path.cwd()))
        print("quantbox chatter that must not reach the report")
        return copy.deepcopy(lab.plans[Path(path).name])

    monkeypatch.setattr(preflight._explain, "explain_path", explain_path)
    return calls


def test_the_default_path_asks_quantbox_for_the_resolved_config_from_the_line(lab, planned, capsys):
    report = preflight.run(lab.root, lab.line, [lab.config], resolve=lab.resolve)
    assert report.exit_code == 0, report.rows
    assert planned == [(str(lab.config.resolve()), lab.line.resolve())]
    captured = capsys.readouterr()
    assert "chatter" not in captured.out and "chatter" in captured.err
    assert Path.cwd() != lab.line.resolve(), "the cwd was not restored"


def test_an_unknown_explain_major_is_a_non_pass_exit_through_main(lab, planned, capsys):
    lab.plan["schema"] = "quantbox/explain@2"
    code = preflight.main(["--lab", str(lab.root), "--line", str(lab.line), str(lab.config)])
    out = capsys.readouterr().out
    assert code == 2 and "quantbox/explain@2" in out and "PREFLIGHT: COULD NOT CHECK" in out


def test_the_default_path_reports_a_run_quantbox_refuses_to_start(lab, planned):
    lab.plans["arm.yaml"] = {"schema": "quantbox/explain@1", "ok": False, "errors": ["dataset: not found"]}
    _only_refusal(
        preflight.run(lab.root, lab.line, [lab.config], resolve=lab.resolve),
        "config[arm.yaml]",
        "dataset: not found",
    )


def _datasets(lab: Lab, tmp_path, monkeypatch, pin: str | None) -> Path:
    """A real datasets root holding perp-daily, and a datasets.lock beside the line's configs."""
    import hashlib

    root = tmp_path / "datasets-root"
    _write(root / "perp-daily" / "manifest.yaml", "market: futures\n")
    (root / "perp-daily" / "prices.parquet").write_bytes(b"PAR1 perp prices PAR1")
    sha = hashlib.sha256(b"PAR1 perp prices PAR1").hexdigest()
    monkeypatch.setenv("QUANTBOX_DATASETS_ROOT", str(root))
    lock = lab.line / "datasets.lock"
    _write(lock, f"perp-daily: {pin if pin is not None else sha}\n" if pin != "" else "{}\n")
    lab.commit()
    return lock


def test_an_unresolved_by_name_read_is_settled_by_quantboxs_real_resolver(lab, tmp_path, monkeypatch):
    """No stand-in: `resolve_dataset` at the lock `lock_for_run` names, as `quantbox
    dataset resolve --config` answers it."""
    lock = _datasets(lab, tmp_path, monkeypatch, pin=None)
    lab.plan["dataset"].update(UNRESOLVED)
    report = lab.run(resolve=preflight.resolve_dataset)
    assert report.exit_code == 0, report.rows
    passed = _rows(report, preflight.PASS)
    assert f"`quantbox dataset resolve` binds the build {lock} pins" in passed["data[arm.yaml]"]
    assert passed["market[arm.yaml]"] == "perp-daily: perp (primary)"


@pytest.mark.parametrize(
    ("pin", "phrase"),
    [("", "NOT pinned"), ("e" * 64, "not the pinned build")],
    ids=["no-pin", "bytes-moved"],
)
def test_the_real_resolver_refuses_an_unpinned_or_moved_build(lab, tmp_path, monkeypatch, pin, phrase):
    _datasets(lab, tmp_path, monkeypatch, pin=pin)
    lab.plan["dataset"].update(UNRESOLVED)
    _only_refusal(lab.run(resolve=preflight.resolve_dataset), "data[arm.yaml]", phrase)


def test_the_real_resolver_that_cannot_find_the_dataset_is_blind(lab, tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTBOX_DATASETS_ROOT", str(tmp_path / "empty-root"))
    lab.plan["dataset"].update(UNRESOLVED)
    report = lab.run(resolve=preflight.resolve_dataset)
    _blind(report, "data[arm.yaml]")
    assert "raised DatasetResolveError" in _rows(report, preflight.BLIND)["data[arm.yaml]"]


def test_dataset_resolve_cli_and_preflight_read_one_lock_rule(lab, tmp_path):
    """`lock_for_run` is the one reader: an explicit `dataset_lock` wins, else the nearest lock."""
    from quantbox.dataset_lock import lock_for_run

    assert lock_for_run(lab.config) is None
    _write(lab.line / "datasets.lock", "{}\n")
    assert lock_for_run(lab.config) == str((lab.line / "datasets.lock").resolve())
    _write(lab.config, CONFIG.replace('dataset: "perp-daily"', 'dataset: "perp-daily"\n      dataset_lock: "x.lock"'))
    assert lock_for_run(lab.config) == "x.lock"


# ── the exit code, through the real entry point ────────────────────────────


def test_the_cli_exit_codes(lab, planned):
    from typer.testing import CliRunner

    from quantbox.cli import app

    def cli(*extra: str):
        return CliRunner().invoke(
            app, ["preflight", "--lab", str(lab.root), "--line", str(lab.line), *extra, str(lab.config)]
        )

    done = cli()
    assert done.exit_code == 0 and "PREFLIGHT: PASS" in done.stdout, done.stdout
    assert cli("--id", "nope").exit_code == 2
    assert cli("--no-such-option").exit_code == 2  # a usage error is never a pass
    (lab.root / "system.yaml").unlink()
    done = cli()
    assert done.exit_code == 1 and "PREFLIGHT: REFUSE" in done.stdout


@pytest.mark.slow
def test_the_installed_command_runs_the_gate(lab):
    """`quantbox preflight` from the console script: a refusal exits 1 and prints the report."""
    (lab.root / "system.yaml").unlink()
    exe = Path(sys.executable).with_name("quantbox")
    done = subprocess.run(
        [str(exe), "preflight", "--lab", str(lab.root), "--line", str(lab.line), str(lab.config)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 1 and "no system.yaml — cannot determine" in done.stdout, done.stdout + done.stderr
    assert done.stdout.rstrip().endswith("— do not run the backtest")


def test_an_unknown_run_manifest_is_said_even_when_the_preregistration_is_missing(lab):
    _manifest(lab, {"schema": "quantbox/run@2"})
    _write(lab.readme, "# line\n\nno block\n")
    lab.commit()
    report = lab.run()
    assert "run-manifest" in _rows(report, preflight.BLIND), report.rows


# ── kind: replication — anchored to the paper, not to a book (TOM-2231) ────

PAPER = """\
  kind: replication
  paper:
    ref: "10.1111/jofi.12513"
    sample: {start: 1926-07, end: 2015-12}
    reported:
      - metric: annualised_alpha
        value: 0.0486
        where: "Table 1, Mkt"
    reference_series: authors-mktrf-vm
    post_sample: {start: 2016-01, end: 2024-12}
"""
REPLICATION = PREREG.replace("  book: perp-book\n", PAPER)


@pytest.fixture
def replication(lab) -> Lab:
    _write(lab.readme, _readme(REPLICATION))
    lab.commit("a replication prereg")
    return lab


def test_a_replication_passes_without_a_book_and_says_what_did_not_apply(replication):
    report = replication.run()
    assert report.exit_code == 0, report.rows
    passed = _rows(report, preflight.PASS)
    assert "kind: replication" in passed["book"] and "no venue" in passed["book"]
    assert "1926-07..2015-12" in passed["paper"] and "2016-01..2024-12" in passed["paper"]
    assert "no book to match" in passed["primary-dataset"]
    assert "_for_long_only` floors do not bind it" in passed["liveness"]
    # the data pin and the timing still ran; nothing about a venue did
    for family in ("data[arm.yaml]", "execution[arm.yaml]", "prereg-committed", "criterion", "claim-test"):
        assert family in passed, family
    checks = {check for _, check, _ in report.rows}
    assert not checks & {"market[arm.yaml]", "shorts[arm.yaml]", "leverage[arm.yaml]", "funding[arm.yaml]"}


def test_a_replication_missing_its_sample_window_refuses(replication):
    _write(replication.readme, _readme(REPLICATION.replace("    sample: {start: 1926-07, end: 2015-12}\n", "")))
    replication.commit()
    _only_refusal(replication.run(), "paper", "`paper.sample` must be a mapping")


@pytest.mark.parametrize(
    ("old", "new", "phrase"),
    [
        ('ref: "10.1111/jofi.12513"', "ref: Moreira and Muir", "not a DOI"),
        ("end: 2024-12}", "end: 2018-11}", "shorter than 3 years"),
        ("post_sample: {start: 2016-01", "post_sample: {start: 2015-06", "not right after the paper's sample"),
        ("post_sample: {start: 2016-01", "post_sample: {start: 2018-01", "a gap chooses which years count"),
        ("end: 2024-12}", "end: 2017-36}", "not YYYY-MM"),
        ("end: 2015-12}", "end: 2015-13}", "not YYYY-MM"),
        ("sample: {start: 1926-07, end: 2015-12}", "sample: {start: 1926, end: 2015-12}", "not YYYY-MM"),
        ("        value: 0.0486\n", '        value: "4.86%"\n', "not a finite number"),
        ("reference_series: authors-mktrf-vm", 'reference_series: ""', "leave it out, or name the series"),
    ],
    ids=[
        "ref",
        "post-sample-short",
        "post-sample-overlaps",
        "post-sample-gap",
        "month-36",
        "month-13",
        "sample-unparseable",
        "reported-text",
        "series-empty",
    ],
)
def test_a_replication_whose_paper_declaration_is_wrong_refuses_by_name(replication, old, new, phrase):
    _write(replication.readme, _readme(REPLICATION.replace(old, new)))
    replication.commit()
    _only_refusal(replication.run(), "paper", phrase)


def test_a_post_sample_of_exactly_three_years_passes(replication):
    _write(replication.readme, _readme(REPLICATION.replace("end: 2024-12}", "end: 2018-12}")))
    replication.commit()
    assert replication.run().exit_code == 0


def test_a_replication_without_reported_numbers_refuses(replication):
    text = REPLICATION.replace(
        '    reported:\n      - metric: annualised_alpha\n        value: 0.0486\n        where: "Table 1, Mkt"\n', ""
    )
    _write(replication.readme, _readme(text))
    replication.commit()
    _only_refusal(replication.run(), "paper", "`paper.reported` must list")


def test_a_replication_that_also_names_a_book_refuses(replication):
    _write(
        replication.readme,
        _readme(REPLICATION.replace("  kind: replication\n", "  kind: replication\n  book: perp-book\n")),
    )
    replication.commit()
    _only_refusal(replication.run(), "book", "a replication is anchored to its `paper:`")


def test_a_replication_with_no_paper_cannot_be_determined_and_refuses(lab):
    _write(lab.readme, _readme(PREREG.replace("  book: perp-book\n", "  kind: replication\n")))
    lab.commit()
    _only_refusal(lab.run(), "book", "what it replicates cannot be determined")


def test_any_other_kind_keeps_the_book_rule(lab):
    _write(lab.readme, _readme(REPLICATION.replace("kind: replication", "kind: replicaton")))
    lab.commit()
    _only_refusal(lab.run(), "book", "not declared in system.yaml")


def test_a_replication_is_still_held_to_the_lab_liveness_floor_and_the_data_pin(replication):
    _write(replication.readme, _readme(REPLICATION.replace("max_short_gross_share: 0.5", "max_short_gross_share: 0.7")))
    replication.commit()
    replication.plan["dataset"].update(tier="cache")
    report = replication.run()
    _refused(report, "liveness", "looser than the lab's 0.6")
    _refused(report, "data[arm.yaml]", "NOT pinned")


def test_the_paper_declaration_is_frozen(replication):
    replication.add_run("20260102T000000Z")
    _write(replication.readme, _readme(REPLICATION.replace("value: 0.0486", "value: 0.0350")))
    replication.commit("move the paper's reported number after the run", date=T2)
    _only_refusal(replication.run(), "prereg-committed", "post-registration")
