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
    replica_parity: findings/parity.md
  - name: spot-book
    market: spot
    venue: {allow_shorts: false, leverage: 1, funding_modelled: false}
    primary_dataset: spot-daily
    replica_parity: findings/parity.md
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

#: A green replica-parity finding (TOM-1412 point 2): confirmed, reviewed, UPHELD.
PARITY = (
    "---\nverdict: confirmed\nreview:\n  status: done\n  verdict: UPHELD\n---\n\nThe replica holds what live holds.\n"
)
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
    _write(made.root / "findings" / "parity.md", PARITY)
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
        "replica-parity",
        "binding",
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


def test_leverage_above_the_live_cap_is_recorded_not_refused(lab):
    """TOM-1119: venue.leverage is a PROMOTION limit, not a research cap. The arm runs; its
    finding carries the flag and is not promotable (exit 3, never 0)."""
    lab.plan["venue"]["max_leverage"] = 5
    report = lab.run()
    assert report.exit_code == 3, report.rows
    assert list(_rows(report, preflight.FLAG)) == ["leverage[arm.yaml]"]
    flagged = _rows(report, preflight.FLAG)["leverage[arm.yaml]"]
    assert "ABOVE LIVE CAP" in flagged and "5 > perp-book venue.leverage 3" in flagged and "NOT PROMOTABLE" in flagged
    out = preflight.render(report)
    assert "finding limits: leverage[arm.yaml]" in out and "PREFLIGHT: FLAG (1)" in out


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
    _write(made.root / "findings" / "parity.md", PARITY)
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
    lab.plan["execution"].update(lag_bars=0, same_bar=True)  # a refusal (leverage is a FLAG since TOM-1119)
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
    _write(made.root / "findings" / "parity.md", PARITY)
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


def test_a_config_path_relative_to_the_callers_cwd_keeps_its_meaning(lab, planned, tmp_path, monkeypatch):
    """PR #319 review, finding 1: a relative config path was resolved AFTER the switch to
    the line, so `research/line/configs/arm.yaml` became `.../line/research/line/...` and an
    existing config read as BLIND. Both quantbox calls get the caller's path."""
    lock = _datasets(lab, tmp_path, monkeypatch, pin=None)
    lab.plan["dataset"].update(UNRESOLVED)
    monkeypatch.chdir(lab.root)
    relative = lab.config.relative_to(lab.root)
    assert not (lab.line / relative).exists(), "the fixture would not tell the two meanings apart"
    report = preflight.run(lab.root, lab.line, [relative], resolve=preflight.resolve_dataset)
    assert report.exit_code == 0, report.rows
    assert planned == [(str(lab.config.resolve()), lab.line.resolve())]
    assert f"binds the build {lock} pins" in _rows(report, preflight.PASS)["data[arm.yaml]"]


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
    data:
      returns: total
      classes:
        - {class: equity, instruments: 1, type: index, start: 1926-07}
        - {class: bond, instruments: 1, type: index, start: 1926-07}
  data:
    returns: total
    classes:
      - {class: equity, type: index, instruments: [MKT]}
      - {class: bond, type: index, instruments: [BND]}
    anchor:
      file: results/anchor.csv
      min_corr: 0.5
      pairs:
        - {class: equity, ours: ours_eq, reference: ref_eq}
        - {class: bond, ours: ours_fi, reference: ref_fi}
"""
REPLICATION = PREREG.replace("  book: perp-book\n", PAPER)
CONFIG_HASH = "c0ffee12"


def monthly_prices(start: str = "1926-07-31", ends: dict | None = None):
    """Month-end prices of MKT and BND, smooth enough that no log return is a 10-sigma jump.
    `ends` cuts a series short ({"BND": "2024-12-31"}); BND ends a month before MKT by
    default, so the composition changes (no survivorship flag)."""
    import numpy as np
    import pandas as pd

    ends = {"BND": "2024-12-31", **(ends or {})}
    index = pd.date_range(start, "2025-01-31", freq="ME")
    k = np.arange(len(index))
    frame = pd.DataFrame(
        {"MKT": 100 * np.exp(0.006 * k + 0.02 * np.sin(k)), "BND": 50 * np.exp(0.003 * k + 0.01 * np.cos(k))},
        index=index,
    )
    for name, end in ends.items():
        frame.loc[frame.index > pd.Timestamp(end), name] = np.nan
    return frame


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parquet(path: Path, frame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.rename_axis("date").reset_index().to_parquet(path, index=False)
    return _sha(path)


def serve_data(
    lab: Lab, frame, kind: str = "prices", run_frame=None, stamp: str = "20260102T000000Z", inline: str | None = None
) -> Path:
    """The replication's primary and ONE data-probe run of its config: the run_manifest.json
    names the config and the data, the data_validation.json is quantbox's own calendar
    report (`instrument_calendar`) of what the run read (`run_frame`, default: what quantbox
    serves). `kind` prices|returns: a by-name dataset whose manifest `serve` block quantbox
    reads itself (a returns table beside the pinned prices.parquet). `inline`: a file read by
    path, `data/<inline>` (.parquet or .csv, written as given), declared in data_inputs.yaml."""
    from quantbox.dataset import served_prices
    from quantbox.instrument_calendar import DATA_VALIDATION_SCHEMA, instrument_calendar

    if inline is not None:
        path = lab.line / "data" / inline
        path.parent.mkdir(parents=True, exist_ok=True)
        table = frame if "date" in frame.columns else frame.rename_axis("date").reset_index()
        table.to_csv(path, index=False) if inline.endswith(".csv") else table.to_parquet(path, index=False)
        pin = _sha(path)
        lab.plan["dataset"] = {"name": inline, "sha256": pin, "source": "inline", "tier": "raw", "market": None}
        rows = f"inputs:\n  - path: data/{inline}\n    sha256: {pin}\n    dataset: perp-daily\n    market: futures\n"
        _write(lab.line / "data_inputs.yaml", rows)
        record = {"name": inline, "sha256": pin, "source": "inline"}
        served = run_frame
    else:
        home = lab.root / "datasets" / "perp-daily"
        pin = _parquet(home / "prices.parquet", frame if kind == "prices" else monthly_prices())
        name = "prices" if kind == "prices" else "returns"
        sha = pin if kind == "prices" else _parquet(home / "returns.parquet", frame)
        serve = {"table": name, "kind": kind, "frequency": "1M", "columns": None}
        serve.update(file=str(home / f"{name}.parquet"), sha256=sha)
        resolved = {"name": "perp-daily", "path": str(home), "sha256": pin, "lock": None, "serve": serve}
        lab.plan["dataset"].update(sha256=pin, resolved=resolved)
        record = {"name": "perp-daily", "sha256": pin, "source": "lock", "resolved": resolved}
        served = served_prices(resolved) if run_frame is None else run_frame
    lab.plan["config_hash"] = CONFIG_HASH
    run = lab.line / "artifacts" / "arm" / f"probe__{stamp}"
    manifest = {"schema": "quantbox/run@1", "config": {"sha256": CONFIG_HASH + "0" * 56}, "dataset": record}
    _write(run / "run_manifest.json", json.dumps(manifest))
    report = instrument_calendar(served).report
    _write(run / "data_validation.json", json.dumps({"schema": DATA_VALIDATION_SCHEMA, "calendar": report}))
    return run / "data_validation.json"


def anchor_csv(lab: Lab, rows: int = 30, text: str | None = None) -> None:
    """Monthly returns of the line's class series beside the public ones, tracking them."""
    import math

    if text is None:
        lines = ["date,ours_eq,ref_eq,ours_fi,ref_fi"]
        for n in range(rows):
            eq, fi = math.sin(n), math.cos(1.7 * n)
            lines.append(f"{1990 + n // 12}-{n % 12 + 1:02d},{eq:.4f},{0.9 * eq + 0.05:.4f},{fi:.4f},{fi:.4f}")
        text = "\n".join(lines) + "\n"
    _write(lab.line / "results" / "anchor.csv", text)


@pytest.fixture
def replication(lab) -> Lab:
    _write(lab.readme, _readme(REPLICATION))
    serve_data(lab, monthly_prices())
    anchor_csv(lab)
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
    # the data step ran, every row of it (TOM-1430)
    for family in ("data-validation", "data-coverage", "data-type", "data-hygiene", "data-anchor"):
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


# ── FLAG: a limit the finding carries (TOM-1119, TOM-1412 point 2) ──────────


@pytest.mark.parametrize(
    ("value", "least", "counts"),
    [
        (0, 0, True),
        (3, 1, True),
        (0, 1, False),
        (-1, 0, False),
        (-1, -5, False),  # `least` never goes below 0
        (True, 0, False),
        (False, 0, False),
        (1.0, 0, False),
        ("1", 0, False),
        (None, 0, False),
    ],
)
def test_one_count_contract_an_int_at_least_zero_never_a_bool(value, least, counts):
    """qute-plugins #297 round 2: `jumps: -1` passed as measured jumps. One contract, no caller-side bounds."""
    assert preflight._is_count(value, least=least) is counts


def test_a_flag_exits_3_and_never_reads_as_a_pass():
    report = preflight.Report()
    report.add(preflight.PASS, "a", "fine")
    report.add(preflight.FLAG, "b", "limited")
    assert report.exit_code == 3
    report.add(preflight.BLIND, "c", "could not look")
    assert report.exit_code == 2
    report.add(preflight.REFUSE, "d", "no")
    assert report.exit_code == 1


def test_a_book_without_a_replica_parity_finding_is_flagged_not_promotable(lab):
    _write(lab.root / "system.yaml", SYSTEM.replace("    replica_parity: findings/parity.md\n", ""))
    report = lab.run()
    assert report.exit_code == 3, report.rows
    assert "declares no `replica_parity`" in _rows(report, preflight.FLAG)["replica-parity"]
    assert "NOT PROMOTABLE" in _rows(report, preflight.FLAG)["replica-parity"]
    assert "finding limits: replica-parity" in preflight.render(report)


@pytest.mark.parametrize(
    ("finding", "phrase"),
    [
        (PARITY.replace("verdict: confirmed", "verdict: inconclusive"), "verdict 'inconclusive'"),
        (PARITY.replace("verdict: UPHELD", "verdict: OVERTURNED"), "'done' / 'OVERTURNED'"),
        (PARITY.replace("status: done", "status: pending"), "'pending'"),
        ("no frontmatter at all\n", "cannot be read"),
    ],
    ids=["not-confirmed", "not-upheld", "not-reviewed", "unreadable"],
)
def test_a_replica_parity_finding_that_is_not_green_is_a_flag(lab, finding, phrase):
    _write(lab.root / "findings" / "parity.md", finding)
    report = lab.run()
    assert report.exit_code == 3, report.rows
    assert phrase in _rows(report, preflight.FLAG)["replica-parity"]


# ── binding: can the rule change the weights at all (TOM-1412 point 1) ──────

BOUND = PREREG.replace("        metric: max_drawdown\n", "        metric: max_drawdown\n        versus: baseline\n") + (
    "  binding:\n"
    "    min_share: 0.2\n"
    "    baseline: configs/baseline.yaml\n"
    "    baseline_run: artifacts/bind/base__20260102T000000Z\n"
    "    arm_run: artifacts/bind/arm__20260102T000000Z\n"
)
BASE_HASH = "ba5e1111"


def weights(changed: int = 3, bars: int = 10, index=None):
    """A traded-weights frame; the arm moves the first `changed` bars' weights."""
    import pandas as pd

    index = pd.date_range("2024-01-01", periods=bars, freq="D") if index is None else index
    base = pd.DataFrame({"A": [0.5] * len(index), "B": [0.5] * len(index)}, index=index)
    arm = base.copy()
    arm.iloc[:changed, 0] = 0.25
    return base, arm


def bound(lab: Lab, base, arm, base_pin: str | None = None, base_schema: str = "quantbox/run@1") -> Lab:
    """A rule measured versus its baseline, with the two runs on disk: each run's manifest
    names its own config and pin, and its traded weights as quantbox writes them."""
    _write(lab.readme, _readme(BOUND))
    _write(lab.line / "configs" / "baseline.yaml", CONFIG)
    lab.plans["baseline.yaml"] = make_plan(lab)
    lab.plans["baseline.yaml"]["config_hash"] = BASE_HASH
    lab.plan["config_hash"] = CONFIG_HASH
    if base_pin is not None:
        lab.plans["baseline.yaml"]["dataset"]["sha256"] = base_pin
    lab.commit("a rule against its baseline")
    pin = lab.plan["dataset"]["sha256"]
    for name, frame, config, data, schema in (
        ("base", base, BASE_HASH, base_pin or pin, base_schema),
        ("arm", arm, CONFIG_HASH, pin, "quantbox/run@1"),
    ):
        run = lab.line / "artifacts" / "bind" / f"{name}__20260102T000000Z"
        _parquet(run / "traded_weights.parquet", frame)
        manifest = {
            "schema": schema,
            "config": {"sha256": config + "0" * 56},
            "dataset": {"name": "perp-daily", "sha256": data, "source": "lock"},
            "files": {"traded_weights": "traded_weights.parquet"},
        }
        _write(run / "run_manifest.json", json.dumps(manifest))
    return lab


def test_a_rule_against_a_baseline_with_no_binding_block_refuses(lab):
    _write(lab.readme, _readme(BOUND.split("  binding:\n")[0]))
    lab.commit()
    _only_refusal(lab.run(), "binding", "pre-registers no `binding")


@pytest.mark.parametrize(
    "broken",
    ["    min_share: 0\n", "    min_share: 1.5\n", "    min_share: true\n", "    min_share: 0.2\n    tol: -1\n"],
    ids=["zero", "above-one", "a-bool", "negative-tol"],
)
def test_a_binding_share_or_tolerance_that_is_not_one_refuses(lab, broken):
    bound(lab, *weights(changed=3))
    _write(lab.readme, _readme(BOUND.replace("    min_share: 0.2\n", broken)))
    lab.commit()
    _only_refusal(lab.run(), "binding", "with min_share in (0, 1] and a tol >= 0 (got min_share")


def test_an_overlay_needs_binding_too(lab):
    lab.plan["overlays"] = [{"name": "overlay.vol_target.v1"}]
    _only_refusal(lab.run(), "binding", "a config applies overlay(s) ['overlay.vol_target.v1']")


def test_a_rule_that_changes_enough_bars_passes_and_counts_them(lab):
    report = bound(lab, *weights(changed=3)).run()
    assert report.exit_code == 0, report.rows
    assert (
        "changes the weights on 3/10 baseline bars (30.0%) >= min_share 20.0%"
        in _rows(report, preflight.PASS)["binding"]
    )


def test_a_rule_that_never_changes_the_weights_refuses(lab):
    """H23: four arms of a correlation-gated cap, bit-identical, 0 of 1449 bars."""
    _only_refusal(bound(lab, *weights(changed=0)).run(), "binding", "0/10 baseline bars (0.0%) < the pre-registered")


def test_a_baseline_and_an_arm_on_different_datasets_cannot_be_compared(lab):
    """qute-plugins #296 round 3: each run matched ITS config's pin, the pair was never
    compared, and a change of data counted as the rule's effect (10/10 bars, exit 0)."""
    base, arm = weights(changed=0)
    arm.iloc[:, 0] = 0.3  # every bar differs — because the data differs, not the rule
    _only_refusal(bound(lab, base, arm, base_pin="d" * 64).run(), "binding", "not the same data")


def test_two_runs_served_different_tables_from_one_pin_cannot_be_compared(lab):
    """PR #321 review, finding 101: both runs read the same pinned prices.parquet, but each
    config's `serve` block served another returns table, and 10/10 changed bars passed."""
    bound(lab, *weights(changed=3))
    pin = lab.plan["dataset"]["sha256"]
    for config, run, sha in (("baseline.yaml", "base", "1" * 64), ("arm.yaml", "arm", "2" * 64)):
        serve = {"table": "returns", "kind": "returns", "file": "/x/returns.parquet", "sha256": sha, "columns": None}
        resolved = {"name": "perp-daily", "path": "/x", "sha256": pin, "serve": serve}
        lab.plans[config]["dataset"]["resolved"] = resolved
        record = lab.line / "artifacts" / "bind" / f"{run}__20260102T000000Z" / "run_manifest.json"
        manifest = json.loads(record.read_text())
        manifest["dataset"]["resolved"] = resolved
        record.write_text(json.dumps(manifest))
    _only_refusal(lab.run(), "binding", "not the same data")


@pytest.mark.parametrize("how", ["shifted", "reordered", "a-bar-twice", "shorter"])
def test_two_runs_not_on_the_identical_bar_index_cannot_be_compared(lab, how):
    import pandas as pd

    base, arm = weights(changed=3)
    if how == "shifted":
        arm.index = arm.index + pd.Timedelta(days=1)
    elif how == "reordered":
        arm = arm.iloc[::-1]
    elif how == "a-bar-twice":
        arm = pd.concat([arm.iloc[:9], arm.iloc[[8]]])
    else:
        arm = arm.iloc[:9]
    _only_refusal(bound(lab, base, arm).run(), "binding", "cannot determine")


def test_bars_are_counted_as_bars_whatever_the_label(lab):
    """Nine hourly bars and one daily bar: ten bars, never two calendar days."""
    import pandas as pd

    index = pd.DatetimeIndex([*pd.date_range("2024-01-01", periods=9, freq="h"), pd.Timestamp("2024-01-02")])
    report = bound(lab, *weights(changed=9, index=index)).run()
    assert "9/10 baseline bars" in _rows(report, preflight.PASS)["binding"], report.rows


def test_a_weight_that_is_not_a_number_is_not_a_flat_position(lab):
    base, arm = weights(changed=3)
    arm.iloc[5, 1] = float("nan")
    _only_refusal(bound(lab, base, arm).run(), "binding", "not finite numbers")


def test_a_binding_run_of_an_unknown_manifest_major_refuses(lab):
    _only_refusal(bound(lab, *weights(changed=3), base_schema="quantbox/run@999").run(), "binding", "quantbox/run@999")


def test_a_run_of_another_config_is_not_the_arm(lab):
    bound(lab, *weights(changed=3))
    lab.plan["config_hash"] = "feedbeef"
    _only_refusal(lab.run(), "binding", "another config's run")


# ── a replication's data step (TOM-1430) ───────────────────────────────────


def test_a_replication_with_no_data_block_refuses(replication):
    start = REPLICATION.index("  data:\n    returns: total\n")
    end = REPLICATION.index("reference: ref_fi}\n") + len("reference: ref_fi}\n")
    _write(replication.readme, _readme(REPLICATION[:start] + REPLICATION[end:]))
    replication.commit()
    _only_refusal(replication.run(), "data-validation", "no `data:` block")


def test_a_replication_with_no_data_probe_run_refuses(replication):
    import shutil

    shutil.rmtree(replication.line / "artifacts")
    _only_refusal(replication.run(), "data-validation", "run the data probe first")


def test_a_run_record_of_another_config_is_not_the_primarys(replication):
    replication.plan["config_hash"] = "feedbeef"
    _only_refusal(replication.run(), "data-validation", "another config's run")


def test_a_run_record_of_an_unknown_major_refuses(replication):
    record = next((replication.line / "artifacts").rglob("data_validation.json"))
    record.write_text(record.read_text().replace("data-validation@1", "data-validation@2"))
    _only_refusal(replication.run(), "data-validation", "quantbox/data-validation@2")


WEEKDAY = (
    REPLICATION.replace("sample: {start: 1926-07, end: 2015-12}", "sample: {start: 2015-01, end: 2020-12}")
    .replace("post_sample: {start: 2016-01, end: 2024-12}", "post_sample: {start: 2021-01, end: 2024-03}")
    .replace("start: 1926-07}", "start: 2015-01}")
)


def weekday_prices(last: str, holidays: tuple[str, ...] = ("2023-04-07", "2023-12-25")):
    """Exchange weekday prices from 2014 through `last`, minus the exchange's holidays."""
    import numpy as np
    import pandas as pd

    index = pd.bdate_range("2014-01-01", last).difference(pd.DatetimeIndex(list(holidays)))
    k = np.arange(len(index))
    return pd.DataFrame({"MKT": 100 * np.exp(0.0003 * k), "BND": 50 * np.exp(0.0001 * k)}, index=index)


@pytest.mark.parametrize(
    ("last", "holidays", "status", "phrase"),
    [
        ("2024-03-28", ("2023-04-07", "2023-12-25"), preflight.PASS, "2014-01-01..2024-03-28"),
        ("2024-03-27", ("2023-04-07", "2023-12-25"), preflight.REFUSE, "ends 4 day(s) before"),
        ("2024-03-28", (), preflight.REFUSE, "at most 3 day(s) apart"),
    ],
    ids=["good-friday-holiday-in-its-own-bars", "a-day-short-of-it", "no-holiday-in-its-bars"],
)
def test_a_window_end_is_judged_on_the_series_own_observed_bars(replication, last, holidays, status, phrase):
    """codex P2 on qute-plugins #297: a complete US-equity series through Thursday 2024-03-28
    covers a post-sample that ends 2024-03 — 29 March was Good Friday. A computed weekday
    calendar demanded the 29th and refused. The series' own bars say a 4-day gap happens."""
    _write(replication.readme, _readme(WEEKDAY))
    frame = weekday_prices(last, holidays)
    serve_data(replication, frame)
    replication.commit()
    report = replication.run()
    row = next(r for st, c, r in report.rows if c == "data-coverage" and st == status)
    assert phrase in row, report.rows


@pytest.mark.parametrize(
    ("ends", "phrase"),
    [({"MKT": "2024-11-30", "BND": "2024-11-30"}, "ends 31 day(s) before the post-sample end 2024-12-31")],
)
def test_a_monthly_series_a_month_short_refuses(replication, ends, phrase):
    serve_data(replication, monthly_prices(ends=ends))
    _refused(replication.run(), "data-coverage", phrase)


def test_a_calendar_day_series_two_days_short_refuses(replication):
    import numpy as np
    import pandas as pd

    index = pd.date_range("2010-01-01", "2024-12-29", freq="D")
    k = np.arange(len(index))
    serve_data(replication, pd.DataFrame({"MKT": 100 * np.exp(1e-4 * k), "BND": 50 * np.exp(5e-5 * k)}, index=index))
    _refused(replication.run(), "data-coverage", "ends 2 day(s) before the post-sample end 2024-12-31")


def returns_of(frame):
    return frame.pct_change(fill_method=None).iloc[1:]


def test_a_returns_table_is_measured_as_returns_not_as_prices(replication):
    """The served table of a `serve.kind: returns` dataset holds returns: half of them are
    negative. Read as prices, every one of those is a non-positive PRICE (a false refusal);
    measured as returns, a price is non-positive only at a return <= -100 %."""
    served = returns_of(monthly_prices(start="1926-06-30"))
    assert (served < 0).any().all(), "the fixture would not tell the two readings apart"
    serve_data(replication, served, kind="returns")
    report = replication.run()
    assert report.exit_code == 0, report.rows
    assert "the served returns table" in _rows(report, preflight.PASS)["data-hygiene"]


def test_a_table_quantbox_refuses_to_serve_refuses_with_quantboxs_reason(replication):
    """A return at or below -100 % is a non-positive total-return price. quantbox refuses
    to serve it, and the gate says so in quantbox's words, never in its own."""
    from quantbox.dataset import served_prices

    clean = returns_of(monthly_prices(start="1926-06-30"))
    served = clean.copy()
    served.iloc[40, 0] = -1.2
    serve_data(replication, clean, kind="returns")
    run_frame = served_prices(replication.plan["dataset"]["resolved"])
    serve_data(replication, served, kind="returns", run_frame=run_frame)
    _refused(replication.run(), "data-hygiene", "quantbox would not serve it: DatasetServeError")


def test_hygiene_measures_the_bytes_quantbox_serves_not_a_file_that_moved(replication):
    path = replication.root / "datasets" / "perp-daily" / "prices.parquet"
    frame = monthly_prices()
    frame.iloc[5, 0] = -3.0
    _parquet(path, frame)  # the bytes on disk are no longer the pinned build
    report = replication.run()
    _refused(report, "data-hygiene", "not the bytes it was resolved at")
    _refused(report, "data-coverage", "cannot determine")


def test_a_plan_that_names_no_served_table_cannot_be_measured(replication):
    replication.plan["dataset"].pop("resolved")
    record = next((replication.line / "artifacts").rglob("run_manifest.json"))
    manifest = json.loads(record.read_text())
    manifest["dataset"].pop("resolved")
    record.write_text(json.dumps(manifest))
    report = replication.run()
    _refused(report, "data-hygiene", "a plan quantbox cannot name a served table for")
    _refused(report, "data-coverage", "cannot determine")


def test_a_non_finite_price_read_by_path_is_counted_and_refuses(replication):
    frame = monthly_prices()
    frame.iloc[100, 1] = float("inf")
    serve_data(replication, frame, run_frame=monthly_prices(), inline="prices.parquet")
    _refused(replication.run(), "data-hygiene", "BND (0 <=0, 1 non-finite)")


def test_a_run_served_another_table_than_its_config_serves_is_another_datas_run(replication):
    """One pin can serve two tables (a `serve` block names its own): the run record must
    have been served the table the config serves, not only read the same pin."""
    record = next((replication.line / "artifacts").rglob("run_manifest.json"))
    manifest = json.loads(record.read_text())
    manifest["dataset"]["resolved"]["serve"]["sha256"] = "e" * 64
    record.write_text(json.dumps(manifest))
    _only_refusal(replication.run(), "data-validation", "another data's run")


@pytest.mark.parametrize(
    ("inline", "shape"), [("prices.csv", "wide"), ("prices.parquet", "long"), ("prices.csv", "long")]
)
def test_a_file_read_by_path_is_read_as_the_data_plugin_reads_it(replication, inline, shape):
    """PR #321 review, finding 1: the gate read parquet wide tables only, and refused a CSV
    or a long table the data plugin serves the run from."""
    frame = monthly_prices()
    if shape == "long":
        table = frame.rename_axis("date").reset_index().melt(id_vars="date", var_name="symbol", value_name="close")
        table = table.dropna()
    else:
        table = frame
    serve_data(replication, table, run_frame=frame, inline=inline)
    report = replication.run()
    assert report.exit_code == 0, report.rows


def test_a_dated_row_is_judged_on_its_own_calendar_date_in_any_timezone(replication):
    """PR #321 review, findings 2 and 102: Warsaw-midnight dates were converted to UTC, so
    the first bar became the day before, fell outside the run's window, and a -1 price on
    it passed. The data plugin labels each row with its own calendar date (TOM-1681)."""
    frame = monthly_prices()
    frame.iloc[0, 0] = -1.0
    zoned = frame.copy()
    zoned.index = zoned.index.tz_localize("Europe/Warsaw")
    serve_data(replication, zoned, run_frame=frame, inline="prices.parquet")
    _refused(replication.run(), "data-hygiene", "MKT (1 <=0, 0 non-finite)")


def test_an_instrument_the_served_table_never_prints_in_the_window_cannot_be_measured(replication):
    import numpy as np

    frame = monthly_prices()
    frame["BND"] = np.nan
    serve_data(replication, frame, run_frame=monthly_prices())
    _refused(replication.run(), "data-hygiene", "no served observation of ['BND']")


def test_a_jump_is_flagged_and_named(replication):
    frame = monthly_prices()
    frame.iloc[600:, 0] *= 40.0
    serve_data(replication, frame)
    report = replication.run()
    assert report.exit_code == 3, report.rows
    assert "jumps > 10.0 sigma: MKT 1" in _rows(report, preflight.FLAG)["data-hygiene"]


def test_data_of_another_type_than_the_papers_is_a_proxy_test(replication):
    _write(
        replication.readme,
        _readme(
            REPLICATION.replace(
                "{class: equity, instruments: 1, type: index,",
                "{class: equity, instruments: 1, type: futures, roll: back-adjusted,",
            )
        ),
    )
    replication.commit()
    report = replication.run()
    assert report.exit_code == 3, report.rows
    assert "PROXY TEST: equity: index for the paper's futures" in _rows(report, preflight.FLAG)["data-type"]


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        ("date,ours_eq,ref_eq,ours_fi,ref_fi\n" + "1990-01,1,1,1,1\n1990-02,2,2,2,2\n" * 12, "on more than one row"),
        ("ours_eq,ref_eq,ours_fi,ref_fi\n" + "1,1,1,1\n" * 30, "no readable `date`"),
    ],
    ids=["two-dates-twelve-times", "undated"],
)
def test_an_anchor_counts_dated_observations_once_each(replication, text, phrase):
    anchor_csv(replication, text=text)
    _only_refusal(replication.run(), "data-anchor", phrase)


def test_an_anchor_pair_of_one_column_against_itself_refuses(replication):
    _write(replication.readme, _readme(REPLICATION.replace("reference: ref_eq}", "reference: ours_eq}")))
    replication.commit()
    _only_refusal(replication.run(), "data-anchor", "a series against itself")


def test_a_named_run_record_is_read_through_main(replication, planned, capsys):
    """`--data-validation` names the record when the runs hold more than one."""
    record = next((replication.line / "artifacts").rglob("data_validation.json"))
    serve_data(replication, monthly_prices(), stamp="20260103T000000Z")
    argv = ["--lab", str(replication.root), "--line", str(replication.line), str(replication.config)]
    assert preflight.main(argv) == 1 and "name the one to read with --data-validation" in capsys.readouterr().out
    assert preflight.main(["--data-validation", str(record), *argv]) == 0, capsys.readouterr().out


# ── release review #322 (v0.21.0): three fail-opens, each reproduction a guard ──
#
# Finding 2 (TOM-2483): the PyYAML fallback read a number the subset parser refuses.
# Finding 3 (TOM-2548): claim-test counts passed as "a finite number" when invalid.
# Finding 101 (TOM-2548): a run dir the walk could not look into read as empty.

AMBIGUOUS = ["010", "0x10", "1_0", "1:30", "1e3", "1.0e3"]
#: An anchor sends system.yaml to the PyYAML fallback BEFORE the subset parser reaches a number.
ANCHORED = "base: &b {x: 1}\n"


@pytest.mark.parametrize("prefix", ["", ANCHORED], ids=["plain", "fallback-by-anchor"])
@pytest.mark.parametrize("spelling", AMBIGUOUS)
def test_an_ambiguous_live_cap_is_blind_on_every_yaml_path(lab, spelling, prefix):
    """The reproduction: `venue.leverage: 010` gave PASS "runner cap 3 <= live cap 8", exit 0."""
    _write(lab.root / "system.yaml", prefix + SYSTEM.replace("      leverage: 3\n", f"      leverage: {spelling}\n"))
    lab.commit()
    report = lab.run()
    _blind(report, "system")
    assert "YAML readers disagree about" in _rows(report, preflight.BLIND)["system"]
    assert not any(check.startswith("leverage[") for _, check, _ in report.rows)


def test_the_fallback_still_reads_a_clean_file_and_a_quoted_spelling_is_text(lab):
    """The control: the anchor alone falls back and passes; quoting `010` makes it text, and
    text is not a cap (refused by name, never read as 8)."""
    _write(lab.root / "system.yaml", ANCHORED + SYSTEM)
    lab.commit()
    assert lab.run().exit_code == 0
    _write(lab.root / "system.yaml", ANCHORED + SYSTEM.replace("      leverage: 3\n", "      leverage: '010'\n"))
    lab.commit()
    _refused(lab.run(), "system", "venue.leverage must be a finite number")


@pytest.mark.parametrize(
    "text",
    ["a: &x 1\nb: 010\n", "a: &x 1\nb: [1, 0x10]\n", "a: &x 1\n010: v\n", "010: v\n", "a: &x 010\nb: *x\n"],
    ids=["value-after-anchor", "in-flow", "key-after-anchor", "key", "anchored-number"],
)
def test_load_yaml_never_hands_back_an_ambiguous_number(tmp_path, text):
    path = tmp_path / "doc.yaml"
    _write(path, text)
    with pytest.raises(preflight.YamlRefused, match="YAML readers disagree about"):
        preflight.load_yaml(path)


@pytest.mark.parametrize(
    ("old", "new", "phrase"),
    [
        ("draws: 2000", "draws: -1", "claim_test.draws"),
        ("draws: 2000", "draws: 0", "claim_test.draws"),
        ("draws: 2000", "draws: 2000.5", "claim_test.draws"),
        ("draws: 2000", "draws: true", "claim_test.draws"),
        ("mean_block: 30", "mean_block: -0.5", "claim_test.mean_block"),
        ("mean_block: 30", "mean_block: 0", "claim_test.mean_block"),
        ("mean_block: 30", "mean_block: 0.5", "claim_test.mean_block"),
        ("mean_block: 30", "mean_block: .inf", "claim_test.mean_block"),
        ("seed: 7", "seed: -1", "claim_test.seed"),
        ("seed: 7", "seed: 7.5", "claim_test.seed"),
        ("seed: 7", "seed: true", "claim_test.seed"),
    ],
)
def test_a_bootstrap_count_outside_its_domain_refuses(lab, old, new, phrase):
    """The reproduction: `draws: -1`, `mean_block: -0.5`, `seed: -1` gave PASS claim-test, exit 0."""
    _write(lab.readme, _readme(PREREG.replace(old, new)))
    lab.commit()
    _only_refusal(lab.run(), "claim-test", phrase)


@pytest.mark.parametrize(
    ("old", "new"), [("seed: 7", "seed: 0"), ("mean_block: 30", "mean_block: 1"), ("mean_block: 30", "mean_block: 2.5")]
)
def test_a_bootstrap_count_at_the_edge_of_its_domain_passes(lab, old, new):
    _write(lab.readme, _readme(PREREG.replace(old, new)))
    lab.commit()
    assert lab.run().exit_code == 0


@pytest.mark.parametrize("floor", ["-2", "1.5"])
def test_an_anchor_floor_that_is_not_a_correlation_refuses(replication, floor):
    """Swept with finding 3: `min_corr: -2` is passed by every correlation, so it is no floor."""
    _write(replication.readme, _readme(REPLICATION.replace("min_corr: 0.5", f"min_corr: {floor}")))
    replication.commit()
    _only_refusal(replication.run(), "data-anchor", "a `min_corr` in [-1, 1]")


def _run_before_the_criterion(lab: Lab) -> Path:
    """The finding-101 setup: the declared root is `artifacts`, and a quantbox run dated
    2025-12-31 (before the criterion, committed 2026-01-01) sits in `artifacts/arm`."""
    lab.plan["artifacts_root"] = str(lab.line / "artifacts")
    arm = lab.line / "artifacts" / "arm"
    _write(arm / "x__20251231T000000Z" / "run_manifest.json", json.dumps({"schema": "quantbox/run@1"}))
    return arm


def test_a_run_before_the_criterion_refuses_the_control_for_finding_101(lab):
    _run_before_the_criterion(lab)
    report = lab.run()
    _only_refusal(report, "prereg-committed", "predates the criterion")
    assert "1 quantbox run manifest(s)" in _rows(report, preflight.PASS)["run-manifest"]


def test_a_symlinked_run_dir_is_blind_never_an_empty_one(lab, tmp_path):
    """The reproduction: the same run behind a symlink gave PASS, "0 evidence item(s) found"."""
    arm = _run_before_the_criterion(lab)
    elsewhere = tmp_path / "elsewhere" / "arm"
    elsewhere.parent.mkdir()
    arm.rename(elsewhere)
    arm.symlink_to(elsewhere, target_is_directory=True)
    report = lab.run()
    _blind(report, "prereg-committed")
    _blind(report, "run-manifest")
    for check in ("prereg-committed", "run-manifest"):
        said = _rows(report, preflight.BLIND)[check]
        assert str(arm) in said and "symlinked directory" in said, said


def test_a_visible_post_registration_still_refuses_beside_a_symlink(lab, tmp_path):
    """A refusal outranks blindness: what the walk DID see still refuses, and says what it could not."""
    target = tmp_path / "elsewhere"
    target.mkdir()
    (lab.line / "artifacts" / "arm").mkdir(parents=True)
    (lab.line / "artifacts" / "arm" / "linked").symlink_to(target, target_is_directory=True)
    (lab.line / "results" / "x__20251231T000000Z").mkdir(parents=True)
    report = lab.run()
    _refused(report, "prereg-committed", "predates")
    assert "symlinked directory" in _rows(report, preflight.REFUSE)["prereg-committed"]


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a chmod 000 directory")
def test_an_unreadable_run_dir_is_blind_never_an_empty_one(lab):
    """The reproduction: the same run under `chmod 000` gave PASS, "0 evidence item(s) found"."""
    arm = _run_before_the_criterion(lab)
    arm.chmod(0)
    try:
        report = lab.run()
    finally:
        arm.chmod(0o755)
    _blind(report, "prereg-committed")
    _blind(report, "run-manifest")
    for check in ("prereg-committed", "run-manifest"):
        assert str(arm) in _rows(report, preflight.BLIND)[check]


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a chmod 000 file")
def test_an_unreadable_run_manifest_is_blind_not_a_manifest_that_claims_no_schema(lab):
    manifest = lab.line / "artifacts" / "arm" / "x__20260102T000000Z" / "run_manifest.json"
    _write(manifest, json.dumps({"schema": "quantbox/run@9"}))
    manifest.chmod(0)
    try:
        report = lab.run()
    finally:
        manifest.chmod(0o644)
    _blind(report, "run-manifest")
    assert str(manifest) in _rows(report, preflight.BLIND)["run-manifest"]


def test_a_symlinked_dir_under_a_replications_runs_is_blind_for_its_data_step(replication, tmp_path):
    """The same walk finds the run's data_validation.json: a second one may lie behind the link."""
    target = tmp_path / "elsewhere" / "probe2"
    target.mkdir(parents=True)
    link = replication.line / "artifacts" / "arm" / "probe2"
    link.symlink_to(target, target_is_directory=True)
    report = replication.run()
    _blind(report, "data-validation")
    assert str(link) in _rows(report, preflight.BLIND)["data-validation"]


def test_a_pruned_symlink_is_not_blind(lab, tmp_path):
    """`.venv` is never walked outside a declared root, whether or not it is a symlink: the
    row already says it was not walked, and a lab venv often is one."""
    target = tmp_path / "venv"
    target.mkdir()
    (lab.line / ".venv").symlink_to(target, target_is_directory=True)
    report = lab.run()
    assert report.exit_code == 0, report.rows
    assert "never walked: ['.venv']" in _rows(report, preflight.PASS)["prereg-committed"]
