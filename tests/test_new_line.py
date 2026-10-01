"""`quantbox new line` and `quantbox line repin` (TOM-1366).

The fast tests run offline against throwaway local git repos standing in for quantbox.
The acceptance test (``slow``) is the ticket's criterion end to end, in a clean temp dir:
scaffold a line pinned to a snapshot of THIS working tree, ``uv sync``, one run, the
arms, the reproduction test, then a repin. It needs PyPI and read access to the
private quantbox-datasets repo, and skips (saying so) where that remote cannot be read.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from quantbox import line
from quantbox.cli import app

REPO = Path(__file__).resolve().parents[1]
FAKE_DATASETS_SHA = "1" * 40


def _git(cwd: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout


def _fake_quantbox(root: Path, tags: dict[str, str]) -> Path:
    """A git repo with one commit per tag: `src/quantbox/line.py` + a uv.lock pinning pandas to the tag's value."""
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    (root / "src/quantbox").mkdir(parents=True)
    (root / "src/quantbox/line.py").write_text("# line support\n")
    for tag, pandas_version in tags.items():
        (root / "uv.lock").write_text(
            "version = 1\n\n[[package]]\n"
            f'name = "pandas"\nversion = "{pandas_version}"\nsource = {{ registry = "https://pypi.org/simple" }}\n'
            '\n[[package]]\nname = "quantbox"\nversion = "0.1.0"\nsource = { editable = "." }\n'
        )
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", tag)
        _git(root, "tag", tag)
    return root


@pytest.fixture
def fake_qb(tmp_path: Path) -> Path:
    return _fake_quantbox(tmp_path / "qb", {"v0.9.0": "2.3.2", "v0.10.0": "2.3.3"})


@pytest.fixture
def datasets_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A one-dataset root (`demo-daily`) built from the committed canonical fixture."""
    root = tmp_path / "datasets"
    ds = root / "demo-daily"
    ds.mkdir(parents=True)
    shutil.copy(REPO / "cookbook/canonical/fixture.parquet", ds / "prices.parquet")
    shutil.copy(REPO / "cookbook/canonical/fixture_volume.parquet", ds / "volume.parquet")
    monkeypatch.setenv("QUANTBOX_DATASETS_ROOT", str(root))
    return root


def _scaffold(tmp_path: Path, fake_qb: Path, **kw) -> dict:
    kw.setdefault("directory", tmp_path / "demo")
    kw.setdefault("quantbox_url", f"file://{fake_qb}")
    kw.setdefault("datasets_ref", FAKE_DATASETS_SHA)
    kw.setdefault("dataset", "demo-daily")
    kw.setdefault("lock", False)
    return line.new_line("demo", **kw)


# ── pure helpers ──────────────────────────────────────────────────────────


def test_latest_tag_is_the_highest_semver_not_the_last_listed():
    assert line.pick_latest_tag(["v0.9.0", "v0.10.0", "v0.10.0rc1", "vfoo", "v0.2.11"]) == "v0.10.0"
    with pytest.raises(line.LineError, match="--quantbox-ref"):
        line.pick_latest_tag(["latest", "v1"])


def test_lock_versions_reads_quantboxs_own_lock_and_skips_split_and_non_registry_packages():
    pins = line.lock_versions((REPO / "uv.lock").read_text(encoding="utf-8"), exclude={"quantbox"})
    assert "quantbox" not in pins
    assert pins and all(re.match(r"^[0-9][0-9A-Za-z.+!-]*$", v) for v in pins.values())
    split = 'version = 1\n\n[[package]]\nname = "numpy"\nversion = "2.2.6"\nsource = { registry = "x" }\n'
    split += '\n[[package]]\nname = "numpy"\nversion = "2.3.1"\nsource = { registry = "x" }\n'
    split += '\n[[package]]\nname = "vbt"\nversion = "1.0"\nsource = { git = "https://x#abc" }\n'
    assert line.lock_versions(split) == {}


def test_replace_pins_rewrites_only_the_managed_block():
    text = f"[tool.uv]\npackage = false\n{line.render_pins({'a': '1'})}\n\n[tool.other]\nx = 1\n"
    out = line.replace_pins(text, {"b": "2", "a": "3"})
    assert '"a==3",' in out and '"b==2",' in out and '"a==1"' not in out
    assert out.endswith("[tool.other]\nx = 1\n")
    with pytest.raises(line.LineError, match="managed pins block"):
        line.replace_pins("[tool.uv]\n", {})


# ── new line (offline) ────────────────────────────────────────────────────


def test_new_line_writes_the_model_c_skeleton(tmp_path, fake_qb, datasets_root):
    result = _scaffold(tmp_path, fake_qb)
    target = tmp_path / "demo"
    for rel in [
        "pyproject.toml",
        "README.md",
        "config.yaml",
        "arms.yaml",
        "datasets.lock",
        "tests/test_reproduces.py",
        ".gitignore",
        ".python-version",
        "findings/.gitkeep",
    ]:
        assert (target / rel).is_file(), rel

    # quantbox pinned to the commit the LATEST tag names, the tag beside it.
    v10 = _git(fake_qb, "rev-parse", "v0.10.0").strip()
    assert result["quantbox"] == {"url": f"file://{fake_qb}", "ref": "v0.10.0", "sha": v10, "extras": "full"}
    pyproject = (target / "pyproject.toml").read_text()
    assert f'"quantbox[full] @ git+file://{fake_qb}@{v10}",  # v0.10.0' in pyproject
    assert f'quantbox-ref = "v0.10.0"\nquantbox-sha = "{v10}"' in pyproject
    assert f"quantbox-datasets.git@{FAKE_DATASETS_SHA}" in pyproject
    # the transitive pins come from quantbox's own lock at that ref
    assert '"pandas==2.3.3",' in pyproject
    assert 'requires-python = ">=3.12,<3.13"' in pyproject

    # datasets.lock pins the bytes quantbox would read
    from quantbox.dataset_lock import resolve_dataset

    pins = yaml.safe_load((target / "datasets.lock").read_text())
    assert pins == {"demo-daily": resolve_dataset("demo-daily", lock=None)["actual_sha256"]}

    # README carries frontmatter and a top-level prereg block naming the dataset
    readme = (target / "README.md").read_text()
    assert readme.startswith("---\nline: demo\n")
    block = re.search(r"```yaml\n(prereg:\n.*?)```", readme, re.S)
    assert block and "primary_dataset: demo-daily" in block.group(1)

    # the arms file loads with the quantbox that wrote it, against the scaffolded base
    from quantbox.arms import load_arms

    spec = load_arms(target / "arms.yaml")
    assert set(spec.overrides) == {"baseline", "signal-ma-20"}
    cfg = yaml.safe_load((target / "config.yaml").read_text())
    assert cfg["plugins"]["data"]["params_init"] == {"dataset": "demo-daily"}
    assert cfg["plugins"]["pipeline"]["params"]["execution"] == {"lag_bars": 1}
    assert "{{" not in "".join(p.read_text() for p in target.rglob("*") if p.is_file())


def test_new_line_validates_against_the_registry(tmp_path, fake_qb, datasets_root):
    """The base config is not a sketch: `quantbox validate` accepts it as written."""
    from quantbox.validate import validate_config

    _scaffold(tmp_path, fake_qb)
    cfg = yaml.safe_load((tmp_path / "demo/config.yaml").read_text())
    errors = [f.message for f in validate_config(cfg) if f.level == "error"]
    assert errors == []


def test_new_line_config_passes_validate_and_explain_cli(tmp_path, fake_qb, datasets_root, monkeypatch):
    """Both doors a user reaches first accept the scaffolded base config, run from the line."""
    _scaffold(tmp_path, fake_qb)
    monkeypatch.chdir(tmp_path / "demo")
    runner = CliRunner()
    res = runner.invoke(app, ["validate", "-c", "config.yaml"])
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["config", "explain", "config.yaml", "--json"])
    assert res.exit_code == 0, res.output
    plan = json.loads(res.stdout)
    assert plan["ok"] is True and plan["errors"] == []
    assert plan["dataset"]["name"] == "demo-daily"


def test_new_line_honours_an_explicit_ref_and_extras(tmp_path, fake_qb, datasets_root):
    result = _scaffold(tmp_path, fake_qb, quantbox_ref="v0.9.0", extras="")
    pyproject = (tmp_path / "demo/pyproject.toml").read_text()
    assert result["quantbox"]["ref"] == "v0.9.0"
    assert '"quantbox @ git+file://' in pyproject and '"pandas==2.3.2",' in pyproject


def test_new_line_refuses_a_ref_that_predates_line_support(tmp_path, datasets_root):
    old = tmp_path / "old"
    old.mkdir()
    _git(old, "init", "-q", "-b", "main")
    (old / "README.md").write_text("old\n")
    _git(old, "add", "-A")
    _git(old, "commit", "-q", "-m", "old")
    _git(old, "tag", "v0.7.0")
    with pytest.raises(line.LineError, match="predates `quantbox new line`"):
        _scaffold(tmp_path, old)
    assert not (tmp_path / "demo").exists() or not any((tmp_path / "demo").iterdir())


def test_new_line_refuses_a_non_empty_target_and_a_bad_slug(tmp_path, fake_qb, datasets_root):
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo/keep.txt").write_text("x")
    with pytest.raises(line.LineError, match="not empty"):
        _scaffold(tmp_path, fake_qb)
    with pytest.raises(line.LineError, match="slug"):
        line.new_line("Bad Slug", directory=tmp_path / "x")


def test_an_unresolvable_dataset_is_written_unpinned_and_said_so(tmp_path, fake_qb, datasets_root):
    result = _scaffold(tmp_path, fake_qb, dataset="no-such-dataset")
    assert result["dataset"]["sha256"] is None and "not found" in result["dataset"]["unpinned_reason"]
    assert yaml.safe_load((tmp_path / "demo/datasets.lock").read_text()) is None  # comments only, no pin


def test_cli_new_line_and_repin(tmp_path, fake_qb, datasets_root):
    runner = CliRunner()
    target = tmp_path / "demo"
    args = ["new", "line", "demo", "--dir", str(target), "--quantbox-url", f"file://{fake_qb}"]
    args += ["--quantbox-ref", "v0.9.0", "--datasets-ref", FAKE_DATASETS_SHA, "--dataset", "demo-daily"]
    res = runner.invoke(app, [*args, "--no-lock", "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["quantbox"]["ref"] == "v0.9.0"

    res = runner.invoke(app, ["line", "repin", str(target), "--ref", "v0.10.0", "--no-lock", "--json"])
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert out["from"]["ref"] == "v0.9.0" and out["to"]["ref"] == "v0.10.0"

    res = runner.invoke(app, [*args, "--no-lock"])  # target is no longer empty
    assert res.exit_code == 1 and "not empty" in res.output


# ── repin (offline) ───────────────────────────────────────────────────────


def test_repin_moves_sha_tag_and_every_derived_pin(tmp_path, fake_qb, datasets_root):
    _scaffold(tmp_path, fake_qb, quantbox_ref="v0.9.0")
    target = tmp_path / "demo"
    before = (target / "pyproject.toml").read_text()
    v9, v10 = (_git(fake_qb, "rev-parse", t).strip() for t in ("v0.9.0", "v0.10.0"))

    result = line.repin(target, lock=False)  # no --ref: the latest tag

    after = (target / "pyproject.toml").read_text()
    assert result["from"] == {"ref": "v0.9.0", "sha": v9}
    assert result["to"]["ref"] == "v0.10.0" and result["to"]["sha"] == v10
    assert f'@{v10}",  # v0.10.0' in after and v9 not in after
    assert '"pandas==2.3.3",' in after and '"pandas==2.3.2"' not in after
    # nothing outside the pin, the tag record and the managed block moved
    strip = re.compile(rf"{v9}|{v10}|v0\.9\.0|v0\.10\.0|2\.3\.[23]")
    assert strip.sub("X", before) == strip.sub("X", after)


def test_repin_refuses_a_ref_that_predates_line_support(tmp_path, fake_qb, datasets_root):
    """Same gate as new_line: a repin must not move a line onto an engine that cannot run it."""
    _scaffold(tmp_path, fake_qb, quantbox_ref="v0.10.0")
    target = tmp_path / "demo"
    before = (target / "pyproject.toml").read_text()
    # an older tag in the same repo that has no line support
    _git(fake_qb, "checkout", "-q", "--orphan", "ancient")
    _git(fake_qb, "rm", "-rq", "--cached", ".")
    (fake_qb / "README.md").write_text("old\n")
    _git(fake_qb, "add", "README.md")
    _git(fake_qb, "commit", "-q", "-m", "ancient")
    _git(fake_qb, "tag", "v0.7.0")
    with pytest.raises(line.LineError, match=r"predates `quantbox new line`.*--ref"):
        line.repin(target, ref="v0.7.0", lock=False)
    assert (target / "pyproject.toml").read_text() == before
    # the default (latest tag) is gated too, not just an explicit --ref
    _git(fake_qb, "tag", "v0.11.0")
    with pytest.raises(line.LineError, match="predates `quantbox new line`"):
        line.repin(target, lock=False)
    assert (target / "pyproject.toml").read_text() == before


def test_repin_refuses_a_pyproject_it_did_not_write(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["quantbox>=0.7"]\n')
    with pytest.raises(line.LineError, match="exactly one"):
        line.repin(tmp_path, ref="v1.0.0", lock=False)


# ── acceptance: new line -> uv sync -> one run, in a clean temp dir ─────────


def _snapshot_worktree(dest: Path, tag: str) -> Path:
    """A git repo whose `tag` is THIS working tree as it is now (uncommitted edits included)."""
    dest.mkdir(parents=True)
    _git(dest, "init", "-q", "-b", "main")
    files = _git(REPO, "ls-files", "-co", "--exclude-standard", "-z").split("\0")
    for rel in filter(None, files):
        src = REPO / rel
        if src.is_file() and not src.is_symlink():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / rel)
    _git(dest, "add", "-A")
    _git(dest, "commit", "-q", "-m", "snapshot")
    _git(dest, "tag", tag)
    return dest


def _clean_env(**extra: str) -> dict[str, str]:
    """No GIT_* (under the pre-push hook they name THIS repo), no borrowed venv, no prompts."""
    drop = ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k not in drop}
    return {**env, "GIT_TERMINAL_PROMPT": "0", **extra}


def _uv(line_dir: Path, *args: str, env: dict | None = None, timeout: int = 1800) -> subprocess.CompletedProcess:
    clean = _clean_env(**(env or {}))
    return subprocess.run(
        ["uv", *args], cwd=line_dir, env=clean, capture_output=True, text=True, timeout=timeout, check=False
    )


@pytest.mark.slow
def test_acceptance_new_line_then_uv_sync_then_one_run(tmp_path, datasets_root):
    probe = subprocess.run(
        ["git", "ls-remote", line.DATASETS_URL, "HEAD"],
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip(f"cannot read {line.DATASETS_URL} (private) from here: {probe.stderr.strip()[:200]}")

    engine = _snapshot_worktree(tmp_path / "quantbox", "v99.0.0")
    target = tmp_path / "work" / "demo"
    target.parent.mkdir()
    cli = Path(sys.executable).parent / "quantbox"
    new = subprocess.run(
        [str(cli), "new", "line", "demo", "--quantbox-url", f"file://{engine}", "--dataset", "demo-daily", "--json"],
        cwd=target.parent,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert new.returncode == 0, new.stderr
    scaffold = json.loads(new.stdout)
    assert scaffold["quantbox"]["ref"] == "v99.0.0" and scaffold["dataset"]["sha256"]
    assert (target / "uv.lock").is_file()

    sync = _uv(target, "sync", "--locked")
    assert sync.returncode == 0, sync.stderr
    # every package the line installs is pinned exactly (bar the two git pins)
    pyproject = (target / "pyproject.toml").read_text()
    lock_pins = line.lock_versions((target / "uv.lock").read_text(), exclude={"quantbox", "quantbox-datasets"})
    missing = [n for n, v in lock_pins.items() if f'"{n}=={v}",' not in pyproject]
    assert not missing, f"resolved but not pinned exactly: {missing}"

    run = _uv(target, "run", "--locked", "quantbox", "run", "-c", "config.yaml", "--json")
    assert run.returncode == 0, run.stderr
    manifest = json.loads(run.stdout)
    assert manifest["schema"] == "quantbox/run@1" and manifest["metrics"]
    assert manifest["dataset"]["sha256"] == scaffold["dataset"]["sha256"]

    arms = _uv(target, "run", "--locked", "quantbox", "arms", "-c", "arms.yaml", "--json")
    assert arms.returncode == 0, arms.stderr
    assert {a["status"] for a in json.loads(arms.stdout)["arms"]} == {"ok"}

    capture = _uv(target, "run", "--locked", "pytest", "-m", "reproduction", "-q", env={"QUANTBOX_CAPTURE_GOLDEN": "1"})
    assert capture.returncode == 0, capture.stdout + capture.stderr
    assert (target / "expected_metrics.json").is_file()
    check = _uv(target, "run", "--locked", "pytest", "-m", "reproduction", "-q")
    assert check.returncode == 0 and "1 passed" in check.stdout, check.stdout + check.stderr

    # repin is one command plus `uv sync`; the reproduction still holds at an identical engine
    _git(engine, "commit", "-q", "--allow-empty", "-m", "next")
    _git(engine, "tag", "v99.0.1")
    repin = subprocess.run(
        [str(cli), "line", "repin", str(target), "--json"],
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert repin.returncode == 0, repin.stderr
    assert json.loads(repin.stdout)["to"]["ref"] == "v99.0.1"
    assert _uv(target, "sync", "--locked").returncode == 0
    again = _uv(target, "run", "--locked", "pytest", "-m", "reproduction", "-q")
    assert again.returncode == 0 and "1 passed" in again.stdout, again.stdout + again.stderr
