"""Tests for quantbox-iface.py — above all that "could not check" never exits like "clean".

No real engine: a fake `uv` on PATH plays an old engine (no run --json, no
plugins schema) or a new one. Run with `python3 -m pytest test_quantbox_iface.py`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent

FAKE_UV = r"""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
mode = os.environ["FAKE_UV_MODE"]
args = sys.argv[1:]
if args[:2] == ["tool", "run"]:              # quantbox >= v0.8.0, run as a tool
    qb = args[args.index("quantbox", 4) + 1:]
    if qb[:2] == ["new", "line"]:
        d = Path(qb[qb.index("--dir") + 1]); d.mkdir(parents=True, exist_ok=True)
        (d / "pyproject.toml").write_text("# >>> quantbox line pins: managed\n# <<< quantbox line pins\n")
        (d / "uv.lock").write_text("lock\n"); (d / "config.yaml").write_text("execution: {lag_bars: 1}\n")
        (d / "README.md").write_text("# x\n\n## Pre-registration\n\n```yaml\nprereg:\n  id: <hypothesis-id>\n```\n\n## Findings\n")
        sys.exit(0)
    if qb[:2] == ["line", "repin"]:
        Path(qb[2], "pyproject.toml").write_text("# >>> quantbox line pins: repinned\n"); sys.exit(0)
    sys.exit(98)
if "pytest" in args:                        # the line's reproduction test
    if os.environ.get("QUANTBOX_CAPTURE_GOLDEN") == "1":
        Path("expected_metrics.json").write_text("{}\n"); sys.exit(0)
    sys.exit(1 if os.environ.get("FAKE_REPRO") == "red" else 0)
if "python" in args and "_load_local_source_class" in args[args.index("-c") + 1]:  # the engine's loader
    load = os.environ.get("FAKE_LOAD", "ok")
    if load == "noloader":
        sys.stderr.write("ImportError: cannot import name '_load_local_source_class'\n"); sys.exit(1)
    out = {}
    for src in json.loads(args[-1]):
        f = Path(src.rsplit(":", 1)[0])      # relative to the cwd, as the runner resolves it
        out[src] = (None if f.is_file() else f"FileNotFoundError: {f}") if load == "ok" else \
            "AttributeError: 'NoneType' object has no attribute '__dict__'"
    print(json.dumps(out)); sys.exit(0)
if "python" in args:                       # run@1 validation in the line env
    sys.exit(0)
if args[-3:] == ["plugins", "schema", "--json"]:
    if mode == "old":
        sys.stderr.write("Invalid value: Unknown action: schema. Use list, info, or doctor.\n"); sys.exit(2)
    if mode == "broken":
        sys.stderr.write("ImportError: boom\n"); sys.exit(1)
    print(json.dumps({"plugins": [
        {"id": "overlay.reversal_derisk.v1", "group": "overlay", "status": "experimental",
         "params_schema": {"type": "object", "properties": {"hold_bars": {}, "multiplier": {}}}, "params": []},
        {"id": "strategy.carver_trend.v2", "group": "strategy", "status": "production",
         "params_schema": None, "params": None}]}))
    sys.exit(0)
if "--help" in args:
    sys.exit(2 if mode == "old" else 0)
if "run" in args:
    if mode == "crash":
        sys.stderr.write("Traceback...\nKeyError: 'prices'\n"); sys.exit(1)
    if mode == "stdout-text":
        print("METRICS: {'sharpe': 9.9}"); sys.exit(0)
    print(json.dumps({"schema": "quantbox/run@1", "run_id": "r1", "metrics": {"sharpe": 0.5},
                      "files": {"returns": "returns.parquet", "traded_weights": "tw.parquet", "metrics": "m.json"}}))
    sys.exit(0)
sys.exit(99)
"""


@pytest.fixture
def mod(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("quantbox_iface", HERE / "quantbox-iface.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    uv = bindir / "uv"
    uv.write_text(FAKE_UV)
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(m, "ENV_CACHE", tmp_path / "envs")
    monkeypatch.setattr(m, "SCRATCH", tmp_path / "scratch")
    return m


def lab_with_line(tmp_path: Path, readme: str, extra: dict[str, str] | None = None) -> Path:
    src = tmp_path / "lab"
    line = src / "research" / "l1"
    line.mkdir(parents=True)
    (line / "README.md").write_text(readme)
    for rel, text in (extra or {}).items():
        (line / rel).parent.mkdir(parents=True, exist_ok=True)
        (line / rel).write_text(text)
    (line / "pyproject.toml").write_text(
        'dependencies = ["quantbox[full] @ git+https://github.com/tomlupo/quantbox.git@abc1234"]\n'
    )
    (line / "uv.lock").write_text("lock\n")
    for cmd in (
        ["init", "-q", "-b", "main"],
        ["add", "-A"],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"],
    ):
        subprocess.run(["git", "-C", str(src), *cmd], check=True)
    return src


PREREG = "# l1\n\n```yaml\nprereg:\n  hypothesis: H1\n```\n"


def call(mod, capsys, argv):
    code = mod.main(argv)
    return code, json.loads(capsys.readouterr().out)


def cap(lab, *needs, kind=None):
    argv = ["capability", "--lab", str(lab), "--line", "l1", "--ref", "main", "--no-fetch"]
    if kind:
        argv += ["--kind", kind]
    for n in needs:
        argv += ["--need", n]
    return argv


def _clone_with_dev_only_line(tmp_path: Path, with_dev: bool) -> Path:
    """A clone whose origin has the line on `dev` only (quantbox-lab, TOM-2400) or no `dev`."""
    src = lab_with_line(tmp_path, PREREG)
    g = ["-c", "user.name=t", "-c", "user.email=t@t"]
    if with_dev:
        subprocess.run(["git", "-C", str(src), "checkout", "-qb", "dev"], check=True)
        subprocess.run(["git", "-C", str(src), "mv", "research/l1", "research/l2"], check=True)
        subprocess.run(["git", "-C", str(src), *g, "commit", "-qm", "dev"], check=True)
        subprocess.run(["git", "-C", str(src), "checkout", "-q", "main"], check=True)
        subprocess.run(["git", "-C", str(src), "rm", "-rq", "research/l1"], check=True)
        subprocess.run(["git", "-C", str(src), *g, "commit", "-qm", "main"], check=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "--bare", str(src), str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "fetch", "-q", "origin", "+refs/heads/*:refs/remotes/origin/*"], check=True)
    return clone


@pytest.mark.parametrize("with_dev, line, ref", [(True, "l2", "origin/dev"), (False, "l1", "origin/main")])
def test_default_ref_is_the_labs_working_branch(mod, capsys, tmp_path, monkeypatch, with_dev, line, ref):
    # quantbox-lab ADR-0001: research lives on `dev`; a lab with no `dev` (robo-lab) works on `main`.
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    lab = _clone_with_dev_only_line(tmp_path, with_dev)
    code, out = call(mod, capsys, ["capability", "--lab", str(lab), "--line", line, "--no-fetch"])
    assert out["ref"] == ref
    assert out["result"] != "needs_line" or "does not exist" not in out["missing"][0]


def test_new_line_is_needs_line(mod, capsys, tmp_path):
    code, out = call(mod, capsys, ["capability", "--lab", str(tmp_path), "--line", "new: overlays"])
    assert (code, out["result"]) == (1, "needs_line")


def test_old_engine_is_needs_line_with_repin(mod, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_UV_MODE", "old")
    code, out = call(mod, capsys, cap(lab_with_line(tmp_path, PREREG)))
    assert (code, out["result"]) == (1, "needs_line")
    assert "plugins schema" in out["missing"][0] and out["engine_pin"] == "abc1234"


def test_schema_failure_is_blind_not_pass(mod, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_UV_MODE", "broken")
    code, out = call(mod, capsys, cap(lab_with_line(tmp_path, PREREG)))
    assert (code, out["result"]) == (3, "blind")


def test_missing_prereg_is_needs_line(mod, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    code, out = call(mod, capsys, cap(lab_with_line(tmp_path, "# l1\nno block\n")))
    assert (code, out["result"]) == (1, "needs_line")
    assert any("prereg" in m for m in out["missing"])


def test_needs_checked_against_catalog(mod, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    lab = lab_with_line(tmp_path, PREREG)
    code, out = call(
        mod, capsys, cap(lab, "overlay:overlay.reversal_derisk.v1", "param:overlay.reversal_derisk.v1.hold_bars")
    )
    assert (code, out["result"]) == (0, "pass")
    code, out = call(
        mod,
        capsys,
        cap(
            lab,
            "param:overlay.reversal_derisk.v1.bogus",
            "overlay:strategy.carver_trend.v2",
            "param:strategy.carver_trend.v2.x",
            "plugin:nope.v1",
        ),
    )
    assert (code, out["result"]) == (1, "needs_capability")
    assert len(out["missing"]) == 4


# -- the replication ladder (TOM-2233, Tom's D2) ---------------------------------------

CALENDAR = "plugin:strategy.halloween_switch.v1"
LINE_PLUGIN = '''from quantbox.contracts import PluginMeta


class HalloweenSwitch:
    meta = PluginMeta(
        name="strategy.halloween_switch.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.9",
        params_schema={"type": "object", "properties": {"months_in": {"description": "Nov-Apr"}}},
    )
'''


@pytest.mark.parametrize(("kind", "action"), [("replication", "write_plugin"), (None, "re_grade"), ("hypothesis", "re_grade")])
def test_a_gap_climbs_the_ladder_only_for_a_replication(mod, capsys, tmp_path, monkeypatch, kind, action):
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    code, out = call(mod, capsys, cap(lab_with_line(tmp_path, PREREG), CALENDAR, kind=kind))
    assert (code, out["result"], out["ladder"]["action"]) == (1, "needs_capability", action)
    if action == "write_plugin":
        assert out["ladder"]["code"] == [CALENDAR] and out["ladder"]["data"] == []


def test_the_prereg_kind_decides_when_no_kind_is_passed(mod, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    readme = "```yaml\nprereg:\n  id: H61\n  kind: replication\n```\n"
    code, out = call(mod, capsys, cap(lab_with_line(tmp_path, readme), CALENDAR))
    assert (out["kind"], out["ladder"]["action"]) == ("replication", "write_plugin")


def test_a_line_source_plugin_is_expressible(mod, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    lab = lab_with_line(tmp_path, PREREG, {"halloween.py": LINE_PLUGIN, "tests/test_x.py": LINE_PLUGIN.replace("halloween", "x")})
    code, out = call(mod, capsys, cap(lab, CALENDAR, "param:strategy.halloween_switch.v1.months_in"))
    assert (code, out["result"]) == (0, "pass")
    assert out["line_plugins"] == [
        {"id": "strategy.halloween_switch.v1", "group": "strategy", "source": "halloween.py:HalloweenSwitch"}
    ]
    code, out = call(mod, capsys, cap(lab, "overlay:strategy.halloween_switch.v1", "param:strategy.halloween_switch.v1.nope"))
    assert (code, out["result"], len(out["missing"])) == (1, "needs_capability", 2)


def test_a_line_plugin_the_engine_cannot_load_is_not_expressible(mod, capsys, tmp_path, monkeypatch):
    # Replay of the 2026-10-28 H60 run (TOM-2514): the `ast` read found the plugin, quantbox's loader
    # refused it, and the pre-flight was the first to say so. The engine's loader now answers.
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    monkeypatch.setenv("FAKE_LOAD", "fail")
    lab = lab_with_line(tmp_path, PREREG, {"halloween.py": LINE_PLUGIN})
    code, out = call(mod, capsys, cap(lab, CALENDAR, kind="replication"))
    assert (code, out["result"]) == (1, "needs_capability")
    assert "does not load in the pinned engine" in out["missing"][0] and "__dict__" in out["missing"][0]
    assert out["ladder"] == {"rung": "b", "action": "write_plugin", "code": [CALENDAR], "data": []}


def test_a_loader_that_cannot_be_called_is_blind_not_pass(mod, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    monkeypatch.setenv("FAKE_LOAD", "noloader")
    lab = lab_with_line(tmp_path, PREREG, {"halloween.py": LINE_PLUGIN})
    code, out = call(mod, capsys, cap(lab, CALENDAR))
    assert (code, out["result"]) == (3, "blind") and "line plugins not checked" in out["reason"]


def test_only_a_needed_line_plugin_is_loaded(mod, capsys, tmp_path, monkeypatch):
    # An old dated plugin no need names never runs: a line with legacy files keeps its pass (H58's path).
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    monkeypatch.setenv("FAKE_LOAD", "fail")
    lab = lab_with_line(tmp_path, PREREG, {"configs/dated/old.py": LINE_PLUGIN})
    code, out = call(mod, capsys, cap(lab, "overlay:overlay.reversal_derisk.v1"))
    assert (code, out["result"]) == (0, "pass")


def test_a_line_plugin_may_not_reuse_an_engine_id(mod, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    clash = LINE_PLUGIN.replace("strategy.halloween_switch.v1", "strategy.carver_trend.v2")
    code, out = call(mod, capsys, cap(lab_with_line(tmp_path, PREREG, {"c.py": clash})))
    assert (code, out["result"]) == (1, "needs_capability") and "bump its suffix" in out["missing"][0]


def test_a_dataset_is_readable_only_when_frozen_and_pinned(mod, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    monkeypatch.setattr(mod, "DATASETS_ROOT", tmp_path / "no-root")
    frozen = {"datasets/ff-factors/manifest.yaml": "name: ff-factors\n"}
    code, out = call(mod, capsys, cap(lab_with_line(tmp_path / "a", PREREG), "dataset:ff-factors", kind="replication"))
    assert (code, out["ladder"]["data"]) == (1, ["dataset:ff-factors"])
    code, out = call(mod, capsys, cap(lab_with_line(tmp_path / "b", PREREG, frozen), "dataset:ff-factors"))
    assert code == 1 and "not pinned" in out["missing"][0]
    pinned = {**frozen, "datasets.lock": "ff-factors: abc123\n"}
    code, out = call(mod, capsys, cap(lab_with_line(tmp_path / "c", PREREG, pinned), "dataset:ff-factors"))
    assert (code, out["result"]) == (0, "pass")


SBBI_SERVE = "name: sbbi\nserve:\n  table: monthly\n  kind: returns\n  columns: [LargeCap_TR, LargeCap_IR]\n"


def test_a_dataset_column_passes_only_when_its_manifest_serves_it(mod, capsys, tmp_path, monkeypatch):
    # TOM-2410: `dataset:sbbi` passed on the manifest alone while LargeCap_IR was not served (H58 dry run).
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    root = tmp_path / "root"
    (root / "sbbi").mkdir(parents=True)
    (root / "sbbi" / "manifest.yaml").write_text(SBBI_SERVE)
    (root / "etf-daily").mkdir()
    (root / "etf-daily" / "manifest.yaml").write_text("name: etf-daily\n")
    monkeypatch.setattr(mod, "DATASETS_ROOT", root)
    lab = lab_with_line(tmp_path / "a", PREREG)
    code, out = call(mod, capsys, cap(lab, "dataset:sbbi", "dataset:sbbi:LargeCap_IR"))
    assert (code, out["result"]) == (0, "pass")
    code, out = call(mod, capsys, cap(lab, "dataset:sbbi:LargeCap_CAR", "dataset:etf-daily:SPY", kind="replication"))
    assert (code, out["result"]) == (1, "needs_capability")
    assert "LargeCap_CAR" in out["missing"][0] and "not served" in out["missing"][0]
    assert "declares no serve.columns" in out["missing"][1]
    assert out["ladder"]["data"] == ["dataset:etf-daily:SPY", "dataset:sbbi:LargeCap_CAR"]
    # A dataset frozen in the line is judged by the line's own manifest, at the ref checked.
    frozen = {"datasets/ff/manifest.yaml": SBBI_SERVE.replace("sbbi", "ff"), "datasets.lock": "ff: abc123\n"}
    lab = lab_with_line(tmp_path / "b", PREREG, frozen)
    code, out = call(mod, capsys, cap(lab, "dataset:ff:LargeCap_IR"))
    assert (code, out["result"]) == (0, "pass")
    code, out = call(mod, capsys, cap(lab, "dataset:ff:Nope"))
    assert code == 1 and "not served" in out["missing"][0]


def test_a_dataset_need_names_one_column_at_most(mod):
    assert mod.parse_need("dataset:sbbi:LargeCap_IR") == ("dataset", "sbbi", "LargeCap_IR")
    assert mod.parse_need("dataset:sbbi") == ("dataset", "sbbi", None)
    for bad in ("dataset:sbbi:", "dataset::LargeCap_IR", "dataset:sbbi:a:b"):
        with pytest.raises(Exception, match="dataset"):
            mod.parse_need(bad)


def line_dir(tmp_path: Path) -> Path:
    d = tmp_path / "line"
    (d / "configs").mkdir(parents=True)
    (d / "uv.lock").write_text("lock\n")
    (d / "configs" / "a.yaml").write_text("artifacts: {root: ./artifacts}\n")
    return d


@pytest.mark.parametrize(
    ("mode", "code", "result"),
    [("old", 3, "blind"), ("stdout-text", 3, "blind"), ("crash", 1, "failed"), ("new", 0, "ok")],
)
def test_run_reads_only_run_at_1(mod, capsys, tmp_path, monkeypatch, mode, code, result):
    monkeypatch.setenv("FAKE_UV_MODE", mode)
    d = line_dir(tmp_path)
    got, out = call(mod, capsys, ["run", "--line-dir", str(d), "--config", "configs/a.yaml"])
    assert (got, out["result"]) == (code, result)
    if mode == "new":
        assert out["files"]["returns"] == str(d / "artifacts" / "r1" / "returns.parquet")
    if mode == "stdout-text":
        assert "manifest" not in out  # a METRICS: line is never read as a result


# -- dev-loop Step 1b: `quantbox preflight` of the pinned engine (TOM-2457, quantbox ADR-0013/0014) ------

# The gate as quantbox v0.21.0 answers it: `--help` is its argparse usage, a run prints rows and ONE
# `PREFLIGHT:` line and exits 0/1/2/3. FAKE_PF=absent plays an engine older than v0.21.0 (Typer's
# "No such command"); FAKE_PF=liar exits 0 under a REFUSE summary. Every gate call's argv is logged.
FAKE_UV_PF = FAKE_UV.replace(
    'if args[-3:] == ["plugins", "schema", "--json"]:\n',
    'if "preflight" in args:\n'
    '    pf = os.environ["FAKE_PF"]\n'
    '    if "--help" in args:\n'
    '        if pf == "absent":\n'
    '            sys.stderr.write("Error: No such command \'preflight\'.\\n"); sys.exit(2)\n'
    '        print("usage: quantbox preflight [-h] --lab LAB --line LINE"); sys.exit(0)\n'
    '    with open(os.environ["FAKE_ARGV_LOG"], "a") as f: f.write(json.dumps(args) + "\\n")\n'
    '    rows = {"pass": (0, ["PASS   prereg   block id=H1", "PREFLIGHT: PASS (1 checks)"]),\n'
    '            "refuse": (1, ["REFUSE binding   no binding block", "PREFLIGHT: REFUSE (1) - do not run the backtest"]),\n'
    '            "blind": (2, ["BLIND  data   cannot read", "PREFLIGHT: COULD NOT CHECK (1) - not a pass"]),\n'
    '            "flag": (3, ["FLAG   replica-parity   no confirmed parity - NOT PROMOTABLE",\n'
    '                         "finding limits: replica-parity - each FLAG row above says what the finding may not be",\n'
    '                         "PREFLIGHT: FLAG (1) - the run may go ahead"]),\n'
    '            "liar": (0, ["REFUSE binding   x", "PREFLIGHT: REFUSE (1)"])}\n'
    '    code, lines = rows[pf]\n'
    '    print("\\n".join(lines)); sys.exit(code)\n'
    'if args[-3:] == ["plugins", "schema", "--json"]:\n',
)
assert FAKE_UV_PF != FAKE_UV


def preflight(mod, capsys, tmp_path, monkeypatch, pf: str, *extra: str):
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    monkeypatch.setenv("FAKE_PF", pf)
    log = tmp_path / "argv.jsonl"
    monkeypatch.setenv("FAKE_ARGV_LOG", str(log))
    (tmp_path / "bin" / "uv").write_text(FAKE_UV_PF)
    d = line_dir(tmp_path)
    (d / "pyproject.toml").write_text(pinned("v0.21.0"))
    code, out = call(mod, capsys, ["preflight", "--lab", str(tmp_path), "--line-dir", str(d), *extra, "configs/a.yaml"])
    return code, out, d, (json.loads(log.read_text().splitlines()[-1]) if log.exists() else None)


@pytest.mark.parametrize(
    ("pf", "code", "result"), [("pass", 0, "pass"), ("refuse", 1, "refused"), ("blind", 2, "blind"), ("flag", 3, "flag")]
)
def test_preflight_passes_the_gates_four_codes_through(mod, capsys, tmp_path, monkeypatch, pf, code, result):
    got, out, d, argv = preflight(mod, capsys, tmp_path, monkeypatch, pf)
    assert (got, out["result"], out["rc"]) == (code, result, code)
    # The line's own pin runs the gate, on absolute paths, from the line dir.
    assert argv[argv.index("--project") + 1] == str(d)
    assert argv[argv.index("--line") + 1] == str(d) and argv[-1] == str(d / "configs" / "a.yaml")
    if pf == "flag":
        # A FLAG run goes ahead, and the finding's limits are named, never dropped.
        assert out["flags"] == ["replica-parity"] and out["limits"].startswith("finding limits: replica-parity")
    else:
        assert out["flags"] == [] and out["limits"] is None


def test_preflight_on_an_engine_without_the_verb_is_blind_and_never_the_plugin(mod, capsys, tmp_path, monkeypatch):
    # Fail closed: a line pinned below v0.21.0 has no `quantbox preflight`. That is BLIND, said in words,
    # and the gate is never run from the qute-research plugin instead.
    got, out, d, argv = preflight(mod, capsys, tmp_path, monkeypatch, "absent")
    assert (got, out["result"]) == (2, "blind")
    assert "has no `quantbox preflight`" in out["reason"] and "v0.21.0" in out["reason"]
    assert "never fall back" in out["reason"] and argv is None  # the gate itself was never called


def test_preflight_whose_summary_contradicts_its_code_is_blind(mod, capsys, tmp_path, monkeypatch):
    got, out, _, _ = preflight(mod, capsys, tmp_path, monkeypatch, "liar")
    assert (got, out["result"]) == (2, "blind") and "not the gate's answer" in out["reason"]


def test_preflight_lab_env_runs_the_lab_roots_pin(mod, capsys, tmp_path, monkeypatch):
    (tmp_path / "uv.lock").write_text("lock\n")
    got, out, _, argv = preflight(mod, capsys, tmp_path, monkeypatch, "pass", "--env", "lab")
    assert (got, out["env_dir"]) == (0, str(tmp_path))
    assert argv[argv.index("--project") + 1] == str(tmp_path)


def test_preflight_without_a_lock_is_blind(mod, capsys, tmp_path, monkeypatch):
    # `--env lab` on a lab dir with no uv.lock: no pinned quantbox to ask, so nothing was checked.
    got, out, _, argv = preflight(mod, capsys, tmp_path, monkeypatch, "pass", "--env", "lab")
    assert (got, out["result"]) == (2, "blind") and "no uv.lock" in out["reason"] and argv is None


def test_placeholder_prereg_is_needs_line(mod, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    code, out = call(mod, capsys, cap(lab_with_line(tmp_path, "```yaml\nprereg:\n  id: <hypothesis-id>\n```\n")))
    assert (code, out["result"]) == (1, "needs_line")
    assert any("placeholders" in m for m in out["missing"])


def git_(d: Path, *a: str) -> None:
    subprocess.run(["git", "-C", str(d), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True)


def worktree(tmp_path: Path, files: dict[str, str], branch: str = "research/devloop-x") -> Path:
    lab = tmp_path / "lab"
    for rel, text in {"research/.gitkeep": "", **files}.items():
        (lab / rel).parent.mkdir(parents=True, exist_ok=True)
        (lab / rel).write_text(text)
    git_(lab, "init", "-q", "-b", "main")
    git_(lab, "add", "-A")
    git_(lab, "commit", "-qm", "x")
    wt = tmp_path / "wt"
    git_(lab, "worktree", "add", "-q", str(wt), "-b", branch)
    return wt


def line(mod, capsys, wt, name, *extra):
    return call(mod, capsys, ["line", "--worktree", str(wt), "--line", name, "--ref", "v0.8.0", *extra])


def test_line_opens_a_new_line_with_placeholders_left(mod, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    code, out = line(mod, capsys, worktree(tmp_path, {}), "new: hypertrend-overlays")
    assert (code, out["result"], out["action"]) == (0, "ready", "new")
    assert out["prereg"] == "placeholders" and out["line"] == "hypertrend-overlays"


def test_line_pins_a_readme_only_line_without_overwriting(mod, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    wt = worktree(tmp_path, {"research/ct/README.md": "# ct\nprose\n", "research/ct/config.yaml": "mine: 1\n"})
    code, out = line(mod, capsys, wt, "ct")
    assert (code, out["action"]) == (0, "pin")
    assert (wt / "research/ct/config.yaml").read_text() == "mine: 1\n"  # never overwritten
    readme = (wt / "research/ct/README.md").read_text()
    assert readme.startswith("# ct\nprose\n") and "prereg:" in readme and "## Findings" not in readme
    assert (wt / "research/ct/pyproject.toml").exists() and not (wt / ".line-scaffold").exists()


def test_line_refuses_legacy_pin_lab_root_and_protected_branch(mod, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    wt = worktree(tmp_path, {"research/old/pyproject.toml": "[project]\n"})
    code, out = line(mod, capsys, wt, "old")
    assert (code, out["result"]) == (1, "refused") and "legacy" in out["reason"]
    code, out = line(mod, capsys, tmp_path / "lab", "old")  # the lab root, not a worktree
    assert (code, out["result"]) == (1, "refused")
    wt2 = tmp_path / "wt2"
    git_(tmp_path / "lab", "worktree", "add", "-q", str(wt2), "-b", "dev")
    code, out = line(mod, capsys, wt2, "new: x")
    assert (code, out["result"]) == (1, "refused") and "'dev'" in out["reason"]
    assert not (wt2 / "research/x").exists()


@pytest.mark.parametrize(("mode", "repro", "code", "action"), [("new", "green", 0, "none"), ("old", "green", 0, "repin"), ("old", "red", 1, "repin")])
def test_line_repins_only_an_engine_without_the_interface(mod, capsys, tmp_path, monkeypatch, mode, repro, code, action):
    monkeypatch.setenv("FAKE_UV_MODE", mode)
    monkeypatch.setenv("FAKE_REPRO", repro)
    wt = worktree(tmp_path, {"research/mc/pyproject.toml": "# >>> quantbox line pins: x\n", "research/mc/README.md": PREREG})
    got, out = line(mod, capsys, wt, "mc")
    assert (got, out["action"]) == (code, action)
    if action == "repin":
        assert out["golden_captured_on_old_pin"] is True and out["reproduction"] == repro


# -- the H58 dry run (TOM-1368, 2026-10-25): the engine floor and the repin's own runs -----------

# Every reproduction run writes a run-stamped dir under the line's artifacts root, as `quantbox run` does.
FAKE_UV_WITH_RUNS = FAKE_UV.replace(
    'if "pytest" in args:                        # the line\'s reproduction test\n',
    'if "pytest" in args:                        # the line\'s reproduction test\n'
    '    import time\n'
    '    Path("artifacts/base", f"2026-10-14__backtest_pipeline_v1__x__{time.time_ns()}").mkdir(parents=True)\n',
)
assert FAKE_UV_WITH_RUNS != FAKE_UV


def pinned(ref: str) -> str:
    """A Model C pyproject as `quantbox new line` writes it: the tag's commit, the tag in two places."""
    return (
        f'dependencies = ["quantbox[full] @ git+https://github.com/tomlupo/quantbox.git@{"a" * 40}",  # {ref}]\n'
        f'[tool.quantbox-line]\nquantbox-ref = "{ref}"\n# >>> quantbox line pins: x\n# <<< quantbox line pins\n'
    )


@pytest.mark.parametrize(("ref", "action"), [("v0.17.0", "repin"), ("v0.18.0", "none"), ("v0.19.0", "none")])
def test_line_repins_a_line_below_the_engine_floor_without_the_flag(mod, capsys, tmp_path, monkeypatch, ref, action):
    # dev-loop.md rung b: "a line still pinned below v0.18.0 is repinned first". The engine has the
    # interface (mode new), so only the floor can make the script repin, and no --repin is passed.
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    monkeypatch.setenv("FAKE_REPRO", "green")
    monkeypatch.setattr(mod, "REPIN_RUNS", tmp_path / "repin-runs", raising=False)
    wt = worktree(tmp_path, {"research/mc/pyproject.toml": pinned(ref), "research/mc/README.md": PREREG})
    got, out = call(mod, capsys, ["line", "--worktree", str(wt), "--line", "mc", "--ref", "v0.19.0"])
    assert (got, out["action"]) == (0, action)
    if action == "repin":
        assert "v0.17.0" in out["repinned_because"] and "v0.18.0" in out["repinned_because"]
    else:
        assert "repinned_because" not in out


# -- a replication line opens with the paper-anchored prereg block (TOM-2426, H58 dry run 2) ----------

# `new line --help` answers as the pinned engine's CLI would: with `--kind` (quantbox#312) or without
# (every tag up to v0.19.0). With `--kind` it prints Typer's rich styling, which splits the option into
# `-` and `-kind` (seen on the real #312 help, 2026-10-26). Each `new line` call records its argv.
FAKE_UV_KIND = FAKE_UV.replace(
    '    if qb[:2] == ["new", "line"]:\n',
    '    if qb[:2] == ["new", "line"] and "--help" in qb:\n'
    '        print("--dir --quantbox-ref --dataset --question" + (" \\x1b[1;36m-\\x1b[0m\\x1b[1;36m-kind\\x1b[0m" if os.environ.get("FAKE_KIND") == "yes" else ""))\n'
    '        sys.exit(0)\n'
    '    if qb[:2] == ["new", "line"]:\n'
    '        with open(os.environ["FAKE_ARGV_LOG"], "a") as f: f.write(json.dumps(qb) + "\\n")\n',
)
assert FAKE_UV_KIND != FAKE_UV


@pytest.mark.parametrize(("has_kind", "warned"), [("yes", False), ("no", True)])
def test_a_replication_line_is_scaffolded_with_its_kind_and_dataset(
    mod, capsys, tmp_path, monkeypatch, has_kind, warned
):
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    monkeypatch.setenv("FAKE_KIND", has_kind)
    log = tmp_path / "argv.jsonl"
    monkeypatch.setenv("FAKE_ARGV_LOG", str(log))
    (tmp_path / "bin" / "uv").write_text(FAKE_UV_KIND)
    wt = worktree(tmp_path, {})
    code, out = line(mod, capsys, wt, "new: dp-lewellen", "--kind", "replication", "--dataset", "sbbi")
    assert (code, out["result"], out["action"]) == (0, "ready", "new")
    argv = json.loads(log.read_text().splitlines()[-1])
    assert argv[argv.index("--dataset") + 1] == "sbbi"
    if warned:
        # The engine cannot write the block: today's scaffold, and the gap is said, never silent.
        assert "--kind" not in argv
        assert out["prereg_kind"] == "book" and "--kind" in out["warning"]
    else:
        assert argv[argv.index("--kind") + 1] == "replication"
        assert out["prereg_kind"] == "replication" and "warning" not in out


def test_a_hypothesis_line_passes_no_kind(mod, capsys, tmp_path, monkeypatch):
    # The default path is unchanged: no probe, no --kind, no --dataset.
    monkeypatch.setenv("FAKE_UV_MODE", "new")
    monkeypatch.setenv("FAKE_KIND", "yes")
    log = tmp_path / "argv.jsonl"
    monkeypatch.setenv("FAKE_ARGV_LOG", str(log))
    (tmp_path / "bin" / "uv").write_text(FAKE_UV_KIND)
    code, out = line(mod, capsys, worktree(tmp_path, {}), "new: x")
    argv = json.loads(log.read_text().splitlines()[-1])
    assert code == 0 and "--kind" not in argv and "--dataset" not in argv


def test_a_repins_own_runs_never_stay_in_the_line(mod, capsys, tmp_path, monkeypatch):
    # The dry run's golden landed in artifacts/base, and the pre-flight read it as a post-registration.
    monkeypatch.setenv("FAKE_UV_MODE", "old")
    monkeypatch.setenv("FAKE_REPRO", "green")
    (tmp_path / "bin" / "uv").write_text(FAKE_UV_WITH_RUNS)
    monkeypatch.setattr(mod, "REPIN_RUNS", tmp_path / "repin-runs", raising=False)
    earlier = "research/mc/artifacts/base/2026-10-01__backtest_pipeline_v1__y__20261001T000000Z/run_manifest.json"
    wt = worktree(
        tmp_path,
        {
            "research/mc/pyproject.toml": "# >>> quantbox line pins: x\n",
            "research/mc/README.md": PREREG,
            "research/mc/config.yaml": "artifacts:\n  root: ./artifacts/base\n",
            earlier: "{}\n",
        },
    )
    got, out = line(mod, capsys, wt, "mc")
    assert (got, out["action"], out["golden_captured_on_old_pin"]) == (0, "repin", True)
    left = sorted(p.name for p in (wt / "research/mc/artifacts/base").iterdir())
    assert left == ["2026-10-01__backtest_pipeline_v1__y__20261001T000000Z"]  # a run that was there stays
    moved = Path(out["repin_runs_moved_to"])
    assert moved.is_relative_to(tmp_path / "repin-runs") and len(out["repin_runs_moved"]) == 2
    assert sorted(p.name for p in moved.iterdir()) == sorted(out["repin_runs_moved"])
