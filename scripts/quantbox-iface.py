#!/usr/bin/env python3
"""Quark's ONE reader of the quantbox research interface (TOM-1368).

Four verbs, all run ON FORGE (the labs live there; from core: `ssh forge
python3 <this file> ...`). All talk to the engine the LINE PINS, never to
whatever quantbox happens to be on PATH: the line's own `pyproject.toml` +
`uv.lock` is the engine a dev-loop run would use.

  capability  scout's expressibility gate. Is the line there, does its README
              carry a fenced `prereg:` block, and does `quantbox plugins schema
              --json` (TOM-1350) of the PINNED engine, or one of the line's own
              `source:` plugins, confirm every plugin, overlay and param the
              hypothesis needs, and is every dataset it names readable by name?
              A line plugin a need resolves to must also LOAD through the pinned
              engine's own loader (the one `quantbox run` calls). For a replication,
              a gap is not a re-grade: `ladder` says what the run writes in the line
              (TOM-2233, Tom's D2).
  line        dev-loop Step 1c (Tom, 2026-10-01: the loop may open or repin a line
              itself). In the run's lab WORKTREE, make research/<line> a Model C line
              (`quantbox new line` / `quantbox line repin`, quantbox >= v0.8.0) and
              leave the `prereg:` placeholders for the run to fill. Never commits.
              It repins a line whose engine lacks the interface or is pinned below
              v0.18.0 (ENGINE_FLOOR) by itself and says why (`repinned_because`); the
              repin's own golden and reproduction runs are moved out of the line
              (`repin_runs_moved_to`), so the pre-flight never counts them as runs.
  run         dev-loop Step 2. `quantbox run --json` (quantbox/run@1, TOM-1348)
              in the line dir; the run@1 manifest is the ONLY source of metrics.
              Nothing is ever parsed from human stdout (`METRICS:` is gone).
  preflight   dev-loop Step 1b, the MECHANICAL half (TOM-2457, quantbox ADR-0013/0014):
              `quantbox preflight` of the engine the line pins (`--env line`, the
              default) or the lab root's (`--env lab`). An engine without the verb
              (older than v0.21.0) is BLIND and the verb says so; there is NO fallback
              to the qute-research plugin's `quant_preflight.py`. The judgement half,
              the `qute-research:prereg-auditor` agent, is not run here.

Exit codes are the contract, so "could not check" never looks like "clean".
`preflight` passes the GATE's own codes through, and they differ from the others at 2 and 3:

  0  pass       1  refused (do not run the backtest)
  2  BLIND: could not check, also an engine without `quantbox preflight` or output
     that is not the gate's
  3  FLAG: the run goes ahead, and the finding carries every FLAG row (`flags`,
     `limits`): it is never promotable, and a proxy test is never `confirmed`

The other verbs:

  0  pass / ok
  1  capability: the hypothesis is not runnable as stated (`result` says
     needs_line | needs_capability, `missing` says what, and on needs_capability
     `ladder.action` says write_plugin (a replication) or re_grade) ; line: refused (a
     legacy-pinned line, the lab root or a protected branch, quantbox refused,
     or the reproduction went red after a repin) ; run: the backtest failed or
     the rebalancer froze (`reason` carries the last error line)
  2  usage
  3  BLIND: the check could not be made (no lab on this box, no uv, the pinned
     engine predates the interface, output was not the declared document).
     Never a pass. The caller reports it as BLIND, in that word.

Output: one JSON document on stdout, always (also on 1 and 3).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

RUN_SCHEMA = "quantbox/run@1"
ENV_CACHE = Path(os.environ.get("QUARK_LINE_ENV_CACHE", Path.home() / ".cache" / "quark-line-envs"))
QUANTBOX_URL = "https://github.com/tomlupo/quantbox.git"
DATASETS_ROOT = Path("/home/tom/workspace/projects/quantbox-datasets/datasets")
PROTECTED = {"main", "master", "dev"}
PLACEHOLDER = re.compile(r"^<[^<>]+>$")
PINS_BEGIN = "# >>> quantbox line pins"
REPIN_HINT = "repin the line to a quantbox release that carries it (>= v0.8.0; `quantbox line repin`)"
# The oldest engine a line may stay on: datasets read by name (TOM-2232) ship in v0.18.0. A line pinned
# below it is repinned by `line` itself, no --repin needed (dev-loop.md rung b, Step 1c).
ENGINE_FLOOR = (0, 18, 0)
# A repin's own golden + reproduction runs land in the line's artifacts root, which the pre-flight walks as
# this hypothesis's run evidence: they predate the prereg and read as a post-registration. They are an
# engine check, not a measurement, so `line` moves them here, outside the lab (TOM-1368 dry run, gap c).
REPIN_RUNS = Path(os.environ.get("QUARK_REPIN_RUNS", Path.home() / "workspace" / "tmp" / "quark-repin-runs"))
ANSI = re.compile(r"\x1b\[[0-9;]*m")
FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.M | re.S)


def emit(code: int, payload: dict) -> int:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return code


def uv_bin() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "uv"
    return str(fallback) if fallback.exists() else None


def tail(text: str, n: int = 1) -> str:
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return " | ".join(lines[-n:]) if lines else ""


def git(lab: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(lab), *args], capture_output=True, text=True)


def _placeholders(node) -> bool:
    if isinstance(node, dict):
        return any(_placeholders(k) or _placeholders(v) for k, v in node.items())
    if isinstance(node, list):
        return any(_placeholders(v) for v in node)
    return isinstance(node, str) and bool(PLACEHOLDER.match(node.strip()))


def prereg_state(readme: str) -> str:
    """`ok`, `absent`, or `placeholders` (the `quantbox new line` template, still unfilled).

    A fenced YAML block whose top level carries `prereg:` is the plugin's preregistration.md shape.
    """
    state = "absent"
    for block in FENCE.findall(readme):
        try:
            doc = yaml.safe_load(block)
        except yaml.YAMLError:
            continue
        if isinstance(doc, dict) and "prereg" in doc:
            if not _placeholders(doc["prereg"]):
                return "ok"
            state = "placeholders"
    return state


def prereg_kind(readme: str) -> str | None:
    """The filled `prereg:` block's `kind` (`replication`, TOM-2231), or None."""
    for block in FENCE.findall(readme):
        try:
            doc = yaml.safe_load(block)
        except yaml.YAMLError:
            continue
        if isinstance(doc, dict) and isinstance(doc.get("prereg"), dict) and not _placeholders(doc["prereg"]):
            kind = doc["prereg"].get("kind")
            return kind if isinstance(kind, str) else None
    return None


def engine_pin(pyproject: str) -> str | None:
    m = re.search(r"quantbox(?:\[[^\]]*\])?\s*@\s*git\+[^\"']*?quantbox(?:\.git)?@([0-9a-f]{7,40}|v[\w.]+)", pyproject)
    return m.group(1) if m else None


def engine_ref(pyproject: str) -> tuple[int, int, int] | None:
    """The release a Model C line pins, as (major, minor, patch); None when it names no `vX.Y.Z`.

    `quantbox new line` pins the tag's COMMIT and names the tag in `[tool.quantbox-line] quantbox-ref`
    and in the dependency's comment; an older line may pin the tag itself.
    """
    for pattern in (
        r"^\s*quantbox-ref\s*=\s*[\"']v(\d+)\.(\d+)\.(\d+)[\"']",
        r"quantbox(?:\[[^\]]*\])?\s*@\s*git\+[^\n]*?#\s*v(\d+)\.(\d+)\.(\d+)",
        r"quantbox(?:\[[^\]]*\])?\s*@\s*git\+[^\"']*?quantbox(?:\.git)?@v(\d+)\.(\d+)\.(\d+)",
    ):
        m = re.search(pattern, pyproject, re.M)
        if m:
            return tuple(int(x) for x in m.groups())
    return None


def below_floor(pyproject: str) -> str | None:
    """Why the line must be repinned for the floor, or None. A commit pin that names no release
    cannot be shown to carry v0.18.0, so it counts as below; a file that pins no quantbox at all
    is left to the interface probe."""
    floor = "v" + ".".join(map(str, ENGINE_FLOOR))
    ref = engine_ref(pyproject)
    if ref is not None:
        return None if ref >= ENGINE_FLOOR else f"pinned v{'.'.join(map(str, ref))} is below the floor {floor}"
    pin = engine_pin(pyproject)
    if pin is not None:
        return f"pinned commit {pin} names no release, so it cannot be shown to be at or above {floor}"
    return None


def artifacts_root(line_dir: Path) -> Path:
    """Where the line's `config.yaml` (the reproduction's config) writes its runs."""
    try:
        cfg = yaml.safe_load((line_dir / "config.yaml").read_text()) or {}
        root = (cfg.get("artifacts") or {}).get("root") if isinstance(cfg, dict) else None
    except (OSError, yaml.YAMLError, AttributeError):
        root = None
    path = Path(root or "artifacts")
    return path if path.is_absolute() else (line_dir / path).resolve()


def line_env(pyproject: str, lock: str) -> Path:
    """A uv project holding ONLY the line's pyproject + lock, keyed by their hash, reused across runs."""
    key = hashlib.sha256((pyproject + "\0" + lock).encode()).hexdigest()[:16]
    env = ENV_CACHE / key
    env.mkdir(parents=True, exist_ok=True)
    (env / "pyproject.toml").write_text(pyproject)
    (env / "uv.lock").write_text(lock)
    return env


def parse_need(spec: str) -> tuple[str, str, str | None]:
    kind, _, rest = spec.partition(":")
    if kind not in ("plugin", "overlay", "param", "dataset") or not rest:
        raise argparse.ArgumentTypeError(
            f"--need {spec!r}: expected plugin:<id>, overlay:<id>, param:<id>.<name> or dataset:<name>[:<column>]"
        )
    if kind == "dataset":  # dataset:<name>, or dataset:<name>:<column> for one served column (TOM-2410)
        name, colon, column = rest.partition(":")
        if not name or (colon and (not column or ":" in column)):
            raise argparse.ArgumentTypeError(f"--need {spec!r}: dataset needs <name> or <name>:<column>")
        return kind, name, column or None
    if kind == "param":
        pid, dot, name = rest.rpartition(".")
        if not dot or not pid or not name:
            raise argparse.ArgumentTypeError(f"--need {spec!r}: param needs <plugin id>.<param name>")
        return kind, pid, name
    return kind, rest, None


def line_plugins(files: dict[str, str]) -> list[dict]:
    """The line's own `source:` plugins: each class whose body sets `meta = PluginMeta(name=..., kind=...)`.

    Read with `ast`, never imported: this finds what each file declares, for every file of the line.
    Whether a plugin a need resolves to actually loads is the engine's answer (`load_line_plugins`).
    `files` maps a path relative to the line dir to its text; `source` is what a config's `source:`
    names (the runner resolves it against its cwd, the line dir). A params_schema that is not a
    literal is None.
    """
    found = []
    for rel, text in sorted(files.items()):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            for stmt in cls.body:
                call = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [getattr(stmt, "target", None)]
                if not (
                    isinstance(call, ast.Call)
                    and any(isinstance(t, ast.Name) and t.id == "meta" for t in targets)
                    and (getattr(call.func, "id", None) or getattr(call.func, "attr", None)) == "PluginMeta"
                ):
                    continue
                kw = {k.arg: k.value for k in call.keywords if k.arg}

                def lit(node):
                    try:
                        return ast.literal_eval(node) if node is not None else None
                    except ValueError:
                        return None

                name, kind = lit(kw.get("name")), lit(kw.get("kind"))
                if isinstance(name, str) and isinstance(kind, str):
                    schema = lit(kw.get("params_schema"))
                    found.append(
                        {
                            "id": name,
                            "group": kind,
                            "params_schema": schema if isinstance(schema, dict) else None,
                            "source": f"{rel}:{cls.name}",
                        }
                    )
    return found


def check_needs(
    catalog: dict, needs: list[tuple[str, str, str | None]], own: list[dict] | None = None, datasets=None
) -> list[tuple[int, str]]:
    """(index of the need, why it is missing) for every need neither the engine nor the line covers.

    `own` are the line's `source:` plugins: they count as expressible (TOM-2233), but never under an
    id the engine already registers (a config naming it would mean two different things).
    `datasets(name, column)` answers whether a dataset name, and the one column a need names (or
    None), is readable by name (None = no dataset was checked).
    """
    engine = {p["id"]: p for p in catalog.get("plugins", [])}
    by_id = {**engine, **{p["id"]: {**p, "origin": "line"} for p in own or [] if p["id"] not in engine}}
    missing: list[tuple[int, str]] = []
    for p in own or []:
        if p["id"] in engine:
            missing.append((-1, f"line plugin {p['source']} reuses the engine's id {p['id']}: bump its suffix"))
    for i, (kind, pid, name) in enumerate(needs):
        if kind == "dataset":
            why = datasets(pid, name) if datasets else f"dataset {pid}: not checked"
            if why:
                missing.append((i, why))
            continue
        plugin = by_id.get(pid)
        if plugin is None:
            missing.append((i, f"{kind} {pid}: no such plugin in the pinned engine or the line"))
            continue
        if kind == "overlay" and plugin.get("group") != "overlay":
            missing.append((i, f"overlay {pid}: registered as a {plugin.get('group')}, not an overlay"))
        if kind == "param":
            props = (plugin.get("params_schema") or {}).get("properties") or {}
            if plugin.get("params_schema") is None:
                missing.append((i, f"param {pid}.{name}: {pid} declares no params_schema, so nothing is confirmable"))
            elif name not in props:
                missing.append(
                    (i, f"param {pid}.{name}: not a parameter of {pid} (has: {', '.join(sorted(props)) or 'none'})")
                )
    return missing


# Run in the line's env with the line dir as cwd: the engine's OWN loader for a `source:` plugin, the
# one `quantbox run` calls (runner._load_local_source_class, v0.18.0 and later). argv[1] is a JSON list
# of sources; it prints {source: null | "<why it does not load>"}. `python -c` puts the cwd first on
# sys.path and `quantbox run` does not, so it is dropped: a plugin importing a sibling module fails here too.
LOADER = """import json, sys
sys.path[:] = [p for p in sys.path if p not in ("", ".")]
from quantbox.runner import _load_local_source_class as load
out = {}
for src in json.loads(sys.argv[1]):
    try:
        load(src)
        out[src] = None
    except Exception as e:
        out[src] = f"{type(e).__name__}: {e}"
print(json.dumps(out))
"""
SCRATCH = Path(os.environ.get("QUARK_TMP", Path.home() / "workspace" / "tmp"))


def needed_line_plugins(needs: list[tuple[str, str, str | None]], own: list[dict], catalog: dict) -> dict[str, list[int]]:
    """{source: [index of every need it answers]} for the line plugins a need resolves to (engine ids win)."""
    engine = {p["id"] for p in catalog.get("plugins", [])}
    by_id = {p["id"]: p["source"] for p in own if p["id"] not in engine}
    used: dict[str, list[int]] = {}
    for i, (kind, pid, _) in enumerate(needs):
        if kind != "dataset" and pid in by_id:
            used.setdefault(by_id[pid], []).append(i)
    return used


def load_line_plugins(
    lab: Path, ref: str, line: str, env: Path, uv: str, used: dict[str, list[int]], timeout: int
) -> list[tuple[int, str]] | str:
    """Load each line plugin a need resolves to with quantbox's own loader, in the pinned engine.

    The `ast` read only finds what a file declares; the engine decides what loads (TOM-2514: a
    `@dataclass` with `from __future__ import annotations` read as expressible, and the run then
    failed at start). The line is extracted at `ref`, so this runs the line's own code: the code
    the measured run would run. Returns the missing rows, or a reason string when it could not check.
    """
    tar = subprocess.run(["git", "-C", str(lab), "archive", "--format=tar", ref, f"research/{line}"], capture_output=True)
    if tar.returncode != 0:
        return f"git archive {ref} research/{line} failed: {tail(tar.stderr.decode(errors='replace'))}"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="quark-capability-", dir=SCRATCH) as tmp:
        with tarfile.open(fileobj=io.BytesIO(tar.stdout)) as tf:
            tf.extractall(tmp, filter="data")
        r = subprocess.run(
            [uv, "run", "--frozen", "--quiet", "--project", str(env), "python", "-c", LOADER, json.dumps(sorted(used))],
            capture_output=True,
            text=True,
            cwd=Path(tmp) / "research" / line,
            timeout=timeout,
        )
    if r.returncode != 0:
        return f"the pinned engine's loader could not be called (exit {r.returncode}): {tail(r.stderr)}"
    try:
        result = json.loads(r.stdout.strip().splitlines()[-1])
        assert isinstance(result, dict) and set(result) == set(used)
    except (ValueError, AssertionError, IndexError):
        return "the loader printed no {source: reason} document for every line plugin it was given"
    return [
        (i, f"line plugin {src} does not load in the pinned engine: {why}")
        for src, why in sorted(result.items())
        if why
        for i in used[src]
    ]


def ladder(kind: str, needs_raw: list[str], cap_missing: list[tuple[int, str]]) -> dict:
    """What a `needs_capability` result asks the dev-loop to do next (Tom's D2, TOM-2233).

    A replication climbs: the engine and data it has did not cover these needs, so the run
    writes each as a line plugin (a `source:` strategy/overlay, or a data plugin that freezes
    its data) and checks again. Anything else keeps the re-grade: the gap goes to meta and Tom.
    """
    if kind != "replication":
        return {"action": "re_grade"}
    gaps = sorted({needs_raw[i] for i, _ in cap_missing if i >= 0})
    return {
        "rung": "b",
        "action": "write_plugin",
        "code": [g for g in gaps if not g.startswith("dataset:")],
        "data": [g for g in gaps if g.startswith("dataset:")],
    }


def cmd_capability(a: argparse.Namespace) -> int:
    lab, line, ref = Path(a.lab), a.line.strip(), a.ref
    out: dict = {"verb": "capability", "lab": str(lab), "line": line, "ref": ref, "needs": a.need_raw}
    if line.lower().startswith("new:"):
        return emit(
            1,
            {
                **out,
                "result": "needs_line",
                "missing": [
                    f"line {line!r} does not exist yet: the dev-loop opens it (`quantbox-iface.py line`, Step 1c) and writes its prereg: block"
                ],
            },
        )
    if not (lab / "HEAD").exists() and not (lab / ".git").exists():
        return emit(3, {**out, "result": "blind", "reason": f"no lab at {lab} on this box (labs live on forge)"})
    if not a.no_fetch:
        f = git(lab, "fetch", "-q", "origin")
        if f.returncode != 0:
            return emit(3, {**out, "result": "blind", "reason": f"git fetch failed: {tail(f.stderr)}"})
    if ref is None:  # the lab's working branch: `dev` where it exists (quantbox-lab ADR-0001), else `main`
        has_dev = git(lab, "rev-parse", "--verify", "-q", "origin/dev^{commit}").returncode == 0
        ref = out["ref"] = "origin/dev" if has_dev else "origin/main"
    base = f"{ref}:research/{line}"
    if git(lab, "cat-file", "-t", base).stdout.strip() != "tree":
        return emit(1, {**out, "result": "needs_line", "missing": [f"research/{line} does not exist at {ref}"]})

    def show(name: str) -> str | None:
        r = git(lab, "show", f"{base}/{name}")
        return r.stdout if r.returncode == 0 else None

    line_missing: list[str] = []
    readme = show("README.md")
    if readme is None:
        line_missing.append("no README.md in the line")
    elif (state := prereg_state(readme)) != "ok":
        line_missing.append(
            "README.md carries no fenced `prereg:` block"
            if state == "absent"
            else "README.md `prereg:` block still holds `<...>` template placeholders"
        )
    pyproject, lock = show("pyproject.toml"), show("uv.lock")
    out["engine_pin"] = engine_pin(pyproject) if pyproject else None
    if pyproject is None or lock is None:
        line_missing.append("the line pins no engine (pyproject.toml + uv.lock)")
        return emit(1, {**out, "result": "needs_line", "missing": line_missing})

    uv = uv_bin()
    if uv is None:
        return emit(3, {**out, "result": "blind", "reason": "uv not found on this box"})
    env = line_env(pyproject, lock)
    r = subprocess.run(
        [uv, "run", "--frozen", "--quiet", "--project", str(env), "quantbox", "plugins", "schema", "--json"],
        capture_output=True,
        text=True,
        cwd=env,
        timeout=a.timeout,
    )
    if r.returncode != 0:
        if r.returncode == 2 and "Unknown action: schema" in r.stderr:
            line_missing.append(
                f"engine pin {out['engine_pin']} has no `quantbox plugins schema --json` (TOM-1350): {REPIN_HINT}"
            )
            return emit(1, {**out, "result": "needs_line", "missing": line_missing})
        return emit(3, {**out, "result": "blind", "reason": f"plugins schema exited {r.returncode}: {tail(r.stderr)}"})
    try:
        catalog = json.loads(r.stdout)
        assert isinstance(catalog.get("plugins"), list)
    except (ValueError, AssertionError, AttributeError):
        return emit(
            3, {**out, "result": "blind", "reason": "plugins schema --json did not print a {plugins: [...]} document"}
        )

    listing = git(lab, "ls-tree", "-r", "--name-only", base).stdout.splitlines()
    sources = {
        rel: show(rel) or ""
        for rel in listing
        if rel.endswith(".py") and not any(part.startswith((".", "test")) for part in Path(rel).parts)
    }
    own = line_plugins(sources)
    lock_text = show("datasets.lock")
    try:
        pins = yaml.safe_load(lock_text) if lock_text else {}
    except yaml.YAMLError:
        pins = None

    def served(name: str, column: str, manifest_text: str | None) -> str | None:
        """None when the manifest's `serve.columns` (quantbox TOM-2232) names `column`."""
        try:
            manifest = yaml.safe_load(manifest_text or "")
        except yaml.YAMLError:
            return f"dataset {name}: its manifest.yaml does not parse, so column {column} is not confirmable"
        block = manifest.get("serve") if isinstance(manifest, dict) else None
        columns = block.get("columns") if isinstance(block, dict) else None
        if not (isinstance(columns, list) and all(isinstance(c, str) for c in columns) and columns):
            return f"dataset {name}: its manifest declares no serve.columns, so column {column} is not confirmable"
        if column not in columns:
            return f"dataset {name}: column {column} is not served (serve.columns: {', '.join(columns)})"
        return None

    def dataset(name: str, column: str | None = None) -> str | None:
        """None when `name` is readable by name: under the datasets root, or frozen in the line and pinned.

        With `column`, also that the manifest serving it names that column in `serve.columns`.
        The root wins when it holds the name, as quantbox's own resolution does (dataset_lock._locate).
        """
        root_manifest = DATASETS_ROOT / name / "manifest.yaml"
        if root_manifest.is_file():
            return served(name, column, root_manifest.read_text()) if column else None
        if f"datasets/{name}/manifest.yaml" not in listing:
            return f"dataset {name}: not in quantbox-datasets and not frozen in the line (datasets/{name}/manifest.yaml)"
        if not isinstance(pins, dict) or not pins.get(name):
            return f"dataset {name}: frozen in the line but not pinned in its datasets.lock"
        return served(name, column, show(f"datasets/{name}/manifest.yaml")) if column else None

    cap = check_needs(catalog, a.need, own, dataset)
    used = needed_line_plugins(a.need, own, catalog)
    if used:
        loaded = load_line_plugins(lab, ref, line, env, uv, used, a.timeout)
        if isinstance(loaded, str):
            return emit(3, {**out, "result": "blind", "reason": f"line plugins not checked: {loaded}"})
        cap += loaded
    cap_missing = list(dict.fromkeys(why for _, why in cap))  # a plugin two needs name fails once
    kind = a.kind or (prereg_kind(readme) if readme else None) or "hypothesis"
    out.update(
        catalog_plugins=len(catalog["plugins"]),
        kind=kind,
        line_plugins=[{k: p[k] for k in ("id", "group", "source")} for p in own],
    )
    if line_missing:
        return emit(1, {**out, "result": "needs_line", "missing": line_missing + cap_missing})
    if cap_missing:
        return emit(
            1, {**out, "result": "needs_capability", "missing": cap_missing, "ladder": ladder(kind, a.need_raw, cap)}
        )
    return emit(0, {**out, "result": "pass", "missing": []})


def latest_tag(url: str) -> str | None:
    r = subprocess.run(["git", "ls-remote", "--tags", "--refs", url, "v*"], capture_output=True, text=True)
    tags = []
    for ln in r.stdout.splitlines():
        m = re.search(r"refs/tags/(v(\d+)\.(\d+)\.(\d+))$", ln)
        if m:
            tags.append((tuple(int(x) for x in m.groups()[1:]), m.group(1)))
    return max(tags)[1] if r.returncode == 0 and tags else None


def cmd_line(a: argparse.Namespace) -> int:
    wt, slug = Path(a.worktree).resolve(), a.line.strip()
    if slug.lower().startswith("new:"):
        slug = slug[4:].strip()
    line_dir = wt / "research" / slug
    out: dict = {"verb": "line", "worktree": str(wt), "line": slug, "line_dir": str(line_dir)}

    def refuse(reason: str, **extra) -> int:
        return emit(1, {**out, **extra, "result": "refused", "reason": reason})

    # The run's own worktree, on its own branch: a line opened by Quark reaches the lab as a PR only.
    if not (wt / ".git").is_file():
        return refuse(f"{wt} is not a linked git worktree; open the line only in the run's fresh worktree")
    branch = git(wt, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    out["branch"] = branch
    if branch in PROTECTED or branch in ("HEAD", ""):
        return refuse(f"worktree is on {branch!r}; a line opened by the loop lands on a research/ branch and a PR")
    uv = uv_bin()
    if uv is None:
        return emit(3, {**out, "result": "blind", "reason": "uv not found on this box"})
    ref = a.ref or latest_tag(QUANTBOX_URL)
    if ref is None:
        return emit(3, {**out, "result": "blind", "reason": f"could not list the v* tags of {QUANTBOX_URL}"})
    out["ref"] = ref
    env = dict(os.environ)
    if "QUANTBOX_DATASETS_ROOT" not in env and DATASETS_ROOT.is_dir():
        env["QUANTBOX_DATASETS_ROOT"] = str(DATASETS_ROOT)
    tool = [uv, "tool", "run", "--quiet", "--from", f"quantbox @ git+{QUANTBOX_URL}@{ref}", "quantbox"]

    def qb(*args: str, cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run([*tool, *args], capture_output=True, text=True, cwd=cwd, env=env, timeout=a.timeout)

    def in_line(*args: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [uv, "run", "--frozen", "--quiet", *args],
            capture_output=True,
            text=True,
            cwd=line_dir,
            env={**env, **(extra_env or {})},
            timeout=a.timeout,
        )

    scaffold = ["--question", a.question] if a.question else []
    pyproject = line_dir / "pyproject.toml"
    will_scaffold = not line_dir.exists() or not any(line_dir.iterdir()) or not pyproject.exists()
    if a.kind == "replication" and will_scaffold:
        # A replication's prereg block is anchored to its paper. `new line --kind replication` writes that
        # block (quantbox#312, TOM-2277); an engine without the flag scaffolds a book block, said loudly.
        scaffold += ["--dataset", a.dataset]
        # Typer's rich help styles `-` and `-kind` apart (`\x1b[1;36m-\x1b[0m\x1b[1;36m-kind`): strip the escapes.
        help_text = ANSI.sub("", qb("new", "line", "--help", cwd=wt).stdout)
        if re.search(r"--kind\b", help_text):
            scaffold += ["--kind", "replication"]
            out["prereg_kind"] = "replication"
        else:
            out["prereg_kind"] = "book"
            out["warning"] = (
                f"quantbox {ref} has no `new line --kind` (quantbox#312, after v0.19.0): the README's prereg "
                "block is a BOOK block. Rewrite it as `kind: replication` per preregistration.md "
                '§ "A paper replication" before Step 1b.'
            )
            print(f"quantbox-iface line: WARNING: {out['warning']}", file=sys.stderr)
    if not line_dir.exists() or not any(line_dir.iterdir()):
        r = qb("new", "line", slug, "--dir", str(line_dir), "--quantbox-ref", ref, *scaffold, cwd=wt)
        if r.returncode != 0:
            return refuse(f"quantbox new line refused: {tail(r.stderr, 3)}")
        out.update(action="new", added=sorted(p.name for p in line_dir.iterdir()))
    elif not pyproject.exists():
        # A line that pins nothing yet (README only, e.g. carry-trend): scaffold beside its files,
        # never over one, and give its README the template's Pre-registration section if it has none.
        tmp = wt / ".line-scaffold" / slug
        r = qb("new", "line", slug, "--dir", str(tmp), "--quantbox-ref", ref, *scaffold, cwd=wt)
        if r.returncode != 0:
            shutil.rmtree(tmp.parent, ignore_errors=True)
            return refuse(f"quantbox new line refused: {tail(r.stderr, 3)}")
        added = []
        for src in sorted(tmp.rglob("*")):
            dst = line_dir / src.relative_to(tmp)
            if src.is_file() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                added.append(str(src.relative_to(tmp)))
        readme = line_dir / "README.md"
        if readme.exists() and prereg_state(readme.read_text()) == "absent":
            tpl = (tmp / "README.md").read_text()
            section = tpl[tpl.index("## Pre-registration") :].split("\n## Findings")[0].rstrip()
            readme.write_text(readme.read_text().rstrip() + "\n\n" + section + "\n")
            added.append("README.md (## Pre-registration appended)")
        shutil.rmtree(tmp.parent, ignore_errors=True)
        out.update(action="pin", added=added)
    elif PINS_BEGIN not in pyproject.read_text():
        return refuse(
            "legacy-pinned line (own pyproject.toml without the managed pins block): converting it would move "
            "a frozen line's numbers, so it stays a lab session's call. Re-grade the note, name this in the report.",
            action="none",
        )
    else:
        floor_reason = below_floor(pyproject.read_text())
        probe = in_line("quantbox", "plugins", "schema", "--json")
        if probe.returncode == 0 and not a.repin and floor_reason is None:
            out.update(action="none", added=[])
        else:
            if floor_reason is not None and probe.returncode == 0 and not a.repin:
                # dev-loop.md says such a line "is repinned first": the script does it, and says so.
                out["repinned_because"] = f"{floor_reason} (datasets by name, TOM-2232)"
                print(f"quantbox-iface line: repinning {slug} to {ref}: {out['repinned_because']}", file=sys.stderr)
            runs_root = artifacts_root(line_dir)
            before = {p.name for p in runs_root.iterdir()} if runs_root.is_dir() else set()

            def keep_runs_out() -> None:
                """Move the runs this repin made out of the line, so they never count as its run evidence."""
                made = sorted(p for p in runs_root.iterdir() if p.name not in before) if runs_root.is_dir() else []
                if not made:
                    return
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                dest = REPIN_RUNS / slug / stamp
                dest.mkdir(parents=True, exist_ok=True)
                for p in made:
                    shutil.move(str(p), str(dest / p.name))
                out["repin_runs_moved_to"] = str(dest)
                out["repin_runs_moved"] = [p.name for p in made]

            # A repin either moves a number or it doesn't; the golden decides, so it must exist first.
            golden = line_dir / "expected_metrics.json"
            out["golden_captured_on_old_pin"] = False
            if not golden.exists():
                g = in_line("pytest", "-m", "reproduction", "-q", extra_env={"QUANTBOX_CAPTURE_GOLDEN": "1"})
                if g.returncode != 0 or not golden.exists():
                    keep_runs_out()
                    return refuse(
                        f"could not capture a golden on the old pin before repinning: {tail(g.stdout or g.stderr, 3)}"
                    )
                out["golden_captured_on_old_pin"] = True
            r = qb("line", "repin", str(line_dir), "--ref", ref, cwd=wt)
            if r.returncode != 0:
                keep_runs_out()
                return refuse(f"quantbox line repin refused: {tail(r.stderr, 3)}")
            repro = in_line("pytest", "-m", "reproduction", "-q")
            keep_runs_out()
            out.update(action="repin", added=[], reproduction="green" if repro.returncode == 0 else "red")
            if repro.returncode != 0:
                return refuse(
                    "reproduction RED after the repin: the engine moved a number. A finding to record, never a "
                    f"tolerance to loosen: {tail(repro.stdout or repro.stderr, 3)}"
                )
    readme = line_dir / "README.md"
    out["prereg"] = prereg_state(readme.read_text()) if readme.exists() else "absent"
    return emit(0, {**out, "result": "ready"})


def validate_manifest(uv: str, line_dir: Path, manifest_text: str, timeout: int) -> str | None:
    """Validate against the run@1 schema SHIPPED with the pinned engine. None = valid, else the reason."""
    code = (
        "import json,sys,importlib.resources as r;from jsonschema import Draft202012Validator as V;"
        "s=json.loads((r.files('quantbox')/'artifact_schemas'/'run_manifest.schema.json').read_text());"
        "e=sorted(V(s).iter_errors(json.load(sys.stdin)),key=str);"
        "print('; '.join(x.message for x in e[:3]));sys.exit(1 if e else 0)"
    )
    r = subprocess.run(
        [uv, "run", "--frozen", "--quiet", "python", "-c", code],
        input=manifest_text,
        capture_output=True,
        text=True,
        cwd=line_dir,
        timeout=timeout,
    )
    if r.returncode == 0:
        return None
    return (r.stdout.strip() or tail(r.stderr)) or f"validator exited {r.returncode}"


def cmd_run(a: argparse.Namespace) -> int:
    line_dir = Path(a.line_dir).resolve()
    out: dict = {"verb": "run", "line_dir": str(line_dir), "config": a.config}
    if not (line_dir / "uv.lock").exists():
        return emit(
            3,
            {**out, "result": "blind", "reason": f"{line_dir} has no uv.lock (not a pinned line, or not on this box)"},
        )
    cfg_path = line_dir / a.config
    if not cfg_path.exists():
        return emit(1, {**out, "result": "failed", "reason": f"config not found: {cfg_path}"})
    uv = uv_bin()
    if uv is None:
        return emit(3, {**out, "result": "blind", "reason": "uv not found on this box"})
    base = [uv, "run", "--frozen", "--quiet", "quantbox", "run"]

    # Capability probe, by EXIT CODE: an engine without --json refuses the option (2)
    # before it reads anything; one with it prints help (0).
    probe = subprocess.run([*base, "--json", "--help"], capture_output=True, text=True, cwd=line_dir, timeout=a.timeout)
    if probe.returncode != 0:
        why = (
            "has no `quantbox run --json` (quantbox/run@1, TOM-1348)"
            if probe.returncode == 2
            else f"probe exited {probe.returncode}: {tail(probe.stderr)}"
        )
        return emit(
            3, {**out, "result": "blind", "reason": f"the line's engine {why}; {REPIN_HINT}. No metrics were read."}
        )

    r = subprocess.run(
        [*base, "--json", "-c", a.config], capture_output=True, text=True, cwd=line_dir, timeout=a.timeout
    )
    manifest = None
    try:
        doc = json.loads(r.stdout) if r.stdout.strip() else None
        if isinstance(doc, dict) and doc.get("schema") == RUN_SCHEMA:
            manifest = doc
    except ValueError:
        pass
    if r.returncode != 0 and manifest is None:
        return emit(1, {**out, "result": "failed", "rc": r.returncode, "reason": f"Backtest failed: {tail(r.stderr)}"})
    if manifest is None:
        return emit(3, {**out, "result": "blind", "reason": f"exit 0 but stdout is not a {RUN_SCHEMA} manifest"})
    bad = validate_manifest(uv, line_dir, r.stdout, a.timeout)
    if bad:
        return emit(3, {**out, "result": "blind", "reason": f"manifest fails the engine's own run@1 schema: {bad}"})

    root = Path(((yaml.safe_load(cfg_path.read_text()) or {}).get("artifacts") or {}).get("root", "artifacts"))
    run_dir = (root if root.is_absolute() else line_dir / root) / manifest["run_id"]
    out.update(
        run_dir=str(run_dir),
        files={k: (str(run_dir / v) if v else None) for k, v in (manifest.get("files") or {}).items()},
        manifest=manifest,
    )
    if r.returncode != 0:  # the manifest was written, then the CLI refused (rebalancer frozen, quantbox#120)
        return emit(
            1,
            {
                **out,
                "result": "failed",
                "rc": r.returncode,
                "reason": f"run exited {r.returncode} after writing its manifest: {tail(r.stderr)}",
            },
        )
    return emit(0, {**out, "result": "ok"})


#: `quantbox preflight` exit code -> (result, the word its `PREFLIGHT:` line carries). ADR-0014: 3 is FLAG.
PREFLIGHT_CODES = {0: ("pass", "PASS"), 1: ("refused", "REFUSE"), 2: ("blind", "COULD NOT CHECK"), 3: ("flag", "FLAG")}
PREFLIGHT_FLOOR = "v0.21.0"
NO_PLUGIN = "never fall back to the qute-research plugin's quant_preflight.py"


def cmd_preflight(a: argparse.Namespace) -> int:
    lab, line_dir = Path(a.lab).resolve(), Path(a.line_dir).resolve()
    env_dir = line_dir if a.env == "line" else lab
    out: dict = {"verb": "preflight", "lab": str(lab), "line_dir": str(line_dir), "env": a.env, "env_dir": str(env_dir)}

    def blind(reason: str, **extra) -> int:
        return emit(2, {**out, **extra, "result": "blind", "reason": reason})

    if not (env_dir / "uv.lock").exists():
        return blind(f"{env_dir} has no uv.lock: no pinned quantbox to run the gate with (not on this box?)")
    pyproject = env_dir / "pyproject.toml"
    out["engine_pin"] = engine_pin(pyproject.read_text()) if pyproject.exists() else None
    uv = uv_bin()
    if uv is None:
        return blind("uv not found on this box")
    base = [uv, "run", "--frozen", "--quiet", "--project", str(env_dir), "quantbox", "preflight"]

    # The verb, by EXIT CODE and by its own usage: `quantbox preflight --help` prints the gate's argparse
    # help (exit 0, names --lab); an engine without the verb is a Typer "No such command" (exit 2).
    probe = subprocess.run([*base, "--help"], capture_output=True, text=True, cwd=line_dir, timeout=a.timeout)
    if probe.returncode != 0 or "--lab" not in probe.stdout:
        said = [ln.strip(" │╭╮╰╯─") for ln in ANSI.sub("", probe.stderr or probe.stdout).splitlines()]
        said = next((ln for ln in said if "No such command" in ln), None) or tail("\n".join(s for s in said if s))
        return blind(
            f"the {a.env} env's quantbox ({out['engine_pin'] or 'pin unknown'}) has no `quantbox preflight` "
            f"(it ships in {PREFLIGHT_FLOOR}; probe exited {probe.returncode}: {said}). "
            f"Repin to {PREFLIGHT_FLOOR} or later; {NO_PLUGIN}. Nothing was checked.",
        )

    def absolute(p: str) -> str:
        return str(Path(p) if Path(p).is_absolute() else (line_dir / p).resolve())

    argv = ["--lab", str(lab), "--line", str(line_dir)]
    if a.prereg:
        argv += ["--prereg", absolute(a.prereg)]
    if a.id:
        argv += ["--id", a.id]
    for d in a.runs_dir:
        argv += ["--runs-dir", absolute(d)]
    if a.data_validation:
        argv += ["--data-validation", absolute(a.data_validation)]
    argv += [absolute(c) for c in a.configs]
    r = subprocess.run([*base, *argv], capture_output=True, text=True, cwd=line_dir, timeout=a.timeout)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    verdict = next((ln for ln in reversed(lines) if ln.startswith("PREFLIGHT: ")), None)
    out.update(
        rc=r.returncode,
        verdict=verdict,
        limits=next((ln for ln in lines if ln.startswith("finding limits: ")), None),
        flags=[ln.split()[1] for ln in lines if ln.startswith("FLAG ") and len(ln.split()) > 1],
        stdout=lines,
    )
    known = PREFLIGHT_CODES.get(r.returncode)
    # The code and the gate's own last line must say the same thing; anything else did not finish looking.
    if known is None or verdict is None or not verdict.startswith(f"PREFLIGHT: {known[1]}"):
        return blind(
            f"`quantbox preflight` exited {r.returncode} with summary {verdict!r}: not the gate's answer "
            f"({tail(r.stderr)}); {NO_PLUGIN}"
        )
    return emit(r.returncode, {**out, "result": known[0]})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="verb", required=True)
    c = sub.add_parser("capability", help="scout: is this hypothesis expressible on its line's pinned engine?")
    c.add_argument("--lab", required=True, help="absolute path of the lab repo (bare or not)")
    c.add_argument("--line", required=True, help="line dir name under research/, or 'new: <name>'")
    c.add_argument("--ref", default=None, help="git ref to read the line at (default: origin/dev, else origin/main)")
    c.add_argument(
        "--need",
        action="append",
        default=[],
        metavar="KIND:ID[.PARAM]",
        help="plugin:<id> | overlay:<id> | param:<plugin id>.<param> | dataset:<name>[:<served column>]; repeatable",
    )
    c.add_argument(
        "--kind",
        choices=("hypothesis", "replication"),
        default=None,
        help="replication: a gap returns ladder write_plugin, not a re-grade (default: the prereg's kind)",
    )
    c.add_argument("--no-fetch", action="store_true")
    c.add_argument("--timeout", type=int, default=900)
    ln = sub.add_parser("line", help="dev-loop: open / pin / repin research/<line> in the run's lab worktree")
    ln.add_argument(
        "--worktree", required=True, help="ABSOLUTE path of the run's fresh lab worktree (never the lab root)"
    )
    ln.add_argument("--line", required=True, help="line dir name under research/, or 'new: <name>'")
    ln.add_argument("--ref", default=None, help="quantbox tag to pin (default: the latest v* tag)")
    ln.add_argument(
        "--repin",
        action="store_true",
        help="repin a Model C line even though its engine has the schema and is at or above v0.18.0 "
        "(a line below that floor is repinned without this flag)",
    )
    ln.add_argument("--question", default=None, help="one sentence for a new line's README")
    ln.add_argument(
        "--kind",
        choices=("hypothesis", "replication"),
        default="hypothesis",
        help="replication: scaffold with `new line --kind replication` (a paper-anchored prereg block)",
    )
    ln.add_argument("--dataset", default=None, help="the primary dataset a replication's base config reads")
    ln.add_argument("--timeout", type=int, default=3000)
    r = sub.add_parser("run", help="dev-loop: one backtest via `quantbox run --json` in the line dir")
    r.add_argument("--line-dir", required=True, help="ABSOLUTE line dir inside the run's fresh lab worktree")
    r.add_argument("--config", required=True, help="config path relative to the line dir")
    r.add_argument("--timeout", type=int, default=3000)
    pf = sub.add_parser("preflight", help="dev-loop Step 1b: `quantbox preflight` of the pinned engine (exit 0/1/2/3)")
    pf.add_argument("--lab", required=True, help="ABSOLUTE path of the run's fresh lab worktree (holds system.yaml)")
    pf.add_argument("--line-dir", required=True, help="ABSOLUTE line dir inside that worktree")
    pf.add_argument(
        "--env",
        choices=("line", "lab"),
        default="line",
        help="whose quantbox runs the gate: the line's pin (default) or the lab root's; never both, never the plugin",
    )
    pf.add_argument("--prereg", default=None, help="file carrying the prereg block (default <line>/README.md)")
    pf.add_argument("--id", default=None, help="hypothesis id, when the file carries several blocks")
    pf.add_argument("--runs-dir", action="append", default=[], help="extra artifacts root; repeatable")
    pf.add_argument("--data-validation", default=None, help="a replication: the run's data_validation.json")
    pf.add_argument("--timeout", type=int, default=900)
    pf.add_argument("configs", nargs="+", help="config paths, relative to the line dir or absolute")
    a = p.parse_args(argv)
    if a.verb == "line" and a.kind == "replication" and not a.dataset:
        p.error("line --kind replication needs --dataset <the primary dataset>")
    if a.verb == "capability":
        a.need_raw = list(a.need)
        try:
            a.need = [parse_need(s) for s in a.need]
        except argparse.ArgumentTypeError as e:
            p.error(str(e))
    try:
        verbs = {"capability": cmd_capability, "line": cmd_line, "run": cmd_run, "preflight": cmd_preflight}
        return verbs[a.verb](a)
    except subprocess.TimeoutExpired as e:
        # A capability check, line build or pre-flight that did not finish looked at nothing (BLIND:
        # 3, or the gate's own 2 for preflight); a backtest that did not finish is a failed run.
        code, result = {"run": (1, "failed"), "preflight": (2, "blind")}.get(a.verb, (3, "blind"))
        return emit(
            code,
            {
                "verb": a.verb,
                "result": result,
                "reason": f"timed out after {e.timeout}s: {' '.join(map(str, e.cmd[-4:]))}",
            },
        )


if __name__ == "__main__":
    sys.exit(main())
