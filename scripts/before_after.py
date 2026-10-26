#!/usr/bin/env python3
""":"
# hotl verify runs this file as `bash <script> <base ref> <out dir>`: these four lines are bash,
# and the start of the docstring to Python. bash hands the file to the project's Python here and
# never reads further.
exec uv run --no-sync python "$0" "$@"

The before/after of a quantbox PR: the result at base and at head, diffed (TOM-2458).

    uv run python scripts/before_after.py [--plan] <base ref> <out dir>

--plan prints the per-file mapping and the configs it would run, runs nothing, and
exits 3 (a plan is not a comparison).

hotl verify's contract (tomlupo/hotl README, "Adopt in a repo" step 6): run from the
HEAD checkout; exit 0 = base and head identical, 1 = they differ (or a step is owed
that this script cannot run) and <out dir>/summary.md says what, anything else = the
compare could not run. Decision: vaults/projects/infra/decisions/
decyzja-2026-10-26-stos-kiss-repo-hotl-fabryka.md, "Lekki compare".

What it does, in order:

1. MAP. Every file of `git diff <merge-base of base and HEAD> HEAD` goes to one step of
   data -> signal -> decision -> engine -> trade, by .hotl/compare.yml (the quant-lib
   default in its comment block plus quantbox's additions). A src/quantbox module no
   glob matches takes the earliest step of the modules that import it (the import
   graph); a module of the import-linter `trade` layer (scripts/layer_map.py) is trade.
   A path that maps to no step runs EVERYTHING. The per-file mapping is printed.
2. FAST EXIT. A diff whose every file is ignored (docs, tests, .github) exits 0 at once.
3. SELECT. Of the configs compare.yml lists, a config runs when its code closure (its
   plugins' modules and the runner, through the import graph at base OR head) holds a
   touched module, when it is touched itself, or when it reads a touched data file.
   An unmapped path selects every config. A package __init__ loaded only because a
   module inside it was imported does not pull in its sibling re-exports (importing one
   strategy does not make every strategy part of the run).
4. RUN. Each selected config runs at base (a git worktree of the merge base, under
   <out dir>/base-tree) and at head, in a fresh process with PYTHONHASHSEED=0 and
   TZ=UTC. The process prints which quantbox it imported, and a side that imported the
   other side's code is refused. A by-name dataset is read from the build that side's
   datasets.lock pins, extracted from the quantbox-datasets clone that
   $QUANTBOX_DATASETS_ROOT names (its working copy, else its git history).

The plan is read from .hotl/compare.yml beside THIS script, not from the checkout:
hotl takes the script from the base branch so a PR cannot rewrite how it is verified.

Known limits. A module loaded by a computed name (an f-string) that no config names
and no string constant names is not in any closure. A config outside compare.yml's
list never runs here. Both are covered by the release (`/ship`), which runs
everything.
5. DIFF. Every file of the two run directories but events.jsonl (timings): JSON field
   by field, parquet frame by frame, text line by line, after masking what differs on
   every run (run id, timestamps, the checkout path, git and installed packages).
   Floats compare at rel 1e-9, abs 1e-12.
6. TRADE. A touched trade-step file, or a touched module that trade code calls (the
   code closure of the import-linter trade layer), owes the live shadow run (quantbox-live
   scripts/shadow_run.py). quantbox is public and has no forge runner, so this script
   cannot run it: summary.md names it and the exit is 1.

A config that fails on one side only is a difference. A config that fails on both
(no datasets root, say) could not be compared: the exit is 2, never 0.
"""

import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path.cwd().resolve()  # the HEAD checkout the compare runs in
SCRIPT_TREE = Path(__file__).resolve().parents[1]  # the tree this script (and its plan) came from
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer_map import layers_of, read_layers  # noqa: E402

PLAN_FILE = ".hotl/compare.yml"
DEFAULT_START = "# --- quant-lib default ---"
DEFAULT_END = "# --- end quant-lib default ---"
EXCLUDED_FILES = {"events.jsonl"}  # wall-clock timings of the run, nothing Tom reads
DROPPED_KEYS = {"git", "installed_packages"}  # the checkout and the venv, not the result
RTOL, ATOL = 1e-9, 1e-12
MAX_LINES = 40  # differences printed per file
ALL = "ALL"

# Wall-clock stamps of the run itself. Only a stamp dated on a day the compare ran is
# masked: a data date ("2024-01-05T00:00:00") is a result and must still differ.
TIMESTAMP_RES = [
    re.compile(r"(\d{8})T\d{6}Z"),
    re.compile(r"(\d{4}-\d{2}-\d{2})[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"),
]
RUN_DAYS: set[str] = set()  # filled by run_days() at the start and the end of the compare


def run_days() -> None:
    import datetime as dt

    for now in (dt.datetime.now(dt.UTC), dt.datetime.now()):
        RUN_DAYS.update({now.strftime("%Y-%m-%d"), now.strftime("%Y%m%d")})


# Runs ONE config in a fresh process. argv: config path, artifacts root, result file.
DRIVER = """
import json, sys
from pathlib import Path
import yaml
import quantbox
from quantbox.registry import PluginRegistry
from quantbox.runner import run_from_config
cfg_path, art_root, out = sys.argv[1:4]
cfg = yaml.safe_load(Path(cfg_path).read_text())
cfg.setdefault("artifacts", {})["root"] = art_root
res = run_from_config(cfg, PluginRegistry.discover(), config_path=cfg_path)
Path(out).write_text(json.dumps({"run_id": res.run_id, "quantbox": quantbox.__file__}))
"""

# Prints the plugin catalogue of the quantbox on sys.path: {kind: {name: "module:Class"}}.
CATALOGUE = """
import json
from quantbox.plugins.builtins import BUILTIN_PLUGINS
print(json.dumps(BUILTIN_PLUGINS))
"""


class CouldNotRun(Exception):
    """The compare could not run: exit 2."""


# --------------------------------------------------------------------------- plan


def load_plan(root: Path) -> dict[str, Any]:
    """compare.yml merged onto the quant-lib default its comment block carries."""
    text = (root / PLAN_FILE).read_text()
    lines = text.splitlines()
    try:
        start, end = lines.index(DEFAULT_START), lines.index(DEFAULT_END)
    except ValueError as exc:
        raise CouldNotRun(
            f"{PLAN_FILE}: no quant-lib default block between {DEFAULT_START!r} and {DEFAULT_END!r}"
        ) from exc
    default = yaml.safe_load("\n".join(re.sub(r"^# ?", "", line) for line in lines[start + 1 : end]))
    repo = yaml.safe_load(text) or {}
    if repo.get("type") != default.get("type"):
        raise CouldNotRun(f"{PLAN_FILE}: type {repo.get('type')!r}, but the default block is {default.get('type')!r}")
    steps = list(default["steps"])
    paths = {s: list(default["paths"].get(s) or []) + list((repo.get("paths") or {}).get(s) or []) for s in steps}
    return {
        "steps": steps,
        "ignore": list(default.get("ignore") or []) + list(repo.get("ignore") or []),
        "paths": paths,
        "import_graph": repo.get("import_graph") or {},
        "configs": list(repo.get("configs") or []),
        "shadow_run": repo.get("shadow_run") or {},
    }


def glob_re(pattern: str) -> re.Pattern[str]:
    """A path glob as a regex: `**/` any directories (or none), `**` anything, `*` and `?` within one segment."""
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(rf"\A{out}\Z")


def first_match(path: str, patterns: list[str]) -> str | None:
    return next((p for p in patterns if glob_re(p).match(path)), None)


# ---------------------------------------------------------------- import graph


@dataclass
class Graph:
    """quantbox's module graph in one tree: module -> the quantbox modules it imports."""

    edges: dict[str, set[str]] = field(default_factory=dict)  # explicit imports (and lazy-load strings)
    parents: dict[str, str] = field(default_factory=dict)  # module -> its package, loaded first
    soft: dict[str, set[str]] = field(default_factory=dict)  # module -> packages it names as strings
    files: dict[str, str] = field(default_factory=dict)  # repo-relative path -> module

    def closure(self, start: set[str]) -> set[str]:
        """The modules whose code a run of *start* can call.

        A package's __init__ that is loaded only because a module INSIDE it was imported
        runs its own imports, but its re-exports of sibling modules are never called by
        that module (an explicit import of a sibling is its own edge). So through such a
        parent edge the closure follows the __init__'s imports from outside the package,
        not into it: importing one strategy does not make every strategy part of a run.
        """
        seen: dict[str, bool] = {}  # module -> reached as a parent only
        todo = [(m, False) for m in start if m in self.edges]
        while todo:
            mod, as_parent = todo.pop()
            if mod in seen and (seen[mod] is False or as_parent):
                continue
            seen[mod] = as_parent
            for dep in self.edges.get(mod, ()):
                if as_parent and dep.startswith(mod + "."):
                    continue
                todo.append((dep, False))
            if mod in self.parents:
                todo.append((self.parents[mod], True))
            # A package named as a string (`f"{pkg}.{name}"`) loads the module the CONFIG
            # names; that module is a root of its own (config_roots), so here only the
            # package's own imports are followed, as for a parent.
            todo.extend((pkg, True) for pkg in self.soft.get(mod, ()))
        return set(seen)

    def importers(self) -> dict[str, set[str]]:
        """module -> the modules that load it (an import, or being its package's member)."""
        rev: dict[str, set[str]] = {m: set() for m in self.edges}
        for mod, deps in self.edges.items():
            for dep in deps:
                rev.setdefault(dep, set()).add(mod)
        for mod, parent in self.parents.items():
            rev.setdefault(parent, set()).add(mod)
        return rev


def module_of(rel: str, package: str) -> str | None:
    """The dotted module of a repo-relative .py path under *package* (src/quantbox), else None."""
    if not rel.endswith(".py") or not rel.startswith(package.rstrip("/") + "/"):
        return None
    parts = list(Path(rel).with_suffix("").parts)[len(Path(package).parts) - 1 :]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def build_graph(tree: Path, package: str, catalogues: set[str]) -> Graph:
    """Static imports, parent packages, and quantbox module names written as strings (outside catalogues)."""
    graph = Graph()
    pkg_dir = tree / package
    if not pkg_dir.is_dir():
        return graph
    sources: dict[str, tuple[ast.AST, bool]] = {}
    for path in sorted(pkg_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(tree).as_posix()
        mod = module_of(rel, package)
        graph.files[rel] = mod
        try:
            sources[mod] = (ast.parse(path.read_text(encoding="utf-8")), path.name == "__init__.py")
        except SyntaxError:
            sources[mod] = (ast.Module(body=[], type_ignores=[]), path.name == "__init__.py")
    known = set(sources)

    def resolve(name: str) -> str | None:
        while name and name not in known:
            name = name.rpartition(".")[0]
        return name or None

    for mod, (node, is_pkg) in sources.items():
        deps: set[str] = set()
        parent = mod.rpartition(".")[0]
        if parent in known:
            graph.parents[mod] = parent  # importing a module imports its package first
        base_pkg = mod if is_pkg else parent
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                for alias in sub.names:
                    hit = resolve(alias.name)
                    if hit:
                        deps.add(hit)
            elif isinstance(sub, ast.ImportFrom):
                if sub.level:
                    anchor = base_pkg.split(".")
                    anchor = anchor[: len(anchor) - (sub.level - 1)] if sub.level > 1 else anchor
                    target = ".".join(anchor + ([sub.module] if sub.module else []))
                else:
                    target = sub.module or ""
                for alias in sub.names:
                    hit = resolve(f"{target}.{alias.name}") or resolve(target)
                    if hit:
                        deps.add(hit)
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str) and mod not in catalogues:
                head = sub.value.partition(":")[0]
                if head.startswith("quantbox.") and head in known:
                    if sources[head][1]:  # a PACKAGE named as a string is a prefix for names a config gives
                        graph.soft.setdefault(mod, set()).add(head)
                    else:
                        deps.add(head)
        deps.discard(mod)
        graph.edges[mod] = deps
    return graph


# ------------------------------------------------------------------ mapping


@dataclass
class Mapped:
    path: str
    step: str | None  # a step name, None (ignored), or ALL (unmapped)
    why: str
    module: str | None = None


def map_files(
    files: list[str], plan: dict[str, Any], graphs: list[Graph], layers: dict[str, list[str]] | None
) -> list[Mapped]:
    steps, package = plan["steps"], plan["import_graph"].get("package", "src/quantbox")
    rank = {s: i for i, s in enumerate(steps)}
    out: list[Mapped] = []
    importers = [g.importers() for g in graphs]

    def glob_step(path: str) -> tuple[str, str] | None:
        for s in steps:
            hit = first_match(path, plan["paths"][s])
            if hit:
                return s, f"glob {hit}"
        return None

    rel_by_mod = {m: r for g in graphs for r, m in g.files.items()}

    def module_step(mod: str, rel: str | None = None) -> tuple[str, str] | None:
        """A module's own step: the trade layer, else the glob its file matches."""
        if layers is not None and "trade" in rank and "trade" in layers_of(mod, layers):
            return "trade", "import-linter layer trade"
        rel = rel or rel_by_mod.get(mod)
        return glob_step(rel) if rel else None

    for path in files:
        mod = module_of(path, package)
        ignored = first_match(path, plan["ignore"])
        if ignored:
            out.append(Mapped(path, None, f"ignored ({ignored})", mod))
            continue
        direct = module_step(mod, path) if mod else glob_step(path)
        if direct:
            out.append(Mapped(path, direct[0], direct[1], mod))
            continue
        if mod:  # refine through the import graph: the earliest step of its importers
            best: tuple[int, str, str] | None = None
            for rev in importers:
                seen, todo = set(), list(rev.get(mod, ()))
                while todo:
                    imp = todo.pop()
                    if imp in seen:
                        continue
                    seen.add(imp)
                    hit = module_step(imp)
                    if hit:
                        if best is None or rank[hit[0]] < best[0]:
                            best = (rank[hit[0]], hit[0], imp)
                        continue  # a stepped importer bounds this branch
                    todo.extend(rev.get(imp, ()))
            if best:
                out.append(Mapped(path, best[1], f"import graph: imported by {best[2]}", mod))
                continue
        out.append(Mapped(path, ALL, "no step: unmapped path, runs everything", mod))
    return out


# ------------------------------------------------------------------ selection


def plugin_names(cfg: Any) -> set[str]:
    """Every string in a config's plugins block (names are among them)."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            found.add(node)

    walk((cfg or {}).get("plugins") or {})
    return found


def config_roots(
    cfg: dict[str, Any], catalogue: dict[str, dict[str, str]] | None, mode_all: set[str], graph: Graph
) -> set[str] | None:
    """The modules a config starts from: the runner, its plugins' modules, and any module it
    names inside a package the code loads by name (``pipeline.params.strategies``).
    None = depends on everything."""
    if catalogue is None or (cfg.get("plugins") or {}).get("profile"):
        return None  # no catalogue to read, or a profile pulls plugins from the manifest
    names = plugin_names(cfg)
    roots = set(mode_all)
    for table in catalogue.values():
        for name, target in table.items():
            if name in names:
                roots.add(target.partition(":")[0])
    for pkg in {p for pkgs in graph.soft.values() for p in pkgs}:
        roots.update(f"{pkg}.{n}" for n in names if f"{pkg}.{n}" in graph.edges)
    return roots


# ------------------------------------------------------------------ running


def git(*args: str, cwd: Path = ROOT) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    proc = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise CouldNotRun(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout


@dataclass
class Side:
    name: str
    tree: Path
    python: str
    env: dict[str, str]
    sha: str = ""


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def pinned_slice(tree: Path, dest: Path) -> tuple[Path | None, list[str]]:
    """The builds *tree*'s datasets.lock pins, extracted under *dest*: (root, notes), or (None, why not).

    The bytes come from the quantbox-datasets clone that $QUANTBOX_DATASETS_ROOT names:
    its working copy when the pin file hashes to the pin, else the newest commit of its
    history whose pin file does. Each side gets the slice ITS lock pins. quantbox's own
    restore (quantbox_datasets.dataset) is not used: it imports the clone's build
    dependencies (yfinance), which a quantbox venv does not carry.
    """
    lock = tree / "datasets.lock"
    pins = yaml.safe_load(lock.read_text()) if lock.is_file() else None
    if not isinstance(pins, dict) or not pins:
        return None, ["no datasets.lock pins"]
    env_root = os.environ.get("QUANTBOX_DATASETS_ROOT", "")
    if not env_root or not Path(env_root).is_dir():
        return None, [f"no datasets root (QUANTBOX_DATASETS_ROOT={env_root or 'unset'})"]
    src = Path(env_root).resolve()
    clone = src.parent
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    notes = []
    for name, pin in sorted(pins.items()):
        pin = str(pin)
        here = src / name / "prices.parquet"
        if here.is_file() and _sha256_bytes(here.read_bytes()) == pin:
            shutil.copytree(src / name, dest / name)
            notes.append(f"{name} @ {pin[:12]}: working copy of {src / name}")
            continue
        rel = (src / name / "prices.parquet").relative_to(clone).as_posix()
        try:
            commits = git("log", "--format=%H", "-n", "60", "--", rel, cwd=clone).split()
        except CouldNotRun as exc:
            notes.append(f"{name} @ {pin[:12]}: NOT AVAILABLE ({exc})")
            continue
        found = None
        for commit in commits:
            blob = subprocess.run(["git", "-C", str(clone), "show", f"{commit}:{rel}"], capture_output=True)
            if blob.returncode == 0 and _sha256_bytes(blob.stdout) == pin:
                found = commit
                break
        if found is None:
            notes.append(f"{name} @ {pin[:12]}: NOT AVAILABLE (no commit of {clone} has that build)")
            continue
        archive = subprocess.run(
            ["git", "-C", str(clone), "archive", found, (src / name).relative_to(clone).as_posix()], capture_output=True
        )
        if archive.returncode != 0:
            notes.append(f"{name} @ {pin[:12]}: NOT AVAILABLE (git archive failed)")
            continue
        import io
        import tarfile

        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(dest / "_x", filter="data")
        shutil.move(str(dest / "_x" / (src / name).relative_to(clone)), str(dest / name))
        shutil.rmtree(dest / "_x", ignore_errors=True)
        notes.append(f"{name} @ {pin[:12]}: {clone.name} commit {found[:12]}")
    reader = datasets_reader()
    if reader is None:
        notes.append(f"reader: NOT AVAILABLE (no {clone}/src/quantbox_datasets/lock.py)")
    else:
        notes.append(f"reader: {reader} at {clone.name} {git('rev-parse', '--short=12', 'HEAD', cwd=clone).strip()}")
    return dest, notes


def side_env(tree: Path, datasets_root: Path | None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["PYTHONPATH"] = os.pathsep.join([str(tree / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.update({"PYTHONHASHSEED": "0", "TZ": "UTC", "MPLBACKEND": "Agg"})
    if datasets_root is not None:
        env["QUANTBOX_DATASETS_ROOT"] = str(datasets_root)
        shim = reader_shim(datasets_root.parent)
        if shim is not None:
            env["PYTHONPATH"] = os.pathsep.join([env["PYTHONPATH"], str(shim)])
    else:
        env.pop("QUANTBOX_DATASETS_ROOT", None)
    return env


def datasets_reader() -> Path | None:
    """quantbox_datasets in the clone $QUANTBOX_DATASETS_ROOT names; quantbox reads a dataset through it."""
    root = os.environ.get("QUANTBOX_DATASETS_ROOT", "")
    pkg = Path(root).resolve().parent / "src" / "quantbox_datasets" if root else None
    return pkg if pkg is not None and (pkg / "lock.py").is_file() else None


def reader_shim(out: Path) -> Path | None:
    """A quantbox_datasets package of the clone's own files, with an EMPTY __init__.

    The clone's __init__ imports its BUILDERS (yfinance and more), which a quantbox venv
    does not carry. The reader modules quantbox calls (lock, dataset) need only pandas
    and yaml. Every other file is linked as is, so the reader code that runs is the
    clone's.
    """
    reader = datasets_reader()
    if reader is None:
        return None
    shim = out / "_datasets_reader"
    pkg = shim / "quantbox_datasets"
    if not pkg.is_dir():
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text(
            '"""Reader shim (scripts/before_after.py): the builders are not imported."""\n'
        )
        for entry in reader.iterdir():
            if entry.name not in ("__init__.py", "__pycache__"):
                (pkg / entry.name).symlink_to(entry)
    return shim


def make_base(base_sha: str, out: Path) -> Side:
    tree = out / "base-tree"
    if tree.exists():
        git("worktree", "remove", "--force", str(tree))
    git("worktree", "add", "--detach", "--quiet", str(tree), base_sha)
    python = sys.executable
    same_deps = all(
        git("rev-parse", f"{base_sha}:{f}").strip() == git("rev-parse", f"HEAD:{f}").strip()
        for f in ("pyproject.toml", "uv.lock")
    )
    if not same_deps:  # the base's own dependencies, in its own venv
        print("base: pyproject.toml or uv.lock differ from head; syncing a base venv")
        env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(tree / ".venv")}
        proc = subprocess.run(
            ["uv", "sync", "--frozen", "--extra", "full", "--directory", str(tree)],
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            raise CouldNotRun(f"base venv: uv sync failed: {proc.stderr[-800:]}")
        python = str(tree / ".venv" / "bin" / "python")
    return Side("base", tree, python, side_env(tree, None))


def catalogue_of(side: Side) -> dict[str, dict[str, str]] | None:
    proc = subprocess.run(
        [side.python, "-c", CATALOGUE], cwd=side.tree, capture_output=True, text=True, env=side.env, timeout=300
    )
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


@dataclass
class RunResult:
    ok: bool
    run_dir: Path | None = None
    error: str = ""
    seconds: float = 0.0


def run_config(side: Side, config: str, out: Path) -> RunResult:
    """One config at one side. Both sides write under the SAME artifacts root, so the config hash is the same."""
    art_root, result_file = out / "_run", out / "_run.json"
    shutil.rmtree(art_root, ignore_errors=True)
    result_file.unlink(missing_ok=True)
    dest = out / side.name / Path(config).stem
    shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    cfg_path = side.tree / config
    if not cfg_path.is_file():
        return RunResult(False, error=f"{config} does not exist at {side.name}")
    proc = subprocess.run(
        [side.python, "-c", DRIVER, str(cfg_path), str(art_root), str(result_file)],
        cwd=side.tree,
        capture_output=True,
        text=True,
        env=side.env,
        timeout=1800,
    )
    seconds = time.monotonic() - start
    (dest.parent / f"{dest.name}.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0 or not result_file.is_file():
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return RunResult(False, error=" | ".join(tail) or f"exit {proc.returncode}", seconds=seconds)
    result = json.loads(result_file.read_text())
    imported = Path(result["quantbox"]).resolve()
    if not imported.is_relative_to(side.tree.resolve()):
        raise CouldNotRun(
            f"{side.name} imported quantbox from {imported}, not from {side.tree}: the sides are not isolated"
        )
    shutil.move(str(art_root / result["run_id"]), str(dest))
    (dest / ".run_id").write_text(result["run_id"])
    return RunResult(True, run_dir=dest, seconds=seconds)


# ------------------------------------------------------------------ diffing


def masker(side: Side, run_id: str, art_root: Path):
    subs = [
        (str(art_root / run_id), "<RUN_DIR>"),
        (str(art_root), "<ART>"),
        (str(side.tree), "<ROOT>"),
        (run_id, "<RUN_ID>"),
        (side.sha, "<COMMIT>"),  # the checkout's commit, which a finding report prints
    ]
    if side.env.get("QUANTBOX_DATASETS_ROOT"):  # each side's own copy of the pinned slice
        subs.insert(0, (side.env["QUANTBOX_DATASETS_ROOT"], "<DATASETS>"))

    def mask(text: str) -> str:
        for old, new in subs:
            text = text.replace(old, new)
        run_days()  # a run that crossed midnight stamps two days
        for rx in TIMESTAMP_RES:
            text = rx.sub(lambda m: "<TS>" if m.group(1) in RUN_DAYS else m.group(0), text)
        return text

    return mask


def masked_json(node: Any, mask) -> Any:
    if isinstance(node, dict):
        return {k: masked_json(v, mask) for k, v in node.items() if k not in DROPPED_KEYS}
    if isinstance(node, list):
        return [masked_json(v, mask) for v in node]
    if isinstance(node, str):
        return mask(node)
    return node


def close(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
    return a == b


def diff_json(a: Any, b: Any, where: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b), key=str):
            sub = f"{where}.{key}" if where else str(key)
            if key not in a:
                out.append(f"{sub}: only at head = {json.dumps(b[key], default=str)[:200]}")
            elif key not in b:
                out.append(f"{sub}: only at base = {json.dumps(a[key], default=str)[:200]}")
            else:
                out.extend(diff_json(a[key], b[key], sub))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{where}: {len(a)} items at base, {len(b)} at head"]
        return [line for i, (x, y) in enumerate(zip(a, b, strict=True)) for line in diff_json(x, y, f"{where}[{i}]")]
    if close(a, b):
        return []
    return [f"{where}: base {json.dumps(a, default=str)[:200]} -> head {json.dumps(b, default=str)[:200]}"]


def diff_parquet(a: Path, b: Path) -> list[str]:
    import numpy as np
    import pandas as pd

    fa, fb = pd.read_parquet(a), pd.read_parquet(b)
    out = []
    if fa.shape != fb.shape:
        out.append(f"shape: base {fa.shape} -> head {fb.shape}")
    if list(fa.columns) != list(fb.columns):
        gone, new = (
            sorted(map(str, set(fa.columns) - set(fb.columns))),
            sorted(map(str, set(fb.columns) - set(fa.columns))),
        )
        out.append(f"columns: only at base {gone[:10]}, only at head {new[:10]}")
    if not fa.index.equals(fb.index):
        out.append(
            f"index: base {len(fa.index)} rows [{fa.index.min()} .. {fa.index.max()}] -> head {len(fb.index)} rows [{fb.index.min()} .. {fb.index.max()}]"
        )
    if out:
        return out
    for col in fa.columns:
        x, y = fa[col], fb[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(x):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            bad = ~np.isclose(xv, yv, rtol=RTOL, atol=ATOL, equal_nan=True)
            if bad.any():
                worst = np.nanmax(np.abs(xv[bad] - yv[bad])) if bad.any() else 0.0
                first = fa.index[bad.argmax()]
                out.append(f"{col}: {int(bad.sum())} of {len(xv)} values differ, max |d| {worst:.3g}, first at {first}")
        elif not x.astype(str).equals(y.astype(str)):
            n = int((x.astype(str) != y.astype(str)).sum())
            out.append(f"{col}: {n} of {len(x)} values differ")
    return out


def diff_runs(base: Side, head: Side, a: Path, b: Path, out: Path) -> tuple[dict[str, list[str]], int]:
    """(file -> its differences, how many files were compared). Zero compared is not identical."""
    mask_a = masker(base, (a / ".run_id").read_text(), out / "_run")
    mask_b = masker(head, (b / ".run_id").read_text(), out / "_run")
    names = sorted(({p.name for p in a.iterdir()} | {p.name for p in b.iterdir()}) - EXCLUDED_FILES - {".run_id"})
    found: dict[str, list[str]] = {}
    for name in names:
        fa, fb = a / name, b / name
        if not fa.exists() or not fb.exists():
            found[name] = [f"written only at {'head' if fb.exists() else 'base'}"]
            continue
        if name.endswith(".parquet"):
            lines = diff_parquet(fa, fb)
        elif name.endswith(".json"):
            lines = diff_json(
                masked_json(json.loads(fa.read_text()), mask_a), masked_json(json.loads(fb.read_text()), mask_b)
            )
        else:
            ta, tb = (
                mask_a(fa.read_text(errors="replace")).splitlines(),
                mask_b(fb.read_text(errors="replace")).splitlines(),
            )
            lines = [] if ta == tb else _text_diff(ta, tb)
        if lines:
            found[name] = lines
    return found, len(names)


def _text_diff(a: list[str], b: list[str]) -> list[str]:
    import difflib

    return [
        line for line in difflib.unified_diff(a, b, "base", "head", n=0, lineterm="") if line[:3] not in ("---", "+++")
    ]


# ------------------------------------------------------------------ report


def write_summary(out: Path, lines: list[str]) -> None:
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def mapping_lines(mapped: list[Mapped]) -> list[str]:
    width = max((len(m.path) for m in mapped), default=10)
    return [f"  {m.path.ljust(width)}  ->  {m.step or '-'}  ({m.why})" for m in mapped]


def main(argv: list[str]) -> int:
    plan_only = "--plan" in argv  # print the mapping and the selection, run nothing, exit 3
    argv = [a for a in argv if a != "--plan"]
    if len(argv) != 2 or argv[0] in ("-h", "--help"):
        print((__doc__ or "").split("\n\n", 1)[-1])  # past the bash preamble
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    base_ref, out = argv[0], Path(argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    run_days()
    # The plan beside THIS script, not the checkout's: hotl takes the script from the
    # base branch so a PR cannot rewrite how it is verified, and the plan goes with it.
    plan = load_plan(SCRIPT_TREE)
    head_sha = git("rev-parse", "HEAD").strip()
    base_sha = git("merge-base", base_ref, "HEAD").strip()
    files = [f for f in git("diff", "--name-only", base_sha, head_sha).splitlines() if f]
    print(f"before/after: base {base_ref} (merge base {base_sha[:12]}) vs head {head_sha[:12]}, {len(files)} file(s)")

    report = [
        "# before/after",
        "",
        f"- base: `{base_ref}`, merge base `{base_sha}`",
        f"- head: `{head_sha}`",
        f"- files in the diff: {len(files)}",
        "",
    ]
    if not files:
        print("no file differs: identical")
        write_summary(out, [*report, "No file differs between base and head: identical."])
        return 0

    ig = plan["import_graph"]
    package, catalogues = ig.get("package", "src/quantbox"), set(ig.get("catalogues") or [])
    head_graph = build_graph(ROOT, package, catalogues)
    try:
        layers = read_layers(ROOT / "pyproject.toml")
    except Exception as exc:  # noqa: BLE001 — the layer map is a refinement; without it trade comes from globs
        print(f"layer map not read ({exc}); trade maps by glob only")
        layers = None

    # A file deleted at head lives only in the base tree: map with both graphs.
    base_side: Side | None = None
    mapped_first = map_files(files, plan, [head_graph], layers)  # enough to see an all-ignored diff
    stepped = [m for m in mapped_first if m.step is not None]
    if not stepped:
        print("per-file mapping:")
        print("\n".join(mapping_lines(mapped_first)))
        print(f"every file is ignored: nothing a run reads changed; identical ({time.monotonic() - started:.1f}s)")
        write_summary(
            out,
            [
                *report,
                "## Mapping",
                "",
                "```",
                *mapping_lines(mapped_first),
                "```",
                "",
                "Every file is ignored (docs, tests, CI): identical, nothing run.",
            ],
        )
        return 0

    base_side = make_base(base_sha, out)
    base_side.sha = base_sha
    head_side = Side("head", ROOT, sys.executable, side_env(ROOT, None), head_sha)
    try:
        base_graph = build_graph(base_side.tree, package, catalogues)
        graphs = [head_graph, base_graph]
        mapped = map_files(files, plan, graphs, layers)  # with the base graph: a file deleted at head
        print("per-file mapping:")
        print("\n".join(mapping_lines(mapped)))
        report += ["## Mapping", "", "```", *mapping_lines(mapped), "```", ""]

        steps = plan["steps"]
        stepped = [m for m in mapped if m.step is not None]
        run_all = any(m.step == ALL for m in stepped)
        earliest = steps[0] if run_all else min((m.step for m in stepped), key=steps.index)
        # The backtest (data .. engine) and the live trade path are two branches after the
        # decision: the configs here run the first, the shadow run the second. So trade is
        # owed by a trade-step file, an unmapped path, or a touched module that trade code
        # calls (the code closure of the import-linter trade layer: decision, metrics, the
        # strategy runner ...), never merely by being an earlier step.
        trade_called: set[str] = set()
        if layers is not None:
            for graph in graphs:
                trade_mods = {m for m in graph.edges if "trade" in layers_of(m, layers)}
                trade_called |= {m.module for m in stepped if m.module} & graph.closure(trade_mods)
        trade_called -= {m.module for m in stepped if m.step == "trade"}
        trade_owed = run_all or bool(trade_called) or any(m.step == "trade" for m in stepped)
        if trade_called:
            note = f"trade code calls {', '.join(sorted(trade_called))}: the shadow run is owed too"
            print(note)
            report += [f"- {note}", ""]
        owed_steps = [s for s in steps[steps.index(earliest) :] if s != "trade" or trade_owed]
        print(
            f"earliest step: {earliest}{' (an unmapped path: everything runs)' if run_all else ''}; steps owed: {' -> '.join(owed_steps)}"
        )
        report += [
            f"- earliest step: **{earliest}**{' (an unmapped path: everything runs)' if run_all else ''}",
            f"- steps owed: {' -> '.join(owed_steps)}",
            "",
        ]

        touched_mods = {m.module for m in stepped if m.module}
        touched_paths = {m.path for m in stepped}
        catalogues_by_side = {"head": catalogue_of(head_side), "base": catalogue_of(base_side)}
        runner_roots = {"quantbox.runner", "quantbox.registry"}
        selected: list[tuple[str, str]] = []
        for config in plan["configs"]:
            why = None
            if run_all:
                why = "an unmapped path runs every config"
            elif config in touched_paths:
                why = "the config itself changed"
            else:
                for side, graph in (("head", head_graph), ("base", base_graph)):
                    path = (ROOT if side == "head" else base_side.tree) / config
                    if not path.is_file():
                        continue
                    text = path.read_text()
                    cfg = yaml.safe_load(text) or {}
                    data_hit = next(
                        (
                            p
                            for p in touched_paths
                            if not p.endswith(".py")
                            and (Path(p).name in text or (Path(p).name == "datasets.lock" and "dataset:" in text))
                        ),
                        None,
                    )
                    if data_hit:
                        why = f"it reads {data_hit}"
                        break
                    roots = config_roots(cfg, catalogues_by_side[side], runner_roots, graph)
                    if roots is None:
                        why = f"its plugins could not be read at {side}: runs to be safe"
                        break
                    hit = sorted(touched_mods & graph.closure(roots))
                    if hit:
                        why = f"its code closure at {side} holds {', '.join(hit[:3])}{' ...' if len(hit) > 3 else ''}"
                        break
            if why:
                selected.append((config, why))
        skipped = [c for c in plan["configs"] if c not in dict(selected)]
        print(f"configs selected: {len(selected)} of {len(plan['configs'])}")
        for config, why in selected:
            print(f"  run   {config}  ({why})")
        for config in skipped:
            print(f"  skip  {config}  (no touched module in its code closure)")
        report += ["## Configs", ""]
        report += [f"- run `{c}`: {w}" for c, w in selected]
        report += [f"- skip `{c}`: no touched module in its code closure" for c in skipped]
        report.append("")
        if plan_only:  # never 0: a plan is not a comparison
            print("--plan: nothing run (exit 3)")
            write_summary(out, [*report, "**PLAN ONLY**: nothing was run (exit 3)."])
            return 3

        # The pinned slice, per side, only when a selected config reads a dataset by name.
        if any("dataset:" in p.read_text() for c, _ in selected for p in (ROOT / c, base_side.tree / c) if p.is_file()):
            report += ["## Pinned slice (datasets.lock)", ""]
            for side in (base_side, head_side):
                root, notes = pinned_slice(side.tree, out / f"{side.name}-datasets")
                side.env = side_env(side.tree, root)
                for note in notes:
                    print(f"slice {side.name}: {note}")
                    report.append(f"- {side.name}: {note}")
            report.append("")

        differs, could_not = [], []
        results: list[str] = []
        for config, _why in selected:
            rb = run_config(base_side, config, out)
            rh = run_config(head_side, config, out)
            timing = f"base {rb.seconds:.1f}s, head {rh.seconds:.1f}s"
            if not rb.ok and not rh.ok:
                could_not.append(config)
                line = f"- `{config}`: COULD NOT RUN on either side ({timing}). base: {rb.error[:300]} / head: {rh.error[:300]}"
            elif rb.ok != rh.ok:
                differs.append(config)
                failed, err = ("base", rb.error) if not rb.ok else ("head", rh.error)
                line = f"- `{config}`: DIFFERS: fails at {failed} only ({timing}): {err[:400]}"
            else:
                found, compared = diff_runs(base_side, head_side, rb.run_dir, rh.run_dir, out)
                if not compared:
                    could_not.append(config)
                    line = f"- `{config}`: COULD NOT COMPARE: the run wrote no file ({timing})"
                elif not found:
                    line = f"- `{config}`: identical, {compared} files compared ({timing})"
                else:
                    differs.append(config)
                    detail = out / "diffs" / f"{Path(config).stem}.md"
                    detail.parent.mkdir(exist_ok=True)
                    body = [f"# {config}", ""]
                    for name, lines in found.items():
                        body += [f"## {name}", "", *[f"- {x}" for x in lines[:MAX_LINES]]]
                        if len(lines) > MAX_LINES:
                            body.append(f"- ... {len(lines) - MAX_LINES} more")
                        body.append("")
                    detail.write_text("\n".join(body))
                    files_list = ", ".join(f"{n} ({len(v)})" for n, v in found.items())
                    line = f"- `{config}`: DIFFERS in {files_list} ({timing}); detail: diffs/{detail.name}"
                    for name, lines in list(found.items())[:4]:
                        line += f"\n  - {name}: " + "; ".join(lines[:3])
            print(line)
            results.append(line)
        report += ["## Results", "", *(results or ["- no config selected: none reaches a touched module"]), ""]

        if trade_owed:
            sr = plan["shadow_run"]
            owed = (
                f"{'An unmapped path runs every step' if run_all else 'The trade path is touched'}, "
                f"so the live shadow run is OWED: {sr.get('rule', '')}. "
                f"Run `{sr.get('command', '').replace('<base sha>', base_sha[:12]).replace('<head sha>', head_sha[:12])}` "
                f"in {sr.get('repo', 'quantbox-live')} on {sr.get('where', 'forge')}. "
                "quantbox is public and has no forge runner, so this script cannot run it."
            )
            print(f"TRADE: {owed}")
            report += ["## Trade step", "", owed, ""]

        elapsed = time.monotonic() - started
        if could_not:
            verdict, code = f"COULD NOT RUN: {len(could_not)} config(s) failed on both sides", 2
        elif differs or trade_owed:
            parts = [f"{len(differs)} config(s) differ"] if differs else []
            parts += ["the live shadow run is owed"] if trade_owed else []
            verdict, code = "DIFFERS: " + "; ".join(parts), 1
        else:
            verdict, code = f"IDENTICAL: {len(selected)} config(s) ran at base and head", 0
        report += [f"**{verdict}** (exit {code}, {elapsed:.1f}s)"]
        write_summary(out, report)
        print(f"{verdict} (exit {code}, {elapsed:.1f}s); report: {out / 'summary.md'}")
        return code
    finally:
        shutil.rmtree(out / "_run", ignore_errors=True)
        (out / "_run.json").unlink(missing_ok=True)
        if base_side is not None and base_side.tree.exists():
            subprocess.run(
                ["git", "-C", str(ROOT), "worktree", "remove", "--force", str(base_side.tree)], capture_output=True
            )


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except CouldNotRun as exc:
        print(f"COULD NOT RUN: {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception:  # noqa: BLE001 — Python exits an uncaught error with 1, which reads as "differs"
        import traceback

        traceback.print_exc()
        print("COULD NOT RUN: the compare itself crashed (traceback above)", file=sys.stderr)
        sys.exit(2)
