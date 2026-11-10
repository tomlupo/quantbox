#!/usr/bin/env python3
""":"
# hotl verify runs this file as `bash <script> <base ref> <out dir>`: these four lines are bash,
# and the start of the docstring to Python. bash hands the file to the project's Python here and
# never reads further.
exec uv run --no-sync python "$0" "$@"

The before/after of a quantbox PR: the result at base and at head, diffed (TOM-2458).

    uv run python scripts/before_after.py [--plan] <base ref> <out dir>

--plan prints the per-file mapping and the configs it would run, runs nothing, and
exits 4 (a plan is not a comparison).

hotl verify's contract (tomlupo/hotl README, "Adopt in a repo" step 6): run from the
HEAD checkout; exit 0 = base and head identical, 1 = they differ (or a step is owed
that this script cannot run) and <out dir>/summary.md says what, anything else = the
compare could not run. Decision: vaults/projects/infra/decisions/
decyzja-2026-10-26-stos-kiss-repo-hotl-fabryka.md, "Lekki compare".

The contract holds for the configs this runner CAN run. Two codes are added:

    0  identical: every selected config ran and is identical
    1  differs: at least one config differs (the shadow run may be owed too)
    2  could not run: a config failed on both sides, or no selected config can run here
    3  PARTIAL: every config that can run here is identical, but at least one could not
       run here because a dataset it reads by name does not resolve on this runner (no
       line copy and no datasets root: public CI has no quantbox-datasets clone).
       summary.md lists them under "Not checked on this runner"; <out dir>/not_checked.txt
       names them on one line. Never 0: a config not looked at is not identical.
    4  --plan: nothing was run
    5  SHADOW OWED: every config that ran is identical (some may be not checked, as in
       3), and the live shadow run is owed. hotl's contract says 1 here; quantbox
       splits it because it is public and the shadow run happens only in quantbox-live
       at the pin bump (Tom, 2026-10-27). A private repo running hotl verify maps
       "shadow owed" through its own runner.

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
   An unmapped path selects every config. The closure is every module the run LOADS,
   so a package __init__ brings in all its siblings: a change to one strategy selects
   every config that loads the strategies package. A name taken from a lazy-export
   package (a `__getattr__` over a `_LAZY` map) loads the module the map names; an
   import of the package itself (or of a module below it, which binds the package
   too) loads every module the map names; a map not read in full (bound twice,
   extended, conditional) loads every module below the package. A
   rename counts as its delete and its add.
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
   every run (the run id, which carries the run's one wall-clock stamp; the checkout and
   slice paths; the commit; git and installed packages; the engine version, which is a
   package version too: run_manifest.json engine.version and, in finding_report.json's
   "Run provenance" table, the "engine" row's "<name> <version>" label, each side's own,
   TOM-2566; the engine name and every other row and number still compare). No other
   date is masked.
   TYPE comes before value (TOM-2547), and no value is compared through a display string.
   Parquet: the arrow tables read from the files, never the pandas frames: the stored
   schema, the pandas metadata (index kind, column-index dtype; not the writer's
   versions), the row count, then each column; a column arrow does not find equal is
   walked row by row and into its lists and structs, and the first differing path
   (column, row, nested index) is reported. A value is decoded only where that is exact: a
   temporal leaf as its integer storage, a dictionary column as its dictionary and its
   indices; a column of any other type is refused (exit 2), never called identical. JSON: a value's type (null, boolean, number,
   string, array, object; true is not 1, and 1 vs 1.0 is one type, number). Floats,
   nested ones too, compare at rel 1e-9, abs 1e-12; every other value compares exactly.
   A text file that is not UTF-8 compares as bytes.
6. TRADE. A touched trade-step file, or a touched module that trade code loads (the
   code closure of the import-linter trade layer), owes the live shadow run (quantbox-live
   scripts/shadow_run.py). quantbox is public and has no forge runner, so this script
   cannot run it: summary.md names it and the exit is 5 (1 if a config also differs).

A config that fails on one side only is a difference. A config that fails on both
could not be compared: the exit is 2, never 0. Before any run, a selected config whose
by-name dataset does not resolve here (quantbox.dataset_lock: no datasets root and no
line copy beside its lock) is NOT AVAILABLE ON THIS RUNNER and is not run: exit 3 when
the rest are identical, 2 when nothing else could run.
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

# No timestamp is masked by its shape. The one wall-clock stamp a run writes is inside its
# run id (`<asof>__<pipeline>__<hash>__<YYYYmmddTHHMMSSZ>`), which is masked as a whole;
# any other date in a run's files is a result and must still differ (PR #320 review).
SINGLE_NAME = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")  # one directory entry, never "..", "/" or absolute


def single_name(value: str, what: str) -> str:
    """*value* if it names one entry in a directory; a name from a lock or a run that would leave it is refused."""
    if not SINGLE_NAME.match(value) or value in (".", ".."):
        raise CouldNotRun(f"{what} {value!r} is not a single file name; refused rather than joined into a path")
    return value


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
        """Every module a run of *start* LOADS: its imports, each module's package, and theirs.

        A package's __init__ runs when any module inside it is imported, and its own
        imports run with it, siblings included. Their import-time code executes, and a
        function it calls can change it from anywhere, so a loaded module counts like a
        called one. No narrower rule is attempted: two rounds of PR #320's review broke
        two of them (findings 2 and 102, then round 2 finding 1).
        """
        seen: set[str] = set()
        todo = [m for m in start if m in self.edges]
        while todo:
            mod = todo.pop()
            if mod in seen:
                continue
            seen.add(mod)
            todo.extend(self.edges.get(mod, ()))
            if mod in self.parents:
                todo.append(self.parents[mod])
            # A package named as a string (`f"{pkg}.{name}"`) loads the module the CONFIG
            # names; that module is a root of its own (config_roots). The package itself
            # loads first, with its own imports.
            todo.extend(self.soft.get(mod, ()))
        return seen

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


def _named(target: ast.expr, name: str) -> bool:
    return isinstance(target, ast.Name) and target.id == name


def lazy_exports(node: ast.AST, resolve) -> tuple[dict[str, set[str]], bool] | None:
    """A module's lazy exports: None when it has no module-level ``__getattr__`` that can load.

    Else (exported name -> the modules taking it loads, read in full?). quantbox's lazy
    packages (plugins.datasources, .pipeline, .broker, .publisher) keep a dict, ``_LAZY``,
    of ``name -> "module:Attr"`` (or a tuple starting with it, or an f-string over a
    module-level constant) that ``__getattr__`` hands to ``quantbox._lazy.load``. Anything
    else is not read in full, and the caller widens it to every module below. Not lazy: a
    ``__getattr__`` whose only calls build the exception it raises, and ``removed_names``
    (quantbox._removed), which refuses names and loads nothing.
    """
    body = getattr(node, "body", [])
    # Every binding of __getattr__ anywhere in the module (under if/try too), and every name
    # bound more than once: a map or constant bound twice is not read in full.
    defs = [n for n in ast.walk(node) if isinstance(n, ast.FunctionDef) and n.name == "__getattr__"]
    stores: dict[str, int] = {}
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store | ast.Del):
            stores[n.id] = stores.get(n.id, 0) + 1
    assigned = [s.value for s in body if isinstance(s, ast.Assign) and any(_named(t, "__getattr__") for t in s.targets)]
    if not defs and stores.get("__getattr__", 0) == 0:
        return None
    if not defs:
        top = assigned[0] if len(assigned) == 1 and stores["__getattr__"] == 1 else None
        call = top.func if isinstance(top, ast.Call) else None
        if isinstance(call, ast.Name) and call.id == "removed_names":
            return None  # quantbox._removed: refuses names, loads nothing
        return {}, False  # a __getattr__ this reader does not know: anything below may load
    func = defs[0]
    if len(defs) != 1 or func not in body or stores.get("__getattr__", 0):
        return {}, False  # defined twice, conditionally, or rebound: not read in full
    in_raise = {id(n) for r in ast.walk(func) if isinstance(r, ast.Raise) for n in ast.walk(r)}
    if not any(isinstance(n, ast.Call) and id(n) not in in_raise for n in ast.walk(func)):
        return None  # it only raises: no name loads a module
    consts: dict[str, str] = {}
    dicts: dict[str, ast.Dict] = {}
    for stmt in body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            if stores.get(name) != 1:
                continue
            if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                consts[name] = stmt.value.value
            elif isinstance(stmt.value, ast.Dict):
                dicts[name] = stmt.value
    # The one map __getattr__ reads; any other module-level binding it reads (a second map, a
    # comprehension, a name bound twice) may name more modules, so the map is not trusted.
    params = {a.arg for a in ast.walk(func.args) if isinstance(a, ast.arg)}
    local = {n.id for n in ast.walk(func) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    read = {n.id for n in ast.walk(func) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    module_bound = {n for n, count in stores.items() if count} - params - local
    maps = [dicts[n] for n in read if n in dicts]
    if len(maps) != 1 or (read & module_bound) - set(dicts):
        return {}, False
    map_name = next(n for n in read if n in dicts)
    # The map may be named nowhere else: `_LAZY.update(...)` or `_LAZY[k] = v` adds names.
    uses = sum(1 for n in ast.walk(node) if isinstance(n, ast.Name) and n.id == map_name)
    if uses != 1 + sum(1 for n in ast.walk(func) if isinstance(n, ast.Name) and n.id == map_name):
        return {}, False

    def text(value: ast.expr) -> str | None:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        if isinstance(value, ast.Tuple) and value.elts:
            return text(value.elts[0])
        if isinstance(value, ast.JoinedStr):
            parts = []
            for part in value.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    parts.append(part.value)
                elif (
                    isinstance(part, ast.FormattedValue)
                    and isinstance(part.value, ast.Name)
                    and part.value.id in consts
                    and part.conversion == -1
                    and part.format_spec is None
                ):
                    parts.append(consts[part.value.id])
                else:
                    return None
            return "".join(parts)
        return None

    out: dict[str, set[str]] = {}
    complete = True
    for key, value in zip(maps[0].keys, maps[0].values, strict=True):
        name = key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else None
        target = (text(value) or "").partition(":")[0]
        if name is None or resolve(target) != target:  # not a quantbox module this tree holds, exactly
            complete = False
            continue
        out[name] = {target}
    return out, complete


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

    # A module with a lazy-export __getattr__ loads a module only when a NAME is taken
    # from it. lazy[mod] maps each exported name to the modules it loads; lazy_all[mod]
    # is what an unknown name (or `import mod`, `*`) may load (PR #320 round 4, codex P2).
    lazy: dict[str, dict[str, set[str]]] = {}
    lazy_all: dict[str, set[str]] = {}
    for mod, (node, _is_pkg) in sources.items():
        found = lazy_exports(node, resolve)
        if found is None:
            continue
        lazy[mod], complete = found
        # A map read in full loads at most its targets; any doubt widens to every module below it.
        below = {m for m in known if m.startswith(mod + ".")}
        lazy_all[mod] = set().union(*lazy[mod].values()) if complete else below | set().union(*lazy[mod].values())
        if not complete:
            lazy[mod] = {}  # no name is trusted from a map not read in full

    def taken(pkg: str, name: str | None) -> set[str]:
        """What taking *name* from lazy-export *pkg* loads (None, "*": any name). Unknown is everything."""
        if name is None or name == "*" or not lazy[pkg]:
            return lazy_all[pkg]
        return lazy[pkg].get(name, set())  # a name the full map lacks is not lazy: nothing more loads

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
                    # `import pkg` binds pkg; `import pkg.sub` binds the top name, so
                    # `pkg.Name` is reachable too: any name of each lazy package on the path.
                    for pkg in lazy:
                        if alias.name == pkg or (alias.asname is None and alias.name.startswith(pkg + ".")):
                            deps |= taken(pkg, None)
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
                        if hit in lazy and hit == target:  # a NAME taken from a lazy-export module
                            deps |= taken(hit, alias.name)
                        elif hit in lazy and hit == f"{target}.{alias.name}":  # the lazy module itself
                            deps |= taken(hit, None)
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


# ------------------------------------------------------------- availability


def dataset_available(name: str, lock: Path | None) -> bool:
    """Does by-name *name* resolve on this runner? quantbox's own resolver (quantbox.dataset_lock) decides.

    False only when there is no datasets root ($QUANTBOX_DATASETS_ROOT, or an installed
    quantbox-datasets clone) AND no copy of *name* in the line beside *lock*. With a root,
    a name it lacks is the run's to report: that config fails on both sides, exit 2.
    """
    from quantbox.dataset_lock import DatasetResolveError, _locate, datasets_root

    try:
        datasets_root()
        return True
    except DatasetResolveError:
        pass
    try:
        _locate(name, lock)  # with no root, only the line's own copy can answer
        return True
    except DatasetResolveError:
        return False


def unavailable_datasets(config: str, trees: tuple[Path, ...]) -> list[str]:
    """The by-name datasets *config* reads, at any of *trees*, that do not resolve on this runner."""
    from quantbox.dataset_lock import lock_for_config

    missing: list[str] = []
    for tree in trees:
        path = tree / config
        if not path.is_file():
            continue
        cfg = yaml.safe_load(path.read_text()) or {}
        params_init = ((cfg.get("plugins") or {}).get("data") or {}).get("params_init") or {}
        name = params_init.get("dataset")
        if not isinstance(name, str) or not name or name in missing:
            continue
        explicit = params_init.get("dataset_lock")  # as the runner binds it: the config's own, else the nearest
        lock = (tree / explicit) if isinstance(explicit, str) else lock_for_config(path)
        if not dataset_available(name, lock):
            missing.append(name)
    return missing


def not_checked_lines(not_available: dict[str, list[str]]) -> list[str]:
    return [
        "## Not checked on this runner",
        "",
        "A dataset these configs read by name does not resolve here: no datasets root",
        "($QUANTBOX_DATASETS_ROOT) and no line copy. They were not run at either side.",
        "",
        *[f"- `{c}`: needs {', '.join(f'`{n}`' for n in names)}" for c, names in not_available.items()],
        "",
    ]


def verdict_of(
    n_selected: int,
    n_runnable: int,
    not_available: dict[str, list[str]],
    could_not: list[str],
    differs: list[str],
    trade_owed: bool,
) -> tuple[str, int]:
    """(verdict, exit code). 0, 1 and 2 keep hotl's meaning for the runnable configs; 3 is PARTIAL,
    5 is identical with the live shadow run owed (quantbox's public-CI split of hotl's 1)."""
    if n_selected and not n_runnable:  # nothing was looked at: never 3, never 0
        return f"COULD NOT RUN: none of {n_selected} selected config(s) can run on this runner", 2
    if could_not:
        return f"COULD NOT RUN: {len(could_not)} config(s) failed on both sides", 2
    if differs:
        owed = "; the live shadow run is owed" if trade_owed else ""
        return f"DIFFERS: {len(differs)} config(s) differ{owed}", 1
    if trade_owed:
        unchecked = f"; {len(not_available)} not checked on this runner" if not_available else ""
        return (
            f"SHADOW OWED: {n_runnable} config(s) ran at base and head and are identical{unchecked}; "
            "the live shadow run is owed",
            5,
        )
    if not_available:
        return (
            f"PARTIAL: {n_runnable} config(s) ran at base and head and are identical; "
            f"{len(not_available)} not checked on this runner",
            3,
        )
    return f"IDENTICAL: {n_runnable} config(s) ran at base and head", 0


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
        # A lock is PR content: a name like "../private" would copy from, and write to,
        # outside the slice (PR #320 review, finding 103).
        name, pin = single_name(str(name), "datasets.lock entry"), str(pin)
        if not (src / name).resolve().is_relative_to(src):
            raise CouldNotRun(f"dataset {name!r} resolves outside {src}; refused")
        # copytree follows links: a link anywhere under the dataset would copy, and the run
        # would read, content from outside it (PR #320 review round 2, finding 2).
        links = [p for p in [src / name, *(src / name).rglob("*")] if p.is_symlink()]
        if links:
            raise CouldNotRun(f"dataset {name!r} holds a symlink ({links[0]}); refused rather than followed")
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
    run_id = single_name(str(result["run_id"]), f"{side.name} run id")  # written by the side's own code
    shutil.move(str(art_root / run_id), str(dest))
    (dest / ".run_id").write_text(run_id)
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


ENGINE_VERSION = "<ENGINE_VERSION>"


def engine_of(run: Path) -> tuple[Any, Any]:
    """(name, version) of the engine a run recorded in its run_manifest.json; (None, None) if none."""
    try:
        engine = json.loads((run / "run_manifest.json").read_text()).get("engine")
    except (OSError, ValueError, AttributeError):
        return None, None
    return (engine.get("name"), engine.get("version")) if isinstance(engine, dict) else (None, None)


def mask_engine_version(name: str, node: Any, engine: tuple[Any, Any]) -> Any:
    """*node* with the engine version a run records masked, and nothing else (TOM-2566).

    Only run_manifest.json's engine.version and the "engine" row of finding_report.json's
    "Run provenance" table, where a cell is exactly that side's "<name> <version>" label.
    """
    engine_name, version = engine
    if version is None or not isinstance(node, dict):
        return node
    if name == "run_manifest.json" and isinstance(node.get("engine"), dict) and "version" in node["engine"]:
        node["engine"]["version"] = ENGINE_VERSION
    elif name == "finding_report.json":
        label = " ".join(str(x) for x in (engine_name, version) if x)
        masked = " ".join(str(x) for x in (engine_name, ENGINE_VERSION) if x)
        for table in node.get("tables") or []:
            if not isinstance(table, dict) or table.get("title") != "Run provenance":
                continue
            for row in table.get("rows") or []:
                if isinstance(row, list) and row and row[0] == "engine":
                    row[1:] = [masked if cell == label else cell for cell in row[1:]]
    return node


def json_type(value: Any) -> str:
    """The JSON type of a decoded value. bool is tested before number: bool is an int in Python."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__  # not from json.loads; its own type, never equal to a JSON one


def close(a: Any, b: Any) -> bool:
    """Equal in JSON TYPE, then in value (TOM-2547). Python's == says True == 1 and 0 == False.

    JSON types: null, boolean, number, string (lists and dicts are walked by diff_json).
    int and float are ONE type, number: RFC 8259 has one number type, and JSON Schema's
    "integer" accepts 1.0. The compare never told 1 from 1.0, and still does not. Two ints
    compare exactly: the float tolerance is for rounding, and a count has none.
    """
    if json_type(a) != json_type(b):
        return False
    if json_type(a) == "number":
        if isinstance(a, int) and isinstance(b, int):
            return a == b
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


def _kind(value: Any) -> str:
    """The kind of a value decoded from arrow (``to_pylist``). bool is tested before int: bool is an int in Python."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, dict):
        return "struct"
    if isinstance(value, list | tuple):
        return "list"
    return "other"  # str, bytes, Decimal, date and time: compared exactly, by ==


def first_difference(a: Any, b: Any, where: str) -> tuple[str, Any, Any] | None:
    """(path, base value, head value) where two decoded arrow values first differ, else None.

    Structural, never through a display string (TOM-2547, #322 round 2): lists and structs
    are walked element by element, so a change deep in a 1001-element list is found. The
    kind comes first: null is not a value, bool is not int, int is not float. A float leaf,
    nested or not, compares within RTOL/ATOL (NaN equals NaN: a missing value in both);
    every other leaf compares exactly.
    """
    if _kind(a) != _kind(b):
        return where, a, b
    kind = _kind(a)
    if kind == "float":
        if (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL):
            return None
        return where, a, b
    if kind == "struct":
        if list(a) != list(b):
            return where, a, b
        return next((hit for k in a if (hit := first_difference(a[k], b[k], f"{where}.{k}"))), None)
    if kind == "list":
        if len(a) != len(b):
            return f"{where} (length {len(a)} -> {len(b)})", a, b
        return next(
            (
                hit
                for i, (x, y) in enumerate(zip(a, b, strict=True))
                if (hit := first_difference(x, y, f"{where}[{i}]"))
            ),
            None,
        )
    return None if a == b else (where, a, b)


PANDAS_WRITER_KEYS = {"creator", "pandas_version"}  # which library wrote the file, not what it holds


def _pandas_meta(schema) -> Any:
    """The pandas metadata a parquet file carries, decoded: the index kind (a RangeIndex lives
    only here), the column-index dtype, each column's pandas type. The writer's versions are dropped."""
    raw = (schema.metadata or {}).get(b"pandas")
    if raw is None:
        return None
    return {k: v for k, v in json.loads(raw).items() if k not in PANDAS_WRITER_KEYS}


def _lossless_type(dtype):
    """The type a column of *dtype* is decoded through, so that to_pylist keeps every stored bit;
    None where that is not established, and the column is refused, never passed (#326 rounds 1-2).

    A leaf whose Python value is exact (null, bool, integers, floats, strings, binaries,
    decimals) decodes as itself. A temporal leaf decodes as its integer storage: to_pylist turns
    time64[ns] into datetime.time and drops the nanoseconds. Its unit and zone are in the schema,
    which is already equal. Lists, structs and maps decode through their children. Anything else,
    a dictionary included (diff_parquet compares a top-level one by its parts), is refused.
    """
    import pyarrow as pa

    t = pa.types
    if t.is_timestamp(dtype) or t.is_date64(dtype) or t.is_time64(dtype) or t.is_duration(dtype):
        return pa.int64()
    if t.is_date32(dtype) or t.is_time32(dtype):
        return pa.int32()
    exact = (t.is_null, t.is_boolean, t.is_integer, t.is_floating, t.is_string, t.is_large_string)
    exact += (t.is_binary, t.is_large_binary, t.is_fixed_size_binary, t.is_decimal)
    if any(is_(dtype) for is_ in exact):
        return dtype
    if t.is_map(dtype):
        key, item = _lossless_type(dtype.key_type), _lossless_type(dtype.item_type)
        if key is None or item is None:
            return None
        return pa.map_(dtype.key_field.with_type(key), dtype.item_field.with_type(item))
    if t.is_list(dtype) or t.is_large_list(dtype) or t.is_fixed_size_list(dtype):
        child = _lossless_type(dtype.value_type)
        if child is None:
            return None
        field = dtype.value_field.with_type(child)
        if t.is_fixed_size_list(dtype):
            return pa.list_(field, dtype.list_size)
        return pa.large_list(field) if t.is_large_list(dtype) else pa.list_(field)
    if t.is_struct(dtype):
        fields = [dtype.field(i) for i in range(dtype.num_fields)]
        children = [_lossless_type(f.type) for f in fields]
        if any(c is None for c in children):
            return None
        return pa.struct([f.with_type(c) for f, c in zip(fields, children, strict=True)])
    return None


def _shown(value: Any) -> str:
    return repr(value)[:200]  # for the report only; no comparison reads it


def _label(column, row: int) -> str:
    """The index value at *row*, for the report only: arrow's own text, which keeps nanoseconds."""
    return " ".join(column.slice(row, 1).to_string().split()).strip("[] ")[:200]


def diff_parquet(a: Path, b: Path) -> list[str]:
    """The two arrow tables, compared by structure (TOM-2547): the stored schema first (names,
    types, nullability, index columns included), then the pandas metadata, the row count, and
    each column's values. pandas is not read: its display coerces, and a decimal and a string
    both read as object. A column that arrow finds equal is equal; any other is decoded through
    _lossless_type and walked row by row with first_difference, which keeps types and applies
    the float tolerance at the leaves. A type with no lossless decoding is refused."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ta, tb = pq.read_table(a), pq.read_table(b)
    sa, sb = ta.schema, tb.schema
    if not sa.equals(sb):
        fields_a = [(f.name, f.type, f.nullable) for f in sa]
        fields_b = [(f.name, f.type, f.nullable) for f in sb]
        changed = [f"{n}: {t}" for n, t, null in fields_a if (n, t, null) not in fields_b]
        new = [f"{n}: {t}" for n, t, null in fields_b if (n, t, null) not in fields_a]
        if not changed and not new:
            return [f"schema: same fields in another order, base {sa.names[:10]} -> head {sb.names[:10]}"]
        return [f"schema: base {changed[:10]} -> head {new[:10]}"]
    meta = diff_json(_pandas_meta(sa), _pandas_meta(sb), "pandas metadata")
    if meta:
        return meta
    if ta.num_rows != tb.num_rows:
        return [f"rows: base {ta.num_rows} -> head {tb.num_rows}"]
    index = [c for c in (_pandas_meta(sa) or {}).get("index_columns") or [] if isinstance(c, str) and c in sa.names]
    ta, tb = ta.unify_dictionaries(), tb.unify_dictionaries()  # one dictionary per column, over chunks
    out = []
    for name, ca, cb in zip(sa.names, ta.columns, tb.columns, strict=True):
        dictionary = pa.types.is_dictionary(ca.type)
        storage = _lossless_type(ca.type.value_type if dictionary else ca.type)
        if storage is None:
            raise CouldNotRun(f"{a.name}: column {name!r} of type {ca.type} has no lossless decoding; not compared")
        if dictionary:
            # A dictionary column (a pandas categorical) compares by its two parts: the dictionary,
            # whose order sorts an ordered categorical and which may hold unused entries, then the
            # indices. Decoding to values would drop the first (#326 round 1).
            da, db = ca.combine_chunks(), cb.combine_chunks()  # one dictionary each, after unify
            va, vb = da.dictionary.cast(storage).to_pylist(), db.dictionary.cast(storage).to_pylist()
            hit = first_difference(va, vb, f"{name}.dictionary")
            if hit:
                out.append(
                    f"{name}: dictionary (categories) differ at {hit[0]}: base {_shown(va)} -> head {_shown(vb)}"
                )
                continue
            xs, ys = da.indices.to_pylist(), db.indices.to_pylist()
        elif ca.equals(cb):  # exact, type-aware; a NaN is never equal here, so such a column walks
            continue
        else:
            xs, ys = ca.cast(storage).to_pylist(), cb.cast(storage).to_pylist()
        bad = [
            (i, hit)
            for i, (x, y) in enumerate(zip(xs, ys, strict=True))
            if (hit := first_difference(x, y, f"{name}[row {i}]"))
        ]
        if bad:
            row, (path, x, y) = bad[0]
            at = f" (index {_label(ta.column(index[0]), row)})" if len(index) == 1 else ""
            out.append(
                f"{name}: {len(bad)} of {ta.num_rows} rows differ, first at {path}{at}: base {_shown(x)} -> head {_shown(y)}"
            )
    return out


def diff_runs(base: Side, head: Side, a: Path, b: Path, out: Path) -> tuple[dict[str, list[str]], int]:
    """(file -> its differences, how many files were compared). Zero compared is not identical."""
    mask_a = masker(base, (a / ".run_id").read_text(), out / "_run")
    mask_b = masker(head, (b / ".run_id").read_text(), out / "_run")
    engine_a, engine_b = engine_of(a), engine_of(b)
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
                mask_engine_version(name, masked_json(json.loads(fa.read_text()), mask_a), engine_a),
                mask_engine_version(name, masked_json(json.loads(fb.read_text()), mask_b), engine_b),
            )
        else:
            lines = diff_text(fa.read_bytes(), fb.read_bytes(), mask_a, mask_b)
        if lines:
            found[name] = lines
    return found, len(names)


def diff_text(a: bytes, b: bytes, mask_a, mask_b) -> list[str]:
    """UTF-8 text, masked, line by line. A file that is not UTF-8 compares as bytes, exactly:
    decoding it with errors="replace" turned two different invalid bytes into one U+FFFD."""
    try:
        ta, tb = mask_a(a.decode("utf-8")).splitlines(), mask_b(b.decode("utf-8")).splitlines()
    except UnicodeDecodeError:
        if a == b:
            return []
        at = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
        return [f"bytes (not UTF-8 text): base {len(a)} -> head {len(b)} bytes, first differing at byte {at}"]
    return [] if ta == tb else _text_diff(ta, tb)


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
    plan_only = "--plan" in argv  # print the mapping and the selection, run nothing, exit 4
    argv = [a for a in argv if a != "--plan"]
    if len(argv) != 2 or argv[0] in ("-h", "--help"):
        print((__doc__ or "").split("\n\n", 1)[-1])  # past the bash preamble
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    base_ref, out = argv[0], Path(argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "not_checked.txt").unlink(missing_ok=True)  # a reused out dir must not carry an earlier run's list
    started = time.monotonic()
    # The plan beside THIS script, not the checkout's: hotl takes the script from the
    # base branch so a PR cannot rewrite how it is verified, and the plan goes with it.
    plan = load_plan(SCRIPT_TREE)
    head_sha = git("rev-parse", "HEAD").strip()
    base_sha = git("merge-base", base_ref, "HEAD").strip()
    # --no-renames: a rename is its delete AND its add, so the OLD path is mapped too (a
    # module moved into tests/ or renamed away still runs what used to import it).
    files = [f for f in git("diff", "--name-only", "--no-renames", base_sha, head_sha).splitlines() if f]
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
        # loads (the closure of the import-linter trade layer: decision, metrics, the
        # strategy runner ...), never merely by being an earlier step.
        touched_mods = {m.module for m in stepped if m.module}
        trade_called: set[str] = set()
        if layers is not None:
            for graph in graphs:
                trade_mods = {m for m in graph.edges if "trade" in layers_of(m, layers)}
                trade_called |= touched_mods & graph.closure(trade_mods)
        trade_called -= {m.module for m in stepped if m.step == "trade"}
        trade_owed = run_all or bool(trade_called) or any(m.step == "trade" for m in stepped)
        if trade_called:
            note = f"trade code loads {', '.join(sorted(trade_called))}: the shadow run is owed too"
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
                        why = f"at {side} its run loads {', '.join(hit[:3])}{' ...' if len(hit) > 3 else ''}"
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
        # A config whose by-name dataset does not resolve on this runner is not run: it
        # cannot be compared here, and running it would only fail on both sides.
        not_available = {
            c: missing for c, _ in selected if (missing := unavailable_datasets(c, (ROOT, base_side.tree)))
        }
        runnable = [(c, w) for c, w in selected if c not in not_available]
        if not_available:
            print(f"not checked on this runner: {len(not_available)} of {len(selected)} selected config(s)")
            for config, names in not_available.items():
                print(f"  not available  {config}  (needs {', '.join(names)})")
            report += not_checked_lines(not_available)
            (out / "not_checked.txt").write_text(
                ", ".join(f"{c} ({', '.join(n)})" for c, n in not_available.items()) + "\n"
            )
        if plan_only:  # never 0: a plan is not a comparison
            print("--plan: nothing run (exit 4)")
            write_summary(out, [*report, "**PLAN ONLY**: nothing was run (exit 4)."])
            return 4
        if selected and not runnable:
            verdict, code = verdict_of(len(selected), 0, not_available, [], [], trade_owed)
            report += [f"**{verdict}** (exit {code}, {time.monotonic() - started:.1f}s)"]
            write_summary(out, report)
            print(f"{verdict} (exit {code}); report: {out / 'summary.md'}")
            return code

        # The pinned slice, per side, only when a runnable config reads a dataset by name.
        if any("dataset:" in p.read_text() for c, _ in runnable for p in (ROOT / c, base_side.tree / c) if p.is_file()):
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
        for config, _why in runnable:
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
        verdict, code = verdict_of(len(selected), len(runnable), not_available, could_not, differs, trade_owed)
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
