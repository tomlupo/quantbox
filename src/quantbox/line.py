"""Research lines: scaffold one (``quantbox new line``) and move its engine pin (``quantbox line repin``).

A line is the Model C atom every lab shares (TOM-1366): its own ``pyproject.toml`` and
``uv.lock``, a README carrying the ``prereg:`` block, a ``datasets.lock``, an arms file
and a reproduction test. What makes a line survive dependency drift is the PIN:

- quantbox is a direct git reference to the 40-char SHA a tag names. The tag is
  recorded beside it (``[tool.quantbox-line]``), the SHA is what is installed: a tag
  can move, and the qute pin gate (``check_research_pins.py``) refuses a tag pin.
- every TRANSITIVE dependency is pinned exactly, in a managed
  ``constraint-dependencies`` block. Its versions come first from the uv.lock
  quantbox itself was tested with at that ref, then from the line's own resolution
  for whatever quantbox's lock does not settle. H08 is why: ``import vectorbt``
  crashed once a fresh resolve picked up pandas 3.

``repin`` is the one command that moves a line to another ref: it rewrites the
SHA and the tag, re-derives the managed block from the new ref and re-locks.

Everything here shells out to ``git`` and ``uv``; nothing imports either.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from importlib.resources import files as _res_files
from pathlib import Path
from typing import Any

QUANTBOX_URL = "https://github.com/tomlupo/quantbox.git"
DATASETS_URL = "https://github.com/tomlupo/quantbox-datasets.git"
DEFAULT_EXTRAS = "full"  # what quantbox-lab's .research-config.yaml pins
DEFAULT_PYTHON = "3.12"
DEFAULT_DATASET = "crypto-spot-daily"

#: A ref older than this file cannot run the skeleton: it predates arms files
#: (TOM-1363), by-name datasets (TOM-1349) and the run manifest (TOM-1348).
LINE_SUPPORT_FILE = "src/quantbox/line.py"

PINS_BEGIN = "# >>> quantbox line pins: rewritten by `quantbox line repin`, do not edit by hand"
PINS_END = "# <<< quantbox line pins"

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
_QB_DEP_RE = re.compile(r'"quantbox(\[[^\]]*\])? @ git\+(?P<url>[^"@]+)@(?P<sha>[^"#]+)",(?P<comment>[^\n]*)')
_REF_LINE_RE = re.compile(r'^quantbox-ref = "[^"]*"$', re.M)
_SHA_LINE_RE = re.compile(r'^quantbox-sha = "[^"]*"$', re.M)


class LineError(RuntimeError):
    """A line cannot be scaffolded or repinned; the message says why and what to pass."""


@dataclass(frozen=True)
class EngineRef:
    """A quantbox ref as a line pins it: the name asked for and the commit it named."""

    url: str
    ref: str
    sha: str


# ── git / uv ──────────────────────────────────────────────────────────────


def _run(cmd: list[str], cwd: str | Path | None = None, *, what: str) -> str:
    try:
        # GIT_* scrubbed: under a git hook they name ANOTHER repository, and `git init` /
        # `fetch` in the temp dir would act on it. Never prompt for credentials: a private
        # remote this box cannot read is an error, not a hang.
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env["GIT_TERMINAL_PROMPT"] = "0"
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise LineError(f"{what}: `{cmd[0]}` is not installed") from exc
    if proc.returncode != 0:
        raise LineError(f"{what} failed ({' '.join(cmd)}):\n{(proc.stderr or proc.stdout).strip()}")
    return proc.stdout


def latest_tag(url: str = QUANTBOX_URL) -> str:
    """The highest ``vX.Y.Z`` tag at *url*. Pre-release and non-semver tags are not candidates."""
    out = _run(["git", "ls-remote", "--tags", "--refs", url, "v*"], what=f"listing tags of {url}")
    tags = [line.rsplit("refs/tags/", 1)[-1] for line in out.splitlines() if "refs/tags/" in line]
    return pick_latest_tag(tags, url)


def pick_latest_tag(tags: list[str], url: str = QUANTBOX_URL) -> str:
    versions = [(tuple(int(p) for p in m.groups()), tag) for tag in tags if (m := _TAG_RE.match(tag))]
    if not versions:
        raise LineError(f"{url} has no vX.Y.Z tag; pass --quantbox-ref")
    return max(versions)[1]


def remote_head(url: str) -> str:
    """The commit *url*'s default branch (HEAD) names."""
    out = _run(["git", "ls-remote", url, "HEAD"], what=f"reading HEAD of {url}")
    sha = out.split()[0] if out.split() else ""
    if not _SHA40_RE.match(sha):
        raise LineError(f"{url} answered no HEAD commit; pass the ref explicitly")
    return sha


def fetch_ref(url: str, ref: str, dest: Path) -> str:
    """Fetch *ref* (tag, branch or SHA) of *url* into *dest* at depth 1; return its commit."""
    dest.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "-q"], dest, what="git init")
    _run(["git", "fetch", "-q", "--depth", "1", url, ref], dest, what=f"fetching {ref} of {url}")
    _run(["git", "checkout", "-q", "FETCH_HEAD"], dest, what=f"checking out {ref}")
    return _run(["git", "rev-parse", "HEAD"], dest, what="rev-parse").strip()


def uv_lock(line_dir: Path, *, offline: bool = False) -> None:
    cmd = ["uv", "lock", "--directory", str(line_dir)]
    if offline:
        cmd.append("--offline")
    _run(cmd, what=f"uv lock in {line_dir}")


# ── uv.lock reading (3.10 has no tomllib, and uv.lock's [[package]] header is regular) ──


def lock_versions(lock_text: str, *, exclude: set[str] = frozenset()) -> dict[str, str]:
    """``{name: version}`` for every REGISTRY package a uv.lock resolves to exactly one version.

    A package the lock splits by marker (two versions for two Pythons) is left out: one
    ``==`` constraint cannot say both, and the line's own lock settles it. Git, path and
    editable sources are not version pins and are left out too.
    """
    seen: dict[str, set[str]] = {}
    for block in lock_text.split("\n[[package]]\n")[1:]:
        name = re.search(r'^name = "([^"]+)"', block, re.M)
        version = re.search(r'^version = "([^"]+)"', block, re.M)
        source = re.search(r"^source = \{ (\w+)", block, re.M)
        if not (name and version and source) or source.group(1) != "registry":
            continue
        seen.setdefault(name.group(1), set()).add(version.group(1))
    return {n: next(iter(v)) for n, v in sorted(seen.items()) if len(v) == 1 and n not in exclude}


def render_pins(pins: dict[str, str]) -> str:
    lines = [PINS_BEGIN, "constraint-dependencies = ["]
    lines += [f'    "{name}=={version}",' for name, version in sorted(pins.items())]
    lines += ["]", PINS_END]
    return "\n".join(lines)


def replace_pins(pyproject: str, pins: dict[str, str]) -> str:
    start, end = pyproject.find(PINS_BEGIN), pyproject.find(PINS_END)
    if start < 0 or end < start:
        raise LineError("pyproject.toml has no managed pins block; was this line made by `quantbox new line`?")
    return pyproject[:start] + render_pins(pins) + pyproject[end + len(PINS_END) :]


# ── templates ────────────────────────────────────────────────────────────

#: template file -> path in the line
_TEMPLATE_FILES = {
    "pyproject.toml.tmpl": "pyproject.toml",
    "README.md.tmpl": "README.md",
    "config.yaml.tmpl": "config.yaml",
    "arms.yaml.tmpl": "arms.yaml",
    "test_reproduces.py.tmpl": "tests/test_reproduces.py",
    "gitignore.tmpl": ".gitignore",
}


def _template(name: str) -> str:
    return _res_files("quantbox").joinpath("line_template").joinpath(name).read_text(encoding="utf-8")


def _fill(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    left = re.findall(r"\{\{[A-Z_]+\}\}", text)
    if left:
        raise LineError(f"template placeholders left unfilled: {sorted(set(left))}")
    return text


def _dataset_pin(dataset: str) -> tuple[str, str]:
    """The datasets.lock entry for *dataset*: the sha256 quantbox would read, or why there is none."""
    from .dataset_lock import DatasetResolveError, resolve_dataset

    try:
        resolved = resolve_dataset(dataset, lock=None)
    except DatasetResolveError as exc:
        return "", str(exc)
    if not resolved["actual_sha256"]:
        return "", f"{resolved['path']} has no prices.parquet"
    return resolved["actual_sha256"], ""


def _datasets_lock_text(dataset: str, sha: str, why_not: str) -> str:
    head = (
        "# quantbox-datasets pins: dataset -> sha256 of prices.parquet. "
        "Written by `quantbox new line`; refresh with `quantbox-datasets pin <name>`.\n"
    )
    if sha:
        return head + f"{dataset}: {sha}\n"
    return head + f"# {dataset} is NOT pinned yet ({why_not}).\n# Run `quantbox-datasets pin {dataset}` here.\n"


# ── the two commands ─────────────────────────────────────────────────────


def _engine(url: str, ref: str | None, workdir: Path, *, ref_flag: str) -> tuple[EngineRef, Path]:
    """Resolve and fetch the engine a line will pin, refusing one that cannot run a line.

    Both ``new_line`` and ``repin`` come through here, so the line-support gate holds for
    an explicit ref and for the latest-tag default alike, before anything is written.
    *ref_flag* is the CLI option the caller's user passes to choose another ref.
    """
    ref = ref or latest_tag(url)
    sha = fetch_ref(url, ref, workdir)
    if not (workdir / LINE_SUPPORT_FILE).is_file():
        raise LineError(
            f"quantbox {ref} ({sha[:12]}) predates `quantbox new line`: it has no arms files, "
            "by-name datasets or run manifest, so a line pinned to it would not run. "
            f"Pass {ref_flag} <a tag or SHA that has them>."
        )
    return EngineRef(url=url, ref=ref, sha=sha), workdir


def _derive_pins(line_dir: Path, engine_tree: Path, *, lock: bool) -> int:
    """Write the managed pins block: quantbox's tested lock first, then the line's own resolution.

    Returns how many packages are pinned. With ``lock=False`` only the first half runs
    (no resolution happens, so nothing more can be known).
    """
    exclude = {"quantbox", "quantbox-datasets"}
    engine_lock = engine_tree / "uv.lock"
    pins = lock_versions(engine_lock.read_text(encoding="utf-8"), exclude=exclude) if engine_lock.is_file() else {}
    pyproject = line_dir / "pyproject.toml"
    pyproject.write_text(replace_pins(pyproject.read_text(encoding="utf-8"), pins), encoding="utf-8")
    if not lock:
        return len(pins)
    uv_lock(line_dir)
    resolved = lock_versions((line_dir / "uv.lock").read_text(encoding="utf-8"), exclude=exclude)
    # The line's resolution only ADDS: a package quantbox's lock settled keeps that version.
    merged = {**resolved, **{k: v for k, v in pins.items() if k in resolved}}
    pyproject.write_text(replace_pins(pyproject.read_text(encoding="utf-8"), merged), encoding="utf-8")
    uv_lock(line_dir)  # the lock records the constraints it was resolved under
    return len(merged)


def new_line(
    slug: str,
    *,
    directory: str | Path | None = None,
    quantbox_ref: str | None = None,
    quantbox_url: str = QUANTBOX_URL,
    datasets_ref: str | None = None,
    datasets_url: str = DATASETS_URL,
    extras: str = DEFAULT_EXTRAS,
    python: str = DEFAULT_PYTHON,
    dataset: str = DEFAULT_DATASET,
    question: str | None = None,
    lock: bool = True,
    today: _dt.date | None = None,
) -> dict[str, Any]:
    """Scaffold the Model C line *slug* in *directory* (default ``./<slug>``); return what was pinned.

    Refuses a non-empty target, a slug that is not a lowercase name, and a quantbox ref
    that predates line support (the skeleton would not run on it).
    """
    if not _SLUG_RE.match(slug):
        raise LineError(f"line slug {slug!r}: use lowercase letters, digits, '-' and '_'")
    target = Path(directory) if directory is not None else Path.cwd() / slug
    if target.exists() and any(target.iterdir()):
        raise LineError(f"{target} exists and is not empty")
    if not re.match(r"^3\.\d+$", python):
        raise LineError(f"--python {python!r}: give a minor version such as 3.12")

    with tempfile.TemporaryDirectory(prefix="quantbox-line-") as tmp:
        engine, tree = _engine(quantbox_url, quantbox_ref, Path(tmp) / "quantbox", ref_flag="--quantbox-ref")
        ds_sha = datasets_ref if datasets_ref and _SHA40_RE.match(datasets_ref) else None
        if ds_sha is None:
            ds_sha = (
                remote_head(datasets_url)
                if datasets_ref is None
                else fetch_ref(datasets_url, datasets_ref, Path(tmp) / "datasets")
            )
        pin_sha, why_not = _dataset_pin(dataset)
        today = today or _dt.date.today()
        major, minor = python.split(".")
        values = {
            "SLUG": slug,
            "QUESTION": question or f"TODO: one sentence — what {slug} investigates",
            "STARTED": today.isoformat(),
            "ASOF": today.isoformat(),
            "PYTHON": python,
            "PYTHON_NEXT": f"{major}.{int(minor) + 1}",
            "EXTRAS": f"[{extras}]" if extras else "",
            "QUANTBOX_URL": engine.url,
            "QUANTBOX_REF": engine.ref,
            "QUANTBOX_SHA": engine.sha,
            "DATASETS_URL": datasets_url,
            "DATASETS_SHA": ds_sha,
            "DATASET": dataset,
            "PINS": render_pins({}),
        }
        target.mkdir(parents=True, exist_ok=True)
        for src, dst in _TEMPLATE_FILES.items():
            path = target / dst
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_fill(_template(src), values), encoding="utf-8")
        (target / ".python-version").write_text(python + "\n", encoding="utf-8")
        (target / "datasets.lock").write_text(_datasets_lock_text(dataset, pin_sha, why_not), encoding="utf-8")
        (target / "findings").mkdir(exist_ok=True)
        (target / "findings" / ".gitkeep").write_text("", encoding="utf-8")
        n_pins = _derive_pins(target, tree, lock=lock)

    return {
        "line": str(target),
        "quantbox": {"url": engine.url, "ref": engine.ref, "sha": engine.sha, "extras": extras},
        "quantbox_datasets": {"url": datasets_url, "sha": ds_sha},
        "dataset": {"name": dataset, "sha256": pin_sha or None, "unpinned_reason": why_not or None},
        "pinned_packages": n_pins,
        "locked": lock,
    }


def repin(
    line_dir: str | Path,
    *,
    ref: str | None = None,
    quantbox_url: str | None = None,
    lock: bool = True,
) -> dict[str, Any]:
    """Move *line_dir*'s quantbox pin to *ref* (default: the latest tag) and re-derive every exact pin.

    The URL defaults to the one the line already pins. Refuses, before writing anything,
    a ref that predates line support. Returns old and new ref/SHA.
    """
    line_dir = Path(line_dir)
    pyproject_path = line_dir / "pyproject.toml"
    if not pyproject_path.is_file():
        raise LineError(f"{line_dir} has no pyproject.toml")
    text = pyproject_path.read_text(encoding="utf-8")
    deps = _QB_DEP_RE.findall(text)
    match = _QB_DEP_RE.search(text)
    if len(deps) != 1 or match is None:
        raise LineError(
            f"{pyproject_path}: expected exactly one `quantbox @ git+<url>@<sha>` dependency, found {len(deps)}"
        )
    if PINS_BEGIN not in text:
        raise LineError(f"{pyproject_path} has no managed pins block; was this line made by `quantbox new line`?")
    old_sha = match.group("sha")
    old_ref = (re.search(r'^quantbox-ref = "([^"]*)"$', text, re.M) or [None, None])[1]
    url = quantbox_url or match.group("url")

    with tempfile.TemporaryDirectory(prefix="quantbox-repin-") as tmp:
        engine, tree = _engine(url, ref, Path(tmp) / "quantbox", ref_flag="--ref")
        new_dep = f'"quantbox{match.group(1) or ""} @ git+{engine.url}@{engine.sha}",  # {engine.ref}'
        text = text[: match.start()] + new_dep + text[match.end() :]
        text = _REF_LINE_RE.sub(f'quantbox-ref = "{engine.ref}"', text)
        text = _SHA_LINE_RE.sub(f'quantbox-sha = "{engine.sha}"', text)
        pyproject_path.write_text(text, encoding="utf-8")
        n_pins = _derive_pins(line_dir, tree, lock=lock)

    return {
        "line": str(line_dir),
        "from": {"ref": old_ref, "sha": old_sha},
        "to": {"url": engine.url, "ref": engine.ref, "sha": engine.sha},
        "pinned_packages": n_pins,
        "locked": lock,
    }
