"""No doc names a ``quantbox.*`` module or attribute that does not exist (TOM-1341).

The architecture docs once advertised ``quantbox.opt`` and ``quantbox.score``
as L1 modules; neither was ever built. A dotted ``quantbox.x.y`` reference in a
doc is resolved against the installed package: the longest importable prefix
must be a module, and every remaining segment an attribute of it. Code blocks
are checked too — a snippet importing a missing module is the same claim. So
is ``from quantbox import run_from_config``: each imported name is checked as
``quantbox.run_from_config`` (TOM-1449; api-layers.md carried exactly that
line, and the dotted pattern alone did not see it).

Three things are not a claim about the current code, and are skipped:

* ``docs/adr/`` and ``CHANGELOG.md`` — dated records of what was decided or
  shipped then; they are not rewritten afterwards.
* entry-point GROUP names (``quantbox.brokers``, ``quantbox.data``, ...) —
  read from ``registry.ENTRYPOINT_GROUPS``, the owner, not restated here.
* a doc carrying ``DESIGN_ONLY_MARKER`` — a design for something not built,
  which must say so in its own visible banner.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from pathlib import Path

import pytest

from quantbox.registry import ENTRYPOINT_GROUPS

REPO = Path(__file__).resolve().parents[1]

DOC_GLOBS = ("*.md", "docs/**/*.md", "templates/**/*.md", "cookbook/**/*.md", "skills/**/*.md")
DESIGN_ONLY_MARKER = "<!-- design-only:"

_REF = re.compile(r"(?<![\w./-])quantbox(?:\.[A-Za-z_]\w*)+")
_FROM_IMPORT = re.compile(r"\bfrom\s+(quantbox(?:\.[A-Za-z_]\w*)*)\s+import\s+([^#\n]+)")


def _line_refs(line: str) -> list[str]:
    """Every dotted name a doc line claims, including each `from quantbox... import x`."""
    refs = _REF.findall(line)
    for m in _FROM_IMPORT.finditer(line):
        for item in m.group(2).replace("(", " ").replace(")", " ").split(","):
            name = item.split(" as ")[0].strip()
            if name.isidentifier():
                refs.append(f"{m.group(1)}.{name}")
    return refs


def _skipped(rel: Path) -> bool:
    return rel.name == "CHANGELOG.md" or ".claude" in rel.parts or rel.parts[:2] == ("docs", "adr")


def _doc_files() -> list[Path]:
    files = {p for g in DOC_GLOBS for p in REPO.glob(g)}
    return sorted(p for p in files if not _skipped(p.relative_to(REPO)))


def _resolves(dotted: str) -> bool:
    if dotted in ENTRYPOINT_GROUPS.values():
        return True
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:i])
        try:
            spec = importlib.util.find_spec(mod_name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            continue
        obj = importlib.import_module(mod_name)
        for attr in parts[i:]:
            if not hasattr(obj, attr):
                return False
            obj = getattr(obj, attr)
        return True
    return False


def _refs() -> list[tuple[str, int, str]]:
    out = []
    for path in _doc_files():
        text = path.read_text(encoding="utf-8")
        if DESIGN_ONLY_MARKER in text:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for ref in _line_refs(line):
                out.append((str(path.relative_to(REPO)), lineno, ref))
    return out


def test_the_scan_sees_docs() -> None:
    """Guard against a vacuous pass: the globs must find docs and references."""
    assert len(_doc_files()) > 10
    assert len(_refs()) > 20


def test_resolver_rejects_a_missing_module() -> None:
    assert _resolves("quantbox.contracts.PluginMeta")
    assert _resolves("quantbox.brokers")  # entry-point group, not a module
    assert not _resolves("quantbox.opt")
    assert not _resolves("quantbox.score.peer_z")
    assert not _resolves("quantbox.contracts.NoSuchThing")


def test_from_import_names_are_claims() -> None:
    assert _line_refs("from quantbox import run_from_config") == ["quantbox.run_from_config"]
    assert not _resolves("quantbox.run_from_config")
    assert _line_refs("from quantbox.runner import (run_from_config as r, Foo)  # x") == [
        "quantbox.runner",
        "quantbox.runner.run_from_config",
        "quantbox.runner.Foo",
    ]
    assert _resolves("quantbox.runner.run_from_config")


@pytest.mark.filterwarnings("ignore")
def test_no_doc_names_a_missing_quantbox_module() -> None:
    missing = [f"{f}:{n}: {ref}" for f, n, ref in _refs() if not _resolves(ref)]
    assert not missing, "docs name quantbox modules/attributes that do not exist:\n" + "\n".join(missing)
