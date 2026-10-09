"""Read the layer map (ADR-0010, TOM-1451) from its one owner: the import-linter
contracts in ``pyproject.toml`` ``[tool.importlinter]``.

Each layer is the ``source_modules`` of the contract whose ``id`` is the layer's
name. Entries are module expressions in import-linter's syntax, restricted to two
forms: an exact module name, or ``X.**`` (every module below X, not X itself).
``tests/test_layer_map.py`` checks that this reading matches import-linter's own
expansion, and that every module under ``src/quantbox`` is in exactly one layer.

Stdlib only: ``scripts/check_no_vectorbt.sh`` imports it in a clean venv that has
no dev group. Usage from a script::

    from layer_map import layer_of, read_layers
    layers = read_layers(Path("pyproject.toml"))
    layer_of("quantbox.engine.vectorbt", layers)  # -> "research"
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 (pytest depends on tomli there)
    import tomli as tomllib

# Layer name -> its rank in import order. A layer may import only a lower rank;
# research and trade share rank 2 and never import each other.
LAYERS: dict[str, int] = {"core": 0, "plugins": 1, "research": 2, "trade": 2}


def read_contracts(pyproject: Path) -> list[dict]:
    """The ``[[tool.importlinter.contracts]]`` tables, in file order."""
    data = tomllib.loads(pyproject.read_text())
    return list(data["tool"]["importlinter"]["contracts"])


def read_layers(pyproject: Path) -> dict[str, list[str]]:
    """Layer name -> the module expressions of that layer's contract."""
    by_id = {c.get("id"): c for c in read_contracts(pyproject)}
    missing = [name for name in LAYERS if name not in by_id]
    if missing:
        raise ValueError(f"{pyproject}: no import-linter contract with id {missing}")
    return {name: list(by_id[name]["source_modules"]) for name in LAYERS}


def matches(expression: str, module: str) -> bool:
    """Whether ``module`` is in ``expression`` (exact name, or ``X.**``)."""
    if expression.endswith(".**"):
        return module.startswith(expression[:-2])
    if "*" in expression:
        raise ValueError(f"unsupported module expression {expression!r}: use a module name or X.**")
    return module == expression


def layers_of(module: str, layers: dict[str, list[str]]) -> list[str]:
    """Every layer whose expressions include ``module`` (exactly one when the map is sound)."""
    return [name for name, exprs in layers.items() if any(matches(e, module) for e in exprs)]


def layer_of(module: str, layers: dict[str, list[str]]) -> str:
    """The one layer of ``module``; raises when it is in none or in several."""
    found = layers_of(module, layers)
    if len(found) != 1:
        raise ValueError(f"{module} is in {len(found)} layers {found}; pyproject.toml [tool.importlinter] owns the map")
    return found[0]


def module_names(package_dir: Path) -> list[str]:
    """Every module under a package directory, as dotted names (packages by their __init__)."""
    names = []
    for path in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(package_dir.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names
