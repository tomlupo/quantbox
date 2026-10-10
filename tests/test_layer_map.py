"""TOM-1451 (4b-3): the layer map in pyproject.toml covers every module, once.

The import-linter contracts in ``pyproject.toml`` ``[tool.importlinter]`` own
the layer map (ADR-0010): core < plugins < research, trade. ``lint-imports``
(ci.yml job ``lint``) checks the imports against them. It cannot see a module
that no contract names: that module would import anything, unchecked. These
tests close that hole:

- every module under ``src/quantbox`` is in exactly one layer;
- each contract forbids exactly the layers above its source, so a module added
  to one layer is forbidden to every layer below it;
- every entry still names a module (a stale entry checks nothing);
- the stdlib reader ``scripts/layer_map.py`` (which the client smoke uses)
  expands each entry exactly as import-linter does.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE = ROOT / "src" / "quantbox"

_spec = importlib.util.spec_from_file_location("layer_map", ROOT / "scripts" / "layer_map.py")
layer_map = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(layer_map)

# Layer -> the layers its contract forbids: every layer above it, plus the sibling
# for research and trade.
FORBIDDEN = {
    "core": {"plugins", "research", "trade"},
    "plugins": {"research", "trade"},
    "research": {"trade"},
    "trade": {"research"},
}
CORE_EXTERNALS = {"vectorbt", "numba", "ccxt"}


@pytest.fixture(scope="module")
def modules() -> list[str]:
    names = layer_map.module_names(PACKAGE)
    # The walk must have looked: 193 modules at 4b-3.
    assert len(names) >= 150, f"walked only {len(names)} modules under {PACKAGE}"
    return names


@pytest.fixture(scope="module")
def contracts() -> dict[str, dict]:
    return {c["id"]: c for c in layer_map.read_contracts(PYPROJECT)}


@pytest.fixture(scope="module")
def layers() -> dict[str, list[str]]:
    return layer_map.read_layers(PYPROJECT)


def _resolve(expressions: list[str], modules: list[str]) -> set[str]:
    return {m for m in modules if any(layer_map.matches(e, m) for e in expressions)}


def test_every_module_is_in_exactly_one_layer(modules, layers):
    unassigned = [m for m in modules if not layer_map.layers_of(m, layers)]
    twice = {m: found for m in modules if len(found := layer_map.layers_of(m, layers)) > 1}
    assert not unassigned, (
        f"{len(unassigned)} module(s) in no layer: {unassigned}. Add each to the source_modules of its "
        "layer's contract in pyproject.toml [tool.importlinter], and to forbidden_modules of every contract "
        "below it; decide the layer by who imports it."
    )
    assert not twice, f"module(s) in more than one layer: {twice}"
    counts = {name: len(_resolve(exprs, modules)) for name, exprs in layers.items()}
    assert sum(counts.values()) == len(modules), counts
    assert all(counts.values()), f"an empty layer checks nothing: {counts}"


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_each_contract_forbids_exactly_the_layers_above_it(layer, modules, layers, contracts):
    contract = contracts[layer]
    expected = set().union(*(_resolve(layers[upper], modules) for upper in FORBIDDEN[layer]))
    forbidden = _resolve(contract["forbidden_modules"], modules)
    assert forbidden == expected, (
        f"contract {layer!r}: forbidden_modules must be exactly the layers {sorted(FORBIDDEN[layer])}; "
        f"missing {sorted(expected - forbidden)}, extra {sorted(forbidden - expected)}"
    )


def test_core_externals_contract_covers_core(modules, layers, contracts):
    contract = contracts["core-externals"]
    assert _resolve(contract["source_modules"], modules) == _resolve(layers["core"], modules)
    assert set(contract["forbidden_modules"]) == CORE_EXTERNALS
    settings = layer_map.tomllib.loads(PYPROJECT.read_text())["tool"]["importlinter"]
    # Without it import-linter refuses an external forbidden module.
    assert settings["include_external_packages"] is True
    assert settings["root_package"] == "quantbox"


def test_contracts_are_module_level_forbidden_contracts_without_ignores(contracts):
    assert set(contracts) == set(FORBIDDEN) | {"core-externals"}
    for cid, contract in contracts.items():
        assert contract["type"] == "forbidden", cid
        # The layers interleave in the package tree (engine is core, engine.vectorbt
        # research), so an entry must mean ONE module, never a package with its children.
        assert contract["as_packages"] is False, cid
        # An ignored import is a hole in the map. Unavoidable ones are listed with a reason.
        assert not contract.get("ignore_imports"), f"{cid}: ignore_imports {contract.get('ignore_imports')}"


def test_every_internal_entry_names_a_module(modules, contracts):
    stale = []
    for cid, contract in contracts.items():
        for key in ("source_modules", "forbidden_modules"):
            for expr in contract[key]:
                if expr.split(".")[0] != "quantbox":
                    continue
                if not _resolve([expr], modules):
                    stale.append(f"{cid}.{key}: {expr}")
    assert not stale, f"entries that match no module check nothing: {stale}"


def test_the_stdlib_reader_expands_entries_as_import_linter_does(modules, contracts):
    grimp = pytest.importorskip("grimp")
    graph = grimp.build_graph("quantbox")
    assert {m for m in graph.modules if m.split(".")[0] == "quantbox"} == set(modules)
    for contract in contracts.values():
        for expr in contract["source_modules"] + contract["forbidden_modules"]:
            if expr.split(".")[0] != "quantbox":
                continue
            theirs = set(graph.find_matching_modules(expr)) if "*" in expr else {expr}
            assert _resolve([expr], modules) == theirs, expr


def test_every_module_the_client_pack_imports_is_core(layers):
    """The client pack mirrors what robo imports (TOM-1451): a client uses core only."""
    import ast

    pack = ROOT / "tests" / "fixtures" / "client_pack" / "src"
    imported = set()
    for path in pack.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "quantbox":
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported |= {a.name for a in node.names if a.name.split(".")[0] == "quantbox"}
    # Count what was looked at: robo's surface is these five modules.
    assert imported >= {
        "quantbox.bootstrap",
        "quantbox.cache.strategy_cache",
        "quantbox.contracts",
        "quantbox.features",
        "quantbox.features.covariance",
    }, imported
    assert {m: layer_map.layer_of(m, layers) for m in sorted(imported)} == dict.fromkeys(sorted(imported), "core")


def test_the_reader_refuses_a_wildcard_it_cannot_expand():
    with pytest.raises(ValueError, match="unsupported module expression"):
        layer_map.matches("quantbox.*", "quantbox.cli")
    with pytest.raises(ValueError, match="in 0 layers"):
        layer_map.layer_of("quantbox.not_a_module", {"core": ["quantbox.cli"]})
