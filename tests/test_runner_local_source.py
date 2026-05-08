"""Tests for local-source plugin loading + safety rails in the runner.

Local-source plugins (``source: path/to/file.py:Class``) let LLM-authored or
one-shot research code participate in a quantbox pipeline without going through
package release. See ``docs/architecture/plugin-authoring.md`` and ADR-0003.

Safety rails enforced here:
  - Local-source forbidden for broker / pipeline kinds.
  - Local-source forbidden in paper / live modes.
  - meta attribute is required.
  - meta.kind must match the expected kind.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantbox.contracts import PluginMeta
from quantbox.exceptions import PluginNotFoundError
from quantbox.runner import _load_local_source_class, _resolve_plugin_cls

# --- Fixtures ---


_SCRATCH_STRATEGY_SRC = '''
"""Synthetic strategy for local-source plugin tests."""
from dataclasses import dataclass
from quantbox.contracts import PluginMeta


@dataclass
class ScratchStrategy:
    meta = PluginMeta(
        name="lab.strategy.scratch.v1",
        kind="strategy",
        version="0.0.1",
        core_compat=">=0.1,<0.2",
        status="research",
        description="Synthetic scratch strategy for tests.",
    )

    def run(self, data, params=None):
        return {"weights": data["prices"].tail(1) * 0}
'''


_NO_META_SRC = '''
"""Plugin class without a ``meta`` attribute — must be rejected."""

class Bogus:
    def run(self, data, params=None):
        return {}
'''


@pytest.fixture
def scratch_strategy_file(tmp_path: Path) -> Path:
    """Write a synthetic strategy file to a tmp dir; return its path."""
    p = tmp_path / "scratch_strat.py"
    p.write_text(_SCRATCH_STRATEGY_SRC)
    return p


@pytest.fixture
def no_meta_file(tmp_path: Path) -> Path:
    """Write a class file with no ``meta`` — must be rejected."""
    p = tmp_path / "no_meta.py"
    p.write_text(_NO_META_SRC)
    return p


# --- _load_local_source_class ---


def test_load_local_source_returns_class(scratch_strategy_file: Path):
    cls = _load_local_source_class(f"{scratch_strategy_file}:ScratchStrategy", expected_kind="strategy")
    assert cls.__name__ == "ScratchStrategy"
    assert cls.meta.name == "lab.strategy.scratch.v1"
    assert cls.meta.kind == "strategy"
    assert cls.meta.status == "research"


def test_load_local_source_kind_mismatch_raises(scratch_strategy_file: Path):
    with pytest.raises(ValueError, match="meta.kind"):
        _load_local_source_class(f"{scratch_strategy_file}:ScratchStrategy", expected_kind="data")


def test_load_local_source_missing_class_raises(scratch_strategy_file: Path):
    with pytest.raises(AttributeError, match="not found"):
        _load_local_source_class(f"{scratch_strategy_file}:DoesNotExist")


def test_load_local_source_no_meta_raises(no_meta_file: Path):
    with pytest.raises(AttributeError, match="meta"):
        _load_local_source_class(f"{no_meta_file}:Bogus")


def test_load_local_source_missing_file_raises(tmp_path: Path):
    bad = tmp_path / "nonexistent.py"
    with pytest.raises(FileNotFoundError):
        _load_local_source_class(f"{bad}:Whatever")


def test_load_local_source_malformed_string_raises():
    with pytest.raises(ValueError, match="path:ClassName"):
        _load_local_source_class("not_a_valid_source_string")


# --- _resolve_plugin_cls (resolution + safety rails) ---


def test_resolve_via_registry_name():
    """When spec has 'name', look up in the registry dict."""

    class FooStrategy:
        meta = PluginMeta(name="foo.strategy.v1", kind="strategy", version="0.1.0", core_compat=">=0.1,<0.2")

    registry = {"foo.strategy.v1": FooStrategy}
    cls = _resolve_plugin_cls({"name": "foo.strategy.v1"}, registry, "strategy", mode="backtest")
    assert cls is FooStrategy


def test_resolve_unknown_name_raises():
    with pytest.raises(PluginNotFoundError):
        _resolve_plugin_cls({"name": "missing.v1"}, {}, "strategy", mode="backtest")


def test_resolve_via_local_source(scratch_strategy_file: Path):
    """When spec has 'source', load from local file."""
    spec = {"source": f"{scratch_strategy_file}:ScratchStrategy"}
    cls = _resolve_plugin_cls(spec, {}, "strategy", mode="backtest")
    assert cls.__name__ == "ScratchStrategy"


def test_resolve_neither_name_nor_source_raises():
    with pytest.raises(ValueError, match="name.*source"):
        _resolve_plugin_cls({"params": {}}, {}, "strategy", mode="backtest")


# Safety rails


@pytest.mark.parametrize("forbidden_kind", ["broker", "pipeline"])
def test_local_source_forbidden_for_dangerous_kinds(scratch_strategy_file: Path, forbidden_kind: str):
    """Brokers and pipelines must come from registered entry points."""
    spec = {"source": f"{scratch_strategy_file}:ScratchStrategy"}
    with pytest.raises(ValueError, match="forbidden"):
        _resolve_plugin_cls(spec, {}, forbidden_kind, mode="backtest")


@pytest.mark.parametrize("prod_mode", ["paper", "live"])
def test_local_source_forbidden_in_production_modes(scratch_strategy_file: Path, prod_mode: str):
    """Local-source plugins are research-only — refused in paper/live."""
    spec = {"source": f"{scratch_strategy_file}:ScratchStrategy"}
    with pytest.raises(ValueError, match="forbidden"):
        _resolve_plugin_cls(spec, {}, "strategy", mode=prod_mode)


def test_registered_plugins_work_in_all_modes():
    """Registered (entry-point) plugins are unaffected by the safety rails."""

    class Reg:
        meta = PluginMeta(name="r.v1", kind="strategy", version="0.1.0", core_compat=">=0.1,<0.2")

    registry = {"r.v1": Reg}
    for mode in ("backtest", "paper", "live"):
        cls = _resolve_plugin_cls({"name": "r.v1"}, registry, "strategy", mode=mode)
        assert cls is Reg
