"""Tests for ``quantbox.bootstrap.config``.

Covers yaml load, profile-override deep-merge, ``extra_data`` merge,
and schema validation via a duck-typed stub ``schema_cls``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantbox.bootstrap import load_config, load_yaml_merged


@pytest.fixture
def config_tree(tmp_path: Path) -> Path:
    """Build a tiny config tree: base yaml + overrides/dev.yaml."""
    base = tmp_path / "config.yaml"
    base.write_text(
        "\n".join(
            [
                "output_dir: output",
                "processes:",
                "  foo:",
                "    report_date: -1",
                "    timeout: 300",
                "market:",
                "  calendar: XWAR",
            ]
        ),
        encoding="utf-8",
    )
    overrides = tmp_path / "overrides"
    overrides.mkdir()
    (overrides / "dev.yaml").write_text(
        "\n".join(
            [
                "output_dir: dev_output",
                "processes:",
                "  foo:",
                "    timeout: 60",
            ]
        ),
        encoding="utf-8",
    )
    (overrides / "prod.yaml").write_text("output_dir: prod_output\n", encoding="utf-8")
    return tmp_path


def test_load_yaml_merged_base_only(config_tree: Path) -> None:
    data = load_yaml_merged(config_tree / "config.yaml")
    assert data["output_dir"] == "output"
    assert data["processes"]["foo"]["timeout"] == 300
    assert data["market"]["calendar"] == "XWAR"


def test_load_yaml_merged_with_profile(config_tree: Path) -> None:
    data = load_yaml_merged(
        config_tree / "config.yaml",
        overrides_dir=config_tree / "overrides",
        override_profile="dev",
    )
    # Override wins for scalar
    assert data["output_dir"] == "dev_output"
    # Nested deep-merge: processes.foo.timeout overridden, report_date preserved
    assert data["processes"]["foo"]["timeout"] == 60
    assert data["processes"]["foo"]["report_date"] == -1
    # Base-only keys untouched
    assert data["market"]["calendar"] == "XWAR"


def test_load_yaml_merged_missing_profile_silent(config_tree: Path) -> None:
    # Absent profile yaml — no error.
    data = load_yaml_merged(
        config_tree / "config.yaml",
        overrides_dir=config_tree / "overrides",
        override_profile="nonexistent",
    )
    assert data["output_dir"] == "output"


class _FakeSchema:
    """Duck-typed stand-in for a pydantic BaseModel."""

    def __init__(self, data: dict) -> None:
        self.data = data

    @classmethod
    def model_validate(cls, data: dict) -> _FakeSchema:
        return cls(data)


def test_load_config_validates(config_tree: Path) -> None:
    cfg = load_config(
        _FakeSchema,
        config_tree / "config.yaml",
        overrides_dir=config_tree / "overrides",
        override_profile="dev",
    )
    assert isinstance(cfg, _FakeSchema)
    assert cfg.data["output_dir"] == "dev_output"


def test_load_config_with_extra_data(config_tree: Path) -> None:
    cfg = load_config(
        _FakeSchema,
        config_tree / "config.yaml",
        extra_data={"strategies": {"strat_a": {"weight": 0.4}}},
    )
    assert cfg.data["strategies"] == {"strat_a": {"weight": 0.4}}
    # Base fields still present
    assert cfg.data["output_dir"] == "output"


def test_load_config_extra_data_merges_nested(config_tree: Path) -> None:
    cfg = load_config(
        _FakeSchema,
        config_tree / "config.yaml",
        extra_data={"processes": {"bar": {"timeout": 10}}},
    )
    # processes.foo from base preserved; processes.bar added
    assert cfg.data["processes"]["foo"]["timeout"] == 300
    assert cfg.data["processes"]["bar"]["timeout"] == 10
