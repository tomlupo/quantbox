"""The runner refuses what ``quantbox validate`` refuses: an unknown plugin (TOM-1529).

Since TOM-1528 ``validate`` refuses a block that names a plugin this environment does
not register. The runner only logged "Validation plugin '...' not found, skipping"
(and the same for monitors), so a typo silently dropped a check. The run now raises
the SAME ``unknown_plugin`` finding, from the same code path (``check_plugin_params``),
as the ``PluginNotFoundError`` a run has always raised for an unknown name, before any
work: no run directory is created.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from quantbox.exceptions import PluginNotFoundError
from quantbox.registry import PluginRegistry
from quantbox.runner import run_from_config
from quantbox.validate import validate_config

REPO = Path(__file__).resolve().parents[1]
SYNTHETIC = REPO / "cookbook" / "configs" / "run_synthetic_backtest.yaml"
REG = PluginRegistry.discover()


def _synthetic(tmp_path: Path) -> dict:
    cfg = yaml.safe_load(SYNTHETIC.read_text(encoding="utf-8"))
    cfg["artifacts"]["root"] = str(tmp_path / "artifacts")
    # Small and fast: the refusal comes before any data is generated.
    cfg["plugins"]["pipeline"]["params"]["prices"]["n_steps"] = 300
    return cfg


@pytest.mark.parametrize(
    ("slot", "name"),
    [("validation", "validation.walk_forwrd.v1"), ("monitors", "monitor.drawdwn.v1")],
)
def test_an_unknown_validation_or_monitor_plugin_fails_the_run_with_validates_error(tmp_path, slot, name):
    cfg = _synthetic(tmp_path)
    cfg["plugins"][slot] = [{"name": name, "params": {}}]

    validate_errors = [f.message for f in validate_config(cfg, REG) if f.level == "error"]
    (expected,) = [m for m in validate_errors if m.startswith("unknown_plugin:")]
    assert f"'{name}'" in expected

    # Validate's finding, verbatim, as the PluginNotFoundError a run raises for an unknown name.
    with pytest.raises(PluginNotFoundError) as exc:
        run_from_config(cfg, REG)
    assert str(exc.value) == expected
    assert exc.value.plugin_name == name
    assert exc.value.group == slot.removesuffix("s")
    assert exc.value.available and name not in exc.value.available
    # Refused before any work: no run directory, no artifact.
    assert not (tmp_path / "artifacts").exists() or not any((tmp_path / "artifacts").iterdir())


def test_a_monitor_is_refused_in_a_mode_that_would_not_run_it(tmp_path):
    """Monitors run in paper/live only; the name is still checked in a backtest, as validate does."""
    cfg = _synthetic(tmp_path)
    assert cfg["run"]["mode"] == "backtest"
    cfg["plugins"]["monitors"] = [{"name": "monitor.no_such.v1"}]
    with pytest.raises(PluginNotFoundError, match=r"unknown_plugin: 'monitor\.no_such\.v1'"):
        run_from_config(cfg, REG)


def test_an_unknown_param_still_only_warns(tmp_path, caplog):
    """Only the unknown PLUGIN is refused; an unknown param keeps its TOM-1350 warning."""
    cfg = _synthetic(tmp_path)
    cfg["plugins"]["validation"] = [{"name": "validation.turnover.v1", "params": {"no_such_knob": 1}}]
    with caplog.at_level("WARNING", logger="quantbox.runner"):
        result = run_from_config(cfg, REG)
    assert any("no_such_knob" in r.getMessage() for r in caplog.records)
    assert [v["plugin"] for v in result.notes["validation"]] == ["validation.turnover.v1"]


def test_a_registered_validation_plugin_runs_and_writes_its_artifact(tmp_path):
    cfg = _synthetic(tmp_path)
    cfg["plugins"]["validation"] = [{"name": "validation.turnover.v1"}]
    result = run_from_config(cfg, REG)
    assert Path(result.artifacts["validation"]).is_file()
    assert [v["plugin"] for v in result.notes["validation"]] == ["validation.turnover.v1"]
