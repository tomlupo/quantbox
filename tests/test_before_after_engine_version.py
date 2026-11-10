"""TOM-2566: scripts/before_after.py masks the engine version a run records, and only that.

The v0.21.0 promotion compare (quantbox#322) read DIFFERS on one thing: the engine
version label, run_manifest.json ``engine.version`` and the ``engine`` row of
finding_report.json's provenance table ("rsims 0.20.0" -> "rsims 0.21.0"). Each case
goes through ``diff_runs`` on real files. The engine name, every other row and every
number must still compare.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("before_after_engine", ROOT / "scripts" / "before_after.py")
ba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ba)


def _side(name: str, version: str, sharpe: float = 1.25) -> tuple[dict, dict]:
    manifest = {"run_id": "r", "engine": {"name": name, "version": version}, "metrics": {"sharpe": sharpe}}
    report = {
        "tables": [
            {
                "title": "Run provenance",
                "columns": ["field", "arm"],
                "rows": [["run_id", "r"], ["pipeline", "p"], ["engine", f"{name} {version}"], ["sharpe", sharpe]],
            }
        ]
    }
    return manifest, report


def _diff(tmp_path: Path, base: tuple[dict, dict], head: tuple[dict, dict]) -> tuple[dict[str, list[str]], int]:
    sides = []
    for side_name, (manifest, report) in (("base", base), ("head", head)):
        run = tmp_path / side_name / "run"
        run.mkdir(parents=True)
        (run / ".run_id").write_text(f"{side_name}-run")
        (run / "run_manifest.json").write_text(json.dumps(manifest))
        (run / "finding_report.json").write_text(json.dumps(report))
        sides.append((ba.Side(side_name, tmp_path / side_name, sys.executable, {}, "0" * 40), run))
    (sb, a), (sh, b) = sides
    return ba.diff_runs(sb, sh, a, b, tmp_path)


def test_engine_version_only_is_identical(tmp_path) -> None:
    found, compared = _diff(tmp_path, _side("rsims", "0.20.0"), _side("rsims", "0.21.0"))
    assert compared == 2
    assert found == {}, found


def test_number_in_the_same_table_still_differs(tmp_path) -> None:
    found, _ = _diff(tmp_path, _side("rsims", "0.20.0"), _side("rsims", "0.21.0", sharpe=1.5))
    assert "rows[3][1]" in " ".join(found.get("finding_report.json", []))
    assert "metrics" in " ".join(found.get("run_manifest.json", []))
    assert not any("engine" in line for lines in found.values() for line in lines), found


def test_engine_name_still_differs(tmp_path) -> None:
    found, _ = _diff(tmp_path, _side("rsims", "0.20.0"), _side("vectorbt", "0.20.0"))
    assert set(found) == {"run_manifest.json", "finding_report.json"}


def test_other_row_carrying_the_version_still_differs(tmp_path) -> None:
    base, head = _side("rsims", "0.20.0"), _side("rsims", "0.21.0")
    base, head = copy.deepcopy(base), copy.deepcopy(head)
    base[1]["tables"][0]["rows"][1][1] = "rsims 0.20.0"
    head[1]["tables"][0]["rows"][1][1] = "rsims 0.21.0"
    found, _ = _diff(tmp_path, base, head)
    assert list(found) == ["finding_report.json"]
    assert "rows[1][1]" in found["finding_report.json"][0]
