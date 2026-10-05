"""Every ``pipeline_smoke`` test lives under ``tests/pipeline/`` (TOM-1500).

CI's smoke job runs ``pytest tests/pipeline/ -m pipeline_smoke``
(``.github/workflows/ci.yml``), and the main job deselects the marker. A smoke
test anywhere else therefore runs in NO CI job: the policies end-to-end test
sat in ``tests/test_rebalancing_policies.py`` that way.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS = Path(__file__).resolve().parent
MARK = re.compile(r"mark\.pipeline_smoke\b")


def test_the_ci_smoke_job_runs_tests_pipeline():
    ci = (TESTS.parent / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pytest tests/pipeline/ -m pipeline_smoke" in ci


def test_every_pipeline_smoke_test_lives_under_tests_pipeline():
    files = [p for p in TESTS.rglob("*.py") if p.resolve() != Path(__file__).resolve()]
    assert files, "scanned no test files"
    marked = [p for p in files if MARK.search(p.read_text(encoding="utf-8"))]
    assert any(p.parent.name == "pipeline" for p in marked), "found no pipeline_smoke test at all"
    outside = sorted(str(p.relative_to(TESTS)) for p in marked if (TESTS / "pipeline") not in p.parents)
    assert outside == [], f"pipeline_smoke tests outside tests/pipeline/ run in no CI job: {outside}"
