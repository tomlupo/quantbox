from dataclasses import dataclass
from typing import Any

from quantbox.dataset import CoverageReport, DatasetManifest
from quantbox.runner import _dataset_block, _run_capability_checks


@dataclass
class _StubPlugin:
    dataset_id: str = "fake"
    dataset_version: str = "1.0.0"
    capabilities: tuple[str, ...] = ("manifest_hash", "coverage_report")
    meta: Any = None

    def manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="fake",
            version="1.0.0",
            date_range={"start": "2026-01-01", "end": "2026-04-01"},
            symbols_count=1,
            data_fields=("prices",),
            extras={"data_files": {}, "instrument_registry_sha256": None, "coverage_policy_pass": True},
        )

    def manifest_hash(self):
        return "deadbeef" * 8

    def coverage_report(self):
        return CoverageReport(per_symbol={}, per_field={}, overall={})


def test_dataset_block_for_plugin_includes_identity_and_capabilities():
    block = _dataset_block(_StubPlugin())
    assert block["tier"] == "plugin"
    assert block["id"] == "fake"
    assert block["version"] == "1.0.0"
    assert "manifest_hash" in block["capabilities_declared"]


def test_dataset_block_for_raw_data_marks_tier_raw():
    block = _dataset_block(None)
    assert block["tier"] == "raw"
    assert "warning" in block


def test_run_capability_checks_returns_per_capability_results():
    results = _run_capability_checks(_StubPlugin(), run_ctx=None)
    assert "manifest_hash" in results
    assert results["manifest_hash"]["passed"] is True
    assert results["coverage_report"]["passed"] is True


def test_run_capability_checks_records_unknown_capability_as_warning():
    plug = _StubPlugin(capabilities=("manifest_hash", "no_such_capability"))
    results = _run_capability_checks(plug, run_ctx=None)
    assert results["no_such_capability"]["passed"] is False
    assert "unknown_capability" in (results["no_such_capability"].get("message") or "")
