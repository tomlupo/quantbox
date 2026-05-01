import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

from quantbox.dataset import CoverageReport, DatasetManifest
from quantbox.strict import get_capability


@dataclass
class _FakePlugin:
    dataset_id: str = "fake"
    dataset_version: str = "1.0.0"
    capabilities: tuple[str, ...] = ()
    _manifest_bytes: bytes = b'{"name":"fake","version":"1.0.0"}'
    _coverage: CoverageReport | None = None
    _data_files: Mapping[str, str] = field(default_factory=dict)
    _registry_hash: str | None = None
    _coverage_policy_pass: bool = True

    def manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="fake",
            version="1.0.0",
            date_range={"start": "2026-01-01", "end": "2026-04-01"},
            symbols_count=3,
            data_fields=("prices",),
            extras={
                "data_files": dict(self._data_files),
                "instrument_registry_sha256": self._registry_hash,
                "coverage_policy_pass": self._coverage_policy_pass,
            },
        )

    def manifest_hash(self) -> str:
        return hashlib.sha256(self._manifest_bytes).hexdigest()

    def coverage_report(self) -> CoverageReport | None:
        return self._coverage


def test_manifest_hash_checker_passes_on_consistent_bytes():
    chk = get_capability("manifest_hash")
    p = _FakePlugin()
    r = chk.check(p, run_ctx=None)
    assert r.passed is True


def test_coverage_report_checker_fails_when_missing():
    chk = get_capability("coverage_report")
    p = _FakePlugin(_coverage=None)
    r = chk.check(p, run_ctx=None)
    assert r.passed is False


def test_coverage_report_checker_passes_when_present():
    chk = get_capability("coverage_report")
    p = _FakePlugin(_coverage=CoverageReport(per_symbol={}, per_field={}, overall={"max_gap_pct": 0.0}))
    assert chk.check(p, None).passed is True


def test_coverage_policy_uses_extras_flag():
    chk = get_capability("coverage_policy")
    assert chk.check(_FakePlugin(_coverage_policy_pass=True), None).passed is True
    assert chk.check(_FakePlugin(_coverage_policy_pass=False), None).passed is False


def test_instrument_registry_checker_requires_hash():
    chk = get_capability("instrument_registry")
    assert chk.check(_FakePlugin(_registry_hash=None), None).passed is False
    assert chk.check(_FakePlugin(_registry_hash="abc"), None).passed is True


def test_bytes_pinned_checker_requires_data_files_with_sha():
    chk = get_capability("bytes_pinned")
    assert chk.check(_FakePlugin(_data_files={}), None).passed is False
    assert chk.check(_FakePlugin(_data_files={"prices.parquet": "sha"}), None).passed is True
