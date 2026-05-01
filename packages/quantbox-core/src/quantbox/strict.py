from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    details: Mapping[str, Any] = field(default_factory=dict)
    message: str | None = None


class CapabilityChecker(Protocol):
    name: str

    def check(self, plugin: Any, run_ctx: Any) -> CheckResult: ...


CAPABILITY_REGISTRY: dict[str, CapabilityChecker] = {}


def register_capability(name: str, checker: CapabilityChecker) -> None:
    CAPABILITY_REGISTRY[name] = checker


def get_capability(name: str) -> CapabilityChecker | None:
    return CAPABILITY_REGISTRY.get(name)


# --- Built-in checkers ---


class _ManifestHashChecker:
    name = "manifest_hash"

    def check(self, plugin, run_ctx) -> CheckResult:
        try:
            h = plugin.manifest_hash()
        except Exception as exc:
            return CheckResult(name=self.name, passed=False, message=f"hash_error:{exc}")
        return CheckResult(name=self.name, passed=bool(h), details={"sha256": h})


class _CoverageReportChecker:
    name = "coverage_report"

    def check(self, plugin, run_ctx) -> CheckResult:
        cov = plugin.coverage_report()
        if cov is None:
            return CheckResult(name=self.name, passed=False, message="coverage_report missing")
        return CheckResult(name=self.name, passed=True)


class _CoveragePolicyChecker:
    name = "coverage_policy"

    def check(self, plugin, run_ctx) -> CheckResult:
        try:
            extras = plugin.manifest().extras
        except Exception as exc:
            return CheckResult(name=self.name, passed=False, message=f"manifest_error:{exc}")
        ok = bool(extras.get("coverage_policy_pass"))
        return CheckResult(name=self.name, passed=ok)


class _InstrumentRegistryChecker:
    name = "instrument_registry"

    def check(self, plugin, run_ctx) -> CheckResult:
        extras = plugin.manifest().extras
        sha = extras.get("instrument_registry_sha256")
        return CheckResult(
            name=self.name,
            passed=bool(sha),
            details={"sha256": sha} if sha else {},
            message=None if sha else "instrument_registry hash missing",
        )


class _BytesPinnedChecker:
    name = "bytes_pinned"

    def check(self, plugin, run_ctx) -> CheckResult:
        extras = plugin.manifest().extras
        files = extras.get("data_files") or {}
        ok = bool(files) and all(files.values())
        return CheckResult(name=self.name, passed=ok, details={"count": len(files)})


for _checker in (
    _ManifestHashChecker(),
    _CoverageReportChecker(),
    _CoveragePolicyChecker(),
    _InstrumentRegistryChecker(),
    _BytesPinnedChecker(),
):
    register_capability(_checker.name, _checker)
