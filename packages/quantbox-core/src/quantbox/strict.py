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
