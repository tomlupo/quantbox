import pytest

from quantbox.strict import (
    CAPABILITY_REGISTRY,
    CheckResult,
    get_capability,
    register_capability,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    snapshot = dict(CAPABILITY_REGISTRY)
    yield
    CAPABILITY_REGISTRY.clear()
    CAPABILITY_REGISTRY.update(snapshot)


def test_check_result_fields():
    r = CheckResult(name="foo", passed=True, details={"a": 1})
    assert r.name == "foo" and r.passed is True and r.details == {"a": 1}


def test_register_and_get_capability():
    class Dummy:
        name = "dummy"

        def check(self, plugin, run_ctx):
            return CheckResult(name="dummy", passed=True)

    register_capability("dummy", Dummy())
    assert get_capability("dummy") is not None
    assert get_capability("dummy").check(None, None).passed is True


def test_get_unknown_capability_returns_none():
    assert get_capability("nonexistent") is None
