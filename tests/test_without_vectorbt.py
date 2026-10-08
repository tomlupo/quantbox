"""quantbox imports and runs without the ``[vectorbt]`` extra (TOM-1334).

The dev env has vectorbt installed, so each check runs in a SUBPROCESS whose
import system refuses ``vectorbt`` and ``numba`` — the two packages the
``[vectorbt]`` extra carries. That reproduces a base install without building
one; the clean-venv CI job (``ci.yml`` → ``no-vectorbt``) proves the real thing.

Every check first asserts the block is live (``import vectorbt`` fails), so a
broken harness reads red, never as a vacuous green.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

# Modules whose whole job IS the vectorbt engine: these may not import without
# the extra, and must say which extra to install when they refuse.
ENGINE_MODULES = {
    "quantbox.adapters.vectorbt",
    "quantbox.plugins.backtesting.vectorbt_engine",
}

_BLOCK = textwrap.dedent(
    """
    import importlib.abc, sys

    class _Block(importlib.abc.MetaPathFinder):
        BLOCKED = __BLOCKED__

        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in self.BLOCKED:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    for _m in list(sys.modules):
        if _m.split(".")[0] in _Block.BLOCKED:
            del sys.modules[_m]
    sys.meta_path.insert(0, _Block())

    _probe = sorted(_Block.BLOCKED)[0]
    try:
        __import__(_probe)
    except ModuleNotFoundError:
        pass
    else:
        raise SystemExit(f"HARNESS BROKEN: {_probe} still importable")
    """
)


def _run(body: str, blocked: tuple[str, ...] = ("vectorbt", "numba")) -> subprocess.CompletedProcess:
    code = _BLOCK.replace("__BLOCKED__", repr(set(blocked))) + textwrap.dedent(body)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert "HARNESS BROKEN" not in proc.stdout + proc.stderr, proc.stderr
    return proc


def test_every_non_engine_module_imports_without_vectorbt():
    proc = _run(
        """
        import importlib, json, pkgutil, traceback
        import quantbox

        ok, failed = [], {}
        for info in pkgutil.walk_packages(quantbox.__path__, "quantbox."):
            try:
                importlib.import_module(info.name)
                ok.append(info.name)
            except BaseException as exc:  # noqa: BLE001 - report every refusal
                failed[info.name] = f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
        print(json.dumps({"ok": ok, "failed": failed}))
        """
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    # Count what was looked at: an empty walk would pass the assertions below.
    assert len(result["ok"]) > 100, result

    unexpected = {m: e for m, e in result["failed"].items() if m not in ENGINE_MODULES}
    assert unexpected == {}, unexpected

    # The engine modules must refuse, and refuse by naming the extra.
    for mod in ENGINE_MODULES:
        err = result["failed"].get(mod)
        assert err is not None, f"{mod} imported without vectorbt — the block is not reaching it"
        assert "quantbox.exceptions.MissingExtraError" in err, err
        assert "quantbox[vectorbt]" in err, err


def test_cli_plugins_list_without_vectorbt():
    proc = _run(
        """
        import sys
        sys.argv = ["quantbox", "plugins", "list", "--json"]
        from quantbox.cli import main
        try:
            main()
        except SystemExit as e:
            if e.code not in (0, None):
                raise
        """
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout[proc.stdout.index("{") :])
    assert "backtest.pipeline.v1" in payload.get("pipelines", []), payload


@pytest.mark.parametrize(
    "snippet",
    [
        "from quantbox.plugins.backtesting.vectorbt_engine import run",
        "import quantbox.plugins.backtesting as b; b.backtest(None, None)",
        "from quantbox.adapters.vectorbt import vbt",
        "import pandas as pd, quantbox.bt as qbt; qbt.run(pd.DataFrame(), pd.DataFrame())",
        "from quantbox.sweep import run_grid; run_grid(None, {}, {}, {})",
        "from quantbox.analysis import run_grid; run_grid(None, {}, {}, {})",  # the deprecated path
    ],
)
def test_asking_for_vectorbt_names_the_extra(snippet: str):
    proc = _run(
        f"""
        from quantbox.exceptions import MissingExtraError
        try:
            {snippet}
        except MissingExtraError as exc:
            assert isinstance(exc, ImportError)
            assert "quantbox[vectorbt]" in str(exc), str(exc)
            print("NAMED-EXTRA")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "NAMED-EXTRA" in proc.stdout, proc.stdout + proc.stderr


def test_broken_vectorbt_install_is_not_reported_as_missing_extra():
    """vectorbt present but one of ITS dependencies missing is a broken install:
    the real ModuleNotFoundError must surface, not "install the extra"."""
    proc = _run(
        """
        from quantbox.exceptions import MissingExtraError
        try:
            import quantbox.plugins.backtesting.vectorbt_engine  # noqa: F401
        except MissingExtraError as exc:
            raise SystemExit(f"WRONG: broken install reported as missing extra: {exc}")
        except ModuleNotFoundError as exc:
            assert exc.name.split(".")[0] == "plotly", exc
            print("REAL-ERROR")
        """,
        blocked=("plotly",),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "REAL-ERROR" in proc.stdout, proc.stdout + proc.stderr
