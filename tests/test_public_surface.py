"""The declared public surface (TOM-1449): four private names got public homes, old paths still work.

- ``quantbox.universe``: ``select_universe``, ``DEFAULT_STABLECOINS`` (was
  ``quantbox.plugins.strategies._universe``).
- ``quantbox.market_cap``: ``MarketCapProvider``, ``load_pit_market_cap`` (was
  ``quantbox.plugins.datasources._utils``).
- ``quantbox.dataset.load_pinned_dataset`` (was
  ``quantbox.plugins.datasources.local_file_data._load_pinned_dataset``).

Every old path resolves to the SAME object as the new one and emits a
DeprecationWarning naming the new path. The package itself never takes an old
path, so importing it emits none. pyproject's ``filterwarnings`` ignores
DeprecationWarning, so every check here catches warnings explicitly.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
import warnings

import pytest

# (old module, old name, new module, new name)
_MOVED = [
    *(
        ("quantbox.plugins.strategies._universe", name, "quantbox.universe", name)
        for name in (
            "DEFAULT_STABLECOINS",
            "DUCKDB_AVAILABLE",
            "select_universe",
            "select_universe_duckdb",
            "select_universe_vectorized",
        )
    ),
    ("quantbox.plugins.datasources._utils", "MarketCapProvider", "quantbox.market_cap", "MarketCapProvider"),
    ("quantbox.plugins.datasources._utils", "CMCMarketCapProvider", "quantbox.market_cap", "MarketCapProvider"),
    ("quantbox.plugins.datasources._utils", "load_pit_market_cap", "quantbox.market_cap", "load_pit_market_cap"),
    (
        "quantbox.plugins.datasources.local_file_data",
        "_load_pinned_dataset",
        "quantbox.dataset",
        "load_pinned_dataset",
    ),
]


@pytest.mark.parametrize(("old", "name", "new", "new_name"), _MOVED)
def test_an_old_path_resolves_to_the_new_object_with_a_deprecation_warning(old, name, new, new_name):
    module = importlib.import_module(old)
    pattern = re.escape(f"{old}.{name} is deprecated: ") + ".*" + re.escape(new)
    with pytest.warns(DeprecationWarning, match=pattern):
        obj = getattr(module, name)
    assert obj is getattr(importlib.import_module(new), new_name)


def test_a_from_import_of_an_old_path_works_and_warns():
    with pytest.warns(DeprecationWarning, match=r"quantbox\.universe"):
        from quantbox.plugins.strategies._universe import select_universe
    from quantbox.universe import select_universe as new

    assert select_universe is new


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("quantbox.plugins.datasources._utils", "no_such_name"),
        ("quantbox.plugins.datasources.local_file_data", "no_such_name"),
        ("quantbox.plugins.strategies._universe", "no_such_name"),
    ],
)
def test_an_unknown_name_on_an_old_path_is_still_an_attribute_error(module, name):
    with pytest.raises(AttributeError):
        getattr(importlib.import_module(module), name)


def test_names_that_did_not_move_still_import_from_the_old_module_without_a_warning():
    import quantbox.plugins.datasources._utils as utils

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert callable(utils.resolve_screen_inputs)
        assert callable(utils.validate_ohlcv)


@pytest.mark.parametrize(
    ("module", "names"),
    [
        ("quantbox.universe", ("select_universe", "DEFAULT_STABLECOINS")),
        ("quantbox.market_cap", ("MarketCapProvider", "load_pit_market_cap", "map_symbol")),
        ("quantbox.dataset", ("load_pinned_dataset",)),
    ],
)
def test_a_new_path_declares_its_names_and_reads_them_without_a_warning(module, names):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        mod = importlib.import_module(module)
        for name in names:
            assert name in mod.__all__
            getattr(mod, name)


_DECLARED = (
    "contracts",
    "registry",
    "runner",
    "metrics",
    "decision",
    "strategy_runner",
    "frequency",
    "store",
    "exceptions",
    "sweep",
    "universe",
    "market_cap",
    "dataset",
)


@pytest.mark.parametrize("name", _DECLARED)
def test_every_public_module_declares_all_and_every_name_resolves_without_a_warning(name):
    mod = importlib.import_module(f"quantbox.{name}")
    assert mod.__all__, f"quantbox.{name} declares no __all__"
    assert len(set(mod.__all__)) == len(mod.__all__)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for attr in mod.__all__:
            getattr(mod, attr)


# Imports every quantbox module and discovers every plugin with a quantbox-raised
# DeprecationWarning turned into an error. A fresh interpreter, so modules this
# test session already imported do not hide an import-time warning.
_IMPORT_ALL = r"""
import importlib, pkgutil, warnings
warnings.filterwarnings("error", category=DeprecationWarning, module=r"quantbox(\.|$)")
import quantbox
from quantbox.registry import PluginRegistry
n_plugins = sum(len(v) for v in vars(PluginRegistry.discover()).values() if isinstance(v, dict))
imported, skipped = 0, []
for info in pkgutil.walk_packages(quantbox.__path__, "quantbox."):
    try:
        importlib.import_module(info.name)
        imported += 1
    except DeprecationWarning:
        raise
    except Exception as exc:  # an optional dependency this environment lacks
        skipped.append(f"{info.name}: {type(exc).__name__}")
print(imported, n_plugins, skipped)
"""


def test_the_package_itself_takes_no_deprecated_path():
    proc = subprocess.run([sys.executable, "-c", _IMPORT_ALL], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    imported, n_plugins = (int(x) for x in proc.stdout.split()[:2])
    # A walk that imported nothing proves nothing.
    assert imported > 100 and n_plugins > 20, proc.stdout
