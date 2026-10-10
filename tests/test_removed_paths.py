"""Every import path removed in 0.13.0 raises ImportError naming its replacement (TOM-1457).

The shims of TOM-1618, 4a (TOM-1449) and 4b (TOM-1451) resolved an old path to
the new object with a DeprecationWarning for one minor version. 4d removes
them. ``quantbox._removed.REMOVED`` is the one list the code reads; the lists
below are this test's own statement of what 4d removes, so a path dropped from
the table, or a shim left behind, fails here.
"""

from __future__ import annotations

import importlib

import pytest

from quantbox._removed import REMOVED

# Removed MODULES: each is a tombstone that raises at import.
REMOVED_MODULES = {
    "quantbox.analysis": ("quantbox.inference", "quantbox.gates", "quantbox.sweep"),
    "quantbox.plugins.strategies._universe": ("quantbox.universe",),
    "quantbox.plugins.backtesting.rsims_engine": ("quantbox.engine.rsims_sim",),
    "quantbox.simulation.models": ("quantbox.montecarlo.models",),
    "quantbox.simulation.engine": ("quantbox.montecarlo.engine",),
    "quantbox.simulation.correlation": ("quantbox.montecarlo.correlation",),
    "quantbox.simulation.stress_testing": ("quantbox.montecarlo.stress_testing",),
}

# Removed NAMES in modules that stay: (module, name) -> the replacement, module.attr.
REMOVED_NAMES = {
    ("quantbox.metrics", "hac_ols"): "quantbox.inference.hac_ols",
    ("quantbox.metrics", "newey_west_auto_lags"): "quantbox.inference.newey_west_auto_lags",
    ("quantbox.metrics", "newey_west_tstat"): "quantbox.inference.newey_west_tstat",
    ("quantbox.metrics", "require_finite"): "quantbox.inference.require_finite",
    ("quantbox.plugins.datasources._utils", "MarketCapProvider"): "quantbox.market_cap.MarketCapProvider",
    ("quantbox.plugins.datasources._utils", "CMCMarketCapProvider"): "quantbox.market_cap.MarketCapProvider",
    ("quantbox.plugins.datasources._utils", "load_pit_market_cap"): "quantbox.market_cap.load_pit_market_cap",
    (
        "quantbox.plugins.datasources.local_file_data",
        "_load_pinned_dataset",
    ): "quantbox.dataset.load_pinned_dataset",
}

# Imports a user had before 0.13.0 that reached the analysis package's old submodules.
_OLD_ANALYSIS_IMPORTS = [
    "import quantbox.analysis.dsr",
    "from quantbox.analysis.hac import newey_west_tstat",
    "from quantbox.analysis.gates import dsr_gate",
    "from quantbox.analysis.parameter_grid import run_grid",
    "from quantbox.analysis import deflated_sharpe_ratio_from_returns",
    "from quantbox import analysis",
]


def _resolve(dotted: str) -> object:
    module, _, attr = dotted.rpartition(".")
    try:
        return importlib.import_module(dotted)
    except ImportError:
        return getattr(importlib.import_module(module), attr)


def test_the_table_lists_exactly_what_4d_removes():
    expected = set(REMOVED_MODULES) | {f"{m}.{n}" for m, n in REMOVED_NAMES}
    assert set(REMOVED) == expected


@pytest.mark.parametrize("old", sorted(REMOVED_MODULES))
def test_a_removed_module_raises_import_error_naming_its_replacement(old):
    with pytest.raises(ImportError) as exc:
        importlib.import_module(old)
    msg = str(exc.value)
    assert f"{old} was removed in quantbox 0.13.0 (TOM-1457)" in msg
    for new in REMOVED_MODULES[old]:
        assert new in msg
    assert not isinstance(exc.value, ModuleNotFoundError)  # the tombstone ran, not a missing file


@pytest.mark.parametrize("old", sorted(REMOVED_MODULES))
def test_a_from_import_of_a_removed_module_raises_too(old):
    parent, _, leaf = old.rpartition(".")
    with pytest.raises(ImportError, match=r"removed in quantbox 0\.13\.0"):
        exec(f"from {parent} import {leaf}", {})


@pytest.mark.parametrize("statement", _OLD_ANALYSIS_IMPORTS)
def test_every_old_analysis_import_names_all_three_homes(statement):
    with pytest.raises(ImportError) as exc:
        exec(statement, {})
    for new in ("quantbox.inference", "quantbox.gates", "quantbox.sweep", "start_is_peak=True"):
        assert new in str(exc.value)


@pytest.mark.parametrize(("module", "name"), sorted(REMOVED_NAMES))
def test_a_removed_name_raises_import_error_naming_its_replacement(module, name):
    new = REMOVED_NAMES[(module, name)]
    with pytest.raises(ImportError) as exc:
        exec(f"from {module} import {name}", {})
    assert f"{module}.{name} was removed in quantbox 0.13.0 (TOM-1457): use {new}" in str(exc.value)
    with pytest.raises(ImportError, match=new.replace(".", r"\.")):
        getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("module", sorted({m for m, _ in REMOVED_NAMES}))
def test_an_unknown_name_on_a_live_module_is_still_an_attribute_error(module):
    mod = importlib.import_module(module)
    with pytest.raises(AttributeError):
        mod.no_such_name  # noqa: B018
    assert not hasattr(mod, "no_such_name")


@pytest.mark.parametrize("new", sorted({*REMOVED_NAMES.values(), *(n for v in REMOVED_MODULES.values() for n in v)}))
def test_every_replacement_resolves(new):
    assert _resolve(new) is not None


def test_the_simulation_package_still_re_exports_the_kernels():
    """``from quantbox.simulation import GBM`` was never deprecated and still works."""
    import quantbox.montecarlo as mc
    import quantbox.simulation as sim

    for name in mc.__all__:
        assert getattr(sim, name) is getattr(mc, name)
