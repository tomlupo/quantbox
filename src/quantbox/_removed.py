"""Import paths removed in quantbox 0.13.0 (TOM-1457, the contract step of ADR-0010).

4a and 4b (TOM-1449, TOM-1451) and TOM-1618 moved code to public homes and kept
each old path for one minor version, as a shim with a ``DeprecationWarning``.
0.13.0 removes those shims. An old path now raises ``ImportError`` and names
its replacement, so a stale import fails at once and says what to write.

:data:`REMOVED` is the one list: old path -> what to import instead. It is read
two ways:

- A removed MODULE keeps a tombstone file that calls :func:`removed` at import
  (``quantbox.analysis``, ``quantbox.simulation.models``, ...).
- A removed NAME in a module that is still live is refused by the module
  ``__getattr__`` that :func:`removed_names` returns. Every other missing name
  stays an ``AttributeError``.

The module walkers (``tests/test_base_install.py``,
``tests/test_without_vectorbt.py``, ``scripts/check_no_vectorbt.sh``) read this
list to tell a tombstone from a broken module; ``tests/test_removed_paths.py``
checks each entry.

YAML plugin-id aliases and call or config aliases are NOT import paths and are
not here: they keep working, with their warnings (Tom, 2026-10-09).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn

REMOVED_IN = "0.13.0"
CARD = "TOM-1457"

_INFERENCE = "quantbox.inference"

# Old path -> the replacement. A module path is a tombstone file; a dotted name
# under a live module is refused by that module's __getattr__.
REMOVED: dict[str, str] = {
    # TOM-1618: metrics describes, inference tests, gates decide (ADR-0009).
    "quantbox.analysis": (
        "quantbox.inference (was analysis.dsr, analysis.hac and the statistics), "
        "quantbox.gates (was analysis.gates; its POSITIVE max_drawdown is "
        "-quantbox.metrics.max_drawdown(r, start_is_peak=True)), "
        "quantbox.sweep (was analysis.parameter_grid and the sweep names)"
    ),
    "quantbox.metrics.hac_ols": _INFERENCE + ".hac_ols",
    "quantbox.metrics.newey_west_auto_lags": _INFERENCE + ".newey_west_auto_lags",
    "quantbox.metrics.newey_west_tstat": _INFERENCE + ".newey_west_tstat",
    "quantbox.metrics.require_finite": _INFERENCE + ".require_finite",
    # TOM-1449 (4a): the declared public surface.
    "quantbox.plugins.strategies._universe": "quantbox.universe",
    "quantbox.plugins.datasources._utils.MarketCapProvider": "quantbox.market_cap.MarketCapProvider",
    "quantbox.plugins.datasources._utils.CMCMarketCapProvider": "quantbox.market_cap.MarketCapProvider",
    "quantbox.plugins.datasources._utils.load_pit_market_cap": "quantbox.market_cap.load_pit_market_cap",
    "quantbox.plugins.datasources.local_file_data._load_pinned_dataset": "quantbox.dataset.load_pinned_dataset",
    # TOM-1451 (4b): the layer map (ADR-0010).
    "quantbox.plugins.backtesting.rsims_engine": "quantbox.engine.rsims_sim",
    "quantbox.simulation.models": "quantbox.montecarlo.models (or: from quantbox.simulation import ...)",
    "quantbox.simulation.engine": "quantbox.montecarlo.engine (or: from quantbox.simulation import ...)",
    "quantbox.simulation.correlation": "quantbox.montecarlo.correlation (or: from quantbox.simulation import ...)",
    "quantbox.simulation.stress_testing": "quantbox.montecarlo.stress_testing (or: from quantbox.simulation import ...)",
}


def _error(old: str) -> ImportError:
    return ImportError(f"{old} was removed in quantbox {REMOVED_IN} ({CARD}): use {REMOVED[old]}", name=old)


def removed(module: str) -> NoReturn:
    """Raise the ``ImportError`` for a removed module; its tombstone file calls this at import."""
    raise _error(module)


def removed_names(module: str) -> Callable[[str], Any]:
    """A module ``__getattr__`` that refuses each name :data:`REMOVED` lists under ``module``.

    It raises ``ImportError``, not ``AttributeError``: ``from module import name``
    turns an ``AttributeError`` into a generic "cannot import name" and drops the
    replacement. Any other missing name is the usual ``AttributeError``.
    """

    def __getattr__(name: str) -> Any:
        if f"{module}.{name}" in REMOVED:
            raise _error(f"{module}.{name}")
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    return __getattr__
