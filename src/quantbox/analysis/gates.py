"""Deprecated path: the acceptance gates live in :mod:`quantbox.gates` (TOM-1618).

Every gate name resolves to the same object in :mod:`quantbox.gates`, and the
statistics this module used to export (``stationary_bootstrap_indices``,
``newey_west_tstat``, ...) to :mod:`quantbox.inference`, with a
``DeprecationWarning``.

Two names keep their OLD meaning here, because a caller of this path was
promised it: ``max_drawdown`` (a POSITIVE fraction — :func:`quantbox.metrics.max_drawdown`
is the package's one drawdown and is negative) and ``largest_drawdown_episode``
(``depth`` positive — :func:`quantbox.inference.largest_drawdown_episode` reports
it negative). Neither exists under the new paths with this sign.
"""

from __future__ import annotations

import warnings

from quantbox._deprecation import moved

_forward = moved(__name__, "quantbox.gates", "quantbox.inference")


def _old_max_drawdown(returns):
    """Largest peak-to-trough loss as a POSITIVE fraction, the starting equity 1.0 a peak (pre-TOM-1618)."""
    from quantbox.metrics import max_drawdown

    return -max_drawdown(returns, start_is_peak=True)


def _old_largest_drawdown_episode(returns):
    """:func:`quantbox.inference.largest_drawdown_episode` with ``depth`` positive (pre-TOM-1618)."""
    from quantbox.inference import largest_drawdown_episode

    ep = largest_drawdown_episode(returns)
    return None if ep is None else {**ep, "depth": -ep["depth"]}


_OLD_MEANING = {
    "max_drawdown": (_old_max_drawdown, "quantbox.metrics.max_drawdown(r, start_is_peak=True), which is NEGATIVE"),
    "largest_drawdown_episode": (
        _old_largest_drawdown_episode,
        "quantbox.inference.largest_drawdown_episode, whose depth is NEGATIVE",
    ),
}


def __getattr__(name: str):
    if name in _OLD_MEANING:
        fn, new = _OLD_MEANING[name]
        warnings.warn(
            f"quantbox.analysis.gates.{name} is deprecated (positive drawdown): use {new} (TOM-1618)",
            DeprecationWarning,
            stacklevel=2,
        )
        return fn
    return _forward(name)
