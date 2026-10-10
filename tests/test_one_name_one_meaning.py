"""metrics describes, inference tests, gates decide (TOM-1618): one name, one meaning.

- No two public functions of metrics / inference / gates share a name with a
  different meaning (the defect: ``gates.max_drawdown`` was positive,
  ``metrics.max_drawdown`` negative).
- One refusal type: ``GateInputError`` IS ``InferenceInputError``, a ValueError.

The old import paths of TOM-1618 (``quantbox.analysis``, the moved
``quantbox.metrics`` names) were shims until 0.13.0; their removal is checked
in ``tests/test_removed_paths.py`` (TOM-1457).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from quantbox import gates, inference, metrics


def test_one_refusal_type_and_it_is_a_value_error():
    assert gates.GateInputError is inference.InferenceInputError
    assert issubclass(inference.InferenceInputError, ValueError)
    with pytest.raises(gates.GateInputError):
        inference.require_finite([0.01, float("nan")])
    with pytest.raises(ValueError):  # what every caller before TOM-1618 caught
        inference.deflated_sharpe_ratio_from_returns([0.001] * 50, 5)


def _public_functions(module) -> dict[str, object]:
    return {
        name: obj
        for name, obj in vars(module).items()
        if not name.startswith("_") and (inspect.isfunction(obj) or inspect.isclass(obj))
    }


def test_no_public_name_means_two_things_across_metrics_inference_and_gates():
    """A name public in two of the three modules must be the same object (one meaning)."""
    modules = [metrics, inference, gates]
    seen: dict[str, object] = {}
    clashes = []
    for module in modules:
        for name, obj in _public_functions(module).items():
            if name in seen and seen[name] is not obj:
                clashes.append(f"{name}: {seen[name]!r} vs {obj!r}")
            seen.setdefault(name, obj)
    assert not clashes, clashes
    assert not hasattr(gates, "max_drawdown")


def test_drawdowns_are_negative_on_every_new_path():
    r = np.array([0.10, -0.10, -0.10, 0.05, 0.20, 0.01, -0.05, 0.02])
    assert metrics.max_drawdown(r) < 0
    assert metrics.max_drawdown(r, start_is_peak=True) < 0
    assert inference.largest_drawdown_episode(r)["depth"] < 0
