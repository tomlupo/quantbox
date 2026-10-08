"""metrics describes, inference tests, gates decide (TOM-1618): one name, one meaning, old paths still work.

- Every import path that existed before TOM-1618 still resolves, to the SAME
  object as the new path, with a DeprecationWarning naming the new home.
- No two public functions of metrics / inference / gates share a name with a
  different meaning (the defect: ``gates.max_drawdown`` was positive,
  ``metrics.max_drawdown`` negative).
- One refusal type: ``GateInputError`` IS ``InferenceInputError``, a ValueError.
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

from quantbox import gates, inference, metrics

# (old module, name, new module) — every name an old path exported before TOM-1618.
_MOVED = [
    *(
        ("quantbox.metrics", name, "quantbox.inference")
        for name in ("newey_west_tstat", "newey_west_auto_lags", "hac_ols", "require_finite")
    ),
    *(
        ("quantbox.analysis.dsr", name, "quantbox.inference")
        for name in (
            "DEGENERATE_RTOL",
            "DSRResult",
            "EULER_MASCHERONI",
            "deflated_sharpe_ratio",
            "deflated_sharpe_ratio_from_returns",
            "expected_max_sr",
            "sr_estimator_std",
        )
    ),
    *(
        ("quantbox.analysis.hac", name, "quantbox.inference")
        for name in ("factor_regression", "newey_west_auto_lags", "newey_west_tstat", "require_finite")
    ),
    *(
        ("quantbox.analysis", name, "quantbox.inference")
        for name in (
            "DEGENERATE_RTOL",
            "DSRResult",
            "deflated_sharpe_ratio",
            "deflated_sharpe_ratio_from_returns",
            "expected_max_sr",
            "factor_regression",
            "newey_west_auto_lags",
            "newey_west_tstat",
            "require_finite",
            "sr_estimator_std",
        )
    ),
    *(
        ("quantbox.analysis.gates", name, "quantbox.gates")
        for name in (
            "DEFAULT_N_TRIALS",
            "GateInputError",
            "dsr_gate",
            "dsr_gate_from_returns",
            "episode_gate",
            "factor_gate",
            "nw_gate",
            "paired_block_bootstrap",
            "parse_n_trials",
        )
    ),
    ("quantbox.analysis.gates", "stationary_bootstrap_indices", "quantbox.inference"),
    ("quantbox.analysis.gates", "newey_west_tstat", "quantbox.inference"),
]


@pytest.mark.parametrize(("old", "name", "new"), _MOVED)
def test_an_old_path_resolves_to_the_new_object_with_a_deprecation_warning(old, name, new):
    module = importlib.import_module(old)
    with pytest.warns(DeprecationWarning, match=new.replace(".", r"\.")):
        obj = getattr(module, name)
    assert obj is getattr(importlib.import_module(new), name)


def test_an_unknown_name_on_an_old_path_is_still_an_attribute_error():
    import quantbox.analysis.dsr as old

    with pytest.raises(AttributeError):
        old.no_such_name  # noqa: B018


def test_the_old_gates_path_keeps_its_positive_drawdowns_with_a_warning():
    import quantbox.analysis.gates as old

    r = np.array([0.10, -0.10, -0.10, 0.05, 0.20, 0.01, -0.05, 0.02])
    with pytest.warns(DeprecationWarning, match="NEGATIVE"):
        mdd = old.max_drawdown(r)
    assert mdd == -metrics.max_drawdown(r, start_is_peak=True) > 0
    with pytest.warns(DeprecationWarning, match="NEGATIVE"):
        ep = old.largest_drawdown_episode(r)
    assert ep["depth"] == -inference.largest_drawdown_episode(r)["depth"] > 0


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
