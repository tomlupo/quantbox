"""Optimizer dust is no position (replay of #331's before/after).

On #331 the synthetic backtest's optimizer left SYN_001 at 1.4e-15 on one numpy
build and SYN_004 at 4.5e-15 on another. aggregated_weights listed every weight
``!= 0``, so the book "changed" on float noise. A weight within WEIGHT_ATOL of
zero is now dropped from the lists that drop zeros; a NaN is kept, as before.
"""

from __future__ import annotations

import math

import pytest

from quantbox._numerics import WEIGHT_ATOL, is_held
from quantbox.plugins.pipeline.trading_pipeline import _target_dict


@pytest.mark.parametrize("w", [0.0, -0.0, 1.4e-15, -4.5e-15, WEIGHT_ATOL, -WEIGHT_ATOL])
def test_dust_is_not_held(w):
    assert not is_held(w)


@pytest.mark.parametrize("w", [1e-9, -1e-9, 2 * WEIGHT_ATOL, 0.24441975633504717, math.nan, math.inf])
def test_a_real_weight_and_nan_are_held(w):
    assert is_held(w)


def test_target_dict_drops_the_dust_rows_the_two_numpy_builds_disagreed_on():
    base = {"SYN_001": 1.4271957006440607e-15, "SYN_002": 0.24441975633504717, "SYN_004": 0.0, "SYN_010": 4.4e-17}
    head = {"SYN_001": 0.0, "SYN_002": 0.24441975543319996, "SYN_004": 4.5e-15, "SYN_010": 6.4e-16}
    assert list(_target_dict(base)) == list(_target_dict(head)) == ["SYN_002"]
