"""The framework's "cancelled to floating-point noise" tests (internal).

:mod:`quantbox.metrics` (a statistic with no answer is NaN) and
:mod:`quantbox.inference` (an input that cannot produce a statistic is refused)
both ask the same question of a series: is its spread real, or rounding noise?
The pipelines ask it of a weight: is it a position, or optimizer dust? They ask
here, so each threshold and test exists once.
"""

from __future__ import annotations

import numpy as np

# A quantity that is mathematically zero does not reliably come out as 0.0 in
# binary floating point. A constant returns series is the canonical example:
# `[0.001] * 200` accumulates rounding to std = 2.17e-19 while `[0.001] * 50`
# gives exactly 0.0, so whether an `== 0` guard fires is a lottery on the
# (value, length) pair rather than a property of the input. Measured on the DSR
# module before the fix: of 32 constant series (8 values x 4 lengths), 20 hit
# the exact guard and 12 sailed past it into the moment path, where scipy hit
# catastrophic cancellation.
#
# The observed noise floor for a constant series is std/|value| ~ 2e-16
# (machine epsilon); 1e-12 leaves ~4000x headroom above it while staying far
# below any real series (std/scale = 1e-12 would imply a Sharpe of ~1e12).
# Being relative, the test is unit-independent: a genuinely tiny-but-real
# series (returns of order 1e-9 with std of order 1e-9) is unaffected. When
# every observation is exactly zero, scale is 0 and the test reduces to
# std <= 0, which still holds.
DEGENERATE_RTOL = 1e-12


# A portfolio weight within this of zero is no position: optimizer dust. A
# solver that lands on the zero bound returns 1e-15-ish, not 0.0, and which
# symbols carry dust moves with the BLAS build (numpy 2.3.5 vs 2.5.4 swapped
# SYN_001 for SYN_004 in the synthetic backtest, #331). The engine already
# treats weights this way: engine/schedule.py compares books "within 1e-12",
# and engine/rsims_sim.py trades no change below 1e-12 absolute. Weight lists
# that drop zeros (aggregated_weights, trade targets) use the same number.
WEIGHT_ATOL = 1e-12


def is_held(weight: float) -> bool:
    """A weight that is a position: NOT within WEIGHT_ATOL of zero. A NaN weight is kept, as ``!= 0`` kept it."""
    return not abs(weight) <= WEIGHT_ATOL


def flat(x, *, ddof: int = 0) -> bool:
    """No spread worth the name: ``std(x, ddof)`` within DEGENERATE_RTOL of ``mean(|x|)`` (never ``== 0``)."""
    a = np.asarray(x, dtype=float)
    return bool(np.std(a, ddof=ddof) <= DEGENERATE_RTOL * float(np.mean(np.abs(a))))
