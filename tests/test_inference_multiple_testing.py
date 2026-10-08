"""``multiple_testing`` and ``bonferroni_alpha``: family-wise corrections over m tests (TOM-1646).

Bonferroni runs each test at ``alpha / m``. Holm (1979) steps down through the
sorted p-values against ``alpha / (m - k + 1)`` and rejects at least what
Bonferroni rejects. The known answers below are worked by hand, not taken from
the library the helper wraps. ADR-0009, amendment of 2026-10-08 (TOM-1646).
"""

from __future__ import annotations

import numpy as np
import pytest

from quantbox.inference import InferenceInputError, MultipleTestingResult, bonferroni_alpha, multiple_testing

# The Holm (1979) textbook example: four hypotheses, alpha = 0.05.
# Sorted p: 0.005, 0.01, 0.03, 0.04 against 0.0125, 0.0167, 0.025, 0.05.
# The third (0.03 > 0.025) stops the step-down: H4 and H1 are rejected, H2 and H3 not.
TEXTBOOK = [0.01, 0.04, 0.03, 0.005]


def test_holm_textbook_example():
    out = multiple_testing(TEXTBOOK, method="holm")
    assert out.reject.tolist() == [True, False, False, True]
    # (m - k + 1) * p_(k) on the sorted p: 0.02, 0.03, 0.06, 0.04 -> running max 0.02, 0.03, 0.06, 0.06
    np.testing.assert_allclose(out.adjusted, [0.03, 0.06, 0.06, 0.02], rtol=1e-12)
    assert out.method == "holm" and out.m == 4 and out.alpha == 0.05
    assert out.per_test_alpha is None  # Holm's level depends on the rank: no single CI level


def test_bonferroni_textbook_example():
    out = multiple_testing(TEXTBOOK, method="bonferroni")
    assert out.reject.tolist() == [True, False, False, True]
    np.testing.assert_allclose(out.adjusted, [0.04, 0.16, 0.12, 0.02], rtol=1e-12)
    assert out.per_test_alpha == pytest.approx(0.0125, rel=1e-15)


def test_bonferroni_is_the_default_method():
    assert multiple_testing(TEXTBOOK).method == "bonferroni"


def test_holm_rejects_what_bonferroni_misses():
    """p = 0.005, 0.011, 0.02, 0.04: Bonferroni (level 0.0125) rejects two, Holm all four."""
    p = [0.04, 0.005, 0.02, 0.011]
    bonf = multiple_testing(p, method="bonferroni")
    holm = multiple_testing(p, method="holm")
    assert bonf.reject.tolist() == [False, True, False, True]
    assert holm.reject.tolist() == [True, True, True, True]
    # sorted: 4*0.005, 3*0.011, 2*0.02, 1*0.04 = 0.02, 0.033, 0.04, 0.04 (already monotone)
    np.testing.assert_allclose(holm.adjusted, [0.04, 0.02, 0.04, 0.033], rtol=1e-12)


def test_adjusted_pvalues_are_capped_at_one():
    out = multiple_testing([0.5, 0.9, 0.2], method="bonferroni")
    np.testing.assert_allclose(out.adjusted, [1.0, 1.0, 0.6], rtol=1e-12)
    assert not out.reject.any()


def test_a_pvalue_on_the_level_is_rejected():
    """The rule is p <= level, so a p-value exactly on the Bonferroni level is a rejection."""
    out = multiple_testing([0.025, 0.5], alpha=0.05, method="bonferroni")
    assert out.reject.tolist() == [True, False]


def test_one_test_is_no_correction():
    for method in ("bonferroni", "holm"):
        out = multiple_testing([0.03], method=method)
        assert out.adjusted.tolist() == [0.03] and out.reject.tolist() == [True]


def test_reject_is_adjusted_at_most_alpha():
    rng = np.random.default_rng(1646)
    p = rng.uniform(0, 0.1, 40)
    for method in ("bonferroni", "holm"):
        out = multiple_testing(p, alpha=0.05, method=method)
        assert (out.reject == (out.adjusted <= 0.05)).all()
        assert (out.adjusted >= p).all()


def test_result_keeps_the_input_order_and_the_raw_pvalues():
    out = multiple_testing(np.array(TEXTBOOK), method="holm")
    assert isinstance(out, MultipleTestingResult)
    assert out.pvalues.tolist() == TEXTBOOK


# --- the per-test level (a CI at 1 - alpha/m, active_etfs) ---


def test_bonferroni_alpha_is_alpha_over_m():
    assert bonferroni_alpha(9) == pytest.approx(0.05 / 9, rel=1e-15)
    assert bonferroni_alpha(4, alpha=0.10) == pytest.approx(0.025, rel=1e-15)
    assert bonferroni_alpha(1) == 0.05


def test_bonferroni_alpha_covers_the_active_etfs_ci_level():
    """robo-lab active_etfs: ``q = 100 * 0.05 / n_tests / 2``, the lower percentile of a two-sided CI."""
    n_tests = 9
    assert 100 * bonferroni_alpha(n_tests) / 2 == pytest.approx(100 * 0.05 / n_tests / 2, rel=1e-15)


@pytest.mark.parametrize("bad", [0, -1, 2.0, True, None])
def test_bonferroni_alpha_refuses_a_test_count_that_is_not_a_whole_number_ge_1(bad):
    with pytest.raises(InferenceInputError, match="n_tests"):
        bonferroni_alpha(bad)


# --- refusals ---


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.05, 1.5, float("nan")])
def test_alpha_outside_the_open_unit_interval_is_refused(alpha):
    with pytest.raises(InferenceInputError, match="alpha"):
        multiple_testing(TEXTBOOK, alpha=alpha)
    with pytest.raises(InferenceInputError, match="alpha"):
        bonferroni_alpha(3, alpha=alpha)


def test_unknown_method_is_refused():
    with pytest.raises(InferenceInputError, match="method"):
        multiple_testing(TEXTBOOK, method="bh")


def test_empty_family_is_refused():
    with pytest.raises(InferenceInputError, match="at least one"):
        multiple_testing([])


def test_nan_pvalue_is_refused_not_dropped():
    """Dropping a NaN p-value would shrink m and loosen every other test's level."""
    with pytest.raises(InferenceInputError, match="NaN"):
        multiple_testing([0.01, float("nan"), 0.03])


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_pvalue_outside_the_unit_interval_is_refused(bad):
    with pytest.raises(InferenceInputError, match=r"\[0, 1\]"):
        multiple_testing([0.01, bad])


def test_a_two_dimensional_family_is_refused():
    with pytest.raises(InferenceInputError, match="1-D"):
        multiple_testing([[0.01, 0.02], [0.03, 0.04]])
