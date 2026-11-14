"""``sharpe_difference_test`` and ``sharpe_difference_gate``: the Ledoit-Wolf HAC Sharpe difference (TOM-3717).

The reference values are PUBLISHED: Ledoit & Wolf (2008), "Robust performance
hypothesis testing with the Sharpe ratio", Journal of Empirical Finance 15(5),
850-859, Section 6. Table 2 gives the per-period Sharpe of each fund, Table 3
the two-sided p-values (in percent, one decimal) of H0: equal Sharpe for the
columns HAC (QS kernel, Andrews 1991 bandwidth) and HACPW (prewhitened):

    mutual funds (Fidelity vs Fidelity Aggressive Growth): HAC 6.3, HACPW 6.7
    hedge funds (JMG Capital Partners vs Coast Enhanced Income): HAC 14.7, HACPW 25.4

The data are the authors' own (``tests/fixtures/ledoit_wolf_2008/returns.csv``,
whose header names the source). The paper's p-values are two-sided, so the
order of the pair does not move them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantbox.gates import GateInputError, sharpe_difference_gate
from quantbox.inference import InferenceInputError, sharpe_difference_test

LW = pd.read_csv(Path(__file__).parent / "fixtures" / "ledoit_wolf_2008" / "returns.csv", comment="#")
MUTUAL = (LW["fidelity"].to_numpy(), LW["fidelity_aggressive_growth"].to_numpy())
HEDGE = (LW["jmg_capital_partners"].to_numpy(), LW["coast_enhanced_income"].to_numpy())


@pytest.mark.parametrize(
    ("pair", "prewhiten", "published_pct"),
    [(MUTUAL, False, 6.3), (MUTUAL, True, 6.7), (HEDGE, False, 14.7), (HEDGE, True, 25.4)],
    ids=["mutual-HAC", "mutual-HACPW", "hedge-HAC", "hedge-HACPW"],
)
def test_matches_ledoit_wolf_2008_table_3(pair, prewhiten, published_pct):
    out = sharpe_difference_test(*pair, prewhiten=prewhiten, unit="percent")
    # The paper prints one decimal of a percent: the computed value must round to it.
    assert 100 * out["pvalue"] == pytest.approx(published_pct, abs=0.05)
    assert out["n_obs"] == 120 and out["prewhiten"] is prewhiten


def test_sharpes_match_ledoit_wolf_2008_table_2():
    """Table 2 prints three decimals; Fidelity's 0.10746 is printed 0.108, so the check is to 0.001."""
    mutual, hedge = sharpe_difference_test(*MUTUAL), sharpe_difference_test(*HEDGE)
    computed = [
        mutual["sharpe_candidate"],
        mutual["sharpe_baseline"],
        hedge["sharpe_candidate"],
        hedge["sharpe_baseline"],
    ]
    assert computed == pytest.approx([0.108, 0.011, 1.014, 1.461], abs=1e-3)
    assert hedge["delta"] == pytest.approx(hedge["sharpe_candidate"] - hedge["sharpe_baseline"], rel=1e-15)


def test_sign_follows_the_candidate():
    fwd, rev = sharpe_difference_test(*HEDGE), sharpe_difference_test(*HEDGE[::-1])
    assert fwd["delta"] < 0 and rev["delta"] == pytest.approx(-fwd["delta"], rel=1e-12)
    assert rev["zstat"] == pytest.approx(-fwd["zstat"], rel=1e-9)
    assert rev["pvalue"] == pytest.approx(fwd["pvalue"], rel=1e-9)
    # One-sided H1: delta > 0. JMG's lower Sharpe is no evidence it beats Coast.
    assert fwd["pvalue_onesided"] > 0.5
    assert fwd["pvalue_onesided"] + rev["pvalue_onesided"] == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("prewhiten", [False, True])
def test_a_fraction_series_gets_the_papers_percent_test(prewhiten):
    """The same returns declared in either unit give one answer: the paper's, in percent."""
    pct = sharpe_difference_test(*HEDGE, prewhiten=prewhiten, unit="percent")
    frac = sharpe_difference_test(HEDGE[0] / 100, HEDGE[1] / 100, prewhiten=prewhiten)
    assert frac["unit"] == "fraction" and pct["unit"] == "percent"
    for key in ("delta", "zstat", "pvalue", "bandwidth"):
        assert frac[key] == pytest.approx(pct[key], rel=1e-9)
    # Undeclared, a percent series read as fractions is a different (unit-dependent) HAC: not the paper's.
    wrong = sharpe_difference_test(*HEDGE, prewhiten=prewhiten)
    assert wrong["delta"] == pct["delta"] and wrong["pvalue"] != pytest.approx(pct["pvalue"], rel=1e-3)


def test_identical_series_have_no_test():
    out = sharpe_difference_test(HEDGE[0], HEDGE[0].copy())
    assert out["delta"] == 0.0
    assert out["se"] is None and out["zstat"] is None and out["pvalue"] is None and out["pvalue_onesided"] is None


def test_refusals():
    c, b = HEDGE
    with pytest.raises(InferenceInputError, match="same rows"):
        sharpe_difference_test(c, b[:-1])
    with pytest.raises(InferenceInputError, match="NaN/Inf"):
        sharpe_difference_test(np.r_[np.nan, c[1:]], b)
    with pytest.raises(InferenceInputError, match="at least 6"):
        sharpe_difference_test(c[:5], b[:5])
    with pytest.raises(InferenceInputError, match="unit must be one of"):
        sharpe_difference_test(c, b, unit="bp")
    with pytest.raises(InferenceInputError, match="zero-variance baseline"):
        sharpe_difference_test(c, np.full_like(b, 0.001))


def test_a_dropped_row_leaves_both_series():
    c = np.r_[np.nan, HEDGE[0][1:]]
    out = sharpe_difference_test(c, HEDGE[1], allow_nonfinite_drop=True)
    ref = sharpe_difference_test(HEDGE[0][1:], HEDGE[1][1:])
    assert out["n_nonfinite_dropped"] == 1 and out["n_obs"] == 119 and out["n_obs_raw"] == 120
    assert out["pvalue"] == ref["pvalue"]


# ── the gate: a declared family against one baseline ───────────────────────

RNG = np.random.default_rng(7)
BASE = RNG.normal(0.0, 0.01, 500)
STRONG = BASE + RNG.normal(0.002, 0.004, 500)  # same shocks, a positive drift: a much higher Sharpe
NOISE = [BASE + RNG.normal(0.0, 0.004, 500) for _ in range(4)]


def test_gate_finds_the_real_winner_under_holm_and_bhy():
    family = {"strong": STRONG, **{f"noise{i}": r for i, r in enumerate(NOISE)}}
    for method in ("holm", "bhy"):
        out = sharpe_difference_gate(family, BASE, method=method)
        assert out["beats"] == ["strong"] and out["gate_pass"] is True
        assert out["m"] == 5 and out["method"] == method
        assert out["candidates"]["strong"]["pvalue_adjusted"] <= 0.05


def test_gate_corrects_over_the_whole_family():
    alone = sharpe_difference_gate({"strong": STRONG}, BASE)["candidates"]["strong"]
    assert alone["pvalue_adjusted"] == alone["pvalue_onesided"]  # m = 1: nothing to correct
    family = sharpe_difference_gate({"strong": STRONG, "noise0": NOISE[0]}, BASE)["candidates"]["strong"]
    assert family["pvalue_adjusted"] >= alone["pvalue_onesided"]


def test_gate_baseline_never_beats_itself():
    out = sharpe_difference_gate({"noise0": NOISE[0]}, BASE)
    assert out["gate_pass"] is False and out["beats"] == []


def test_gate_refusals():
    with pytest.raises(GateInputError, match="non-empty mapping"):
        sharpe_difference_gate({}, BASE)
    with pytest.raises(GateInputError, match="degenerate HAC variance"):
        sharpe_difference_gate({"same": BASE.copy()}, BASE)
    with pytest.raises(GateInputError, match="method must be one of"):
        sharpe_difference_gate({"strong": STRONG}, BASE, method="fdr_bh")
