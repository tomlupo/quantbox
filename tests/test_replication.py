"""``metrics.correlation``, ``gates.replication_gate`` and ``quantbox gates replicate`` (TOM-3699).

Metrics describe (rho and the overlap count on a declared frequency, NaN masked
pairwise), gates decide (PASS at rho >= min_corr, DEFECT below the floor, FAIL
between or on too few periods). ADR-0009's split.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from quantbox.cli import app
from quantbox.gates import GateInputError, replication_gate
from quantbox.metrics import correlation

runner = CliRunner()

# returns: business days 2010..2019; the reference starts mid-June 2012 and runs to 2021
_RETURNS_DAYS = pd.bdate_range("2010-01-01", "2019-12-31")
_REFERENCE_DAYS = pd.bdate_range("2012-06-15", "2021-12-31")
# complete months both series show: July 2012 (June has no anchor in the reference) .. December 2019
_COMMON_MONTHS = 6 + 7 * 12


def _z(seed: int) -> pd.Series:
    days = _RETURNS_DAYS.union(_REFERENCE_DAYS)
    return pd.Series(np.random.default_rng(seed).standard_normal(len(days)), index=days)


@pytest.fixture(scope="module")
def series() -> dict[str, pd.Series]:
    z1, z2 = _z(3699), _z(3700)

    def mix(rho: float) -> pd.Series:
        return (0.01 * (rho * z1 + np.sqrt(1 - rho**2) * z2)).reindex(_REFERENCE_DAYS)

    returns = (0.01 * z1).reindex(_RETURNS_DAYS).rename("returns")
    return {
        "returns": returns,
        "same": returns.copy(),
        "rho06": mix(0.6),
        "rho078": mix(0.78),
        "flipped": -(0.01 * z1).reindex(_REFERENCE_DAYS),
    }


@pytest.fixture()
def files(tmp_path, series) -> dict[str, str]:
    out = {}
    for name, s in series.items():
        path = tmp_path / f"{name}.csv"
        s.rename("returns").rename_axis("date").to_csv(path)
        out[name] = str(path)
    return out


def _gate(*argv: str):
    res = runner.invoke(app, ["gates", "replicate", *argv, "--json"])
    return res, (json.loads(res.stdout) if res.exit_code in (0, 1) else None)


# --- the card's proof -------------------------------------------------------


def test_an_identical_series_gives_rho_one_and_passes(files):
    res, out = _gate("--returns", files["returns"], "--reference", files["same"], "--period", "ME")
    assert res.exit_code == 0, res.output
    assert out["rho"] == 1.0 and out["verdict"] == "PASS" and out["gate_pass"] is True
    assert out["n_overlap"] == 11 + 9 * 12  # February 2010 .. December 2019: January has no anchor
    assert out["window"] == {"period": "ME", "start": None, "end": None}


def test_a_series_correlated_about_06_is_a_defect(files):
    res, out = _gate("--returns", files["returns"], "--reference", files["rho06"], "--period", "ME")
    assert res.exit_code == 1, res.output
    assert 0.5 < out["rho"] < 0.7
    assert out["verdict"] == "DEFECT" and out["gate_pass"] is False


def test_the_overlap_count_is_the_number_of_common_months(series):
    c = correlation(series["returns"], series["rho06"], "ME")
    assert c.n_overlap == _COMMON_MONTHS
    assert (c.first, c.last) == (pd.Timestamp("2012-07-31"), pd.Timestamp("2019-12-31"))


def test_the_text_line_names_the_defect(files):
    res = runner.invoke(
        app, ["gates", "replicate", "--returns", files["returns"], "--reference", files["rho06"], "--period", "ME"]
    )
    assert res.exit_code == 1
    assert res.stdout.startswith("replicate: DEFECT (rho=0.")


# --- pairwise NaN masking and the declared frequency -------------------------


def test_a_nan_day_masks_its_month_on_both_sides(series):
    ref = series["rho06"].copy()
    ref.loc["2015-03-10"] = np.nan
    c = correlation(series["returns"], ref, "ME")
    assert c.n_overlap == _COMMON_MONTHS - 1
    full = correlation(series["returns"], series["rho06"], "ME")
    assert c.rho != full.rho  # March 2015 is out, not paired as 0%


def test_without_a_frequency_the_rows_pair_by_date(series):
    ref = series["rho06"].copy()
    ref.iloc[10] = np.nan
    common = series["returns"].index.intersection(ref.index)
    c = correlation(series["returns"], ref)
    assert c.n_overlap == len(common) - 1
    pair = pd.concat([series["returns"], ref], axis=1, join="inner").dropna()
    assert c.rho == pytest.approx(pair.corr().iloc[0, 1], abs=1e-12)


def test_start_and_end_read_like_period_returns(series):
    # the window's first month keeps its anchor in the rows before the window
    c = correlation(series["returns"], series["rho06"], "ME", "2015-01-01", "2015-12-31")
    assert c.n_overlap == 12 and c.first == pd.Timestamp("2015-01-31")


def test_quarters_pair_whole_quarters(series):
    # Q3 2012 (from July) .. Q4 2019; Q2 2012 starts before the reference's first row
    assert correlation(series["returns"], series["rho06"], "QE").n_overlap == 2 + 7 * 4


# --- the verdict bands -------------------------------------------------------


def test_between_floor_and_min_corr_is_a_fail_not_a_defect(series):
    out = replication_gate(series["returns"], series["rho078"], freq="ME")
    assert 0.70 <= out["rho"] < 0.85 and out["verdict"] == "FAIL"


def test_a_sign_flipped_copy_is_a_defect(series):
    out = replication_gate(series["returns"], series["flipped"], freq="ME")
    # compounding is not linear, so the monthly copy is -0.998, not -1
    assert out["rho"] < -0.99 and out["verdict"] == "DEFECT"


def test_too_few_periods_fail_whatever_rho(series):
    out = replication_gate(series["returns"], series["same"], freq="YE", min_obs=12)
    assert out["rho"] == 1.0 and out["n_overlap"] == 9 and out["verdict"] == "FAIL" and not out["obs_pass"]


def test_the_thresholds_are_arguments(series):
    out = replication_gate(series["returns"], series["rho06"], freq="ME", min_corr=0.5, floor=0.3)
    assert out["verdict"] == "PASS" and (out["min_corr"], out["floor"]) == (0.5, 0.3)


@pytest.mark.parametrize(
    "kw",
    [{"floor": 0.9}, {"min_corr": 1.5}, {"floor": float("nan")}, {"min_obs": 2}, {"min_obs": True}],
)
def test_incoherent_thresholds_are_refused(series, kw):
    with pytest.raises(GateInputError):
        replication_gate(series["returns"], series["same"], **kw)


# --- refusals: no verdict on an input that cannot give one --------------------


def test_percent_units_are_refused_when_compounded_but_not_needed_without(series):
    pct = series["rho06"] * 100
    with pytest.raises(GateInputError, match="percent units"):
        replication_gate(series["returns"], pct, freq="ME")
    assert correlation(series["returns"], pct).rho == pytest.approx(
        correlation(series["returns"], series["rho06"]).rho, abs=1e-12
    )


def test_a_timezone_mismatch_is_refused(series):
    aware = series["same"].tz_localize("UTC")
    with pytest.raises(GateInputError, match="timezone"):
        replication_gate(series["returns"], aware, freq="ME")


def test_no_shared_period_is_refused(series):
    late = series["rho06"]["2020-01-01":]
    with pytest.raises(GateInputError, match="share no finite period"):
        replication_gate(series["returns"], late, freq="ME")


def test_a_flat_reference_has_no_correlation(series):
    # a constant DAILY return compounds to unequal months, so the flat monthly series is 0%
    assert np.isnan(correlation(series["returns"], pd.Series(0.001, index=_REFERENCE_DAYS)).rho)
    flat = pd.Series(0.0, index=_REFERENCE_DAYS)
    assert np.isnan(correlation(series["returns"], flat, "ME").rho)
    with pytest.raises(GateInputError, match="flat"):
        replication_gate(series["returns"], flat, freq="ME")


def test_cli_refusals_exit_2(files, tmp_path):
    res, _ = _gate("--returns", files["returns"], "--reference", files["same"], "--period", "WE")
    assert res.exit_code == 2
    dup = tmp_path / "dup.csv"
    dup.write_text("date,returns\n2020-01-02,0.01\n2020-01-02,0.02\n2020-01-03,0.0\n")
    res, _ = _gate("--returns", files["returns"], "--reference", str(dup))
    assert res.exit_code == 2 and "duplicate date" in res.stderr
