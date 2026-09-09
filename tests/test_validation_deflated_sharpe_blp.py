"""Tests for DeflatedSharpeBLPValidation plugin (validation.deflated_sharpe_blp.v1).

Covers the analytic Bailey & Lopez de Prado (2014) DSR: PSR/DSR bounds, the
skew/kurtosis correction, the trial_sharpes vs. se-proxy sigma_SR modes, and
plugin metadata. Self-contained -- no conftest.py required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.analysis.dsr import expected_max_sr
from quantbox.plugins.validation.deflated_sharpe_blp import DeflatedSharpeBLPValidation


class TestDeflatedSharpeBLPValidation:
    @pytest.fixture
    def plugin(self) -> DeflatedSharpeBLPValidation:
        return DeflatedSharpeBLPValidation()

    @staticmethod
    def _make_returns(n_days: int = 1000, mean: float = 0.0008, std: float = 0.02, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
        rets = rng.normal(mean, std, size=n_days)
        return pd.DataFrame({"returns": rets}, index=dates)

    @staticmethod
    def _empty_weights() -> pd.DataFrame:
        return pd.DataFrame()

    def test_result_has_required_keys(self, plugin: DeflatedSharpeBLPValidation) -> None:
        result = plugin.validate(self._make_returns(), self._empty_weights(), None, {})
        assert "findings" in result
        assert "metrics" in result
        assert "passed" in result

    def test_metrics_contain_expected_keys(self, plugin: DeflatedSharpeBLPValidation) -> None:
        result = plugin.validate(self._make_returns(), self._empty_weights(), None, {})
        metrics = result["metrics"]
        for key in (
            "observed_sharpe",
            "skewness",
            "kurtosis",
            "sharpe_standard_error",
            "sigma_sr",
            "sigma_sr_source",
            "expected_max_sharpe_null",
            "psr",
            "dsr",
        ):
            assert key in metrics

    def test_dsr_and_psr_are_probabilities(self, plugin: DeflatedSharpeBLPValidation) -> None:
        result = plugin.validate(self._make_returns(), self._empty_weights(), None, {"n_trials": 5})
        metrics = result["metrics"]
        assert 0.0 <= metrics["psr"] <= 1.0
        assert 0.0 <= metrics["dsr"] <= 1.0

    def test_dsr_leq_psr(self, plugin: DeflatedSharpeBLPValidation) -> None:
        """DSR compares against a higher (or equal) bar than plain PSR(0), so DSR <= PSR."""
        result = plugin.validate(self._make_returns(), self._empty_weights(), None, {"n_trials": 10})
        metrics = result["metrics"]
        assert metrics["dsr"] <= metrics["psr"] + 1e-9

    def test_more_trials_reduces_dsr(self, plugin: DeflatedSharpeBLPValidation) -> None:
        """More trials -> higher expected-max-Sharpe-by-chance -> lower DSR, all else equal."""
        returns = self._make_returns()
        result_1 = plugin.validate(returns, self._empty_weights(), None, {"n_trials": 1})
        result_50 = plugin.validate(returns, self._empty_weights(), None, {"n_trials": 50})
        assert result_50["metrics"]["dsr"] <= result_1["metrics"]["dsr"]
        assert result_50["metrics"]["expected_max_sharpe_null"] >= result_1["metrics"]["expected_max_sharpe_null"]

    def test_trial_sharpes_mode_used_when_supplied(self, plugin: DeflatedSharpeBLPValidation) -> None:
        returns = self._make_returns()
        result = plugin.validate(returns, self._empty_weights(), None, {"trial_sharpes": [0.3, 0.5, 0.6, 0.55, 0.58]})
        assert result["metrics"]["sigma_sr_source"] == "trial_sharpes"
        assert result["metrics"]["n_trials"] == 5

    def test_se_proxy_mode_when_no_trial_sharpes(self, plugin: DeflatedSharpeBLPValidation) -> None:
        returns = self._make_returns()
        result = plugin.validate(returns, self._empty_weights(), None, {"n_trials": 6})
        assert result["metrics"]["sigma_sr_source"] == "se_proxy_approximation"
        assert any(f["rule"] == "sigma_sr_approximated" for f in result["findings"])

    def test_single_trial_no_multiple_testing_penalty(self, plugin: DeflatedSharpeBLPValidation) -> None:
        """With n_trials=1 (default), DSR reduces to PSR(0) -- no multiple-testing penalty."""
        returns = self._make_returns()
        result = plugin.validate(returns, self._empty_weights(), None, {})
        metrics = result["metrics"]
        assert metrics["expected_max_sharpe_null"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["dsr"] == pytest.approx(metrics["psr"], abs=1e-9)

    def test_skew_kurtosis_reported_for_known_distribution(self, plugin: DeflatedSharpeBLPValidation) -> None:
        """A large Gaussian sample should have skew ~0 and kurtosis ~3."""
        returns = self._make_returns(n_days=5000, seed=7)
        result = plugin.validate(returns, self._empty_weights(), None, {})
        metrics = result["metrics"]
        assert abs(metrics["skewness"]) < 0.2
        assert abs(metrics["kurtosis"] - 3.0) < 0.5

    def test_h32_sweep_worked_example(self, plugin: DeflatedSharpeBLPValidation) -> None:
        """Real motivating case: the H32 min_periods sweep's 5 full-sample Sharpes
        (2026-07-13, quantbox-lab issue #59) as the trial distribution, testing the
        min_periods=126 variant (observed_sharpe ~0.722) against them.
        """
        # ~3187 daily returns with mean/std chosen so the annualized Sharpe matches
        # the min_periods=126 sweep run (0.7220134158888037, trading_days=365).
        n = 3187
        target_sharpe = 0.7220134158888037
        rng = np.random.default_rng(126)
        std = 0.02
        mean = target_sharpe * std / np.sqrt(365)
        rets = rng.normal(mean, std, size=n)
        # Rescale to hit the exact target annualized Sharpe.
        rets = rets - rets.mean() + mean
        actual_sr = (rets.mean() / rets.std(ddof=1)) * np.sqrt(365)
        rets = rets * (target_sharpe / actual_sr)
        returns = pd.DataFrame({"returns": rets}, index=pd.date_range("2017-08-17", periods=n, freq="D"))

        trial_sharpes = [
            0.6919138246717664,
            0.7070977107825139,
            0.7220134158888037,
            0.7214006140958918,
            0.7210332073938897,
        ]
        result = plugin.validate(returns, self._empty_weights(), None, {"trial_sharpes": trial_sharpes})
        metrics = result["metrics"]
        assert metrics["sigma_sr_source"] == "trial_sharpes"
        assert metrics["n_trials"] == 5
        # sigma_SR across 5 nearly-identical Sharpes is small -> small multiple-testing
        # penalty -> DSR should be close to (but <=) PSR(0).
        assert metrics["dsr"] <= metrics["psr"] + 1e-9
        assert metrics["expected_max_sharpe_null"] < metrics["observed_sharpe"]

    def test_plugin_meta_attributes(self) -> None:
        meta = DeflatedSharpeBLPValidation.meta
        assert meta.name == "validation.deflated_sharpe_blp.v1"
        assert meta.kind == "validation"
        assert "validation" in meta.tags

    def test_too_few_observations(self, plugin: DeflatedSharpeBLPValidation) -> None:
        returns = pd.DataFrame({"returns": [0.01, -0.005]}, index=pd.date_range("2024-01-01", periods=2))
        result = plugin.validate(returns, self._empty_weights(), None, {})
        assert result["passed"] is False
        assert any(f["rule"] == "insufficient_observations" for f in result["findings"])


class TestFailClosedOnUncomputableInput:
    """A validation gate that cannot COMPUTE a verdict must not emit a passing one.

    Both cases below previously returned a confident, plausible-looking number
    instead of refusing: a zero-variance series was scored as if it had a
    Sharpe, and a non-positive ``n_trials`` was coerced to 1, silently deleting
    the entire multiple-testing deflation.
    """

    @pytest.fixture
    def plugin(self) -> DeflatedSharpeBLPValidation:
        return DeflatedSharpeBLPValidation()

    @staticmethod
    def _series(values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"returns": values}, index=pd.date_range("2024-01-01", periods=len(values)))

    @staticmethod
    def _good_returns(n: int = 300) -> pd.DataFrame:
        rng = np.random.default_rng(11)
        return pd.DataFrame(
            {"returns": rng.normal(0.001, 0.01, size=n)},
            index=pd.date_range("2024-01-01", periods=n),
        )

    # --- Defect 1: degenerate (zero-variance) input ---

    @pytest.mark.parametrize("value", [0.0, 0.01, -0.02])
    def test_constant_returns_are_refused_not_scored(self, plugin: DeflatedSharpeBLPValidation, value: float) -> None:
        result = plugin.validate(self._series([value] * 50), pd.DataFrame(), None, {})
        assert result["passed"] is False
        assert any(f["rule"] == "degenerate_returns" and f["level"] == "error" for f in result["findings"])
        # The point of the fix: no NUMBER is emitted that a downstream reader
        # could mistake for a computed verdict.
        assert result["metrics"]["dsr"] is None
        assert result["metrics"]["psr"] is None

    def test_non_finite_returns_are_refused(self, plugin: DeflatedSharpeBLPValidation) -> None:
        returns = self._good_returns()
        returns.iloc[5, 0] = np.nan
        result = plugin.validate(returns, pd.DataFrame(), None, {})
        assert result["passed"] is False
        assert any(f["rule"] == "non_finite_returns" and f["level"] == "error" for f in result["findings"])
        assert result["metrics"]["dsr"] is None

    # --- Defect 2: the multiple-testing penalty must not silently disappear ---

    @pytest.mark.parametrize("n_trials", [0, -1, -5])
    def test_non_positive_n_trials_is_refused(self, plugin: DeflatedSharpeBLPValidation, n_trials: int) -> None:
        result = plugin.validate(self._good_returns(), pd.DataFrame(), None, {"n_trials": n_trials})
        assert result["passed"] is False
        assert any(f["rule"] == "invalid_n_trials" and f["level"] == "error" for f in result["findings"])
        assert result["metrics"]["dsr"] is None

    def test_non_positive_n_trials_does_not_reproduce_the_single_trial_answer(
        self, plugin: DeflatedSharpeBLPValidation
    ) -> None:
        """The exact defect shape: n_trials of 0, -5 and 1 all returned the same DSR."""
        returns = self._good_returns()
        one = plugin.validate(returns, pd.DataFrame(), None, {"n_trials": 1})
        assert one["metrics"]["dsr"] is not None
        for bad in (0, -5):
            result = plugin.validate(returns, pd.DataFrame(), None, {"n_trials": bad})
            assert result["metrics"]["dsr"] != one["metrics"]["dsr"]

    @pytest.mark.parametrize("n_trials", [2.7, "5", None])
    def test_non_integral_n_trials_is_refused(self, plugin: DeflatedSharpeBLPValidation, n_trials: object) -> None:
        result = plugin.validate(self._good_returns(), pd.DataFrame(), None, {"n_trials": n_trials})
        assert result["passed"] is False
        assert any(f["rule"] == "invalid_n_trials" for f in result["findings"])
        assert result["metrics"]["dsr"] is None

    # --- Anti-drift: the deflation term is the framework's, not a local copy ---

    @pytest.mark.parametrize("n_trials", [1, 2, 7, 50, 1000])
    def test_deflation_term_matches_the_framework_module(
        self, plugin: DeflatedSharpeBLPValidation, n_trials: int
    ) -> None:
        """expected_max_sharpe_null must equal sigma_SR * analysis.dsr.expected_max_sr(N).

        This module reimplemented that formula once and it drifted. Assert the
        delegation numerically so a re-inlined copy cannot pass silently.
        """
        result = plugin.validate(self._good_returns(), pd.DataFrame(), None, {"n_trials": n_trials})
        metrics = result["metrics"]
        expected = metrics["sigma_sr"] * expected_max_sr(n_trials)
        assert metrics["expected_max_sharpe_null"] == pytest.approx(expected, rel=1e-12, abs=1e-12)
