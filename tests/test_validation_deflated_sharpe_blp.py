"""Tests for DeflatedSharpeBLPValidation plugin (validation.deflated_sharpe_blp.v1).

Covers the analytic Bailey & Lopez de Prado (2014) DSR: PSR/DSR bounds, the
skew/kurtosis correction, the trial_sharpes vs. se-proxy sigma_SR modes, and
plugin metadata. Self-contained -- no conftest.py required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
        result = plugin.validate(
            returns, self._empty_weights(), None, {"trial_sharpes": [0.3, 0.5, 0.6, 0.55, 0.58]}
        )
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

        trial_sharpes = [0.6919138246717664, 0.7070977107825139, 0.7220134158888037, 0.7214006140958918, 0.7210332073938897]
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
