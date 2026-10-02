"""The overlay stage (TOM-1364, ADR-0004): a chain of modifiers on a base strategy's decided book.

THE TOY. Asset ``A`` rises one point a bar until bar ``F``, where it drops to 50
and stays there; ``USD`` is constant. EWMAC(2,8) of ``A`` is positive on every
bar before ``F`` and negative from ``F`` on, so it flips sign exactly once, on
bar ``F``. The base strategy holds ``A`` at 1.0 on every bar.

``overlay.reversal_derisk.v1`` (hold_bars=H, multiplier=0.5) must therefore
halve the DECIDED weight on bars ``F .. F+H-1`` and nowhere else, and the
TRADED weight — after the run's execution lag — on bars ``F+lag .. F+H-1+lag``.
An overlay that shifted its own output would land one bar further.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from quantbox.cli import app
from quantbox.overlays import apply_overlays
from quantbox.plugins.overlays import CorrGrossCapOverlay, RegimeReweightOverlay, ReversalDeriskOverlay
from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline
from quantbox.registry import PluginRegistry
from quantbox.run_manifest import validate_run_manifest
from quantbox.runner import run_from_config
from quantbox.store import FileArtifactStore
from quantbox.validate import validate_config

N = 40
F = 20
H = 5
DERISK = {"fast_span": 2, "slow_span": 8, "multiplier": 0.5, "hold_bars": H}
REPO = Path(__file__).resolve().parents[1]
H22_CONFIG = REPO / "cookbook" / "configs" / "run_backtest_overlay_reversal_derisk.yaml"
cli = CliRunner()


def _prices() -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=N, freq="D").values)
    a = 100.0 + np.arange(N, dtype=float)
    a[F:] = 50.0
    return pd.DataFrame({"A": a, "USD": 100.0}, index=idx)


def _hold_a() -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=_prices().index, columns=["A", "USD"])
    w["A"] = 1.0
    return w


def _expected_a(lag: int) -> np.ndarray:
    """A's weight on each bar: 1.0, halved on the de-risk window shifted by ``lag`` (0 before the first fill)."""
    out = np.ones(N)
    out[F + lag : F + H + lag] = 0.5
    out[:lag] = 0.0
    return out


class _HoldA:
    meta = type("M", (), {"name": "strategy.hold_a.v1"})()

    def run(self, data: Any, params: Any = None) -> dict[str, Any]:
        return {"weights": _hold_a()}


class _Data:
    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame({"symbol": ["A", "USD"]})

    def load_market_data(self, universe: Any, asof: str, params: dict[str, Any]) -> dict[str, pd.DataFrame]:
        return {"prices": _prices()}


def _pipeline_run(tmp_path, params: dict[str, Any], chain, **kw):
    store = FileArtifactStore(str(tmp_path), "run")
    result = BacktestPipeline().run(
        mode="backtest",
        asof="2024-02-09",
        params={"fees": 0.0, "strategies": [{"name": "strategy.hold_a.v1", "weight": 1.0}], **params},
        data=_Data(),
        store=store,
        broker=None,
        risk=[],
        strategies=[_HoldA()],
        overlays=chain,
        **kw,
    )
    return result, store


# ----------------------------------------------------------------------
# Known answer: the effect lands on the bar the execution convention says
# ----------------------------------------------------------------------


def test_reversal_derisk_halves_exactly_the_window_starting_on_the_flip_bar():
    out = ReversalDeriskOverlay().apply(_hold_a(), {"prices": _prices()}, DERISK)
    np.testing.assert_array_equal(out["A"].to_numpy(), _expected_a(0))
    assert (out["USD"] == 0.0).all()


@pytest.mark.parametrize("engine", ["vectorbt", "rsims"])
@pytest.mark.parametrize("lag", [1, 2])
def test_overlay_effect_lands_on_the_execution_conventions_bar_with_no_extra_shift(tmp_path, engine, lag):
    result, store = _pipeline_run(
        tmp_path, {"engine": engine, "execution": {"lag_bars": lag}}, [(ReversalDeriskOverlay(), DERISK)]
    )
    decided = store.read_parquet("weights_history").set_index("date")
    traded = store.read_parquet("traded_weights").set_index("date")
    np.testing.assert_array_equal(decided["A"].to_numpy(), _expected_a(0))
    np.testing.assert_array_equal(traded["A"].to_numpy(), _expected_a(lag))
    assert result.notes["overlays"] == [{"name": "overlay.reversal_derisk.v1", "version": "0.1.0", "params": DERISK}]
    base = store.read_parquet("base_weights_history").set_index("date")
    assert (base["A"] == 1.0).all()


def test_no_overlays_leaves_the_run_unchanged(tmp_path):
    result, store = _pipeline_run(tmp_path, {}, None)
    assert result.notes["overlays"] == []
    assert "base_weights_history" not in result.artifacts
    traded = store.read_parquet("traded_weights").set_index("date")
    np.testing.assert_array_equal(traded["A"].to_numpy(), np.r_[0.0, np.ones(N - 1)])


def test_variants_take_the_same_overlay_chain(tmp_path):
    store = FileArtifactStore(str(tmp_path), "run")
    result = BacktestPipeline().run(
        mode="backtest",
        asof="2024-02-09",
        params={"fees": 0.0, "variants": [{"name": "v", "strategy": {"name": "v"}}]},
        data=_Data(),
        store=store,
        broker=None,
        risk=[],
        variant_plugins={"v": _HoldA()},
        overlays=[(ReversalDeriskOverlay(), DERISK)],
    )
    traded = store.read_parquet("traded_weights").set_index("date")
    np.testing.assert_array_equal(traded["A"].to_numpy(), _expected_a(1))
    assert [o["name"] for o in result.notes["overlays"]] == ["overlay.reversal_derisk.v1"]


# ----------------------------------------------------------------------
# The chain
# ----------------------------------------------------------------------


class _Scale:
    def __init__(self, name: str, k: float):
        self.meta = type("M", (), {"name": name, "version": "0"})()
        self.k = k

    def apply(self, weights, data, params):
        return weights * self.k


class _Shift:
    meta = type("M", (), {"name": "overlay.shifts_itself", "version": "0"})()

    def apply(self, weights, data, params):
        return weights.shift(1).dropna()


def test_chain_applies_in_config_order_and_records_it():
    cap = CorrGrossCapOverlay()
    two = pd.DataFrame({"A": [1.0] * N, "B": [1.0] * N}, index=_prices().index)
    prices = pd.DataFrame({"A": _prices()["A"], "B": _prices()["A"] * 2}, index=two.index)  # perfectly correlated
    params = {"window": 5, "corr_threshold": 0.5, "max_gross": 1.0}
    capped_then_doubled, rec = apply_overlays(two, {"prices": prices}, [(cap, params), (_Scale("x2", 2.0), {})])
    doubled_then_capped, _ = apply_overlays(two, {"prices": prices}, [(_Scale("x2", 2.0), {}), (cap, params)])
    bar = two.index[F - 1]  # window filled, A and B still moving (both flat after F: no correlation defined)
    assert capped_then_doubled.loc[bar].abs().sum() == pytest.approx(2.0)
    assert doubled_then_capped.loc[bar].abs().sum() == pytest.approx(1.0)
    assert [r["name"] for r in rec] == ["overlay.corr_gross_cap.v1", "x2"]


def test_an_overlay_that_shifts_or_drops_rows_is_refused():
    with pytest.raises(ValueError, match="never shifts"):
        apply_overlays(_hold_a(), {"prices": _prices()}, [(_Shift(), {})])


class _DropInPlace:
    meta = type("M", (), {"name": "overlay.drops_in_place", "version": "0"})()

    def apply(self, weights, data, params):
        weights.drop(columns=["A"], inplace=True)
        return weights


class _ZeroInPlaceReturnsOther:
    meta = type("M", (), {"name": "overlay.zeros_in_place", "version": "0"})()

    def apply(self, weights, data, params):
        out = weights.copy()
        weights.loc[:, :] = 0.0
        data.pop("prices", None)
        return out


def test_an_overlay_that_drops_a_column_in_place_is_refused():
    with pytest.raises(ValueError, match="never shifts"):
        apply_overlays(_hold_a(), {"prices": _prices()}, [(_DropInPlace(), {})])


def test_an_overlay_cannot_mutate_the_callers_weights_or_data():
    base = _hold_a()
    data = {"prices": _prices()}
    out, _ = apply_overlays(base, data, [(_ZeroInPlaceReturnsOther(), {})])
    assert (base["A"] == 1.0).all()
    assert "prices" in data
    pd.testing.assert_frame_equal(out, _hold_a())


class _Sparse:
    """A strategy that speaks on bars 0 and 2 only: [1, NaN, 0, NaN, ...] for A."""

    meta = type("M", (), {"name": "strategy.sparse.v1"})()

    def run(self, data: Any, params: Any = None) -> dict[str, Any]:
        w = pd.DataFrame(np.nan, index=_prices().index, columns=["A", "USD"])
        w.iloc[0] = [1.0, 0.0]
        w.iloc[2] = [0.0, 0.0]
        return {"weights": w}


@pytest.mark.parametrize("engine", ["vectorbt", "rsims"])
@pytest.mark.parametrize("tranches", [1, 2, 3])
def test_an_identity_overlay_leaves_a_sparse_book_bit_for_bit(tmp_path, engine, tranches):
    """NaN cells mean "hold" (vectorbt) / "flat" (rsims); the overlay path must not
    resolve them earlier than the baseline does, or tranching averages different numbers."""

    def run(sub, chain):
        store = FileArtifactStore(str(tmp_path / sub), "run")
        result = BacktestPipeline().run(
            mode="backtest",
            asof="2024-02-09",
            params={
                "fees": 0.0,
                "engine": engine,
                "risk": {"tranches": tranches},
                "strategies": [{"name": "strategy.sparse.v1", "weight": 1.0}],
            },
            data=_Data(),
            store=store,
            broker=None,
            risk=[],
            strategies=[_Sparse()],
            overlays=chain,
        )
        return result, store.read_parquet("traded_weights").set_index("date")

    base_res, base = run("base", None)
    over_res, over = run("over", [(_Scale("identity", 1.0), {})])
    pd.testing.assert_frame_equal(over, base)
    assert over_res.metrics == base_res.metrics


class _SparseHoldA:
    """A strategy that speaks once: A=1.0 on bar 0, NaN ("no new target") after."""

    meta = type("M", (), {"name": "strategy.sparse_hold_a.v1"})()

    def run(self, data: Any, params: Any = None) -> dict[str, Any]:
        w = pd.DataFrame(np.nan, index=_prices().index, columns=["A", "USD"])
        w.iloc[0] = [1.0, 0.0]
        return {"weights": w}


def _sparse_traded(tmp_path, sub, engine, lag, chain):
    store = FileArtifactStore(str(tmp_path / sub), "run")
    BacktestPipeline().run(
        mode="backtest",
        asof="2024-02-09",
        params={
            "fees": 0.0,
            "engine": engine,
            "execution": {"lag_bars": lag},
            "strategies": [{"name": "strategy.sparse_hold_a.v1", "weight": 1.0}],
        },
        data=_Data(),
        store=store,
        broker=None,
        risk=[],
        strategies=[_SparseHoldA()],
        overlays=chain,
    )
    return store.read_parquet("traded_weights").set_index("date")


@pytest.mark.parametrize("lag", [1, 2])
def test_an_expired_overlay_returns_a_sparse_hold_book_to_its_base_position(tmp_path, lag):
    """vectorbt reads NaN as "hold the last target": once the de-risk window closes,
    the book must go BACK to the base 1.0, not hold the last reduced 0.5 forever."""
    traded = _sparse_traded(tmp_path, "over", "vectorbt", lag, [(ReversalDeriskOverlay(), DERISK)])
    np.testing.assert_array_equal(traded["A"].to_numpy(), _expected_a(lag))


@pytest.mark.parametrize("lag", [1, 2])
def test_an_expired_overlay_on_a_sparse_flat_book_matches_the_base_run(tmp_path, lag):
    """rsims reads NaN as "flat": scaling a flat cell changes nothing, so the book is the base one."""
    base = _sparse_traded(tmp_path, "base", "rsims", lag, None)
    over = _sparse_traded(tmp_path, "over", "rsims", lag, [(ReversalDeriskOverlay(), DERISK)])
    pd.testing.assert_frame_equal(over, base)


# ----------------------------------------------------------------------
# The other two overlays, known answers
# ----------------------------------------------------------------------


def test_regime_reweight_scales_from_the_first_bar_the_vol_spike_is_known_and_not_before():
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    rets = np.where(np.arange(80) % 2 == 0, 0.001, -0.001)
    rets[60:] = np.where(np.arange(60, 80) % 2 == 0, 0.05, -0.05)
    rets[0] = 0.0
    prices = pd.DataFrame({"A": 100.0 * np.cumprod(1 + rets)}, index=idx)
    weights = pd.DataFrame({"A": 1.0}, index=idx)
    params = {"short_window": 7, "long_window": 30, "threshold": 1.4, "high_vol_multiplier": 0.25}
    out = RegimeReweightOverlay().apply(weights, {"prices": prices}, params)["A"]
    assert (out.iloc[:60] == 1.0).all()  # warm-up and calm bars: untouched
    assert out.iloc[60] == 0.25  # the spike's own bar
    assert (
        RegimeReweightOverlay()
        .apply(weights, {"prices": prices}, {**params, "reference_symbol": "A"})
        .equals(out.to_frame())
    )


def test_corr_gross_cap_gates_only_correlated_multi_asset_bars():
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 60)
    prices = pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + r),
            "B": 100 * np.cumprod(1 + 2 * r),  # moves with A
            "C": 100 * np.cumprod(1 - r),  # moves against A
        },
        index=idx,
    )
    params = {"window": 10, "corr_threshold": 0.6, "max_gross": 1.0}
    together = pd.DataFrame({"A": 1.0, "B": 1.0, "C": 0.0}, index=idx)
    apart = pd.DataFrame({"A": 1.0, "B": 0.0, "C": 1.0}, index=idx)
    alone = pd.DataFrame({"A": 2.0, "B": 0.0, "C": 0.0}, index=idx)
    ov = CorrGrossCapOverlay()
    capped = ov.apply(together, {"prices": prices}, params)
    assert (capped.iloc[:10] == together.iloc[:10]).all().all()  # window not filled yet
    np.testing.assert_allclose(capped.iloc[10:].abs().sum(axis=1), 1.0)
    pd.testing.assert_frame_equal(ov.apply(apart, {"prices": prices}, params), apart)
    pd.testing.assert_frame_equal(ov.apply(alone, {"prices": prices}, params), alone)


# ----------------------------------------------------------------------
# From YAML, no Python: plugins schema, validate, run, manifest
# ----------------------------------------------------------------------


def _write_prices(tmp_path: Path) -> Path:
    long = _prices().rename_axis("date").reset_index().melt("date", var_name="symbol", value_name="close")
    path = tmp_path / "prices.parquet"
    long.to_parquet(path, index=False)
    return path


def _yaml_config(tmp_path: Path, overlays: list[dict] | None) -> dict:
    cfg = yaml.safe_load(f"""
run: {{mode: backtest, asof: "2024-02-09", pipeline: backtest.pipeline.v1}}
artifacts: {{root: "{tmp_path / "artifacts"}"}}
plugins:
  pipeline:
    name: backtest.pipeline.v1
    params: {{engine: vectorbt, fees: 0.0, universe: {{symbols: [A, USD]}}}}
  strategies:
    - name: strategy.static_weights.v1
      params_init: {{weights: {{A: 1.0}}}}
  data:
    name: local_file_data
    params_init: {{prices_path: "{_write_prices(tmp_path)}"}}
""")
    if overlays is not None:
        cfg["plugins"]["overlays"] = overlays
    return cfg


def test_every_overlay_is_in_plugins_schema_json():
    payload = json.loads(cli.invoke(app, ["plugins", "schema", "--json"]).output)
    overlays = {p["id"]: p for p in payload["plugins"] if p["group"] == "overlay"}
    assert set(overlays) == {"overlay.reversal_derisk.v1", "overlay.regime_reweight.v1", "overlay.corr_gross_cap.v1"}
    params = {r["name"] for r in overlays["overlay.reversal_derisk.v1"]["params"]}
    assert params == {"fast_span", "slow_span", "multiplier", "hold_bars"}


def test_a_chain_runs_from_yaml_and_the_manifests_traded_weights_reflect_it(tmp_path):
    chain = [
        {"name": "overlay.reversal_derisk.v1", "params": DERISK},
        {"name": "overlay.corr_gross_cap.v1", "params": {"window": 10}},
    ]
    cfg = _yaml_config(tmp_path, chain)
    assert not [f.message for f in validate_config(cfg, PluginRegistry.discover()) if f.level == "error"]
    result = run_from_config(cfg, PluginRegistry.discover())
    run_dir = tmp_path / "artifacts" / result.run_id
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert validate_run_manifest(manifest) == []
    assert [(o["name"], o["params"]) for o in manifest["overlays"]] == [(c["name"], c["params"]) for c in chain]
    assert [p["name"] for p in manifest["plugin_versions"]["overlays"]] == [c["name"] for c in chain]
    traded = pd.read_parquet(run_dir / manifest["files"]["traded_weights"]).set_index("date")
    np.testing.assert_array_equal(traded["A"].to_numpy(), _expected_a(1))


def test_without_overlays_the_manifest_traded_weights_are_the_base_book(tmp_path):
    result = run_from_config(_yaml_config(tmp_path, None), PluginRegistry.discover())
    run_dir = tmp_path / "artifacts" / result.run_id
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["overlays"] == []
    traded = pd.read_parquet(run_dir / manifest["files"]["traded_weights"]).set_index("date")
    np.testing.assert_array_equal(traded["A"].to_numpy(), np.r_[0.0, np.ones(N - 1)])


def test_unknown_overlay_param_is_refused_by_validate(tmp_path):
    cfg = _yaml_config(tmp_path, [{"name": "overlay.reversal_derisk.v1", "params": {"hold_days": 5}}])
    msgs = [f.message for f in validate_config(cfg, PluginRegistry.discover()) if f.level == "error"]
    assert any("'hold_days'" in m and "overlay.reversal_derisk.v1" in m and "plugins.overlays[0]" in m for m in msgs)


def test_overlays_on_a_pipeline_that_does_not_apply_them_are_refused(tmp_path):
    cfg = _yaml_config(tmp_path, [{"name": "overlay.reversal_derisk.v1"}])
    cfg["plugins"]["pipeline"] = {"name": "trade.full_pipeline.v1", "params": {}}
    reg = PluginRegistry.discover()
    assert any("does not apply overlays" in f.message for f in validate_config(cfg, reg) if f.level == "error")
    with pytest.raises(ValueError, match="does not apply overlays"):
        run_from_config(cfg, reg)


# ----------------------------------------------------------------------
# H22 (Quark's blocked reversal_derisk) is a config change
# ----------------------------------------------------------------------


def test_h22_config_validates_with_the_cli():
    result = cli.invoke(app, ["validate", "-c", str(H22_CONFIG)])
    assert result.exit_code == 0, result.output


def test_h22_config_runs_and_the_overlay_changes_the_traded_book(tmp_path):
    """The config as committed, with only its repo-relative paths anchored (pytest's cwd may differ)."""
    cfg = yaml.safe_load(H22_CONFIG.read_text())
    cfg["artifacts"]["root"] = str(tmp_path / "artifacts")
    data_init = cfg["plugins"]["data"]["params_init"]
    data_init["prices_path"] = str(REPO / data_init["prices_path"])
    result = run_from_config(cfg, PluginRegistry.discover(), config_path=H22_CONFIG)
    run_dir = tmp_path / "artifacts" / result.run_id
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert validate_run_manifest(manifest) == []
    assert [o["name"] for o in manifest["overlays"]] == ["overlay.reversal_derisk.v1"]
    base = pd.read_parquet(manifest["artifacts"]["base_weights_history"]).set_index("date")
    decided = pd.read_parquet(manifest["artifacts"]["weights_history"]).set_index("date")
    changed = int(((base.fillna(0.0) - decided).abs() > 1e-12).to_numpy().sum())
    assert changed > 0, "the overlay never fired on the H22 config — the run proves nothing"
