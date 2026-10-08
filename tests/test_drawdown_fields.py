"""Explicit drawdown fields (TOM-1627): ``max_drawdown`` is signed, ``max_drawdown_abs`` is its depth.

Tom, 2026-10-08: "musimy mieć dd i abs dd, żeby było jasne". Before this, the metrics
reported a drawdown as a negative fraction and the gate JSON as a positive depth, under
names that did not say which. Every drawdown output now carries BOTH names:

- ``max_drawdown``      <= 0, the signed fraction (-0.25 = a 25% drawdown);
- ``max_drawdown_abs``  >= 0, the same drawdown as a positive depth (0.25).

No existing key changes value: the metrics' ``max_drawdown`` stays negative, the gate
JSON's positive keys (the ``max_drawdown`` leg value, ``episode.depth``) stay positive.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantbox import metrics
from quantbox.finding_export import export_finding_report
from quantbox.gates import episode_gate, paired_block_bootstrap
from quantbox.registry import PluginRegistry
from quantbox.run_manifest import load_run_schema, validate_run_manifest
from quantbox.runner import run_from_config

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_run"


def _returns(n: int = 300, seed: int = 4, drift: float = 0.0005) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(drift, 0.02, n), index=pd.date_range("2023-01-01", periods=n, freq="D"))


def _assert_pair(block: dict, signed: float | None = None) -> None:
    assert block["max_drawdown"] <= 0, block
    assert block["max_drawdown_abs"] >= 0, block
    assert block["max_drawdown_abs"] == pytest.approx(-block["max_drawdown"], abs=1e-15)
    if signed is not None:
        assert block["max_drawdown"] == pytest.approx(signed, abs=1e-15)


# =========================================================================== metrics


def test_the_metrics_dict_carries_the_signed_drawdown_and_its_depth():
    r = _returns()
    m = metrics.compute_backtest_metrics(r)
    dd = metrics.compute_drawdown_series((1 + r).cumprod()).min()
    _assert_pair(m, signed=float(dd))
    assert m["max_drawdown"] < 0


def test_a_series_with_no_drawdown_carries_zero_in_both():
    r = pd.Series([0.01] * 10, index=pd.date_range("2024-01-01", periods=10, freq="D"))
    m = metrics.compute_backtest_metrics(r)
    assert m["max_drawdown"] == 0.0
    assert m["max_drawdown_abs"] == 0.0


# =========================================================================== run@1


def _inputs(root: Path, n: int = 120) -> Path:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(9)
    prices = pd.DataFrame({"A": 100.0 * np.cumprod(1 + rng.normal(0.0, 0.02, n)), "B": 50.0}, index=idx)
    path = root / "prices.parquet"
    prices.rename_axis("date").reset_index().melt("date", var_name="symbol", value_name="close").to_parquet(
        path, index=False
    )
    return path


@pytest.mark.parametrize("engine", ["vectorbt", "rsims"])
def test_run_metrics_and_the_run_manifest_carry_both(tmp_path, engine):
    cfg = {
        "run": {"mode": "backtest", "asof": "2024-04-29", "pipeline": "backtest.pipeline.v1"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "plugins": {
            "pipeline": {
                "name": "backtest.pipeline.v1",
                "params": {"engine": engine, "fees": 0.0, "universe": {"symbols": ["A", "B"]}},
            },
            "strategies": [
                {"name": "strategy.static_weights.v1", "weight": 1.0, "params_init": {"weights": {"A": 1.0}}}
            ],
            "data": {"name": "local_file_data", "params_init": {"prices_path": str(_inputs(tmp_path))}},
        },
    }
    result = run_from_config(cfg, PluginRegistry.discover())
    run_dir = tmp_path / "artifacts" / result.run_id
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    on_disk = json.loads((run_dir / manifest["files"]["metrics"]).read_text())

    _assert_pair(result.metrics)
    _assert_pair(manifest["metrics"])
    _assert_pair(on_disk)
    assert manifest["metrics"]["max_drawdown"] < 0
    assert validate_run_manifest(manifest) == []


def test_the_run_schema_declares_both_signs_in_a_minor_bump():
    schema = load_run_schema()
    assert schema["x-schema-minor"] == 8
    props = schema["properties"]["metrics"]["properties"]
    assert props["max_drawdown"]["maximum"] == 0
    assert props["max_drawdown_abs"]["minimum"] == 0


def test_the_run_schema_refuses_a_positive_signed_drawdown():
    manifest = json.loads((GOLDEN / "run_manifest.json").read_text())
    manifest["metrics"] = {"max_drawdown": 0.2, "max_drawdown_abs": 0.2}
    assert validate_run_manifest(manifest)


# =========================================================================== gate JSON


def test_the_episode_carries_both_and_keeps_its_positive_depth():
    r = _returns().to_numpy()
    out = episode_gate(r)
    episode = out["episode"]
    _assert_pair(episode)
    assert episode["depth"] > 0  # the existing key, unchanged
    assert episode["max_drawdown_abs"] == episode["depth"]


def test_a_max_drawdown_episode_leg_carries_both_for_each_series_and_keeps_its_value():
    c, b = _returns(seed=1).to_numpy(), _returns(seed=2).to_numpy()
    out = episode_gate(c, b, metric="max_drawdown", compare="ratio", pass_if="below", threshold=0.7)
    for leg in ("full", "ex_episode"):
        drawdowns = out[leg]["drawdowns"]
        _assert_pair(drawdowns["candidate"])
        _assert_pair(drawdowns["baseline"])
    full = out["full"]["drawdowns"]
    # the leg value is unchanged: a ratio of positive depths
    assert out["full"]["value"] == pytest.approx(
        full["candidate"]["max_drawdown_abs"] / full["baseline"]["max_drawdown_abs"]
    )


def test_a_single_series_max_drawdown_leg_value_is_its_depth():
    r = _returns().to_numpy()
    out = episode_gate(r, metric="max_drawdown", pass_if="below", threshold=0.5)
    drawdowns = out["full"]["drawdowns"]
    assert drawdowns["baseline"] is None
    assert out["full"]["value"] == pytest.approx(drawdowns["candidate"]["max_drawdown_abs"])
    assert out["full"]["value"] > 0


def test_a_max_drawdown_bootstrap_carries_both_on_the_full_sample():
    c, b = _returns(seed=1).to_numpy(), _returns(seed=2).to_numpy()
    out = paired_block_bootstrap(
        c, b, metric="max_drawdown", compare="diff", pass_if="below", threshold=0.0, draws=50, seed=1
    )
    drawdowns = out["drawdowns"]
    _assert_pair(drawdowns["candidate"])
    _assert_pair(drawdowns["baseline"])
    assert out["point_estimate"] == pytest.approx(
        drawdowns["candidate"]["max_drawdown_abs"] - drawdowns["baseline"]["max_drawdown_abs"]
    )


def test_a_sharpe_leg_reports_no_drawdowns():
    c, b = _returns(seed=1).to_numpy(), _returns(seed=2).to_numpy()
    assert "drawdowns" not in paired_block_bootstrap(c, b, draws=20)
    assert "drawdowns" not in episode_gate(c, b)["full"]


# =========================================================================== finding_report export


def test_the_export_carries_both_rows_and_both_cards_even_for_a_run_written_before():
    metrics_json = json.loads((GOLDEN / "metrics.json").read_text())
    assert "max_drawdown_abs" not in metrics_json  # written before TOM-1627
    payload = export_finding_report(GOLDEN)

    rows = {r["label"]: r["values"] for r in payload["metrics"]["rows"]}
    assert rows["Max drawdown"] == [pytest.approx(metrics_json["max_drawdown"])]
    assert rows["Max drawdown (depth)"] == [pytest.approx(-metrics_json["max_drawdown"])]
    cards = {k["label"]: k["value"] for k in payload["kpis"]}
    assert cards["Max drawdown"] == pytest.approx(metrics_json["max_drawdown"])
    assert cards["Max drawdown (depth)"] == pytest.approx(-metrics_json["max_drawdown"])
