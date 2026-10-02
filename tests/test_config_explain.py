"""``quantbox config explain --json`` (TOM-1362): what the runner will do with a config, without running it.

The contract: on every field ``explain`` shares with the run@1 manifest, the plan
and the record of an actual run AGREE — for vectorbt, rsims, a dataset pinned in
datasets.lock and one named by inline paths. A config the runner could not run
(unknown plugin, unresolvable dataset) exits non-zero with the reason.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest
import yaml
from test_dataset_resolve import _build, fake_datasets, root  # noqa: F401 — fixtures
from test_run_manifest_v1 import _config
from typer.testing import CliRunner

from quantbox.cli import app
from quantbox.explain import SCHEMA_ID, SHARED_FIELDS, explain_config, validate_explain
from quantbox.registry import PluginRegistry
from quantbox.runner import run_from_config


def _lock_config(tmp_path: Path, ds_root: Path, engine: str) -> tuple[dict, Path]:
    sha = _build(ds_root, "perp-daily", market="perp", funding=True)
    lab = tmp_path / "lab"
    lab.mkdir()
    (lab / "datasets.lock").write_text(yaml.safe_dump({"perp-daily": sha}))
    cfg = yaml.safe_load(f"""
run: {{mode: backtest, asof: "2024-02-20", pipeline: backtest.pipeline.v1, n_trials: 3}}
artifacts: {{root: "{tmp_path / "artifacts"}"}}
plugins:
  pipeline:
    name: backtest.pipeline.v1
    params:
      engine: {engine}
      fees: 0.0
      execution: {{lag_bars: 2}}
      venue: {{allow_shorts: true}}
      universe: {{symbols: [AAA, BBB]}}
  strategies:
    - name: strategy.static_weights.v1
      weight: 1.0
      params_init: {{weights: {{AAA: 0.5, BBB: -0.5}}}}
  data:
    name: local_file_data
    params_init: {{dataset: perp-daily}}
""")
    config_path = lab / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return cfg, config_path


def _inline_config(tmp_path: Path, engine: str) -> tuple[dict, Path]:
    cfg = _config(tmp_path, engine, n_trials=5)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return cfg, config_path


def _shared(doc: dict) -> dict:
    out = {k: doc.get(k) for k in SHARED_FIELDS}
    out["dataset"] = {k: (doc["dataset"] or {}).get(k) for k in ("name", "sha256", "source", "tier")}
    out["config_sha256"] = doc["config"]["sha256"]
    return out


def _explain_and_run(cfg: dict, config_path: Path) -> tuple[dict, dict]:
    registry = PluginRegistry.discover()
    planned = explain_config(copy.deepcopy(cfg), registry, config_path=config_path)
    result = run_from_config(copy.deepcopy(cfg), registry, config_path=config_path)
    run_dir = Path(cfg["artifacts"]["root"]) / result.run_id
    return planned, json.loads((run_dir / "run_manifest.json").read_text())


@pytest.mark.parametrize("engine", ["vectorbt", "rsims"])
def test_explain_agrees_with_the_run_manifest_inline_paths(tmp_path, engine):
    cfg, config_path = _inline_config(tmp_path, engine)
    planned, recorded = _explain_and_run(cfg, config_path)

    assert planned["ok"] is True, planned["errors"]
    assert validate_explain(planned) == []
    assert _shared(planned) == _shared(recorded)
    # The facts the pre-flight needs are really there, not agreeing Nones.
    assert planned["engine"]["name"] == engine
    assert planned["dataset"]["source"] == "inline" and planned["dataset"]["sha256"]
    assert planned["funding"]["modelled"] is (engine == "rsims")
    assert planned["funding"]["sha256"]
    assert planned["execution"]["lag_bars"] == 1
    assert planned["venue"] == {"declared": True, "allow_shorts": False, "max_leverage": 1.5}
    assert planned["n_trials"] == 5
    assert Path(recorded["artifacts"]["returns"]).resolve().parent.parent == Path(planned["artifacts_root"])


@pytest.mark.parametrize("engine", ["vectorbt", "rsims"])
def test_explain_agrees_with_the_run_manifest_lock_pinned(tmp_path, root, fake_datasets, engine):  # noqa: F811
    cfg, config_path = _lock_config(tmp_path, root, engine)
    planned, recorded = _explain_and_run(cfg, config_path)

    assert planned["ok"] is True, planned["errors"]
    assert validate_explain(planned) == []
    assert _shared(planned) == _shared(recorded)
    assert planned["dataset"]["source"] == "lock" and planned["dataset"]["name"] == "perp-daily"
    assert planned["dataset"]["market"] == "perp"
    assert planned["funding"]["source_path"] == str(root / "perp-daily" / "funding_rates.parquet")
    assert planned["funding"]["modelled"] is (engine == "rsims")
    assert planned["execution"]["lag_bars"] == 2
    assert planned["venue"]["allow_shorts"] is True and planned["venue"]["max_leverage"] == 99.0


def test_explain_reads_no_data_and_writes_no_artifacts(tmp_path, root, fake_datasets):  # noqa: F811
    cfg, config_path = _lock_config(tmp_path, root, "rsims")
    planned = explain_config(cfg, PluginRegistry.discover(), config_path=config_path)

    assert planned["ok"] is True, planned["errors"]
    assert fake_datasets == []  # resolved against the lock, never loaded
    assert not (tmp_path / "artifacts").exists()


def test_explain_lists_strategies_with_resolved_params_and_every_plugin_id(tmp_path):
    cfg, config_path = _inline_config(tmp_path, "vectorbt")
    planned = explain_config(cfg, PluginRegistry.discover(), config_path=config_path)

    [strategy] = planned["strategies"]
    assert strategy["name"] == "strategy.static_weights.v1"
    assert strategy["weight"] == 1.0
    assert strategy["params_init"]["weights"] == {"A": 1.0}
    assert planned["plugins_resolved"] is True
    assert {p["role"] for p in planned["plugin_ids"]} == {"pipeline", "data", "strategies[0]"}


def test_explain_lists_variant_strategies_named_by_a_bare_string_or_a_spec(tmp_path):
    # The runner accepts `strategy: <registry id>` as well as `strategy: {name: ...}`;
    # explain must report both, not crash on the string form.
    cfg, config_path = _inline_config(tmp_path, "vectorbt")
    cfg["plugins"]["pipeline"]["params"]["variants"] = [
        {"name": "bare", "strategy": "strategy.static_weights.v1"},
        {"name": "spec", "strategy": {"name": "strategy.static_weights.v1", "params": {"x": 1}}},
    ]
    planned = explain_config(cfg, PluginRegistry.discover(), config_path=config_path)

    assert planned["ok"] is True, planned["errors"]
    assert validate_explain(planned) == []
    by_variant = {s["variant"]: s for s in planned["strategies"] if "variant" in s}
    assert set(by_variant) == {"bare", "spec"}
    assert by_variant["bare"]["name"] == "strategy.static_weights.v1"
    assert by_variant["bare"]["params"] == {}
    assert by_variant["spec"]["params"] == {"x": 1}
    assert {"variants[bare]", "variants[spec]"} <= {p["role"] for p in planned["plugin_ids"]}


def test_explain_resolves_source_strategies_and_source_variants(tmp_path):
    # TOM-1363 made `source: file.py:Class` valid for a strategy AND a variant;
    # explain's pre-flight must resolve it as the runner does, not look it up by name.
    from test_arms import _write_base

    base_path = _write_base(tmp_path)
    cfg = yaml.safe_load(base_path.read_text())
    source = cfg["plugins"]["strategies"][0]["source"]
    cfg["plugins"]["pipeline"]["params"]["variants"] = [
        {"name": "half", "strategy": {"source": source, "params": {"frac": 0.5}}},
    ]
    planned = explain_config(cfg, PluginRegistry.discover(), config_path=base_path)

    assert planned["ok"] is True, planned["errors"]
    assert validate_explain(planned) == []
    ids = {p["role"]: p for p in planned["plugin_ids"]}
    assert ids["strategies[0]"]["name"] == source and ids["strategies[0]"]["resolved"] is True
    assert ids["variants[half]"]["name"] == source and ids["variants[half]"]["resolved"] is True
    names = {s.get("variant"): s["name"] for s in planned["strategies"]}
    assert names[None] == names["half"]


def _cli(config_path: Path) -> tuple[int, dict, str]:
    result = CliRunner().invoke(app, ["config", "explain", str(config_path), "--json"])
    return result.exit_code, json.loads(result.stdout), result.stderr


def test_cli_prints_the_plan_as_json(tmp_path):
    _, config_path = _inline_config(tmp_path, "rsims")
    code, doc, _ = _cli(config_path)

    assert code == 0
    assert doc["schema"] == SCHEMA_ID == "quantbox/explain@1"
    assert doc["engine"]["name"] == "rsims"
    assert validate_explain(doc) == []


def test_cli_unresolvable_plugin_exits_non_zero_with_the_reason(tmp_path):
    cfg, config_path = _inline_config(tmp_path, "vectorbt")
    cfg["plugins"]["strategies"][0]["name"] = "strategy.does_not_exist.v1"
    config_path.write_text(yaml.safe_dump(cfg))
    code, doc, stderr = _cli(config_path)

    assert code == 1
    assert doc["ok"] is False and doc["plugins_resolved"] is False
    assert any("strategy.does_not_exist.v1" in e for e in doc["errors"])
    assert "strategy.does_not_exist.v1" in stderr
    assert validate_explain(doc) == []


def test_cli_unresolvable_dataset_exits_non_zero_with_the_reason(tmp_path, root, fake_datasets):  # noqa: F811
    cfg, config_path = _lock_config(tmp_path, root, "vectorbt")
    (config_path.parent / "datasets.lock").write_text(yaml.safe_dump({"perp-daily": "0" * 64}))
    code, doc, stderr = _cli(config_path)

    assert code == 1
    assert doc["ok"] is False
    assert any("perp-daily" in e and "0" * 64 in e for e in doc["errors"])
    assert "perp-daily" in stderr


def test_cli_malformed_execution_block_exits_non_zero(tmp_path):
    cfg, config_path = _inline_config(tmp_path, "vectorbt")
    cfg["plugins"]["pipeline"]["params"]["execution"] = {"lag_bar": 0}
    config_path.write_text(yaml.safe_dump(cfg))
    code, doc, _ = _cli(config_path)

    assert code == 1
    assert any("lag_bar" in e for e in doc["errors"])


# ---------------------------------------------------------------------------
# Explain and run AGREE on refusals: a config the run refuses, explain refuses,
# from the SAME check rather than a copy of it. Each case below once said
# `ok: true` in explain and failed only after the run had loaded its data.
# ---------------------------------------------------------------------------


def _both_refuse(cfg: dict, config_path: Path, needle: str) -> dict:
    """Explain says ok=False naming *needle*, and the run raises naming it too."""
    registry = PluginRegistry.discover()
    planned = explain_config(copy.deepcopy(cfg), registry, config_path=config_path)
    assert planned["ok"] is False, f"explain said ok for a config the run refuses ({needle})"
    assert any(re.search(needle, e) for e in planned["errors"]), planned["errors"]
    assert validate_explain(planned) == []
    with pytest.raises(Exception, match=needle):
        run_from_config(copy.deepcopy(cfg), registry, config_path=config_path)
    return planned


def _variant(name: str = "v1", **extra) -> dict:
    strategy = {"name": "strategy.static_weights.v1", "params_init": {"weights": {"A": 1.0}}}
    return {"name": name, "strategy": strategy, **extra}


def test_variants_under_rsims_are_refused_by_explain_and_run(tmp_path):
    cfg, config_path = _inline_config(tmp_path, "rsims")
    cfg["plugins"]["pipeline"]["params"]["variants"] = [_variant()]
    _both_refuse(cfg, config_path, "engine='vectorbt' only")


@pytest.mark.parametrize("where", ["overrides", "variant"])
@pytest.mark.parametrize("key", ["execution", "venue"])
def test_run_level_keys_in_a_variant_are_refused_by_explain_and_run(tmp_path, where, key):
    cfg, config_path = _inline_config(tmp_path, "vectorbt")
    block = {"lag_bars": 2} if key == "execution" else {"allow_shorts": True}
    v = _variant(overrides={key: block}) if where == "overrides" else _variant(**{key: block})
    cfg["plugins"]["pipeline"]["params"]["variants"] = [v]
    _both_refuse(cfg, config_path, f"'{key}' is run-level")


def test_variant_allow_short_against_a_declared_venue_is_refused_by_explain_and_run(tmp_path):
    cfg, config_path = _inline_config(tmp_path, "vectorbt")  # declares venue
    cfg["plugins"]["pipeline"]["params"]["variants"] = [_variant(overrides={"risk": {"allow_short": True}})]
    _both_refuse(cfg, config_path, "conflicts with the run-level `venue` block")


def test_a_missing_inline_prices_file_is_refused_by_explain_and_run(tmp_path):
    cfg, config_path = _inline_config(tmp_path, "vectorbt")
    cfg["plugins"]["data"]["params_init"]["prices_path"] = str(tmp_path / "nope.parquet")
    planned = _both_refuse(cfg, config_path, r"nope\.parquet")
    assert any(e.startswith("dataset:") for e in planned["errors"])


def test_no_prices_source_at_all_is_refused_by_explain_and_run(tmp_path):
    cfg, config_path = _inline_config(tmp_path, "vectorbt")
    del cfg["plugins"]["data"]["params_init"]["prices_path"]
    _both_refuse(cfg, config_path, "no prices source")


def test_an_unparseable_frequency_is_refused_by_explain_and_run(tmp_path):
    cfg, config_path = _inline_config(tmp_path, "vectorbt")
    cfg["plugins"]["pipeline"]["params"]["frequency"] = "every-tuesday"
    _both_refuse(cfg, config_path, "every-tuesday")


@pytest.mark.parametrize("key", ["fees", "fixed_fees", "slippage", "trading_days"])
def test_a_non_numeric_cost_is_refused_by_explain_and_run(tmp_path, key):
    cfg, config_path = _inline_config(tmp_path, "vectorbt")
    cfg["plugins"]["pipeline"]["params"][key] = "lots"
    _both_refuse(cfg, config_path, f"'{key}'")


@pytest.mark.parametrize("key", ["trade_buffer", "initial_cash", "margin"])
def test_a_non_numeric_rsims_param_is_refused_by_explain_and_run(tmp_path, key):
    cfg, config_path = _inline_config(tmp_path, "rsims")
    cfg["plugins"]["pipeline"]["params"][key] = "lots"
    _both_refuse(cfg, config_path, f"'{key}'")


def test_a_non_numeric_variant_override_is_refused_by_explain_and_run(tmp_path):
    cfg, config_path = _inline_config(tmp_path, "vectorbt")
    cfg["plugins"]["pipeline"]["params"]["variants"] = [_variant(overrides={"fees": "lots"})]
    _both_refuse(cfg, config_path, "Variant 'v1'.*'fees'")


@pytest.mark.parametrize("value", ["yes", "no", 1])
def test_a_non_boolean_full_report_is_refused_by_explain_and_run(tmp_path, value):
    # TOM-1365: a truthy "no" would otherwise write the tens-of-MB HTML report.
    cfg, config_path = _inline_config(tmp_path, "vectorbt")
    cfg["plugins"]["pipeline"]["params"]["full_report"] = value
    _both_refuse(cfg, config_path, "'full_report' must be true or false")


def test_plan_records_full_report_off_by_default():
    from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline

    assert BacktestPipeline().plan({})["full_report"] is False
    assert BacktestPipeline().plan({"full_report": True})["full_report"] is True


def test_a_missing_engine_extra_is_refused_by_explain_and_run(tmp_path, monkeypatch):
    from quantbox.plugins.pipeline import backtest_pipeline

    monkeypatch.setattr(backtest_pipeline, "_engine_installed", lambda engine: engine != "vectorbt")
    cfg, config_path = _inline_config(tmp_path, "vectorbt")
    _both_refuse(cfg, config_path, r"\[vectorbt\] extra")


def test_strict_mode_on_a_raw_dataset_is_refused_by_explain_and_run(tmp_path):
    cfg, config_path = _inline_config(tmp_path, "vectorbt")  # inline paths: tier raw
    cfg["run"]["strict"] = True
    _both_refuse(cfg, config_path, "strict mode rejects")


def test_overlays_on_a_pipeline_that_does_not_apply_them_are_refused_by_explain_and_run(tmp_path):
    cfg, config_path = _inline_config(tmp_path, "vectorbt")
    cfg["plugins"]["pipeline"]["name"] = cfg["run"]["pipeline"] = "trade.full_pipeline.v1"
    cfg["plugins"]["overlays"] = [{"name": "overlay.regime_reweight.v1"}]
    _both_refuse(cfg, config_path, "does not apply overlays")


DERISK = {"fast_span": 2, "slow_span": 8, "multiplier": 0.5, "hold_bars": 5}


@pytest.mark.parametrize("engine", ["vectorbt", "rsims"])
def test_explain_reports_the_overlay_chain_the_run_records(tmp_path, engine):
    cfg, config_path = _inline_config(tmp_path, engine)
    cfg["plugins"]["overlays"] = [
        {"name": "overlay.reversal_derisk.v1", "params": DERISK},
        {"name": "overlay.regime_reweight.v1"},
    ]
    planned, recorded = _explain_and_run(cfg, config_path)

    assert planned["ok"] is True, planned["errors"]
    assert validate_explain(planned) == []
    assert "overlays" in SHARED_FIELDS
    assert _shared(planned) == _shared(recorded)
    assert [o["name"] for o in planned["overlays"]] == ["overlay.reversal_derisk.v1", "overlay.regime_reweight.v1"]
    assert planned["overlays"][0]["params"] == DERISK
    ids = {p["role"]: p for p in planned["plugin_ids"]}
    assert ids["overlays[0]"]["group"] == "overlay" and ids["overlays[0]"]["resolved"] is True


def test_explain_reports_an_empty_overlay_chain_as_the_run_does(tmp_path):
    cfg, config_path = _inline_config(tmp_path, "vectorbt")
    planned, recorded = _explain_and_run(cfg, config_path)
    assert planned["overlays"] == recorded["overlays"] == []


def test_an_unknown_overlay_id_is_refused_by_explain_and_run(tmp_path):
    cfg, config_path = _inline_config(tmp_path, "vectorbt")
    cfg["plugins"]["overlays"] = [{"name": "overlay.does_not_exist.v1"}]
    planned = _both_refuse(cfg, config_path, "overlay.does_not_exist.v1")
    assert planned["plugins_resolved"] is False
