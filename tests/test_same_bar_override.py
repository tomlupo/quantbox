"""The explicit same-bar override (TOM-1399, docs/adr/0006).

``execution.lag_bars: 0`` is refused unless the SAME block also carries
``same_bar: {allow: true, reason: "<non-empty>"}`` (helpers:
``allow_same_bar=True, same_bar_reason=...``). With it the run goes ahead, the
run@1 manifest stamps ``execution.same_bar: true`` plus the reason, and the run
is classified ``run.kind: research`` — never a backtest — wherever a result is
read back: the manifest, ``config explain``, the finding-report export, the
gates.

Same toy as ``test_execution_timing``: `A` jumps +10% on bar ``J``. A weight
decided on ``J-1`` earns the jump only when filled same-bar, so the number
itself says which timing ran.

THE GATE IS ``quantbox.execution._check_lag``. Deleting its override condition
(accepting ``lag == 0`` unconditionally) turns every ``*_refused_*`` test here
red — that is what they are for.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from test_execution_timing import BRANCHES, JUMP, J, _prices, _run_pipeline, _weights_decided_on
from typer.testing import CliRunner

from quantbox.cli import app
from quantbox.execution import apply_execution_lag, resolve_execution, resolve_lag_bars
from quantbox.explain import explain_config, validate_explain
from quantbox.finding_export import export_finding_report
from quantbox.plugins.backtesting import backtest, optimize
from quantbox.registry import PluginRegistry
from quantbox.run_manifest import validate_run_manifest
from quantbox.runner import run_from_config

REFUSED = "lag_bars must be >= 1"
NAMES_THE_OVERRIDE = "same_bar: {allow: true, reason:"
REASON = "monthly-only data: the month-end close is the only price, so same-bar is closer to reality"
OVERRIDE = {"lag_bars": 0, "same_bar": {"allow": True, "reason": REASON}}


# ----------------------------------------------------------------------
# the resolver: ONE gate
# ----------------------------------------------------------------------


def test_same_bar_without_the_override_is_refused_and_the_error_names_the_override():
    with pytest.raises(ValueError, match=REFUSED) as exc:
        resolve_execution({"lag_bars": 0})
    assert NAMES_THE_OVERRIDE in str(exc.value)


def test_the_override_lets_same_bar_through_and_carries_its_reason():
    timing = resolve_execution(copy.deepcopy(OVERRIDE))
    assert timing.lag_bars == 0
    assert timing.same_bar is not None and timing.same_bar.reason == REASON
    assert resolve_lag_bars(copy.deepcopy(OVERRIDE)) == 0


@pytest.mark.parametrize(
    "same_bar",
    [
        {"allow": True, "reason": ""},
        {"allow": True, "reason": "   "},
        {"allow": True},
        {"allow": True, "reason": None},
        {"allow": True, "reason": 7},
    ],
)
def test_an_empty_or_missing_reason_is_refused(same_bar):
    with pytest.raises(ValueError, match="reason"):
        resolve_execution({"lag_bars": 0, "same_bar": same_bar})


@pytest.mark.parametrize(
    ("same_bar", "message"),
    [
        ({"allow": "yes", "reason": REASON}, "allow"),
        ({"allow": 1, "reason": REASON}, "allow"),
        ({"reason": REASON}, "allow"),
        ({"allow": True, "reason": REASON, "because": "x"}, "unknown key"),
        (True, "mapping"),
    ],
)
def test_a_malformed_override_is_refused(same_bar, message):
    with pytest.raises(ValueError, match=message):
        resolve_execution({"lag_bars": 0, "same_bar": same_bar})


def test_a_withheld_override_does_not_allow_same_bar():
    with pytest.raises(ValueError, match=REFUSED):
        resolve_execution({"lag_bars": 0, "same_bar": {"allow": False, "reason": REASON}})


def test_an_override_on_a_next_bar_run_is_refused():
    """An override with nothing to override would classify a next-bar run as research."""
    with pytest.raises(ValueError, match="only with lag_bars 0"):
        resolve_execution({"lag_bars": 1, "same_bar": {"allow": True, "reason": REASON}})
    with pytest.raises(ValueError, match="only with lag_bars 0"):
        resolve_execution({"same_bar": {"allow": True, "reason": REASON}})


def test_a_negative_lag_is_refused_even_with_the_override():
    with pytest.raises(ValueError, match=REFUSED):
        resolve_execution({"lag_bars": -1, "same_bar": {"allow": True, "reason": REASON}})


def test_apply_execution_lag_needs_the_resolved_override_for_lag_zero():
    w = _weights_decided_on(3)
    with pytest.raises(ValueError, match=REFUSED):
        apply_execution_lag(w, 0)
    same = apply_execution_lag(w, 0, same_bar=resolve_execution(copy.deepcopy(OVERRIDE)).same_bar)
    pd.testing.assert_frame_equal(same, w)


# ----------------------------------------------------------------------
# the pipeline (`quantbox run -c`, BacktestPipeline)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("branch", BRANCHES)
def test_the_pipeline_refuses_same_bar_without_the_override(tmp_path, branch):
    with pytest.raises(ValueError, match=REFUSED):
        _run_pipeline(tmp_path, {**BRANCHES[branch], "execution": {"lag_bars": 0}}, decided_on=J - 1)
    assert not list(tmp_path.rglob("*.parquet"))


@pytest.mark.parametrize("branch", BRANCHES)
def test_with_the_override_every_branch_fills_same_bar(tmp_path, branch):
    result, store = _run_pipeline(
        tmp_path, {**BRANCHES[branch], "execution": copy.deepcopy(OVERRIDE)}, decided_on=J - 1
    )
    assert result.metrics["total_return"] == pytest.approx(JUMP, abs=1e-9)
    execution = result.notes["execution"]
    assert execution["lag_bars"] == 0
    assert execution["same_bar"] is True
    assert execution["same_bar_reason"] == REASON
    assert execution["description"].startswith("same-bar (lag_bars=0)")
    assert "RESEARCH" in (store.root / "summary.md").read_text()


# ----------------------------------------------------------------------
# validate, config explain, run — through the CLI and the runner
# ----------------------------------------------------------------------


def _inputs(root: Path, n: int = 120) -> Path:
    import numpy as np

    idx = pd.date_range("2023-06-01", periods=n, freq="D")
    rng = np.random.default_rng(3)
    prices = pd.DataFrame(
        {"A": 100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n)), "B": 50.0 * np.cumprod(1 + rng.normal(0, 0.02, n))},
        index=idx,
    )
    path = root / "prices.parquet"
    prices.rename_axis("date").reset_index().melt("date", var_name="symbol", value_name="close").to_parquet(
        path, index=False
    )
    return path


def _config(tmp_path: Path, execution: dict | None, **run: object) -> tuple[dict, Path]:
    params = {"engine": "vectorbt", "fees": 0.0, "universe": {"symbols": ["A", "B"]}}
    if execution is not None:
        params["execution"] = execution
    cfg = {
        "run": {"mode": "backtest", "asof": "2023-09-28", "pipeline": "backtest.pipeline.v1", **run},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "plugins": {
            "pipeline": {"name": "backtest.pipeline.v1", "params": params},
            "strategies": [
                {"name": "strategy.static_weights.v1", "weight": 1.0, "params_init": {"weights": {"A": 1.0}}}
            ],
            "data": {"name": "local_file_data", "params_init": {"prices_path": str(_inputs(tmp_path))}},
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return cfg, path


def _cli(*args: str):
    return CliRunner().invoke(app, list(args))


@pytest.mark.parametrize("command", [("validate", "-c"), ("config", "explain"), ("run", "-c")])
def test_every_cli_entry_point_refuses_same_bar_without_the_override(tmp_path, command):
    _, path = _config(tmp_path, {"lag_bars": 0})
    result = _cli(*command, str(path))
    assert result.exit_code != 0, result.output
    text = result.output + (str(result.exception) if result.exception else "")
    assert "same_bar" in text, text
    assert not list((tmp_path / "artifacts").rglob("run_manifest.json"))


@pytest.mark.parametrize("command", [("validate", "-c"), ("config", "explain")])
def test_validate_and_explain_accept_the_override(tmp_path, command):
    _, path = _config(tmp_path, copy.deepcopy(OVERRIDE))
    result = _cli(*command, str(path))
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("bad", [{"lag_bars": 0, "same_bar": {"allow": True, "reason": ""}}, {"lag_bars": 0}])
def test_validate_and_explain_refuse_an_empty_or_absent_reason(tmp_path, bad):
    _, path = _config(tmp_path, bad)
    for command in (("validate", "-c"), ("config", "explain")):
        assert _cli(*command, str(path)).exit_code != 0


def test_the_params_schema_knows_same_bar():
    """`quantbox validate` with the params check ON: the override is a declared param, not a typo."""
    from quantbox.validate import validate_config

    cfg = {
        "run": {"mode": "backtest", "asof": "2024-01-01"},
        "artifacts": {},
        "plugins": {
            "pipeline": {"name": "backtest.pipeline.v1", "params": {"execution": copy.deepcopy(OVERRIDE)}},
            "data": {"name": "local_file_data"},
        },
    }
    assert [f.message for f in validate_config(cfg) if f.level == "error"] == []
    cfg["plugins"]["pipeline"]["params"]["execution"]["same_bar"]["typo"] = 1
    assert any("same_bar" in f.message for f in validate_config(cfg) if f.level == "error")


def test_explain_plans_a_research_run(tmp_path):
    cfg, path = _config(tmp_path, copy.deepcopy(OVERRIDE))
    doc = explain_config(cfg, PluginRegistry.discover(), config_path=path)
    assert doc["ok"] is True, doc["errors"]
    assert validate_explain(doc) == []
    assert doc["execution"]["same_bar"] is True and doc["execution"]["same_bar_reason"] == REASON
    assert doc["run"] == {"kind": "research"}


def _run(tmp_path: Path, execution: dict | None, **run: object) -> tuple[dict, Path]:
    cfg, path = _config(tmp_path, execution, **run)
    result = run_from_config(copy.deepcopy(cfg), PluginRegistry.discover(), config_path=path)
    run_dir = tmp_path / "artifacts" / result.run_id
    return json.loads((run_dir / "run_manifest.json").read_text()), run_dir


def test_the_runner_refuses_same_bar_without_the_override(tmp_path):
    from quantbox.exceptions import ConfigValidationError

    cfg, _ = _config(tmp_path, {"lag_bars": 0})
    with pytest.raises(ConfigValidationError, match=REFUSED):
        run_from_config(cfg, PluginRegistry.discover())


def test_a_same_bar_run_is_recorded_as_research(tmp_path):
    manifest, _ = _run(tmp_path, copy.deepcopy(OVERRIDE))
    assert validate_run_manifest(manifest) == []
    assert manifest["execution"]["lag_bars"] == 0
    assert manifest["execution"]["same_bar"] is True
    assert manifest["execution"]["same_bar_reason"] == REASON
    assert manifest["run"] == {"kind": "research"}


def test_a_next_bar_run_is_recorded_as_a_backtest(tmp_path):
    manifest, _ = _run(tmp_path, None)
    assert validate_run_manifest(manifest) == []
    assert manifest["execution"]["same_bar"] is False
    assert "same_bar_reason" not in manifest["execution"]
    assert manifest["run"] == {"kind": "backtest"}


def test_the_schema_refuses_a_same_bar_manifest_that_claims_to_be_a_backtest(tmp_path):
    manifest, _ = _run(tmp_path, copy.deepcopy(OVERRIDE))
    lying = {**manifest, "run": {"kind": "backtest"}}
    assert validate_run_manifest(lying) != []
    unclassified = {k: v for k, v in manifest.items() if k != "run"}
    assert validate_run_manifest(unclassified) != []
    no_reason = {**manifest, "execution": {k: v for k, v in manifest["execution"].items() if k != "same_bar_reason"}}
    assert validate_run_manifest(no_reason) != []


def test_strict_mode_refuses_a_research_run(tmp_path):
    cfg, path = _config(tmp_path, copy.deepcopy(OVERRIDE), strict=True)
    doc = explain_config(copy.deepcopy(cfg), PluginRegistry.discover(), config_path=path)
    assert doc["ok"] is False
    assert any("same-bar" in e and "RESEARCH" in e for e in doc["errors"])
    with pytest.raises(RuntimeError, match="strict mode refuses a same-bar run"):
        run_from_config(cfg, PluginRegistry.discover(), config_path=path)


# ----------------------------------------------------------------------
# backtest() and optimize()
# ----------------------------------------------------------------------


def _decided_fn(prices, params):
    return _weights_decided_on(params["decided_on"]).loc[prices.index]


def test_backtest_refuses_same_bar_without_the_override():
    with pytest.raises(ValueError, match=REFUSED) as exc:
        backtest(_prices(), _weights_decided_on(J - 1), lag_bars=0)
    assert "allow_same_bar" in str(exc.value) or NAMES_THE_OVERRIDE in str(exc.value)


@pytest.mark.parametrize("reason", [None, "", "  "])
def test_backtest_refuses_the_override_without_a_reason(reason):
    with pytest.raises(ValueError, match="reason"):
        backtest(_prices(), _weights_decided_on(J - 1), lag_bars=0, allow_same_bar=True, same_bar_reason=reason)


def test_backtest_refuses_an_override_on_a_next_bar_run():
    with pytest.raises(ValueError, match="only with lag_bars 0"):
        backtest(_prices(), _weights_decided_on(J - 1), allow_same_bar=True, same_bar_reason=REASON)


def test_backtest_with_the_override_fills_same_bar_and_says_research():
    result = backtest(
        _prices(), _weights_decided_on(J - 1), fees=0.0, lag_bars=0, allow_same_bar=True, same_bar_reason=REASON
    )
    assert result["metrics"]["total_return"] == pytest.approx(JUMP, abs=1e-9)
    assert result["execution"]["same_bar"] is True and result["execution"]["same_bar_reason"] == REASON
    assert result["run"] == {"kind": "research"}
    assert backtest(_prices(), _weights_decided_on(J - 1), fees=0.0)["run"] == {"kind": "backtest"}


def test_optimize_refuses_same_bar_without_the_override():
    with pytest.raises(ValueError, match=REFUSED):
        optimize(_prices(), _decided_fn, {"decided_on": [J - 1]}, fees=0.0, lag_bars=0)
    with pytest.raises(ValueError, match="reason"):
        optimize(_prices(), _decided_fn, {"decided_on": [J - 1]}, fees=0.0, lag_bars=0, allow_same_bar=True)


def test_optimize_with_the_override_fills_same_bar_and_says_research():
    result = optimize(
        _prices(),
        _decided_fn,
        {"decided_on": [J - 1]},
        metric="total_return",
        fees=0.0,
        lag_bars=0,
        allow_same_bar=True,
        same_bar_reason=REASON,
    )
    assert result["best_metric"] == pytest.approx(JUMP, abs=1e-9)
    assert result["execution"]["same_bar"] is True
    assert result["run"] == {"kind": "research"}


# ----------------------------------------------------------------------
# read back: the finding-report export and the gates
# ----------------------------------------------------------------------


def test_the_finding_report_export_marks_a_research_run(tmp_path):
    _, run_dir = _run(tmp_path, copy.deepcopy(OVERRIDE))
    payload = export_finding_report(run_dir)
    json.dumps(payload, allow_nan=False)

    # the hero card does NOT report the finding's backtest_sharpe
    assert payload["kpis"], payload
    assert not any(k.get("key") == "backtest_sharpe" for k in payload["kpis"])
    assert all("RESEARCH" in k["note"] and k["tone"] == "bad" for k in payload["kpis"])
    assert "RESEARCH" in payload["series"]["title"]

    prov = {r[0]: r[1:] for r in payload["tables"][0]["rows"]}
    assert prov["run.kind"] == ["research"]
    assert prov["execution.same_bar"] == ["true"]
    assert prov["execution.same_bar_reason"] == [REASON]

    (axis,) = [a for a in payload["audit"]["axes"] if a["name"] == "Execution timing"]
    assert axis["status"] == "fail" and REASON in axis["note"]
    for path, value in _leaves(payload):
        if not path.endswith(".benchmark"):
            assert not isinstance(value, bool), path


def test_the_finding_report_export_of_a_backtest_is_unchanged(tmp_path):
    _, run_dir = _run(tmp_path, None)
    payload = export_finding_report(run_dir)
    assert any(k.get("key") == "backtest_sharpe" for k in payload["kpis"])
    assert "audit" not in payload
    prov = {r[0]: r[1:] for r in payload["tables"][0]["rows"]}
    assert prov["run.kind"] == ["backtest"]


def _leaves(obj, path="$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, obj


def test_the_gates_mark_a_research_runs_returns(tmp_path):
    _, run_dir = _run(tmp_path, copy.deepcopy(OVERRIDE))
    returns = str(run_dir / "returns.parquet")
    out = _cli("gates", "nw", "--returns", returns, "--min-oos-periods", "10", "--json")
    assert out.exit_code in (0, 1), out.output
    verdict = json.loads(out.stdout)
    assert verdict["run_kind"] == "research"
    assert REASON in verdict["research_note"]
    text = _cli("gates", "nw", "--returns", returns, "--min-oos-periods", "10")
    assert "RESEARCH" in text.stdout


def test_the_gates_say_nothing_extra_for_a_backtests_returns(tmp_path):
    _, run_dir = _run(tmp_path, None)
    out = _cli("gates", "nw", "--returns", str(run_dir / "returns.parquet"), "--min-oos-periods", "10", "--json")
    verdict = json.loads(out.stdout)
    assert verdict.get("run_kind") in (None, "backtest")
    assert "research_note" not in verdict


# ----------------------------------------------------------------------
# entry points the override does NOT open
# ----------------------------------------------------------------------


def test_the_sweep_and_the_signal_helpers_still_refuse_same_bar():
    from quantbox.execution import resolve_sweep_lag_bars

    with pytest.raises(ValueError, match=REFUSED):
        resolve_sweep_lag_bars(0, None)


def test_the_sweep_cli_refuses_the_override_by_name(tmp_path):
    cfg = tmp_path / "sweep.yaml"
    cfg.write_text(yaml.safe_dump({"strategy": "strategy.nope.v1", "data": {}, "execution": copy.deepcopy(OVERRIDE)}))
    result = _cli("sweep", "-c", str(cfg))
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError) and "does not take execution.same_bar" in str(result.exception)


def test_arms_carry_the_override_to_every_arm(tmp_path):
    from quantbox.arms import load_arms

    _, base = _config(tmp_path, None)
    arms = tmp_path / "arms.yaml"
    arms.write_text(
        yaml.safe_dump(
            {
                "base": str(base),
                "execution": copy.deepcopy(OVERRIDE),
                "overrides": {"a": {"plugins.pipeline.params.fees": 0.0}},
            }
        )
    )
    spec = load_arms(arms)
    assert spec.base["plugins"]["pipeline"]["params"]["execution"] == OVERRIDE
