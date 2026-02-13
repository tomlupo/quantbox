"""Tests for quantbox.runner.run_from_config().

Self-contained: no conftest.py needed. All plugins and registry
interactions are mocked so tests run without real plugin imports.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from quantbox.contracts import PluginMeta, RunResult
from quantbox.exceptions import ConfigValidationError, PluginNotFoundError
from quantbox.runner import run_from_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_result(
    run_id: str = "test_run",
    pipeline_name: str = "test.pipeline.v1",
    mode: str = "backtest",
    asof: str = "2026-01-15",
) -> RunResult:
    return RunResult(
        run_id=run_id,
        pipeline_name=pipeline_name,
        mode=mode,
        asof=asof,
        artifacts={},
        metrics={"n_assets": 10},
        notes={},
    )


def _make_registry(
    *,
    pipelines: dict | None = None,
    data: dict | None = None,
    brokers: dict | None = None,
    risk: dict | None = None,
    strategies: dict | None = None,
    publishers: dict | None = None,
    rebalancing: dict | None = None,
) -> MagicMock:
    """Build a mock PluginRegistry with the supplied plugin dicts."""
    reg = MagicMock()
    reg.pipelines = pipelines or {}
    reg.data = data or {}
    reg.brokers = brokers or {}
    reg.risk = risk or {}
    reg.strategies = strategies or {}
    reg.publishers = publishers or {}
    reg.rebalancing = rebalancing or {}
    return reg


def _minimal_config(
    *,
    mode: str = "backtest",
    asof: str = "2026-01-15",
    pipeline_name: str = "test.pipeline.v1",
    data_name: str = "test.data.v1",
    broker_block: dict | None = None,
    strategies: list[dict] | None = None,
    risk: list[dict] | None = None,
    pipeline_params: dict | None = None,
) -> dict[str, Any]:
    """Return the smallest valid config dict for run_from_config()."""
    plugins: dict[str, Any] = {
        "pipeline": {
            "name": pipeline_name,
            "params": pipeline_params or {},
        },
        "data": {
            "name": data_name,
        },
    }
    if broker_block is not None:
        plugins["broker"] = broker_block
    if strategies is not None:
        plugins["strategies"] = strategies
    if risk is not None:
        plugins["risk"] = risk

    return {
        "run": {
            "mode": mode,
            "asof": asof,
            "pipeline": pipeline_name,
        },
        "artifacts": {"root": "/tmp/quantbox_test_artifacts"},
        "plugins": plugins,
    }


def _make_pipeline_cls(run_result: RunResult | None = None, kind: str = "research"):
    """Return a callable that produces a mock pipeline instance."""
    instance = MagicMock()
    instance.kind = kind
    instance.meta = PluginMeta(
        name="test.pipeline.v1",
        kind="pipeline",
        version="1.0.0",
        core_compat=">=0.1.0",
    )
    instance.run.return_value = run_result or _make_run_result()
    cls = MagicMock(return_value=instance)
    return cls, instance


def _make_data_cls():
    """Return a callable that produces a mock data plugin instance."""
    instance = MagicMock()
    instance.meta = PluginMeta(
        name="test.data.v1",
        kind="data",
        version="1.0.0",
        core_compat=">=0.1.0",
    )
    cls = MagicMock(return_value=instance)
    return cls, instance


def _make_strategy_cls(name: str = "strategy.a.v1"):
    """Return a callable that produces a mock strategy plugin instance."""
    instance = MagicMock()
    instance.meta = PluginMeta(name=name, kind="strategy", version="1.0.0", core_compat=">=0.1.0")
    cls = MagicMock(return_value=instance)
    return cls, instance


def _make_risk_cls(name: str = "risk.a.v1"):
    """Return a callable that produces a mock risk plugin instance."""
    instance = MagicMock()
    instance.meta = PluginMeta(name=name, kind="risk", version="1.0.0", core_compat=">=0.1.0")
    cls = MagicMock(return_value=instance)
    return cls, instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("quantbox.runner.repo_root", return_value=MagicMock(spec=["__truediv__"]))
@patch("quantbox.runner.FileArtifactStore")
class TestRunFromConfig:
    """Tests for run_from_config()."""

    # 1. Happy path -------------------------------------------------------

    def test_happy_path_pipeline_run_called(self, MockStore, mock_repo_root):
        """Pipeline.run() is called with correct args on a minimal valid config."""
        mock_repo_root.return_value.__truediv__ = lambda self, x: MagicMock(exists=MagicMock(return_value=False))
        pipe_cls, pipe_inst = _make_pipeline_cls()
        data_cls, data_inst = _make_data_cls()

        registry = _make_registry(
            pipelines={"test.pipeline.v1": pipe_cls},
            data={"test.data.v1": data_cls},
        )
        cfg = _minimal_config()

        result = run_from_config(cfg, registry)

        # Pipeline class was instantiated (with params_init if any)
        pipe_cls.assert_called_once_with()
        # Data class was instantiated
        data_cls.assert_called_once_with()
        # Pipeline.run() was called exactly once
        pipe_inst.run.assert_called_once()
        call_kwargs = pipe_inst.run.call_args.kwargs
        assert call_kwargs["mode"] == "backtest"
        assert call_kwargs["asof"] == "2026-01-15"
        assert call_kwargs["data"] is data_inst
        assert call_kwargs["broker"] is None
        assert call_kwargs["risk"] == []
        assert call_kwargs["strategies"] is None
        assert call_kwargs["rebalancer"] is None
        assert call_kwargs["aggregator"] is None
        assert isinstance(result, RunResult)

    # 2. Unknown pipeline name → PluginNotFoundError ----------------------

    def test_unknown_pipeline_raises(self, MockStore, mock_repo_root):
        """An unregistered pipeline name must raise PluginNotFoundError."""
        data_cls, _ = _make_data_cls()
        registry = _make_registry(
            pipelines={},  # empty — nothing registered
            data={"test.data.v1": data_cls},
        )
        cfg = _minimal_config(pipeline_name="nonexistent.pipeline.v1")

        with pytest.raises(PluginNotFoundError) as exc_info:
            run_from_config(cfg, registry)

        assert exc_info.value.plugin_name == "nonexistent.pipeline.v1"
        assert exc_info.value.group == "pipeline"

    # 3. Unknown data plugin → PluginNotFoundError ------------------------

    def test_unknown_data_raises(self, MockStore, mock_repo_root):
        """An unregistered data plugin name must raise PluginNotFoundError."""
        pipe_cls, _ = _make_pipeline_cls()
        registry = _make_registry(
            pipelines={"test.pipeline.v1": pipe_cls},
            data={},  # empty
        )
        cfg = _minimal_config(data_name="nonexistent.data.v1")

        with pytest.raises(PluginNotFoundError) as exc_info:
            run_from_config(cfg, registry)

        assert exc_info.value.plugin_name == "nonexistent.data.v1"
        assert exc_info.value.group == "data"

    # 4. Missing required config keys → ConfigValidationError -------------

    def test_missing_run_key_raises(self, MockStore, mock_repo_root):
        """Config without 'run' top-level key raises KeyError (accessed before validation)."""
        registry = _make_registry()
        cfg = {
            # missing "run"
            "artifacts": {"root": "/tmp/test"},
            "plugins": {
                "pipeline": {"name": "x"},
                "data": {"name": "y"},
            },
        }

        with pytest.raises(KeyError):
            run_from_config(cfg, registry)

    def test_missing_plugins_key_raises(self, MockStore, mock_repo_root):
        """Config without 'plugins' top-level key must raise ConfigValidationError."""
        registry = _make_registry()
        cfg = {
            "run": {"mode": "backtest", "asof": "2026-01-01", "pipeline": "x"},
            "artifacts": {"root": "/tmp/test"},
            # missing "plugins"
        }

        with pytest.raises(ConfigValidationError):
            run_from_config(cfg, registry)

    def test_missing_artifacts_key_raises(self, MockStore, mock_repo_root):
        """Config without 'artifacts' top-level key must raise ConfigValidationError."""
        registry = _make_registry()
        cfg = {
            "run": {"mode": "backtest", "asof": "2026-01-01", "pipeline": "x"},
            # missing "artifacts"
            "plugins": {
                "pipeline": {"name": "x"},
                "data": {"name": "y"},
            },
        }

        with pytest.raises(ConfigValidationError):
            run_from_config(cfg, registry)

    def test_invalid_mode_raises(self, MockStore, mock_repo_root):
        """An invalid run.mode must raise ConfigValidationError."""
        registry = _make_registry()
        cfg = _minimal_config(mode="invalid_mode")

        with pytest.raises(ConfigValidationError):
            run_from_config(cfg, registry)

    # 5. Result structure: verify returned dict has expected keys ----------

    def test_result_structure(self, MockStore, mock_repo_root):
        """RunResult returned by run_from_config has all expected attributes."""
        mock_repo_root.return_value.__truediv__ = lambda self, x: MagicMock(exists=MagicMock(return_value=False))
        expected = _make_run_result(
            run_id="abc",
            pipeline_name="test.pipeline.v1",
            mode="backtest",
            asof="2026-01-15",
        )
        pipe_cls, _ = _make_pipeline_cls(run_result=expected)
        data_cls, _ = _make_data_cls()
        registry = _make_registry(
            pipelines={"test.pipeline.v1": pipe_cls},
            data={"test.data.v1": data_cls},
        )
        cfg = _minimal_config()

        result = run_from_config(cfg, registry)

        assert result.run_id == "abc"
        assert result.pipeline_name == "test.pipeline.v1"
        assert result.mode == "backtest"
        assert result.asof == "2026-01-15"
        assert isinstance(result.artifacts, dict)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.notes, dict)
        assert result.metrics["n_assets"] == 10

    # 6. Dry-run mode: pipeline.run() is NOT called -----------------------
    #
    # run_from_config itself does not implement dry-run — it always calls
    # pipeline.run(). The dry-run flag is handled at CLI level by
    # skipping run_from_config entirely. Therefore we verify that if
    # the mode is "backtest" and pipeline.run() is called, the result
    # is still returned. A true dry-run test would need to test the CLI
    # layer, but we can verify that the mode is correctly propagated.
    #
    # However, checking that mode is propagated is already covered above.
    # Instead, let's verify the pipeline_params dict is correctly built
    # for a dry-run scenario where the caller might inspect params.

    def test_pipeline_params_propagated(self, MockStore, mock_repo_root):
        """Pipeline params from config are forwarded to pipeline.run()."""
        mock_repo_root.return_value.__truediv__ = lambda self, x: MagicMock(exists=MagicMock(return_value=False))
        pipe_cls, pipe_inst = _make_pipeline_cls()
        data_cls, _ = _make_data_cls()
        registry = _make_registry(
            pipelines={"test.pipeline.v1": pipe_cls},
            data={"test.data.v1": data_cls},
        )
        cfg = _minimal_config(pipeline_params={"lookback_days": 90, "top_n": 20})

        run_from_config(cfg, registry)

        call_kwargs = pipe_inst.run.call_args.kwargs
        assert call_kwargs["params"]["lookback_days"] == 90
        assert call_kwargs["params"]["top_n"] == 20

    # 7. Strategy list handling: multiple strategies resolve correctly -----

    def test_multiple_strategies_resolved(self, MockStore, mock_repo_root):
        """Multiple strategies in config are all instantiated and passed."""
        mock_repo_root.return_value.__truediv__ = lambda self, x: MagicMock(exists=MagicMock(return_value=False))
        pipe_cls, pipe_inst = _make_pipeline_cls()
        data_cls, _ = _make_data_cls()
        strat_a_cls, strat_a_inst = _make_strategy_cls("strategy.alpha.v1")
        strat_b_cls, strat_b_inst = _make_strategy_cls("strategy.beta.v1")

        registry = _make_registry(
            pipelines={"test.pipeline.v1": pipe_cls},
            data={"test.data.v1": data_cls},
            strategies={
                "strategy.alpha.v1": strat_a_cls,
                "strategy.beta.v1": strat_b_cls,
            },
        )
        cfg = _minimal_config(
            strategies=[
                {"name": "strategy.alpha.v1", "weight": 0.6, "params": {"p": 1}},
                {"name": "strategy.beta.v1", "weight": 0.4, "params": {"p": 2}},
            ]
        )

        run_from_config(cfg, registry)

        # Both strategy classes were instantiated
        strat_a_cls.assert_called_once()
        strat_b_cls.assert_called_once()
        # Both instances passed to pipeline.run()
        call_kwargs = pipe_inst.run.call_args.kwargs
        strategies_passed = call_kwargs["strategies"]
        assert len(strategies_passed) == 2
        assert strat_a_inst in strategies_passed
        assert strat_b_inst in strategies_passed
        # _strategies_cfg injected into pipeline params
        assert "_strategies_cfg" in call_kwargs["params"]

    # 8. Risk plugin list: multiple risk plugins instantiated -------------

    def test_multiple_risk_plugins_instantiated(self, MockStore, mock_repo_root):
        """Multiple risk plugins in config are all instantiated and passed."""
        mock_repo_root.return_value.__truediv__ = lambda self, x: MagicMock(exists=MagicMock(return_value=False))
        pipe_cls, pipe_inst = _make_pipeline_cls()
        data_cls, _ = _make_data_cls()
        risk_a_cls, risk_a_inst = _make_risk_cls("risk.basic.v1")
        risk_b_cls, risk_b_inst = _make_risk_cls("risk.stress.v1")

        registry = _make_registry(
            pipelines={"test.pipeline.v1": pipe_cls},
            data={"test.data.v1": data_cls},
            risk={
                "risk.basic.v1": risk_a_cls,
                "risk.stress.v1": risk_b_cls,
            },
        )
        cfg = _minimal_config(
            risk=[
                {"name": "risk.basic.v1", "params": {"max_weight": 0.3}},
                {"name": "risk.stress.v1", "params": {"var_limit": 0.05}},
            ]
        )

        run_from_config(cfg, registry)

        risk_a_cls.assert_called_once()
        risk_b_cls.assert_called_once()
        call_kwargs = pipe_inst.run.call_args.kwargs
        risk_passed = call_kwargs["risk"]
        assert len(risk_passed) == 2
        assert risk_a_inst in risk_passed
        assert risk_b_inst in risk_passed
        # Merged risk params injected into pipeline params
        assert call_kwargs["params"]["_risk_cfg"]["max_weight"] == 0.3
        assert call_kwargs["params"]["_risk_cfg"]["var_limit"] == 0.05

    # 9. Optional broker: when broker config is missing, broker=None ------

    def test_no_broker_config_passes_none(self, MockStore, mock_repo_root):
        """When config has no broker block, pipeline.run() receives broker=None."""
        mock_repo_root.return_value.__truediv__ = lambda self, x: MagicMock(exists=MagicMock(return_value=False))
        pipe_cls, pipe_inst = _make_pipeline_cls()
        data_cls, _ = _make_data_cls()
        registry = _make_registry(
            pipelines={"test.pipeline.v1": pipe_cls},
            data={"test.data.v1": data_cls},
        )
        cfg = _minimal_config()  # no broker_block

        run_from_config(cfg, registry)

        call_kwargs = pipe_inst.run.call_args.kwargs
        assert call_kwargs["broker"] is None

    def test_broker_only_used_for_trading_pipeline(self, MockStore, mock_repo_root):
        """Even with broker config, broker=None if pipeline.kind != 'trading' or mode != paper/live."""
        mock_repo_root.return_value.__truediv__ = lambda self, x: MagicMock(exists=MagicMock(return_value=False))
        # Pipeline kind is "research" (default), mode is "backtest"
        pipe_cls, pipe_inst = _make_pipeline_cls(kind="research")
        data_cls, _ = _make_data_cls()
        broker_cls = MagicMock()
        registry = _make_registry(
            pipelines={"test.pipeline.v1": pipe_cls},
            data={"test.data.v1": data_cls},
            brokers={"test.broker.v1": broker_cls},
        )
        cfg = _minimal_config(
            mode="backtest",
            broker_block={"name": "test.broker.v1"},
        )

        run_from_config(cfg, registry)

        # Broker class should NOT be instantiated for research+backtest
        broker_cls.assert_not_called()
        call_kwargs = pipe_inst.run.call_args.kwargs
        assert call_kwargs["broker"] is None

    # 10. asof date propagation -------------------------------------------

    def test_asof_date_propagation(self, MockStore, mock_repo_root):
        """The asof date from config is forwarded to pipeline.run()."""
        mock_repo_root.return_value.__truediv__ = lambda self, x: MagicMock(exists=MagicMock(return_value=False))
        pipe_cls, pipe_inst = _make_pipeline_cls()
        data_cls, _ = _make_data_cls()
        registry = _make_registry(
            pipelines={"test.pipeline.v1": pipe_cls},
            data={"test.data.v1": data_cls},
        )
        target_date = "2025-12-31"
        cfg = _minimal_config(asof=target_date)

        run_from_config(cfg, registry)

        call_kwargs = pipe_inst.run.call_args.kwargs
        assert call_kwargs["asof"] == target_date

    def test_asof_different_dates(self, MockStore, mock_repo_root):
        """Different asof dates are correctly forwarded."""
        mock_repo_root.return_value.__truediv__ = lambda self, x: MagicMock(exists=MagicMock(return_value=False))
        for date in ("2024-06-15", "2026-02-13", "2023-01-01"):
            pipe_cls, pipe_inst = _make_pipeline_cls()
            data_cls, _ = _make_data_cls()
            registry = _make_registry(
                pipelines={"test.pipeline.v1": pipe_cls},
                data={"test.data.v1": data_cls},
            )
            cfg = _minimal_config(asof=date)

            run_from_config(cfg, registry)

            call_kwargs = pipe_inst.run.call_args.kwargs
            assert call_kwargs["asof"] == date
