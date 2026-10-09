"""``quantbox validate``: every check of a run config, the funding guard on its plan included.

The checks that need no plan of the run live in :mod:`quantbox.config_checks`,
which the runner calls. This module adds the one check that needs the plan,
through ``quantbox config explain`` (:mod:`quantbox.explain`). Keeping that
reach here, above the runner, is what keeps ``validate``, ``explain`` and the
runner out of an import cycle (TOM-1451).
"""

from __future__ import annotations

from typing import Any

from .config_checks import check_config, check_plugin_params, is_backtest_config
from .exceptions import UNKNOWN_PLUGIN, ValidationFinding

__all__ = ["UNKNOWN_PLUGIN", "ValidationFinding", "check_plugin_params", "validate_config"]


def validate_config(
    cfg: dict[str, Any], registry: Any = None, *, check_params: bool = True, config_path: Any = None
) -> list[ValidationFinding]:
    """Findings for a run config.

    ``check_params`` checks every plugin block's ``params_init`` / ``params``
    against that plugin's params schema (see ``check_plugin_params``) and, for a
    backtest config that passes every other check, the funding guard on its plan
    (``_check_planned_funding``); ``registry`` defaults to ``PluginRegistry.discover()``.
    """
    findings = check_config(cfg, registry, check_params=check_params)
    if check_params and "plugins" in cfg and is_backtest_config(cfg) and not any(f.level == "error" for f in findings):
        findings.extend(_check_planned_funding(cfg, registry, config_path))
    return findings


def _check_planned_funding(cfg: dict[str, Any], registry: Any, config_path: Any) -> list[ValidationFinding]:
    """The funding guard on the run's plan (TOM-1609, TOM-1619): explain's refusal, with its finding.

    Whether the data carries a funding series, and which market it is, is known only
    once the data plugin is built and its files and market planned, so validate asks
    ``quantbox config explain``, which
    calls the SAME check the run calls (:func:`quantbox.funding_guard.check_funding`).
    A config explain cannot plan is reported as not checked, never as clean.
    """
    from .explain import explain_config

    refused: list[ValidationFinding] = []
    try:
        if registry is None:
            from .registry import PluginRegistry

            registry = PluginRegistry.discover()
        doc = explain_config(cfg, registry, config_path=config_path, findings=refused)
    except Exception as exc:  # noqa: BLE001 — a plan that cannot be built is "not checked"
        return [ValidationFinding("warning", f"funding_not_checked: the run could not be planned ({exc})")]
    if refused:
        return refused
    if not doc.get("ok"):
        why = "; ".join(doc.get("errors") or []) or "unknown"
        return [
            ValidationFinding(
                "warning",
                f"funding_not_checked: the run could not be planned ({why}); `quantbox config explain` shows it",
            )
        ]
    return []
