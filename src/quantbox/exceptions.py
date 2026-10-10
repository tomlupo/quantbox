"""Custom exceptions for Quantbox.

Hierarchy::

    QuantboxError
    ├── ConfigValidationError   — YAML config failed validation
    ├── PluginNotFoundError     — plugin name not in registry
    ├── PluginLoadError         — entry point or import failed
    ├── DataLoadError           — data plugin couldn't fetch/load data
    ├── BrokerExecutionError    — broker failed to place/fill orders
    └── MissingExtraError       — an optional extra (e.g. ``[vectorbt]``) is not installed

All exceptions carry structured context in ``details`` for LLM agents
to parse and recover from programmatically.

Example::

    try:
        result = run_from_config(cfg, registry)
    except ConfigValidationError as e:
        print(e.findings)          # List[ValidationFinding]
    except PluginNotFoundError as e:
        print(e.plugin_name)       # str
        print(e.available)         # List[str]
    except BrokerExecutionError as e:
        print(e.broker_name)       # str
        print(e.details)           # dict with context
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "UNKNOWN_PLUGIN",
    "BrokerExecutionError",
    "ConfigValidationError",
    "DataLoadError",
    "MissingExtraError",
    "PluginLoadError",
    "PluginNotFoundError",
    "QuantboxError",
    "ValidationFinding",
]

#: Finding code of a block naming a plugin this environment does not register. ``validate``
#: and the runner refuse it alike (TOM-1528, TOM-1529).
UNKNOWN_PLUGIN = "unknown_plugin"


@dataclass
class ValidationFinding:
    """One finding on a run config; :class:`ConfigValidationError` carries a list of them.

    Defined here, the leaf every checker imports, so that :mod:`quantbox.validate`,
    :mod:`quantbox.funding_guard` and :mod:`quantbox.explain` share it without an
    import cycle (TOM-1451). ``quantbox.validate.ValidationFinding`` is the same class.
    """

    level: str  # "error" or "warning"
    message: str
    #: Machine-readable kind, set where a consumer acts on it (``UNKNOWN_PLUGIN``); None otherwise.
    code: str | None = None
    #: What the finding is about, for a consumer that acts on it: for ``UNKNOWN_PLUGIN``,
    #: ``{"plugin_name", "group", "where"}``.
    subject: dict[str, str] | None = None


class QuantboxError(Exception):
    """Base exception for all quantbox errors."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details: dict[str, Any] = details or {}


class ConfigValidationError(QuantboxError):
    """Configuration validation failed.

    Attributes:
        findings: List of ValidationFinding objects describing each issue.
    """

    def __init__(self, message: str, findings: list[ValidationFinding]) -> None:
        super().__init__(message, details={"findings_count": len(findings)})
        self.findings = findings


class PluginNotFoundError(QuantboxError):
    """Plugin name not found in the registry.

    Attributes:
        plugin_name: The name that was looked up.
        group: Plugin group searched (e.g. "pipeline", "broker").
        available: Names that do exist in that group.
    """

    def __init__(
        self,
        plugin_name: str,
        group: str,
        available: list[str],
        message: str | None = None,
    ) -> None:
        # ``message``: the run passes validate's ``unknown_plugin`` finding verbatim (TOM-1529).
        msg = message or (
            f"plugin_not_found: '{plugin_name}' in group '{group}'. Available: {', '.join(sorted(available))}"
        )
        super().__init__(msg, details={"plugin_name": plugin_name, "group": group})
        self.plugin_name = plugin_name
        self.group = group
        self.available = available


class PluginLoadError(QuantboxError):
    """Entry-point or import for a plugin failed.

    Attributes:
        plugin_name: The entry-point name that failed to load.
        cause: The underlying exception.
    """

    def __init__(self, plugin_name: str, cause: Exception) -> None:
        msg = f"plugin_load_failed: '{plugin_name}': {cause}"
        super().__init__(msg, details={"plugin_name": plugin_name, "cause": str(cause)})
        self.plugin_name = plugin_name
        self.cause = cause


class DataLoadError(QuantboxError):
    """Data plugin failed to fetch or load market data.

    Attributes:
        data_plugin: Name of the data plugin.
    """

    def __init__(self, data_plugin: str, message: str, **kwargs: Any) -> None:
        msg = f"data_load_failed ({data_plugin}): {message}"
        super().__init__(msg, details={"data_plugin": data_plugin, **kwargs})
        self.data_plugin = data_plugin


class BrokerExecutionError(QuantboxError):
    """Broker failed to place or fill orders.

    Attributes:
        broker_name: Name of the broker plugin.
    """

    def __init__(self, broker_name: str, message: str, **kwargs: Any) -> None:
        msg = f"broker_execution_failed ({broker_name}): {message}"
        super().__init__(msg, details={"broker_name": broker_name, **kwargs})
        self.broker_name = broker_name


class MissingExtraError(QuantboxError, ImportError):
    """A feature needs an optional extra that is not installed.

    Also an ``ImportError``, so ``except ImportError`` callers keep working;
    the message names the extra to install, never just the missing module.

    Attributes:
        extra: The extra to install (e.g. ``"vectorbt"``).
        feature: What was asked for (e.g. ``"the vectorbt backtest engine"``).
    """

    def __init__(self, extra: str, feature: str, missing: str | None = None) -> None:
        msg = (
            f"missing_extra: {feature} needs the [{extra}] extra, which is not installed. "
            f"Install it with `uv add 'quantbox[{extra}]'` (or `pip install 'quantbox[{extra}]'`); "
            f"[full] includes it."
        )
        if missing:
            msg += f" (missing module: {missing})"
        super().__init__(msg, details={"extra": extra, "feature": feature, "missing": missing})
        self.extra = extra
        self.feature = feature
        self.name = missing
