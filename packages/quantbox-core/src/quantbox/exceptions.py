"""Custom exceptions for Quantbox.

Hierarchy::

    QuantboxError
    ├── ConfigValidationError   — YAML config failed validation
    ├── PluginNotFoundError     — plugin name not in registry
    ├── PluginLoadError         — entry point or import failed
    ├── DataLoadError           — data plugin couldn't fetch/load data
    └── BrokerExecutionError    — broker failed to place/fill orders

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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .validate import ValidationFinding


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
    ) -> None:
        msg = (
            f"plugin_not_found: '{plugin_name}' in group '{group}'. "
            f"Available: {', '.join(sorted(available))}"
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
