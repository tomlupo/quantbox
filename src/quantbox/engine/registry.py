"""``engine:`` value -> its adapter. The one table of engine names in quantbox."""

from __future__ import annotations

from quantbox._lazy import load
from quantbox.exceptions import MissingExtraError

from .base import EngineAdapter
from .rsims import RsimsAdapter

#: ``engine:`` when a config or call does not name one.
DEFAULT_ENGINE = "vectorbt"

#: ``engine:`` name -> its adapter. The vectorbt adapter is research-layer code (ADR-0010):
#: it is named here and imported on first use (:func:`quantbox._lazy.load`), never at import.
_ADAPTERS: dict[str, type[EngineAdapter] | str] = {
    "vectorbt": "quantbox.engine.vectorbt:VectorbtAdapter",
    RsimsAdapter.name: RsimsAdapter,
}


def _adapter(name: str) -> type[EngineAdapter] | None:
    cls = _ADAPTERS.get(name)
    if isinstance(cls, str):
        cls = _ADAPTERS[name] = load(cls, extra=name)
    return cls


def engine_names() -> list[str]:
    """Every ``engine:`` value the seam accepts."""
    return list(_ADAPTERS)


def get_engine(engine: str | EngineAdapter | None = None, *, require_installed: bool = True) -> EngineAdapter:
    """The adapter for *engine* (a name, or an adapter passed through); None = :data:`DEFAULT_ENGINE`.

    Raises ``ValueError`` on an unknown name and :class:`~quantbox.exceptions.MissingExtraError`
    when the adapter's extra is not installed (unless ``require_installed=False``, for a
    caller that only reads a declared default).
    """
    if isinstance(engine, EngineAdapter):
        return engine
    name = str(engine if engine is not None else DEFAULT_ENGINE).lower()
    cls = _adapter(name)
    if cls is None:
        names = " or ".join(repr(n) for n in _ADAPTERS)
        raise ValueError(f"Unknown engine: {engine!r}. Use {names}.")
    if require_installed and not cls.installed():
        raise MissingExtraError(cls.extra or cls.name, f"the {cls.name} backtest engine", cls.name)
    return cls()


def engine_distribution(name: str) -> str:
    """The distribution whose version is *name*'s version; an unknown name is its own distribution."""
    cls = _adapter(str(name).lower())
    return cls.distribution if cls is not None else str(name)
