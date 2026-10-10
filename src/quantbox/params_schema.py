"""Plugin parameter schemas — the params contract, resolved in one place (TOM-1350).

Every plugin declares ``meta.params_schema`` (a JSON Schema object). The schema
is completed from the plugin's constructor: each parameter that is not private
contributes its JSON type and its default, so the two cannot drift apart. The
declaration supplies what a signature cannot carry — descriptions, bounds, enums
— and every key a plugin reads from ``params`` at run time that is not a
constructor parameter.

The resolved schema is CLOSED: a key that is not a property is an unknown
parameter, and ``quantbox validate`` refuses it. ``params`` (run time) is checked
against the whole property set; ``params_init`` goes to the constructor, so it is
checked against the constructor's parameters alone.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import types
import typing
from typing import Any, Literal, Union

from jsonschema import Draft202012Validator

from .exceptions import MissingExtraError

# group label -> PluginRegistry attribute. The configurable plugin groups only:
# datasets and capabilities are not configured through params.
PLUGIN_GROUPS: dict[str, str] = {
    "pipeline": "pipelines",
    "strategy": "strategies",
    "data": "data",
    "broker": "brokers",
    "rebalancing": "rebalancing",
    "risk": "risk",
    "publisher": "publishers",
    "validation": "validations",
    "monitor": "monitors",
    "overlay": "overlays",
    "feature": "features",
}

_SCALARS: dict[Any, str] = {bool: "boolean", int: "integer", float: "number", str: "string", type(None): "null"}
_ARRAYS = (list, tuple, set, frozenset, typing.Sequence, typing.Iterable)
_OBJECTS = (dict, typing.Mapping)


def _json_type(tp: Any) -> str | list[str] | None:
    """JSON Schema ``type`` for a Python annotation; None when it cannot be named (``Any``)."""
    if tp in _SCALARS:
        return _SCALARS[tp]
    origin = typing.get_origin(tp)
    if origin in (Union, types.UnionType):
        out: list[str] = []
        for arg in typing.get_args(tp):
            t = _json_type(arg)
            if t is None:
                return None
            for name in [t] if isinstance(t, str) else t:
                if name not in out:
                    out.append(name)
        return out[0] if len(out) == 1 else out
    if origin is Literal:
        return _json_type(Union[tuple(type(v) for v in typing.get_args(tp))])  # noqa: UP007
    base = origin or tp
    if base in _ARRAYS or (isinstance(base, type) and issubclass(base, (list, tuple, set, frozenset))):
        return "array"
    if base in _OBJECTS or (isinstance(base, type) and issubclass(base, dict)):
        return "object"
    return None


def _json_default(value: Any) -> tuple[bool, Any]:
    """(ok, value) — the default as JSON would carry it (tuples become lists)."""
    try:
        return True, json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return False, None


class InitParam(typing.NamedTuple):
    name: str
    hint: Any
    default: Any  # dataclasses.MISSING when the parameter is required


def config_fields(cls: type) -> list[InitParam]:
    """Constructor parameters a config may set (``params_init``): not private, not ``meta``.

    Read from the constructor SIGNATURE, not the dataclass fields: a plugin that
    writes its own ``__init__`` (the paper-broker stubs) accepts what it declares
    there, whatever its fields say. A plugin without a constructor of its own
    accepts nothing here.
    """
    if cls.__init__ is object.__init__:
        return []
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return []
    try:
        hints = typing.get_type_hints(cls.__init__)
    except Exception:  # an unresolvable annotation leaves the type to the declaration
        hints = {}
    fields = {f.name: f for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else {}
    out = []
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or p.name.startswith("_") or p.name == "meta":
            continue
        default: Any = dataclasses.MISSING if p.default is p.empty else p.default
        f = fields.get(p.name)
        if f is not None and f.default_factory is not dataclasses.MISSING:
            default = f.default_factory()
        out.append(InitParam(p.name, hints.get(p.name, Any), default))
    return out


def _derived_property(param: InitParam) -> dict[str, Any]:
    prop: dict[str, Any] = {}
    jt = _json_type(param.hint)
    if jt is not None:
        prop["type"] = jt
    if typing.get_origin(param.hint) is Literal:
        prop["enum"] = list(typing.get_args(param.hint))
    if param.default is not dataclasses.MISSING:
        ok, val = _json_default(param.default)
        if ok:
            prop["default"] = val
    return prop


def resolve_params_schema(cls: type) -> dict[str, Any] | None:
    """The plugin's params schema, completed from its constructor; None when it declares none."""
    meta = getattr(cls, "meta", None)
    declared = getattr(meta, "params_schema", None)
    if declared is None:
        return None
    schema = copy.deepcopy(declared)
    schema.setdefault("type", "object")
    props: dict[str, Any] = schema.setdefault("properties", {})
    for param in config_fields(cls):
        props[param.name] = {**_derived_property(param), **props.get(param.name, {})}
    schema.setdefault("additionalProperties", False)
    return schema


def describe_params(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per parameter: name, type, default, description, required."""
    required = set(schema.get("required", []))
    rows = []
    for name, prop in schema.get("properties", {}).items():
        rows.append(
            {
                "name": name,
                "type": prop.get("type"),
                "default": prop.get("default"),
                "description": prop.get("description", ""),
                "required": name in required,
            }
        )
    return rows


def check_params(
    schema: dict[str, Any], params: dict[str, Any], allowed: set[str] | None = None
) -> tuple[list[str], list[str]]:
    """(unknown keys, value violations) of ``params`` against a resolved schema.

    ``allowed`` narrows the accepted keys below the property set (the constructor's
    parameters, for ``params_init``).
    """
    props = schema.get("properties", {})
    accepted = set(props) if allowed is None else set(props) & allowed
    unknown = [k for k in params if k not in accepted]
    known = {k: v for k, v in params.items() if k in accepted}
    validator = Draft202012Validator({**schema, "required": []})
    violations = []
    for err in sorted(validator.iter_errors(known), key=lambda e: list(e.path)):
        where = ".".join(str(p) for p in err.path) or "<params>"
        violations.append(f"{where}: {err.message}")
    return unknown, violations


def catalog(registry: Any) -> dict[str, Any]:
    """Every registered plugin with its id, status and resolved params schema.

    A builtin whose module needs an extra that is not installed cannot be
    imported, so its meta and schema are unknown: its row carries
    ``missing_extra`` (the extra to install) and nulls, and it is still listed.
    """
    plugins = []
    for group, attr in PLUGIN_GROUPS.items():
        mapping = getattr(registry, attr, None) or {}
        for name in sorted(mapping.keys()):
            try:
                cls = mapping[name]
            except MissingExtraError as exc:
                plugins.append(
                    {
                        "id": name,
                        "group": group,
                        "kind": None,
                        "status": None,
                        "version": None,
                        "description": "",
                        "params_schema": None,
                        "params": None,
                        "missing_extra": exc.extra,
                    }
                )
                continue
            meta = getattr(cls, "meta", None)
            schema = resolve_params_schema(cls)
            plugins.append(
                {
                    "id": name,
                    "group": group,
                    "kind": getattr(meta, "kind", None),
                    "status": getattr(meta, "status", None),
                    "version": getattr(meta, "version", None),
                    "description": getattr(meta, "description", ""),
                    "params_schema": schema,
                    "params": describe_params(schema) if schema is not None else None,
                }
            )
    return {"plugins": plugins}


# The shape `quantbox plugins schema --json` prints. Kept beside the producer
# so a consumer (an agent, a scout) can validate what it parsed.
CATALOG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["plugins"],
    "properties": {
        "plugins": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "group", "status", "params_schema", "params"],
                "properties": {
                    "id": {"type": "string"},
                    "group": {"enum": list(PLUGIN_GROUPS)},
                    "kind": {"type": ["string", "null"]},
                    "status": {"type": ["string", "null"]},
                    "version": {"type": ["string", "null"]},
                    "description": {"type": "string"},
                    # Present only when the plugin's extra is not installed.
                    "missing_extra": {"type": "string"},
                    "params_schema": {"type": ["object", "null"]},
                    "params": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "object",
                            "required": ["name", "type", "default", "description", "required"],
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": ["string", "array", "null"]},
                                "description": {"type": "string"},
                                "required": {"type": "boolean"},
                            },
                        },
                    },
                },
            },
        }
    },
}
