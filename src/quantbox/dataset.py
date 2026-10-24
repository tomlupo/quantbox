from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from .contracts import PluginMeta

__all__ = [
    "CoverageReport",
    "DatasetManifest",
    "DatasetPlugin",
    "load_pinned_dataset",
    "refuse_non_finite",
    "served_prices",
    "total_return_index",
]


def _resolve_pinned(name: str, lock: str | None = None) -> dict[str, Any]:
    """Where a by-name dataset will be read from, refused when its bytes are not the pin.

    :func:`quantbox.dataset_lock.resolve_dataset` — the root from ``$QUANTBOX_DATASETS_ROOT``,
    the pin from *lock* (default: the nearest ``datasets.lock``).
    """
    from quantbox.dataset_lock import require_match, resolve_dataset

    return require_match(resolve_dataset(name, lock=lock))


def load_pinned_dataset(
    name: str, lock: str | None = None, resolved: dict[str, Any] | None = None
) -> tuple[Any, dict[str, Any]]:
    """A quantbox-datasets Dataset and the resolution it was served from.

    Resolved by :func:`quantbox.dataset_lock.resolve_dataset` (unless *resolved*
    already is) and refused before any read when the bytes are not the pinned
    build. This was ``quantbox.plugins.datasources.local_file_data._load_pinned_dataset``
    (TOM-1449); since 0.13.0 that name raises ``ImportError`` naming this one (TOM-1457).
    """
    if resolved is None:
        resolved = _resolve_pinned(name, lock)
    try:
        from quantbox_datasets.lock import load
    except ImportError as exc:
        raise ImportError(
            f"dataset={name!r} needs quantbox-datasets installed (it carries quantbox_datasets.lock); "
            "quantbox does not depend on it — install it from its clone and point "
            "QUANTBOX_DATASETS_ROOT at <clone>/datasets"
        ) from exc
    return load(name, root=resolved["root"], sha256=resolved["sha256"], pinned=False), resolved


def _bar_before(first: pd.Timestamp, frequency: str | None, index: pd.DatetimeIndex) -> pd.Timestamp:
    """The date one bar before *first*: a month-end before a month-end, else the index's own step back."""
    if frequency == "1M":
        return first - (pd.offsets.MonthEnd(1) if first.is_month_end else pd.DateOffset(months=1))
    if frequency == "1w":
        return first - pd.Timedelta(weeks=1)
    if frequency == "1d" or len(index) < 2:
        return first - pd.Timedelta(days=1)
    return first - (index[1] - index[0])


def _dates(mask: pd.Series) -> list[str]:
    """The first five dates where *mask* holds, as ISO strings, for an error message."""
    return [d.date().isoformat() for d in mask[mask].index[:5]]


def _numeric(frame: pd.DataFrame, name: str, what: str) -> pd.DataFrame:
    """*frame* as numbers, refused when a value that is there is not a number, or is not finite.

    A null stays a missing value (a series not yet listed, or no longer). A value that
    fails numeric conversion, or an infinity, is a data fault: coerced to NaN at a
    series' edge it would pass as missing history (PR #303 review, finding 2/101). A
    column that pandas converts without being numbers (bool, dates, durations) is
    refused by its dtype.
    """
    from quantbox.dataset_lock import DatasetServeError

    values = frame.apply(pd.to_numeric, errors="coerce")
    for col in values.columns:
        source, converted = frame[col].dtype, values[col].dtype
        if (
            pd.api.types.is_bool_dtype(converted)
            or not pd.api.types.is_numeric_dtype(converted)
            or pd.api.types.is_datetime64_any_dtype(source)
            or pd.api.types.is_timedelta64_dtype(source)
        ):
            raise DatasetServeError(f"dataset {name!r}: {what} series {col!r} holds {source} values, not numbers")
        malformed = values[col].isna() & frame[col].notna()
        if malformed.any():
            raise DatasetServeError(
                f"dataset {name!r}: {what} series {col!r} has a value that is not a number on "
                f"{_dates(malformed)}: {frame[col][malformed].head(5).tolist()}"
            )
        infinite = values[col].isin([float("inf"), float("-inf")])
        if infinite.any():
            raise DatasetServeError(
                f"dataset {name!r}: {what} series {col!r} has a non-finite value on {_dates(infinite)}"
            )
    return values


def refuse_non_finite(frame: pd.DataFrame, name: str, what: str) -> pd.DataFrame:
    """*frame* unchanged, refused when a numeric column holds an infinity.

    For a by-name dataset's served funding file, read wide or long by the inline reader:
    its non-numeric columns (``symbol``, a date) are left to that reader.
    """
    from quantbox.dataset_lock import DatasetServeError

    numeric = frame.select_dtypes("number")
    infinite = sorted(str(c) for c in numeric.columns if numeric[c].isin([float("inf"), float("-inf")]).any())
    if infinite:
        raise DatasetServeError(f"dataset {name!r}: {what} column(s) {infinite} hold a non-finite value")
    return frame


def total_return_index(returns: pd.DataFrame, frequency: str | None = None, *, name: str = "dataset") -> pd.DataFrame:
    """Prices for a table of simple per-bar returns: :data:`quantbox.dataset_lock.TOTAL_RETURN_CONSTRUCTION`.

    Every series starts at 1.0 one bar before its first return, so ``pct_change`` of the
    index gives back every return, the first included; when a series' first return is
    the table's first row, that bar is added before it. A series may start late and end
    early (NaN outside its span). Refused (:class:`~quantbox.dataset_lock.DatasetServeError`):
    a value that is not a number or not finite (anywhere, the edges included), a NaN
    INSIDE a series' span, a missing month in a monthly table, a duplicated or unsorted
    date, a return at or below -100 %.
    """
    from quantbox.dataset_lock import DatasetServeError

    if returns.empty:
        raise DatasetServeError(f"dataset {name!r}: the returns table is empty")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise DatasetServeError(f"dataset {name!r}: a returns table needs a date index")
    index = returns.index
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise DatasetServeError(f"dataset {name!r}: the returns table's dates are unsorted or duplicated")
    if frequency == "1M":
        months = index.tz_localize(None).to_period("M") if index.tz is not None else index.to_period("M")
        steps = (months[1:] - months[:-1]).map(lambda d: d.n)
        missing = [str(months[i]) for i, step in enumerate(steps) if step != 1]
        if missing:
            raise DatasetServeError(
                f"dataset {name!r}: the monthly returns table has a gap after {missing[:5]}; "
                "a missing month would be compounded as no return"
            )
    # Before any span is found: a malformed edge value must not read as "not yet listed".
    values = _numeric(returns, name, "returns")
    for col in values.columns:
        series = values[col]
        first, last = series.first_valid_index(), series.last_valid_index()
        if first is None:
            raise DatasetServeError(f"dataset {name!r}: returns series {col!r} has no value")
        holes = series.loc[first:last].isna()
        if holes.any():
            raise DatasetServeError(
                f"dataset {name!r}: returns series {col!r} has a gap (no return) on "
                f"{[d.date().isoformat() for d in holes[holes].index[:5]]}"
            )
        if (series <= -1.0).any():
            raise DatasetServeError(f"dataset {name!r}: returns series {col!r} has a return at or below -100 %")
    if values.iloc[0].notna().any():
        before = _bar_before(index[0], frequency, index)
        values = pd.concat(
            [
                pd.DataFrame(
                    float("nan"), index=pd.DatetimeIndex([before], name=index.name, tz=index.tz), columns=values.columns
                ),
                values,
            ]
        )
    growth = (1.0 + values).cumprod()
    # The bar before each series' first return is its base, 1.0.
    for col in values.columns:
        start = values.index.get_loc(values[col].first_valid_index())
        growth.iloc[start - 1, growth.columns.get_loc(col)] = 1.0
    return growth


def served_prices(resolution: dict[str, Any]) -> pd.DataFrame:
    """The prices a by-name dataset with a manifest ``serve`` block is served as, from its resolution.

    The table is parsed from the bytes that hash to the resolution's sha256, read once
    (:func:`quantbox.dataset_lock.served_bytes`), narrowed to ``serve.columns``, refused
    when a value is not a finite number, and for ``kind: returns`` turned into a
    total-return index (:func:`total_return_index`). quantbox reads it itself, so a
    dataset a research line froze needs no quantbox-datasets install.
    """
    from quantbox.dataset_lock import DatasetServeError, served_bytes
    from quantbox.parquet_io import read_parquet_bytes

    serve, name = resolution["serve"], resolution["name"]
    frame = read_parquet_bytes(served_bytes(serve, name))
    if "date" in frame.columns:
        frame = frame.set_index("date")
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index), name=frame.index.name)
    if serve.get("columns"):
        absent = [c for c in serve["columns"] if c not in frame.columns]
        if absent:
            raise DatasetServeError(f"dataset {name!r}: serve.columns {absent} are not in {serve['file']}")
        frame = frame[serve["columns"]]
    if serve["kind"] == "returns":
        return total_return_index(frame, serve.get("frequency"), name=name)
    return _numeric(frame, name, "prices").sort_index()


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    version: str
    date_range: Mapping[str, str]  # {"start": "...", "end": "..."}
    symbols_count: int
    data_fields: tuple[str, ...]
    source: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoverageReport:
    per_symbol: Mapping[str, Any]
    per_field: Mapping[str, Any]
    overall: Mapping[str, Any]
    extras: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class DatasetPlugin(Protocol):
    meta: PluginMeta
    dataset_id: str
    dataset_version: str
    capabilities: tuple[str, ...]

    def load_prices(self) -> pd.DataFrame: ...
    def load_universe(self) -> pd.DataFrame: ...
    def load_fx(self) -> pd.DataFrame | None: ...

    def manifest(self) -> DatasetManifest: ...
    def manifest_hash(self) -> str: ...
    def coverage_report(self) -> CoverageReport | None: ...
