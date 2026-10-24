from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .contracts import PluginMeta

__all__ = [
    "CoverageReport",
    "DatasetManifest",
    "DatasetPlugin",
    "finite_numbers",
    "load_pinned_dataset",
    "served_prices",
    "total_return_index",
    "unique_rows",
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


_NOT_A_NUMBER = "type"  # a bool, a date, any object that is not a real number
_NOT_PARSED = "text"  # a string that does not parse as a number ("bad", "nan")


def _cell(value: Any) -> tuple[float, str | None]:
    """One raw source value as a float, or NaN and why it is not a number.

    A true null (None, NaN, pd.NA, NaT) is NaN with no reason. A bool is not a number,
    although Python and pandas count it as one. A string is parsed by ``float``; the
    text "nan" is not a null, so it is refused. ±inf is returned as is; the caller refuses it.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return math.nan, None
    if isinstance(value, (bool, np.bool_)):
        return math.nan, _NOT_A_NUMBER
    if isinstance(value, (int, float, np.integer, np.floating, Decimal)):
        return float(value), None
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return math.nan, _NOT_PARSED
        return (math.nan, _NOT_PARSED) if math.isnan(number) else (number, None)
    return math.nan, _NOT_A_NUMBER


def _labels(where: np.ndarray, labels: pd.Index) -> list[str]:
    """The first five labels where *where* holds: ISO dates for a date label, else the label."""
    return [d.date().isoformat() if isinstance(d, pd.Timestamp) else str(d) for d in labels[where][:5]]


def finite_numbers(
    frame: pd.DataFrame, name: str, what: str, *, columns: list[Any] | None = None, labels: Any = None
) -> pd.DataFrame:
    """*frame* with *columns* (default: all) as numbers, refused unless every RAW value is a true null
    or a finite real number. The ONE value rule for every served table: prices, returns, funding.

    Run on the source values, before any ``to_numeric``, ``astype`` or pivot: those turn a
    malformed edge value into missing history, a bool into a 0/1 return, a date into
    nanoseconds, and average +inf with -inf into NaN (PR #303 and #305 reviews). A null
    stays NaN (a series not yet listed, or no longer). Refused
    (:class:`~quantbox.dataset_lock.DatasetServeError`): a bool, a date or any other
    object; a string that does not parse as a number (a numeric string such as "0.1"
    parses, as a CSV-born table holds them); ±inf. A plain int or float column is kept
    as read; any other validated column becomes float64. *labels* names the rows in an
    error (default: the index).
    """
    from quantbox.dataset_lock import DatasetServeError

    labels = frame.index if labels is None else pd.Index(labels)
    out = frame.copy()
    for col in frame.columns if columns is None else columns:
        raw = frame[col].to_numpy(dtype=object)
        dtype = frame[col].dtype
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            # A float or int column holds nothing but numbers and nulls: no per-value walk.
            values = frame[col].to_numpy(dtype=float, na_value=math.nan)
            not_a_number = not_parsed = np.zeros(len(raw), dtype=bool)
        else:
            cells = [_cell(v) for v in raw]
            values = np.array([c[0] for c in cells], dtype=float)
            reasons = np.array([c[1] for c in cells], dtype=object)
            not_a_number, not_parsed = reasons == _NOT_A_NUMBER, reasons == _NOT_PARSED
        if not_a_number.any():
            kinds = sorted({type(v).__name__ for v in raw[not_a_number]})
            raise DatasetServeError(
                f"dataset {name!r}: {what} series {col!r} holds {', '.join(kinds)} values, not numbers, on "
                f"{_labels(not_a_number, labels)}: {raw[not_a_number][:5].tolist()}"
            )
        if not_parsed.any():
            raise DatasetServeError(
                f"dataset {name!r}: {what} series {col!r} has a value that is not a number on "
                f"{_labels(not_parsed, labels)}: {raw[not_parsed][:5].tolist()}"
            )
        infinite = np.isinf(values)
        if infinite.any():
            raise DatasetServeError(
                f"dataset {name!r}: {what} series {col!r} has a non-finite value on {_labels(infinite, labels)}"
            )
        if not isinstance(dtype, np.dtype) or dtype.kind not in "iuf":
            out[col] = values  # a plain int or float column is served as it was read
    return out


def unique_rows(frame: pd.DataFrame, name: str, what: str) -> pd.DataFrame:
    """*frame*, refused when a key has more than one row: a served table is reshaped, never aggregated.

    The key is (``date``, ``symbol``) for a long table, ``date`` for a wide one with a date
    column, else the index. Two rows for one key would be averaged by a pivot, and an
    average of finite values can overflow (PR #305 review round 2).
    """
    from quantbox.dataset_lock import DatasetServeError

    key = [c for c in ("date", "symbol") if c in frame.columns] if "date" in frame.columns else []
    duplicated = frame.duplicated(subset=key, keep=False) if key else frame.index.duplicated(keep=False)
    if duplicated.any():
        where = frame.loc[duplicated, key] if key else frame.index[duplicated].to_frame(index=False)
        rows = [
            tuple(v.date().isoformat() if isinstance(v, pd.Timestamp) else v for v in r)
            for r in where.drop_duplicates().head(5).itertuples(index=False)
        ]
        raise DatasetServeError(
            f"dataset {name!r}: {what} has more than one row for key {key or ['index']} {rows}; "
            "a served table is reshaped, never aggregated"
        )
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
    values = finite_numbers(returns, name, "returns")
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
    with np.errstate(over="ignore"):  # an overflow is refused where the frame is served (served_prices)
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
    when a date has more than one row or a value is not a finite number, and for
    ``kind: returns`` turned into a total-return index (:func:`total_return_index`). The
    frame returned is checked again by :func:`finite_numbers`, so every served value is
    finite or a true null whatever arithmetic made it. quantbox reads it itself, so a
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
    unique_rows(frame, name, serve["table"])
    if serve["kind"] == "returns":
        served = total_return_index(frame, serve.get("frequency"), name=name)
    else:
        served = finite_numbers(frame, name, "prices").sort_index()
    # The guarantee is checked where the frame leaves, after every transform: compounding
    # finite returns can still overflow to inf.
    return finite_numbers(served, name, "served prices")


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
