"""Parity tests for :mod:`quantbox.parquet_io`.

The helper exists to dodge a native teardown abort (TOM-435), which no test can
observe deterministically. What a test *can* pin is the thing that would make
the dodge unsafe: the fast path must return exactly what ``pd.read_parquet``
returns, index and dtypes included.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quantbox.parquet_io import _local_file, read_parquet


def _frames() -> dict[str, pd.DataFrame]:
    return {
        "datetime_index": pd.DataFrame(
            {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.5, float("nan"), 3.0]},
            index=pd.date_range("2020-01-01", periods=3, name="date"),
        ),
        "range_index": pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]}),
        "multi_index": pd.DataFrame(
            {"v": [1, 2]},
            index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["s", "n"]),
        ),
        "empty": pd.DataFrame({"x": pd.Series(dtype="float64")}),
    }


@pytest.mark.parametrize("name", sorted(_frames()))
def test_matches_pandas(tmp_path, name):
    df = _frames()[name]
    path = tmp_path / f"{name}.parquet"
    df.to_parquet(path)
    pd.testing.assert_frame_equal(pd.read_parquet(path), read_parquet(path))


def test_column_projection_keeps_the_index(tmp_path):
    """``columns=`` must not silently drop the stored index to a RangeIndex."""
    df = _frames()["datetime_index"]
    path = tmp_path / "projected.parquet"
    df.to_parquet(path)
    got = read_parquet(path, columns=["a"])
    pd.testing.assert_frame_equal(pd.read_parquet(path, columns=["a"]), got)
    assert isinstance(got.index, pd.DatetimeIndex)


def test_index_false_roundtrip(tmp_path):
    df = _frames()["range_index"]
    path = tmp_path / "noindex.parquet"
    df.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(path), read_parquet(path))
    pd.testing.assert_frame_equal(pd.read_parquet(path, columns=["y"]), read_parquet(path, columns=["y"]))


def test_extra_kwargs_fall_back_to_pandas(tmp_path):
    """Unsupported keywords must still work, not raise."""
    df = _frames()["range_index"]
    path = tmp_path / "kwargs.parquet"
    df.to_parquet(path, index=False)
    got = read_parquet(path, dtype_backend="numpy_nullable")
    pd.testing.assert_frame_equal(pd.read_parquet(path, dtype_backend="numpy_nullable"), got)


def test_file_object_delegates(tmp_path):
    df = _frames()["range_index"]
    path = tmp_path / "handle.parquet"
    df.to_parquet(path, index=False)
    with path.open("rb") as fh:
        expected = pd.read_parquet(fh)
    with path.open("rb") as fh:
        pd.testing.assert_frame_equal(expected, read_parquet(fh))


@pytest.mark.parametrize(
    "value",
    ["s3://bucket/key.parquet", "https://example.com/a.parquet", "/no/such/file.parquet"],
)
def test_non_local_inputs_are_not_fast_pathed(value):
    assert _local_file(value) is None


def test_directory_is_not_fast_pathed(tmp_path):
    """A partitioned dataset is a directory — pandas/pyarrow-dataset owns it."""
    assert _local_file(tmp_path) is None


def test_file_without_pandas_metadata(tmp_path):
    """A file written by pyarrow/duckdb carries no pandas metadata block.

    ``use_pandas_metadata=True`` must be a no-op there rather than an error,
    including on a projected read — the warehouse and DuckDB writers both
    produce such files.
    """
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "nometa.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}), path)
    pd.testing.assert_frame_equal(pd.read_parquet(path), read_parquet(path))
    pd.testing.assert_frame_equal(pd.read_parquet(path, columns=["a"]), read_parquet(path, columns=["a"]))


def test_extension_dtypes_round_trip(tmp_path):
    """Categorical and nullable-integer columns must survive the fast path."""
    df = pd.DataFrame({"c": pd.Categorical(["x", "y", "x"]), "n": pd.array([1, None, 3], dtype="Int64")})
    path = tmp_path / "ext.parquet"
    df.to_parquet(path)
    pd.testing.assert_frame_equal(pd.read_parquet(path), read_parquet(path))


def test_no_library_code_calls_pandas_read_parquet_directly():
    """The chokepoint must be enforced, not merely documented.

    ``parquet_io`` only helps if every read goes through it, and the crash it
    dodges is per-call-site — one stray ``pd.read_parquet`` reintroduces it for
    that read alone. Parsing the AST (rather than grepping) means docstring
    examples and the module's own documented fallback don't register as
    call sites.
    """
    import ast

    src = Path(__file__).resolve().parents[1] / "src" / "quantbox"
    allowed = {src / "parquet_io.py"}  # owns the one deliberate fallback
    offenders = []
    for py in sorted(src.rglob("*.py")):
        if py in allowed:
            continue
        for node in ast.walk(ast.parse(py.read_text(), filename=str(py))):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "read_parquet":
                base = fn.value
                if isinstance(base, ast.Name) and base.id in {"pd", "pandas"}:
                    offenders.append(f"{py.relative_to(src.parent.parent)}:{node.lineno}")
    assert not offenders, (
        "these call pandas.read_parquet directly instead of quantbox.parquet_io.read_parquet, "
        "reintroducing the teardown abort (TOM-435): " + ", ".join(offenders)
    )
