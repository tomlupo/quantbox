"""TOM-2547, #322 round 2: scripts/before_after.py compares parquet values by STRUCTURE.

The release review reproduced a changed nested value that compared identical: a list
column holding ``list(range(1001))``, element 500 changed from 500 to 999999, gave
``diff_runs -> ({}, 1)``. ``astype(str)`` printed the array as ``[0 1 2 ... 998 999 1000]``.
Each case below writes real files and goes through ``diff_runs``, the path the compare
takes. The "equal" cases guard the other failure: a false difference.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("before_after_structural", ROOT / "scripts" / "before_after.py")
ba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ba)


def _runs(tmp_path: Path, name: str, base, head) -> tuple[dict[str, list[str]], int]:
    """One artifact per side (a DataFrame, an arrow Table or raw bytes), diffed as the compare does."""
    sides = []
    for side_name, payload in (("base", base), ("head", head)):
        run = tmp_path / side_name / "run"
        run.mkdir(parents=True)
        (run / ".run_id").write_text(f"{side_name}-run")
        if isinstance(payload, pa.Table):
            pq.write_table(payload, run / name)
        elif isinstance(payload, pd.DataFrame):
            payload.to_parquet(run / name)
        else:
            (run / name).write_bytes(payload)
        sides.append((ba.Side(side_name, tmp_path / side_name, sys.executable, {}, "0" * 40), run))
    (base, a), (head, b) = sides
    return ba.diff_runs(base, head, a, b, tmp_path)


def _list_frame(values) -> pd.DataFrame:
    return pd.DataFrame({"x": [values]})


# ------------------------------------------------------------------ the review's reproduction


def test_changed_element_deep_in_a_long_list_differs(tmp_path) -> None:
    changed = list(range(1001))
    changed[500] = 999999
    found, compared = _runs(tmp_path, "n.parquet", _list_frame(list(range(1001))), _list_frame(changed))
    assert compared == 1
    assert found.get("n.parquet"), "element 500 of a 1001-element list changed and compared identical"
    assert "x[row 0][500]" in found["n.parquet"][0], found["n.parquet"]


# ------------------------------------------------------------------ the class: nested values


def test_struct_column_nested_change_differs(tmp_path) -> None:
    def table(v):
        return pa.table({"s": pa.array([{"v": v, "w": 0.5}])})

    changed = list(range(1001))
    changed[700] = -1
    found, _ = _runs(tmp_path, "s.parquet", table(list(range(1001))), table(changed))
    assert found.get("s.parquet"), "a change inside a struct's list compared identical"
    assert "s[row 0].v[700]" in found["s.parquet"][0], found["s.parquet"]


def test_struct_float_within_tolerance_is_equal(tmp_path) -> None:
    def table(x):
        return pa.table({"s": pa.array([{"x": x, "n": 1}])})

    assert _runs(tmp_path, "s.parquet", table(0.1 + 0.2), table(0.3)) == ({}, 1)


def test_list_of_floats_within_tolerance_is_equal(tmp_path) -> None:
    assert _runs(tmp_path, "f.parquet", _list_frame([0.1 + 0.2, 1.0]), _list_frame([0.3, 1.0])) == ({}, 1)


def test_list_of_floats_outside_tolerance_differs(tmp_path) -> None:
    """|d| 4e-10 is outside rel 1e-9 of 0.3 (3e-10), but numpy prints both lists as [0.3 1. ]."""
    found, _ = _runs(tmp_path, "f.parquet", _list_frame([0.3, 1.0]), _list_frame([0.3000000004, 1.0]))
    assert found.get("f.parquet"), "a nested float outside the tolerance compared identical"


def test_null_inside_a_list_is_not_nan(tmp_path) -> None:
    """A null and a NaN both read back as nan through pandas; arrow keeps them apart."""
    a = pa.table({"x": pa.array([[1.0, None]], type=pa.list_(pa.float64()))})
    b = pa.table({"x": pa.array([[1.0, float("nan")]], type=pa.list_(pa.float64()))})
    found, _ = _runs(tmp_path, "z.parquet", a, b)
    assert found.get("z.parquet"), "null vs NaN inside a list compared identical"


def test_null_inside_a_list_vs_a_value_differs(tmp_path) -> None:
    found, _ = _runs(tmp_path, "z.parquet", _list_frame([1, None, 3]), _list_frame([1, 2, 3]))
    assert found.get("z.parquet")


def test_nan_inside_a_list_at_both_sides_is_equal(tmp_path) -> None:
    frame = _list_frame([1.0, float("nan")])
    assert _runs(tmp_path, "z.parquet", frame, frame.copy()) == ({}, 1)


def test_list_of_bool_vs_list_of_int_differs(tmp_path) -> None:
    found, _ = _runs(tmp_path, "b.parquet", _list_frame([True, False]), _list_frame([1, 0]))
    assert found.get("b.parquet"), "list<bool> vs list<int64> compared identical"


# ------------------------------------------------------------------ #326 round 1: dictionaries


def _categorical(values, categories, ordered=False) -> pd.DataFrame:
    return pd.DataFrame({"x": pd.Categorical(values, categories=categories, ordered=ordered)})


def test_ordered_category_order_change_differs(tmp_path) -> None:
    """The review's reproduction: the same values, the categories reordered, so sorting changes."""
    a = _categorical(["a", "b"], ["a", "b"], ordered=True)
    b = _categorical(["a", "b"], ["b", "a"], ordered=True)
    found, _ = _runs(tmp_path, "c.parquet", a, b)
    assert found.get("c.parquet"), "an ordered categorical's order changed and compared identical"
    assert "dictionary" in found["c.parquet"][0], found["c.parquet"]


def test_unused_category_change_differs(tmp_path) -> None:
    a = _categorical(["a", "a"], ["a", "z"])
    b = _categorical(["a", "a"], ["a", "y"])
    found, _ = _runs(tmp_path, "c.parquet", a, b)
    assert found.get("c.parquet"), "an unused category was replaced and compared identical"


def test_same_categorical_is_equal(tmp_path) -> None:
    frame = _categorical(["a", "b", None], ["a", "b", "c"], ordered=True)
    assert _runs(tmp_path, "c.parquet", frame, frame.copy()) == ({}, 1)


def test_dictionary_below_the_top_level_is_refused(tmp_path) -> None:
    """A list of dictionary values would decode and lose its dictionary: refused, never identical."""
    import pytest

    def table(dictionary):
        indices = pa.array([0], type=pa.int8())
        values = pa.DictionaryArray.from_arrays(indices, pa.array(dictionary))
        return pa.table({"l": pa.ListArray.from_arrays(pa.array([0, 1], type=pa.int32()), values)})

    with pytest.raises(ba.CouldNotRun, match="below its top level"):
        _runs(tmp_path, "l.parquet", table(["a", "z"]), table(["a", "y"]))


# ------------------------------------------------------------------ the class: other lossy paths


def test_range_index_change_differs(tmp_path) -> None:
    """A RangeIndex is stored only in the pandas metadata, not as a column."""
    a = pd.DataFrame({"x": [0.1, 0.2]})
    b = pd.DataFrame({"x": [0.1, 0.2]}, index=pd.RangeIndex(5, 7))
    found, _ = _runs(tmp_path, "r.parquet", a, b)
    assert found.get("r.parquet")


def test_non_utf8_text_compares_as_bytes(tmp_path) -> None:
    """errors="replace" read both invalid bytes as one U+FFFD."""
    found, _ = _runs(tmp_path, "blob.bin", b"head\xff\n", b"head\xfe\n")
    assert found.get("blob.bin"), "two different non-UTF-8 files compared identical"
    assert _runs(tmp_path / "same", "blob.bin", b"head\xff\n", b"head\xff\n") == ({}, 1)
