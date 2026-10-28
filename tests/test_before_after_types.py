"""TOM-2547: scripts/before_after.py compares TYPE before value.

The v0.21.0 release review (quantbox#322, findings 1 and 102) reproduced a type
change that compared as identical: parquet ``weight=[0.5]`` against
``weight=["0.5"]``, and JSON ``true`` against ``1``. Each case below goes through
``diff_runs`` on real files, the path the compare takes, and must report a
difference. The last tests guard against the other failure: a false difference
where only the value representation is the same type.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("before_after_types", ROOT / "scripts" / "before_after.py")
ba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ba)


def _runs(tmp_path: Path, name: str, base, head) -> tuple[dict[str, list[str]], int]:
    """Write one artifact per side and diff the two run directories as the compare does."""
    sides = []
    for side_name, payload in (("base", base), ("head", head)):
        run = tmp_path / side_name / "run"
        run.mkdir(parents=True)
        (run / ".run_id").write_text(f"{side_name}-run")
        if name.endswith(".parquet"):
            payload.to_parquet(run / name)
        else:
            (run / name).write_text(json.dumps(payload))
        sides.append((ba.Side(side_name, tmp_path / side_name, sys.executable, {}, "0" * 40), run))
    (base, a), (head, b) = sides
    return ba.diff_runs(base, head, a, b, tmp_path)


# ------------------------------------------------------------------ the review's reproductions


def test_parquet_number_vs_string_differs(tmp_path) -> None:  # finding 1
    found, compared = _runs(tmp_path, "w.parquet", pd.DataFrame({"weight": [0.5]}), pd.DataFrame({"weight": ["0.5"]}))
    assert compared == 1
    assert "w.parquet" in found, "float 0.5 vs string '0.5' compared identical"


@pytest.mark.parametrize(("a", "b"), [(True, 1), (False, 0), (1, True), (0.0, False)])
def test_json_bool_vs_number_differs(tmp_path, a, b) -> None:  # finding 102
    found, compared = _runs(tmp_path, "r.json", {"flag": a}, {"flag": b})
    assert compared == 1
    assert found.get("r.json"), f"JSON {a!r} vs {b!r} compared identical"


# ------------------------------------------------------------------ the class: parquet


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([1, 0], [True, False]),  # int vs bool
        ([True, False], [1, 0]),  # bool vs int
        ([1, 2], [1.0, 2.0]),  # int64 vs float64
        (pd.array([1, None], dtype="Int64"), [1.0, np.nan]),  # nullable int vs float
    ],
)
def test_parquet_column_type_change_differs(tmp_path, a, b) -> None:
    found, _ = _runs(tmp_path, "w.parquet", pd.DataFrame({"x": a}), pd.DataFrame({"x": b}))
    assert "w.parquet" in found, f"{a!r} vs {b!r} compared identical"


def test_parquet_decimal_vs_string_differs(tmp_path) -> None:
    """Both read back as pandas object columns and print '0.5'; only the parquet schema tells them apart."""
    from decimal import Decimal

    a = pd.DataFrame({"x": [Decimal("0.5")]})
    b = pd.DataFrame({"x": ["0.5"]})
    found, _ = _runs(tmp_path, "w.parquet", a, b)
    assert "w.parquet" in found


def test_parquet_null_mask_change_differs(tmp_path) -> None:
    """None and the string 'None' print the same; the null mask tells them apart."""
    a = pd.DataFrame({"x": pd.Series(["a", None], dtype=object)})
    b = pd.DataFrame({"x": pd.Series(["a", "None"], dtype=object)})
    found, _ = _runs(tmp_path, "w.parquet", a, b)
    assert "w.parquet" in found


def test_parquet_index_dtype_change_differs(tmp_path) -> None:
    """pandas Index.equals ignores dtype: [1, 2] equals [1.0, 2.0]."""
    a = pd.DataFrame({"x": [0.1, 0.2]}, index=pd.Index([1, 2]))
    b = pd.DataFrame({"x": [0.1, 0.2]}, index=pd.Index([1.0, 2.0]))
    found, _ = _runs(tmp_path, "w.parquet", a, b)
    assert "w.parquet" in found


def test_parquet_large_int_is_compared_exactly(tmp_path) -> None:
    """An int column was cast to float and compared within rel 1e-9; a change of 1 in 2**53 vanished."""
    a = pd.DataFrame({"x": [2**53]})
    b = pd.DataFrame({"x": [2**53 + 1]})
    found, _ = _runs(tmp_path, "w.parquet", a, b)
    assert "w.parquet" in found


# ------------------------------------------------------------------ the class: JSON


@pytest.mark.parametrize(("a", "b"), [("0.5", 0.5), (None, 0), (None, False), ([1], {"0": 1}), ("true", True)])
def test_json_other_type_changes_differ(tmp_path, a, b) -> None:
    found, _ = _runs(tmp_path, "r.json", {"v": a}, {"v": b})
    assert found.get("r.json"), f"JSON {a!r} vs {b!r} compared identical"


# ------------------------------------------------------------------ no false difference


def test_same_types_and_values_are_identical(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "w": [0.5, np.nan, 0.25],
            "n": [1, 2, 3],
            "ok": [True, False, True],
            "s": ["a", None, "c"],
        },
        index=pd.date_range("2026-01-01", periods=3, tz="UTC"),
    )
    found, compared = _runs(tmp_path, "w.parquet", frame, frame.copy())
    assert (found, compared) == ({}, 1)


def test_nullable_float_column_compares(tmp_path) -> None:
    """A nullable Float64 column with a null compares (nulls leave before the cast) and finds the change."""
    a = pd.DataFrame({"w": pd.array([0.5, None], dtype="Float64")})
    b = pd.DataFrame({"w": pd.array([0.6, None], dtype="Float64")})
    found, _ = _runs(tmp_path, "w.parquet", a, b)
    assert "w.parquet" in found
    assert _runs(tmp_path / "same", "w.parquet", a, a.copy()) == ({}, 1)


def test_float_tolerance_is_unchanged(tmp_path) -> None:
    """Floats keep rel 1e-9, abs 1e-12: a last-bit change is not a difference."""
    a = pd.DataFrame({"w": [0.1 + 0.2]})
    b = pd.DataFrame({"w": [0.3]})
    assert _runs(tmp_path, "w.parquet", a, b) == ({}, 1)


def test_json_int_vs_float_is_not_a_difference(tmp_path) -> None:
    """JSON has one number type (RFC 8259; JSON Schema's integer accepts 1.0). The old code agreed."""
    assert _runs(tmp_path, "r.json", {"n": 1, "b": True}, {"n": 1.0, "b": True}) == ({}, 1)
