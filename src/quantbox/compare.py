"""``quantbox compare A B --rtol`` — whether two results agree within a tolerance, and by how much.

A claim such as "reproduced bit for bit" or "97 of 97 values identical" is replaced
by a measured one (TOM-3764): every number on both sides, how many differ at all,
the largest relative difference and where it is, and the verdict "equal within
rtol" or not.

A and B are the same kind:

* two JSON files (a line's analysis or results): every leaf, paired by its path;
* two run directories, or two directories of arms: the arms as
  :func:`quantbox.finding_export.load_arms` reads them, paired by name; per arm,
  every engine metric (``metrics.json``) and the returns, paired by date. The
  manifests are not compared: their paths, timestamps and commits differ by design.

The relative difference of two numbers is ``|a - b| / max(|a|, |b|)``, 0 when they
are equal (both 0, the same infinity, or both NaN). Anything that is not a number
must be equal exactly; a key, an arm or a date on one side only, a list of another
length, NaN against a number, is a structural difference. The two agree when there
is no structural difference and the largest relative difference is at most
``rtol``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_ID = "quantbox/compare@1"
DEFAULT_RTOL = 1e-9
_SHOWN = 10  # structural differences listed in the text output


class _Tally:
    """Every pair of numbers seen, reduced as it is added; and every structural difference."""

    def __init__(self) -> None:
        self.n_values = 0
        self.n_differ = 0
        self.max_rel = 0.0
        self.max_at: str | None = None
        self.mismatches: list[str] = []

    def numbers(self, where: str, a: Any, b: Any, labels: list[str] | None = None) -> None:
        x = np.atleast_1d(np.asarray(a, dtype=float))
        y = np.atleast_1d(np.asarray(b, dtype=float))
        nan_x, nan_y = np.isnan(x), np.isnan(y)
        one_nan = nan_x ^ nan_y
        same = (x == y) | (nan_x & nan_y)
        inf = np.isinf(x) | np.isinf(y)
        # NaN against a number, or an infinity against anything else: no relative difference exists
        bad = np.flatnonzero(one_nan | (inf & ~same))
        for i in bad[:_SHOWN]:
            self.mismatches.append(f"{self._at(where, labels, i)}: {float(x[i])!r} vs {float(y[i])!r}")
        if bad.size > _SHOWN:
            self.mismatches.append(f"{where}: {bad.size - _SHOWN} more NaN or infinite value(s) on one side only")
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(x - y) / np.maximum(np.abs(x), np.abs(y))
        rel = np.where(same | one_nan | inf, 0.0, rel)
        self.n_values += x.size
        self.n_differ += int(np.count_nonzero(rel))
        if rel.size and rel.max() > self.max_rel:
            i = int(rel.argmax())
            self.max_rel, self.max_at = float(rel[i]), self._at(where, labels, i)

    @staticmethod
    def _at(where: str, labels: list[str] | None, i: int) -> str:
        return f"{where}[{labels[i]}]" if labels is not None else where


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _walk(a: Any, b: Any, where: str, tally: _Tally) -> None:
    """Pair two JSON values leaf by leaf (module docstring)."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in [*a, *(k for k in b if k not in a)]:
            path = f"{where}.{key}" if where else str(key)
            if key not in b or key not in a:
                tally.mismatches.append(f"{path}: only in {'A' if key in a else 'B'}")
            else:
                _walk(a[key], b[key], path, tally)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            tally.mismatches.append(f"{where}: {len(a)} items vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _walk(x, y, f"{where}[{i}]", tally)
    elif _is_number(a) and _is_number(b):
        tally.numbers(where, a, b)
    elif type(a) is not type(b) or a != b:
        tally.mismatches.append(f"{where}: {a!r} vs {b!r}")


def _compare_json(a: Path, b: Path, tally: _Tally) -> None:
    docs = []
    for p in (a, b):
        try:
            docs.append(json.loads(p.read_text(encoding="utf-8")))
        except ValueError as exc:
            raise ValueError(f"{p}: not JSON ({exc})") from exc
    _walk(docs[0], docs[1], "", tally)


def _compare_runs(a: Path, b: Path, tally: _Tally) -> None:
    from .finding_export import load_arms

    arms_a = {arm.name: arm for arm in load_arms(a)}
    arms_b = {arm.name: arm for arm in load_arms(b)}
    for name in [*arms_a, *(n for n in arms_b if n not in arms_a)]:
        if name not in arms_a or name not in arms_b:
            tally.mismatches.append(f"arm {name!r}: only in {'A' if name in arms_a else 'B'}")
            continue
        x, y = arms_a[name], arms_b[name]
        _walk(x.metrics, y.metrics, f"{name}.metrics", tally)
        if not x.returns.index.equals(y.returns.index):
            only_a = x.returns.index.difference(y.returns.index)
            only_b = y.returns.index.difference(x.returns.index)
            tally.mismatches.append(
                f"{name}.returns: dates differ ({len(x.returns)} vs {len(y.returns)}; "
                f"{len(only_a)} only in A, {len(only_b)} only in B)"
            )
            continue
        labels = [d.isoformat() for d in x.returns.index]
        tally.numbers(f"{name}.returns", x.returns.to_numpy(), y.returns.to_numpy(), labels)


def compare(a: str | Path, b: str | Path, *, rtol: float = DEFAULT_RTOL) -> dict[str, Any]:
    """The ``quantbox/compare@1`` verdict on A and B (module docstring).

    Raises :class:`ValueError` for a refused input (a bad ``rtol``, A and B of
    different kinds, a file that is not JSON) and :class:`FileNotFoundError` when
    a path or a run under it does not exist.
    """
    if not (isinstance(rtol, (int, float)) and math.isfinite(rtol) and rtol >= 0):
        raise ValueError(f"rtol must be a finite number >= 0, got {rtol!r}")
    a, b = Path(a), Path(b)
    for p in (a, b):
        if not p.exists():
            raise FileNotFoundError(f"{p}: no such file or directory")
    tally = _Tally()
    if a.is_file() and b.is_file():
        kind = "json"
        _compare_json(a, b, tally)
    elif a.is_dir() and b.is_dir():
        kind = "runs"
        _compare_runs(a, b, tally)
    else:
        raise ValueError(f"{a} and {b} are not the same kind: compare two JSON files or two run/arms directories")
    return {
        "schema": SCHEMA_ID,
        "a": str(a),
        "b": str(b),
        "kind": kind,
        "rtol": float(rtol),
        "n_values": tally.n_values,
        "n_differ": tally.n_differ,
        "max_rel_diff": tally.max_rel,
        "max_rel_diff_at": tally.max_at,
        "mismatches": tally.mismatches,
        "equal": not tally.mismatches and tally.max_rel <= rtol,
    }


def headline(result: dict[str, Any]) -> str:
    """The one line a reader quotes: ``equal within rtol 1e-09: 75 numbers, 37 differ, max rel diff 9.3e-14 (...)``."""
    verdict = "equal" if result["equal"] else "NOT equal"
    line = (
        f"{verdict} within rtol {result['rtol']:g}: {result['n_values']} numbers, "
        f"{result['n_differ']} differ, max rel diff {result['max_rel_diff']:.2g}"
    )
    if result["max_rel_diff_at"]:
        line += f" ({result['max_rel_diff_at']})"
    if result["mismatches"]:
        line += f"; {len(result['mismatches'])} structural difference(s)"
    return line
