"""The line's trial log, ``<line dir>/trials.jsonl`` (TOM-3929).

Every result row of ``quantbox run``, ``arms`` and ``sweep`` inside a research line
appends one JSON object per line, and ``quantbox gates dsr --line <dir>`` takes its
n_trials from the log, so the Deflated Sharpe deflates by what was tried, not by a
hand count. The format is fixed by the owner and read by other repos
(qute-research ``quant_preflight.py``, the lab's gauntlet script); do not change it::

    {"ts": "<UTC ISO-8601>", "command": "run|arms|sweep", "config_sha256": "<hex>",
     "quantbox_version": "<x.y.z>", "data_pins": {"<dataset>": "<sha256>"},
     "metric": {"sharpe": <float>}}          # + "arm" on arms and sweep rows

n_trials is the number of DISTINCT ``config_sha256`` values: re-running an identical
config is not a new trial. Blank lines are ignored; any other line that is not a JSON
object naming a ``config_sha256`` is an error, never skipped, because a half-read
log undercounts. Only quantbox writes the file, by appending.

``config_sha256`` hashes the trial's config as canonical JSON (sorted keys) with the
keys that say where the output goes and how the batch counts left out
(:data:`NOT_THE_TRIAL`): an arms batch re-run lands in a new timestamped directory,
and adding an arm restamps ``run.n_trials`` on every arm, yet neither is a new trial.

A line is the nearest directory, from the config upwards, whose ``README.md`` carries
a fenced yaml block opening with ``prereg:`` (the block ``quantbox preflight`` and
qute-research's pre-flight read); the walk stops at a git root. ``--line`` names it
explicitly. No line, no log: the commands behave as before.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOG_NAME = "trials.jsonl"

#: Dotted config paths that are not part of a trial's identity.
NOT_THE_TRIAL: tuple[tuple[str, ...], ...] = (("artifacts",), ("run", "n_trials"))

_PREREG_RE = re.compile(r"^```+[ \t]*ya?ml[ \t]*\n(?:[ \t]*(?:#[^\n]*)?\n)*prereg:", re.M)


class TrialLogError(ValueError):
    """The trial log cannot be read (missing, or a line that is not a trial row)."""


def _plain(obj: Any) -> Any:
    if hasattr(obj, "item"):  # numpy / pandas scalars
        return obj.item()
    return str(obj)  # dates and paths as YAML handed them over


def config_sha256(cfg: Mapping[str, Any]) -> str:
    """sha256 of *cfg* as canonical JSON (sorted keys), without :data:`NOT_THE_TRIAL`."""
    trial = copy.deepcopy(dict(cfg))
    for path in NOT_THE_TRIAL:
        node: Any = trial
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(path[-1], None)
    text = json.dumps(trial, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_plain)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def trial_row(
    command: str,
    cfg: Mapping[str, Any],
    *,
    data_pins: Mapping[str, str | None],
    sharpe: Any,
    arm: str | None = None,
) -> dict[str, Any]:
    """One log row in the owner's format; a missing or non-finite Sharpe is ``null``."""
    from quantbox import __version__

    row: dict[str, Any] = {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": command,
        "config_sha256": config_sha256(cfg),
        "quantbox_version": __version__,
        "data_pins": dict(data_pins),
        "metric": {"sharpe": _finite(sharpe)},
    }
    if arm is not None:
        row["arm"] = arm
    return row


def is_line(directory: Path) -> bool:
    readme = directory / "README.md"
    try:
        return readme.is_file() and bool(_PREREG_RE.search(readme.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError):
        return False


def find_line(start: str | Path) -> Path | None:
    """The nearest line at or above *start* (a directory), stopping at a git root."""
    here = Path(start).resolve()
    for directory in (here, *here.parents):
        if is_line(directory):
            return directory
        if (directory / ".git").exists():
            return None
    return None


def resolve_line(explicit: str | None, config: str | Path) -> Path | None:
    """``--line`` when given (it must be a directory), else the line holding *config*."""
    if explicit is not None:
        directory = Path(explicit).resolve()
        if not directory.is_dir():
            raise TrialLogError(f"--line {explicit}: not a directory")
        return directory
    return find_line(Path(config).resolve().parent)


def append(line_dir: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Append *rows* to the line's log, one strict-JSON object per line, in one write."""
    path = Path(line_dir) / LOG_NAME
    text = "".join(json.dumps(dict(r), sort_keys=True, allow_nan=False) + "\n" for r in rows)
    if text:
        with path.open("a", encoding="utf-8") as f:
            f.write(text)
    return path


def count(line_dir: str | Path) -> int:
    """n_trials of a line: the distinct ``config_sha256`` values in its log.

    Raises :class:`TrialLogError` when there is no log, or a non-blank line is not a
    JSON object naming a ``config_sha256``.
    """
    path = Path(line_dir) / LOG_NAME
    if not path.is_file():
        raise TrialLogError(f"no trial log at {path}: no run, arms or sweep has logged a trial in this line")
    configs: set[str] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except ValueError as exc:
            raise TrialLogError(f"{path} line {number} is not JSON ({exc})") from None
        sha = row.get("config_sha256") if isinstance(row, dict) else None
        if not isinstance(sha, str) or not sha:
            raise TrialLogError(f"{path} line {number} names no `config_sha256`")
        configs.add(sha)
    return len(configs)
