"""``quantbox preflight`` — the gate run BEFORE any measured backtest (ADR-0013).

Ported from qute-research 0.13.0 ``scripts/quant_preflight.py``, behaviour-identical,
so the gate is versioned and pinned with the engine it judges. Where the plugin
asked quantbox through a subprocess, this module calls the same quantbox code
in-process (the sites are listed in ADR-0013).

A backtest that passes its own criterion can still say nothing about the book
it names: the primary dataset is a market the book does not trade, the config
holds positions the venue cannot, the claim has no significance test of its
own, the criterion was written after the first run. Every one of those is
visible in FILES before a single bar is simulated, so they are refused here,
with an exit code, instead of being caveated afterwards.

    quantbox preflight --lab <lab root> --line <line dir> [--prereg <file>]
                       [--id <hypothesis id>] [--runs-dir <dir> ...] <config> ...

WHAT IT READS — nothing else
----------------------------
  * `<lab>/system.yaml`            the lab's declaration (book, market, venue,
                                   liveness floor). Contract:
                                   skills/research/references/system.md
                                   A `kind: replication` block is anchored to its
                                   `paper:` instead of a book (TOM-2231); only the
                                   liveness floor is read from here then
  * the pre-registration block    a fenced ```yaml block whose top-level key
                                   is `prereg:`, in `<line>/README.md` (or
                                   `--prereg`). Prose around it is for people;
                                   the block is what this gate reads.
  * quantbox's PLAN for each config :func:`quantbox.explain.explain_path`
                                   (quantbox/explain@1, what `quantbox config
                                   explain --json` prints), run from the line: the
                                   engine, the dataset (name, pin, market),
                                   funding, execution timing and venue AS THE
                                   RUNNER WILL APPLY THEM, and the artifact root.
                                   The config itself is never parsed here and no
                                   dataset layout is assumed (TOM-1367). A by-name
                                   read the plan leaves unresolved is settled by
                                   :func:`quantbox.dataset_lock.resolve_dataset`
                                   at the lock :func:`~quantbox.dataset_lock.lock_for_run`
                                   names (what `quantbox dataset resolve --config`
                                   prints)
  * `<line>/data_inputs.yaml`      content hashes, for data a config reads by path
  * git                            when the frozen part of the block last changed
  * the whole line directory       for run evidence (any `*manifest*.json`, any
                                   run-stamped directory) to compare that against;
                                   a `quantbox/run@N` manifest of a major this
                                   quantbox does not write (:data:`quantbox.run_manifest.SCHEMA_ID`)
                                   is BLIND

DETERMINISTIC ON PURPOSE. Anything needing judgement — is the criterion the
right one, is every earlier config counted — belongs to the `prereg-auditor`
agent. One enforcer per rule: this script owns what a file can answer, and
quantbox owns what its runner will do.

BLIND IS NOT CLEAN
------------------
A file this gate needs and cannot parse, a git read that failed, a plan quantbox
could not produce, a plan or a run manifest of a schema major this quantbox does not read, a
number YAML readers disagree about, a duplicate key, an unexpected exception
inside the gate itself: that check reports BLIND and the run exits 2. A refusal
outranks blindness (exit 1), and nothing blind ever exits 0.

NOTHING TRUSTS A TYPE. A threshold is a plain finite number or it is refused by
name; a plan field of the wrong type is BLIND, never coerced.

NOT AN ADVERSARY'S GATE. Commit dates are trusted; a backdated or squashed
history reads as "frozen early" (preregistration.md § Stated limits).

EXIT CODES
----------
  0  every check passed
  1  at least one REFUSE — do not run the backtest
  2  no refusal, but at least one check COULD NOT LOOK (also: usage errors)

THE LAB'S YAML. system.yaml, the prereg block and data_inputs.yaml are read by the
block-style subset parser below, kept from the plugin unchanged: it REFUSES TO
GUESS. Anchors, aliases, tags, merge keys and multi-line flow collections raise,
and such a file is then read by PyYAML with duplicate keys refused. A duplicate
key or a second document is BLIND under either reader. A config is never read
here: quantbox's plan is.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from . import explain as _explain
from . import run_manifest as _run_manifest
from .dataset_lock import lock_for_run
from .dataset_lock import resolve_dataset as _resolve_dataset
from .run_history import RUN_TS_FORMAT

__all__ = ["BLIND", "PASS", "REFUSE", "Report", "Unavailable", "main", "render", "run"]

PASS, REFUSE, BLIND = "PASS", "REFUSE", "BLIND"


class Unavailable(RuntimeError):
    """quantbox could not produce an answer — the caller reports BLIND, never a pass."""


_SCHEMA_RE = re.compile(r"^quantbox/([a-z_]+)@(\d+)$")


def _major(schema_id: str) -> int:
    match = _SCHEMA_RE.match(schema_id)
    if match is None:  # pragma: no cover — quantbox's own constants
        raise ValueError(f"not a quantbox schema id: {schema_id!r}")
    return int(match.group(2))


#: Schema majors this gate reads, per quantbox document kind: the ones THIS quantbox
#: writes. A document naming any other major is BLIND, never read as the nearest one.
SUPPORTED: dict[str, frozenset[int]] = {
    "run": frozenset({_major(_run_manifest.SCHEMA_ID)}),
    "explain": frozenset({_major(_explain.SCHEMA_ID)}),
}


def schema_problem(doc, kind: str) -> str | None:
    """Why `doc` is not a `quantbox/<kind>@N` this quantbox reads — or None when it is."""
    if not isinstance(doc, dict):
        return f"not a JSON object ({type(doc).__name__}), so not a quantbox/{kind} document"
    raw = doc.get("schema")
    match = _SCHEMA_RE.match(raw) if isinstance(raw, str) else None
    if match is None:
        return f"`schema` is {raw!r}, not `quantbox/{kind}@<N>` — which contract this is cannot be told"
    if match.group(1) != kind:
        return f"`schema` is {raw!r}, a quantbox/{match.group(1)} document where quantbox/{kind} was expected"
    supported = SUPPORTED[kind]
    if int(match.group(2)) not in supported:
        known = ", ".join(f"quantbox/{kind}@{n}" for n in sorted(supported))
        return (
            f"`schema` is {raw!r}; this quantbox reads {known} only — a major it does not know is not "
            "read as the nearest one it resembles. Run the pre-flight with the quantbox that wrote it"
        )
    return None


def is_quantbox_document(doc, kind: str) -> bool:
    """Does `doc` CLAIM to be a quantbox/<kind> document, at any version?"""
    raw = doc.get("schema") if isinstance(doc, dict) else None
    return isinstance(raw, str) and raw.startswith(f"quantbox/{kind}@")


MANDATES = ("alpha", "zero-edge-test", "client-shadow")
VERDICTS = ("confirmed", "refuted", "inconclusive", "superseded")

#: What a dataset manifest may call the market a book declares. Vocabulary, not
#: lab facts: a lab never edits this, it declares `market:` and the dataset's
#: own manifest says what it is.
MARKET_ALIASES = {
    "perp": {"perp", "perps", "perpetual", "perpetuals", "futures", "swap"},
    "spot": {"spot"},
    "etf": {"etf", "etfs"},
    "fund": {"fund", "funds", "mutual-fund"},
}

#: The part of a pre-registration that may not move once a run exists. `n_trials`
#: is deliberately absent — it GROWS with every arm, by design — and so are `id`
#: and anything this gate does not read. Everything a check compares against IS
#: here: loosening `liveness`, adding a secondary dataset or writing a
#: `lag0_reason` after the numbers are in is the same move as editing a threshold.
FROZEN_FIELDS = (
    "kind",
    "book",
    "paper",
    "primary_dataset",
    "secondary_datasets",
    "criterion",
    "claim_test",
    "ex_largest_episode",
    "execution",
    "funding",
    "liveness",
    "runs",
    "not_runs",
)


# ── the YAML subset ────────────────────────────────────────────────────────


class YamlUnsupported(ValueError):
    """A construct this parser will not guess at (PyYAML may read it instead)."""


class YamlRefused(YamlUnsupported):
    """YAML that NO reader gets to decide: a duplicate key, a second document.

    PyYAML would answer — last key wins, first document wins — and that answer
    is exactly the ambiguity a gate must not inherit. Never falls back.
    """


_KEY_RE = re.compile(r"""^(?:"([^"]+)"|'([^']+)'|([^\s:#\-\[\]{}&*!|>'"%@`][^:#]*?)):(?:\s+(.*))?$""")
#: What a PyYAML-loading engine reads as a boolean (YAML 1.1), exact spellings.
_TRUE = {"true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"}
_FALSE = {"false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF"}
_NULL = {"null", "Null", "NULL", "~"}
_INT_RE = re.compile(r"^[-+]?(?:0|[1-9]\d*)$")
_FLOAT_RE = re.compile(r"^[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]\d+)?$")
_INF_RE = re.compile(r"^([-+]?)\.(?:inf|Inf|INF)$")
_NAN_RE = re.compile(r"^\.(?:nan|NaN|NAN)$")
#: Number-SHAPED text that YAML 1.1 and 1.2 readers disagree about, or that this
#: parser would otherwise hand back as a harmless-looking string: `1_0` and `0x10`
#: are 10 and 16 to PyYAML, `010` is 8, `1:30` is 90, `1e3` is a STRING to PyYAML
#: and 1000 to a 1.2 reader. A gate that guesses here passes leverage 16 as text.
_AMBIGUOUS_NUMBER_RE = re.compile(
    r"^[-+]?(?:"
    r"\d[\d_]*_[\d_]*(?:\.[\d_]*)?(?:[eE][-+]?\d+)?"  # underscores
    r"|\d*\.[\d_]*_[\d_]*(?:[eE][-+]?\d+)?"
    r"|0[xX][0-9a-fA-F_]+|0[oO][0-7_]+|0[bB][01_]+"  # hex / octal / binary
    r"|0\d+"  # leading-zero octal
    r"|\d[\d_]*(?::[0-5]?\d)+(?:\.[\d_]*)?"  # sexagesimal
    r"|\d+[eE][-+]?\d+"  # exponent without a dot
    r"|(?:\d+\.\d*|\.\d+)[eE]\d+"  # exponent without a sign
    r")$"
)
_BLOCK_SCALAR_RE = re.compile(r"^[|>][-+]?\d*$")


def _strip_comment(line: str) -> str:
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


def _closes(text: str) -> bool:
    """Does a scalar opening with a quote also close on this line?"""
    quote = text[0]
    body = text[1:].replace("\\" + quote, "") if quote == '"' else text[1:].replace("''", "")
    return body.endswith(quote)


def _scalar(text: str):
    text = text.strip()
    if not text:
        return None
    if text[0] in "&*!" or text.startswith("<<"):
        raise YamlUnsupported(f"anchor, alias, tag or merge key: {text!r}")
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        inner = text[1:-1]
        return inner.replace("''", "'") if text[0] == "'" else inner
    if text in _NULL:
        return None
    if text in _TRUE or text in _FALSE:
        return text in _TRUE
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    if _INF_RE.match(text):
        return float("-inf") if text.startswith("-") else float("inf")
    if _NAN_RE.match(text):
        return float("nan")
    if _AMBIGUOUS_NUMBER_RE.match(text):
        raise YamlUnsupported(
            f"number-shaped scalar that YAML readers disagree about: {text!r} — write it as a plain decimal"
        )
    return text


def _flow(text: str):
    """One-line flow collection: `[a, [b, c]]`, `{k: v}`."""
    pos = 0

    def ws():
        nonlocal pos
        while pos < len(text) and text[pos] in " \t":
            pos += 1

    def atom(stops: str):
        nonlocal pos
        ws()
        if pos < len(text) and text[pos] in "\"'":
            quote, start = text[pos], pos + 1
            end = text.find(quote, start)
            if end < 0:
                raise YamlUnsupported(f"unterminated quote in flow: {text!r}")
            pos = end + 1
            return text[start:end]
        start = pos
        while pos < len(text) and text[pos] not in stops:
            pos += 1
        return _scalar(text[start:pos])

    def value(stops: str):
        nonlocal pos
        ws()
        if pos >= len(text):
            raise YamlUnsupported(f"flow collection does not close on its line: {text!r}")
        if text[pos] == "[":
            pos += 1
            out = []
            while True:
                ws()
                if pos >= len(text):
                    raise YamlUnsupported(f"flow collection does not close on its line: {text!r}")
                if text[pos] == "]":
                    pos += 1
                    return out
                out.append(value(",]"))
                ws()
                if pos < len(text) and text[pos] == ",":
                    pos += 1
        if text[pos] == "{":
            pos += 1
            out = {}
            while True:
                ws()
                if pos >= len(text):
                    raise YamlUnsupported(f"flow collection does not close on its line: {text!r}")
                if text[pos] == "}":
                    pos += 1
                    return out
                key = atom(":")
                if pos >= len(text) or text[pos] != ":":
                    raise YamlUnsupported(f"flow mapping without a value: {text!r}")
                pos += 1
                if str(key) in out:
                    raise YamlRefused(f"duplicate key {key!r} in a flow mapping")
                out[str(key)] = value(",}")
                ws()
                if pos < len(text) and text[pos] == ",":
                    pos += 1
        return atom(stops)

    result = value("")
    ws()
    if pos != len(text):
        raise YamlUnsupported(f"trailing text after flow collection: {text!r}")
    return result


def _value(text: str):
    text = text.strip()
    if text and text[0] in "[{":
        return _flow(text)
    return _scalar(text)


class _Parser:
    def __init__(self, text: str):
        self.rows: list[tuple[int, str, str]] = []  # indent, content, raw
        seen_content = False
        for raw in text.lstrip("\ufeff").splitlines():
            if raw.strip() in ("---", "..."):
                if raw.strip() == "---" and seen_content:
                    raise YamlRefused("a second YAML document — which one counts is a guess")
                continue
            seen_content = seen_content or bool(_strip_comment(raw).strip())
            content = _strip_comment(raw).rstrip()
            indent = len(raw) - len(raw.lstrip(" "))
            if "\t" in raw[: indent + 1]:
                raise YamlUnsupported("tab indentation")
            self.rows.append((indent, content.strip(), raw))
        self.i = 0

    def _skip_blank(self):
        while self.i < len(self.rows) and not self.rows[self.i][1]:
            self.i += 1

    def parse(self):
        self._skip_blank()
        if self.i >= len(self.rows):
            return None
        out = self._block(self.rows[self.i][0])
        self._skip_blank()
        if self.i < len(self.rows):
            raise YamlUnsupported(f"unexpected dedent/indent at: {self.rows[self.i][2]!r}")
        return out

    def _block(self, indent: int):
        content = self.rows[self.i][1]
        if content == "-" or content.startswith("- "):
            return self._sequence(indent)
        if _KEY_RE.match(content):
            return self._mapping(indent)
        self.i += 1
        return _value(content)

    def _sequence(self, indent: int):
        out = []
        while True:
            self._skip_blank()
            if self.i >= len(self.rows):
                return out
            row_indent, content, raw = self.rows[self.i]
            if row_indent != indent or not (content == "-" or content.startswith("- ")):
                if row_indent > indent:
                    raise YamlUnsupported(f"unexpected indent at: {raw!r}")
                return out
            rest = content[1:].lstrip()
            if not rest:
                self.i += 1
                self._skip_blank()
                if self.i < len(self.rows) and self.rows[self.i][0] > indent:
                    out.append(self._block(self.rows[self.i][0]))
                else:
                    out.append(None)
                continue
            if _KEY_RE.match(rest) and rest[:1] not in "[{":
                # `- key: value` opens a mapping whose keys sit where `key` starts
                inner = indent + (len(content) - len(rest))
                self.rows[self.i] = (inner, rest, raw)
                out.append(self._mapping(inner))
            else:
                self.i += 1
                out.append(_value(rest))

    def _mapping(self, indent: int):
        out: dict = {}
        while True:
            self._skip_blank()
            if self.i >= len(self.rows):
                return out
            row_indent, content, raw = self.rows[self.i]
            if row_indent < indent:
                return out
            if row_indent > indent:
                raise YamlUnsupported(f"unexpected indent at: {raw!r}")
            match = _KEY_RE.match(content)
            if not match:
                if content == "-" or content.startswith("- "):
                    return out
                raise YamlUnsupported(f"not a `key: value` line: {raw!r}")
            key = match.group(1) or match.group(2) or match.group(3).strip()
            if key == "<<":
                raise YamlUnsupported("merge key `<<`")
            if key in out:
                raise YamlRefused(f"duplicate key {key!r} — which value counts is a guess")
            rest = (match.group(4) or "").strip()
            self.i += 1
            if _BLOCK_SCALAR_RE.match(rest):
                out[key] = self._block_scalar(indent)
            elif rest and rest[0] in "\"'" and not _closes(rest):
                out[key] = self._quoted_continuation(rest)
            elif rest and rest[0] not in "\"'[{" and (": " in rest or rest.startswith("- ") or "\t" in raw):
                raise YamlUnsupported(f"plain scalar no YAML reader accepts: {raw!r}")
            elif rest:
                value = _value(rest)
                if isinstance(value, str) and rest[:1] not in "\"'":
                    value = self._continuation(indent, value)
                out[key] = value
            else:
                self._skip_blank()
                if self.i < len(self.rows):
                    nxt_indent, nxt, _ = self.rows[self.i]
                    is_item = nxt == "-" or nxt.startswith("- ")
                    if nxt_indent > indent or (nxt_indent == indent and is_item):
                        out[key] = self._block(nxt_indent)
                        continue
                out[key] = None

    def _quoted_continuation(self, first: str) -> str:
        """A quoted scalar that closes on a later line. Folded approximately: no
        check here reads such a value, the point is not to go blind on the file."""
        quote, parts = first[0], [first[1:]]
        while self.i < len(self.rows):
            raw = self.rows[self.i][2].strip()
            self.i += 1
            if _closes(quote + raw):
                parts.append(raw[:-1])
                return " ".join(part.rstrip("\\").strip() for part in parts)
            parts.append(raw)
        raise YamlUnsupported(f"quoted scalar never closes: {first!r}")

    def _continuation(self, indent: int, value: str) -> str:
        """A plain scalar folded onto deeper-indented lines."""
        while self.i < len(self.rows):
            row_indent, content, _ = self.rows[self.i]
            if not content or row_indent <= indent:
                break
            value = f"{value} {content}"
            self.i += 1
        return value

    def _block_scalar(self, indent: int) -> str:
        lines = []
        while self.i < len(self.rows):
            row_indent, _, raw = self.rows[self.i]
            if raw.strip() and row_indent <= indent:
                break
            lines.append(raw.strip())
            self.i += 1
        return "\n".join(lines).strip()


def parse_yaml(text: str):
    return _Parser(text).parse()


def load_yaml(path: Path):
    """The subset parser; PyYAML only for what the subset refuses, if it is there.

    Same arrangement as `qute_review_policy`: the stdlib reader is THE parser,
    PyYAML the bonus. A config using anchors or merge keys is read on a box
    that has PyYAML (a lab venv does) and goes BLIND on one that does not —
    never half-read.
    """
    text = path.read_text(encoding="utf-8")
    try:
        return parse_yaml(text)
    except YamlRefused:
        raise
    except YamlUnsupported as unsupported:
        try:
            import yaml  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError:
            raise YamlUnsupported(f"{unsupported} (and PyYAML is not importable)") from None

        class _NoDuplicates(yaml.SafeLoader):
            def construct_mapping(self, node, deep=False):
                keys = [k.value for k, _ in node.value if isinstance(k.value, str)]
                repeated = sorted({k for k in keys if keys.count(k) > 1 and k != "<<"})
                if repeated:
                    raise YamlRefused(f"duplicate key(s) {repeated}")
                return super().construct_mapping(node, deep=deep)

        try:
            return yaml.load(text, Loader=_NoDuplicates)  # noqa: S506 — SafeLoader subclass
        except yaml.YAMLError as exc:
            raise YamlUnsupported(f"not valid YAML: {exc}") from None


# ── reading the declared files ─────────────────────────────────────────────

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})\s*([\w+-]*)\s*$")


def prereg_blocks(text: str) -> list[dict]:
    """Every TOP-LEVEL fenced yaml block whose top-level key is `prereg:`.

    Fences nest by length: a block opened with four backticks closes only on four
    or more, so a ```yaml example INSIDE a ````markdown fence is documentation,
    not THE pre-registration. A yaml fence that opens with `prereg:` and fails to
    parse raises, so a broken pre-registration is blind, never absent.
    """
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        opened = _FENCE_RE.match(lines[i])
        if not opened:
            i += 1
            continue
        mark, language = opened.group(1), opened.group(2).lower()
        j = i + 1
        while j < len(lines):
            closing = _FENCE_RE.match(lines[j])
            if (
                closing
                and closing.group(1)[0] == mark[0]
                and len(closing.group(1)) >= len(mark)
                and not closing.group(2)
            ):
                break
            j += 1
        if language in ("yaml", "yml"):
            body = "\n".join(lines[i + 1 : j])
            first = next((ln for ln in body.splitlines() if _strip_comment(ln).strip()), "")
            if first.strip().startswith("prereg:"):
                parsed = parse_yaml(body)
                if isinstance(parsed, dict) and isinstance(parsed.get("prereg"), dict):
                    out.append(parsed["prereg"])
        i = j + 1
    return out


def select_prereg(blocks: list[dict], wanted: str | None) -> dict | None:
    if wanted is not None:
        hits = [b for b in blocks if str(b.get("id")) == wanted]
        return hits[0] if len(hits) == 1 else None
    return blocks[0] if len(blocks) == 1 else None


def fingerprint(prereg: dict) -> str:
    frozen = {k: prereg.get(k) for k in FROZEN_FIELDS}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, default=str).encode()).hexdigest()


def canonical_market(raw) -> str | None:
    text = str(raw).strip().lower()
    for market, names in MARKET_ALIASES.items():
        if text in names:
            return market
    return text or None


def _is_number(value) -> bool:
    """A plain FINITE number. Not a bool, not a numeric string, not inf, not nan."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if done.returncode != 0:
        raise OSError(f"git {' '.join(args)} failed: {done.stderr.strip()[:200]}")
    return done.stdout


def _iso(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── run evidence ───────────────────────────────────────────────────────────

_RUN_STAMP_RE = re.compile(r"(\d{8}T\d{6})Z")
_MANIFEST_RE = re.compile(r"manifest.*\.json$", re.IGNORECASE)
#: Never walked: version control and interpreter environments — tens of thousands of
#: files that are not the line's. The row SAYS they were not walked.
_NEVER_WALKED = {".git", ".hg", ".svn", ".venv", "venv", ".tox"}
#: Walked for run-STAMPED directories only: a package tree is full of `manifest.json`
#: files that are not runs, but a run parked there still carries its stamp.
_STAMPS_ONLY = {"node_modules", "__pycache__"}
_NOT_EVIDENCE = {".gitkeep", ".gitignore", ".DS_Store"}


@dataclass(frozen=True)
class Evidence:
    path: Path
    when: float
    source: str  # "dir stamp" | "mtime"


def _stamp_of(name: str) -> float | None:
    match = _RUN_STAMP_RE.search(name)
    if not match:
        return None
    try:
        stamp = datetime.strptime(match.group(0), RUN_TS_FORMAT)  # the run id's own stamp format
    except ValueError:
        return None
    return stamp.replace(tzinfo=UTC).timestamp()


def run_evidence(root: Path, everything: bool, found: dict, undatable: list[str], unwalked: set[str]) -> None:
    """Collect what says "a run happened" under `root` into `found`.

    `everything=True` is for a DECLARED runs root: every file under it is a run's
    output, whatever it is called and however the directory is named (hidden,
    `node_modules`, inside a dataset directory — a declared root is always walked).
    Elsewhere only run MANIFESTS (`*manifest*.json`) and run-stamped directories
    count. Evidence whose time cannot be read goes to `undatable`, never away.
    """
    if not root.is_dir():
        return
    for folder, dirs, files in os.walk(root):
        here = Path(folder)
        stamps_only = not everything and any(part in _STAMPS_ONLY for part in here.relative_to(root).parts)
        kept = []
        for name in dirs:
            if not everything and name in _NEVER_WALKED:
                unwalked.add(name)
            else:
                kept.append(name)
        dirs[:] = kept
        for name in dirs:
            when = _stamp_of(name)
            if when is not None:
                found.setdefault((here / name).resolve(), Evidence(here / name, when, "dir stamp"))
        if stamps_only:
            continue
        for name in files:
            if name in _NOT_EVIDENCE or not (everything or _MANIFEST_RE.search(name)):
                continue
            path = here / name
            stamped = next((p for p in [here, *here.parents] if p.resolve() in found), None)
            if stamped is not None:
                continue  # its run-stamped directory already stands for it
            try:
                found.setdefault(path.resolve(), Evidence(path, path.stat().st_mtime, "mtime"))
            except OSError as exc:
                undatable.append(f"{path}: {exc}")


def declared_paths(block: dict, key: str, line: Path) -> tuple[list[Path], list[str]]:
    """`runs:` (a list of paths) / `not_runs:` (a list of `{path, reason}`), resolved
    against the line; plus what is wrong with the declaration."""
    value, problems, out = block.get(key), [], []
    if value is None:
        return [], []
    if not isinstance(value, list):
        return [], [f"`{key}` must be a list"]
    for item in value:
        path = item.get("path") if isinstance(item, dict) else item
        if key == "not_runs" and not (
            isinstance(item, dict) and isinstance(item.get("reason"), str) and item["reason"].strip()
        ):
            problems.append(f"`not_runs` entry {item!r} needs a `path` and a written `reason`")
        elif not isinstance(path, str) or not path.strip():
            problems.append(f"`{key}` entry {item!r} is not a path")
        else:
            out.append(Path(path) if Path(path).is_absolute() else line / path)
    return out, problems


def _under(path: Path, roots: list[Path]) -> bool:
    return any(_within(path, root) for root in roots)


# ── the gate ───────────────────────────────────────────────────────────────


@dataclass
class Report:
    rows: list[tuple[str, str, str]] = field(default_factory=list)
    read: list[str] = field(default_factory=list)

    def add(self, status: str, check: str, reason: str) -> None:
        self.rows.append((status, check, reason))

    @property
    def exit_code(self) -> int:
        statuses = {status for status, _, _ in self.rows}
        if REFUSE in statuses:
            return 1
        if BLIND in statuses or not self.rows:
            return 2
        return 0


#: What the scaffold templates actually emit: a whole `<words>` token standing alone or
#: as a path segment (`<number>`, `<owner>/<prod-repo>`, `docs/<book>.md`), or a leading
#: TBD / TODO / FIXME / XXX *word*. NOT any text between two comparison signs —
#: `"MDD < 30% and Calmar > 1"` is a criterion — and not an empty string: an empty
#: REQUIRED field is refused by the check that requires it, an empty optional one is fine.
_PLACEHOLDER_RE = re.compile(
    r"(?:^|[\s/])<[A-Za-z][^<>]*[^\s<>]>(?=$|[\s/.,;:])|(?:^|[\s/])<[A-Za-z]>(?=$|[\s/.,;:])"
    r"|^\s*(?:TBD|TODO|FIXME|XXX)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def _placeholders(node, trail: str = "") -> list[str]:
    """Paths still carrying the template's text — `<...>`, TBD/TODO, or nothing at
    all — in a VALUE or in a KEY. One sweep for both declared files, so a field is
    never unfilled-but-green by the luck of which check happens to read it."""
    if isinstance(node, dict):
        hits = []
        for key, value in node.items():
            path = f"{trail}.{key}" if trail else str(key)
            if isinstance(key, str) and _PLACEHOLDER_RE.search(key):
                hits.append(f"{path} (key)")
            hits += _placeholders(value, path)
        return hits
    if isinstance(node, list):
        return [hit for n, v in enumerate(node) for hit in _placeholders(v, f"{trail}[{n}]")]
    return [trail] if isinstance(node, str) and _PLACEHOLDER_RE.search(node) else []


def venue_of(book: dict) -> dict:
    """The book's venue with the SAFE answer for anything it fails to declare:
    no shorts, no leverage. `check_system` refuses an incomplete venue by name;
    nothing downstream may turn the same gap into a traceback — or a permission."""
    venue = book.get("venue") if isinstance(book.get("venue"), dict) else {}
    return {
        "allow_shorts": venue.get("allow_shorts") is True,
        "leverage": venue.get("leverage") if _is_number(venue.get("leverage")) else 0,
    }


def check_system(report: Report, lab: Path) -> dict | None:
    path = lab / "system.yaml"
    if not path.is_file():
        report.add(REFUSE, "system", f"no system.yaml — cannot determine ({path})")
        return None
    report.read.append(str(path))
    try:
        system = load_yaml(path)
    except (YamlUnsupported, OSError, UnicodeDecodeError) as exc:
        report.add(BLIND, "system", f"system.yaml unreadable: {exc}")
        return None
    if not isinstance(system, dict):
        report.add(REFUSE, "system", "system.yaml is not a mapping — cannot determine")
        return None
    problems = []
    if not isinstance(system.get("system"), str):
        problems.append("`system` (name) missing")
    if system.get("mandate") not in MANDATES:
        problems.append(f"`mandate` must be one of {'|'.join(MANDATES)}")
    if not system.get("production"):
        problems.append("`production` missing")
    books = system.get("books")
    if not isinstance(books, list) or not books:
        problems.append("`books` must be a non-empty list")
        books = []
    names = []
    for book in books:
        if not isinstance(book, dict):
            problems.append(f"book entry {book!r} is not a mapping")
            continue
        name, venue = book.get("name"), book.get("venue")
        names.append(name)
        for key in ("name", "market", "primary_dataset"):
            if not isinstance(book.get(key), str) or not book.get(key):
                problems.append(f"book {name!r}: `{key}` must be a non-empty string")
        if "mandate" in book and book["mandate"] not in MANDATES:
            problems.append(f"book {name!r}: `mandate` override must be one of {'|'.join(MANDATES)}")
        if not isinstance(venue, dict):
            problems.append(f"book {name!r}: `venue` missing")
            continue
        if not isinstance(venue.get("allow_shorts"), bool):
            problems.append(f"book {name!r}: venue.allow_shorts must be true|false")
        if not _is_number(venue.get("leverage")):
            problems.append(f"book {name!r}: venue.leverage must be a finite number")
        if not isinstance(venue.get("funding_modelled"), bool):
            problems.append(f"book {name!r}: venue.funding_modelled must be true|false")
    repeated = sorted({str(n) for n in names if names.count(n) > 1})
    if repeated:
        problems.append(f"duplicate book name(s) {repeated} — which declaration counts is a guess")
    liveness = system.get("liveness")
    if not isinstance(liveness, dict) or not liveness:
        problems.append("`liveness` must map threshold names to numbers")
    else:
        for key, value in liveness.items():
            if not isinstance(key, str) or not key.startswith(("min_", "max_")):
                problems.append(
                    f"liveness key {key!r} must start with min_ or max_ — "
                    "a threshold with no direction is never compared"
                )
            if not _is_number(value):
                problems.append(f"liveness.{key} is {value!r}, not a finite number")
    modes = system.get("modes")
    if not isinstance(modes, dict) or not all(isinstance(v, bool) for v in modes.values()):
        problems.append("`modes` must map mode names to true|false")
    placeholders = _placeholders(system)
    if placeholders:
        problems.append("unfilled template placeholder at " + ", ".join(placeholders))
    if problems:
        report.add(
            REFUSE,
            "system",
            "system.yaml incomplete — cannot determine: " + "; ".join(problems),
        )
        return None
    report.add(
        PASS,
        "system",
        f"{system['system']} ({system['mandate']}), {len(books)} book(s)",
    )
    return system


def check_prereg(report: Report, path: Path, wanted: str | None) -> tuple[dict | None, list[dict]]:
    """`(the selected block, every OTHER block in the file)`."""
    if not path.is_file():
        report.add(REFUSE, "prereg", f"no pre-registration file at {path}")
        return None, []
    report.read.append(str(path))
    try:
        blocks = prereg_blocks(path.read_text(encoding="utf-8"))
    except (YamlUnsupported, OSError, UnicodeDecodeError) as exc:
        report.add(BLIND, "prereg", f"pre-registration block unreadable: {exc}")
        return None, []
    if not blocks:
        report.add(
            REFUSE,
            "prereg",
            f"{path.name} carries no top-level fenced yaml block opening with `prereg:` — a "
            "criterion in prose cannot be checked, and an unchecked criterion is not a pre-registration",
        )
        return None, []
    prereg = select_prereg(blocks, wanted)
    if prereg is None:
        ids = [str(b.get("id")) for b in blocks]
        report.add(
            BLIND,
            "prereg",
            f"{len(blocks)} prereg blocks {ids}; --id selects exactly one",
        )
        return None, []
    report.add(PASS, "prereg", f"block id={prereg.get('id')!r}")
    others = [b for b in blocks if b is not prereg]
    return prereg, others


def secondary_datasets(prereg: dict) -> list[str] | None:
    """The declared secondaries, or None when the key is not a list of names."""
    value = prereg.get("secondary_datasets")
    if value is None:
        return []
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    return None


def check_prereg_content(report: Report, prereg: dict, n_configs: int) -> None:
    criterion = prereg.get("criterion") if isinstance(prereg.get("criterion"), dict) else {}
    legs = criterion.get("legs")
    bad = []
    if not isinstance(legs, list) or not legs:
        bad.append("criterion.legs must be a non-empty list")
    else:
        for n, leg in enumerate(legs):
            if not isinstance(leg, dict) or not leg.get("metric"):
                bad.append(f"leg {n}: needs a `metric`")
            elif not _is_number(leg.get("threshold")):
                bad.append(f"leg {n}: threshold is {leg.get('threshold')!r}, not a finite numeric `threshold`")
    if not criterion.get("pass_rule"):
        bad.append("criterion.pass_rule missing")
    report.add(
        REFUSE if bad else PASS,
        "criterion",
        "; ".join(bad) or f"{len(legs)} numeric leg(s) + pass rule",
    )

    test = prereg.get("claim_test") if isinstance(prereg.get("claim_test"), dict) else {}
    bad = []
    if not test.get("method"):
        bad.append("no `claim_test.method` — the CLAIM has no significance test of its own")
    else:
        probability = test.get("min_probability")
        if not _is_number(probability) or not 0 < probability <= 1:
            bad.append(f"claim_test.min_probability is {probability!r}, not a number in (0, 1]")
        if "bootstrap" in str(test.get("method")).lower():
            for key in ("draws", "mean_block", "seed"):
                if not _is_number(test.get(key)):
                    bad.append(f"claim_test.{key} must be a number")
    report.add(REFUSE if bad else PASS, "claim-test", "; ".join(bad) or str(test.get("method")))

    episode = prereg.get("ex_largest_episode")
    relabel = episode.get("relabel_to") if isinstance(episode, dict) else None
    if not isinstance(relabel, str) or relabel not in VERDICTS:
        report.add(
            REFUSE,
            "ex-largest-episode",
            f"`ex_largest_episode.relabel_to` is {relabel!r}, not one of " + "|".join(VERDICTS) + " — what "
            "a criterion that only passes WITH its largest drawdown episode becomes is undeclared",
        )
    else:
        report.add(PASS, "ex-largest-episode", f"relabel_to={relabel}")

    trials = prereg.get("n_trials")
    if not _is_count(trials) or trials < 1:
        report.add(
            REFUSE,
            "trial-count",
            f"`n_trials` is {trials!r} — no integer `n_trials` >= 1",
        )
    elif _is_count(trials) and trials < n_configs:
        report.add(
            REFUSE,
            "trial-count",
            f"n_trials={trials} < the {n_configs} config(s) about to run",
        )
    else:
        report.add(PASS, "trial-count", f"n_trials={trials} (completeness is the auditor's)")

    unfilled = _placeholders(prereg, "prereg")
    report.add(
        REFUSE if unfilled else PASS,
        "placeholders",
        ("unfilled template placeholder at " + ", ".join(unfilled)) if unfilled else "none left in the block",
    )

    if secondary_datasets(prereg) is None:
        report.add(
            REFUSE,
            "secondary-datasets",
            f"`secondary_datasets` is {prereg.get('secondary_datasets')!r}, "
            "not a list of dataset names — a string would be matched by substring, so it is refused",
        )
    else:
        report.add(PASS, "secondary-datasets", str(secondary_datasets(prereg)))


def check_liveness(report: Report, system: dict, book: dict | None, prereg: dict) -> None:
    """`book=None` is a replication: it holds no venue, so a `_for_long_only` floor —
    which binds a venue that cannot short — binds nothing; every other floor does. Not
    `venue_of`'s safe default on purpose: a long-short paper (TSMOM) shorts by construction,
    and the lab floor is about what a VENUE can hold (system.yaml; TOM-2231, D1)."""
    declared = prereg.get("liveness") if isinstance(prereg.get("liveness"), dict) else {}
    shorts = venue_of(book)["allow_shorts"] if book is not None else True
    bad = []
    for key, floor in system["liveness"].items():
        if str(key).endswith("_for_long_only") and shorts:
            continue
        value = declared.get(key)
        if value is None:
            bad.append(f"liveness.{key} not pre-registered")
        elif not _is_number(value):
            bad.append(f"liveness.{key} is {value!r}, not a finite number")
        elif not _is_number(floor):
            bad.append(f"the lab's liveness.{key} is {floor!r}, not a finite number")
        if not (_is_number(value) and _is_number(floor)):
            continue  # reported above; what is not a number is never COMPARED
        if str(key).startswith("min_") and value < floor:
            bad.append(f"liveness.{key}={value} is looser than the lab's {floor}")
        if str(key).startswith("max_") and value > floor:
            bad.append(f"liveness.{key}={value} is looser than the lab's {floor}")
    report.add(
        REFUSE if bad else PASS,
        "liveness",
        "; ".join(bad)
        or "every lab threshold pre-registered"
        + ("" if book is not None else " (a replication holds no venue: `_for_long_only` floors do not bind it)"),
    )


def check_book(report: Report, system: dict, prereg: dict) -> dict | None:
    wanted = prereg.get("book")
    matches = [b for b in system["books"] if isinstance(b, dict) and b.get("name") == wanted]
    book = matches[0] if matches else None
    if len(matches) != 1:
        names = [b.get("name") for b in system["books"] if isinstance(b, dict)]
        report.add(
            REFUSE,
            "book",
            f"prereg book {wanted!r} is not declared in system.yaml exactly once {names}",
        )
        return None
    if book is None:
        return None
    primary = prereg.get("primary_dataset")
    if primary != book.get("primary_dataset"):
        report.add(
            REFUSE,
            "primary-dataset",
            f"pre-registered primary {primary!r} is not the book's primary_dataset "
            f"{book.get('primary_dataset')!r} ({book.get('name')} trades {book.get('market')}) — a longer "
            "or cleaner dataset can only be secondary",
        )
    else:
        report.add(
            PASS,
            "primary-dataset",
            f"{primary} == books[{book.get('name')}].primary_dataset",
        )
    return book


# ── a paper replication: anchored to the paper, not to a book ──────────────

#: `prereg.kind` that anchors a block to the paper it replicates (TOM-2231, Tom D1:
#: no paper book goes into system.yaml, which keeps describing live books only).
#: Any other `kind`, or none, is held to a book exactly as before.
REPLICATION = "replication"
#: The paper's out-of-sample stretch the replication must also run on (TOM-1412 point 3).
POST_SAMPLE_MIN_YEARS = 3
_DOI_RE = re.compile(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?10\.\d{4,9}/\S+$", re.IGNORECASE)
_URL_RE = re.compile(r"^https?://[^\s/]+\.[^\s/]+\S*$", re.IGNORECASE)
_PERIOD_RE = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$")


def _period(value, end: bool) -> date | None:
    """`YYYY-MM` or `YYYY-MM-DD` as the first day it names — or, for an `end`, the last."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):  # PyYAML reads an unquoted YYYY-MM-DD as a date
        return value
    match = _PERIOD_RE.match(value.strip()) if isinstance(value, str) else None
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        return None
    try:
        if match.group(3):
            return date(year, month, int(match.group(3)))
        if not end:
            return date(year, month, 1)
        return date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
    except ValueError:
        return None


def _window(paper: dict, key: str, problems: list[str]) -> tuple[date, date] | None:
    value = paper.get(key)
    if not isinstance(value, dict):
        problems.append(f"`paper.{key}` must be a mapping {{start, end}}, each YYYY-MM or YYYY-MM-DD")
        return None
    start, end = _period(value.get("start"), end=False), _period(value.get("end"), end=True)
    if start is None or end is None:
        problems.append(
            f"`paper.{key}` start/end are {value.get('start')!r}/{value.get('end')!r}, not YYYY-MM or YYYY-MM-DD"
        )
        return None
    if end < start:
        problems.append(f"`paper.{key}` ends ({end}) before it starts ({start})")
        return None
    return start, end


def _years_on(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year + years)
    except ValueError:  # 29 February
        return day.replace(year=day.year + years, day=28)


def paper_problems(paper: dict) -> list[str]:
    """What is missing or wrong in a replication's `paper:` declaration."""
    problems: list[str] = []
    ref = paper.get("ref")
    if not isinstance(ref, str) or not (_DOI_RE.match(ref.strip()) or _URL_RE.match(ref.strip())):
        problems.append(f"`paper.ref` is {ref!r}, not a DOI (10.xxxx/...) or an http(s) URL")
    sample = _window(paper, "sample", problems)
    reported = paper.get("reported")
    if not isinstance(reported, list) or not reported:
        problems.append("`paper.reported` must list the paper's numbers being replicated, each {metric, value}")
    else:
        for n, row in enumerate(reported):
            if not isinstance(row, dict) or not isinstance(row.get("metric"), str) or not row["metric"].strip():
                problems.append(f"paper.reported[{n}] needs a `metric`")
            elif not _is_number(row.get("value")):
                problems.append(f"paper.reported[{n}].value is {row.get('value')!r}, not a finite number")
    series = paper.get("reference_series")
    if series is not None and not (isinstance(series, str) and series.strip()):
        problems.append(f"`paper.reference_series` is {series!r} — leave it out, or name the series")
    post = _window(paper, "post_sample", problems)
    if sample is not None and post is not None:
        if post[0] != sample[1] + timedelta(days=1):
            problems.append(
                f"`paper.post_sample` starts {post[0]}, not right after the paper's sample (ends {sample[1]}) "
                "— an overlap re-tests the sample, a gap chooses which years count"
            )
        if post[1] < _years_on(post[0], POST_SAMPLE_MIN_YEARS) - timedelta(days=1):
            problems.append(
                f"`paper.post_sample` {post[0]}..{post[1]} is shorter than {POST_SAMPLE_MIN_YEARS} years "
                "— a replication is also tested after the paper's sample"
            )
    return problems


def check_replication(report: Report, prereg: dict) -> bool:
    """A `kind: replication` block: is it anchored to its paper? True when it is.

    It names no book, so nothing about a venue (market, shorts, leverage, funding)
    has anything to be compared against and none is checked; the data pin, the
    dataset list, execution timing and the lab's liveness floor still are.
    Whether the pinned data COVERS both windows is the auditor's.
    """
    paper = prereg.get("paper")
    if prereg.get("book") is not None:
        report.add(
            REFUSE,
            "book",
            f"`kind: replication` also names book {prereg.get('book')!r} — a replication is anchored to its "
            "`paper:`; drop `book`, or drop `kind: replication` to test a system.yaml book",
        )
        return False
    if not isinstance(paper, dict):
        report.add(
            REFUSE,
            "book",
            f"`kind: replication` and `paper` is {paper!r}, not a mapping — what it replicates cannot be determined",
        )
        return False
    report.add(
        PASS,
        "book",
        f"kind: replication, anchored to paper {paper.get('ref')!r} — no system.yaml book, so no venue "
        "check (market, shorts, leverage, funding) applies",
    )
    problems = paper_problems(paper)
    report.add(
        REFUSE if problems else PASS,
        "paper",
        "; ".join(problems)
        or (
            f"sample {paper['sample']['start']}..{paper['sample']['end']}, {len(paper['reported'])} reported "
            f"number(s), post-sample {paper['post_sample']['start']}..{paper['post_sample']['end']}"
            + (f", reference series {paper['reference_series']}" if paper.get("reference_series") else "")
        ),
    )
    primary = prereg.get("primary_dataset")
    named = isinstance(primary, str) and bool(primary.strip())
    if named:
        report.add(PASS, "primary-dataset", f"{primary} (a replication's own primary; no book to match)")
    else:
        report.add(REFUSE, "primary-dataset", f"pre-registered primary is {primary!r}, not a dataset name")
    return named and not problems


# ── the plan quantbox reports for a config ─────────────────────────────────


@contextlib.contextmanager
def _asked_from(cwd: Path):
    """Run quantbox's own code as `quantbox <verb>` run from *cwd* would: a relative
    `artifacts.root` or `dataset_lock` resolves from there. Anything quantbox prints
    goes to stderr, so the report alone is on stdout (as `config explain --json` keeps it)."""
    with contextlib.chdir(cwd), contextlib.redirect_stdout(sys.stderr):
        yield


def explain_config(path: Path, cwd: Path) -> dict:
    """quantbox's plan for *path* (:func:`quantbox.explain.explain_path`, what `quantbox
    config explain <path> --json` prints), asked from where the line's runs are launched.

    Raises `Unavailable` when quantbox raises instead of answering: the caller
    reports BLIND. A plan with `ok: false` is RETURNED — quantbox saying the run
    cannot start is an answer, not blindness.
    """
    try:
        with _asked_from(cwd):
            return _explain.explain_path(str(path.resolve()))
    except Exception as exc:  # noqa: BLE001 — a crash is "could not ask", never a plan
        raise Unavailable(f"`quantbox config explain {path.resolve()}` raised {type(exc).__name__}: {exc}") from None


def _inline_rows(line: Path) -> tuple[list[dict] | None, str | None]:
    """The rows of `<line>/data_inputs.yaml`, or `(None, why not)`."""
    inputs = line / "data_inputs.yaml"
    if not inputs.is_file():
        return None, f"no {inputs}"
    try:
        declared = load_yaml(inputs)
    except (YamlUnsupported, OSError, UnicodeDecodeError) as exc:
        return None, f"{inputs} unreadable: {exc}"
    rows = declared.get("inputs") if isinstance(declared, dict) else None
    if not isinstance(rows, list):
        return None, f"{inputs.name} has no `inputs:` list"
    return [r for r in rows if isinstance(r, dict)], None


def resolve_dataset(name: str, config: Path, cwd: Path) -> dict:
    """The build `run -c <config>` binds for a by-name read: :func:`quantbox.dataset_lock.resolve_dataset`
    at :func:`~quantbox.dataset_lock.lock_for_run` (what `quantbox dataset resolve <name> --config
    <config> --json` prints). Raises `Unavailable` with quantbox's reason."""
    try:
        with _asked_from(cwd):
            return _resolve_dataset(name, lock=lock_for_run(config.resolve()))
    except Exception as exc:  # noqa: BLE001 — DatasetResolveError or a crash: either way, no answer
        raise Unavailable(
            f"`quantbox dataset resolve {name} --config {config.resolve()}` raised {type(exc).__name__}: {exc}"
        ) from None


def check_data(
    report: Report, tag: str, plan: dict, prereg: dict, line: Path, resolve=None
) -> tuple[str | None, str | None]:
    """Is what the run will read PINNED, and is it a dataset the prereg names?

    Two ways to be pinned, both read from quantbox's plan, never from config keys:
    a dataset read by name at the build its `datasets.lock` pins (`source: lock`,
    `tier: lock`), or a file read by path whose content hash (`dataset.sha256`)
    is declared in the line's `data_inputs.yaml` — that row's `dataset` is then
    the dataset's name, and its optional `market` the market. Returns
    `(dataset name, raw market)`, either None when it cannot be told.
    """
    check = f"data[{tag}]"
    dataset = plan.get("dataset") if isinstance(plan.get("dataset"), dict) else {}
    source, sha = dataset.get("source"), dataset.get("sha256")
    name, market = None, None
    if source == "lock":
        name, market = dataset.get("name"), dataset.get("market")
        if dataset.get("tier") == "lock" and isinstance(sha, str) and sha:
            report.add(PASS, check, f"{name} read by name at the build datasets.lock pins (sha256 {sha[:12]}…)")
        elif isinstance(sha, str) and sha:
            # resolved (its bytes were hashed) and still no pin: quantbox SAYS it is unpinned
            report.add(
                REFUSE,
                check,
                f"{name!r} is read by name but NOT pinned in a datasets.lock (tier {dataset.get('tier')!r}) — "
                "pin it, so the run reads a build anyone can fetch again",
            )
        else:
            # the plan never resolved the name (a data plugin explain cannot see into):
            # ask quantbox the narrower question — which build does a run bind for it?
            name, market = _by_name(report, check, name, dataset, resolve)
    elif source == "inline":
        rows, problem = _inline_rows(line)
        hits = [r for r in rows or [] if isinstance(sha, str) and sha and r.get("sha256") == sha]
        if hits:
            name = hits[0].get("dataset") if isinstance(hits[0].get("dataset"), str) else None
            market = hits[0].get("market")
            report.add(
                PASS,
                check,
                f"read by path; its sha256 {sha[:12]}… is declared in data_inputs.yaml ({hits[0].get('path')})",
            )
        else:
            report.add(
                REFUSE,
                check,
                f"read by path ({dataset.get('name')!r}, sha256 {sha!r}) and that content hash is declared "
                f"nowhere — {problem or 'no data_inputs.yaml row carries it'}. Read it by name through "
                "datasets.lock, or declare the file in data_inputs.yaml and run research_freeze.py",
            )
    else:
        report.add(BLIND, check, f"quantbox's plan gives dataset.source {source!r}, neither lock nor inline")
        return None, None
    if name is not None:
        secondary = secondary_datasets(prereg) or []
        if name != prereg.get("primary_dataset") and name not in secondary:
            report.add(
                REFUSE,
                f"dataset[{tag}]",
                f"{name!r} is neither the pre-registered primary nor in secondary_datasets {secondary}",
            )
    return name, market if isinstance(market, str) and market else None


def _by_name(report: Report, check: str, name, dataset: dict, resolve) -> tuple[str | None, str | None]:
    """A by-name read the plan left unresolved, settled by `quantbox dataset resolve`."""
    why = f"tier {dataset.get('tier')!r}, sha256 null: {dataset.get('warning') or 'no reason given'}"
    if resolve is None or not isinstance(name, str):
        report.add(BLIND, check, f"{name!r} is read by name, but quantbox's plan does not establish its build ({why})")
        return name, None
    try:
        got = resolve(name)
    except Unavailable as exc:
        report.add(
            BLIND,
            check,
            f"{name!r} is read by name; quantbox's plan does not establish its build ({why}) and "
            f"`dataset resolve` could not either: {exc}",
        )
        return name, None
    pin, matches = got.get("sha256"), got.get("matches")
    if isinstance(pin, str) and pin and matches is True:
        report.add(
            PASS,
            check,
            f"{name} read by name; `quantbox dataset resolve` binds the build {got.get('lock')} pins "
            f"(sha256 {pin[:12]}…{', restored from history' if got.get('restored') else ''})",
        )
    elif isinstance(pin, str) and pin:
        report.add(
            REFUSE,
            check,
            f"{name!r}: the bytes on disk are not the pinned build — {got.get('error') or 'matches is not true'}",
        )
    else:
        report.add(
            REFUSE,
            check,
            f"{name!r} is read by name but NOT pinned (`quantbox dataset resolve`: lock {got.get('lock')!r} has no "
            "pin for it) — pin it, so the run reads a build anyone can fetch again",
        )
    market = got.get("market")
    return name, market if isinstance(market, str) else None


def check_config(report: Report, path: Path, plan: dict, book: dict, prereg: dict, line: Path, resolve=None) -> None:
    """Judge quantbox's plan for one config against the book and the pre-registration."""
    tag = path.name
    venue_can = venue_of(book)
    book_market = canonical_market(book.get("market"))
    dataset, raw_market = check_data(report, tag, plan, prereg, line, resolve)
    market = canonical_market(raw_market) if raw_market else None
    is_primary = dataset is not None and dataset == prereg.get("primary_dataset")
    if dataset is None:
        pass  # the data row already refused or went blind; a market of nothing is not a check
    elif market is None:
        report.add(
            BLIND,
            f"market[{tag}]",
            f"quantbox's plan gives no market for {dataset} (dataset.market null: its manifest declares none, "
            "or quantbox did not resolve it); read by path, its data_inputs.yaml row carries no `market`",
        )
    elif is_primary and market != book_market:
        report.add(
            REFUSE,
            f"market[{tag}]",
            f"declared PRIMARY, but {dataset} is a {market} dataset and {book.get('name')} trades {book_market}",
        )
    else:
        report.add(PASS, f"market[{tag}]", f"{dataset}: {market} ({'primary' if is_primary else 'secondary'})")

    # shorts: what the venue cannot hold, the runner must not allow — as quantbox will APPLY it
    venue = plan.get("venue") if isinstance(plan.get("venue"), dict) else None
    allows = venue.get("allow_shorts") if venue is not None else None
    if not isinstance(allows, bool):
        report.add(BLIND, f"shorts[{tag}]", f"quantbox's plan gives venue.allow_shorts {allows!r}, not true|false")
    elif venue_can["allow_shorts"] and market is None:
        report.add(BLIND, f"shorts[{tag}]", "the book may short on its OWN market, and this config's market is unknown")
    elif venue_can["allow_shorts"] and market == book_market:
        report.add(PASS, f"shorts[{tag}]", "venue holds shorts and the config trades the book's market")
    else:
        why = (
            f"this config trades {market}, not the book's {book_market}"
            if venue_can["allow_shorts"]
            else f"{book.get('name')} venue.allow_shorts is not true"
        )
        report.add(
            REFUSE if allows else PASS,
            f"shorts[{tag}]",
            f"{why}, but the runner will allow shorts (venue.allow_shorts: true) — set it false"
            if allows
            else f"{why}; the runner applies venue.allow_shorts: false",
        )

    cap = venue_can["leverage"]
    lev = venue.get("max_leverage") if venue is not None else None
    if not _is_number(lev):
        report.add(BLIND, f"leverage[{tag}]", f"quantbox's plan gives venue.max_leverage {lev!r}, not a finite number")
    elif lev > cap:
        report.add(REFUSE, f"leverage[{tag}]", f"the runner caps gross leverage at {lev} > venue leverage {cap}")
    else:
        report.add(PASS, f"leverage[{tag}]", f"runner cap {lev} <= {cap}")

    check_execution(report, tag, plan, prereg)
    if market is not None and market == book_market:
        check_funding(report, tag, plan, book, prereg)


def funding_input_problem(source: Path | None) -> str | None:
    """Why `source` cannot be what the funding term is charged from — or None."""
    if source is None:
        return "no funding input"
    try:
        size = source.stat().st_size
        with source.open("rb") as handle:
            head = handle.read(4)
            handle.seek(max(size - 4, 0))
            tail = handle.read(4)
    except OSError as exc:
        return f"{source}: {exc.strerror or exc}"
    if size < 12:
        return f"{source} is {size} byte(s) — a stub, not funding rates"
    if source.suffix == ".parquet" and not (head == b"PAR1" and tail == b"PAR1"):
        return f"{source} is not a parquet file (no PAR1 magic)"
    return None


def check_funding(report: Report, tag: str, plan: dict, book: dict, prereg: dict) -> None:
    """A book whose venue charges funding gets results that INCLUDE it — on its own market.

    Whether funding is charged is quantbox's answer (`funding.modelled`: the engine
    charges funding AND a funding file exists). This gate adds one thing quantbox's
    plan does not: that the file is not a stub.
    """
    venue = book.get("venue") if isinstance(book.get("venue"), dict) else {}
    if venue.get("funding_modelled") is not True:
        return
    check = f"funding[{tag}]"
    funding = plan.get("funding") if isinstance(plan.get("funding"), dict) else {}
    engine = plan.get("engine") if isinstance(plan.get("engine"), dict) else {}
    raw = funding.get("source_path")
    source = Path(raw) if isinstance(raw, str) and raw else None
    missing = []
    if funding.get("modelled") is not True:
        missing.append(f"quantbox will not charge funding (engine {engine.get('name')!r}, funding file {raw!r})")
    else:
        unreadable = funding_input_problem(source)
        if unreadable:
            missing.append(f"funding input unreadable: {unreadable}")
    if not missing:
        report.add(PASS, check, f"engine {engine.get('name')} charges funding from {source} ({_size(source)} bytes)")
        return
    declared = prereg.get("funding") if isinstance(prereg.get("funding"), dict) else {}
    reason = declared.get("reason")
    if declared.get("mode") == "excluded" and isinstance(reason, str) and reason.strip():
        report.add(PASS, check, f"funding EXCLUDED by pre-registration: {reason.strip()} — the finding must repeat it")
    else:
        report.add(
            REFUSE,
            check,
            f"{book.get('name')} declares funding_modelled: true, but "
            + "; ".join(missing)
            + " — model it, or pre-register `funding: {mode: excluded, reason: ...}`",
        )


def _size(path: Path | None) -> str:
    try:
        return str(path.stat().st_size) if path is not None else "?"
    except OSError:
        return "?"


def check_execution(report: Report, tag: str, plan: dict, prereg: dict) -> None:
    """Timing as the runner will APPLY it, or a stated timing with a +1-bar promise."""
    check = f"execution[{tag}]"
    execution = plan.get("execution") if isinstance(plan.get("execution"), dict) else None
    stated = prereg.get("execution") if isinstance(prereg.get("execution"), dict) else {}
    lag = execution.get("lag_bars") if execution is not None else None
    if execution is None:
        if (
            isinstance(stated.get("timing"), str)
            and stated["timing"].strip()
            and stated.get("plus_one_bar_result") is True
        ):
            report.add(
                PASS,
                check,
                "quantbox reports no execution timing for this pipeline, but the "
                "pre-registration states the timing and promises a +1-bar result",
            )
        else:
            report.add(
                REFUSE,
                check,
                "quantbox reports no execution timing for this pipeline — timing is UNKNOWN; "
                "pre-register `execution.timing` and `execution.plus_one_bar_result: true`",
            )
    elif not _is_count(lag) or lag < 0:
        report.add(BLIND, check, f"quantbox's plan gives execution.lag_bars {lag!r}, not an integer >= 0")
    elif (lag == 0 or execution.get("same_bar") is True) and not (
        isinstance(stated.get("lag0_reason"), str) and stated["lag0_reason"].strip()
    ):
        report.add(
            REFUSE,
            check,
            f"lag_bars: {lag} is same-bar execution ({execution.get('description')}) and the "
            "pre-registration gives no `execution.lag0_reason`",
        )
    else:
        report.add(PASS, check, f"lag_bars={lag} — {execution.get('description')}")


def check_run_manifests(report: Report, roots: list[Path]) -> None:
    """Every `run_manifest.json` that claims to be quantbox's must be a major this quantbox reads.

    A manifest that claims no quantbox schema (a pre-run@1 run, another tool's) is
    only run EVIDENCE, dated by `check_timing`; one claiming `quantbox/run@N` for
    an N not in `SUPPORTED` (the major :data:`quantbox.run_manifest.SCHEMA_ID` names)
    is BLIND — its fields may not mean what this quantbox would read them as.
    """
    seen: set[Path] = set()
    known, unknown = 0, []
    for root in roots:
        if not root.is_dir():
            continue
        for folder, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in _NEVER_WALKED]
            if "run_manifest.json" not in files:
                continue
            path = (Path(folder) / "run_manifest.json").resolve()
            if path in seen:
                continue
            seen.add(path)
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValueError):
                continue  # not JSON: claims no schema; still dated as evidence
            if not is_quantbox_document(doc, "run"):
                continue
            problem = schema_problem(doc, "run")
            if problem:
                unknown.append(f"{path}: {problem}")
            else:
                known += 1
    if unknown:
        report.add(
            BLIND,
            "run-manifest",
            f"{len(unknown)} run manifest(s) this quantbox cannot read: " + "; ".join(unknown[:3]),
        )
    else:
        reads = ", ".join(f"quantbox/run@{n}" for n in sorted(SUPPORTED["run"]))
        report.add(PASS, "run-manifest", f"{known} quantbox run manifest(s), every one {reads}")


def frozen_since(top: Path, rel: str, block: dict, commits: list[list[str]]) -> tuple[float | None, str | None]:
    """The oldest commit, walking back from HEAD, since which `block`'s frozen part is unchanged."""
    current, frozen_at, frozen_sha = fingerprint(block), None, None
    for sha, stamp in commits:  # newest first
        try:
            blocks = prereg_blocks(_git(top, "show", f"{sha}:{rel}"))
        except (OSError, YamlUnsupported, subprocess.TimeoutExpired):
            break
        then = select_prereg(blocks, str(block.get("id")))
        if then is None or fingerprint(then) != current:
            break
        frozen_at, frozen_sha = float(stamp), sha
    return frozen_at, frozen_sha


def _declared_inputs(line: Path) -> set[Path]:
    """Files the line declares as DATA in `data_inputs.yaml` — a dataset's manifest is not a run."""
    try:
        doc = load_yaml(line / "data_inputs.yaml")
    except (YamlUnsupported, OSError, UnicodeDecodeError):
        return set()
    rows = doc.get("inputs") if isinstance(doc, dict) else None
    rows = rows if isinstance(rows, list) else []
    return {(line / r["path"]).resolve() for r in rows or [] if isinstance(r, dict) and isinstance(r.get("path"), str)}


def check_timing(
    report: Report,
    prereg_path: Path,
    prereg: dict,
    line: Path,
    lab: Path,
    config_roots: list[Path],
    extra_roots: list[Path],
    others: list[dict],
    now: float | None = None,
) -> None:
    """The frozen part of the block was committed before ANY run evidence that is not
    DECLARED to be another block's. Attribution is declared (`runs:`), never inferred
    from what a path happens to be called."""
    check = "prereg-committed"
    # 1 — whose runs land where: declared, checked for contradictions
    mine, problems = declared_paths(prereg, "runs", line)
    excluded, more = declared_paths(prereg, "not_runs", line)
    problems += more
    if mine:
        stray = [str(r) for r in config_roots if not _under(r, mine)]
        if stray:
            problems.append(f"artifacts.root {stray} is outside this block's declared `runs` {[str(p) for p in mine]}")
    else:
        mine = list(config_roots)
    theirs: dict[str, list[Path]] = {}
    for other in others:
        paths, _ = declared_paths(other, "runs", line)
        theirs[str(other.get("id"))] = paths
        for path in paths:
            clash = [str(m) for m in mine if _within(path, m) or _within(m, path)]
            if clash:
                problems.append(
                    f"`runs` {path} of block {other.get('id')!r} overlaps this block's {clash} — "
                    "one run directory cannot belong to two hypotheses"
                )
    if problems:
        report.add(REFUSE, check, "; ".join(problems))
        return
    if not mine and not extra_roots:
        report.add(
            BLIND,
            check,
            "no config declares `artifacts.root`, the block declares no `runs`, and no "
            "--runs-dir was given — where this hypothesis's runs land is unknown",
        )
        return
    reasons = {
        str(path): str(item.get("reason")).strip()
        for path, item in zip(excluded, prereg.get("not_runs") or [], strict=False)
    }

    # 2 — when the frozen part last changed
    try:
        top = Path(_git(prereg_path.parent, "rev-parse", "--show-toplevel").strip())
        rel = prereg_path.resolve().relative_to(top.resolve()).as_posix()
        if _git(top, "status", "--porcelain", "--", rel).strip():
            report.add(
                REFUSE,
                check,
                f"{rel} has uncommitted changes — a criterion that is not committed is not frozen",
            )
            return
        log = _git(top, "log", "--format=%H %ct", "--", rel).split("\n")
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        report.add(BLIND, check, f"cannot read git history of {prereg_path}: {exc}")
        return
    commits = [ln.split() for ln in log if ln.strip()]
    if not commits:
        report.add(REFUSE, check, f"{rel} was never committed")
        return
    frozen_at, frozen_sha = frozen_since(top, rel, prereg, commits)
    if frozen_at is None:
        report.add(
            BLIND,
            check,
            "HEAD's pre-registration block does not match the file on disk",
        )
        return
    # an attribution is only as old as the declaration that makes it: a block added (or
    # given a `runs:`) AFTER a run cannot disown that run on someone's behalf
    claimed_since = {str(o.get("id")): frozen_since(top, rel, o, commits)[0] for o in others}

    # 3 — look. Declared roots are walked whatever they are called; the line is walked whole.
    found: dict = {}
    undatable: list[str] = []
    unwalked: set[str] = set()
    walked: list[str] = []
    for root in mine + extra_roots:
        run_evidence(root, True, found, undatable, unwalked)
        walked.append(f"{root} (declared: every file)")
    run_evidence(line, False, found, undatable, unwalked)
    walked.append(f"{line} (manifests + run-stamped dirs)")
    for root in mine:
        parent = root.parent
        if _within(parent, line):
            continue
        if _within(parent, lab):
            run_evidence(parent, False, found, undatable, unwalked)
            walked.append(f"{parent} (parent of a declared root)")
        else:
            walked.append(f"NOT {parent}: outside the lab, only the declared root itself was walked")
    if undatable:
        report.add(
            BLIND,
            check,
            f"run evidence that cannot be dated: {'; '.join(undatable[:3])}",
        )
        return

    # 4 — attribute by DECLARATION, exclude by DECLARATION, and say both
    inputs = _declared_inputs(line)
    counted: list[Evidence] = []
    tally: dict[str, int] = {}
    for item in sorted(found.values(), key=lambda e: e.when):
        owner = next((name for name, paths in theirs.items() if _under(item.path, paths)), None)
        since = claimed_since.get(owner) if owner is not None else None
        if owner is not None and since is not None and item.when > since:
            tally[f"attributed to block {owner!r} by its `runs`"] = (
                tally.get(f"attributed to block {owner!r} by its `runs`", 0) + 1
            )
            continue
        if owner is not None:
            note = f"under block {owner!r}'s `runs` but NOT newer than that declaration, so counted here"
            tally[note] = tally.get(note, 0) + 1
        elif item.path.resolve() in inputs:
            tally["excluded as a declared data input"] = tally.get("excluded as a declared data input", 0) + 1
            continue
        elif _under(item.path, excluded):
            key = next(str(p) for p in excluded if _within(item.path, p))
            note = f"excluded by not_runs {key} ({reasons.get(key, '')})"
            tally[note] = tally.get(note, 0) + 1
            continue
        counted.append(item)
    notes = [f"{n} {what}" for what, n in tally.items()]
    if unwalked:
        notes.append(f"never walked: {sorted(unwalked)}")
    looked = f"walked: {'; '.join(walked)}; {len(found)} evidence item(s) found" + (
        "; " + "; ".join(notes) if notes else ""
    )
    frozen = f"frozen at {frozen_sha[:10]} ({_iso(frozen_at)})"
    if not counted:
        report.add(
            PASS,
            check,
            f"{frozen}; no run evidence counts against this block — {looked}",
        )
        return
    first, last = counted[0], counted[-1]
    now = time.time() if now is None else now
    if last.when > now + 86400:
        report.add(
            REFUSE,
            check,
            f"run evidence dated in the FUTURE: {last.path} ({_iso(last.when)} from "
            f"{last.source}) — a run time that cannot be true cannot show the criterion came first",
        )
    elif first.when <= frozen_at:
        report.add(
            REFUSE,
            check,
            f"first run evidence {first.path} ({_iso(first.when)} from {first.source}) "
            f"predates the criterion's last change {frozen_sha[:10]} ({_iso(frozen_at)}) — that "
            "is a post-registration. If it is another hypothesis's run, THAT block declares it under "
            f"`runs:`; if it is not a run, list it under `not_runs:` with a reason — {looked}",
        )
    else:
        report.add(
            PASS,
            check,
            f"{frozen}, before the earliest of {len(counted)} counted item(s) "
            f"({_iso(first.when)} from {first.source}: {first.path}) — {looked}",
        )


def plan_for(report: Report, path: Path, line: Path, explain) -> dict | None:
    """quantbox's plan for one config, version-checked — or a BLIND/REFUSE row and None."""
    report.read.append(str(path))
    tag = path.name
    if not path.is_file():
        report.add(BLIND, f"config[{tag}]", f"no such config: {path}")
        return None
    try:
        plan = explain(path, line)
    except Unavailable as exc:
        report.add(BLIND, f"config[{tag}]", f"could not ask quantbox what this config runs: {exc}")
        return None
    problem = schema_problem(plan, "explain")
    if problem:
        report.add(BLIND, f"config[{tag}]", f"quantbox config explain: {problem}")
        return None
    if plan.get("ok") is not True:
        errors = plan.get("errors") if isinstance(plan.get("errors"), list) else []
        report.add(
            REFUSE,
            f"config[{tag}]",
            "quantbox cannot start this run: " + ("; ".join(map(str, errors)) or "ok: false, no reason given"),
        )
        return None
    return plan


def run(
    lab: Path,
    line: Path,
    configs: list[Path],
    prereg_path: Path | None = None,
    wanted: str | None = None,
    runs_dirs: list[Path] | None = None,
    report: Report | None = None,
    explain=None,
    resolve=resolve_dataset,
) -> Report:
    """`explain(config, cwd) -> explain@N dict` and `resolve(name, config, cwd) -> dict` are
    quantbox's answers; tests pass their own. `explain=None` is :func:`explain_config`,
    looked up when the run starts."""
    explain = explain_config if explain is None else explain
    report = Report() if report is None else report
    system = check_system(report, lab)
    prereg_path = prereg_path or line / "README.md"
    prereg, others = check_prereg(report, prereg_path, wanted)
    plans: list[tuple[Path, dict]] = []
    for path in configs:
        plan = plan_for(report, path, line, explain)
        if plan is not None:
            plans.append((path, plan))
    config_roots = [Path(p["artifacts_root"]) for _, p in plans if isinstance(p.get("artifacts_root"), str)]

    def resolver_for(path: Path):
        return (lambda name: resolve(name, path, line)) if resolve is not None else None

    if prereg is not None:
        check_prereg_content(report, prereg, len(configs))
        check_timing(report, prereg_path, prereg, line, lab, config_roots, list(runs_dirs or []), others)
    # the version contract holds whatever else is wrong: an unreadable run is never left unsaid
    check_run_manifests(report, [line, *config_roots, *(runs_dirs or [])])
    if system is not None and prereg is not None and prereg.get("kind") == REPLICATION:
        if check_replication(report, prereg):
            check_liveness(report, system, None, prereg)
            for path, plan in plans:
                check_data(report, path.name, plan, prereg, line, resolver_for(path))
                check_execution(report, path.name, plan, prereg)
    elif system is not None and prereg is not None:
        book = check_book(report, system, prereg)
        if book is not None:
            check_liveness(report, system, book, prereg)
            for path, plan in plans:
                check_config(report, path, plan, book, prereg, line, resolver_for(path))
    return report


def render(report: Report) -> str:
    out = [f"{status:<6} {check:<28} {reason}" for status, check, reason in report.rows]
    out.append("files read: " + (", ".join(dict.fromkeys(report.read)) or "(none)"))
    refused = sum(1 for status, _, _ in report.rows if status == REFUSE)
    blind = sum(1 for status, _, _ in report.rows if status == BLIND)
    if report.exit_code == 1:
        note = f" (+{blind} could not check)" if blind else ""
        out.append(f"PREFLIGHT: REFUSE ({refused}){note} — do not run the backtest")
    elif report.exit_code == 2:
        out.append(f"PREFLIGHT: COULD NOT CHECK ({blind}) — not a pass")
    else:
        out.append(f"PREFLIGHT: PASS ({len(report.rows)} checks)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quantbox preflight", description="Refuse a backtest whose result could not mean anything."
    )
    parser.add_argument("--lab", required=True, type=Path, help="lab repo root (holds system.yaml)")
    parser.add_argument("--line", required=True, type=Path, help="research line directory")
    parser.add_argument(
        "--prereg",
        type=Path,
        help="file carrying the prereg block (default <line>/README.md)",
    )
    parser.add_argument(
        "--id",
        dest="wanted",
        help="hypothesis id, when the file carries several blocks",
    )
    parser.add_argument(
        "--runs-dir",
        action="append",
        type=Path,
        default=[],
        help="extra artifacts root",
    )
    parser.add_argument("configs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    report = Report()
    try:
        run(
            args.lab,
            args.line,
            args.configs,
            args.prereg,
            args.wanted,
            args.runs_dir,
            report=report,
        )
    except Exception as exc:  # noqa: BLE001 — a traceback is not a reason, and must never read as a pass
        where = traceback.extract_tb(exc.__traceback__)[-1]
        report.add(
            BLIND,
            "internal",
            f"unexpected {type(exc).__name__}: {exc} "
            f"({Path(where.filename).name}:{where.lineno}) — the gate did not finish looking",
        )
    print(render(report))
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
