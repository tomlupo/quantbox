---
adr: 0013
title: The pre-flight gate lives in quantbox as `quantbox preflight`, next to `quantbox gates`
status: accepted
date: 2026-10-26
supersedes:
superseded_by:
amends:
status_changes:
  - 2026-10-26: accepted with TOM-2457 — decision note vaults/projects/infra/decisions/decyzja-2026-10-26-stos-kiss-repo-hotl-fabryka.md (Tom, 2026-10-26, "propose KISS modification of hotl and ci of repos"): "logika pre-flightu (dane, pokrycie, higiena, dźwignia, moc testu) przechodzi z qute-research do quantboxa jako `quantbox preflight`, wersjonowany i pinowany razem z silnikiem"
---

# ADR-0013: The pre-flight gate lives in quantbox as `quantbox preflight`

## Context

The pre-flight is the gate a research line runs before any measured backtest. It
refuses a run whose result could not mean anything: a primary dataset the book
does not trade, shorts or leverage the venue cannot hold, a claim with no test,
a criterion written after the first run. Until now it was `scripts/quant_preflight.py`
in the qute-research plugin.

The plugin is versioned apart from the engine it judges. It reached quantbox
through subprocesses (`quantbox config explain --json`, `quantbox dataset resolve
--json`) and hard-coded which schema majors it reads. The new gates of #295, #296
and #297 went further and rewrote quantbox internals: dataset identity, served
tables, the bar calendar. Those PRs took three or more review rounds each. The
decision of 2026-10-26 (see `status_changes`) moves the domain gates into the
product; qute-research keeps the prompts only and calls the repo's CLI.

## Decision

1. **`quantbox preflight` is the gate.** The module is `quantbox.preflight`, in the
   research layer (its audience is the lab); the CLI verb loads it through
   `quantbox._lazy`. It sits next to `quantbox gates` (`gates.py`, `gates_cli.py`):
   the gates decide on a return series after a run, the pre-flight decides on the
   files before one.
2. **Behaviour-identical port of qute-research 0.13.0.** Same inputs (`--lab`,
   `--line`, `--prereg`, `--id`, `--runs-dir`, configs), same exit codes (0 pass,
   1 refuse, 2 could not check, 2 on a usage error), same output lines. The
   plugin's tests are ported to `tests/test_preflight.py`.
3. **No second copy of quantbox.** Where the plugin asked quantbox from outside,
   the port calls quantbox's own code:

   | plugin site | port |
   |---|---|
   | `explain_config`: subprocess `quantbox config explain <config> --json`, cwd = the line | `quantbox.explain.explain_path`, the one reader the CLI verb also uses, run with cwd = the line and stdout sent to stderr |
   | `resolve_dataset`: subprocess `quantbox dataset resolve <name> --config <config> --json` | `quantbox.dataset_lock.resolve_dataset` at `dataset_lock.lock_for_run(config)`, the one lock rule the CLI verb also uses |
   | `quantbox_cli.SUPPORTED`: hard-coded `run@1`, `explain@1` | the majors of `quantbox.run_manifest.SCHEMA_ID` and `quantbox.explain.SCHEMA_ID` |
   | the run-dir stamp parsed as `%Y%m%dT%H%M%S` | `quantbox.run_history.RUN_TS_FORMAT`; the search stays the plugin's (a stamp anywhere in a name), so no run evidence is lost |
   | `quantbox_cli.command()`: which quantbox to ask (`$QUANTBOX_CMD`, PATH, `uv run`) | removed: the quantbox that runs the gate is the one the line pins |

   The YAML subset parser for the lab's own files (system.yaml, the prereg block,
   data_inputs.yaml) is kept unchanged. It is not a copy of quantbox: no quantbox
   reader refuses duplicate keys and number spellings YAML readers disagree about.
4. **Versioned and pinned with the engine.** A line runs the pre-flight of the
   quantbox its `pyproject.toml` pins. A gate change ships in a quantbox release
   and reaches a line through `quantbox line repin`, like an engine change.
5. **Later gates follow the same rule.** A gate that reads `data_validation.json`,
   a run manifest or a dataset reads it through the quantbox module that writes
   it, never by its own parse of the layout (#296, #297 land here on this rule).

## Consequences

- (+) The gate reads the real product objects, so the class "second copy of
  dataset resolution" goes away.
- (+) A new gate is a quantbox PR with quantbox's tests and before/after.
- (−) Until qute-research calls `quantbox preflight`, two copies exist. The
  plugin's copy is frozen at 0.13.0; a fix lands here.
- (−) Three reason texts change, all on paths that fire only when quantbox
  cannot answer or a manifest is of an unknown major:
  - a config whose plan raises reads "`quantbox config explain <path>` raised
    <Error>: <message>" instead of the subprocess's stderr tail;
  - a by-name dataset quantbox cannot resolve reads "`quantbox dataset resolve
    <name> --config <config>` raised DatasetResolveError: <message>" instead of
    "... exited 1: ERROR: <message>";
  - a run manifest of an unknown major reads "this quantbox reads / cannot read"
    and "Run the pre-flight with the quantbox that wrote it" instead of "this
    plugin" and "Update qute-research".
  Status, check id and exit code are unchanged on all three.
