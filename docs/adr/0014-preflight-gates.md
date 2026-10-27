---
adr: 0014
title: The pre-flight's new gates — FLAG, the binding count and a replication's data step — measure quantbox's own objects
status: accepted
date: 2026-10-27
supersedes:
superseded_by:
amends: 0013
status_changes:
  - 2026-10-27: accepted with TOM-2471 — decision note vaults/projects/infra/decisions/decyzja-2026-10-26-stos-kiss-repo-hotl-fabryka.md (domain gates live in the product); Tom 2026-10-25 (replica parity and leverage above the live cap are FLAGs, not blocks)
---

# ADR-0014: The pre-flight's new gates measure quantbox's own objects

## Context

qute-plugins #296 (TOM-1119, TOM-1412 points 1-2) and #297 (TOM-1430) added gates to
the plugin's pre-flight. Both PRs ran three review rounds and stayed NO-SHIP. Every
finding had one shape: the plugin re-implemented a quantbox object from outside and
got an edge wrong. It parsed a run's date axis, compared each run with its own pin but
not with each other, read the pinned prices where quantbox serves a returns table,
counted a bool or a negative number as a count, and computed a weekday calendar that
had no Good Friday. ADR-0013 decision 5 says such gates read quantbox's own modules.
This ADR records how the gates do that in `quantbox preflight`.

## Decision

1. **FLAG is a fourth status; it exits 3.** A FLAG does not stop a run. It limits what
   its finding may be called, and a `finding limits:` line names every FLAG row. A
   refusal still exits 1 and blindness 2; nothing flagged exits 0.
2. **`venue.leverage` is a promotion limit (TOM-1119).** A runner cap above it is FLAG
   "ABOVE LIVE CAP ... NOT PROMOTABLE", never REFUSE.
3. **Replica parity is a FLAG (TOM-1412 point 2, Tom 2026-10-25).** A book test PASSES
   `replica-parity` only when `books[].replica_parity` names a finding with `verdict:
   confirmed` and a review `done` / `UPHELD`. One code line marks where it becomes a
   REFUSE. No config switch exists.
4. **The binding count is measured here (TOM-1412 point 1).** A leg `versus: baseline`
   or an overlay needs `binding: {min_share, baseline, baseline_run, arm_run, tol}`.
   The gate reads both runs' own `run_manifest.json` (a run@N major this quantbox
   reads, the config's hash, the config's pin). It refuses unless BOTH runs read the
   SAME dataset bytes. It reads both `traded_weights` parquet files as quantbox wrote
   them and counts BARS on an IDENTICAL index; it never parses, sorts or realigns the
   axis. A NaN weight refuses. No probe file and no sha binding of a probe exist,
   because nothing is measured outside the gate.
5. **A replication's data step measures the SERVED table (TOM-1430).** The table is
   the one `dataset_identity` names: a manifest `serve` block's table and kind, or the
   pin file. Its bytes are read once through `dataset_lock.served_bytes`, which refuses
   bytes that are not the pin or the vouched hash. A `kind: returns` table is measured
   as returns: a price is non-positive only at a return <= -100 %. `data_validation.json`
   is read through its schema major, never recomputed. There is no hygiene probe, and
   `prereg.data.hygiene` is not read.
6. **Coverage ends are judged on the series' own observed bars.** A class has covered
   a window end it stopped N days before when its own bars sit N or more days apart
   somewhere in its last year (its first year at the start side). No weekday, month or
   holiday calendar is computed.
7. **One count contract.** `_is_count(value, least)` is an int, never a bool, and at
   least `max(least, 0)`. No caller compares a count with a bound itself.

## Consequences

- (+) The five review classes of #296 and #297 cannot recur: the gate holds no second
  copy of a run's axis, a dataset's identity, its served table or a calendar.
- (+) A line needs no probe step before the pre-flight. Two runs and a data-probe run
  are enough.
- (−) The prereg contract differs from the plugin's draft: `binding.probe` and
  `data.hygiene` are gone, and `binding` names the two run directories. No line uses
  either today. qute-research's `preregistration.md` changes when it calls `quantbox
  preflight`.
- (−) A book test exits 3, not 0, until system.yaml names a green `replica_parity`.
  A book prereg with a leg `versus: baseline` or an overlay refuses until it adds
  `binding`. H23 (the incident) refuses now.
- (−) Coverage is lenient by the series' own largest gap: a weekday series that stops
  on a Thursday before a 4-day holiday gap passes, even where that Friday traded.
