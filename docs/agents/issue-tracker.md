# Tracker binding

<!-- This file follows Matt Pocock's setup-matt-pocock-skills convention: skills
     (his and qute's) read it to know where work lives. The HTML marker below is
     the machine-readable declaration — qute's routing engine reads it; the prose
     is for agents and humans. Keep exactly one marker. -->

<!-- qute-tracker: linear team=TOM -->

## Task source — Linear

**Linear is the task source** (qute-code-kit ADR-0004): all work items — tasks, planning,
priority, agent assignment — live in Linear (team `TOM`, project **quantbox**). Jimek monitors
Linear for assigned tasks; `conductor.yml` declares how the repo's workflows run. Humans and
agents pull work from Linear only.

- Auth: `LINEAR_API_KEY` env var (personal API key) — interactive/local sessions only.
- qute `/task` and `/repo-status` route here automatically via the marker above.
- **Orchestrated workspaces** (Jimek / Symphony-Elixir style): the API key is held
  host-side and stripped from your environment by design — use the orchestrator's
  advertised `linear` tool for state changes and comments, not the qute backend.

## GitHub Issues — issue record only

GitHub Issues on `tomlupo/quantbox` track **issues, not tasks**: bugs, defects, technical debt
attached to the code. An issue becomes work only when a Linear task references it
("fix issue #X"). Never pull work from the Issues list directly. File records with
`/task ... --to github`.

## Ideas

Ideas go to Linear (label `research` for research ideas) — never to `RESEARCH_IDEAS.md`,
session notes, or files inside the repo tree.
