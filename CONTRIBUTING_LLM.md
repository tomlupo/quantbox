# Contributing with LLMs (safe editing protocol)

Rules to keep QuantBox stable while allowing agentic development.

## Do
- Keep `quantbox` core **small** and avoid strategy logic in core.
- When changing any artifact schema:
  - bump `schema_version` in the producing plugin meta
  - add/modify JSON schema under `/schemas`
  - add a contract test
- Add tests for any new plugin or core behavior.
- Prefer **additive** changes and new plugin versions over breaking changes.

## Don't
- Do not rename existing entry-point ids (e.g. `fund_selection.simple.v1`) unless you create a new version id.
- Do not embed secrets in YAML configs or artifacts. Use env vars / secret refs.
- Do not make pipelines depend on `print()` parsing. Use structured events.

## Required checks after edits
1. `quantbox plugins list`
2. `quantbox validate -c <config>`
3. `quantbox run -c <config>`
4. `pytest -q` (if tests are installed)
