# Project Templates

QuantBox ships project skeletons via `quantbox new`. A new quant project starts from a template, gets the SDK pinned, gets foundation skills installed, and works in 60 seconds.

---

## What templates are

A template is a directory under `templates/` containing:

- Pre-wired `pyproject.toml` with quantbox SDK pinned + sensible defaults.
- `CLAUDE.md` and `AGENTS.md` reflecting quantbox conventions.
- `.claude/skills/` and `.claude/rules/` with foundation pieces installed.
- Project structure (`src/`, `cookbook/configs/`, `research/`, `data/`, `output/`).
- One working example pipeline (config + script) that proves the install end-to-end.
- `.github/workflows/` with CI for tests + schema validation.

Templates encode shared shape. The SDK provides shared behavior. Together they make new projects coherent without imposing a runtime framework.

---

## The four shipped templates

| Template | For | Pre-wired |
|---|---|---|
| `research` | Backtesting, hypothesis testing, dashboards | vectorbt + riskfolio + plotly + jupyter; `quantbox-research` skill default; minimal lifecycle (research-only) |
| `trading` | Live broker, scheduler, risk gates, Discord publisher | ccxt/binance + telegram + healthchecks; `quantbox-backtest` + `quantbox-promote` skills; full lifecycle states |
| `advisory` | Multi-profile portfolio, fund selection, reports, web app | vectorbt + lightgbm + riskfolio + FastAPI; `dm-evo`-shape (regulated advisory) |
| `client` | Turnkey client engagement | varies; lightweight default with branding hooks |

All four share:
- Foundation skills (`quantbox-core`, `quantbox-config`, `quantbox-research`, `quantbox-promote`).
- Lifecycle conventions (`meta.status`, EXPERIMENTS.md, STATUS.md).
- Data layer pattern (raw → intermediate → processed → published) via `paths.toml`.
- Convention `output/{artifact}/{date}/run_{tag}/` for run artifacts.

What differs:
- Adapters baked in.
- Project structure (advisory has a `apps/` for the web app; trading has a `bots/` for live processes).
- Default skills installed.
- Default `meta.status` for plugins (research-only vs full lifecycle).

---

## `quantbox new` CLI

```bash
quantbox new <target_dir> [--template TEMPLATE] [--name NAME]
```

| Flag | Default | Notes |
|---|---|---|
| `--template` | `research` | One of `research`, `trading`, `advisory`, `client` |
| `--name` | basename of `target_dir` | Project package name |
| `--no-init` | (unset → init) | Skip `git init` and `uv sync` |
| `--with-skills LIST` | foundation set | Add additional skills from `quantbox/skills/` |

Behavior:

1. Create `target_dir` if missing.
2. Copy template tree into `target_dir`.
3. Substitute `{{ project_name }}` placeholders.
4. Install skills from `quantbox/skills/` into `target_dir/.claude/skills/`.
5. Install rules from `quantbox/rules/` into `target_dir/.claude/rules/`.
6. Generate `pyproject.toml` with quantbox SDK pinned at current version.
7. Run `git init && uv sync` (unless `--no-init`).
8. Run the example pipeline as a smoke test (`quantbox run -c cookbook/configs/example.yaml`).
9. Print "next steps" — including how to upgrade later.

The smoke test step is the contract: if it fails, the template is broken.

---

## `quantbox upgrade` — keeping projects in sync

Templates rot. Projects diverge. To re-apply the template against an existing project (3-way merge):

```bash
quantbox upgrade
```

Built on Copier (or equivalent). Records the template version in `.quantbox-template.yaml`; subsequent upgrades diff from the recorded base.

Conflicts surface as standard merge markers. Project-specific changes are preserved.

---

## Convention enforcement — `quantbox doctor`

Detects projects that drifted from current conventions:

```bash
quantbox doctor
```

Checks:

| Check | Diagnoses |
|---|---|
| Foundation skills present | Missing `quantbox-research`, etc. |
| `paths.toml` exists | Data layer override missing |
| Plugin entry points have `meta.status` | Stale plugins from older quantbox versions |
| Schemas pass `--strict` validation | Drift in published artifacts |
| `pyproject.toml` quantbox version matches lockfile | Out-of-sync deps |
| `.claude/rules/research-workflow.md` present | Project missing the convention rule |

Doctor only reports — it doesn't auto-fix. Use `upgrade` to fix.

---

## Template anatomy

```
templates/research/
├── {{ project_name }}/                    # placeholder, substituted at copy time
├── pyproject.toml                          # template form, pre-pins quantbox
├── CLAUDE.md                               # mentions conventions, references quantbox docs
├── AGENTS.md
├── README.md
├── .gitignore
├── .claude/
│   ├── skills/                             # foundation skills installed here
│   ├── rules/                              # research-workflow.md, git-workflow.md
│   └── settings.json
├── src/{{ project_name }}/                 # project package
│   ├── __init__.py
│   └── plugins/                            # placeholder for project plugins
├── cookbook/configs/
│   ├── example_backtest.yaml               # working example
│   └── example_research.yaml
├── pipelines/
│   └── example_backtest.py                 # the script that runs cookbook/configs/example_backtest.yaml
├── research/                               # scratch-plugins land here
├── data/
│   ├── raw/.gitkeep
│   ├── intermediate/.gitkeep
│   ├── processed/.gitkeep
│   └── published/.gitkeep
├── output/.gitkeep                         # ArtifactStore default
├── paths.toml                              # data-layer override
├── tests/
│   └── test_smoke.py                       # asserts example_backtest runs
├── .github/workflows/
│   ├── tests.yml
│   └── schema-validation.yml
└── .quantbox-template.yaml                 # records template version for upgrades
```

---

## Adding or modifying a template

| Change | Where |
|---|---|
| Add a new template | `templates/{name}/` + register in `cli.py` `--template` choices |
| Update the SDK pin everyone gets | `templates/_common/pyproject_quantbox_pin.toml` |
| Update foundation skills installed by all templates | `cli.new` — list of skills to copy |
| Add a project shape (e.g., `ml-research`) | New template if shape is different enough; otherwise extend `research` |

Don't fork templates for one-off needs. If a project needs something the template doesn't, the project adds it locally; the template stays canonical.

---

## Convergence — existing projects

Existing projects (quantbox-live, quantbox-lab, dm-evo, etc.) don't retrofit from templates. They converge through SDK adoption — using foundation skills, adapters, and lifecycle conventions over time.

A project might choose to re-stamp from a template at a major refactor (`quantbox upgrade --reset` against a fresh template), but it's optional. The SDK doesn't require it.

---

## Anti-patterns

| Anti-pattern | Fix |
|---|---|
| Template-specific skills (skills only in `templates/research/skills/`) | Skills live in `quantbox/skills/`; templates *install* them |
| Template-specific adapters | Adapters live in the SDK; templates pin them |
| Template-specific Python source | Project domain code lives in projects, not templates |
| Forking a template for one-off domain shape | Use the closest-matching template; let project diverge as needed |
| Templates drifting from SDK conventions | `quantbox doctor` catches this; CI checks generated outputs match current docs |

---

## See also

- [principles.md](principles.md) — the doctrine that templates encode.
- [skills.md](skills.md) — what skills get installed.
- [api-layers.md](api-layers.md) — what the example pipelines exercise.
- [playbooks/add-a-skill.md](../playbooks/add-a-skill.md) — how to add a new skill that templates pick up.
