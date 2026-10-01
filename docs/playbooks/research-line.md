# Research lines: create one, move its engine pin

A research line is a Model C project: its own `pyproject.toml` and `uv.lock`, nothing
shared with the lab around it. How the pins are derived is the `quantbox.line` module
docstring; this page is the two commands.

## Create

```bash
quantbox new line <slug>                       # pinned to the latest v* tag
quantbox new line <slug> --quantbox-ref v0.8.0 # or a named tag / SHA
cd <slug>
export QUANTBOX_DATASETS_ROOT=/path/to/quantbox-datasets/datasets
uv sync
uv run quantbox run -c config.yaml
```

What it writes:

| File | What it is |
|---|---|
| `pyproject.toml` | quantbox at the 40-char SHA the tag names (tag in `[tool.quantbox-line]`), quantbox-datasets at a SHA, and every transitive dependency `==`-pinned in a managed `constraint-dependencies` block |
| `uv.lock` | resolved under those pins; `uv sync --locked` reproduces it |
| `README.md` | line frontmatter and the `prereg:` block, placeholders the pre-run gate refuses until filled |
| `datasets.lock` | sha256 of the base dataset's `prices.parquet`, read from `$QUANTBOX_DATASETS_ROOT` ([TOM-1349](../../src/quantbox/dataset_lock.py) format) |
| `config.yaml` | the base backtest: the dataset by name, `execution.lag_bars: 1` |
| `arms.yaml` | the arms over that base ([`quantbox.arms`](../../src/quantbox/arms.py) format) |
| `tests/test_reproduces.py` | `-m reproduction`: re-runs `config.yaml`, diffs the run manifest's metrics against `expected_metrics.json` |

A ref that predates line support is refused by name, never scaffolded into a line that
cannot run. When `$QUANTBOX_DATASETS_ROOT` cannot be read, `datasets.lock` is written
unpinned and the command warns; pin it with `quantbox-datasets pin <name>`.

Capture the golden once the run is the one you mean, and commit it:
`QUANTBOX_CAPTURE_GOLDEN=1 uv run pytest -m reproduction`.

## Repin

One command, then the reproduction:

```bash
quantbox line repin <line-dir> --ref vX.Y.Z   # omit --ref for the latest v* tag
cd <line-dir> && uv sync && uv run pytest -m reproduction
```

`repin` rewrites the quantbox SHA and its tag, rebuilds the managed pins block from the
new ref's own `uv.lock` plus the line's resolution, and refreshes `uv.lock`
(`--no-lock` skips that last step). A red reproduction after a repin means the engine
moved a number: record it as a finding, never loosen the tolerance or rewrite the golden
to make it green.

`repin` works on lines `quantbox new line` made: it refuses a `pyproject.toml` without
exactly one `quantbox @ git+…@<sha>` dependency and the managed block. Like `new line`,
it refuses a ref that predates line support (explicit `--ref` or the latest-tag default)
before it writes anything.
