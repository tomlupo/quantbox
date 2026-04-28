# Dataset — {logical name}

> One-paragraph human description. What this dataset *means* and how it fits the broader pipeline.

---

## What this is

A {wide | long}-format DataFrame of <thing>, emitted by <producer plugin/pipeline> and consumed by <consumers>. <One sentence on its role in the system.>

## Schema

Machine-readable: [`schemas/{name}.schema.json`](../../schemas/{name}.schema.json)

| Column | Type | Semantics | Source |
|---|---|---|---|
| `date` | `datetime64[ns]` | Last business day of the period | upstream price source |
| `symbol` | `str` | analizy.pl-aligned fund code | `dm_evo.data.fund_master.v1` |
| `score` | `float64` | Composite 0–100 score within scoring_category | computed |
| `rank` | `int64` | 1 = best within (date, scoring_category) | computed |

## Coverage

- **Date range**: 2010-01-01 → present (continuous, no gaps)
- **Frequency**: weekly (Friday EOD) | daily | monthly month-end
- **Freshness contract**: latest date ≥ last <Friday | trading day> at runtime; pipelines fail loudly if stale > 7 days
- **Holes**: holidays carry forward last value; permanent NaN forbidden in `score` column

## Producer

- **Plugin**: `{namespace}.strategy.{slug}.v1`
- **Pipeline**: `pipelines/datasets/{slug}/compute.py`
- **Cadence**: weekly Tuesday 07:00 (Warsaw)
- **Output path**: `data/published/{slug}/{name}.parquet`
- **Run manifest**: each run records the input dataset content-hashes; see [`../architecture/lifecycle.md`](../architecture/lifecycle.md)

## Consumers

| Consumer | What it reads | Purpose |
|---|---|---|
| `{namespace}.pipeline.allocation.v1` | latest snapshot | Portfolio construction |
| API endpoint `/scores/latest` | latest snapshot | Serve to frontend |
| `quantbox-revalidate` cron | full history | Drift checks against locked baseline |

## Quality contract

- `score` is non-NULL for every `(date, symbol)` where universe membership applies
- `(date, symbol)` is unique
- `score` ∈ [0, 100]
- Within a single date, `rank` is dense across all funds in `scoring_category`
- Monotonicity: `rank == 1` row's `score` ≥ all other `score` values in the same `(date, scoring_category)`

The producer pipeline asserts these contracts before writing. Consumers may assume them and need not re-validate.

## Loading examples

### Python (pandas)

```python
import pandas as pd
df = pd.read_parquet("data/published/fund_selection/scores.parquet")
print(df.shape, df.columns.tolist())
print(df.head())
```

### Python (DuckDB)

```python
import duckdb
con = duckdb.connect()
df = con.execute("""
    SELECT date, symbol, score, rank
    FROM 'data/published/fund_selection/scores.parquet'
    WHERE date = (
        SELECT MAX(date) FROM 'data/published/fund_selection/scores.parquet'
    )
    ORDER BY rank
""").fetch_df()
```

### CLI (DuckDB)

```bash
duckdb -c "SELECT COUNT(*), MIN(date), MAX(date)
            FROM 'data/published/fund_selection/scores.parquet'"
```

## Known issues

- Funds added mid-period have only forward data; treat as left-censored (`first_seen` column not in this dataset; query `data/published/fund_master.parquet` if needed).
- Rare cross-currency rebases on certain providers can shift `score` by ±5 points within a single date; flagged in producer logs as `WARN: rebase_shift`.
- Pre-2015 data has weekly cadence holes during Polish bank holidays not handled by upstream calendar; downstream consumers must forward-fill.

## See also

- Schema (machine-readable): [`schemas/{name}.schema.json`](../../schemas/{name}.schema.json)
- Producer plugin: [`src/{path}/plugin.py`](../../src/{path}/plugin.py)
- Methodology: [`../methodology/{methodology-slug}.md`](../methodology/{methodology-slug}.md)
- Lifecycle: [`../architecture/lifecycle.md`](../architecture/lifecycle.md)
