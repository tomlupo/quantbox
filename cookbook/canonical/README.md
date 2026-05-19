# Canonical Reproductions

End-to-end backtests on a committed synthetic fixture, with frozen
expected headline metrics. These guard against silent numerical drift
when pinning a new pandas/vbt/numba/quantbox version.

## What's here

```
canonical/
├── build_fixture.py       # regenerate the parquet fixture (deterministic)
├── fixture.parquet        # 5 symbols × 252 days, seeded GBM with regime flip
├── fixture_volume.parquet # synthetic volume panel
├── configs/
│   ├── momentum.yaml             # cross_asset_momentum.v1
│   └── trend_catcher_simple.yaml # trend_catcher_simple.v1
├── expected/
│   ├── momentum.json
│   └── trend_catcher_simple.json
├── run_all.py             # run + diff every canonical (no pytest needed)
└── regen_goldens.py       # re-bless every golden after an intentional change
```

## Run them

```bash
# Diff against committed goldens
uv run python cookbook/canonical/run_all.py

# Re-bless after intentional semantic changes
uv run python cookbook/canonical/regen_goldens.py
```

CI also runs them via `pytest tests/canonical_reproductions/` under the
`canonical_reproduction` marker.

## Tolerance

`atol=1e-4, rtol=1e-3` on the headline metrics:
`total_return, cagr, sharpe, max_drawdown, annual_volatility`.

Tight enough to catch real drift; loose enough that hardware-level
floating-point variance across runners stays in the noise.

## When a reproduction diverges

1. Confirm the change is intentional (look at recent diffs in strategy
   or pipeline code).
2. If yes: `uv run python cookbook/canonical/regen_goldens.py` and
   commit the new `expected/*.json` in the same PR with a clear note
   explaining what semantic changed and why.
3. If no: that's a bug — investigate before merging.
