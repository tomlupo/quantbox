# Pre-trade approval gate

After `orders.parquet` is written, the pipeline computes `orders_digest` (sha256, first 16 chars).
If `approval_required: true` in paper/live mode, the pipeline will only execute if an approval file exists.

Default approval file (if approval_path is null):
- `./approvals/<orders_digest>.json`

Example:
```json
{
  "approved": true,
  "orders_digest": "deadbeefcafebabe",
  "who": "tom",
  "when": "2026-02-01T09:00:00+01:00",
  "note": "ok to execute"
}
```

The pipeline writes:
- `orders_digest.json`
- `approval_status.json`

If approval is missing/mismatched, `fills.parquet` stays empty.
