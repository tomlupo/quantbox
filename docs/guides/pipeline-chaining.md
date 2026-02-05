# Pipeline chaining (manual, v0)

QuantBox core is intentionally minimal. For now, chaining is manual:
- run research pipeline -> produces an artifact (e.g. allocations.parquet)
- run trading pipeline -> consumes that artifact via a path

In v1, you can add either:
1) a meta-pipeline plugin that calls two pipelines internally, or
2) a small utility that resolves "latest run id" for a pipeline from artifacts root.
