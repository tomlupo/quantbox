# Recipe: Add a pipeline plugin

1) Create package: `packages/quantbox-plugin-pipeline-<name>/`
2) Add entry point group:

```toml
[project.entry-points."quantbox.pipelines"]
"<pipeline_id>" = "<module>:<ClassName>"
```

3) Implement `PipelinePlugin.run(...)` and set:
- `kind = "research"` or `"trading"`
- `meta.params_schema` (JSON Schema)
- `meta.outputs` list (artifact names)

4) Write artifacts using `store.put_parquet(name, df)` and `store.put_json(...)`

5) Add an example config under `configs/`.
