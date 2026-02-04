# Recipe: Add a pipeline plugin

Two options:

## Option A: Built-in plugin (recommended for core plugins)
1) Add module under `packages/quantbox-core/src/quantbox/plugins/pipeline/`
2) Implement `PipelinePlugin.run(...)` and set:
- `kind = "research"` or `"trading"`
- `meta.params_schema` (JSON Schema)
- `meta.outputs` list (artifact names)
3) Register in built-ins map:
- `packages/quantbox-core/src/quantbox/plugins/builtins.py`
4) Add an example config under `configs/`.

## Option B: External plugin (separate repo/package)
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
