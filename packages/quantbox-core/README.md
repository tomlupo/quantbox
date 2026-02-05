# quantbox (core)

Core runtime + plugin registry for **QuantBox**.

This package contains:
- plugin contracts (`quantbox.contracts`)
- entry-point discovery registry (`quantbox.registry`)
- artifact store (`quantbox.store`)
- runner (`quantbox.runner`)
- CLI (`quantbox.cli`)

Core intentionally contains **no strategy logic** and **no broker-specific code**.

## Using quantbox as a dependency

Other projects depend on this package (name `quantbox`), not the workspace root.

**From PyPI (or a published wheel):**

```toml
dependencies = ["quantbox>=0.1.0"]
```

**Local editable (ongoing development):** In your other project’s `pyproject.toml`, point at this package so changes here take effect without reinstalling:

```toml
[project]
dependencies = ["quantbox"]

[tool.uv.sources]
quantbox = { path = "../quantbox/packages/quantbox-core", editable = true }
```

Adjust the `path` to where the quantbox repo lives relative to your project. Then run `uv sync` in your project.

## Extending with your own plugins

You do **not** need to change the quantbox repo. In your own repo or package:

1. Depend on `quantbox`.
2. Implement a pipeline (or data/broker) plugin and register it via entry points:

   ```toml
   [project.entry-points."quantbox.pipelines"]
   "my_research.momentum.v1" = "my_plugins.pipeline:MomentumPipeline"
   ```

3. Install your package in the same environment as `quantbox`. Run from your project: use your own `configs/` and `plugins/manifest.yaml` (the runner uses the manifest under current working directory). The registry merges built-in and entry-point plugins by name.

See `recipes/add_pipeline_plugin.md` and `recipes/add_broker_plugin.md` in the repo for Option A (built-in) vs Option B (external entry points).
