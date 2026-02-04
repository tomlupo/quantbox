# QuantBox Artifact Contracts (v1)

This folder is the canonical place for **artifact schemas**.

Artifacts are stored as Parquet, but schemas are expressed as lightweight JSON descriptors
so they can be validated by tools (including LLM agents).

Current v1 artifacts (starter):
- universe
- prices
- scores
- rankings
- allocations

JSON schemas live in `/schemas/*.schema.json`.
