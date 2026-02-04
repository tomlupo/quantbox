# Changelog

Ordered by import (oldest to newest).

## Import quantbox_repo.zip (6918dd0)
- Packages: packages/quantbox-core/README.md, packages/quantbox-core/pyproject.toml, packages/quantbox-core/src/quantbox/__init__.py, packages/quantbox-core/src/quantbox/cli.py, packages/quantbox-core/src/quantbox/contracts.py, packages/quantbox-core/src/quantbox/registry.py, (+11 more)
- Configs: configs/run_fund_selection.yaml
- Scripts: scripts/make_sample_data.py
- Other: .gitignore, README.md

## Import quantbox_repo_llm_friendly.zip (f4781e2)
- Packages: packages/quantbox-core/pyproject.toml, packages/quantbox-core/src/quantbox/cli.py, packages/quantbox-core/src/quantbox/contracts.py, packages/quantbox-core/src/quantbox/llm_utils.py, packages/quantbox-core/src/quantbox/runner.py, packages/quantbox-core/src/quantbox/store.py, (+3 more)
- Docs: docs/LLM_OPERATIONS.md
- Schemas: schemas/allocations.schema.json, schemas/prices.schema.json, schemas/rankings.schema.json, schemas/scores.schema.json, schemas/universe.schema.json
- Tests: tests/test_plugin_discovery.py, tests/test_run_manifest.py
- Recipes: recipes/add_broker_plugin.md, recipes/add_pipeline_plugin.md
- Other: CONTRIBUTING_LLM.md, README.md, contracts/ARTIFACTS.md

## Import quantbox_repo_llm_trading_bridge.zip (08ce148)
- Packages: packages/quantbox-plugin-broker-binance-stub/pyproject.toml, packages/quantbox-plugin-broker-binance-stub/src/quantbox_plugin_broker_binance_stub/__init__.py, packages/quantbox-plugin-broker-binance-stub/src/quantbox_plugin_broker_binance_stub/broker.py, packages/quantbox-plugin-broker-ibkr-stub/pyproject.toml, packages/quantbox-plugin-broker-ibkr-stub/src/quantbox_plugin_broker_ibkr_stub/__init__.py, packages/quantbox-plugin-broker-ibkr-stub/src/quantbox_plugin_broker_ibkr_stub/broker.py, (+4 more)
- Docs: docs/LLM_OPERATIONS.md, docs/PIPELINE_CHAINING.md
- Configs: configs/run_trade_from_allocations.yaml
- Schemas: schemas/fills.schema.json, schemas/orders.schema.json, schemas/portfolio_daily.schema.json, schemas/targets.schema.json
- Tests: tests/test_plugin_discovery.py, tests/test_trading_bridge.py
- Other: README.md, contracts/ARTIFACTS.md

## Import quantbox_repo_llm_trading_bridge_real_brokers.zip (3665ae9)
- Packages: packages/quantbox-plugin-broker-binance/README.md, packages/quantbox-plugin-broker-binance/pyproject.toml, packages/quantbox-plugin-broker-binance/src/quantbox_plugin_broker_binance/__init__.py, packages/quantbox-plugin-broker-binance/src/quantbox_plugin_broker_binance/broker.py, packages/quantbox-plugin-broker-ibkr/README.md, packages/quantbox-plugin-broker-ibkr/pyproject.toml, (+2 more)
- Docs: docs/BROKER_SECRETS.md
- Configs: configs/run_trade_from_allocations_binance_live.yaml, configs/run_trade_from_allocations_ibkr_paper.yaml
- Tests: tests/test_plugin_discovery.py
- Other: README.md

## Import quantbox_repo_llm_full_upgrade.zip (28502e0)
- Packages: packages/quantbox-plugin-data-duckdb-parquet/README.md, packages/quantbox-plugin-data-duckdb-parquet/src/quantbox_plugin_data_duckdb_parquet/plugin.py, packages/quantbox-plugin-pipeline-alloc2orders/src/quantbox_plugin_alloc2orders/pipeline.py
- Docs: docs/TRADING_BRIDGE_ADVANCED.md
- Configs: configs/instruments.yaml, configs/run_fund_selection.yaml, configs/run_trade_from_allocations.yaml, configs/run_trade_from_allocations_binance_live.yaml, configs/run_trade_from_allocations_ibkr_paper.yaml
- Scripts: scripts/make_sample_data.py
- Tests: tests/test_trading_bridge.py
- Other: README.md

## Import quantbox_repo_llm_full_upgrade_auto_latest_approval.zip (b90ff61)
- Packages: packages/quantbox-core/src/quantbox/run_history.py, packages/quantbox-plugin-pipeline-alloc2orders/src/quantbox_plugin_alloc2orders/pipeline.py
- Docs: docs/APPROVAL_GATE.md
- Configs: configs/run_trade_from_allocations.yaml
- Scripts: scripts/approve_orders.py
- Tests: tests/test_trading_bridge.py
- Other: README.md
