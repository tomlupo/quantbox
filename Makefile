.PHONY: dev dev-full plugins doctor run-sample

# Create venv and sync workspace dependencies
# Requires: uv

dev:
	uv venv
	. .venv/bin/activate && uv sync

# Install optional broker deps

dev-full:
	uv venv
	. .venv/bin/activate && uv sync --extra ibkr --extra binance

plugins:
	. .venv/bin/activate && uv run quantbox plugins list

doctor:
	. .venv/bin/activate && uv run quantbox plugins doctor

run-sample:
	. .venv/bin/activate && uv run quantbox run -c configs/run_fund_selection.yaml
