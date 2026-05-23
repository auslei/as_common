.PHONY: sync test lint format

sync:
	uv run python sync_common.py

test:
	uv run pytest

lint:
	uv run ruff check src/common tests

format:
	uv run black src/common tests
