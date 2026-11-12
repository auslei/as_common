# Investor-IQ Common Module (as-common)

Shared utilities and core abstractions for the Investor-IQ ecosystem. A lightweight, stable foundation used by multiple microservices (agents, extraction services, etc.).

## Features

- **Unified LLM Interface:** Multi-provider support — Ollama, OpenAI, DeepSeek, LM Studio, OpenRouter, LlamaCpp.
- **Job Queuing:** FastAPI-integrated serialized queue for long-running GPU/CPU tasks.
- **Structured Extraction:** Pydantic-based schema validation and LLM output parsing with automatic repair.
- **Utilities:** Common helpers for ML, file operations, document conversion, media downloading, and more.

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

### Installation

```bash
# Core only
uv sync

# With optional features
uv sync --extra all
# Or individually: --extra ml  --extra media  --extra pdf
```

### Running Tests

```bash
uv run pytest                  # unit tests
uv run pytest tests/manual/    # integration tests (requires live LLM providers)
uv run pytest --cov            # with coverage report
```

## Directory Structure

```
src/common/         # Core source code
tests/              # Unit tests
tests/manual/       # Integration tests (not run by default)
sync_common.py      # Sync script for sibling as_* projects
Makefile            # Common dev targets
```

## Syncing to Sibling Projects

`as_common` is shared by copying `src/common/` directly into each sibling `as_*` project's `common/` directory. After making changes, run:

```bash
make sync
```

This scans `../as_*/` for any project with a `common/` directory and:
1. Copies updated `.py` files and `MODULE_CONTEXT.md`
2. Adds any new dependencies to each project's `pyproject.toml`

Then run `uv sync` in each updated project to install new dependencies.

## Development

```bash
make sync      # push changes to sibling as_* projects
make test      # run unit tests
make lint      # ruff check
make format    # black format
```

Follows the standards in `GLOBAL_STANDARDS.md`. Always use `uv` for dependency management, type hints on all function signatures, and `get_logger()` from `logger.py` for logging.

## License

(Specify license here or refer to a LICENSE file)
