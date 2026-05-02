# Common Module (ic-common)

Shared utilities and core abstractions for the Investor-IQ ecosystem — a lightweight, stable, dependency-efficient library used by multiple services (agents, extraction services, etc.).

See **GLOBAL_STANDARDS.md** for coding standards, design principles, and workflow rules.

## Core Purpose

- **Unified LLM Interface**: Multi-provider support via `UnifiedChatModel` (Ollama, OpenAI, DeepSeek, LlamaCpp, LM Studio)
- **Job Queuing**: FastAPI-integrated serialized queue for long-running GPU/CPU tasks (`job_queue.py`)
- **Structured Extraction**: Pydantic-based LLM output parsing with multi-stage recovery (`pydantic_helper.py`)
- **Utilities**: ML device detection, file ops, document conversion, media downloading, logging

## Commands

```bash
uv sync                        # install core deps
uv sync --extra all            # install with ml/media/pdf extras
uv run pytest                  # run unit tests (tests/)
uv run pytest tests/manual/    # run integration tests (require live LLM providers)
uv run black src/common tests  # format
uv run ruff check src/common tests  # lint
```

Set `PYTHONPATH=.` when running scripts directly.

## Module Reference

See `src/common/MODULE_CONTEXT.md` for the full public API of each module.

Key modules:

- `unified_llm.py` — `UnifiedChatModel`: LangChain `BaseChatModel`, lazy init, structured output via `schema=`, thinking toggle via `thinking=`
- `llm_utils.py` — `get_llm()` factory, message content stripping, token counting, Ollama VRAM release
- `LLMProvider.py` — `LLMSettings`, `LLMProvider` enums, `LLAMA_CPP_MODEL_PATHS`
- `pydantic_helper.py` — `build_structured_prompt()`, `parse_structured_output()` with LLM repair fallback
- `job_queue.py` — serialized FIFO queue, deduplication, `make_router()` for FastAPI
- `config.py` — singleton `config` from `config.yaml` via Pydantic
- `logger.py` — `get_logger(name, icon)` with Rich console; `LOG__TO_FILE=true` persists to `logs/`
- `models.py` — shared domain models (`ExtractionResult`, `ProjectMetadata`)
- `ml_utils.py` — `get_device()`, `get_gpu_usage()`, `millisec_to_time()`
- `docker_utils.py` — `resolve_project_dir()` for host/container path consistency
- `document_utils.py` — Markdown-to-PDF via Pandoc
- `download_utils.py` — YouTube/media download with caching via yt-dlp

## Directory Structure

- `src/common/` — source code
- `tests/` — unit tests
- `tests/manual/` — integration tests (require active LLM providers)
