# CLAUDE.md

**as-common** (v0.1.0) — the stable, dependency-light Python foundation the other `as_*` services copy in. Four domains: unified LLM interface, job queue, structured extraction, misc utilities (ML, files, docs, media, logging).

Usage examples live in `README.md`; this file is the working map.

## Commands

```bash
uv sync                         # core deps; --extra all for ml/media/pdf
uv run pytest                   # unit tests (manual tests excluded via pyproject)
uv run pytest tests/manual/     # integration tests — need live LLM providers
uv run pytest --cov             # coverage
uv run black src/common tests   # format
uv run ruff check --fix src/common tests   # lint
make sync                       # copy src/common into sibling ../as_*/common/ dirs
```

## Environment variables

- `PYTHONPATH=.` — required when running scripts directly
- `DEBUG=true` — DEBUG-level logging · `LOG__TO_FILE=true` — write to `logs/`
- `OLLAMA_BASE_URL`/`OLLAMA_API_URL` (default `http://localhost:11434`), `LMSTUDIO_BASE_URL` (default `http://localhost:1234/v1`)
- `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY` — per-provider
- `FORCE_DEVICE` — override device detection (cuda/mps/cpu)

## Module map (`src/common/`)

```
unified_llm.py      # UnifiedChatModel — LangChain BaseChatModel over all providers
llm_utils.py        # factory, message parsing, token counting
LLMProvider.py      # provider enums & settings
job_queue.py        # FastAPI-integrated serialized task queue
pydantic_helper.py  # structured-output prompt building & parsing
config.py           # YAML config singleton
logger.py           # get_logger() — Rich console logging
models.py           # domain models
document_utils.py   # markdown→PDF (Pandoc)   download_utils.py  # media (yt-dlp)
utils.py ml_utils.py docker_utils.py file_utils.py
```

## Behaviour to know (non-obvious)

- **UnifiedChatModel**: raw models cached in `_RAW_LLM_CACHE`; `thinking=` overridable per call; `schema=Model` triggers JSON-schema injection + multi-stage parse + LLM repair; `thinking=False` strips `<think>`/`<tool_call>` tags.
- **JobQueue**: serialized FIFO, `max_concurrency=1` for GPU safety; dedups identical jobs; lifecycle `queued→running→completed/failed`; keeps last 100; `make_router()` exposes `/jobs`.
- **pydantic_helper**: 4-stage output recovery — direct JSON → bracket extraction → schema-echo normalization → LLM repair.
- **config**: resolution order `PACKAGE_CONFIG_PATH` → `packages/{NAME}/config.yaml` → `src/{NAME}/config.yaml` → root `config.yaml`.
- **Debugging LLM**: `DEBUG=true`; `model.invoke(prompt, thinking=False)` to strip artifacts; `get_token_count()` from `llm_utils.py`; manual probes in `tests/manual/integration_llm.py`.

## Conventions (this repo's stack rules)

- `uv` for all deps (`uv add`, `uv lock`) — never pip/poetry.
- `snake_case` vars/functions, `PascalCase` classes; Black + Ruff, imports alphabetized.
- Type hints on every signature; Pydantic (`pydantic_helper.py`) for complex structures & tool args.
- Logging via `get_logger()` from `logger.py` — never `print()` or stdlib `logging`.
- Keep files under ~30KB (license/obfuscation tooling); split into sub-modules as they grow.
- Descriptive docstrings on tool definitions — they're used as LLM metadata.
- In graph workflows, verify the `State` schema before changing node logic.
- Shared logic lives in `src/common`; update `src/common/MODULE_CONTEXT.md` when a public API changes.

## Procedures

- **New utility**: add to the right `src/common/` module (new file if >30KB) → type hints + docstring → `get_logger()` → `tests/test_<module>.py` → update `MODULE_CONTEXT.md` if public.
- **New LLM provider**: init in `UnifiedChatModel._init_underlying_model()` → env/API-key validation → stop tokens in `STOP_TOKENS`/`get_tokens_for_model()` → test in `tests/test_unified_llm.py` + `tests/manual/integration_llm.py`.
- **Sync to siblings**: `make sync` scans `../as_*/` for a `common/` dir, copies changed `.py` + `MODULE_CONTEXT.md`, merges new deps into their `pyproject.toml`; then `uv sync` each.

## Pointers

- `src/common/MODULE_CONTEXT.md` — public API reference
- `README.md` — quick start + usage examples
- `pyproject.toml` — deps, test config, build settings
