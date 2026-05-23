# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ic-common** is a shared Python library (v0.1.0) providing core abstractions for the Investor-IQ ecosystem. It serves as a lightweight, stable, dependency-efficient foundation used by multiple microservices (agents, extraction services, etc.).

**Key Domains:**
- **Unified LLM Interface**: Multi-provider support (OpenAI, DeepSeek, Ollama, LlamaCpp, LM Studio)
- **Job Queuing**: FastAPI-integrated serialized queue for long-running GPU/CPU tasks
- **Structured Extraction**: Pydantic-based schema validation and LLM output parsing
- **Utilities**: ML, file operations, document conversion, media downloading, logging

## Build, Test & Development Commands

### Prerequisites
- Python 3.12+
- `uv` for dependency management (recommended over pip/poetry)

### Installation & Dependency Management

```bash
# Install core dependencies
uv sync

# Install with optional extras (ML, media, PDF support)
uv sync --extra all
# Or individually:
uv sync --extra ml --extra media --extra pdf
```

### Running Tests

```bash
# Run all unit tests (excludes manual/integration tests)
uv run pytest

# Run a single test file
uv run pytest tests/test_unified_llm.py

# Run a specific test function
uv run pytest tests/test_unified_llm.py::TestUnifiedChatModel::test_basic_invoke

# Run with coverage report
uv run pytest --cov

# Run manual/integration tests (require live LLM providers)
uv run pytest tests/manual/
```

### Code Quality

```bash
# Format code (Black)
uv run black src/common tests

# Lint with Ruff
uv run ruff check src/common tests

# Fix linting issues
uv run ruff check --fix src/common tests
```

### Environment Variables

Key variables for development:
- `PYTHONPATH=.` — required when running scripts directly
- `DEBUG=true` — enables DEBUG-level logging
- `LOG__TO_FILE=true` — write logs to `logs/` directory
- `OLLAMA_BASE_URL` or `OLLAMA_API_URL` — Ollama endpoint (default: `http://localhost:11434`)
- `OPENAI_API_KEY` — required for OpenAI models
- `DEEPSEEK_API_KEY` — required for DeepSeek models
- `LMSTUDIO_BASE_URL` — LM Studio endpoint (default: `http://localhost:1234/v1`)
- `OPENROUTER_API_KEY` — required for OpenRouter models
- `FORCE_DEVICE` — override device detection (cuda/mps/cpu)

## Architecture & Core Modules

### Module Organization

```
src/common/
├── unified_llm.py          # Main LLM abstraction layer
├── llm_utils.py            # LLM helpers (factory, message parsing, token counting)
├── LLMProvider.py          # LLM settings & provider enums
├── job_queue.py            # FastAPI-integrated task queue
├── pydantic_helper.py      # Structured output: prompt building & parsing
├── config.py               # Configuration management (YAML-based)
├── logger.py               # Centralized logging with Rich console
├── models.py               # Domain models (ProjectMetadata, etc.)
├── document_utils.py       # Markdown-to-PDF via Pandoc
├── download_utils.py       # Media downloading via yt-dlp
├── utils.py                # General utilities (markdown rendering, finance normalization)
├── ml_utils.py             # Device detection, GPU usage, timing
├── docker_utils.py         # Container path resolution
└── file_utils.py           # Basic filesystem helpers
```

### Key Design Patterns

**1. Unified LLM Interface (`unified_llm.py`)**

The `UnifiedChatModel` class is a LangChain-compatible `BaseChatModel` that abstracts all provider differences:

- **Supported Providers**: ollama, openai, deepseek, llama_cpp, lms (LM Studio), openrouter
- **Lazy Initialization**: Models are cached and reused via `_RAW_LLM_CACHE` to avoid re-initialization
- **Dynamic Parameters**: `invoke()`, `ainvoke()`, `stream()`, `astream()` methods accept optional `thinking` override at call-time for reasoning model toggling
- **Structured Output**: Pass `schema=MyPydanticModel` to any invoke method for automatic JSON schema injection, multi-stage parsing, and LLM repair
- **Artifact Cleanup**: `thinking=False` automatically strips `<think>` and `<tool_call>` tags from model output

**Example:**
```python
from common.unified_llm import UnifiedChatModel
from pydantic import BaseModel

class Result(BaseModel):
    answer: str

model = UnifiedChatModel(provider="ollama", model_name="qwen3:14b")

# Simple invocation
response = model.invoke("What is 2+2?")

# Structured output with schema
result = model.invoke("Extract key info", schema=Result)

# Toggle reasoning at runtime
reasoning = model.invoke(prompt, thinking=True)
non_reasoning = model.invoke(prompt, thinking=False)
```

**2. Job Queue (`job_queue.py`)**

Serialized FIFO queue with optional concurrency, designed for microservices using FastAPI:

- One job runs at a time by default (`max_concurrency=1` for GPU safety)
- Deduplication: identical jobs (same type + request data) return existing job ID
- Lifecycle: `queued` → `running` → `completed`/`failed`
- Automatic cleanup: retains only the 100 most recent completed/failed jobs
- FastAPI integration: `make_router()` exposes `/jobs`, `/jobs/{job_id}`, and DELETE endpoints

**Usage in FastAPI:**
```python
from common.job_queue import JobQueue
from contextlib import asynccontextmanager

job_queue = JobQueue("my_service")

async def run_job(job):
    if job.job_type == "process_file":
        # Long-running work
        return await some_async_work(**job.request)
    raise ValueError(f"Unknown job type: {job.job_type}")

@asynccontextmanager
async def lifespan(app):
    await job_queue.start(run_job)
    yield
    await job_queue.stop()

app = FastAPI(lifespan=lifespan)
app.include_router(job_queue.make_router())
```

**3. Structured Output Parsing (`pydantic_helper.py`)**

Handles LLM output → Pydantic model conversion with multi-stage fallback:

1. **Direct JSON parsing**: Attempt to parse raw LLM output as JSON
2. **Bracket extraction**: Find and extract JSON between first `{` and last `}`
3. **Schema normalization**: Handle schema-echo artifacts (LLM returning the schema instead of data)
4. **LLM repair**: If above fail and an LLM is provided, ask the LLM to fix the JSON

**Usage:**
```python
from common.pydantic_helper import build_structured_prompt, parse_structured_output

prompt = "Extract company info from this text: [...]"
schema_prompt = build_structured_prompt(prompt, CompanyInfo)

response = model.invoke(schema_prompt)
parsed = parse_structured_output(response.content, CompanyInfo, llm=model)
```

**4. Logging (`logger.py`)**

Centralized logger with Rich console formatting:

```python
from common.logger import get_logger

logger = get_logger("module_name", icon="🚀")
logger.info("This is an info message")
logger.error("This is an error", exc_info=True)
```

Set `LOG__TO_FILE=true` to write logs to `logs/app_YYYYMMDD_HHMMSS.log`.

### Configuration Management (`config.py`)

Singleton pattern using YAML-based configuration:

```python
from src.common.config import config

# Access config properties
api_key = config.api_key
llm_settings = config.core_app.chat.llm
```

Config resolution priority:
1. `PACKAGE_CONFIG_PATH` environment variable
2. `packages/{PACKAGE_NAME}/config.yaml`
3. `src/{PACKAGE_NAME}/config.yaml`
4. Root `config.yaml`

### Development Standards

See **GLOBAL_STANDARDS.md** for the full rules on coding style, type safety, logging, error handling, modularity, and workflow.

## Common Workflows

### Adding a New Utility Function

1. Create in appropriate module under `src/common/` (or new file if it spans 30KB)
2. Add type hints and docstring
3. Use `get_logger()` for any logging
4. Add unit tests in `tests/test_<module>.py`
5. Update `src/common/MODULE_CONTEXT.md` if it's a public API

### Extending LLM Support

1. Add provider initialization in `UnifiedChatModel._init_underlying_model()` (unified_llm.py)
2. Add environment variable validation (e.g., API key checks)
3. Add stop tokens if needed in `STOP_TOKENS` dict and `get_tokens_for_model()`
4. Test with `tests/test_unified_llm.py` and `tests/manual/integration_llm.py`

### Debugging LLM Issues

- Enable `DEBUG=true` for verbose logging
- Check `tests/manual/integration_llm.py` for provider-specific manual tests
- Use `model.invoke(prompt, thinking=False)` to strip reasoning artifacts
- For token counting, use `get_token_count(messages)` from `llm_utils.py`

## Testing Notes

- **Unit Tests**: Run with `uv run pytest` (excludes manual tests by default via `pyproject.toml` config)
- **Manual Tests**: Require active LLM providers (Ollama, OpenAI, etc.); use only for integration validation
- **Async Tests**: Use `@pytest.mark.asyncio` decorator; pytest-asyncio is configured in `pyproject.toml`
- **Coverage**: Run `uv run pytest --cov` to check test coverage

## Key External Dependencies

**Core:**
- `langchain-core`, `langchain-community`, `langchain-ollama`, `langchain-openai`, `langchain-deepseek` — LLM abstractions
- `fastapi`, `httpx` — HTTP/FastAPI utilities
- `pydantic>=2.0` — data validation
- `python-dotenv` — environment variable loading
- `rich` — console logging
- `requests` — HTTP client
- `pyyaml` — YAML config parsing
- `markdown2` — markdown rendering

**Optional (extras):**
- `pdf`: `pypandoc` — markdown-to-PDF conversion (requires pandoc system package)
- `media`: `yt-dlp` — media downloading

## References

- **GLOBAL_STANDARDS.md**: Project-wide development rules and conventions
- **GEMINI.md**: Detailed architecture and module responsibilities
- **MODULE_CONTEXT.md** (in `src/common/`): Public API reference and module purposes
- **README.md**: Quick start and feature overview
- **pyproject.toml**: Dependencies, test config, build settings
