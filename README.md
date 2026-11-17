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

## Usage

### Unified LLM interface

```python
from common.unified_llm import UnifiedChatModel
from pydantic import BaseModel

class Result(BaseModel):
    answer: str

model = UnifiedChatModel(provider="ollama", model_name="qwen3:14b")

response = model.invoke("What is 2+2?")                 # simple
result = model.invoke("Extract key info", schema=Result)  # structured output
reasoning = model.invoke(prompt, thinking=True)          # toggle reasoning per call
```

### Job queue (FastAPI)

```python
from common.job_queue import JobQueue
from contextlib import asynccontextmanager

job_queue = JobQueue("my_service")

async def run_job(job):
    if job.job_type == "process_file":
        return await some_async_work(**job.request)
    raise ValueError(f"Unknown job type: {job.job_type}")

@asynccontextmanager
async def lifespan(app):
    await job_queue.start(run_job)
    yield
    await job_queue.stop()

app = FastAPI(lifespan=lifespan)
app.include_router(job_queue.make_router())   # exposes /jobs, /jobs/{id}, DELETE
```

### Structured output parsing

```python
from common.pydantic_helper import build_structured_prompt, parse_structured_output

schema_prompt = build_structured_prompt("Extract company info: [...]", CompanyInfo)
response = model.invoke(schema_prompt)
parsed = parse_structured_output(response.content, CompanyInfo, llm=model)  # repairs on failure
```

### Logging

```python
from common.logger import get_logger

logger = get_logger("module_name", icon="🚀")
logger.info("info message")
logger.error("boom", exc_info=True)
```

Set `LOG__TO_FILE=true` to write to `logs/app_YYYYMMDD_HHMMSS.log`.

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

Conventions: always use `uv` for dependency management, type hints on all function signatures, and `get_logger()` from `logger.py` for logging. See `CLAUDE.md` for the full convention list.

## License

(Specify license here or refer to a LICENSE file)
