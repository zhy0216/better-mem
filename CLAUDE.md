# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fact-centric long-term memory service for conversational AI agents. Python 3.12+, FastAPI, PostgreSQL (pgvector), Redis, arq worker.

## Commands

```bash
make install      # Install deps (uv pip install -e ".[dev]")
make up           # Start Postgres + Redis via docker-compose
make down         # Stop Docker services
make migrate      # Run DB migrations (python -m src.migrations.migrate)
make run          # Start API server (uvicorn, port 8000, auto-reload)
make worker       # Start arq background worker
make test         # Run tests (pytest tests/ -v)
make lint         # Check code (ruff check + format --check)
make fmt          # Fix code (ruff format + check --fix)
make typecheck    # Run pyright
```

Run a single test: `pytest tests/test_extract/test_contradiction.py -v`

## Architecture

Two main pipelines share a common store layer:

**Extract pipeline** (write path): Messages → Redis buffer → flush to DB → session detection (rule-based, not LLM) → LLM fact extraction → local embedding (BAAI/bge-m3, 1024-dim) → contradiction detection → store facts → conditional profile synthesis.

**Retrieve pipeline** (read path): Query → local embedding → parallel vector + keyword search → RRF merge → LLM context assembly → response. Falls back to raw facts if assembly fails.

### Source layout (`src/`)

- `main.py` / `config.py` / `deps.py` — App init, pydantic-settings (env prefix `MEM_`), FastAPI DI
- `models/` — Pydantic models: `fact`, `profile`, `message`, `api` (request/response DTOs)
- `store/` — Data layer: `database` (asyncpg pool), `fact_store`, `profile_store`, `buffer_store`, `cache` (Redis)
- `api/` — Routers: `memorize` (POST /v1/memorize), `recall` (POST /v1/recall), `profile`, `facts` CRUD, `health`
- `extract/` — `fact_extractor`, `session_detector`, `contradiction`, `profile_synthesizer`, `prompts`
- `retrieve/` — `searcher` (hybrid search), `assembler` (LLM context), `ranker` (RRF), `prompts`
- `services/` — `embedding` (sentence-transformers), `llm` (litellm wrapper), `tokenizer`
- `worker/` — arq tasks (`process_group`, `scan_groups`, `decay_sweep`), scheduler, settings
- `migrations/` — SQL files + simple migration runner

### Key design decisions

- **Atomic unit is a Fact** (single proposition), not an episode — smallest queryable unit
- **Session detection is rule-based** (time gap, message count, token count thresholds) to avoid per-message LLM cost
- **Contradiction handling uses soft-delete chains** (superseded_by/supersedes) preserving history
- **Embeddings are local** (bge-m3) for cost/latency; LLM calls (litellm) used only for extraction and assembly
- **Multi-tenancy is column-based** (tenant_id on all tables)
- **Error strategy**: critical path retries with backoff; best-effort paths (contradiction, profile) skip on failure

## Code style

- Ruff: line-length 100, rules E/F/I/UP, target py312
- Pyright: basic mode
- All I/O is async (asyncpg, aioredis, litellm)
- Tests use pytest-asyncio with `asyncio_mode = "auto"`

## Infrastructure

- PostgreSQL 16 with pgvector extension (docker: `pgvector/pgvector:pg16`)
- Redis 7 (docker: `redis:7-alpine`)
- Config via `.env` file (copy from `env.template`)
