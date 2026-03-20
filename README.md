# Memory Service

Fact-centric long-term memory system for conversational AI agents. Extracts structured facts from conversations, stores them with vector embeddings, and retrieves relevant context via hybrid search.

## Architecture

- **API server** (FastAPI) — ingests messages, serves recall queries
- **Background worker** (arq) — async fact extraction, profile synthesis, decay sweeps
- **PostgreSQL + pgvector** — fact storage and vector search
- **Redis** — message buffer, recall cache, arq queue

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Docker & Docker Compose

## Setup

### 1. Start infrastructure

```bash
make up
```

This starts PostgreSQL (with pgvector) on port 5432 and Redis on port 6379.

### 2. Configure environment

```bash
cp env.template .env
```

Edit `.env` and set at minimum:

| Variable | Description |
|---|---|
| `MEM_LLM_API_KEY` | API key for the LLM provider |
| `MEM_EXTRACT_MODEL` | Model for fact extraction (default: `gpt-4.1-mini`) |
| `MEM_ASSEMBLE_MODEL` | Model for context assembly (default: `gpt-4.1-mini`) |
| `MEM_LLM_BASE_URL` | Base URL if using a custom/self-hosted endpoint (optional) |
| `MEM_EMBEDDING_MODEL` | Sentence-transformer model (default: `BAAI/bge-m3`) |
| `MEM_EMBEDDING_DEVICE` | `cpu` or `cuda` (default: `cpu`) |

The database and Redis URLs default to the Docker Compose services and don't need changing for local development.

### 3. Install dependencies

```bash
make install
```

### 4. Run migrations

```bash
make migrate
```

### 5. Start the API server

```bash
make run
```

Server listens on `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 6. Start the background worker

In a separate terminal:

```bash
make worker
```

The worker handles async fact extraction, profile synthesis, and memory decay sweeps.

## API

### `POST /v1/memorize`

Ingest conversation messages. Set `extract_mode` to `"async"` (default, queues extraction) or `"sync"` (blocks until facts are extracted and returned).

```json
{
  "group_id": "conv-123",
  "tenant_id": "default",
  "extract_mode": "async",
  "messages": [
    { "role": "user", "content": "I love hiking.", "speaker_id": "user-1" },
    { "role": "assistant", "content": "That sounds fun!" }
  ]
}
```

### `POST /v1/recall`

Retrieve relevant memories for a query via hybrid search. Set `assemble: true` to get an LLM-assembled context string.

```json
{
  "user_id": "user-1",
  "query": "What does the user enjoy doing outdoors?",
  "top_k": 20,
  "assemble": true,
  "include_profile": true
}
```

### `GET /v1/facts`

List stored facts for a user.

```
GET /v1/facts?user_id=user-1&tenant_id=default&status=active
```

### `PATCH /v1/facts/{fact_id}`

Update a fact's content, tags, importance, or metadata.

### `DELETE /v1/facts/{fact_id}`

Soft-delete a fact.

### `GET /v1/profile/{user_id}`

Retrieve the synthesized user profile (personality traits, summary, etc.).

### `GET /health`

Health check.

## Development

```bash
make test       # run tests
make lint       # check style
make fmt        # auto-format
make typecheck  # pyright
```

## Configuration Reference

All settings use the `MEM_` prefix and can be set in `.env` or as environment variables.

| Variable | Default | Description |
|---|---|---|
| `MEM_DATABASE_URL` | `postgresql://memory:memory@localhost:5432/memory` | PostgreSQL connection |
| `MEM_REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `MEM_EXTRACT_MODEL` | `gpt-4.1-mini` | LLM for fact extraction |
| `MEM_ASSEMBLE_MODEL` | `gpt-4.1-mini` | LLM for context assembly |
| `MEM_LLM_API_KEY` | — | LLM provider API key |
| `MEM_LLM_BASE_URL` | — | Custom LLM base URL |
| `MEM_EMBEDDING_MODEL` | `BAAI/bge-m3` | Sentence-transformer model |
| `MEM_EMBEDDING_DIM` | `1024` | Embedding dimension |
| `MEM_EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `MEM_SESSION_TIME_GAP_MINUTES` | `30` | Gap that splits sessions |
| `MEM_EXTRACT_SCAN_INTERVAL_SECONDS` | `30` | Worker polling interval |
| `MEM_PROFILE_FACT_THRESHOLD` | `10` | Facts needed to trigger profile synthesis |
| `MEM_PROFILE_TIME_THRESHOLD_HOURS` | `24` | Time threshold for profile re-synthesis |
| `MEM_DEFAULT_DECAY_RATE` | `0.01` | Memory importance decay rate |
| `MEM_DECAY_SWEEP_INTERVAL_HOURS` | `6` | How often the decay sweep runs |
| `MEM_PORT` | `8000` | API server port |
| `MEM_LOG_LEVEL` | `info` | Logging level |
