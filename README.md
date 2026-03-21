# Better-Mem

Proposition-centric, belief-aware long-term memory system for conversational AI agents. Extracts structured propositions from conversations, tracks belief confidence with evidence, and retrieves relevant context via hybrid search.

## Core Concepts

- **Proposition** — the atomic content unit: a single, self-contained statement (e.g. "Zhang San plans to visit Tokyo in April 2024")
- **Belief** — the system's current confidence in a proposition, updated as new evidence arrives (1:1 with proposition)
- **Evidence** — traceable supporting or contradicting signals attached to a proposition
- **Semantic Key** — a slot identifier (e.g. `residence`, `favorite_editor`) enabling same-slot competition and conflict resolution

## Architecture

- **API server** (FastAPI) — ingests messages, serves recall queries, proposition & belief management
- **Background worker** (arq) — async proposition extraction, belief updates, profile synthesis, decay sweeps
- **PostgreSQL + pgvector** — proposition/belief/evidence storage and vector search
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
| `MEM_EXTRACT_MODEL` | Model for proposition extraction (default: `gpt-4.1-mini`) |
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

The worker handles async proposition extraction, belief updates, profile synthesis, and decay sweeps.

## API

### `POST /v1/memorize`

Ingest conversation messages. Set `extract_mode` to `"async"` (default, queues extraction) or `"sync"` (blocks until propositions are extracted and returned).

```json
{
  "group_id": "conv-123",
  "user_id": "user-1",
  "tenant_id": "default",
  "extract_mode": "async",
  "messages": [
    { "role": "user", "content": "I love hiking.", "speaker_id": "user-1" },
    { "role": "assistant", "content": "That sounds fun!" }
  ]
}
```

### `POST /v1/recall`

Retrieve relevant memories via hybrid search. Propositions are ranked by a weighted combination of semantic relevance, belief confidence, utility importance, freshness, and access frequency. Set `assemble: true` to get an LLM-assembled context string.

```json
{
  "user_id": "user-1",
  "query": "What does the user enjoy doing outdoors?",
  "top_k": 20,
  "assemble": true,
  "include_profile": true
}
```

### `GET /v1/propositions`

List stored propositions for a user.

```
GET /v1/propositions?user_id=user-1&tenant_id=default&status=active
```

### `DELETE /v1/propositions/{proposition_id}`

Soft-delete a proposition (marks its belief as deprecated).

### `POST /v1/propositions/{proposition_id}/evidence`

Inject evidence for a proposition. Belief confidence is recalculated automatically.

```json
{
  "direction": "support",
  "evidence_type": "utterance",
  "weight": 1.0,
  "quoted_text": "I really enjoy hiking in the mountains."
}
```

### `GET /v1/propositions/{proposition_id}/evidence`

Retrieve evidence trail for a proposition (audit & explainability).

### `PATCH /v1/beliefs/{proposition_id}`

Manually adjust a proposition's belief (confidence, importance, status).

### `GET /v1/profile/{user_id}`

Retrieve the synthesized user profile (personality traits, summary, etc.). Profile is generated from high-confidence, high-utility propositions.

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
| `MEM_EXTRACT_MODEL` | `gpt-4.1-mini` | LLM for proposition extraction |
| `MEM_ASSEMBLE_MODEL` | `gpt-4.1-mini` | LLM for context assembly |
| `MEM_LLM_API_KEY` | — | LLM provider API key |
| `MEM_LLM_BASE_URL` | — | Custom LLM base URL |
| `MEM_EMBEDDING_MODEL` | `BAAI/bge-m3` | Sentence-transformer model |
| `MEM_EMBEDDING_DIM` | `1024` | Embedding dimension |
| `MEM_EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `MEM_SESSION_TIME_GAP_MINUTES` | `30` | Gap that splits sessions |
| `MEM_EXTRACT_SCAN_INTERVAL_SECONDS` | `30` | Worker polling interval |
| `MEM_PROFILE_FACT_THRESHOLD` | `10` | Propositions needed to trigger profile synthesis |
| `MEM_PROFILE_TIME_THRESHOLD_HOURS` | `24` | Time threshold for profile re-synthesis |
| `MEM_PROFILE_FORCE_ON_DECLARATION` | `true` | Immediately re-synthesize profile on user declarations |
| `MEM_DEFAULT_DECAY_RATE` | `0.01` | Default belief freshness decay rate |
| `MEM_DECAY_SWEEP_INTERVAL_HOURS` | `6` | How often the decay sweep runs |
| `MEM_PORT` | `8000` | API server port |
| `MEM_LOG_LEVEL` | `info` | Logging level |
