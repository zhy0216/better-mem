# Memory System Architecture

> A fact-centric, retrieval-first long-term memory system for conversational AI agents.

## 1. Design Principles

| Principle                 | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| Retrieval-first           | Storage format is dictated by "how will this be queried", not by the shape of raw data |
| Minimal granularity       | The atomic unit is a **Fact** — a single, self-contained proposition. Higher-order concepts (episodes, profiles) are assembled on demand |
| Forgetting is a feature   | Memory decay, contradiction detection, and version management keep the memory clean and current |
| Async-first               | Memory extraction never blocks the conversation flow; it runs in background workers |
| Separation of concerns    | The memory system does storage and retrieval. Reasoning and prediction belong to the application layer |
| Infrastructure simplicity | If one database can do the job, don't use three              |

## 2. High-Level Architecture

```
                        ┌─────────────────────────────┐
                        │       Client / Agent         │
                        └──────────┬──────────────────-┘
                                   │  HTTP / gRPC
                        ┌──────────▼──────────────────-┐
                        │        API Layer              │
                        │   POST /memorize              │
                        │   POST /recall                │
                        │   GET  /profile/{user_id}     │
                        │   DELETE /facts/{fact_id}      │
                        └──────────┬──────────────────-┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
    ┌─────────▼────────┐ ┌────────▼─────────┐ ┌────────▼─────────┐
    │   Extract Layer   │ │  Retrieve Layer   │ │  Manage Layer     │
    │                   │ │                   │ │                   │
    │ - Fact extraction │ │ - Hybrid search   │ │ - Fact CRUD       │
    │ - Contradiction   │ │ - Context assembly│ │ - Profile refresh │
    │   detection       │ │                   │ │ - Decay sweep     │
    │ - Embedding       │ │                   │ │ - Stats           │
    └─────────┬────────┘ └────────┬─────────┘ └────────┬─────────┘
              │                    │                     │
              └────────────────────┼────────────────────┘
                                   │
                        ┌──────────▼──────────────────-┐
                        │        Store Layer            │
                        │                               │
                        │  PostgreSQL + pgvector         │
                        │  Redis (buffer + cache + queue)│
                        └──────────────────────────────-┘
```

## 3. Data Model

### 3.1 Core Tables (PostgreSQL)

#### `facts` — The single source of truth

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE facts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       TEXT NOT NULL DEFAULT 'default',

    -- ownership
    user_id         TEXT NOT NULL,
    group_id        TEXT,

    -- content
    content         TEXT NOT NULL,
    fact_type       TEXT NOT NULL DEFAULT 'observation',
        -- observation: 观察到的事实 ("张三说他喜欢咖啡")
        -- declaration: 用户直接声明 ("我是后端工程师")
        -- plan:        未来计划 ("张三下周要去北京出差")
        -- preference:  偏好 ("张三喜欢用 Vim")
        -- relation:    关系 ("张三是李四的同事")

    -- temporal
    occurred_at     TIMESTAMPTZ NOT NULL,
    valid_from      TIMESTAMPTZ,
    valid_until     TIMESTAMPTZ,

    -- versioning & contradiction
    superseded_by   UUID REFERENCES facts(id),
    supersedes      UUID REFERENCES facts(id),
    status          TEXT NOT NULL DEFAULT 'active',
        -- active:     当前有效
        -- superseded: 已被新 fact 替代
        -- expired:    已过有效期
        -- deleted:    用户删除

    -- retrieval signals
    importance      FLOAT NOT NULL DEFAULT 0.5,
    access_count    INT NOT NULL DEFAULT 0,
    last_accessed   TIMESTAMPTZ,
    decay_rate      FLOAT NOT NULL DEFAULT 0.01,

    -- source tracing
    source_type     TEXT NOT NULL DEFAULT 'conversation',
        -- conversation, document, api, manual
    source_id       TEXT,
    source_meta     JSONB,

    -- vector
    embedding       vector(1024),

    -- full-text search
    tsv             tsvector GENERATED ALWAYS AS (
                        to_tsvector('simple', content)
                    ) STORED,

    -- extension
    tags            TEXT[] DEFAULT '{}',
    metadata        JSONB DEFAULT '{}',

    -- timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX idx_facts_embedding ON facts
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

CREATE INDEX idx_facts_tsv ON facts USING gin(tsv);

CREATE INDEX idx_facts_user_time ON facts (tenant_id, user_id, occurred_at DESC)
    WHERE status = 'active';

CREATE INDEX idx_facts_group ON facts (tenant_id, group_id, occurred_at DESC)
    WHERE group_id IS NOT NULL AND status = 'active';

CREATE INDEX idx_facts_type ON facts (fact_type)
    WHERE status = 'active';

CREATE INDEX idx_facts_valid_until ON facts (valid_until)
    WHERE valid_until IS NOT NULL AND status = 'active';

CREATE INDEX idx_facts_tags ON facts USING gin(tags);
```

#### `profiles` — Synthesized user understanding

```sql
CREATE TABLE profiles (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    user_id         TEXT NOT NULL,
    scope           TEXT NOT NULL DEFAULT 'global',
        -- global:  跨所有 group 的综合画像
        -- group:   特定 group 内的画像

    group_id        TEXT,

    -- structured profile data
    profile_data    JSONB NOT NULL DEFAULT '{}',
        -- {
        --   "skills": [{"name": "Python", "level": "expert", "evidence_fact_ids": ["..."]}],
        --   "personality": [{"trait": "analytical", "evidence_fact_ids": ["..."]}],
        --   "preferences": [{"key": "editor", "value": "Vim", "evidence_fact_ids": ["..."]}],
        --   "goals": [...],
        --   "relations": [{"target_user_id": "...", "relation": "colleague", "evidence_fact_ids": ["..."]}],
        --   "summary": "..."
        -- }

    -- versioning
    version         INT NOT NULL DEFAULT 1,
    fact_count      INT NOT NULL DEFAULT 0,
    last_fact_id    UUID REFERENCES facts(id),

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE(tenant_id, user_id, scope, group_id)
);
```

#### `message_buffer` — Pending messages awaiting fact extraction

```sql
CREATE TABLE message_buffer (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    group_id        TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    content         JSONB NOT NULL,
        -- {
        --   "role": "user",
        --   "speaker_id": "...",
        --   "speaker_name": "...",
        --   "content": "...",
        --   "timestamp": "..."
        -- }
    status          TEXT NOT NULL DEFAULT 'pending',
        -- pending:    等待处理
        -- processing: 正在提取
        -- consumed:   已提取完成
    batch_id        UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_buffer_group ON message_buffer (tenant_id, group_id, created_at)
    WHERE status = 'pending';
```

### 3.2 Redis Keys

```
buffer:{tenant_id}:{group_id}          -- List, 新消息快速缓冲（写入后异步落库）
lock:extract:{tenant_id}:{group_id}    -- String, 防止同一 group 并发提取
cache:profile:{tenant_id}:{user_id}    -- String, 用户画像缓存 (TTL 5min)
cache:recall:{hash}                    -- String, 检索结果缓存 (TTL 2min)
```

## 4. API Design

### 4.1 Memorize — 写入记忆

```
POST /v1/memorize
```

```json
{
    "tenant_id": "default",
    "group_id": "group_abc",
    "messages": [
        {
            "role": "user",
            "speaker_id": "user_001",
            "speaker_name": "Zhang San",
            "content": "I'm planning a trip to Tokyo next month. Budget is about $3000.",
            "timestamp": "2024-03-14T10:30:00Z"
        },
        {
            "role": "user",
            "speaker_id": "user_002",
            "speaker_name": "Li Si",
            "content": "You should check out the Shibuya district!",
            "timestamp": "2024-03-14T10:31:00Z"
        }
    ],
    "source_type": "conversation",
    "extract_mode": "async"
        // async: 消息入缓冲区，后台提取（默认）
        // sync:  同步提取并返回 facts
}
```

Response (async mode):

```json
{
    "status": "accepted",
    "batch_id": "550e8400-e29b-41d4-a716-446655440000",
    "message_count": 2
}
```

Response (sync mode):

```json
{
    "status": "completed",
    "facts": [
        {
            "id": "...",
            "content": "Zhang San is planning a trip to Tokyo next month (April 2024).",
            "fact_type": "plan",
            "occurred_at": "2024-03-14T10:30:00Z",
            "valid_from": "2024-04-01",
            "valid_until": "2024-04-30",
            "importance": 0.7,
            "tags": ["travel", "tokyo"]
        },
        {
            "id": "...",
            "content": "Zhang San's travel budget for the Tokyo trip is approximately $3000.",
            "fact_type": "plan",
            "occurred_at": "2024-03-14T10:30:00Z",
            "importance": 0.6,
            "tags": ["travel", "budget"]
        },
        {
            "id": "...",
            "content": "Li Si recommended Zhang San to visit the Shibuya district in Tokyo.",
            "fact_type": "observation",
            "occurred_at": "2024-03-14T10:31:00Z",
            "importance": 0.5,
            "tags": ["travel", "recommendation"]
        }
    ]
}
```

### 4.2 Recall — 检索记忆

```
POST /v1/recall
```

```json
{
    "tenant_id": "default",
    "user_id": "user_001",
    "query": "What are Zhang San's travel plans?",
    "top_k": 20,
    "filters": {
        "group_id": null,
        "fact_types": null,
        "time_range": {
            "start": "2024-01-01",
            "end": null
        },
        "tags": null,
        "status": ["active"]
    },
    "include_profile": true,
    "assemble": true
        // true:  返回 LLM 组装后的结构化上下文
        // false: 返回原始 facts 列表
}
```

Response (assemble=true):

```json
{
    "context": "Zhang San is planning a trip to Tokyo in April 2024 with a budget of approximately $3000. Li Si has recommended visiting the Shibuya district.",
    "facts": [
        {
            "id": "...",
            "content": "Zhang San is planning a trip to Tokyo next month (April 2024).",
            "score": 0.92,
            "source": "vector"
        },
        {
            "id": "...",
            "content": "Zhang San's travel budget for the Tokyo trip is approximately $3000.",
            "score": 0.87,
            "source": "keyword"
        }
    ],
    "profile_snippet": {
        "summary": "Zhang San: backend engineer, likes travel, budget-conscious",
        "relevant_traits": ["travel enthusiast"]
    },
    "total_candidates": 3,
    "search_time_ms": 45
}
```

Response (assemble=false):

```json
{
    "facts": [
        {
            "id": "...",
            "content": "...",
            "fact_type": "plan",
            "occurred_at": "...",
            "score": 0.92,
            "source": "vector",
            "importance": 0.7,
            "metadata": {}
        }
    ],
    "total_candidates": 3,
    "search_time_ms": 32
}
```

### 4.3 Profile — 画像管理

```
GET /v1/profile/{user_id}?scope=global
```

```json
{
    "user_id": "user_001",
    "scope": "global",
    "version": 5,
    "fact_count": 42,
    "profile_data": {
        "skills": [
            {"name": "Python", "level": "expert", "evidence_count": 3}
        ],
        "personality": [
            {"trait": "analytical", "evidence_count": 2}
        ],
        "preferences": [
            {"key": "editor", "value": "Vim", "evidence_count": 1}
        ],
        "goals": [
            {"description": "Learn Rust", "evidence_count": 1}
        ],
        "summary": "Backend engineer with expertise in Python, analytical personality, prefers Vim."
    },
    "updated_at": "2024-03-14T12:00:00Z"
}
```

### 4.4 Fact Management

```
DELETE /v1/facts/{fact_id}                    -- 软删除
PATCH  /v1/facts/{fact_id}                    -- 更新 metadata/tags
GET    /v1/facts?user_id=...&type=...&limit=  -- 列出 facts
POST   /v1/facts/{fact_id}/supersede          -- 手动标记替代关系
```

## 5. Extract Pipeline

### 5.1 Flow

```
Messages arrive
       │
       ▼
  ┌─────────────┐     ┌──────────────────┐
  │ API Handler  │────▶│ Redis Buffer     │   (instant, < 5ms)
  │ (sync resp)  │     │ RPUSH to list    │
  └─────────────┘     └───────┬──────────┘
                              │ arq worker picks up
                              ▼
                    ┌──────────────────┐
                    │ Flush to DB      │   message_buffer table
                    │ (batch insert)   │
                    └───────┬──────────┘
                            │
                            ▼
                    ┌──────────────────┐
                    │ Session Detect   │   按 group_id 分组，
                    │                  │   检查是否有足够消息
                    └───────┬──────────┘
                            │ 触发条件满足
                            ▼
                    ┌──────────────────┐
                    │ Fact Extraction  │   LLM 调用
                    │ (core logic)     │
                    └───────┬──────────┘
                            │
                    ┌───────┴───────────┐
                    │                   │
                    ▼                   ▼
          ┌──────────────┐    ┌──────────────────┐
          │ Embedding    │    │ Contradiction    │
          │ (local model)│    │ Detection        │
          └──────┬───────┘    └───────┬──────────┘
                 │                    │
                 └────────┬───────────┘
                          ▼
                 ┌──────────────────┐
                 │ Store Facts      │   INSERT into facts table
                 │ (with tx)        │
                 └───────┬──────────┘
                         │
                         ▼
                 ┌──────────────────┐
                 │ Profile Update   │   条件触发
                 │ (if threshold)   │
                 └──────────────────┘
```

### 5.2 Session Detection — When to extract

不使用 LLM 做边界检测。使用基于规则的触发条件：

```python
class SessionDetector:
    """Determines when accumulated messages should be processed."""

    # Configuration
    TIME_GAP_THRESHOLD = timedelta(minutes=30)    # 30 分钟无新消息
    MAX_MESSAGES = 50                              # 累积超过 50 条
    MAX_TOKENS = 8192                              # 累积超过 8K tokens
    MIN_MESSAGES = 2                               # 至少 2 条消息才处理

    async def should_extract(self, group_id: str) -> bool:
        """Check if extraction should be triggered."""
        pending = await self.get_pending_messages(group_id)

        if len(pending) < self.MIN_MESSAGES:
            return False

        # Trigger 1: 时间间隔超过阈值（对话可能已结束）
        last_msg_time = pending[-1].timestamp
        if now() - last_msg_time > self.TIME_GAP_THRESHOLD:
            return True

        # Trigger 2: 消息数量超限
        if len(pending) >= self.MAX_MESSAGES:
            return True

        # Trigger 3: Token 数量超限
        total_tokens = sum(count_tokens(m.content) for m in pending)
        if total_tokens >= self.MAX_TOKENS:
            return True

        return False
```

触发检查由 **arq 定时任务**每 30 秒扫描一次活跃的 group_id。

### 5.3 Fact Extraction — LLM Prompt

```python
FACT_EXTRACTION_PROMPT = """You are a fact extraction engine. Extract atomic facts from the conversation below.

Rules:
1. Each fact must be a single, self-contained proposition
2. Each fact must include WHO did/said WHAT and WHEN
3. Resolve pronouns to specific names
4. Resolve relative times to absolute dates based on the conversation timestamp
5. Classify each fact into one of: observation, declaration, plan, preference, relation
6. Estimate importance (0.0 - 1.0): routine small talk = 0.1, specific plans = 0.7, life events = 0.9
7. For plans/events with a future time scope, include valid_from and valid_until
8. Filter out greetings, acknowledgments, and low-information content
9. For each fact, add relevant tags (2-5 tags)

Conversation timestamp: {timestamp}
Participants: {participants}

Conversation:
{conversation}

Return a JSON object:
{{
    "facts": [
        {{
            "content": "Zhang San plans to visit Tokyo in April 2024.",
            "fact_type": "plan",
            "importance": 0.7,
            "valid_from": "2024-04-01",
            "valid_until": "2024-04-30",
            "tags": ["travel", "tokyo", "plan"]
        }}
    ]
}}"""
```

调用方式：

```python
async def extract_facts(messages: list[dict], timestamp: str) -> list[FactCreate]:
    conversation = format_messages(messages)
    participants = extract_participants(messages)

    response = await litellm.acompletion(
        model=settings.EXTRACT_MODEL,          # e.g. "gpt-4.1-mini"
        messages=[
            {"role": "system", "content": FACT_EXTRACTION_PROMPT.format(
                timestamp=timestamp,
                participants=", ".join(participants),
                conversation=conversation,
            )},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    data = json.loads(response.choices[0].message.content)
    return [FactCreate(**f) for f in data["facts"]]
```

### 5.4 Contradiction Detection

在新 facts 写入前，检测是否与已有 facts 矛盾：

```python
async def detect_contradictions(
    new_facts: list[FactCreate],
    user_id: str,
) -> list[ContradictionPair]:
    """Find existing facts that may be contradicted by new facts."""

    contradictions = []
    for new_fact in new_facts:
        # Step 1: 向量检索相似的已有 facts
        similar_facts = await fact_store.search_similar(
            embedding=new_fact.embedding,
            user_id=user_id,
            top_k=5,
            score_threshold=0.85,   # 高相似度才可能矛盾
        )

        if not similar_facts:
            continue

        # Step 2: LLM 判断是否矛盾
        result = await llm_check_contradiction(new_fact, similar_facts)

        for old_fact, relation in result:
            if relation == "contradicts":
                contradictions.append(ContradictionPair(
                    old_fact_id=old_fact.id,
                    new_fact=new_fact,
                    relation="supersedes",
                ))
            elif relation == "updates":
                contradictions.append(ContradictionPair(
                    old_fact_id=old_fact.id,
                    new_fact=new_fact,
                    relation="supersedes",
                ))

    return contradictions
```

矛盾处理在一个事务中完成：

```python
async def store_facts_with_contradictions(
    new_facts: list[FactCreate],
    contradictions: list[ContradictionPair],
):
    async with db.transaction():
        # 1. 插入新 facts
        saved = await fact_store.insert_batch(new_facts)

        # 2. 标记被替代的旧 facts
        for contradiction in contradictions:
            new_id = saved[contradiction.new_fact_index].id
            await fact_store.mark_superseded(
                old_fact_id=contradiction.old_fact_id,
                superseded_by=new_id,
            )
```

### 5.5 Embedding

使用本地模型，不调用外部 API：

```python
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")
        # 1024 dims, multilingual (zh + en)

    async def embed(self, text: str) -> list[float]:
        # sentence-transformers is sync, run in thread pool
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(
            None, self.model.encode, text
        )
        return vec.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        vecs = await loop.run_in_executor(
            None, self.model.encode, texts
        )
        return [v.tolist() for v in vecs]
```

## 6. Retrieve Pipeline

### 6.1 Two-Stage Retrieval

```
Query arrives
       │
       ▼
  ┌─────────────────────────────────┐
  │ Stage 1: Recall (high recall)   │
  │                                 │
  │  ┌──────────┐  ┌─────────────┐  │
  │  │ Vector   │  │ Full-text   │  │
  │  │ Search   │  │ Search      │  │
  │  │ top_k=50 │  │ top_k=50    │  │
  │  └────┬─────┘  └─────┬───────┘  │
  │       └───────┬───────┘          │
  │               ▼                  │
  │      Merge + Deduplicate         │
  │      (~80 candidates)            │
  └───────────────┬─────────────────┘
                  │
                  ▼
  ┌─────────────────────────────────┐
  │ Stage 2: Assemble (high prec)   │
  │                                 │
  │  LLM selects relevant facts     │
  │  + organizes into context       │
  │  + injects profile if requested │
  └───────────────┬─────────────────┘
                  │
                  ▼
            Response
```

### 6.2 Stage 1: Hybrid Search

```python
async def hybrid_search(
    query: str,
    user_id: str,
    top_k: int = 50,
    filters: SearchFilters | None = None,
) -> list[ScoredFact]:
    """Run vector + keyword search in parallel, merge results."""

    query_embedding = await embedding_service.embed(query)

    # 两路并行搜索
    vector_results, keyword_results = await asyncio.gather(
        fact_store.vector_search(
            embedding=query_embedding,
            user_id=user_id,
            top_k=top_k,
            filters=filters,
        ),
        fact_store.keyword_search(
            query=query,
            user_id=user_id,
            top_k=top_k,
            filters=filters,
        ),
    )

    # 合并去重，使用 RRF (Reciprocal Rank Fusion)
    return reciprocal_rank_fusion(
        vector_results,
        keyword_results,
        k=60,
    )
```

SQL for vector search:

```sql
SELECT id, content, fact_type, occurred_at, importance, metadata,
       1 - (embedding <=> $1::vector) AS score
FROM facts
WHERE tenant_id = $2
  AND user_id = $3
  AND status = 'active'
  AND ($4::text IS NULL OR group_id = $4)
  AND ($5::timestamptz IS NULL OR occurred_at >= $5)
  AND ($6::timestamptz IS NULL OR occurred_at <= $6)
  AND ($7::text[] IS NULL OR fact_type = ANY($7))
ORDER BY embedding <=> $1::vector
LIMIT $8;
```

SQL for keyword search:

```sql
SELECT id, content, fact_type, occurred_at, importance, metadata,
       ts_rank(tsv, plainto_tsquery('simple', $1)) AS score
FROM facts
WHERE tenant_id = $2
  AND user_id = $3
  AND status = 'active'
  AND tsv @@ plainto_tsquery('simple', $1)
  AND ($4::text IS NULL OR group_id = $4)
  AND ($5::timestamptz IS NULL OR occurred_at >= $5)
  AND ($6::timestamptz IS NULL OR occurred_at <= $6)
ORDER BY score DESC
LIMIT $7;
```

### 6.3 Stage 2: Context Assembly

```python
ASSEMBLE_PROMPT = """Given the user's query and a set of retrieved facts, do three things:
1. Select only the facts that are genuinely relevant to the query
2. Organize them into a coherent, chronological context paragraph
3. Note any information gaps

Query: {query}

Retrieved facts:
{facts}

User profile summary:
{profile_summary}

Return JSON:
{{
    "context": "A coherent paragraph summarizing relevant information...",
    "selected_fact_ids": ["id1", "id2"],
    "confidence": 0.85,
    "information_gaps": ["No information about accommodation preferences"]
}}"""

async def assemble_context(
    query: str,
    candidates: list[ScoredFact],
    profile: Profile | None = None,
) -> AssembledContext:
    facts_text = "\n".join(
        f"[{f.id}] ({f.occurred_at}) {f.content}" for f in candidates
    )
    profile_summary = profile.summary if profile else "No profile available"

    response = await litellm.acompletion(
        model=settings.ASSEMBLE_MODEL,         # e.g. "gpt-4.1-mini"
        messages=[
            {"role": "system", "content": ASSEMBLE_PROMPT.format(
                query=query,
                facts=facts_text,
                profile_summary=profile_summary,
            )},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    data = json.loads(response.choices[0].message.content)
    return AssembledContext(**data)
```

## 7. Profile Synthesis

### 7.1 When to update

Profile 更新由以下条件触发（不是每次 fact 提取都更新）：

```python
class ProfileUpdateTrigger:
    FACT_COUNT_THRESHOLD = 10      # 每新增 10 条 facts 更新一次
    TIME_THRESHOLD = timedelta(hours=24)  # 或距上次更新超过 24 小时
    FORCE_ON_DECLARATION = True    # declaration 类型的 fact 立即触发

    async def should_update(self, user_id: str, new_facts: list[Fact]) -> bool:
        # 条件 1: 用户直接声明了自己的特征
        if self.FORCE_ON_DECLARATION:
            if any(f.fact_type == "declaration" for f in new_facts):
                return True

        # 条件 2: 新 facts 累积超过阈值
        profile = await profile_store.get(user_id)
        if profile is None:
            return True
        facts_since = await fact_store.count_since(user_id, profile.updated_at)
        if facts_since >= self.FACT_COUNT_THRESHOLD:
            return True

        # 条件 3: 时间超过阈值
        if now() - profile.updated_at > self.TIME_THRESHOLD:
            return True

        return False
```

### 7.2 Profile extraction prompt

```python
PROFILE_SYNTHESIS_PROMPT = """Analyze the following facts about a user and update their profile.

Existing profile:
{existing_profile}

New facts (since last update):
{new_facts}

Instructions:
1. Merge new information with the existing profile
2. If new facts contradict existing profile entries, prefer the newer information
3. Each profile attribute must reference the fact IDs that support it
4. Profile attributes:
   - skills: [{name, level (beginner/intermediate/expert), evidence_fact_ids}]
   - personality: [{trait, evidence_fact_ids}]
   - preferences: [{key, value, evidence_fact_ids}]
   - goals: [{description, status (active/completed/abandoned), evidence_fact_ids}]
   - relations: [{target_user_id, target_name, relation, evidence_fact_ids}]
   - summary: One paragraph summary of the user

Return the complete updated profile as JSON."""
```

## 8. Memory Decay

### 8.1 Decay Formula

```python
def compute_relevance_score(fact: Fact, now: datetime) -> float:
    """Compute current relevance score for a fact.

    Score = importance * recency_factor * access_boost

    - importance: 初始重要性 (0-1), 由 LLM 在提取时设定
    - recency_factor: 时间衰减, 指数衰减
    - access_boost: 每次被检索命中时提升
    """
    # Time decay (exponential)
    age_days = (now - fact.occurred_at).total_seconds() / 86400
    recency_factor = math.exp(-fact.decay_rate * age_days)

    # Access boost (logarithmic, diminishing returns)
    access_boost = 1.0 + 0.1 * math.log1p(fact.access_count)

    return fact.importance * recency_factor * access_boost
```

### 8.2 Decay sweep (background job)

```python
async def decay_sweep():
    """Periodic job to expire facts that are no longer relevant.

    Runs every 6 hours via arq cron.
    Does NOT delete facts — just marks them as expired.
    """
    # 1. 过有效期的 facts
    await db.execute("""
        UPDATE facts
        SET status = 'expired', updated_at = now()
        WHERE status = 'active'
          AND valid_until IS NOT NULL
          AND valid_until < now()
    """)

    # 2. 低于阈值的 facts (可选, 保守策略不启用)
    # 这一步可以配置为只标记不执行, 让检索时用 score 过滤
```

### 8.3 Access tracking

每次检索命中时更新：

```python
async def track_access(fact_ids: list[UUID]):
    await db.execute("""
        UPDATE facts
        SET access_count = access_count + 1,
            last_accessed = now(),
            updated_at = now()
        WHERE id = ANY($1)
    """, fact_ids)
```

## 9. Project Structure

```
memory-service/
├── src/
│   ├── main.py                     # FastAPI app, lifespan, middleware
│   ├── config.py                   # pydantic-settings, env vars
│   ├── deps.py                     # FastAPI dependencies (get_db, get_services)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fact.py                 # Fact, FactCreate, FactUpdate, ScoredFact
│   │   ├── profile.py             # Profile, ProfileData
│   │   ├── message.py             # Message, MessageBuffer
│   │   └── api.py                 # Request/Response DTOs
│   │
│   ├── store/
│   │   ├── __init__.py
│   │   ├── database.py            # asyncpg pool management
│   │   ├── fact_store.py          # Fact CRUD + search queries
│   │   ├── profile_store.py       # Profile CRUD
│   │   ├── buffer_store.py        # Message buffer operations
│   │   └── cache.py               # Redis cache wrapper
│   │
│   ├── extract/
│   │   ├── __init__.py
│   │   ├── session_detector.py    # When to trigger extraction
│   │   ├── fact_extractor.py      # LLM-based fact extraction
│   │   ├── contradiction.py       # Contradiction detection + resolution
│   │   ├── profile_synthesizer.py # Profile update logic
│   │   └── prompts.py             # All extraction prompts
│   │
│   ├── retrieve/
│   │   ├── __init__.py
│   │   ├── searcher.py            # Hybrid search (vector + keyword)
│   │   ├── assembler.py           # LLM context assembly
│   │   ├── ranker.py              # RRF + decay scoring
│   │   └── prompts.py             # Assembly prompts
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding.py           # Local embedding model (bge-m3)
│   │   ├── llm.py                 # litellm wrapper, retry logic
│   │   └── tokenizer.py           # Token counting
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── memorize.py            # POST /v1/memorize
│   │   ├── recall.py              # POST /v1/recall
│   │   ├── profile.py             # GET /v1/profile/{user_id}
│   │   ├── facts.py               # Fact CRUD endpoints
│   │   └── health.py              # Health check
│   │
│   ├── worker/
│   │   ├── __init__.py
│   │   ├── tasks.py               # arq task definitions
│   │   ├── scheduler.py           # Periodic jobs (decay sweep, profile refresh)
│   │   └── settings.py            # arq worker settings
│   │
│   └── migrations/
│       ├── 001_initial.sql         # Tables, indexes, extensions
│       └── migrate.py             # Simple migration runner
│
├── tests/
│   ├── conftest.py                # Fixtures (test db, mock llm)
│   ├── test_extract/
│   │   ├── test_fact_extractor.py
│   │   ├── test_contradiction.py
│   │   └── test_session_detector.py
│   ├── test_retrieve/
│   │   ├── test_searcher.py
│   │   └── test_assembler.py
│   ├── test_store/
│   │   └── test_fact_store.py
│   └── test_api/
│       ├── test_memorize.py
│       └── test_recall.py
│
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml              # PostgreSQL + Redis
├── env.template
└── Makefile
```

## 10. Tech Stack Summary

| Component             | Choice                  | Version         |
| --------------------- | ----------------------- | --------------- |
| Language              | Python                  | 3.12+           |
| Web framework         | FastAPI                 | 0.115+          |
| ASGI server           | Uvicorn                 | 0.30+           |
| Database              | PostgreSQL              | 16+             |
| Vector extension      | pgvector                | 0.7+            |
| DB driver             | asyncpg                 | 0.30+           |
| Cache / Queue backend | Redis                   | 7+              |
| Task queue            | arq                     | 0.26+           |
| LLM client            | litellm                 | 1.50+           |
| Embedding model       | BAAI/bge-m3             | local, 1024-dim |
| Embedding runtime     | sentence-transformers   | 3.0+            |
| Tokenizer             | tiktoken                | 0.7+            |
| Validation            | Pydantic                | 2.x             |
| Logging               | structlog               | 24.x            |
| Testing               | pytest + pytest-asyncio | 8.x             |
| Linting               | ruff                    | 0.6+            |
| Type checking         | pyright                 | 1.1+            |
| Package manager       | uv                      | 0.4+            |
| Container             | Docker + docker-compose | -               |

## 11. Configuration

```python
# src/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://user:pass@localhost:5432/memory"
    DB_POOL_MIN: int = 5
    DB_POOL_MAX: int = 20

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # LLM
    EXTRACT_MODEL: str = "gpt-4.1-mini"
    ASSEMBLE_MODEL: str = "gpt-4.1-mini"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str | None = None
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_RETRIES: int = 3

    # Embedding
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_DEVICE: str = "cpu"    # or "cuda"

    # Extract
    SESSION_TIME_GAP_MINUTES: int = 30
    SESSION_MAX_MESSAGES: int = 50
    SESSION_MAX_TOKENS: int = 8192
    SESSION_MIN_MESSAGES: int = 2
    EXTRACT_SCAN_INTERVAL_SECONDS: int = 30

    # Profile
    PROFILE_FACT_THRESHOLD: int = 10
    PROFILE_TIME_THRESHOLD_HOURS: int = 24
    PROFILE_FORCE_ON_DECLARATION: bool = True

    # Decay
    DEFAULT_DECAY_RATE: float = 0.01
    DECAY_SWEEP_INTERVAL_HOURS: int = 6

    # Multi-tenancy
    DEFAULT_TENANT_ID: str = "default"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"

    model_config = {"env_prefix": "MEM_", "env_file": ".env"}
```

## 12. docker-compose.yml

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: memory
      POSTGRES_PASSWORD: memory
      POSTGRES_DB: memory
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U memory"]
      interval: 5s
      timeout: 3s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redisdata:/data

volumes:
  pgdata:
  redisdata:
```

## 13. Error Handling Strategy

```
Critical path (must succeed):
  - Message buffer write       → retry 3x, then return 500
  - Fact store write           → retry 3x with backoff, dead-letter on failure

Best-effort path (can degrade gracefully):
  - Contradiction detection    → skip on failure, log warning
  - Profile update             → skip on failure, retry next cycle
  - Context assembly           → fall back to raw facts list
  - Cache operations           → skip on failure, serve from DB

Never retry:
  - Validation errors (400)
  - Auth errors (401/403)
  - Not found (404)
```

## 14. Key Design Decisions Log

| Decision               | Choice                         | Alternatives considered           | Rationale                                                    |
| ---------------------- | ------------------------------ | --------------------------------- | ------------------------------------------------------------ |
| Atomic unit            | Fact (single proposition)      | Episode / MemCell / Message       | Facts are the smallest retrievable unit; episodes can be assembled from facts but not vice versa |
| Boundary detection     | Rule-based (time gap + limits) | LLM-based                         | Avoids per-message LLM cost; edge cases are tolerable since facts are independent |
| Storage                | PostgreSQL + pgvector          | MongoDB + Milvus + ES             | Single DB simplifies ops, transactions, consistency; pgvector is sufficient for < 100M vectors |
| Embedding              | Local bge-m3                   | API-based (OpenAI, etc.)          | Cost and latency; embedding is called on every fact          |
| Contradiction handling | Supersede chain (soft)         | Hard delete / Ignore              | Preserves history; both old and new facts are queryable      |
| Profile update         | Threshold-triggered            | Every extraction / Scheduled      | Balances freshness and cost; declarations trigger immediate update |
| Foresight/prediction   | Not in memory layer            | In memory layer (like EverMemOS)  | Predictions belong to the application layer; memory stores facts, not inferences |
| Context assembly       | LLM at query time              | Pre-computed summaries            | More flexible; different queries need different context shapes |
| Multi-tenancy          | tenant_id column               | Schema-per-tenant / DB-per-tenant | Simplest; sufficient until hundreds of tenants               |


