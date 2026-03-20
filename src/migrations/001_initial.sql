CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS facts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       TEXT NOT NULL DEFAULT 'default',

    user_id         TEXT NOT NULL,
    group_id        TEXT,

    content         TEXT NOT NULL,
    fact_type       TEXT NOT NULL DEFAULT 'observation',

    occurred_at     TIMESTAMPTZ NOT NULL,
    valid_from      TIMESTAMPTZ,
    valid_until     TIMESTAMPTZ,

    superseded_by   UUID REFERENCES facts(id),
    supersedes      UUID REFERENCES facts(id),
    status          TEXT NOT NULL DEFAULT 'active',

    importance      FLOAT NOT NULL DEFAULT 0.5,
    access_count    INT NOT NULL DEFAULT 0,
    last_accessed   TIMESTAMPTZ,
    decay_rate      FLOAT NOT NULL DEFAULT 0.01,

    source_type     TEXT NOT NULL DEFAULT 'conversation',
    source_id       TEXT,
    source_meta     JSONB,

    embedding       vector(1024),

    tsv             tsvector GENERATED ALWAYS AS (
                        to_tsvector('simple', content)
                    ) STORED,

    tags            TEXT[] DEFAULT '{}',
    metadata        JSONB DEFAULT '{}',

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_facts_embedding ON facts
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS idx_facts_tsv ON facts USING gin(tsv);

CREATE INDEX IF NOT EXISTS idx_facts_user_time ON facts (tenant_id, user_id, occurred_at DESC)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_facts_group ON facts (tenant_id, group_id, occurred_at DESC)
    WHERE group_id IS NOT NULL AND status = 'active';

CREATE INDEX IF NOT EXISTS idx_facts_type ON facts (fact_type)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_facts_valid_until ON facts (valid_until)
    WHERE valid_until IS NOT NULL AND status = 'active';

CREATE INDEX IF NOT EXISTS idx_facts_tags ON facts USING gin(tags);

CREATE TABLE IF NOT EXISTS profiles (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    user_id         TEXT NOT NULL,
    scope           TEXT NOT NULL DEFAULT 'global',
    group_id        TEXT,

    profile_data    JSONB NOT NULL DEFAULT '{}',

    version         INT NOT NULL DEFAULT 1,
    fact_count      INT NOT NULL DEFAULT 0,
    last_fact_id    UUID REFERENCES facts(id),

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE(tenant_id, user_id, scope, group_id)
);

CREATE TABLE IF NOT EXISTS message_buffer (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    group_id        TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    content         JSONB NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    batch_id        UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_buffer_group ON message_buffer (tenant_id, group_id, created_at)
    WHERE status = 'pending';
