-- Migration: Replace facts table with propositions + beliefs + evidence

-- ============================================================
-- 1. Create propositions table
-- ============================================================
CREATE TABLE propositions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id           TEXT NOT NULL DEFAULT 'default',
    user_id             TEXT NOT NULL,
    group_id            TEXT,
    subject_id          TEXT,

    canonical_text      TEXT NOT NULL,
    proposition_type    TEXT NOT NULL,
    semantic_key        TEXT,

    valid_from          TIMESTAMPTZ,
    valid_until         TIMESTAMPTZ,
    first_observed_at   TIMESTAMPTZ,
    last_observed_at    TIMESTAMPTZ,

    embedding           vector(1024),
    tsv                 tsvector GENERATED ALWAYS AS (
                            to_tsvector('simple', canonical_text)
                        ) STORED,

    tags                TEXT[] DEFAULT '{}',
    metadata            JSONB DEFAULT '{}',

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_propositions_embedding ON propositions
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

CREATE INDEX idx_propositions_tsv ON propositions USING gin(tsv);

CREATE INDEX idx_propositions_user_time ON propositions
    (tenant_id, user_id, last_observed_at DESC);

CREATE INDEX idx_propositions_semantic_key ON propositions
    (tenant_id, user_id, semantic_key)
    WHERE semantic_key IS NOT NULL;

CREATE INDEX idx_propositions_group ON propositions
    (tenant_id, group_id, last_observed_at DESC)
    WHERE group_id IS NOT NULL;

CREATE INDEX idx_propositions_type ON propositions (proposition_type);

CREATE INDEX idx_propositions_valid_until ON propositions (valid_until)
    WHERE valid_until IS NOT NULL;

CREATE INDEX idx_propositions_tags ON propositions USING gin(tags);

-- ============================================================
-- 2. Create beliefs table
-- ============================================================
CREATE TABLE beliefs (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    proposition_id      UUID NOT NULL UNIQUE
                            REFERENCES propositions(id) ON DELETE CASCADE,

    confidence          FLOAT NOT NULL DEFAULT 0.5,
    prior               FLOAT NOT NULL DEFAULT 0.5,
    source_reliability  FLOAT NOT NULL DEFAULT 0.8,

    utility_importance  FLOAT NOT NULL DEFAULT 0.5,
    freshness_decay     FLOAT NOT NULL DEFAULT 0.01,

    support_count       INT NOT NULL DEFAULT 0,
    contradiction_count INT NOT NULL DEFAULT 0,

    access_count        INT NOT NULL DEFAULT 0,
    last_accessed       TIMESTAMPTZ,

    status              TEXT NOT NULL DEFAULT 'active',

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_beliefs_status_conf ON beliefs (status, confidence DESC);
CREATE INDEX idx_beliefs_importance ON beliefs (utility_importance DESC);

-- ============================================================
-- 3. Create evidence table
-- ============================================================
CREATE TABLE evidence (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    proposition_id      UUID NOT NULL REFERENCES propositions(id) ON DELETE CASCADE,

    evidence_type       TEXT NOT NULL,
    direction           TEXT NOT NULL DEFAULT 'support',

    source_type         TEXT NOT NULL,
    source_id           TEXT,
    source_meta         JSONB,

    speaker_id          TEXT,
    quoted_text         TEXT,
    observed_at         TIMESTAMPTZ,

    weight              FLOAT NOT NULL DEFAULT 1.0,
    metadata            JSONB DEFAULT '{}',

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_evidence_proposition ON evidence (proposition_id, observed_at DESC);
CREATE INDEX idx_evidence_source ON evidence (source_type, source_id);

-- ============================================================
-- 4. Migrate data from facts -> propositions + beliefs + evidence
-- ============================================================
INSERT INTO propositions (
    id, tenant_id, user_id, group_id,
    canonical_text, proposition_type, semantic_key,
    valid_from, valid_until,
    first_observed_at, last_observed_at,
    embedding, tags, metadata,
    created_at, updated_at
)
SELECT
    id, tenant_id, user_id, group_id,
    content, fact_type, NULL,
    valid_from, valid_until,
    occurred_at, occurred_at,
    embedding, tags, metadata,
    created_at, updated_at
FROM facts
WHERE status IN ('active', 'superseded');

INSERT INTO beliefs (
    proposition_id,
    confidence, prior, source_reliability,
    utility_importance, freshness_decay,
    support_count, contradiction_count,
    access_count, last_accessed,
    status,
    created_at, updated_at
)
SELECT
    p.id,
    CASE WHEN f.status = 'superseded' THEN 0.2 ELSE
        CASE f.fact_type
            WHEN 'declaration' THEN 0.85
            WHEN 'preference'  THEN 0.80
            WHEN 'relation'    THEN 0.75
            WHEN 'observation' THEN 0.65
            WHEN 'plan'        THEN 0.60
            ELSE 0.50
        END
    END AS confidence,
    CASE f.fact_type
        WHEN 'declaration' THEN 0.85
        WHEN 'preference'  THEN 0.80
        WHEN 'relation'    THEN 0.75
        WHEN 'observation' THEN 0.65
        WHEN 'plan'        THEN 0.60
        ELSE 0.50
    END AS prior,
    0.8,
    f.importance,
    f.decay_rate,
    CASE WHEN f.status = 'superseded' THEN 0 ELSE 1 END,
    CASE WHEN f.status = 'superseded' THEN 1 ELSE 0 END,
    f.access_count,
    f.last_accessed,
    CASE WHEN f.status = 'superseded' THEN 'deprecated' ELSE 'active' END,
    f.created_at,
    f.updated_at
FROM facts f
JOIN propositions p ON p.id = f.id;

INSERT INTO evidence (
    proposition_id,
    evidence_type, direction,
    source_type, source_id, source_meta,
    speaker_id, quoted_text, observed_at,
    weight, metadata,
    created_at
)
SELECT
    p.id,
    'utterance', 'support',
    f.source_type, f.source_id, f.source_meta,
    NULL, f.content, f.occurred_at,
    1.0, '{}',
    f.created_at
FROM facts f
JOIN propositions p ON p.id = f.id;

-- ============================================================
-- 5. Update profiles FK to point to propositions instead of facts
-- ============================================================
ALTER TABLE profiles DROP CONSTRAINT IF EXISTS profiles_last_fact_id_fkey;
ALTER TABLE profiles RENAME COLUMN last_fact_id TO last_proposition_id;
ALTER TABLE profiles ADD CONSTRAINT profiles_last_proposition_id_fkey
    FOREIGN KEY (last_proposition_id) REFERENCES propositions(id);

-- ============================================================
-- 6. Drop old facts table
-- ============================================================
DROP TABLE facts CASCADE;
