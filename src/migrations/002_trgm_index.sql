CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE INDEX IF NOT EXISTS idx_facts_content_trgm ON facts
    USING gin (content gin_trgm_ops);
