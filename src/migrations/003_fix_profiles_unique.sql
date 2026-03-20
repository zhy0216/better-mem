-- Drop the broken UNIQUE constraint that doesn't enforce uniqueness when group_id IS NULL
-- (PostgreSQL treats NULL != NULL so multiple rows with NULL group_id can satisfy the constraint)
ALTER TABLE profiles DROP CONSTRAINT IF EXISTS profiles_tenant_id_user_id_scope_group_id_key;

-- Partial index for global profiles (group_id IS NULL) - enforces one global profile per user per scope
CREATE UNIQUE INDEX IF NOT EXISTS profiles_global_uniq
    ON profiles (tenant_id, user_id, scope)
    WHERE group_id IS NULL;

-- Partial index for group profiles (group_id IS NOT NULL) - enforces one group profile per (user, scope, group)
CREATE UNIQUE INDEX IF NOT EXISTS profiles_group_uniq
    ON profiles (tenant_id, user_id, scope, group_id)
    WHERE group_id IS NOT NULL;

-- Remove duplicate global profiles, keeping only the most recently updated one per (tenant, user, scope)
DELETE FROM profiles
WHERE group_id IS NULL
  AND id NOT IN (
      SELECT DISTINCT ON (tenant_id, user_id, scope) id
      FROM profiles
      WHERE group_id IS NULL
      ORDER BY tenant_id, user_id, scope, updated_at DESC
  );
