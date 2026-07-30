-- One database (mva_pipeline, created automatically by POSTGRES_DB) shared by all
-- agents, namespaced per agent via Postgres schemas so table names can never
-- collide and no agent's migrations can touch another agent's tables.
CREATE SCHEMA IF NOT EXISTS agent1 AUTHORIZATION postgres;

CREATE USER mva_user WITH PASSWORD 'mva_password';
CREATE SCHEMA IF NOT EXISTS agent2 AUTHORIZATION mva_user;

-- Owning a schema does NOT put it on the role's search_path — mva_user's
-- default search_path ("$user", public) resolves to nothing (no schema
-- named "mva_user" exists), not to agent2. Without this, Alembic
-- migration 001 (which relies on the connecting role's search_path
-- instead of explicit schema-qualification — later migrations fixed
-- this, see 002/003's own comments) fails with "permission denied for
-- schema public" on a genuinely fresh instance.
ALTER ROLE mva_user SET search_path TO agent2, public;

-- Agent 3 (Analytics Agent) — conversation memory only. Uses the postgres
-- superuser like agent1, not a dedicated role like agent2: this schema is
-- created idempotently by Agent 3's own init_db() at every startup too
-- (this script only runs once, on a fresh volume), so there's no benefit
-- to a separate role at this project's scale.
CREATE SCHEMA IF NOT EXISTS agent3 AUTHORIZATION postgres;
