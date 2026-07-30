# Shared Postgres

The single PostgreSQL server used by every agent in the pipeline. One database (`mva_pipeline`), one server instance — each agent gets its own schema inside it, so table names can never collide and no agent's migrations can touch another agent's tables.

```
mva_pipeline (database)
├── agent1  (Schema Intelligence Layer's tables, owned by postgres)
├── agent2  (MVA Data Profiling Engine's tables, owned by mva_user)
└── agent3  (Analytics Agent's conversation memory, owned by postgres)
```

This intentionally replaced an earlier setup where each agent ran its own separate Postgres server — consolidated into one server, then further consolidated from one-database-per-agent into one shared database with per-agent schemas.

Runs as a **native Windows Postgres instance** (data dir `C:\PGData\mva-pipeline`, port `5433` — not the default `5432`, to avoid clashing with any local Postgres install you might already have), not Docker. This project ran the shared server as a Docker container earlier on; that setup has been fully retired and removed in favor of a native instance on the same port, so no other service's config changed.

## Usage

Started automatically as part of the root [`start-all.ps1`](../start-all.ps1) — nothing to do here manually in the normal case. To start/check it directly yourself:

```powershell
$pgBin = "C:\Program Files\PostgreSQL\17\bin"
$pgData = "C:\PGData\mva-pipeline"
& "$pgBin\pg_ctl.exe" -D $pgData status
& "$pgBin\pg_ctl.exe" -D $pgData start
```

On a genuinely fresh data directory, `init/01-create-agent-schemas.sql` is what bootstraps all three schemas, the `mva_user` role, **and its `search_path`** — replay it once via `psql` against a new instance (it isn't run automatically outside of Docker):

```powershell
$env:PGPASSWORD = "postgres"
& "$pgBin\psql.exe" -h localhost -p 5433 -U postgres -d mva_pipeline -f "..\Shared-Postgres\init\01-create-agent-schemas.sql"
```

It won't re-run against an already-initialized instance, and doesn't need to for the existing one. **Undocumented gotcha this script closes**: owning a schema does not put it on that role's `search_path` — `mva_user`'s default (`"$user", public`) resolves to nothing, since no schema is literally named `mva_user`. Without `ALTER ROLE mva_user SET search_path TO agent2, public;` (now part of the script), Agent 2's very first Alembic migration (`001`, which relies on the connecting role's search_path rather than explicit schema-qualification — every migration from `002` onward fixed this, see their own inline comments) fails with `permission denied for schema public` on a genuinely fresh instance. Confirmed already applied on this project's live instance; only relevant again when standing up a new one from scratch.

Agent 3's own `init_db()` (`Analytics-Agent/app/services/database.py`) creates its schema/table idempotently at every service startup instead, which is why it doesn't need a dedicated role the way Agent 2 does: `CREATE SCHEMA IF NOT EXISTS` needs no elevated one-time setup, unlike `CREATE USER`.

Every other repo in this pipeline (`Schema-Intelligence-Layer`, `Data-Profiling-Agent`, `Analytics-Agent`) points at this same server — none of them run their own Postgres.

## Connection details

| | |
|---|---|
| Host | `localhost` |
| Port | `5433` |
| Database | `mva_pipeline` |
| Agent 1 role | `postgres` / `postgres` — schema `agent1` |
| Agent 2 role | `mva_user` / `mva_password` — schema `agent2` |
| Agent 3 role | `postgres` / `postgres` — schema `agent3` |

These are local development defaults, not production credentials — this instance isn't exposed beyond `localhost`.

## Stopping

```powershell
& "C:\Program Files\PostgreSQL\17\bin\pg_ctl.exe" -D "C:\PGData\mva-pipeline" stop
```
