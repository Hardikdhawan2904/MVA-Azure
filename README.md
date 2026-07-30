# MVA — Multi-Agent Data Pipeline

**One system** that ingests a CSV/Excel dataset and takes it end-to-end: quality gating, schema classification, structural profiling, hierarchy inference, business-rule validation, AI-readiness scoring, chart generation, and AI-proposed rule suggestions with a human approve/reject loop.

Internally it's organized as four cooperating services plus a shared database — not because they're separate projects, but because each stage (classification, profiling, orchestration, Insurance Q&A) is cleanly separable and independently testable. One repo, one dependency set, one way to run it.

## ⚡ Azure OpenAI Setup (do this first)

This copy of the project runs on **Azure OpenAI**, not Groq. One config
file, 3 values, and every service picks it up automatically:

1. Clone this repo.
2. From the repo root, run:
   ```bash
   cp .env.example .env
   ```
3. Open the new `.env` file and fill in these 3 lines (they start blank):
   ```
   AZURE_OPENAI_API_KEY=
   AZURE_OPENAI_ENDPOINT=
   AZURE_OPENAI_DEPLOYMENT=
   ```
   - `AZURE_OPENAI_ENDPOINT` is your resource's base URL, e.g.
     `https://your-resource-name.openai.azure.com` (no trailing path).
   - `AZURE_OPENAI_DEPLOYMENT` is the *deployment name* you gave the model
     in Azure AI Studio — not necessarily the underlying model's own name.
4. Run `start-all.ps1` (see Quick Start below for the full first-time
   setup, including the one-time Python/database steps).

That's it — Agent 1, Agent 2, Agent 3, and the Orchestrator all read
these same 3 values from this one file. Nothing else to configure per
service. **Don't commit your filled-in `.env`** — it's already
gitignored on purpose, since this repo is public; only the blank
`.env.example` template is tracked.

## Architecture

```
                     ┌──────────────────────────────────────┐
   Upload ─────────▶ │  Agent Orchestrator                    │
                     │  Stage 0A: Dataset Registry (fingerprint│
                     │  the upload; a duplicate short-circuits │
                     │  straight to Agent 3, skipping 1 + 2)   │
                     └─────────┬──────────────────────────────┘
                               │ (new content, new version, or
                               │  force_revalidate)
              ┌────────────────┴────────────────┐
              ▼                                  ▼
   ┌─────────────────────┐          ┌─────────────────────────┐
   │  Agent 1             │          │  Agent 2                │
   │  Schema Intelligence  │ ───────▶ │  MVA Data Profiling      │
   │  Layer                │  domain  │  Engine                  │
   │  (quality gate,       │  +      │  (profiling, quality,    │
   │  classification,      │  column │  hierarchy, readiness,   │
   │  column descriptions) │  descrs │  charts, rule suggestions)│
   └──────────┬───────────┘          └────────────┬─────────────┘
              │                                    │
              └──────────────┬─────────────────────┘
                              ▼
                   ┌─────────────────────┐
                   │   Shared Postgres    │
                   │  (agent1 / agent2    │
                   │   schemas, one DB)   │
                   └──────────┬──────────┘
                              │ (.csv upload +
                              │  business_question only)
                              ▼
                   ┌─────────────────────┐
                   │  Agent 3 (optional)  │
                   │  Analytics Agent     │
                   │  FastAPI + LangGraph │
                   └─────────────────────┘
```

Each service runs as its own FastAPI process (so they can be started, stopped, and observed independently), but they share one virtual environment, one dependency list, and one repo history. The orchestrator is the only thing that chains them together — Agent 3 included, called over HTTP exactly like Agent 1/2. See "Agent 3 — Analytics Agent" below.

## What's inside

| Folder | Role | Port |
|---|---|---|
| [`Schema-Intelligence-Layer`](./Schema-Intelligence-Layer) | Quality gate, LLM column descriptions, business domain classification | 8000 |
| [`Data-Profiling-Agent`](./Data-Profiling-Agent) | Deep structural profiling, quality/readiness scoring, hierarchy inference, chart generation, AI rule suggestions | 8001 |
| [`Agent-Orchestrator`](./Agent-Orchestrator) | Chains Agent 1 → Agent 2 → (optionally) Agent 3 into one call | 8002 |
| [`Shared-Postgres`](./Shared-Postgres) | The one Postgres server everything persists to, schema-isolated per service | 5433 |
| [`Analytics-Agent`](./Analytics-Agent) | Agent 3 — domain-agnostic analytics engine (KPI/variance/root-cause/forecast/anomaly/segmentation/clustering/correlation/...), Insurance as the fully-built reference domain | 8003 |

Each has its own README going deeper on that piece specifically — this file is the map, not a duplicate.

## Quick Start

1. One virtual environment for the whole thing, from the repo root:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt -r requirements-dev.txt
   ```
2. Two layers of `.env` files:
   - **Root** (`cp .env.example .env`, fill in `AZURE_OPENAI_API_KEY`/`AZURE_OPENAI_ENDPOINT`/`AZURE_OPENAI_DEPLOYMENT`): shared values genuinely identical across Agent 1, Agent 2, Agent 3, and the Orchestrator — the LLM provider credentials, the `POSTGRES_*` connection details, `LOG_LEVEL`. Each service loads this as a fallback *underneath* its own local `.env` (local always wins on any key both define). Every LLM call in Agent 1/2/3 uses Azure OpenAI, degrading to a deterministic/template fallback if the call fails or the credentials are missing — the pipeline never hard-fails because of the LLM.
   - **Per-service** (`cp .env.example .env` inside `Schema-Intelligence-Layer/`, `Agent-Orchestrator/`, and `Analytics-Agent/`): only what's genuinely local to that service — e.g. Agent 1's `AZURE_OPENAI_DEPLOYMENT` override, Agent 3's `DATASET_PATH`/`HOST`/`PORT`, the Orchestrator's `AGENT1_BASE_URL`/`AGENT2_BASE_URL`/etc.
   - `Data-Profiling-Agent` (Agent 2) is the exception — it has its own differently-shaped config (`DATABASE_URL`, `LLM_API_KEY`, a separate Postgres role) and keeps its own fully self-contained `.env`, untouched by the root file.
3. **First time only** — bootstrap the database. `start-all.ps1` starts the Postgres *process* (native Windows install, not Docker — data dir `C:\PGData\mva-pipeline`, port 5433) but doesn't create the database, schemas, or roles on a genuinely fresh install:
   ```powershell
   & "C:\Program Files\PostgreSQL\17\bin\initdb.exe" -D "C:\PGData\mva-pipeline" -U postgres --pwprompt   # only if the data dir doesn't exist yet
   & "C:\Program Files\PostgreSQL\17\bin\pg_ctl.exe" -D "C:\PGData\mva-pipeline" start
   & "C:\Program Files\PostgreSQL\17\bin\createdb.exe" -h localhost -p 5433 -U postgres mva_pipeline
   & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -h localhost -p 5433 -U postgres -d mva_pipeline -f "Shared-Postgres\init\01-create-agent-schemas.sql"
   cd Data-Profiling-Agent; ..\venv\Scripts\python.exe -m alembic upgrade head; cd ..
   ```
   Agent 1 and Agent 3 need no manual step — both create their own schema/tables idempotently at every startup. See [`Shared-Postgres/README.md`](./Shared-Postgres/README.md) for exactly what the bootstrap SQL does and why (including a real, previously-undocumented `search_path` gotcha it now closes). Already-initialized instances (including this repo's own dev machine) skip this step entirely — `start-all.ps1` just starts the existing instance.
4. Start everything at once:

```powershell
powershell -File start-all.ps1
```

This starts the shared Postgres instance (idempotently via `pg_ctl` — safe to run whether or not it's already running) plus all four services (using the one shared venv), each in its own terminal window. Then:

- Full pipeline (recommended entry point): `http://127.0.0.1:8002/docs`
- Agent 1 alone: `http://127.0.0.1:8000/docs`
- Agent 2 alone: `http://127.0.0.1:8001/docs`
- Agent 3 alone: `http://127.0.0.1:8003/docs`

## The pipeline, end to end

1. **Upload** a CSV/XLSX file to the orchestrator's `/pipeline/run`.
2. **Dataset Registry (Stage 0A)** fingerprints the raw bytes (SHA-256, exact-match only) and checks the Master Dataset Repository. Every upload — hit or miss — gets a lightweight `DatasetCopy` record; deleting a copy later never touches the underlying data. Two outcomes:
   - **Duplicate content, already validated**: Agent 1 and Agent 2 are skipped entirely — their cached results are served straight through, byte-identical to the original run. Pass `force_revalidate=true` to bypass the cache and force a fresh run anyway (e.g. to correct a bad past classification).
   - **New content, a new version of a previously-seen filename, or `force_revalidate`**: continue to step 3 as normal.
3. **Agent 1** runs a configurable quality gate (10 checks — nulls, duplicates, corrupted values, etc.). Files that fail stop here with a `422`.
4. Passing files get LLM-generated column descriptions and a business-domain classification.
5. **Agent 2** receives Agent 1's output — including that domain classification, applied automatically with no manual input — and runs its full profiling pipeline: structural stats, semantic type detection, secondary-domain classification, hierarchy inference, business-rule evaluation (both YAML-configured and previously human-approved rules), quality/readiness scoring, and chart generation.
6. The LLM also proposes up to 5 candidate business rules from what it saw in that run's columns. These sit as `proposed` until a human approves or rejects them via Agent 2's API — approved rules then automatically apply to every future upload in that domain, closing the loop.
7. If the caller passed a `business_question` **and** it's a CSV, **Agent 3** answers that one question using Agent 2's ML-readiness score — see below. Runs for any of Agent 2's 5 supported domains, not just Insurance, and runs on every request regardless of Stage 0A's cache outcome (its answer depends on the specific question, not just dataset identity). Otherwise this step is skipped.
8. All agents' results come back together in one response (`agent1`, `agent2`, `agent3`, plus `fingerprint`/`copy_id`/`was_cached` from Stage 0A).
9. **Follow-up questions** don't need to repeat steps 1-6 — `POST /pipeline/ask` (with the earlier response's `agent2.run_id`) re-asks Agent 3 alone, skipping Agent 1's quality gate and Agent 2's full profiling entirely.

See [`Agent-Orchestrator/README.md`](./Agent-Orchestrator/README.md) for the Dataset Registry's full design (Master Dataset / DatasetCopy schema, reference counting, the `/datasets/*` admin endpoints for listing and deleting masters/copies).

## Agent 3 — Analytics Agent (optional; runs for any supported domain)

Agent 3 (`Analytics-Agent/`, port 8003) is a FastAPI service like Agent 1/2 — a thin `app/main.py`/`app/routes/` shell over a LangGraph `StateGraph` (`app/agents/analytics_agent/`). Internally it's a domain-agnostic, dataset-driven analytics engine (capability resolution → KPI discovery → question interpretation → planning → scheduling → model selection → execution → evidence → narration, see `Analytics-Agent/README.md` for the full stage-by-stage breakdown) via DuckDB + ML/LLM tools (`app/services/`) — Insurance is its one fully-built reference domain (curated KPIs, pre-computed variance drivers, business rules), and a dataset with no matching domain plugin still gets a real generic report (trend/forecast/correlation/anomaly/segmentation/... on whatever columns the dataset actually has), not a skip. The orchestrator calls its `POST /analyze` over httpx exactly like it calls Agent 1/2:

- **Runs whenever**: the upload is a `.csv` and `business_question` was supplied — for any of Agent 2's 5 supported domains (`Finance`, `Payments`, `Customer`, `HR`, `Insurance`), not just Insurance. Otherwise `agent3` in the response is `{"status": "skipped", "reason": "..."}`, and the reason names the actual capability gap (wrong file type, no question) rather than the domain. Root-cause and comparative-style analyses additionally prefer whichever column the question itself names (e.g. "net profit" → `net_profit_actual`) over a generic structural default, when the dataset has a matching column.
- **Inputs it's given**: the same uploaded file (posted straight through, no temp file on the orchestrator's side anymore), and Agent 2's `ml_readiness`/`llm_readiness` scores *and* their full breakdown (from `agent2.readiness_assessments` — strengths/blocking_issues/evidence, not just the score) as Form fields.
- **Explains itself**: every `status: "ok"` response carries `execution_trace` (step-by-step: intent → ML gate/engine → LLM gate/engine, each with real per-step timing and, where a real model ran, its version/accuracy from `ml/model_registry.json`) and `execution_summary` (a compact rollup) — see [`API_REFERENCE.md`](./API_REFERENCE.md#execution_trace--execution_summary).
- **Best-effort**: if it's unreachable or returns a non-200, `agent3.status == "failed"` with a reason — this never fails Agent 1/2's already-successful result.
- **Setup**: covered by the normal Quick Start above (`pip install -r requirements.txt` includes its deps — `duckdb`, `prophet`, `lightgbm`, `xgboost`, `scikit-learn`, `shap` — and it falls back to the root `.env`'s `AZURE_OPENAI_*` credentials, same as Agent 1). Nothing separate to clone or install.
- Configured via `Agent-Orchestrator/.env`: `ANALYTICS_AGENT_BASE_URL` (defaults to `http://127.0.0.1:8003`).
- A standalone local-testing CLI (no HTTP server needed) is still available at `Analytics-Agent/scripts/cli.py --query "..."`.
- Originally built as a separate project by a colleague (github.com/VirenKhapra/Analytics-agent-for-project-3) and vendored in here; its own repo still exists independently if contributing changes back upstream.
- **Asking a follow-up question?** Use `POST /pipeline/ask` instead of `/pipeline/run` — re-uploads the file (Agent 3 needs real rows to query; nothing durably stores them elsewhere) but skips Agent 1 and Agent 2's actual pipelines, reusing Agent 2's already-persisted readiness scores by `run_id`. See [`Agent-Orchestrator/README.md`](./Agent-Orchestrator/README.md#re-asking-agent-3-without-re-running-the-whole-pipeline).

## How scoring works (at a glance)

This file is the map, not a duplicate — each service's own README has the
full formulas. The short version, so you know where to look:

| Score | Computed by | Formula shape | Full detail |
|---|---|---|---|
| Quality gate (pass/fail) | Agent 1 | 10 weighted checks, weights sum to 100, `passing_score = 75` | [`Schema-Intelligence-Layer/README.md`](./Schema-Intelligence-Layer/README.md#7-how-scoring-works-the-quality-gate) |
| Overall data quality score | Agent 2 | `Σ(weight × score) / Σ(weight)` over assessed dimensions only (`not_assessable` dimensions excluded from both sides, never treated as zero) | [`Data-Profiling-Agent/README.md`](./Data-Profiling-Agent/README.md#data-quality-dimensions) |
| AI readiness (analytics / ml / llm / overall) | Agent 2 | Per-assessment point-additions, 0-100; `≥80 ready`, `≥60 partially_ready`, `<60 not_ready`. Each reports `score` / `dataset_score` / `task_compatibility_score` — three different questions, not three names for the same number | [`Data-Profiling-Agent/README.md`](./Data-Profiling-Agent/README.md#ai-readiness) |
| Capability resolution (structural + execution) | Agent 3 | Structural = can this analysis run at all (dataset shape); execution = does Agent 2's `ml_readiness_score` clear Agent 3's 75.0 threshold — never a new score, Agent 2's number reused directly | [`Analytics-Agent/README.md`](./Analytics-Agent/README.md#how-scoring-works) |

**One correction worth internalizing**: Agent 3 never receives Agent 2's
`dataset_score`/`task_compatibility_score` split — the Orchestrator's
`_readiness_and_features()` forwards `ml_readiness`/`llm_readiness`'s
plain composite `.score` field. If a dataset-only readiness number and a
question-specific one ever look like they should both be visible to Agent
3, they aren't — only the blended composite makes the trip.

## Known constraints worth knowing

- Agent 2 supports 5 primary domains (`Finance`, `Payments`, `Customer`, `HR`, `Insurance`) — each backed by a real config file defining its secondary domains, hierarchy templates, chart templates, and business rules. Agent 1's classification is open-vocabulary (14+ suggested domains), so `Agent-Orchestrator`'s `extract_domain_and_metadata` node canonicalizes known synonyms (e.g. `"Human Resources"` → `"HR"`, case variants) onto Agent 2's exact 5 strings before forwarding — a domain that's genuinely unsupported (or a synonym not yet in the map) still stops the pipeline with a clear error rather than guessing.
- LLM features (column descriptions, domain classification, rule suggestions) use Azure OpenAI, degrading gracefully to deterministic fallbacks if the credentials are missing/rate-limited/unreachable — the pipeline never hard-fails because of the LLM.
- Agent 3's own engine is domain-agnostic (see `Analytics-Agent/README.md`'s Domain Plugin architecture) — only CSV, not Excel, and its real ML models (Prophet/LightGBM/IsolationForest/XGBoost/K-Means) are trained and persisted against the Insurance dataset only, so a non-Insurance upload's ML-eligible analyses fit fresh per request rather than predicting against a cached model. A genuinely unsupported domain (not one of the 5 above) still stops the pipeline at Agent 2's own `UNSUPPORTED_DOMAIN` check, before Agent 3 is ever reached — that's Agent 2's capability boundary, not Agent 3's gate.
