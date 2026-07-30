# MVA Data Profiling Engine

**Multi-Variance Analysis — Schema, Quality, Hierarchy, Readiness, and Chart Intelligence**

A production-structured backend that profiles CSV/XLSX datasets, classifies them into domains, validates hierarchy structures, assesses data quality, evaluates AI-readiness, and generates typed chart specifications.

## Architecture

```
File Upload → Validation → Profiling → Type Refinement → Semantic Candidates
    → Schema Intelligence → Domain Classification → Category Classification
    → Hierarchy Inference → Business Rules → Quality Assessment
    → AI Readiness → Chart Generation → Persist Results → Cleanup
```

Every stage produces typed results. Non-critical failures do not destroy successful upstream results.

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 16+ — runs from a shared server used by all agents in the pipeline (see below), not from this repo directly

### Local Development

This project shares one virtual environment with the rest of the pipeline — see the [root README](../README.md). From the repo root:

```bash
# Start the shared Postgres server (once) -- native Windows instance, not Docker;
# the normal way to do this is just running ..\start-all.ps1, which starts Postgres
# plus all four services together. To start only Postgres directly:
& "C:\Program Files\PostgreSQL\17\bin\pg_ctl.exe" -D "C:\PGData\mva-pipeline" status  # check
& "C:\Program Files\PostgreSQL\17\bin\pg_ctl.exe" -D "C:\PGData\mva-pipeline" start   # start if needed

# Create the shared environment and install (once, for all three services)
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cd Data-Profiling-Agent

# Copy env file
cp .env.example .env
# Defaults already match the shared Postgres server (localhost:5433, mva_pipeline db,
# this project's tables live in the `agent2` schema) — no edits needed for local dev

# Run migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload --port 8001

# Run tests
python -m pytest tests/ -v
```

Note: this project has no Docker setup of its own — Postgres runs as a native
Windows instance (`C:\PGData\mva-pipeline`, port 5433) shared across all agents;
see [`Shared-Postgres/README.md`](../Shared-Postgres/README.md).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/profile-runs` | Create profiling run. Accepts an optional `sheet_name` form field, required only when the uploaded XLSX workbook has more than one non-empty sheet — names which sheet to load. |
| GET | `/api/v1/profile-runs/{id}` | Run summary |
| GET | `/api/v1/profile-runs/{id}/result` | Full result |
| GET | `/api/v1/profile-runs/{id}/columns` | Column profiles |
| GET | `/api/v1/profile-runs/{id}/quality` | Quality assessments |
| GET | `/api/v1/profile-runs/{id}/readiness` | AI readiness |
| GET | `/api/v1/profile-runs/{id}/hierarchy` | Hierarchy chain |
| GET | `/api/v1/profile-runs/{id}/rule-evaluations` | Business rule evaluation results (per-rule pass/fail counts and score, persisted for every run) |
| GET | `/api/v1/profile-runs/{id}/charts` | Chart specs |
| POST | `/api/v1/profile-runs/{id}/charts/{cid}/drill-down` | Drill-down |
| GET | `/api/v1/rule-suggestions` | List AI-proposed rule suggestions (optional `?status=` filter) |
| GET | `/api/v1/rule-suggestions/{id}` | Get a single suggestion |
| POST | `/api/v1/rule-suggestions/{id}/approve` | Approve a suggestion — materializes it into an active rule definition that future runs in the same primary domain will evaluate |
| POST | `/api/v1/rule-suggestions/{id}/reject` | Reject a suggestion — no rule definition is created |

## Example Usage

```bash
curl -X POST http://localhost:8001/api/v1/profile-runs \
  -F "file=@payments.csv" \
  -F "primary_domain=Payments" \
  -F 'schema_metadata={"columns":[{"column_name":"amount","description":"Payment amount","mandatory":true}]}'
```

## Supported Domains

| Primary | Secondary Domains |
|---------|-------------------|
| Payments | Authorization, Clearing, Settlement, Fraud |
| Customer | CRM, Customer Satisfaction, Loyalty |
| HR | Employee, Payroll, Recruitment |
| Finance | Revenue, P&L, Forecasting |

## Adding a New Domain

1. Create `config/domains/insurance.yaml` following the existing structure
2. Define secondary domains with keywords and semantic roles
3. Add hierarchy templates
4. Add chart templates
5. Add business rules

**No Python code changes required.** The engine loads configuration dynamically.

## Configuration

All domain-specific behavior is in `config/` YAML files:

- `config/domains/*.yaml` — domain definitions, secondary domains, templates, rules
- `config/quality_weights.yaml` — quality dimension weights (actively used by `calculate_overall_score`)
- `config/readiness_weights.yaml` — documents the intended AI readiness weight profiles, but **`ReadinessEngine` does not currently load this file** — its weights are hardcoded directly in `app/services/readiness/readiness_engine.py`. Editing this YAML has no effect today.
- `config/hierarchy_thresholds.yaml` — FD validation thresholds
- `config/chart_policy.yaml` — chart generation policy
- `config/application.yaml` — global thresholds

## Key Design Principles

### Deterministic Before LLM
- Physical types: Pandas + parse ratios (never LLM)
- Statistics: NumPy/Pandas (never LLM)
- Identifier detection: cardinality analysis (never LLM)
- Rule enforcement: typed engine (never LLM)
- FD validation: groupby aggregation (never LLM)

### LLM Only For Semantic Reasoning
- Confirm/override semantic types
- Classify ambiguous secondary domains
- Propose business rule candidates
- Generate recommendation text

### Raw Data Lifecycle
- Uploaded file → temp directory (UUID-scoped)
- Loaded into DataFrame transiently
- Processed through pipeline
- Temp file deleted on success AND failure
- Only derived metadata persisted to PostgreSQL
- No raw rows in database

## Data Quality Dimensions

| Dimension | Formula | When Not Assessable |
|-----------|---------|---------------------|
| Completeness | 1 - null_count/total for mandatory cols | No mandatory columns defined |
| Uniqueness | 1 - dupes/total for expected-unique cols | No expected-unique columns |
| Validity | pass/checked for range/allowed rules | No validity rules configured |
| Conformity | pass/checked for regex rules | No conformity rules |
| Consistency | 1 - contradictions/checked | No cross-field rules |
| Business Rules | pass/total across all active rules | No rules evaluated |
| Timeliness | Requires SLA config | Always (v1) |
| Integrity | Requires reference data | Always (v1) |
| Accuracy | Requires trusted reference | Always (v1) |
| Semantic Quality | Weighted avg of confidences | No SI results |

**Overall score** = `Σ(weight × score) / Σ(weight)` for assessed dimensions only.

## AI Readiness

`ReadinessEngine` (`app/services/readiness/readiness_engine.py`) computes
4 assessments — `analytics_readiness`, `ml_readiness`, `llm_readiness`,
`overall_ai_readiness` — each 0-100, `status` thresholds `≥80 ready`,
`≥60 partially_ready`, `<60 not_ready` (`app/core/constants.py`'s
`READINESS_READY_THRESHOLD`/`READINESS_PARTIALLY_READY_THRESHOLD`).

**Analytics readiness** (dataset-only, no task input): completeness × 20 +
validity × 15 + consistency × 15 + metrics-available (15 if ≥2 metrics, 8
if ≥1) + dimensions-available (15 if ≥3, 8 if ≥1) + grain-identified
(+10) + temporal-fields (+10), capped at 100.

**ML readiness**: completeness × 20 + consistency × 15 (dataset-only) +
feature-coverage × 15 (dataset-wide ratio, *or* task-specific when a
`feature_recommendation` was supplied for this question) + an
identifier-contamination penalty (up to −10) + row-count adequacy (0/5/
10/15 by row-count tier) + cardinality-health (5 or 10) + uniqueness × 10.

**LLM readiness**: description-coverage (up to 25, tiered) +
semantic-quality × 20 + schema-clarity ((1 − avg_null_ratio) × 15) +
sample-availability × 10 + context-rich-metadata (up to 15) — all
dataset-only — plus an optional task-dependent "question-suitability"
boost (up to +10) when `feature_recommendation.recommended_approach ==
"llm"`.

**The `score` vs. `dataset_score` vs. `task_compatibility_score` split**
— every assessment reports all three, and they answer different
questions:
- **`score`** — that assessment's own blended composite (for ML, this
  already mixes dataset-only and task-specific evidence internally).
- **`dataset_score`** — "how good is this data, independent of any
  question" — analytics always contributes one; ml/llm only when their
  task-independent sub-scores are computable.
- **`task_compatibility_score`** — "how well does this data fit *this
  specific question*" — only ml and llm ever produce one (analytics has
  no task-dependent input by design); `None` when no `business_question`
  was asked.

`overall_ai_readiness` combines these three ways, in code
(`ReadinessEngine.assess_all`):
```
overall.score                     = mean(analytics.score, ml.score, llm.score)
overall.dataset_score             = mean of whichever of analytics/ml/llm.dataset_score are not None
overall.task_compatibility_score  = mean of whichever of ml/llm.task_compatibility_score are not None
```

**What Agent 3 actually receives**: the Orchestrator's
`_readiness_and_features()` forwards `ml_readiness`/`llm_readiness`'s own
plain **`score`** field (not `overall_ai_readiness`, and not the
`dataset_score`/`task_compatibility_score` split specifically) — Agent
3's `ML_READINESS_THRESHOLD`/`LLM_READINESS_THRESHOLD` (both 75.0) gate
against that forwarded value.

**`config/readiness_weights.yaml` is not actually wired up** — see Known
Limitations below. The weight values described above are hardcoded
directly in `readiness_engine.py`, not read from that file.

## AI Rule Suggestions

After profiling, the pipeline asks the LLM to propose up to 5 candidate business rules based on that run's column profiles (null ratios, distinct counts, sample values) — e.g. *"this column is never null, add a non-null rule"* or *"only two values observed, add an allowed-values rule."* Suggestions are always structured, engine-compatible rules (one of the 7 types `RuleEngine` supports), never free text.

- Suggestions are persisted per-run with `status: proposed` and never auto-activate.
- `GET /rule-suggestions` / `GET /rule-suggestions/{id}` — review them (also embedded directly in `GET /profile-runs/{id}/result` under `rule_suggestions`, so no separate call is required to see their IDs).
- `POST /rule-suggestions/{id}/approve` — materializes the suggestion into an active `rule_definitions` row, scoped to that run's primary domain.
- From that point on, **every future upload in the same primary domain is automatically checked against it** — no code change, no re-approval. Results are persisted per-run and queryable via `GET /profile-runs/{id}/rule-evaluations`.
- `POST /rule-suggestions/{id}/reject` — no rule gets created.

Generation degrades gracefully: if the LLM call fails or is rate-limited, `rule_suggestions` comes back as an empty list rather than failing the run.

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific phase
python -m pytest tests/unit/test_quality.py -v

# With coverage
python -m pytest tests/ --cov=app --cov-report=term-missing
```

## Running Migrations

On an already-bootstrapped instance (this project's normal dev setup), just:
```bash
# Apply all migrations
alembic upgrade head

# Generate new migration after model changes
alembic revision --autogenerate -m "description"

# Rollback one step
alembic downgrade -1
```

On a genuinely fresh Postgres instance, `alembic upgrade head` fails on
migration `001` with `permission denied for schema public` unless
`mva_user`'s `search_path` was set to include `agent2` first — see the
root README's Quick Start step 3 and
[`Shared-Postgres/README.md`](../Shared-Postgres/README.md) for the
one-time bootstrap SQL that handles this. Not needed against this
project's already-initialized dev instance.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | postgresql://... | PostgreSQL connection |
| MAX_UPLOAD_SIZE_MB | 25 | Max file size |
| MAX_DATASET_ROWS | 200000 | Max rows |
| MAX_DATASET_COLUMNS | 200 | Max columns |
| PROCESSING_TIMEOUT_SECONDS | 120 | Pipeline timeout |
| MIN_CUBE_GROUP_SIZE | 5 | Small-group suppression |
| LLM_PROVIDER | local | LLM backend (`local` or `azure`) — see below |
| LLM_API_KEY | | (unused — Azure OpenAI credentials come from AZURE_OPENAI_API_KEY instead, see root README) |
| LLM_BURST_COOLDOWN_SECONDS | 5 | Delay before the rule-suggestion LLM call, after the schema-intelligence LLM call — avoids bursting two requests in quick succession against the same rate-limited endpoint |
| LOG_LEVEL | INFO | Logging level |

## Known Limitations

Fixed during the handover pass (see the fix instead of working around it):
`profile_runs.py`'s hardcoded `timeout=120.0` now reads
`settings.processing_timeout_seconds`; `ConfigurationRepository` now
resolves `config/` relative to its own package location instead of the
process's current working directory; a catch-all
`@app.exception_handler(Exception)` now guarantees every error, not just
the 5 explicitly-typed ones, returns the documented `{"error": {...}}`
envelope instead of Starlette's default unstructured 500.

Still open — real findings that need a conscious product/architecture
decision rather than a same-session patch:

- **Auth is wired into ~1 of ~17 endpoints.** `app/api/auth.py` exists and
  `POST /profile-runs` uses `Depends(get_current_user)`, but every rule-
  suggestion mutation (`approve`/`reject`) and every `GET` sub-resource is
  unauthenticated. Not fixed here because enabling it on every route would
  break every current caller (including this session's own manual
  testing) that isn't sending credentials — a decision for whoever owns
  the deployment target. Fix, when ready: add `Depends(get_current_user)`
  to each remaining route in `app/api/routes/`.
- **A run that exceeds the processing timeout is never persisted.**
  `create_profile_run` only calls `repo.create_run()`/`persist_*` inside
  `if completed_job and completed_job.result` — a slow run's `run_id` is
  handed back to the caller, but `GET /profile-runs/{run_id}/result` 404s
  forever, even after the background job finishes successfully moments
  later. Real bug, not fixed here because the correct fix (persist a
  `status: processing` row eagerly at job creation, then update it on
  completion) touches core job orchestration and deserves its own test
  coverage rather than a patch bundled with everything else in this pass.
- **Blocking synchronous I/O runs directly on the async event loop** —
  the LLM calls and `wait_for_completion`'s blocking wait inside this
  service's async routes. A real throughput ceiling under concurrent
  load, not a correctness bug at the team/dev-tool scale this system runs
  at today. Same underlying issue as Agent 1's blocking DB/LLM calls (see
  `Schema-Intelligence-Layer/README.md`'s Known Limitations) — general fix
  pattern is `run_in_threadpool`/`asyncio.to_thread`, or a pooled async
  driver, applied once across all 4 services rather than four independent
  partial patches.
- **`JobManager._jobs` grows without bound** — no TTL, no eviction, no
  size cap; every profiling job's result stays in memory for the life of
  the process. Fine for a dev/demo box, not fine for a long-running
  server under sustained traffic. Deciding an eviction policy (LRU by
  last-access? hard TTL?) changes what `GET /profile-runs/{run_id}/result`
  can promise callers, so it's left as a product decision, not silently
  fixed. Parallel to Agent 1's unbounded `_dataframes` cache.
- **Drill-down cubes are computed every run and never persisted** — real
  work thrown away each time; the endpoint already hardcodes `cubes=[]`
  with an in-code comment acknowledging the gap. Building real persistence
  is a genuine feature addition (new repository method + wiring), not a
  bug fix.
- **`config/readiness_weights.yaml` documents intended AI-readiness
  weights but isn't actually loaded by `ReadinessEngine`** — the weights
  described in the AI Readiness section above are hardcoded directly in
  `readiness_engine.py` instead. The YAML file is aspirational/
  documentation-only today.
- LLM integration requires an API key; without one, the deterministic
  fallback formatter is used instead.
- Background-thread processing (the job abstraction is already shaped for
  an async/queue migration, but runs on threads today).
- XLSX workbooks with multiple non-empty sheets require the `sheet_name`
  form field to disambiguate which sheet to load; omitting it on a
  multi-sheet file returns a `MULTIPLE_XLSX_SHEETS` error.
- Individual rule-suggestion generation calls are bounded to the first 30
  columns and 5 suggestions per run; not every column gets considered on
  very wide datasets.
- Default DB credentials (`mva_user`/`mva_password`) ship as fallbacks in
  `.env.example` — a repo-wide convention across all 4 services, flagged
  here as "change before any production deployment" rather than altered,
  since altering it would just move the same fallback problem elsewhere.
