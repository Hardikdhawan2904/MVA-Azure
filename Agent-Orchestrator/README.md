# Agent Orchestrator

Coordinates the multi-agent data pipeline: uploads a dataset once, sends it through **Agent 1 (Schema Intelligence Layer)** for classification and quality gating, then feeds Agent 1's output into **Agent 2 (MVA Data Profiling Engine)** for deep profiling — and, when the upload is a CSV with a business question attached, forwards it on to **Agent 3 (Analytics Agent)** to answer that question, for any of Agent 2's 5 supported domains. Returns all results together in a single response.

Expressed as a LangGraph `StateGraph` (`app/agents/orchestration_agent/`) rather than a chain of early returns — mostly sequential HTTP calls with conditional stop-on-error edges (no LLM calls or tools of its own), plus one best-effort optional stage for Agent 3. Each future agent added to the pipeline becomes another node in the same graph.

## Architecture

```
Upload → Dataset Registry (Stage 0A: fingerprint + duplicate/version check)
       → cache hit? → Agent 3 directly, Agent 1 + Agent 2 skipped, cached results served
       → cache miss? → Agent 1 (classify + quality gate) → Agent 2 (profile, using Agent 1's classification)
       → Agent 3 (optional — any of Agent 2's 5 supported domains, + business_question + CSV only)
       → combined response
```

If Agent 1 rejects the file at its quality gate, the pipeline stops there — Agent 2 (and therefore Agent 3) is never called.

**`primary_domain` is never accepted from the caller.** It's taken directly from Agent 1's `business_domain` classification and forwarded to Agent 2 automatically. If Agent 1's classification isn't one of Agent 2's supported domains (`Finance`, `Payments`, `Customer`, `HR`, `Insurance`), the pipeline stops with a clear `agent1_classification_missing` or domain-mismatch error rather than guessing.

Agent 3 is best-effort and never fails an otherwise-successful pipeline: outside its scope it's cleanly skipped, and if it's unreachable or errors, `agent3.status == "failed"` with a reason while `agent1`/`agent2` still return normally.

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/pipeline/run` | Runs a file through the pipeline. Accepts `file` (required), `sheet_name` (optional — required only for multi-sheet XLSX workbooks), `force_reclassify` (optional boolean, re-runs Agent 1's classification if the filename already exists in its catalog), `force_revalidate` (optional boolean, bypasses the Dataset Registry's cache and re-runs Agent 1 + Agent 2 fresh even for byte-identical content already processed before — implies `force_reclassify`), `business_question` (optional — also drives Agent 3, see below), `target_column` (optional — explicit override for Agent 2's target-column guessing). |
| POST | `/pipeline/ask` | Re-ask Agent 3 a *different* `business_question` against a dataset that already went through `/pipeline/run`, without repeating Agent 1's quality gate or Agent 2's full profiling. See below. |
| GET | `/datasets/masters` | List Master Datasets registered in the Dataset Registry (fingerprint, filename, version, row/column counts, reference count). |
| GET | `/datasets/masters/{fingerprint}/copies` | List the upload/copy history for a Master Dataset. |
| DELETE | `/datasets/copies/{copy_id}` | Soft-delete one upload's copy record. Never touches the underlying Master Dataset. |
| DELETE | `/datasets/masters/{fingerprint}?force=` | Delete a Master Dataset and its physical file. Refuses with `409` if active copies still reference it, unless `force=true`. |
| GET | `/health` | Health check — also pings Agent 1 and Agent 2 so it's obvious what's actually reachable (Agent 3 isn't pinged here since it's optional/best-effort). |

### Response shape

```json
{
  "agent1": { "...": "Agent 1's full response (raw row data stripped out)" },
  "agent2": { "...": "Agent 2's full profiling result, including rule_suggestions" },
  "agent3": { "...": "Agent 3's answer, or {\"status\": \"skipped\", ...} outside its scope" },
  "primary_domain_used": "Finance",
  "fingerprint": "sha256 of the raw uploaded bytes",
  "copy_id": "uuid — this upload's own DatasetCopy record, always new even on a cache hit",
  "was_cached": false
}
```

`agent1.dataframe_records` (the raw uploaded rows) is deliberately stripped from the response before returning — Agent 1 already persists it, and passing it through balloons responses to tens of MB on large files, which Swagger struggles to render.

### Dataset Registry (Stage 0A)

Deterministic infrastructure, not an agent — sits in front of Agent 1/Agent 2 and answers one question: "has this exact content been fully processed before?" Implemented in `app/services/dataset_registry/` (`dataset_registry.py`'s `DatasetRegistry` facade, `fingerprint.py`, `storage.py`, `metadata_cache.py`, `duplicate_detector.py`, `version_manager.py`, `reference_counter.py`), wired in as the graph's new entry point (`resolve_dataset_identity`) in `app/agents/orchestration_agent/graph.py`.

- **Identity**: SHA-256 of the raw uploaded bytes — exact match only, no fuzzy/near-duplicate detection. A byte-identical re-upload shares a **Master Dataset**; any change at all (even one cell) gets its own.
- **Versioning**: a re-upload under a filename the Registry has already seen, but with different content, is treated as a new *version* of the same logical dataset (`latest_version` increments, `previous_fingerprint` links back) rather than either overwriting history or being mistaken for something unrelated.
- **Storage**: Master Dataset bytes are written once, content-addressed, under `MASTER_DATASET_STORAGE_DIR` (default `./data/master-datasets/<hash[:2]>/<hash>.<ext>` — relative to this service's own working directory, so an unset env var works out of the box on any machine/OS instead of assuming a specific Windows path exists). Every upload — hit or miss — creates its own lightweight `DatasetCopy` record; deleting a copy only ever removes that reference, never the Master Dataset. The Master Dataset itself is a reference-counted, separate, deliberate action — `DELETE /datasets/masters/{fingerprint}` refuses unless forced when active copies still exist.
- **Caching**: on a cache hit, Agent 1's and Agent 2's cached responses are served directly (`master_dataset_results`, keyed by fingerprint) and neither service is actually called. Agent 3 is never cached/skipped — its answer depends on the specific `business_question`, not just dataset identity, so it runs on every request regardless of the Stage 0A outcome.
- **Non-fatal by design**: a Postgres outage or unwritable storage directory degrades to "the cache always misses, the pipeline runs exactly as it did before this feature existed" — never breaks the core relay function.
- Bootstraps its own Postgres schema (`orchestrator`, 3 tables: `master_datasets`, `dataset_copies`, `master_dataset_results`) on startup, in the same shared native Postgres instance as Agent 1/2/3 — see [`Shared-Postgres/README.md`](../Shared-Postgres/README.md).

### Automatic column-quality inference (no manual configuration needed)

Before forwarding to Agent 2, `extract_domain_and_metadata()` (`app/agents/orchestration_agent/nodes/pipeline.py`) automatically infers two things directly from the uploaded file — nothing to configure by hand:

- **`mandatory`/`expected_unique` per column** (`_infer_mandatory_and_unique`): identifier-shaped columns (name ends in `id`/`identifier`/`uuid`/`guid`, or Agent 1's LLM description explicitly calls it a unique identifier) are marked `expected_unique`, never `mandatory`; everything else defaults the other way.
- **Cross-field consistency rules** (`_infer_consistency_rules`): scans numeric column pairs in the uploaded file itself and proposes a `column_comparison` rule (e.g. `actual <= budget`) only when the relationship empirically holds for ≥99% of jointly-valid rows across a real sample of the data — never guessed from column names, capped at 2 auto-proposed rules. This is what lets Agent 2's `consistency` quality dimension (and therefore `ml_readiness`) reflect real data-driven rules without anyone writing YAML.

Both directly feed Agent 2's `schema_metadata`/`request_rules` fields — see [`Data-Profiling-Agent/README.md`](../Data-Profiling-Agent/README.md) for how Agent 2 actually uses them in scoring.

**`request_rules` override**: `/pipeline/run` also accepts an optional `request_rules` form field (a JSON string of additional Agent 2 business rules, forwarded verbatim on top of the automatic inference above) — deliberately *not* shown in Swagger UI, since the normal case needs no manual configuration at all, but it's a real, functional parameter for direct API callers who want to assert a relationship the auto-inference can't verify from its own sample. Read directly off the raw request rather than declared as a FastAPI parameter, which is why it doesn't appear in `/docs`.

### Agent 3 (Analytics Agent)

Runs whenever **both** of: the upload is a `.csv`, and `business_question` was supplied — for any of Agent 2's 5 supported domains (`Finance`, `Payments`, `Customer`, `HR`, `Insurance`), gated on file type and question only, never domain. Called over `httpx` — the same shape as the call to Agent 2 — posting the file straight through along with Agent 2's `ml_readiness`/`llm_readiness` scores, their full readiness breakdown (`agent2.readiness_assessments[]` — strengths/blocking_issues/evidence, not just the score, extracted by `_readiness_and_features()` in `app/agents/orchestration_agent/nodes/pipeline.py`), and (if present) its `feature_recommendation.feature_columns`. Agent 3's response — including its `execution_trace`/`execution_summary` — is forwarded back through `agent3_body` untouched. See the [root API reference](../API_REFERENCE.md#agent-3--analytics-agent-optional-third-stage) for the full response shapes, and [`Analytics-Agent/README.md`](../Analytics-Agent/README.md) for what it does internally.

### Re-asking Agent 3 without re-running the whole pipeline

`POST /pipeline/ask` — for asking a follow-up question against a dataset already processed by `/pipeline/run`. Takes `file` (re-upload — required, see why below), `business_question` (the new question), and `run_id` (Agent 2's `run_id`, copied from the earlier `/pipeline/run` response's `agent2.run_id`). It fetches Agent 2's *already-persisted* result with one lightweight `GET` (not a new profiling run) and calls Agent 3 directly — Agent 1's quality gate and Agent 2's full profiling pipeline never re-run.

```json
{"agent3": {"status": "ok", "query": "...", "response": "...", "ml_readiness_score_used": 29.47, "llm_readiness_score_used": 95.77,
            "execution_trace": [ "..." ], "execution_summary": { "...": "..." }},
 "primary_domain_used": "Insurance"}
```

`agent3.status` works exactly like `/pipeline/run` (`ok` / `skipped` / `failed`, never a hard error). Two error cases are specific to this endpoint, since they're caller mistakes rather than "outside Agent 3's scope": `404` if `run_id` isn't known to Agent 2, `502` if Agent 2 is unreachable.

**Why the file still has to be re-uploaded:** neither Agent 1 nor Agent 2 durably stores the raw dataset rows anywhere — Agent 1's dataframe cache is in-memory only (wiped on restart), and Agent 2 deletes its temp-uploaded copy at the end of every run. Agent 3 needs real rows to query via DuckDB, and there's nowhere durable this endpoint can fetch them back from on its own. Only the readiness scores and feature recommendation (which *are* durably persisted) get reused by `run_id`.

## Local Setup

This project shares one virtual environment with the rest of the pipeline — set up once from the **repo root** (see the [root README](../README.md)), not from inside this folder:

```bash
cd ..
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cd Agent-Orchestrator

# Copy env file and adjust if Agent 1 / Agent 2 run on different hosts/ports
cp .env.example .env

..\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8002
```

Agent 1, Agent 2, and (if you want the analytics Q&A stage) Agent 3 must already be running (see their own READMEs) — the orchestrator only coordinates between them, it has no database or LLM of its own.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| AGENT1_BASE_URL | http://127.0.0.1:8000 | Base URL for Agent 1 |
| AGENT2_BASE_URL | http://127.0.0.1:8001 | Base URL for Agent 2 |
| AGENT2_API_PREFIX | /api/v1 | Agent 2's API prefix |
| REQUEST_TIMEOUT_SECONDS | 120.0 | Timeout for calls to Agent 1/Agent 2 |
| ANALYTICS_AGENT_BASE_URL | http://127.0.0.1:8003 | Base URL for Agent 3 |
| ANALYTICS_AGENT_TIMEOUT_SECONDS | 120.0 | Timeout for the call to Agent 3 |
| POSTGRES_HOST | localhost | Shared Postgres host (Dataset Registry, schema `orchestrator`) |
| POSTGRES_PORT | 5433 | Shared Postgres port |
| POSTGRES_DB | mva_pipeline | Shared Postgres database |
| POSTGRES_USER | postgres | Shared Postgres user |
| POSTGRES_PASSWORD | postgres | Shared Postgres password |
| MASTER_DATASET_STORAGE_DIR | ./data/master-datasets | Content-addressed physical storage for Master Dataset bytes (relative to this service's working directory by default) |
| MAX_UPLOAD_SIZE_MB | 100 | Rejects `POST /pipeline/run`/`/pipeline/ask` uploads over this size with a `413`, before any bytes are hashed or forwarded downstream |

No API keys or secrets live here — LLM credentials belong to Agent 1, Agent 2, and Agent 3 individually. The `POSTGRES_*`/`MASTER_DATASET_STORAGE_DIR` keys are usually inherited from the shared root `.env` (see the [root README](../README.md)) — only needed here if this service points at a different Postgres instance than the rest of the pipeline.

## Running Tests

```bash
cd ..
..\venv\Scripts\python.exe -m pytest Agent-Orchestrator/tests/ -v
```

Uses the shared root venv (this project doesn't carry its own `pytest`) — covers the pure-data helpers in `app/agents/orchestration_agent/nodes/pipeline.py`, currently `_readiness_and_features()` (extracting scores *and* full readiness breakdowns from Agent 2's result, including the missing-assessments and `None`-input edge cases).

## Example

```bash
curl -X POST http://localhost:8002/pipeline/run \
  -F "file=@payments.csv"
```

No `primary_domain` field — it's derived automatically from Agent 1's classification.

```bash
# A dataset with a business question — also invokes Agent 3 (works for
# any of Agent 2's 5 supported domains, Insurance shown here as one example)
curl -X POST http://localhost:8002/pipeline/run \
  -F "file=@insurance_variance_data_native.csv" \
  -F "business_question=Show Gross Written Premium for FY2025"
```

```bash
# Follow-up question against the same dataset — only Agent 3 runs
curl -X POST http://localhost:8002/pipeline/ask \
  -F "file=@insurance_variance_data_native.csv" \
  -F "business_question=Why did underwriting result decline in FY2025?" \
  -F "run_id=<agent2.run_id from the /pipeline/run response above>"
```

## How Scoring Works

The Orchestrator **computes zero scores itself** — it's a relay and a
deterministic-inference layer, not a scoring engine. Two things worth
understanding about how it shapes what gets scored downstream:

1. **It doesn't recompute Agent 1's quality gate or Agent 2's quality/
   readiness scores** — `agent1`/`agent2` in the response are those
   services' own outputs, forwarded verbatim (see
   `Schema-Intelligence-Layer/README.md` and
   `Data-Profiling-Agent/README.md` for the exact formulas). The one
   exception is `_readiness_and_features()` (`nodes/pipeline.py`), which
   extracts Agent 2's `ml_readiness`/`llm_readiness` **plain composite
   `.score`** (never the `dataset_score`/`task_compatibility_score`
   split) plus each assessment's full breakdown, purely to forward them
   to Agent 3 — no new arithmetic happens here.
2. **Automatic column-quality inference** (`_infer_mandatory_and_unique`/
   `_infer_consistency_rules`, documented above) runs *before* Agent 2
   scores anything — it decides which columns count as `mandatory`/
   `expected_unique` and which cross-field consistency rules apply, which
   directly changes Agent 2's `completeness`/`consistency`/`uniqueness`
   dimension scores and therefore `ml_readiness`. This is real, if
   indirect, influence over the final score — worth knowing when a
   quality score looks different than expected on a re-run with a
   different sample of the same data (the ≥99%-of-sample consistency-rule
   check can occasionally propose or drop a rule between runs on a
   borderline dataset).

## Known Limitations

Fixed during the handover pass: upload size limits (`MAX_UPLOAD_SIZE_MB`,
413 on exceed, checked before any hashing/forwarding) and a
`max_length=2000` cap on `business_question` were added to both
`/pipeline/run` and `/pipeline/ask` (previously unbounded, forwarded
verbatim into Agent 3's LLM prompt and several regex scans);
`MASTER_DATASET_STORAGE_DIR`'s default changed from a hardcoded absolute
Windows path to a relative one so a fresh clone works cross-platform
without manual configuration; the Dataset Registry's `create_copy()` +
`update_reference_count()` pair now runs in one transaction instead of
two separately-committed connections (a crash between them could
under-count `reference_count` — the delete guardrail itself recomputes a
live count so this could never cause a wrongful deletion, but the count
`GET /datasets/masters` reports could lie).

Still open — real findings that need a conscious decision rather than a
same-session patch:

- No retry/circuit-breaker logic on the calls to Agent 1/Agent 2/Agent 3 — a transient failure surfaces immediately as a `502` (Agent 1/2) or `agent3.status == "failed"` (Agent 3) rather than being retried
- No authentication in v1
- Domain auto-derivation canonicalizes known synonyms of Agent 1's broader vocabulary onto Agent 2's 5 supported domains (`_canonicalize_domain()` / `_DOMAIN_SYNONYMS` in `app/agents/orchestration_agent/nodes/pipeline.py` — e.g. `"Human Resources"` → `"HR"`, case variants), but the map is finite: an unrecognized synonym still fails Agent 2's exact-match check. Notably, Agent 1's classification prompt doesn't suggest `"Payments"` or `"Customer"` as domain names at all (see `Schema-Intelligence-Layer/app/prompts/llm_service_prompt.py`), so those two domains rarely get classified into by Agent 1 in the first place — no amount of synonym-mapping here fixes that; it would need a prompt change in Agent 1.
- `POST /pipeline/ask` still requires re-uploading the file, even though the Dataset Registry (Stage 0A) now durably stores every upload's raw bytes, content-addressed by fingerprint (`app/services/dataset_registry/storage.py`). The durable storage this limitation used to say didn't exist now does — `/pipeline/ask` just isn't wired to look the file up by `run_id`/fingerprint yet, it's a genuinely closeable gap now rather than a missing capability. Read directly by `SQLTool`, that lookup would let a future version drop the re-upload requirement entirely.
- **No retention/eviction policy for the Dataset Registry's physical
  storage** — `MASTER_DATASET_STORAGE_DIR` grows forever; only a manual
  per-fingerprint `DELETE /datasets/masters/{fingerprint}?force=` exists.
  The caching feature's entire premise is "every unique upload lives
  forever," which is fine until disk fills. A suggested next step is
  LRU-by-`last_referenced_at` eviction, not implemented — a product
  decision about retention policy, not a bug.
- Blocking synchronous I/O (the Dataset Registry's psycopg2 calls) runs
  directly on this service's async LangGraph nodes — same cross-service
  issue documented in `Schema-Intelligence-Layer/README.md`'s Known
  Limitations; not fixed here for the same reason (deserves one general
  fix across all 4 services, not four independent partial patches).
- Default DB credentials (`postgres`/`postgres`) ship as fallbacks — a
  repo-wide convention, flagged as "change before production" rather than
  altered in isolation here.
