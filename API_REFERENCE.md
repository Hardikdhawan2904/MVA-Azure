# MVA Pipeline — API Reference

Four services — two data-intake agents, an orchestrator, and one optional analytics Q&A agent — each a separate FastAPI process with its own Swagger UI, sharing one venv and one dependency list. This doc lists every real endpoint, what it's for, and how to open each service's interactive docs locally.

**Flow:** Upload → Agent 1 *(classify + quality-gate)* → Orchestrator *(relay)* → Agent 2 *(profile + score)* → Agent 3 *(optional Q&A, any of the 5 supported domains)* → Combined result

---

## Connect

| Service | Port | Swagger UI | Health check |
|---|---|---|---|
| **Agent 1 — Schema Intelligence** | `8000` | http://127.0.0.1:8000/docs | `GET /health` |
| **Agent 2 — Data Profiling Engine** | `8001` | http://127.0.0.1:8001/docs | `GET /api/v1/health` |
| **Orchestrator** | `8002` | http://127.0.0.1:8002/docs | `GET /health` |
| **Agent 3 — Analytics Agent** | `8003` | http://127.0.0.1:8003/docs | `GET /health` |

**Agent 1** validates uploads, runs the 10-check quality gate, and classifies business domain via LLM.
**Agent 2** does deep column profiling, quality scoring, hierarchy detection, chart + rule generation.
**Orchestrator** runs a file through Agent 1 then Agent 2 (then optionally Agent 3) in one call, no manual domain entry needed.
**Agent 3** is a domain-agnostic analytics engine (Insurance is its fully-built reference domain); the Orchestrator's pipeline routes any of Agent 2's 5 supported domains to it, gated only on file type (`.csv`) and a supplied `business_question`.

---

## Starting all four locally

1. Postgres must be running first — a native Windows instance (see `Shared-Postgres/README.md`), listening on port `5433`. Normally started automatically by `start-all.ps1`.
2. Each service uses the shared virtual environment at the repo root (`venv/`) and its own `app.main:app` entry point:

```bash
cd Schema-Intelligence-Layer   && ..\venv\Scripts\python -m uvicorn app.main:app --port 8000 --reload
cd Data-Profiling-Agent     && ..\venv\Scripts\python -m uvicorn app.main:app --port 8001 --reload
cd Agent-Orchestrator          && ..\venv\Scripts\python -m uvicorn app.main:app --port 8002 --reload
cd Analytics-Agent             && ..\venv\Scripts\python -m uvicorn app.main:app --port 8003 --reload
```

3. Or run `start-all.ps1` from the repo root — it launches all four (plus Postgres) in separate windows in one shot.
4. Once a service prints `Application startup complete`, its `/docs` URL above is live — open it in a browser to try requests directly, no separate client needed.

---

## Agent 1 — Schema Intelligence Layer

Base URL: `http://127.0.0.1:8000`

First stop for any raw upload. Validates the file, scores it against 10 deterministic quality checks, and — for new files or ones explicitly flagged — classifies the business domain and describes every column via an LLM. Its classification is what downstream agents rely on; there's no manual override.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/upload-dataset` | Upload a CSV or Excel file. Runs the quality gate (aborts with 422 on FAIL), classifies the domain, catalogs it in Postgres, and caches the parsed rows in memory. Params: `file` (multipart upload), `force_reclassify` (bool — re-run LLM classification instead of reusing a cached result) |
| `GET` | `/datasets` | List every cataloged dataset with its domain, row/column counts, and quality score. |
| `GET` | `/datasets/{dataset_id}` | Full catalog record for one dataset — classification, column descriptions, full quality report. |
| `GET` | `/datasets/{dataset_id}/dataframe` | The cached row data as JSON records — only available while the server that processed it hasn't restarted. Param: `limit` (optional, cap rows returned) |
| `GET` | `/health` | Liveness check — returns `{"status": "healthy"}` when the service is up. |

---

## Agent 2 — Data Profiling Engine

Base URL: `http://127.0.0.1:8001/api/v1`

Takes a raw file plus the primary domain Agent 1 already determined, and produces column-level profiling, secondary classification, hierarchy detection, quality scoring across 9 dimensions, chart candidates, and business-rule evaluation in one pass.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/profile-runs` | Start a profiling run. Runs synchronously and returns once complete. Params: `file` (multipart upload), `primary_domain` (must be Finance, Payments, Customer, HR, or Insurance), `sheet_name` (required only for multi-sheet Excel workbooks) |
| `GET` | `/profile-runs/{run_id}` | Run summary — status, domain, row/column counts, timestamps. |
| `GET` | `/profile-runs/{run_id}/result` | The complete result — every column profile, quality dimension, chart, hierarchy edge, and rule evaluation in one payload. |
| `GET` | `/profile-runs/{run_id}/columns` | Just the column-level profiles (types, stats, semantic classification) for this run. |
| `GET` | `/profile-runs/{run_id}/quality` | Quality assessment scores across all 9 dimensions plus the overall weighted score. |
| `GET` | `/profile-runs/{run_id}/readiness` | Analytics / ML / LLM readiness assessments — is this dataset actually usable downstream. |
| `GET` | `/profile-runs/{run_id}/hierarchy` | The detected dimensional hierarchy (e.g. region → country → branch) with per-edge confidence. |
| `GET` | `/profile-runs/{run_id}/rule-evaluations` | Results of evaluating this domain's business rules against the uploaded data. |
| `GET` | `/profile-runs/{run_id}/charts` | Generated chart specs — domain-specific where the data supports them, generic otherwise — with aggregated data attached. |
| `POST` | `/profile-runs/{run_id}/charts/{chart_id}/drill-down` | Drill into one level of a hierarchy chart (e.g. from country down into its cities) for a specific path. |
| `GET` | `/rule-suggestions` | List all LLM-proposed business rules across runs, optionally filtered by approval status. Param: `status` (optional, e.g. proposed / approved / rejected) |
| `GET` | `/rule-suggestions/{suggestion_id}` | A single proposed rule's full detail. |
| `POST` | `/rule-suggestions/{suggestion_id}/approve` | Approve a proposed rule so it's evaluated against future uploads in this domain. |
| `POST` | `/rule-suggestions/{suggestion_id}/reject` | Reject a proposed rule. |
| `GET` | `/health` | Liveness + database connectivity check. |

---

## Orchestrator

Base URL: `http://127.0.0.1:8002`

The one call to make if you just want a file profiled end to end. Sends the upload to Agent 1, takes whatever domain it decides on, and forwards straight into Agent 2 — no domain has to be picked by hand. If a `business_question` was supplied and the upload is a `.csv`, also forwards to Agent 3, for any of Agent 2's 5 supported domains. Stops cleanly with a clear error at whichever stage fails.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/pipeline/run` | Runs a file through the pipeline and returns all results together under one response. Params: `file` (multipart upload), `sheet_name` (required only for multi-sheet Excel workbooks), `force_reclassify` (bool — re-run Agent 1's LLM classification), `force_revalidate` (bool — bypass the Dataset Registry's cache entirely and re-run Agent 1 + Agent 2 fresh even for byte-identical content already processed before; implies `force_reclassify`), `business_question` (optional — also drives Agent 3, see below), `target_column` (optional) |
| `POST` | `/pipeline/ask` | Re-ask Agent 3 a different `business_question` against a dataset already processed by `/pipeline/run`, without repeating Agent 1's quality gate or Agent 2's full profiling. See below. |
| `GET` | `/health` | Liveness check that also reports whether Agent 1 and Agent 2 are currently reachable (Agent 3 isn't pinged here since it's optional/best-effort). |

`/pipeline/run`'s response body is `{"agent1": {...}, "agent2": {...}, "agent3": {...}, "primary_domain_used": "...", "fingerprint": "...", "copy_id": "...", "was_cached": false}` — `agent3` is always present with a `status` field (`ok` / `skipped` / `failed`), never `null`, whenever the overall response reaches this shape at all. `was_cached: true` means Agent 1 and Agent 2 were skipped entirely and their results were served from the Dataset Registry's cache (see below) — `agent1`/`agent2` are still byte-identical to what a live run would have produced.

### Dataset Registry (Stage 0A)

Not a separate service — deterministic infrastructure inside the Orchestrator, sitting in front of Agent 1/Agent 2. Fingerprints every upload (SHA-256 of the raw bytes, exact-match only) and, on a duplicate, skips Agent 1 and Agent 2 entirely rather than re-running them. Distinct from Agent 1's own `/datasets*` endpoints above (Agent 1's catalog metadata) — these manage the Registry's own Master Dataset / DatasetCopy records.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/datasets/masters` | List Master Datasets — fingerprint, filename, version, row/column counts, reference count. Param: `limit` (default 100) |
| `GET` | `/datasets/masters/{fingerprint}/copies` | List the upload/copy history for one Master Dataset. Param: `include_deleted` (default false) |
| `DELETE` | `/datasets/copies/{copy_id}` | Soft-delete one upload's copy record. Never touches the Master Dataset or its physical file. |
| `DELETE` | `/datasets/masters/{fingerprint}` | Hard-delete a Master Dataset and its physical file. Refuses with `409` if active copies still reference it, unless `?force=true`. |

### Agent 3 — Analytics Agent (optional third stage)

Vendored into this repo at `Analytics-Agent/` (originally a colleague's separate project, github.com/VirenKhapra/Analytics-agent-for-project-3) — a FastAPI service like Agent 1/2, called by the orchestrator over `httpx` the same way it calls Agent 2. It installs from the same root `requirements.txt` and runs under the same shared venv as the other three folders. Internally it's a domain-agnostic, dataset-driven analytics engine (see `Analytics-Agent/README.md`) with Insurance as its fully-built reference domain (curated KPI lookup, variance, root-cause, forecast, anomaly detection, segmentation); a dataset with no matching domain plugin still gets a real generic multi-analysis report instead of a skip, and the Orchestrator's gate below routes any of Agent 2's 5 supported domains to it, not just Insurance.

Runs whenever **both** of: the upload is a `.csv`, and `business_question` was supplied — for any of Agent 2's 5 supported domains (`Finance`, `Payments`, `Customer`, `HR`, `Insurance`). Fed Agent 2's `ml_readiness`/`llm_readiness` scores **and** their full breakdown (`agent2.readiness_assessments[]` — strengths/blocking_issues/evidence, not just the bare score) as Form fields, so Agent 3's `execution_trace` can explain *why* a readiness gate passed or failed, not just report a number. Response shapes:

```jsonc
// Ran successfully:
"agent3": {
  "status": "ok", "query": "...", "conversation_id": "...",
  "ml_readiness_score_used": 39.48, "llm_readiness_score_used": 95.77,
  "response": "<narrative text>",
  "execution_trace": [ /* step-by-step decision log — see below */ ],
  "execution_summary": { /* compact rollup — see below */ }
}

// Outside its scope (wrong domain / no question / not CSV):
"agent3": {"status": "skipped", "reason": "..."}

// Invoked but errored/unreachable — never fails the overall pipeline:
"agent3": {"status": "failed", "reason": "..."}
```

Configured via `Agent-Orchestrator/.env`: `ANALYTICS_AGENT_BASE_URL` (defaults to `http://127.0.0.1:8003`), `ANALYTICS_AGENT_TIMEOUT_SECONDS`.

#### `execution_trace` / `execution_summary`

Every `status: "ok"` response (from `/pipeline/run`, `/pipeline/ask`, or `/analyze` directly) carries these two fields, built once from the LangGraph run's final state — `null` on `status: "error"` rather than fabricating an explanation for a genuine crash.

`execution_trace` is a list of steps (`intent_detection` → one step per scheduled analysis (usually just one) → `narration`, when narration ran). Each entry:

```jsonc
{
  "step": "forecast",
  "engine": "Prophet",                 // or the deterministic fallback name
  "gate": {
    "name": "ml_readiness", "score": 82.0, "threshold": 75.0, "passed": true,
    "breakdown": { "evidence": [{"dimension": "feature_coverage", "value": 1.0}, ...], "strengths": [...], "blocking_issues": [...] }
  },
  "reason": "ML readiness (82.0%) met the 75.0% threshold — using the trained Prophet model. Strongest on feature_coverage (100%); weakest on data_freshness (55%).",
  "duration_ms": 360.3,                // real per-node wall-clock time, via graph.stream()
  "model_version": { "refit_per_query": true, "last_run_at": "2026-07-17T22:24:02" }
}
```

- `gate` is `null` for analysis types with no ML/LLM readiness check (`kpi_summary`, `kpi_variance`, `root_cause`, `trend`, and every "fully generic" structural analysis type — clustering, classification, correlation, ...). `gate.breakdown` is `null` when the caller didn't supply one (a direct `/analyze` call outside the Orchestrator has no breakdown to forward).
- `model_version` only appears when the ML-gated step actually ran a model (not on the fallback path). Prophet refits every query, so it reports `last_run_at`, not a training date; IsolationForest/K-Means report `trained_at` plus an explicitly-`null` `accuracy_metric` (both unsupervised — no fabricated number). root_cause's XGBoost corroboration and forecast's LightGBM key-drivers evidence cite their real registry accuracy/r² in `reason` when present.
- A bare/generic `business_question` against a dataset with no curated KPI to resolve runs **report mode** instead: one `execution_trace` step per scheduled analysis (budget-bounded, up to 8), `execution_summary.intent == "report"`, and `response` has one narrated section per analysis type.

`execution_summary` is a compact rollup: `{"intent", "tools_used", "ml_engine", "narration_engine", "execution_time_seconds", "fallback_used"}` — `fallback_used` is `true` if either readiness gate didn't pass, or Azure OpenAI was attempted but its call itself failed.

#### `POST /pipeline/ask` — re-ask without re-running Agent 1/2

For a follow-up question against a dataset already run through `/pipeline/run`. Params: `file` (multipart upload — must be re-uploaded; see why below), `business_question` (str), `run_id` (Agent 2's `run_id`, from the earlier `/pipeline/run` response's `agent2.run_id` field). Fetches Agent 2's *already-persisted* result with one lightweight `GET /profile-runs/{run_id}/result` — no new profiling work — then calls Agent 3 directly. Agent 1 never runs at all for this endpoint.

```jsonc
{"agent3": {"status": "ok", "query": "...", "response": "...", "conversation_id": "...",
            "ml_readiness_score_used": 29.47, "llm_readiness_score_used": 95.77,
            "execution_trace": [ /* same shape as /pipeline/run — see above */ ], "execution_summary": { /* ... */ }},
 "primary_domain_used": "Insurance"}
```

`agent3.status` behaves exactly as above (`ok` / `skipped` / `failed`). Two error responses are specific to this endpoint (caller mistakes, not "outside Agent 3's scope"): `404` if `run_id` is unknown to Agent 2, `502` if Agent 2 is unreachable. `agent3.conversation_id` is a fresh one every call — the Orchestrator doesn't track sessions, so this endpoint always starts a new Agent 3 conversation rather than continuing one from a prior `/pipeline/run`/`/pipeline/ask` call.

Requires re-uploading the file because neither Agent 1 nor Agent 2 durably stores raw dataset rows anywhere (Agent 1's dataframe cache is in-memory only; Agent 2 deletes its temp-uploaded copy after every run) — Agent 3 needs real rows to query via DuckDB. Only the readiness scores and feature recommendation, which *are* durably persisted, get reused by `run_id`.

#### `POST /analyze` (called directly, standalone)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/analyze` | Answer a business question against an uploaded CSV (or run multi-analysis report mode with a bare/generic question — see above). Params: `file` (multipart upload), `business_question` (str), `conversation_id` (optional str — omit for a new conversation; pass back a prior response's `conversation_id` to continue it), `ml_readiness` (float, default 99.75), `llm_readiness` (float, default 99.75), `feature_recommendation` (optional JSON string — simple per-column classification, used only to validate the hardcoded ML feature lists in `config/ml_config.yml` still match this dataset), `ml_readiness_breakdown` / `llm_readiness_breakdown` (optional JSON strings — Agent 2's full readiness assessment for the respective gate, surfaced in `execution_trace`'s gate objects; never blocks the request if omitted or malformed), `column_profiles` / `hierarchy` / `charts` / `full_feature_recommendation` (optional JSON strings — Agent 2's richer per-column semantic classification/drill-down hierarchy/chart candidates/target-column recommendation; builds a real `DatasetContext` instead of falling back to local schema inference), `detected_domain` (optional str — Agent 1's business_domain classification; selects the matching `DomainPlugin`, or the generic no-KPI-catalog engine if none matches) |
| `GET` | `/health` | Liveness check — Agent 3 has no downstream agents to ping. |

`conversation_id` is always present in the response, generated fresh if the caller didn't supply one. Conversation memory (filter/KPI carryover, LLM prior-turn context) is persisted in Postgres (`agent3` schema on the same [`Shared-Postgres`](../Shared-Postgres) instance Agent 1/2 use) keyed by this id — non-fatal if that database is unreachable, degrading to a single-turn, non-persistent answer rather than failing the request. The Orchestrator never passes `conversation_id` (`/pipeline/run` and `/pipeline/ask` are both stateless), so multi-turn continuity only applies when calling `POST /analyze` directly.

A standalone local-testing CLI is also available without running the HTTP server: `Analytics-Agent/scripts/cli.py --query "..."` (reads the reference dataset configured via `Analytics-Agent/.env`'s `DATASET_PATH`); `--interactive` mode reuses one `conversation_id` across the whole session.

---

Ports assume the default local setup (Agent 1 → 8000, Agent 2 → 8001, Orchestrator → 8002, Agent 3 → 8003) with Postgres on 5433. If your friend is running on a different machine, swap `127.0.0.1` for that machine's address and make sure the ports are reachable — Swagger itself needs nothing beyond the running service.
