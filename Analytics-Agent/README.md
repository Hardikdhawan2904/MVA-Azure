# Analytics Agent (Agent 3)

A domain-agnostic, dataset-driven analytics engine — resolves what's analytically possible on an uploaded dataset, discovers KPIs, plans and schedules analyses within a budget, selects an ML or deterministic algorithm per analysis, executes it, and narrates the evidence in business language (Azure OpenAI, falling back to a deterministic template formatter if unavailable/rate-limited or `llm_readiness` is below threshold).

Insurance is the **reference domain** — a fully-built `DomainPlugin` (`app/services/domain_plugins/insurance/`) with a curated KPI catalog (Gross Written Premium, Loss Ratio, Combined Ratio, Underwriting Result, ...), 14 pre-computed variance drivers, and 18 business rules, all preserved byte-identical through the redesign below. A dataset with no matching plugin runs through the exact same pipeline with an empty KPI catalog (`GenericDomainPlugin`) instead of inheriting any of Insurance's assumptions — confirmed working end-to-end against non-Insurance fixtures (HR payroll, finance, payments).

Originally a separate project by a colleague (github.com/VirenKhapra/Analytics-agent-for-project-3), vendored into this monorepo, rebuilt as a FastAPI + LangGraph service, and then redesigned from an Insurance-only intent-dispatch agent into the generic engine described here — see [Decisions Log](#decisions-log) below for the full history.

## Architecture

```
POST /analyze (file + business_question + conversation_id + ml_readiness + llm_readiness
               + optional Agent 2 fields: column_profiles, hierarchy, charts,
               feature_recommendation, full_feature_recommendation, detected_domain)

  Stage 0  build_dataset_context     — DatasetContext from Agent 2's column_profiles/
                                        hierarchy/charts (or LocalSchemaInferer, if none
                                        of that was forwarded)
  Stage 1  resolve_capabilities      — per capability: structural support (can this run
                                        at all, given the dataset's shape?) split from
                                        execution support (does Agent 2's ml_readiness_score
                                        clear this agent's own threshold?) — never
                                        recomputes ml_readiness/llm_readiness, consumes
                                        Agent 2's scores verbatim
  Stage 2  discover_kpis             — named business KPIs synthesized from semantic-role
                                        combinations (Profit Margin, Salary Distribution,
                                        Variance vs Budget, ...), not hardcoded columns
  Stage 3  interpret_question        — narrows candidate analyses to what the question
                                        asks; resolves which curated KPI (if a domain
                                        plugin has one) and which filters (fiscal_year,
                                        region, ...) the question is about
  Stage 4  plan_analytics            — WHAT is analytically relevant (structural pattern
                                        rules); a domain plugin's enhance_plan() then
                                        injects/overrides curated-KPI-grounded analyses
  Stage 5  schedule                  — WHEN/HOW MANY, within a budget (max 8 parallel,
                                        max 3 ML, max 2 expensive) — requested/KPI-grounded
                                        analyses are never trimmed
  Stage 6+7 execute_analyses         — per scheduled analysis: ModelSelector picks an
                                        algorithm (ML or deterministic, config/
                                        model_registry.yml — never a fixed intent→model
                                        dict), the matching Analyzer executes it
  Stage 8  (EvidenceBuilder)         — one Evidence per analysis; flattens to the old flat
                                        shape for a single analysis, nests under
                                        analyses[type] for multi-analysis "report mode"
  Stage 9  narrate                   — Azure OpenAI, then a deterministic template formatter —
                                        unchanged logic, only its input shape changed
  record_memory                      — sliding-window conversation history, persisted to
                                        Postgres by conversation_id
  → response
```

A thin `app/main.py`/`app/routes/analyze.py` shell over a LangGraph `StateGraph` (`app/agents/analytics_agent/`), same shape as Agent 1/2. Unlike the other two data-intake agents, this graph is built **fresh per request** rather than once at import time — every request analyzes a *different* uploaded dataset, so its DuckDB connection, rule engine, and readiness-gated model selection all have to be bound to that request's own inputs.

### Domain Plugin architecture

`DomainPlugin` (`app/services/domain_plugins/base.py`) is the single extension point for domain-specific behavior — additive only, never `if domain == X: ... else: ...` in the generic engine:

| Plugin | Curated KPIs | Driver columns | Preferred deterministic strategy | Notes |
|---|---|---|---|---|
| `InsurancePlugin` | 17 (Gross Written Premium, Loss Ratio, Combined Ratio, Underwriting Result, ...) | 14 pre-computed variance drivers | Linear Trend (forecast), Z-Score (anomaly), Insurance Combined Ratio Buckets (segmentation) | Reference domain — byte-identical to the pre-redesign agent for every existing query shape |
| `FinancePlugin` | 3 (Net Profit, Revenue, Operating Cost) | None (correlation-based root cause) | None (registry default order) | Thin starter — `ThinKPIDomainPlugin`-based |
| `HRPlugin` | 3 (Headcount, Attrition Rate, Payroll Cost) | None (correlation-based root cause) | None (registry default order) | Thin starter — `ThinKPIDomainPlugin`-based |
| `PaymentsPlugin` | 3 (Transaction Volume, Authorization Success Rate, Fraud Rate) | None (correlation-based root cause) | None (registry default order) | Thin starter — `ThinKPIDomainPlugin`-based |
| `CustomerPlugin` | 3 (Customer Satisfaction Score, Churn Rate, Customer Lifetime Value) | None (correlation-based root cause) | None (registry default order) | Thin starter — `ThinKPIDomainPlugin`-based |
| `GenericDomainPlugin` | None (empty catalog) | None (correlation-based root cause instead) | None (registry default order) | The true "no plugin matched" fallback — used whenever `detected_domain` isn't forwarded or doesn't match any registered plugin, so an unrelated dataset never silently inherits Insurance's KPI catalog |

The four "thin starter" plugins (`app/services/domain_plugins/thin_kpi_plugin.py`) are a curated KPI catalog plus `kpi_summary`/`kpi_variance` question-answering only — no driver columns (no labeled-mode root cause), no ML-feature-column overrides, unlike `InsurancePlugin`'s fully-built implementation. They still resolve KPI aliases (e.g. "net profit" → `net_profit`) via `get_intent_vocabulary()` and inject the matching analysis via `enhance_plan()` — deliberately narrower than Insurance's own `enhance_plan()`, but never worse than `GenericDomainPlugin`'s generic report for anything they don't cover.

`PluginRegistry.find_plugin(detected_domain)` matches on Agent 1's canonicalized domain string; adding a new domain is one new plugin (KPI definitions + optional driver columns), zero changes to the generic Stage 1-9 core.

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze` | Answer a business question against an uploaded CSV (or run report mode with no question — see below). Params: `file` (multipart upload), `business_question` (str — empty/generic strings like `"Analyze this dataset"` trigger multi-analysis report mode instead of a single curated-KPI answer), `conversation_id` (optional str — omit for a new conversation), `ml_readiness` / `llm_readiness` (float, default 99.75). Optional Agent 2 fields, all inert/gracefully-degrading if omitted or malformed: `feature_recommendation` (JSON string — simple per-column classification, used only to cross-check the hardcoded ML feature-column lists in `config/ml_config.yml`), `ml_readiness_breakdown` / `llm_readiness_breakdown` (JSON strings — Agent 2's full readiness assessment, surfaced in `execution_trace`'s gate objects), `column_profiles` (JSON string, a list — Agent 2's per-column semantic classification, builds the real `DatasetContext` instead of falling back to `LocalSchemaInferer`), `hierarchy` (JSON string, a dict — Agent 2's detected drill-down hierarchy), `charts` (JSON string, a list — Agent 2's pre-selected chart candidates), `full_feature_recommendation` (JSON string, a dict — target_column/problem_type/recommended_approach/feature-drop columns/confidence), `detected_domain` (str — Agent 1's business_domain classification; selects the matching `DomainPlugin`, or `GenericDomainPlugin` if none matches). |
| GET | `/health` | Liveness check — no downstream agents to ping. |

### Response shape

```json
{
  "status": "ok",
  "query": "Forecast underwriting result for next 6 months",
  "response": "## Summary\n...",
  "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "ml_readiness_score_used": 82.0,
  "llm_readiness_score_used": 95.77,
  "execution_trace": [
    {"step": "intent_detection", "engine": null, "gate": null, "reason": "Detected intent='forecast', kpi='underwriting_result'", "duration_ms": 4.6},
    {"step": "forecast", "engine": "Prophet", "gate": {"name": "ml_readiness", "score": 82.0, "threshold": 75.0, "passed": true, "breakdown": {"evidence": [{"dimension": "feature_coverage", "value": 0.96}], "strengths": [], "blocking_issues": []}}, "reason": "ML readiness (82.0%) met the 75.0% threshold — using the trained Prophet model. Strongest/only factor: feature_coverage (96%).", "duration_ms": 360.3, "model_version": {"refit_per_query": true, "last_run_at": "2026-07-17T22:24:02"}},
    {"step": "narration", "engine": "Azure OpenAI", "gate": {"name": "llm_readiness", "score": 95.77, "threshold": 75.0, "passed": true, "breakdown": null}, "reason": "LLM readiness (95.8%) met the 75.0% threshold — narrated by Azure OpenAI.", "duration_ms": 270.7}
  ],
  "execution_summary": {
    "intent": "forecast", "tools_used": ["RuleEngine", "SQLTool", "Prophet", "ExplanationTool"],
    "ml_engine": "Prophet", "narration_engine": "Azure OpenAI", "execution_time_seconds": 0.663, "fallback_used": false
  }
}
```

`narration_engine`/the narration step's `engine` is `"Azure OpenAI"` on a real LLM narration, or `"Template Formatter"` / `"Template Formatter (Azure OpenAI error)"` on the deterministic fallback (`ExplanationTool` in `app/services/tools/explanation_tool.py`) — `fallback_used` is only ever `true` because of a genuine ML-readiness gate miss or a narration fallback.

`execution_trace`/`execution_summary` are built once from the LangGraph run's final state — `null` on `status: "error"` rather than fabricating an explanation for a genuine crash. One step per scheduled analysis: a curated-KPI query (`kpi_summary`/`kpi_variance`/`root_cause`/`trend`/`forecast`) or a keyword-narrowed generic query schedules exactly one, so `execution_trace` looks exactly like the example above; a bare/generic `business_question` against a dataset with no matching curated KPI runs **report mode** — one step per scheduled analysis (up to the budget), `execution_summary.intent` is `"report"`, and `response` has one narrated section per analysis type. `gate.breakdown` is `null` when the caller didn't supply `ml_readiness_breakdown`/`llm_readiness_breakdown`. `model_version` only appears on an ML-gated step that actually ran a model — never on the deterministic-fallback path. `duration_ms` on every step is real per-node wall-clock time from `graph.stream()`.

Normally called by [`Agent-Orchestrator`](../Agent-Orchestrator) as the pipeline's optional third stage — see the [root API reference](../API_REFERENCE.md#agent-3--analytics-agent-optional-third-stage) for that integration, including the `column_profiles`/`hierarchy`/`charts`/`detected_domain` forwarding that builds a real `DatasetContext` and selects the right `DomainPlugin`. The orchestrator never passes `conversation_id` (it's stateless), so multi-turn memory only applies to a direct `POST /analyze` caller. Can also be exercised locally without an HTTP server via `scripts/cli.py` (see below).

## Conversation Memory

Backed by Postgres — one table (`agent3.conversation_turns`) on the same [`Shared-Postgres`](../Shared-Postgres) instance Agent 1/2 already use, isolated in its own schema. This matters more than it might look: `run_analytics_graph()` builds a brand-new graph (and therefore a brand-new `MemoryManager`) on *every single HTTP request* — without real persistence there was no way for two separate `POST /analyze` calls to share history at all, restart or no restart. `conversation_id` is what ties requests back to the same history.

- Pass the same `conversation_id` on the next call to continue a conversation — filter/KPI carryover ("what about EMEA?" after an FY2025 GWP question) and the LLM narrator's prior-turn context both depend on this. Filter/KPI carryover is only meaningful for a domain plugin with a curated KPI catalog (Insurance today) — a generic dataset's report-mode requests don't have a "current KPI" to carry forward.
- History survives service restarts.
- **Non-fatal if Postgres is down**: `init_db()` at startup logs an error and continues rather than crashing the service; `MemoryManager` degrades to in-process-only, non-persistent history for that request if a load or save fails.
- Schema/table are created automatically at startup (`app/services/database.py::init_db()`, called from `app/main.py`'s lifespan) — nothing to migrate by hand.

## Local Setup

This project shares one virtual environment with the rest of the pipeline — set up once from the **repo root** (see the [root README](../README.md)), not from inside this folder:

```bash
cd ..
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cd Analytics-Agent

# Copy env file and add your Azure OpenAI credentials
cp .env.example .env

..\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8003
```

Needs the shared Postgres running for conversation memory — a native Windows Postgres instance now (not Docker; `C:\PGData\mva-pipeline`, port 5433), started automatically by the root [`start-all.ps1`](../start-all.ps1), or directly via `pg_ctl -D C:\PGData\mva-pipeline start`. See [Conversation Memory](#conversation-memory) above. Not a hard requirement: if it's unreachable, the service still starts and answers questions, just without memory persisting.

### Standalone CLI (no HTTP server needed)

```bash
..\venv\Scripts\python.exe scripts/cli.py --query "Show Gross Written Premium for FY2025"
..\venv\Scripts\python.exe scripts/cli.py --interactive
```

Reads the dataset from `DATASET_PATH` (see Environment Variables) instead of an upload. Each query calls the graph fresh, same as a real HTTP request — but `--interactive` mode generates one `conversation_id` at startup and reuses it across turns, so memory carries forward within a session the same way a real client passing the same `conversation_id` back on each `POST /analyze` call would.

## Environment Variables

`AZURE_OPENAI_API_KEY`/`AZURE_OPENAI_ENDPOINT`/`AZURE_OPENAI_DEPLOYMENT` and the `POSTGRES_*` connection details default from the shared **repo-root** `.env` (`../.env`) now — see the [root README](../README.md#quick-start). This service's own `.env` only needs to set them if overriding the shared value specifically for Agent 3.

| Variable | Default | Description |
|----------|---------|-------------|
| AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT *(root `.env`)* | *(empty)* | Azure OpenAI credentials — used for narration. Falls back to a deterministic template formatter if unset/unreachable/rate-limited. |
| HOST | 0.0.0.0 | Bind host. |
| PORT | 8003 | Bind port. |
| DATASET_PATH | `../test_data/insurance_variance_data_native.csv` | **Local testing only** — used by `scripts/cli.py` and `train.py`. Has no effect on `POST /analyze`, which always uses the uploaded file. |
| POSTGRES_HOST *(root `.env`)* | localhost | Shared Postgres host — same instance as Agent 1/2. |
| POSTGRES_PORT *(root `.env`)* | 5433 | Shared Postgres port. |
| POSTGRES_DB *(root `.env`)* | mva_pipeline | Shared database name. |
| POSTGRES_USER *(root `.env`)* | postgres | Connects as the `postgres` superuser, like Agent 1 — not a dedicated role like Agent 2's `mva_user`, since `agent3`'s schema is created idempotently by this service's own `init_db()` at every startup. |
| POSTGRES_PASSWORD *(root `.env`)* | postgres | |

Domain config lives in `config/*.yml`/`config/rules/*.json`/`app/services/domain_plugins/*/`, not environment variables — see `app/config.py`'s module docstring for the full priority order. `config/model_registry.yml` (Stage 6's algorithm-selection catalog — every ML and deterministic algorithm, keyed by analysis type) is distinct from `ml/model_registry.json` (runtime metadata: last-trained/last-run timestamps, read by `execution_trace`'s `model_version` field).

## Model Training

```bash
..\venv\Scripts\python.exe train.py                      # train all 4 models
..\venv\Scripts\python.exe train.py --model xgboost       # train one model
..\venv\Scripts\python.exe train.py --confusion-matrix    # show XGBoost confusion matrix
```

Trains Prophet-free LightGBM, IsolationForest, XGBoost, and K-Means against `DATASET_PATH`, saving artifacts to `ml/trained/*.pkl` (query-time code predicts against these persisted models instead of refitting per request). `app/main.py`'s startup lifespan runs `app/services/boot_trainer.py`'s freshness check once per process lifetime — if `DATASET_PATH` is newer than the saved models (or any are missing), it retrains automatically before the service starts accepting requests. **Insurance-only today** — per-dataset model lifecycle (fit-fresh or cache by dataset fingerprint for a non-Insurance upload) is planned but not yet built; every non-Insurance ML analyzer fits fresh on each request until then.

## Running Tests

```bash
..\venv\Scripts\python.exe -m pytest tests/ -v
```

19 files, 304 tests. Covers every stage of the pipeline above: capability resolution, KPI discovery, question interpretation, planning, scheduling, the model registry/selector, all 16 analyzers (+ the 2 curated-KPI analyzers), the domain plugin system (including a byte-for-byte check that the Insurance plugin's copied config files match the originals), the evidence builder, LangGraph routing, the execution trace/summary builder, Postgres-backed conversation memory (against the real shared instance), and a 12-case end-to-end harness exercising every curated-KPI query shape (real Azure OpenAI calls — slower and occasionally rate-limit-prone, but confirms the whole path works, not just its parts).

## How Scoring Works

Agent 3 **never computes ML or LLM readiness itself** — it only ever
consumes the two numbers Agent 2 already computed
(`ml_readiness_score`/`llm_readiness_score`, forwarded by the
Orchestrator as Agent 2's plain composite `.score`, not the
`dataset_score`/`task_compatibility_score` split — see
`Data-Profiling-Agent/README.md`'s AI Readiness section for the full
formula those numbers come from). Everything below is what Agent 3 does
*with* those two numbers, plus its own purely structural/scheduling logic.

**Stage 1 — Capability resolution (structural vs. execution).** For each
of 8 capabilities (forecasting, clustering, classification, regression,
segmentation, correlation, association_rules, time_series),
`AnalyticsCapabilityResolver` reports two independent verdicts:
- `structural.supported` — can this analysis mechanically run at all,
  given the dataset's shape (right column types present, enough rows)?
  Pure `DatasetContext` question, zero relation to any readiness score.
- `execution.supported` — *if* structurally possible, does
  `ml_readiness_score` clear `ML_READINESS_THRESHOLD` (75.0,
  `app/config.py`, read from `ml_config.yml`'s `readiness_threshold`)? If
  so, `execution.confidence = ml_readiness_score / 100.0` — Agent 2's own
  number re-expressed as a fraction, never recomputed.

`AnalyticsPlanner` (Stage 4) only ever reads `structural.supported` — an
analysis gets *planned* whenever it's mechanically possible, full stop.
`ModelSelector` (Stage 6) reads `execution.supported` (further gated by
the Scheduler's ML budget below) to decide real model vs. deterministic
strategy for an already-planned analysis. `llm_readiness_score` never
gates an analysis — only narration (Stage 9): below
`LLM_READINESS_THRESHOLD` (also 75.0), `ExplanationTool` skips the Azure OpenAI
call entirely and goes straight to the deterministic template formatter.

**Stage 5 — Scheduling budget** (`config/scheduling_budget.yml`):

| Budget | Value | What it bounds |
|---|---|---|
| `max_parallel_analyses` | 8 | Total analyses executed per request, after prioritizing requested → KPI-grounded → by `PlannedAnalysis.priority`. Requested (question-matched) analyses are never trimmed by this cap, even if they alone exceed it. |
| `max_ml_analyses` | 3 | Of the scheduled analyses, only this many (same priority order) may attempt a real ML-backed strategy — `ml_execution_allowed` can only ever downgrade toward a cheaper/deterministic strategy, never upgrade past what Stage 1 already determined viable. |
| `max_expensive_operations` | 2 | Of the ML-allowed set, only this many may select an algorithm tagged `cost_tier: "expensive"` in `config/model_registry.yml` (~35 registered algorithms, cheap/moderate/expensive) — the rest select the cheapest viable algorithm for their analysis type, enforced by `ModelSelector`, not the Scheduler itself. |

This is the explicit, load-bearing safeguard against pathological blowup
on wide datasets — without it, a 50+ column upload could otherwise
trigger dozens of model fits in one request.

**Why this design keeps Agent 2 as the single source of truth for
readiness**: `AnalyticsCapabilityResolver`'s function signature never
receives the raw DataFrame or Agent 2's evidence/strengths/blocking_issues
— only the two already-computed scores plus `DatasetContext` — so there
is no code path in Agent 3 that can silently reimplement or drift from
Agent 2's scoring logic.

## Known Limitations

Fixed during the handover pass: `knowledge_update_service.py`'s broken
`from tools.schemas import KPIDefinition` import (silently swallowed by a
broad `except Exception`, making auto-KPI-generation dead for every
domain) now correctly imports from `app.services.schemas`; upload size
limits and a `business_question` length cap were added to `POST
/analyze` (previously only Agent 1 enforced any upload limit); explicit
30s timeouts were added to the two direct Azure OpenAI calls in
`explanation_tool.py`/`knowledge_update_service.py`; `KPISummaryStrategy`/
`KPIVarianceStrategy` now return an explicit `{"error": "..."}` naming
the missing column instead of silently returning near-empty evidence when
a curated KPI's expected columns aren't present in the uploaded dataset
(the single most likely first complaint on the new starter plugins);
`graph.py`'s error path no longer leaks raw exception text into the
caller's response (logged server-side instead); every `open()` across the
domain-plugin/rule-engine JSON/YAML loaders now sets `encoding="utf-8"`
explicitly (Windows otherwise silently uses the system code page, a
latent bug for the first non-ASCII KPI label anyone adds).

Still open — real findings that need a conscious decision rather than a
same-session patch:

- **Real ML models (Prophet/LightGBM/IsolationForest/XGBoost/K-Means) are
  trained once against `DATASET_PATH` (Insurance) and persisted** — a
  non-Insurance upload's ML-eligible analyses always fit fresh per
  request rather than predicting against a cached model. Deterministic
  fallbacks (the large majority of what a generic dataset actually gets,
  since ML requires clearing the 75.0 readiness threshold) are
  unaffected. Per-dataset model caching keyed by content fingerprint is
  the natural fix, not yet built.
- **ML feature-column lists** (`config/ml_config.yml` — which ratio
  columns feed IsolationForest/KMeans, which columns LightGBM treats as
  categorical) are hand-curated Insurance-domain expertise, not derived
  from Agent 2's `feature_recommendation` — used only as a diagnostic
  cross-check, never to swap which columns a model actually uses. Only
  affects Insurance's own ML paths; a non-Insurance dataset's generic
  analyzers (Stage 7) derive their feature columns from `DatasetContext`
  directly.
- Only CSV uploads, no Excel (`SQLTool` reads via DuckDB's
  `read_csv_auto`, a genuine capability limit — see
  `Agent-Orchestrator/README.md`'s `Agent3Capabilities` gate).
- No authentication in v1.
- Multi-turn conversation memory only works when a caller explicitly
  passes `conversation_id` back on each request — `Agent-Orchestrator`
  doesn't do this today (it's intentionally stateless), so
  pipeline-driven questions each start a fresh conversation. Only a
  direct `POST /analyze` caller (or `scripts/cli.py --interactive`) gets
  continuity.
- Finance/HR/Payments/Customer now have thin starter domain plugins
  (curated KPI catalog + `kpi_summary`/`kpi_variance` only, see the
  Domain Enhancement Layer table above) — none have driver columns or
  ML-feature-column overrides yet, so labeled-mode root cause and
  Insurance-style ML corroboration still only apply to Insurance. A
  domain with no registered plugin at all still gets
  `GenericDomainPlugin`'s fully functional generic report.
- Blocking synchronous I/O (Azure OpenAI calls, DuckDB queries) runs directly on
  this service's async event loop — same cross-service issue documented
  in `Schema-Intelligence-Layer/README.md`'s Known Limitations; not fixed
  here for the same reason (deserves one general fix across all 4
  services, not four independent partial patches).

## Decisions Log

| Date | Decision |
|---|---|
| 2026-07-15 | Groq API chosen for faster inference; DuckDB chosen for SQL (no database server needed); all monetary values in USD. |
| (this copy) | Groq replaced with Azure OpenAI throughout — this copy of the project runs on the company's Azure OpenAI resource instead. |
| 2026-07-17 | Vendored into the `mva` monorepo as Agent 3, initially wired into `Agent-Orchestrator` as a CLI subprocess. |
| 2026-07-17 | Fixed `eval()`-based rule evaluation (security hole + a dormant bug — 4 rules using uppercase `AND` had silently never fired since inception) with an AST-whitelist evaluator; added real model persistence (`ml/persistence.py`); wired in the previously-dead `MemoryManager`; added `llm_readiness` gating (mirrors `ml_readiness`) and `ml/feature_validation.py`. |
| 2026-07-17 | Rebuilt as a full FastAPI + LangGraph service (`app/main.py`, `app/routes/`, `app/agents/analytics_agent/`, `app/services/`) to match Agent 1/2/3's shared architecture, and `Agent-Orchestrator`'s `call_agent3` rewired from a subprocess invocation to a plain `httpx` call. The old `main.py`/`tools/`/flat `ml/*.py` CLI structure was retired once the new service was verified working end-to-end. |
| 2026-07-18 | Added a Postgres schema (`agent3`) and moved conversation memory from an in-process, always-empty-per-request list to real cross-request persistence keyed by `conversation_id`. |
| 2026-07-18 | Fixed `AnalyticsTool.variance()` computing favorable/unfavorable purely from the arithmetic sign, never reading each KPI's own `higher_is_better` flag — every "lower is better" ratio KPI had its variance direction backwards. |
| 2026-07-18 | Added `execution_trace`/`execution_summary` to every `POST /analyze` response — a step-by-step decision log built once from the graph's final state. |
| 2026-07-18 | Enriched the trace further: Agent 2's full readiness assessment flows into `execution_trace`'s gate objects; real per-step `duration_ms` via `graph.stream(..., stream_mode="updates")`; `model_version` metadata from `ml/model_registry.json`. |
| 2026-07-21 (Phase 0) | Began the Agent 3 redesign — a domain-agnostic, dataset-driven analytics engine, not an Insurance-only intent dispatcher. Added `DatasetContext`/`DatasetContextBuilder`/`LocalSchemaInferer` and the Orchestrator's additive forwarding of `column_profiles`/`hierarchy`/`charts`/`feature_recommendation`/`detected_domain` — inert this phase, every existing handler unchanged. |
| 2026-07-21 (Phase 1) | Built Stages 1-6 offline (not yet wired to execution): `AnalyticsCapabilityResolver` (structural/execution split, never recomputes Agent 2's readiness scores — enforced at the function signature), `SemanticKPIDiscovery`, `BusinessQuestionInterpreter`, `AnalyticsPlanner`/`AnalyticsScheduler` (WHAT vs. WHEN/HOW MANY, budget-bounded), `ModelRegistry`/`ModelSelector` (every algorithm, ML or deterministic, registers the same way in `config/model_registry.yml` — never a fixed intent→model dict). |
| 2026-07-21 (Phase 2) | Extracted Insurance's hardcoded content into `DomainPlugin`/`PluginRegistry`/`InsurancePlugin` — generalized `RuleEngine`/`RootCauseTool`/`SQLTool` to accept config/columns as constructor params, defaulting to Insurance's exact values. Verified byte-identical via a live 4-service diff against a pre-refactor snapshot. |
| 2026-07-22 (Phase 3) | Built all 16 `Analyzer` classes (Stage 7) and `Evidence`/`EvidenceBuilder` (Stage 8), plus ~25 new deterministic/ML strategy classes the registry already referenced — root cause gained a generic correlation-based mode (deterministic primary, unchanged for datasets with known driver columns) alongside Insurance's exact labeled-driver mode. |
| 2026-07-22 (Phase 4) | Replaced the old 7-handler intent-dispatch graph with the linear Stage 1-9 topology above. Closed a gap the registry-driven design hadn't accounted for — curated-KPI queries (`kpi_summary`/`kpi_variance`, ports of the old `show_kpi`/`variance` handlers) and wired `ModelSelector` to actually consult a domain plugin's preferred deterministic strategy (built in Phase 2, never called until now). Live end-to-end testing against a non-Insurance HR dataset surfaced and fixed 4 real bugs: a keyword-interpreter/scheduler mismatch that split one query into four; a segmentation strategy returning an unnested evidence shape; **no true "no domain matched" fallback** (any unlabeled dataset was silently inheriting Insurance's KPI catalog — `GenericDomainPlugin` fixes this); and a column-order mismatch between the generic planner and the forecast/trend analyzers (visible as `1970-01-01` epoch dates in forecast output). Golden-snapshot diff against the pre-redesign response: only the deliberate `show_kpi`→`kpi_summary` rename differs, every deterministic evidence number matches exactly. |
| 2026-07-22 (Phase 4 follow-up) | Live-testing bug fixes against a Finance dataset with no domain plugin: `rule_kpi_grounded_root_cause` was proposing a discovered KPI's *entire* `source_columns` list as root-cause "drivers" — including its own budget/reference column — trivially self-explaining "100% of the variance"; narrowed to `[source_columns[0]]` only. Added `## Group Comparison` rendering to `ExplanationTool` for `ComparativeAnalyzer` evidence, previously silently dropped by the generic scalar-only renderer. Added heuristic driver-name-pattern narrowing (`variance`/`driver`/`contribution`/`impact`) to correlation-based root cause's candidate pool. |
| 2026-07-22 (Phase 4.5) | Opened the Orchestrator's Agent 3 gate to all 5 of Agent 2's supported domains — the last place in the system still encoding "Agent 3 = Insurance-only," a stale assumption Phase 4 had already invalidated everywhere else. Replaced the domain check with `Agent3Capabilities` (file type + `business_question` only — both genuine capability limits, not domain assumptions). Live-verified all 5 domains through the real pipeline; found and fixed a related gap where Agent 1's open-vocabulary classifier labels CRM/loyalty datasets `"E-commerce"` as often as `"Customer"` — added as a domain synonym. |
| 2026-07-22 (Phase 4.6) | Live-testing surfaced two further bugs against a Finance dataset: (1) `rule_success_rate`'s KPI-discovery check accepted a column as a transaction status whenever Agent 2 labeled its type generically `"status"` — a balance-sheet flag got picked up this way, producing a nonsensical "Success Rate" KPI that won the discovery race ahead of the dataset's real KPIs; fixed by requiring a genuinely status-*named* column. (2) Root-cause/comparative-analysis planning rules had no way to know which column a question was actually about, always defaulting to "first structurally valid column in the dataset" — added `QuestionIntent.preferred_metrics`/`preferred_dimensions`, populated by `BusinessQuestionInterpreter` via a phrase match against the dataset's own column names, consumed by the two affected planning rules ahead of their structural default. Verified end-to-end against the real data: every reported correlation/aggregate number in the fixed response matched hand-recomputation from the raw CSV exactly. |
