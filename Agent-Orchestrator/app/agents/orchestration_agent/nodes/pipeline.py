"""Orchestrator graph nodes.

Mirrors the exact HTTP call sequence and error handling the old linear
run_pipeline() function used (app/routes/pipeline.py, pre-graph version) —
this file does not change what gets called or how errors are classified,
it only re-expresses each stage as a graph node with an explicit conditional
edge deciding whether the pipeline stops or continues, instead of a chain of
early `return`s inside one function body.
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
import pandas as pd

from app.config import settings
from app.agents.orchestration_agent.state import PipelineState
from app.services.dataset_registry.dataset_registry import DatasetRegistry

logger = logging.getLogger(__name__)

_dataset_registry: DatasetRegistry | None = None


def get_dataset_registry() -> DatasetRegistry:
    """Stage 0A facade, lazily constructed once per process — stateless
    beyond the configured storage dir, safe to share across requests."""
    global _dataset_registry
    if _dataset_registry is None:
        _dataset_registry = DatasetRegistry(settings.MASTER_DATASET_STORAGE_DIR)
    return _dataset_registry

# Agent 2's completeness/uniqueness quality dimensions are rule-driven, not
# computed from raw data automatically (see Data-Profiling-Agent's
# app/services/quality/completeness.py / uniqueness.py) -- they only assess
# columns explicitly flagged mandatory/expected_unique in schema_metadata.
# This used to hardcode both flags False for every column, meaning no
# upload -- however clean -- could ever have completeness/uniqueness
# assessed through the normal pipeline, permanently capping ml_readiness
# ~30 points below what a genuinely clean, well-labeled dataset should
# earn. Agent 1 doesn't expose per-column semantic roles (that's Agent 2's
# own job, running *after* this request), so the only signal available at
# this integration point is the column name itself and its LLM-written
# description -- a real, if simple, heuristic rather than no signal at all.
_IDENTIFIER_NAME_RE = re.compile(r"(^|_)(id|identifier|uuid|guid)$", re.IGNORECASE)


def _infer_mandatory_and_unique(column_name: str, description: str) -> tuple[bool, bool]:
    """Identifier-shaped columns (name ends in id/identifier/uuid/guid, or
    the LLM description explicitly calls it a unique identifier) are
    expected_unique, never mandatory in the completeness sense -- a missing
    ID usually means a missing row, not an incomplete one. Every other
    column defaults to mandatory: a substantive business column that's
    mostly null is a genuine completeness problem worth scoring."""
    is_identifier_like = bool(_IDENTIFIER_NAME_RE.search(column_name)) or "unique identifier" in (description or "").lower()
    return (not is_identifier_like, is_identifier_like)


# Agent 2's "consistency" quality dimension needs an actual cross-field
# business rule (column_comparison) -- unlike mandatory/expected_unique,
# there's no safe way to *guess* one from a column's name alone (asserting
# a false relationship, e.g. "revenue >= marketing_spend", would actively
# score a wrong claim as quality). But a relationship that genuinely,
# empirically holds across (near) every row of the real uploaded data
# isn't a guess -- it's an observed fact about this specific file. This
# scans numeric column pairs for exactly that: a >=, <=, or == relationship
# holding for at least 99% of jointly-valid rows, real sample size
# required, near-constant/flag-like columns excluded (a binary flag
# trivially "holds" against everything, which isn't a meaningful
# invariant), capped at 2 rules so a wide dataset doesn't get flooded with
# spurious asserted relationships. Best-effort and non-fatal throughout --
# parsing failures or a dataset with no qualifying pair just mean no
# auto-inferred rules, never a broken pipeline.
_MIN_ROWS_FOR_CONSISTENCY_INFERENCE = 30
_CONSISTENCY_INVARIANT_THRESHOLD = 0.99
_MAX_AUTO_INFERRED_CONSISTENCY_RULES = 2
_FLAG_LIKE_MAX_DISTINCT_VALUES = 3


def _read_dataframe_best_effort(filename: str, content: bytes, sheet_name: str | None) -> pd.DataFrame | None:
    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(content), sheet_name=sheet_name or 0)
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        return None


def _infer_consistency_rules(filename: str, content: bytes, sheet_name: str | None) -> list[dict[str, Any]]:
    df = _read_dataframe_best_effort(filename, content, sheet_name)
    if df is None or len(df) < _MIN_ROWS_FOR_CONSISTENCY_INFERENCE:
        return []

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if not _IDENTIFIER_NAME_RE.search(c) and df[c].nunique(dropna=True) > _FLAG_LIKE_MAX_DISTINCT_VALUES
    ]

    candidates: list[tuple[float, dict[str, Any]]] = []
    for i, left in enumerate(numeric_cols):
        for right in numeric_cols[i + 1:]:
            both_valid = df[left].notna() & df[right].notna()
            total = int(both_valid.sum())
            if total < _MIN_ROWS_FOR_CONSISTENCY_INFERENCE:
                continue
            lv, rv = df.loc[both_valid, left], df.loc[both_valid, right]
            ge_frac = float((lv >= rv).mean())
            le_frac = float((lv <= rv).mean())

            if ge_frac >= _CONSISTENCY_INVARIANT_THRESHOLD and le_frac >= _CONSISTENCY_INVARIANT_THRESHOLD:
                operator, frac = "==", min(ge_frac, le_frac)
            elif ge_frac >= _CONSISTENCY_INVARIANT_THRESHOLD:
                operator, frac = ">=", ge_frac
            elif le_frac >= _CONSISTENCY_INVARIANT_THRESHOLD:
                operator, frac = "<=", le_frac
            else:
                continue

            op_slug = {"==": "eq", ">=": "ge", "<=": "le"}[operator]
            candidates.append((frac, {
                "type": "column_comparison",
                "rule_key": f"auto_inferred_{left}_{op_slug}_{right}",
                "left_column": left,
                "right_column": right,
                "operator": operator,
                "severity": "medium",
            }))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [rule for _frac, rule in candidates[:_MAX_AUTO_INFERRED_CONSISTENCY_RULES]]

# Agent 1's classification is open-vocabulary LLM output (its prompt only
# *suggests* 14 domain names, e.g. "Human Resources" — see
# Schema-Intelligence-Layer/app/prompts/llm_service_prompt.py); Agent 2's
# domain config lookup is an exact, case-sensitive match against 5 fixed
# strings (Finance/Payments/Customer/HR/Insurance, read from each
# config/domains/*.yaml's own `domain:` field). Without this, a perfectly
# correct classification like "Human Resources" fails Agent 2's check for
# no reason other than wording. Maps known synonyms onto Agent 2's exact
# strings; anything unrecognized passes through unchanged so Agent 2's own
# UNSUPPORTED_DOMAIN error still fires for genuinely unsupported domains.
_DOMAIN_SYNONYMS = {
    "finance": "Finance",
    "financial": "Finance",
    "financial services": "Finance",
    "insurance": "Insurance",
    "payments": "Payments",
    "payment": "Payments",
    "payment processing": "Payments",
    "customer": "Customer",
    "crm": "Customer",
    "customer relationship management": "Customer",
    "e-commerce": "Customer",
    "ecommerce": "Customer",
    "sales": "Customer",
    "retail": "Customer",
    "hr": "HR",
    "human resources": "HR",
    "human resource": "HR",
}

# Fallback for whatever wording _DOMAIN_SYNONYMS's exact-match list hasn't
# hit yet. Agent 1's classification is LLM-driven and non-deterministic --
# it keeps inventing new phrasings for domains Agent 2 already supports
# ("E-commerce", then "Sales", for the same underlying retail dataset,
# confirmed live both times this session) instead of settling on one. An
# exact-match dict alone turns every new phrasing into a fire drill; these
# keyword patterns catch the vocabulary a domain is reliably described
# with, checked only when the exact match above misses.
_DOMAIN_KEYWORD_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(insur|underwrit|claim|policy|premium)\w*", re.IGNORECASE), "Insurance"),
    (re.compile(r"\b(financ|bank|accounting)\w*", re.IGNORECASE), "Finance"),
    (re.compile(r"\b(payment|billing|transaction processing)\w*", re.IGNORECASE), "Payments"),
    (re.compile(r"\b(human resource|payroll|employee|workforce|recruit)\w*", re.IGNORECASE), "HR"),
    (re.compile(r"\b(retail|sales|e-?commerce|customer|shopping|store)\w*", re.IGNORECASE), "Customer"),
]


def _canonicalize_domain(raw_domain: str) -> str:
    """Normalize Agent 1's classification wording onto Agent 2's exact
    supported-domain strings before it's used anywhere downstream."""
    normalized = raw_domain.strip().lower()
    if normalized in _DOMAIN_SYNONYMS:
        return _DOMAIN_SYNONYMS[normalized]
    for pattern, canonical in _DOMAIN_KEYWORD_PATTERNS:
        if pattern.search(raw_domain):
            return canonical
    return raw_domain


def _safe_json(resp: httpx.Response) -> dict[str, Any]:
    try:
        return resp.json()
    except ValueError:
        return {"raw_response": resp.text}


def _strip_raw_rows(agent1_body: dict) -> dict:
    """Agent 1's response embeds every row of the uploaded file
    (`dataframe_records`). The orchestrator doesn't need it (Agent 1 already
    persists it) and passing it through balloons the response to tens of MB
    for large files. Drop it before it goes anywhere else."""
    if isinstance(agent1_body, dict) and "dataframe_records" in agent1_body:
        agent1_body = {k: v for k, v in agent1_body.items() if k != "dataframe_records"}
    return agent1_body


@dataclass(frozen=True)
class Agent3Capabilities:
    """What Agent 3 can currently process — the one place this
    Orchestrator's Agent-3-routing logic reads from (Phase 4.5, plan
    "zany-giggling-crayon"). Hardcoded today, mirroring Agent 3's real
    constraints (its SQLTool only reads CSV via DuckDB's read_csv_auto;
    every analysis needs a business_question, even report mode's generic
    one) — not a domain restriction. Agent 3 itself became domain-agnostic
    in Phase 4 (a DomainPlugin per domain, GenericDomainPlugin as the
    fallback for anything unregistered); before Phase 4.5, this file was
    the *last* place still encoding "Agent 3 = Insurance-only" as a
    routing rule rather than a capability check.

    If Agent 3 later exposes this dynamically (e.g. its own
    GET /capabilities, once it supports more than one file type or an
    optional business_question), only this object's construction needs to
    change — _agent3_skip_reason() and both its call sites don't."""

    supported_file_extensions: frozenset[str] = field(default_factory=lambda: frozenset({".csv"}))
    requires_business_question: bool = True

    def unsupported_file_reason(self, filename: str) -> str | None:
        if not any(filename.lower().endswith(ext) for ext in self.supported_file_extensions):
            return (
                f"Analytics Agent only reads {'/'.join(sorted(self.supported_file_extensions))} "
                f"datasets (DuckDB read_csv_auto) — this upload isn't one."
            )
        return None

    def missing_question_reason(self, business_question: str | None) -> str | None:
        if self.requires_business_question and not business_question:
            return "Analytics Agent answers one business question at a time — no business_question was supplied."
        return None


AGENT3_CAPABILITIES = Agent3Capabilities()


def _agent3_skip_reason(filename: str, business_question: str | None) -> str | None:
    """None means Agent 3 should run; otherwise the string is why it won't.

    Deliberately domain-agnostic since Phase 4.5 — Agent 3 resolves its
    own DomainPlugin (or GenericDomainPlugin) from whatever
    `detected_domain` this request forwards; the Orchestrator no longer
    needs an opinion about which domains Agent 3 "supports"."""
    return (
        AGENT3_CAPABILITIES.missing_question_reason(business_question)
        or AGENT3_CAPABILITIES.unsupported_file_reason(filename)
    )


async def _get_agent2_result(run_id: str) -> tuple[int, dict[str, Any]]:
    """GET Agent 2's already-persisted full result for run_id — no new
    profiling work, just a read. Returns (status_code, body); raises only
    on a network-level failure (connection refused, timeout, ...) — what a
    non-200 status *means* differs by caller (fetch_agent2_result's node
    falls back to the abbreviated agent2_body on anything but 200, since
    run_id is known-valid there; /pipeline/ask treats a 404 as a genuine
    caller error, since the run_id came from outside this request)."""
    result_url = f"{settings.AGENT2_BASE_URL}{settings.AGENT2_API_PREFIX}/profile-runs/{run_id}/result"
    async with httpx.AsyncClient() as client:
        resp = await client.get(result_url, timeout=settings.REQUEST_TIMEOUT_SECONDS)
    return resp.status_code, _safe_json(resp)


async def _analyze_via_agent3(
    business_question: str,
    filename: str,
    content: bytes,
    content_type: str,
    ml_score: float | None,
    llm_score: float | None,
    feature_columns: list,
    ml_breakdown: dict | None = None,
    llm_breakdown: dict | None = None,
    column_profiles: list | None = None,
    hierarchy: dict | None = None,
    charts: list | None = None,
    full_feature_recommendation: dict | None = None,
    detected_domain: str | None = None,
) -> dict[str, Any]:
    """The actual call to Agent 3's POST /analyze. Shared by the full
    pipeline's call_agent3 node and the /pipeline/ask route — same request
    shape either way, they just differ in where ml_score/llm_score/
    feature_columns/breakdowns/dataset-context fields come from (a run just
    completed vs. re-fetched from Agent 2's persisted result by run_id).

    column_profiles/hierarchy/charts/full_feature_recommendation/
    detected_domain (Agent 3 redesign, Phase 0 — plan "zany-giggling-
    crayon", Decision A1): Agent 2's per-column semantic classification,
    forwarded so Agent 3 can build a rich DatasetContext instead of falling
    back to its own lightweight local inference. All optional/additive —
    Agent 3 degrades gracefully if any are omitted."""
    url = f"{settings.ANALYTICS_AGENT_BASE_URL}/analyze"
    data = {"business_question": business_question}
    if ml_score is not None:
        data["ml_readiness"] = ml_score
    if llm_score is not None:
        data["llm_readiness"] = llm_score
    if feature_columns:
        data["feature_recommendation"] = json.dumps(feature_columns)
    if ml_breakdown:
        data["ml_readiness_breakdown"] = json.dumps(ml_breakdown)
    if llm_breakdown:
        data["llm_readiness_breakdown"] = json.dumps(llm_breakdown)
    if column_profiles:
        data["column_profiles"] = json.dumps(column_profiles)
    if hierarchy:
        data["hierarchy"] = json.dumps(hierarchy)
    if charts:
        data["charts"] = json.dumps(charts)
    if full_feature_recommendation:
        data["full_feature_recommendation"] = json.dumps(full_feature_recommendation)
    if detected_domain:
        data["detected_domain"] = detected_domain

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                files={"file": (filename, content, content_type)},
                data=data,
                timeout=settings.ANALYTICS_AGENT_TIMEOUT_SECONDS,
            )
    except httpx.HTTPError as e:
        return {"status": "failed", "reason": f"Could not reach Analytics Agent at {url}: {e}"}

    agent3_body = _safe_json(resp)

    if resp.status_code != 200:
        return {"status": "failed", "reason": f"Analytics Agent returned {resp.status_code}: {agent3_body}"}

    return agent3_body


def _readiness_and_features(
    agent2_full_result: dict,
) -> tuple[float | None, float | None, list, dict | None, dict | None]:
    """Extract (ml_score, llm_score, feature_columns, ml_breakdown, llm_breakdown)
    from Agent 2's full result — shared by call_agent3 and /pipeline/ask,
    same extraction either way.

    ml_breakdown/llm_breakdown are the full readiness assessment dicts
    (strengths, blocking_issues, evidence, recommendations,
    weight_profile_version) — Agent 2's ReadinessEngine already computes
    these, but until now only the bare score float was ever forwarded to
    Agent 3, so it could never explain *why* a gate passed or failed."""
    readiness_assessments = (agent2_full_result or {}).get("readiness_assessments", [])
    ml_assessment = next(
        (r for r in readiness_assessments if r.get("assessment_type") == "ml_readiness"), None,
    )
    llm_assessment = next(
        (r for r in readiness_assessments if r.get("assessment_type") == "llm_readiness"), None,
    )
    ml_score = ml_assessment.get("score") if ml_assessment else None
    llm_score = llm_assessment.get("score") if llm_assessment else None
    feature_columns = (agent2_full_result or {}).get("feature_recommendation", {}).get("feature_columns") or []
    return ml_score, llm_score, feature_columns, ml_assessment, llm_assessment


def _dataset_context_fields(
    agent2_full_result: dict,
) -> tuple[list | None, dict | None, list | None, dict | None]:
    """Extract (column_profiles, hierarchy, charts, full_feature_recommendation)
    from Agent 2's full result — the per-column semantic classification a
    domain-agnostic Agent 3 needs (Decision A1, plan "zany-giggling-crayon").
    Distinct from _readiness_and_features() above: that extracts the
    readiness-gating scores/breakdowns, this extracts the dataset-shape
    understanding those gates don't carry. Shared by call_agent3 and
    /pipeline/ask, same extraction either way."""
    result = agent2_full_result or {}
    column_profiles = result.get("column_profiles") or None
    hierarchy = result.get("hierarchy") or None
    charts = result.get("charts") or None
    full_feature_recommendation = result.get("feature_recommendation") or None
    return column_profiles, hierarchy, charts, full_feature_recommendation


class OrchestratorGraphNodes:
    """Each node is a bound async method — grouped in a class purely for
    namespacing, no shared mutable state between requests (a fresh
    httpx.AsyncClient is opened per HTTP call, same lifecycle as before)."""

    # ── Node 0A: Dataset Registry — fingerprint, duplicate/version detection ──
    # Not an AI agent — deterministic infrastructure (see the architecture
    # doc's "Dataset Registry & Lifecycle Management" section). On a
    # duplicate upload with an already-cached pipeline result, this node
    # fabricates agent1_body/agent2_full_result/primary_domain directly —
    # the exact three fields finalize() already reads off state — so
    # call_agent1/call_agent2/fetch_agent2_result never run at all. A
    # genuinely new dataset, a new version, or a force_revalidate/
    # force_reclassify request falls through to the existing chain
    # unchanged.

    async def resolve_dataset_identity(self, state: PipelineState) -> dict[str, Any]:
        file_extension = Path(state["filename"]).suffix or ""

        try:
            result = get_dataset_registry().register(
                filename=state["filename"], file_extension=file_extension, content=state["content"],
            )
        except Exception as e:
            # Non-fatal degrade: Postgres/storage unavailable -> cache
            # always misses, pipeline runs exactly as it did before this
            # feature existed.
            logger.warning(f"Dataset Registry unavailable — running without cache: {e}")
            return {"file_extension": file_extension, "fingerprint": None, "copy_id": None, "was_cached": False}

        force = bool(state.get("force_revalidate") or state.get("force_reclassify"))

        if result.was_duplicate and result.cached_result is not None and not force:
            return {
                "file_extension": file_extension,
                "fingerprint": result.master.fingerprint,
                "copy_id": result.copy.copy_id,
                "was_cached": True,
                "agent1_body": result.cached_result.agent1_body,
                "agent2_full_result": result.cached_result.agent2_full_result,
                "primary_domain": result.cached_result.primary_domain,
            }

        return {
            "file_extension": file_extension,
            "fingerprint": result.master.fingerprint,
            "copy_id": result.copy.copy_id,
            "was_cached": False,
            # force_revalidate reaching Agent 1 as force_reclassify is what
            # makes "correct a bad past classification" actually work end
            # to end, instead of only bypassing this node's own cache.
            "force_reclassify": force,
        }

    def route_after_identity(self, state: PipelineState) -> Literal["cache_hit", "cache_miss"]:
        return "cache_hit" if state.get("was_cached") else "cache_miss"

    # ── Node 1: Agent 1 (Schema Intelligence Layer) ─────────────────────────

    async def call_agent1(self, state: PipelineState) -> dict[str, Any]:
        url = f"{settings.AGENT1_BASE_URL}/upload-dataset"
        params = {"force_reclassify": "true"} if state.get("force_reclassify") else {}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    files={"file": (state["filename"], state["content"], state["content_type"])},
                    params=params,
                    timeout=settings.REQUEST_TIMEOUT_SECONDS,
                )
        except httpx.HTTPError as e:
            return {
                "error_status_code": 502,
                "error_content": {
                    "stage": "agent1_unreachable",
                    "detail": f"Could not reach Agent 1 (Schema Intelligence Layer) at {url}: {e}",
                },
            }

        agent1_body = _strip_raw_rows(_safe_json(resp))

        if resp.status_code == 422:
            return {
                "agent1_status_code": resp.status_code,
                "agent1_body": agent1_body,
                "error_status_code": 422,
                "error_content": {"stage": "agent1_quality_gate", "agent1": agent1_body},
            }

        if resp.status_code != 200:
            return {
                "agent1_status_code": resp.status_code,
                "agent1_body": agent1_body,
                "error_status_code": resp.status_code,
                "error_content": {"stage": "agent1_error", "agent1": agent1_body},
            }

        return {"agent1_status_code": resp.status_code, "agent1_body": agent1_body}

    def route_after_agent1(self, state: PipelineState) -> Literal["stop", "continue"]:
        return "stop" if state.get("error_content") is not None else "continue"

    # ── Node 2: derive Agent 2's inputs from Agent 1's classification ──────

    def extract_domain_and_metadata(self, state: PipelineState) -> dict[str, Any]:
        agent1_body = state["agent1_body"]
        primary_domain = agent1_body.get("business_domain")
        if not primary_domain:
            return {
                "error_status_code": 422,
                "error_content": {
                    "stage": "agent1_classification_missing",
                    "detail": "Agent 1 completed but returned no business_domain to drive Agent 2 with.",
                    "agent1": agent1_body,
                },
            }

        primary_domain = _canonicalize_domain(primary_domain)

        column_descriptions = agent1_body.get("column_descriptions") or {}
        columns_metadata = []
        for col_name, description in column_descriptions.items():
            mandatory, expected_unique = _infer_mandatory_and_unique(col_name, description)
            columns_metadata.append({
                "column_name": col_name,
                "description": description,
                "mandatory": mandatory,
                "expected_unique": expected_unique,
            })
        schema_metadata = {"columns": columns_metadata}

        request_rules = state.get("request_rules")
        if not request_rules:
            # Caller didn't supply any business rules -- try to find genuine,
            # empirically-observed ones in the data itself rather than
            # leaving "consistency" permanently unassessable. Best-effort:
            # an inference failure or a dataset with no qualifying pair
            # just means no rules, same as today.
            inferred = _infer_consistency_rules(state["filename"], state["content"], state.get("sheet_name"))
            if inferred:
                request_rules = json.dumps(inferred)

        return {"primary_domain": primary_domain, "schema_metadata": schema_metadata, "request_rules": request_rules}

    def route_after_domain(self, state: PipelineState) -> Literal["stop", "continue"]:
        return "stop" if state.get("error_content") is not None else "continue"

    # ── Node 3: Agent 2 (Data Profiling Engine) ─────────────────────────────

    async def call_agent2(self, state: PipelineState) -> dict[str, Any]:
        url = f"{settings.AGENT2_BASE_URL}{settings.AGENT2_API_PREFIX}/profile-runs"
        data = {
            "primary_domain": state["primary_domain"],
            "schema_metadata": json.dumps(state["schema_metadata"]),
            "request_rules": state.get("request_rules") or "[]",
        }
        if state.get("sheet_name"):
            data["sheet_name"] = state["sheet_name"]
        if state.get("business_question"):
            data["business_question"] = state["business_question"]
        if state.get("target_column"):
            data["target_column"] = state["target_column"]

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    files={"file": (state["filename"], state["content"], state["content_type"])},
                    data=data,
                    timeout=settings.REQUEST_TIMEOUT_SECONDS,
                )
        except httpx.HTTPError as e:
            return {
                "error_status_code": 502,
                "error_content": {
                    "stage": "agent2_unreachable",
                    "detail": f"Could not reach Agent 2 (Data Profiling Engine) at {url}: {e}",
                },
            }

        agent2_body = _safe_json(resp)

        if resp.status_code != 200:
            note = "Agent 1 completed successfully; Agent 2 rejected the request."
            if resp.status_code == 422 and isinstance(agent2_body, dict) and \
                    (agent2_body.get("error") or {}).get("code") == "UNSUPPORTED_DOMAIN":
                note = (
                    f"Agent 1 classified this dataset as '{state['primary_domain']}', which Agent 2 does "
                    f"not support (only Finance, Payments, Customer, HR, Insurance). No override is available through "
                    f"the orchestrator — call Agent 2 directly with an explicit primary_domain if needed."
                )
            return {
                "agent2_status_code": resp.status_code,
                "agent2_body": agent2_body,
                "error_status_code": resp.status_code,
                "error_content": {
                    "stage": "agent2_error",
                    "agent1": state["agent1_body"],
                    "agent2": agent2_body,
                    "note": note,
                },
            }

        return {
            "agent2_status_code": resp.status_code,
            "agent2_body": agent2_body,
            "run_id": agent2_body["run_id"],
        }

    def route_after_agent2(self, state: PipelineState) -> Literal["stop", "continue"]:
        return "stop" if state.get("error_content") is not None else "continue"

    # ── Node 4: fetch Agent 2's full result ─────────────────────────────────

    async def fetch_agent2_result(self, state: PipelineState) -> dict[str, Any]:
        run_id = state["run_id"]
        try:
            status_code, body = await _get_agent2_result(run_id)
            agent2_full_result = body if status_code == 200 else state["agent2_body"]
        except httpx.HTTPError as e:
            return {
                "error_status_code": 502,
                "error_content": {
                    "stage": "agent2_result_fetch_failed",
                    "agent1": state["agent1_body"],
                    "agent2": state["agent2_body"],
                    "detail": f"Agent 2 run {run_id} completed, but fetching the full result failed: {e}",
                },
            }
        return {"agent2_full_result": agent2_full_result}

    def route_after_fetch(self, state: PipelineState) -> Literal["stop", "continue"]:
        return "stop" if state.get("error_content") is not None else "continue"

    # ── Node: persist Dataset Registry cache entry (cache-miss path only) ────
    # Runs once Agent 1 + Agent 2 have both succeeded, so the next
    # byte-identical upload becomes a cache hit. Never fails the pipeline —
    # a caching write failure is logged and swallowed, same principle as
    # call_agent3's own best-effort framing below.

    def persist_master_dataset(self, state: PipelineState) -> dict[str, Any]:
        fingerprint = state.get("fingerprint")
        if not fingerprint:
            return {}
        try:
            agent1_body = state.get("agent1_body") or {}
            agent2_full_result = state.get("agent2_full_result") or {}
            row_count = agent1_body.get("row_count") if isinstance(agent1_body, dict) else None
            column_count = agent1_body.get("column_count") if isinstance(agent1_body, dict) else None
            get_dataset_registry().persist_result(
                fingerprint=fingerprint,
                primary_domain=state.get("primary_domain") or "",
                agent1_body=agent1_body,
                agent2_full_result=agent2_full_result,
                row_count=row_count,
                column_count=column_count,
            )
        except Exception as e:
            logger.warning(f"Failed to persist Dataset Registry cache entry for {fingerprint}: {e}")
        return {}

    # ── Node 5: Agent 3 (Analytics Agent) ────────────────────────────────────
    # Vendored at mva/Analytics-Agent (originally github.com/VirenKhapra/
    # Analytics-agent-for-project-3), now a FastAPI service (port 8003) like
    # Agent 1/2, called the same way as call_agent2 above. A domain-agnostic
    # analytics engine since Phase 4 of its own redesign — Insurance is its
    # fully-built reference domain (curated KPIs, pre-computed variance
    # drivers), everything else gets a real generic report via
    # GenericDomainPlugin, not a skip. detected_domain (this request's
    # canonicalized primary_domain) is forwarded so Agent 3 can resolve the
    # right DomainPlugin itself — the Orchestrator doesn't need an opinion
    # about which domains it "supports" (Phase 4.5). Optional and best-
    # effort regardless of domain: skips cleanly outside Agent 3's actual
    # capabilities (no business_question, non-CSV upload — see
    # Agent3Capabilities above) and never populates error_content/
    # error_status_code on its own failures — Agent 1 and Agent 2 already
    # succeeded by the time this node runs, so a broken or unreachable
    # Agent 3 shouldn't fail an otherwise-successful pipeline.

    async def call_agent3(self, state: PipelineState) -> dict[str, Any]:
        business_question = state.get("business_question")
        skip_reason = _agent3_skip_reason(state["filename"], business_question)
        if skip_reason:
            return {"agent3_body": {"status": "skipped", "reason": skip_reason}}

        ml_score, llm_score, feature_columns, ml_breakdown, llm_breakdown = _readiness_and_features(
            state.get("agent2_full_result"),
        )
        column_profiles, hierarchy, charts, full_feature_recommendation = _dataset_context_fields(
            state.get("agent2_full_result"),
        )
        agent3_body = await _analyze_via_agent3(
            business_question, state["filename"], state["content"], state["content_type"],
            ml_score, llm_score, feature_columns, ml_breakdown, llm_breakdown,
            column_profiles, hierarchy, charts, full_feature_recommendation, state.get("primary_domain"),
        )
        return {"agent3_body": agent3_body}

    # ── Node 6: finalize ─────────────────────────────────────────────────────

    def finalize(self, state: PipelineState) -> dict[str, Any]:
        return {
            "result": {
                "agent1": state["agent1_body"],
                "agent2": state["agent2_full_result"],
                "agent3": state.get("agent3_body"),
                "primary_domain_used": state["primary_domain"],
                "fingerprint": state.get("fingerprint"),
                "copy_id": state.get("copy_id"),
                "was_cached": state.get("was_cached", False),
            }
        }
