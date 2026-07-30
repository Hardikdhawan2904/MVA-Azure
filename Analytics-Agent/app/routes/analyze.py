"""app/routes/analyze.py — POST /analyze, the Analytics Agent's one real
endpoint.

Accepts the uploaded dataset directly (multipart/form-data), rather than a
filesystem path — the difference from the old CLI/subprocess integration
that made Phase 2's orchestrator rewrite (subprocess → httpx call) possible.
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.agents.analytics_agent.graph import run_analytics_graph
from app.config import MAX_UPLOAD_SIZE_MB
from app.schemas.responses import AnalysisResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analytics"])


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile = File(..., description="CSV dataset to answer the question against"),
    business_question: str = Form(..., max_length=2000, description="The business question to answer"),
    conversation_id: str | None = Form(
        default=None,
        description="Omit for a new conversation (a fresh id is generated and returned in the response). "
                    "Pass back the conversation_id from a prior response to continue it — filter/KPI "
                    "carryover ('what about EMEA?') and the LLM's prior-turn context both depend on this.",
    ),
    ml_readiness: float = Form(
        default=99.75,
        description="ML readiness score from Agent 2 — gates ML model execution vs. the deterministic fallback",
    ),
    llm_readiness: float = Form(
        default=99.75,
        description="LLM readiness score from Agent 2 — gates Azure OpenAI narration vs. the deterministic template formatter",
    ),
    feature_recommendation: str | None = Form(
        default=None,
        description="Optional JSON string of Agent 2's per-column feature classification for this "
                    "upload — used only to validate the hardcoded ML feature column lists still match "
                    "this dataset's schema; never blocks the request if omitted or malformed.",
    ),
    ml_readiness_breakdown: str | None = Form(
        default=None,
        description="Optional JSON string of Agent 2's full ml_readiness assessment (strengths, "
                    "blocking_issues, evidence) — surfaced in execution_trace so the ml_readiness gate "
                    "can explain *why*, not just report the score. Never blocks the request if omitted "
                    "or malformed.",
    ),
    llm_readiness_breakdown: str | None = Form(
        default=None,
        description="Same as ml_readiness_breakdown, for Agent 2's llm_readiness assessment.",
    ),
    column_profiles: str | None = Form(
        default=None,
        description="Optional JSON string (a list) of Agent 2's per-column semantic classification "
                    "(column_profiles from its profiling result) — builds a richer DatasetContext for "
                    "the Agent 3 redesign's generic analytics pipeline. Inert if omitted; falls back to "
                    "local schema inference. Never blocks the request if malformed.",
    ),
    hierarchy: str | None = Form(
        default=None,
        description="Optional JSON string (a dict) of Agent 2's detected drill-down hierarchy for this "
                    "dataset. Inert if omitted or malformed.",
    ),
    charts: str | None = Form(
        default=None,
        description="Optional JSON string (a list) of Agent 2's pre-selected chart candidates for this "
                    "dataset. Inert if omitted or malformed.",
    ),
    full_feature_recommendation: str | None = Form(
        default=None,
        description="Optional JSON string (a dict) of Agent 2's complete feature_recommendation object "
                    "(target_column, problem_type, recommended_approach, feature/drop columns, "
                    "confidence) — distinct from the simpler `feature_recommendation` list above. Inert "
                    "if omitted or malformed.",
    ),
    detected_domain: str | None = Form(
        default=None,
        description="Optional — Agent 1's business_domain classification for this upload, as forwarded "
                    "by the Orchestrator. Resolves which DomainPlugin applies (PluginRegistry.find_plugin) "
                    "-- falls back to GenericDomainPlugin when omitted or unmatched.",
    ),
) -> AnalysisResponse:
    def _parse_json_field(raw: str | None, field_name: str) -> Any | None:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"{field_name} Form field was not valid JSON — ignoring it.")
            return None

    parsed_feature_recommendation = _parse_json_field(feature_recommendation, "feature_recommendation")
    parsed_ml_breakdown = _parse_json_field(ml_readiness_breakdown, "ml_readiness_breakdown")
    parsed_llm_breakdown = _parse_json_field(llm_readiness_breakdown, "llm_readiness_breakdown")
    parsed_column_profiles = _parse_json_field(column_profiles, "column_profiles")
    parsed_hierarchy = _parse_json_field(hierarchy, "hierarchy")
    parsed_charts = _parse_json_field(charts, "charts")
    parsed_full_feature_recommendation = _parse_json_field(
        full_feature_recommendation, "full_feature_recommendation",
    )

    content = await file.read()
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the maximum upload size of {MAX_UPLOAD_SIZE_MB}MB "
                   f"(got {len(content) / (1024 * 1024):.1f}MB).",
        )

    result = run_analytics_graph(
        file_content=content,
        business_question=business_question,
        conversation_id=conversation_id,
        ml_readiness_score=ml_readiness,
        llm_readiness_score=llm_readiness,
        feature_recommendation=parsed_feature_recommendation,
        ml_readiness_breakdown=parsed_ml_breakdown,
        llm_readiness_breakdown=parsed_llm_breakdown,
        column_profiles=parsed_column_profiles,
        hierarchy=parsed_hierarchy,
        charts=parsed_charts,
        full_feature_recommendation=parsed_full_feature_recommendation,
        detected_domain=detected_domain,
    )

    return AnalysisResponse(**result)
