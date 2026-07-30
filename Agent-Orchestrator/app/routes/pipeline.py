"""
Pipeline route — orchestrates the handoff between Agent 1 (Schema Intelligence Layer)
and Agent 2 (MVA Data Profiling Engine) via a LangGraph StateGraph (app/agents/orchestration_agent/graph.py).
Each future agent added to the pipeline becomes another node in that same graph.
"""

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
import httpx

from app.agents.orchestration_agent.graph import run_orchestrator_pipeline
from app.agents.orchestration_agent.nodes.pipeline import (
    _agent3_skip_reason, _analyze_via_agent3, _dataset_context_fields, _get_agent2_result, _readiness_and_features,
)
from app.config import settings

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


@router.post("/run")
async def run_pipeline(
    request: Request,
    file: UploadFile = File(..., description="CSV or Excel dataset to run through the full pipeline"),
    sheet_name: str | None = Form(
        default=None,
        description="Required only when the XLSX workbook has more than one non-empty sheet; "
                    "names which sheet Agent 2 should load.",
    ),
    force_reclassify: bool = Form(
        default=False,
        description="If this filename already exists in Agent 1's catalog, re-run its LLM column "
                    "descriptions and domain classification instead of reusing the original result.",
    ),
    force_revalidate: bool = Form(
        default=False,
        description="Bypass the Dataset Registry's cache (Stage 0A) entirely and re-run the full "
                    "Agent 1 + Agent 2 pipeline fresh, even for byte-identical content already "
                    "processed before. Implies force_reclassify. Useful for correcting a bad past "
                    "classification/profiling result.",
    ),
    business_question: str | None = Form(
        default=None,
        max_length=2000,
        description="Optional — when supplied, Agent 2 classifies the dataset's target/feature/drop "
                    "columns and an ML-vs-LLM approach for this question (feature_recommendation).",
    ),
    target_column: str | None = Form(
        default=None,
        description="Optional explicit override — if the caller already knows the target column, "
                    "Agent 2 skips LLM target-guessing and uses it directly.",
    ),
):
    """
    Runs a dataset through the full agent pipeline:

    1. Agent 1 (Schema Intelligence Layer) — quality gate, metadata extraction,
       LLM column descriptions, business domain classification.
    2. Agent 2 (MVA Data Profiling Engine) — deep structural profiling, quality/readiness
       scoring, chart generation. Seeded with Agent 1's column descriptions and its own
       primary_domain, taken directly from Agent 1's classification (business_domain) —
       no domain input is accepted from the caller.

    If Agent 1 rejects the file at its quality gate, the pipeline stops there —
    Agent 2 is never called. If Agent 1's classification isn't one of Agent 2's
    supported domains (Finance, Payments, Customer, HR, Insurance), the pipeline
    stops with a clear error rather than guessing.

    3. Agent 3 (Analytics Agent, vendored at Analytics-Agent/, port 8003) —
       optional. Runs for any of Agent 2's 5 supported domains, gated only on
       file type (`.csv`) and a supplied business_question — it answers
       exactly that one question using Agent 2's ML/LLM-readiness scores.
       Outside that scope it's skipped (`agent3.status == "skipped"`) without
       affecting Agent 1/2's results; a broken Agent 3 invocation likewise
       never fails the pipeline (`agent3.status == "failed"` with a reason).

    To ask Agent 3 a *different* question against the same dataset
    afterwards, use `POST /pipeline/ask` instead of calling this again —
    it skips Agent 1's quality gate and Agent 2's full profiling entirely.

    Not shown above (and deliberately not a declared, Swagger-visible
    parameter): an optional `request_rules` multipart field — a JSON string
    of additional Agent 2 business rules (e.g. `[{"type": "column_comparison",
    "rule_key": "...", "left_column": "...", "right_column": "...",
    "operator": "<="}]`), forwarded to Agent 2 verbatim on top of the
    mandatory/expected_unique inference already applied automatically. Left
    unset (the normal case), the Orchestrator infers consistency rules
    straight from the uploaded file itself — nothing to configure. It's kept
    as a direct-API-caller escape hatch for asserting a relationship the
    auto-inference can't verify from its own sample, without prompting every
    Swagger UI user for something that's already automatic.
    """
    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the maximum upload size of {settings.MAX_UPLOAD_SIZE_MB}MB "
                   f"(got {len(content) / (1024 * 1024):.1f}MB).",
        )
    filename = file.filename or "upload"
    content_type = file.content_type or "application/octet-stream"

    form = await request.form()
    request_rules = form.get("request_rules")
    request_rules = str(request_rules) if request_rules else None

    return await run_orchestrator_pipeline(
        filename=filename,
        content_type=content_type,
        content=content,
        sheet_name=sheet_name,
        force_reclassify=force_reclassify,
        force_revalidate=force_revalidate,
        business_question=business_question,
        target_column=target_column,
        request_rules=request_rules,
    )


@router.post("/ask")
async def ask_agent3(
    file: UploadFile = File(
        ..., description="The same dataset re-uploaded — Agent 3 needs real rows to query, and there's "
                          "nowhere durable this service can fetch them back from on its own."
    ),
    business_question: str = Form(..., max_length=2000, description="The new business question to answer."),
    run_id: str = Form(
        ..., description="Agent 2's run_id from an earlier POST /pipeline/run response "
                          "(the agent2.run_id field)."
    ),
):
    """
    Re-ask Agent 3 a different business_question against a dataset that
    already went through a full `POST /pipeline/run` — without repeating
    Agent 1's quality gate/classification or Agent 2's full profiling
    pipeline. Only reads Agent 2's already-persisted result for `run_id`
    (one lightweight GET) and calls Agent 3 — Agent 1 and Agent 2's actual
    pipelines never run again.

    Response mirrors `/pipeline/run`'s shape, minus the agents that didn't
    run: `{"agent3": {...}, "primary_domain_used": "..."}`. `agent3.status`
    is `"skipped"` (not an error) outside Agent 3's scope — non-Insurance
    domain, non-CSV upload — same as in `/pipeline/run`.

    Errors distinct from `agent3.status` (these are caller mistakes, not
    "outside Agent 3's scope"): `404` if `run_id` isn't known to Agent 2,
    `502` if Agent 2 is unreachable.
    """
    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the maximum upload size of {settings.MAX_UPLOAD_SIZE_MB}MB "
                   f"(got {len(content) / (1024 * 1024):.1f}MB).",
        )
    filename = file.filename or "upload"
    content_type = file.content_type or "application/octet-stream"

    try:
        status_code, body = await _get_agent2_result(run_id)
    except httpx.HTTPError as e:
        return JSONResponse(
            status_code=502,
            content={"detail": f"Could not reach Agent 2 (Data Profiling Engine) to fetch run {run_id}: {e}"},
        )

    if status_code == 404:
        return JSONResponse(
            status_code=404,
            content={"detail": f"No Agent 2 run found for run_id={run_id!r} — run POST /pipeline/run first."},
        )
    if status_code != 200:
        return JSONResponse(
            status_code=502,
            content={"detail": f"Agent 2 returned {status_code} fetching run {run_id}: {body}"},
        )

    agent2_full_result = body
    primary_domain = agent2_full_result.get("primary_domain")

    skip_reason = _agent3_skip_reason(filename, business_question)
    if skip_reason:
        return {
            "agent3": {"status": "skipped", "reason": skip_reason},
            "primary_domain_used": primary_domain,
        }

    ml_score, llm_score, feature_columns, ml_breakdown, llm_breakdown = _readiness_and_features(agent2_full_result)
    column_profiles, hierarchy, charts, full_feature_recommendation = _dataset_context_fields(agent2_full_result)
    agent3_body = await _analyze_via_agent3(
        business_question, filename, content, content_type,
        ml_score, llm_score, feature_columns, ml_breakdown, llm_breakdown,
        column_profiles, hierarchy, charts, full_feature_recommendation, primary_domain,
    )
    return {"agent3": agent3_body, "primary_domain_used": primary_domain}
