"""Typed state for the profiling LangGraph — threads pipeline data between nodes.

Unlike the old linear execute() method, graph nodes only communicate through this
shared state dict, so every intermediate value a later node needs has to be an
explicit field here (not just a local Python variable in one function's scope).
"""

import uuid
from typing import Any, TypedDict


class ProfilingState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    run_id: uuid.UUID
    file_content: bytes
    filename: str
    primary_domain: str
    schema_metadata: dict[str, Any] | None
    request_rules: list[dict[str, Any]] | None
    sheet_name: str | None
    approved_rule_dicts: list[dict[str, Any]] | None
    business_question: str | None
    target_column_override: str | None

    # ── File loading ────────────────────────────────────────────────────────
    file_type: Any
    file_path: Any
    df: Any  # pandas.DataFrame

    # ── Profiling ───────────────────────────────────────────────────────────
    dataset_profile_obj: Any
    col_profiles: list[Any]
    refined_types: list[Any]
    grain_result: Any
    sem_candidates: list[Any]

    # ── Schema intelligence (LLM) ───────────────────────────────────────────
    si_result: Any
    schema_intelligence_attempted: bool
    schema_intelligence_retry_count: int

    # ── Classification ──────────────────────────────────────────────────────
    sec_domain: Any
    classifications: list[Any]

    # ── Hierarchy ───────────────────────────────────────────────────────────
    hierarchy_result: Any

    # ── Business rules ──────────────────────────────────────────────────────
    rules: list[Any]
    role_map: dict[str, str]
    rule_results: list[Any]

    # ── Quality ─────────────────────────────────────────────────────────────
    mandatory_cols: list[str]
    unique_cols: list[str]
    quality_dims: list[dict[str, Any]]
    overall_quality: dict[str, Any]
    quality_gate_passed: bool

    # ── Rule suggestions (LLM) ──────────────────────────────────────────────
    rule_suggestions: list[dict[str, Any]]
    rule_suggestion_retry_count: int

    # ── Target/feature/drop recommendation ───────────────────────────────────
    feature_recommendation: dict[str, Any]

    # ── Readiness / charts / drill-down (skipped on a weak quality gate) ────
    readiness_assessments: list[dict[str, Any]]
    charts: list[dict[str, Any]]
    drill_down_cubes: list[dict[str, Any]]

    # ── Output ──────────────────────────────────────────────────────────────
    column_output: list[dict[str, Any]]
    status: str
    started_at: str | None
    completed_at: str | None
    error: dict[str, Any] | None
    warnings: list[str]
